import Foundation
import Darwin

/// One process group per request. This runner must be shared by all public GPU
/// jobs. A cleanup-pending outcome retains its admission lock until OS exit is
/// observed. Native workers must not escape the owned group with setsid/setpgid.
actor NativeProcessRunner {
    static let shared = NativeProcessRunner()
    struct Policy: Sendable {
        var grace: Duration = .seconds(5)
        var reap: Duration = .seconds(5)
        var stdoutLimit = 4 << 20
        var stderrLimit = 1 << 20
    }
    struct Result: Sendable {
        let pid: Int32
        let exitCode: Int32?
        let signal: Int32?
        let stdout: Data
        let stderr: Data
        let cancellationRequested: Bool
        let forcedStop: Bool
        let cleanupPending: Bool
        let failure: String?
    }
    enum Failure: Error { case busy, cleanupPending, invalidLaunch, system(String, Int32) }
    private final class FD {
        var value: Int32
        init(_ value: Int32) { self.value = value }
        func close() { if value >= 0 { Darwin.close(value); value = -1 } }
        deinit { close() }
    }
    private final class Child {
        let pid: pid_t
        var status: Int32?
        var identityLost = false
        init(_ pid: pid_t) { self.pid = pid }
        // Observe without reaping: the PID remains reserved until every signal
        // has been sent. No kill(pid/group) is allowed after waitpid succeeds.
        func exited() -> Bool {
            if status != nil { return true }
            if identityLost { return false }
            var info = siginfo_t()
            if waitid(P_PID, id_t(pid), &info, WEXITED | WNOHANG | WNOWAIT) == 0 { return info.si_pid == pid }
            if errno != EINTR { identityLost = true }
            return false
        }
        func send(_ signal: Int32) {
            guard status == nil, !identityLost else { return }
            _ = kill(-pid, signal)
            // The group signal already reaches its leader. Do not interrupt a
            // cooperative cleanup twice; target PID separately only if escaped.
            if getpgid(pid) != pid { _ = kill(pid, signal) }
        }
        func reapIfExited() {
            guard status == nil, exited(), !identityLost else { return }
            var raw: Int32 = 0
            let result = waitpid(pid, &raw, WNOHANG)
            if result == pid { status = raw }
            else if result < 0 && errno != EINTR { identityLost = true }
        }
        var groupGone: Bool { kill(-pid, 0) == -1 && errno == ESRCH }
        var cleaned: Bool { status != nil && groupGone }
    }
    private var active = false
    private var pending: Child?
    private let policy: Policy
    init(policy: Policy = Policy()) { self.policy = policy }

    /// No signals after reaping. If identity was lost, leave admission closed;
    /// absence of an unrelated/reused PID is not proof of this job's cleanup.
    func pollCleanup() -> Bool {
        guard let child = pending else { return !active }
        child.reapIfExited()
        if child.cleaned { pending = nil; active = false; return true }
        return false
    }
    var cleanupPending: Bool { pending != nil }

    private func pipePair() throws -> (FD, FD) {
        var descriptors: [Int32] = [0, 0]
        guard pipe(&descriptors) == 0 else { throw Failure.system("pipe", errno) }
        let read = FD(descriptors[0]), write = FD(descriptors[1])
        guard fcntl(read.value, F_SETFD, FD_CLOEXEC) == 0,
              fcntl(write.value, F_SETFD, FD_CLOEXEC) == 0,
              fcntl(read.value, F_SETFL, O_NONBLOCK) == 0 else { throw Failure.system("fcntl", errno) }
        return (read, write)
    }
    private func launch(executable: URL, arguments: [String], environment: [String: String],
                        stdout: FD, stderr: FD, stdin: FD?) throws -> Child {
        guard executable.isFileURL, executable.path.hasPrefix("/"),
              ([executable.path] + arguments + environment.flatMap { [$0.key, $0.value] }).allSatisfy({ !$0.contains("\0") }),
              environment.keys.allSatisfy({ !$0.isEmpty && !$0.contains("=") }) else { throw Failure.invalidLaunch }
        var actions: posix_spawn_file_actions_t?
        var attributes: posix_spawnattr_t?
        func check(_ code: Int32, _ name: String) throws { if code != 0 { throw Failure.system(name, code) } }
        try check(posix_spawn_file_actions_init(&actions), "file_actions_init")
        defer { posix_spawn_file_actions_destroy(&actions) }
        try check(posix_spawnattr_init(&attributes), "spawnattr_init")
        defer { posix_spawnattr_destroy(&attributes) }
        if let stdin {
            try check(posix_spawn_file_actions_adddup2(&actions, stdin.value, STDIN_FILENO), "stdin")
        } else {
            try check(posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0), "stdin")
        }
        try check(posix_spawn_file_actions_adddup2(&actions, stdout.value, STDOUT_FILENO), "stdout")
        try check(posix_spawn_file_actions_adddup2(&actions, stderr.value, STDERR_FILENO), "stderr")
        try check(posix_spawnattr_setpgroup(&attributes, 0), "pgroup")
        var mask = sigset_t(), defaults = sigset_t()
        sigemptyset(&mask); sigemptyset(&defaults); sigaddset(&defaults, SIGINT); sigaddset(&defaults, SIGTERM); sigaddset(&defaults, SIGPIPE)
        try check(posix_spawnattr_setsigmask(&attributes, &mask), "sigmask")
        try check(posix_spawnattr_setsigdefault(&attributes, &defaults), "sigdefault")
        try check(posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_CLOEXEC_DEFAULT)), "flags")
        var argv = ([executable.path] + arguments).map { strdup($0) } + [nil]
        var envp = environment.sorted(by: { $0.key < $1.key }).map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { argv.forEach { free($0) }; envp.forEach { free($0) } }
        guard argv.dropLast().allSatisfy({ $0 != nil }), envp.dropLast().allSatisfy({ $0 != nil }) else { throw Failure.system("strdup", ENOMEM) }
        var pid: pid_t = 0
        try check(posix_spawn(&pid, executable.path, &actions, &attributes, &argv, &envp), "posix_spawn")
        return Child(pid)
    }
    private func drain(_ fd: FD, into data: inout Data, limit: Int, observer: (@Sendable (Data) throws -> Void)? = nil) -> String? {
        guard fd.value >= 0 else { return nil }
        var buffer = [UInt8](repeating: 0, count: 16 * 1024)
        var consumed = 0
        while consumed < 256 * 1024 {
            let count = read(fd.value, &buffer, buffer.count)
            if count == 0 { fd.close(); return nil }
            if count < 0 {
                if errno == EINTR { continue }
                if errno == EAGAIN || errno == EWOULDBLOCK { return nil }
                return "worker_pipe_read_failed"
            }
            consumed += count
            let allowed = min(count, max(0, limit - data.count))
            data.append(contentsOf: buffer.prefix(allowed))
            if let observer {
                do { try observer(Data(buffer.prefix(allowed))) }
                catch { return "worker_event_invalid: \(error)" }
            }
            if allowed < count { return "worker_log_limit" }
        }
        return nil
    }
    func run(executable: URL, arguments: [String],
             environment: [String: String] = ProcessInfo.processInfo.environment,
             admission: WorkerLaunchAdmission? = nil,
             onStderr: (@Sendable (Data) throws -> Void)? = nil) async throws -> Result {
        guard pending == nil else { throw Failure.cleanupPending }
        guard !active else { throw Failure.busy }
        guard policy.grace >= .zero, policy.reap >= .zero, policy.stdoutLimit > 0, policy.stderrLimit > 0 else { throw Failure.invalidLaunch }
        try Task.checkCancellation()
        active = true
        defer { if pending == nil { active = false } }
        let (outRead, outWrite) = try pipePair(), (errRead, errWrite) = try pipePair()
        let gate = try admission.map { _ in try pipePair() }
        if let gate {
            guard fcntl(gate.0.value, F_SETFL, 0) == 0, fcntl(gate.1.value, F_SETNOSIGPIPE, 1) == 0 else { throw Failure.system("admission_pipe", errno) }
        }
        let child = try launch(executable: executable, arguments: arguments + (admission == nil ? [] : ["--supervised"]),
                               environment: environment, stdout: outWrite, stderr: errWrite, stdin: gate?.0)
        gate?.0.close()
        outWrite.close(); errWrite.close()
        defer { outRead.close(); errRead.close() }
        var stdout = Data(), stderr = Data(), failure: String?
        if let admission, let gate {
            do {
                try admission.persist(WorkerProcessIdentity.capture(child.pid))
                if !Task.isCancelled {
                    var byte: UInt8 = 1
                    guard write(gate.1.value, &byte, 1) == 1 else { throw Failure.system("admission_write", errno) }
                }
            } catch { failure = "worker_admission_failed: \(error)" }
            gate.1.close()
        }
        var stopAt: ContinuousClock.Instant?, killAt: ContinuousClock.Instant?
        var cancelled = false, forced = false
        while true {
            if let error = drain(outRead, into: &stdout, limit: policy.stdoutLimit) { failure = failure ?? error }
            if let error = drain(errRead, into: &stderr, limit: policy.stderrLimit, observer: onStderr) { failure = failure ?? error }
            if Task.isCancelled { cancelled = true }
            if child.identityLost { failure = failure ?? "worker_process_identity_lost"; break }
            if child.exited() {
                // Kill surviving members before releasing the leader's PID.
                if killAt == nil { child.send(SIGKILL); killAt = .now }
                child.reapIfExited()
            }
            if child.cleaned {
                // Pipes may still contain buffered output after the writer exits.
                while outRead.value >= 0 || errRead.value >= 0 {
                    let before = stdout.count + stderr.count
                    if let error = drain(outRead, into: &stdout, limit: policy.stdoutLimit) { failure = failure ?? error }
                    if let error = drain(errRead, into: &stderr, limit: policy.stderrLimit, observer: onStderr) { failure = failure ?? error }
                    if failure != nil { break }
                    // Group absence guarantees no owned writer remains; escaped
                    // processes are outside this worker's supported contract.
                    if (outRead.value >= 0 || errRead.value >= 0), stdout.count + stderr.count == before { failure = "worker_pipe_not_closed"; break }
                }
                break
            }
            if (cancelled || failure != nil), stopAt == nil { stopAt = .now; child.send(SIGTERM) }
            if let stopAt, killAt == nil, stopAt.duration(to: .now) >= policy.grace {
                child.send(SIGKILL); killAt = .now; forced = true
            }
            if let killAt, killAt.duration(to: .now) >= policy.reap { break }
            // Cancellation must not turn this bounded cleanup loop into a busy spin.
            await Task.detached { try? await Task.sleep(for: .milliseconds(25)) }.value
        }
        let cleanupPending = !child.cleaned
        if cleanupPending { pending = child }
        let raw = child.status
        let exited = raw.map { ($0 & 0x7f) == 0 } ?? false
        return Result(pid: child.pid, exitCode: exited ? raw.map { ($0 >> 8) & 0xff } : nil,
                      signal: exited ? nil : raw.map { $0 & 0x7f }, stdout: stdout, stderr: stderr,
                      cancellationRequested: cancelled, forcedStop: forced, cleanupPending: cleanupPending, failure: failure)
    }
}
