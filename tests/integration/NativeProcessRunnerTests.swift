import Foundation
import Darwin

@main struct NativeProcessRunnerTests {
    static func main() async throws {
        let executable = URL(fileURLWithPath: CommandLine.arguments[1])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tc-process-tests-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        func ready(_ url: URL) async throws -> [Int32] {
            for _ in 0..<200 {
                if let text = try? String(contentsOf: url, encoding: .utf8), text.hasSuffix("\n") {
                    let values = text.split(separator: " ").compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
                    if values.count == 3 { return values }
                }
                try await Task.sleep(for: .milliseconds(25))
            }
            throw NSError(domain: "Fixture never became ready", code: 1)
        }
        let runner = NativeProcessRunner()
        do {
            _ = try await runner.run(executable: directory.appendingPathComponent("missing"), arguments: [])
            fatalError("Missing executable accepted")
        } catch NativeProcessRunner.Failure.system {}
        let normal = try await runner.run(executable: executable, arguments: ["normal", directory.appendingPathComponent("normal").path])
        precondition(normal.exitCode == 7 && normal.signal == nil && !normal.cleanupPending && normal.failure == nil)
        precondition(normal.stdout == Data("terminal\n".utf8) && normal.stderr == Data("diagnostic\n".utf8))
        print("spawn failure releases admission; normal exit/stdout/stderr PASS")
        let large = try await runner.run(executable: executable, arguments: ["large", directory.appendingPathComponent("large").path])
        precondition(large.exitCode == 0 && large.failure == nil && !large.cleanupPending)
        precondition(large.stdout.count == 3 << 20 && large.stderr.count == 512 << 10)
        print("bounded pipes drain large output without deadlock PASS")
        let unrelated = Process(); unrelated.executableURL = URL(fileURLWithPath: "/bin/sleep"); unrelated.arguments = ["60"]
        try unrelated.run()
        defer { if unrelated.isRunning { unrelated.terminate() }; unrelated.waitUntilExit() }
        for mode in ["cooperative", "ignore", "child"] {
            let path = directory.appendingPathComponent(mode)
            let task = Task { try await runner.run(executable: executable, arguments: [mode, path.path]) }
            let ids = try await ready(path)
            precondition(ids[0] == ids[1] && ids[0] != getpgrp())
            do {
                _ = try await runner.run(executable: executable, arguments: [])
                fatalError("Concurrent worker accepted")
            } catch NativeProcessRunner.Failure.busy {}
            let began = ContinuousClock.now
            task.cancel()
            let result = try await task.value
            let elapsed = began.duration(to: .now)
            precondition(result.cancellationRequested && !result.cleanupPending && result.failure == nil)
            if mode == "cooperative" {
                precondition(result.exitCode == 2 && !result.forcedStop && elapsed < .seconds(5))
                precondition(result.stdout == Data("cancelled\n".utf8))
            } else {
                precondition(result.signal == SIGKILL && result.forcedStop && elapsed >= .seconds(5) && elapsed < .seconds(11))
            }
            precondition(kill(-result.pid, 0) == -1 && errno == ESRCH)
            precondition(unrelated.isRunning)
            print("\(mode) cancellation/group cleanup, unrelated process intact PASS (\(elapsed))")
        }
        let orphanPath = directory.appendingPathComponent("early-parent")
        let early = try await runner.run(executable: executable, arguments: ["early_parent", orphanPath.path])
        precondition(early.exitCode == 0 && !early.cleanupPending && early.failure == nil)
        precondition(kill(-early.pid, 0) == -1 && errno == ESRCH)
        print("leader exit cleans surviving child group PASS")
        let limited = NativeProcessRunner(policy: .init(grace: .seconds(5), reap: .seconds(5), stdoutLimit: 8192, stderrLimit: 8192))
        let flood = try await limited.run(executable: executable, arguments: ["flood", directory.appendingPathComponent("flood").path])
        precondition(flood.failure == "worker_log_limit" && flood.stdout.count == 8192 && !flood.cleanupPending)
        print("output overflow initiates owned cleanup and caps retained bytes PASS")
        let immediate = NativeProcessRunner(policy: .init(grace: .zero, reap: .zero))
        let pendingPath = directory.appendingPathComponent("pending")
        let pendingTask = Task { try await immediate.run(executable: executable, arguments: ["ignore", pendingPath.path]) }
        _ = try await ready(pendingPath); pendingTask.cancel()
        let pending = try await pendingTask.value
        precondition(pending.cleanupPending && pending.forcedStop)
        do {
            _ = try await immediate.run(executable: executable, arguments: [])
            fatalError("Cleanup-pending worker accepted another job")
        } catch NativeProcessRunner.Failure.cleanupPending {}
        var cleaned = false
        for _ in 0..<200 {
            if await immediate.pollCleanup() { cleaned = true; break }
            try await Task.sleep(for: .milliseconds(25))
        }
        precondition(cleaned)
        let after = try await immediate.run(executable: executable, arguments: ["normal", directory.appendingPathComponent("after").path])
        // A zero reap test policy may also defer the ordinary terminal observation.
        if after.cleanupPending { for _ in 0..<200 { if await immediate.pollCleanup() { break }; try await Task.sleep(for: .milliseconds(25)) } }
        let finalCleaned = await immediate.pollCleanup()
        precondition(finalCleaned)
        print("zero-deadline injected cleanup_pending blocks admission until OS-confirmed reaping PASS")
        print("NativeProcessRunner tests PASS; no GPU/model execution")
    }
}
