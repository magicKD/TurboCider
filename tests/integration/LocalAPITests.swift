import Foundation

// Pause only after receiving a real owned service response. The service can
// therefore terminate/restart while an old App continuation remains pending.
@MainActor private final class PausedAPITransport {
    @MainActor final class Reply {
        var started = false
        private var released = false
        private var continuation: CheckedContinuation<Void, Never>?
        let fail: Bool
        let markStale: Bool
        init(fail: Bool, markStale: Bool) { self.fail = fail; self.markStale = markStale }
        func wait() async {
            started = true
            if released { return }
            await withCheckedContinuation { continuation = $0 }
        }
        func release() {
            released = true
            let pending = continuation; continuation = nil; pending?.resume()
        }
    }
    private let executable: URL
    private let socket: String
    private var pending: [Reply] = []
    private var replies: [Reply] = []
    init(executable: URL, socket: String) { self.executable = executable; self.socket = socket }
    func pause(fail: Bool = false, markStale: Bool = false) -> Reply {
        let reply = Reply(fail: fail, markStale: markStale)
        pending.append(reply); replies.append(reply); return reply
    }
    func releaseAll() { for reply in replies { reply.release() } }
    func send(_ request: [String: Any]) async throws -> [String: Any] {
        let reply = pending.isEmpty ? nil : pending.removeFirst()
        let payload = try JSONSerialization.data(withJSONObject: request)
        let data = try await LibraryTool.run(["rpc", socket, "{request}"], payload: payload, executable: executable)
        guard var raw = try JSONSerialization.jsonObject(with: data) as? [String: Any], raw["ok"] as? Bool == true else {
            throw NativeFailure(message: "Paused API transport received an invalid real response")
        }
        if let reply {
            await reply.wait()
            if reply.fail { throw NativeFailure(message: "old RPC failure") }
            if reply.markStale {
                var result = raw["result"] as? [String: Any] ?? [:]
                result["active_job"] = "old-job"; result["session_model"] = "old-model"; result["history_count"] = 7
                raw["result"] = result
            }
        }
        return raw
    }
}

@main struct LocalAPITests {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-api-test-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let socket = "/private/tmp/tc-api-\(UUID().uuidString.prefix(8)).sock"
        defer { try? FileManager.default.removeItem(atPath: socket + ".lock") }
        let executable = Bundle.main.executableURL!.deletingLastPathComponent().appendingPathComponent("turbocider")
        let store = NativeJobStore(directory: root.appendingPathComponent("studio"))
        let transport = PausedAPITransport(executable: executable, socket: socket)
        defer { transport.releaseAll() }
        let api = LocalAPIController(directory: root, executable: executable, socketPath: socket,
                                     rpcTransport: { try await transport.send($0) })
        defer { api.terminateNow() }
        func check(_ ok: Bool, _ text: String) throws { if !ok { throw NativeFailure(message: text) } }
        func wait(_ text: String, until condition: () -> Bool) async throws {
            for _ in 0..<500 {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            throw NativeFailure(message: "Timed out: \(text)")
        }
        await api.start(store: store)
        try check(api.running && store.externalServiceActive, "App service did not start: \(api.error ?? "")")
        await api.refresh()
        try check(api.error == nil && api.jobCount == 0 && api.activeJob.isEmpty, "App service status failed")
        do {
            try await store.load(modelURL: root, modelID: "z-image-turbo")
            throw NativeFailure(message: "Embedded model loaded while API owned execution")
        } catch { try check(error.localizedDescription.contains("API"), "Wrong execution ownership error") }
        let other = LocalAPIController(directory: root.appendingPathComponent("other"), executable: executable, socketPath: socket)
        let otherStore = NativeJobStore(directory: root.appendingPathComponent("other-studio"))
        await other.start(store: otherStore)
        try check(!other.running && other.error != nil, "Second App attached to another process's socket")
        await api.refresh(); try check(api.error == nil, "Second start damaged the running service")
        api.stop()
        for _ in 0..<100 { if !api.changing && !api.running { break }; try await Task.sleep(for: .milliseconds(100)) }
        try check(!api.running && !store.externalServiceActive && !FileManager.default.fileExists(atPath: socket), "App stop left socket or execution reservation")
        await api.start(store: store)
        try check(api.running, "App service cannot restart: \(api.error ?? "")")
        api.stop()
        for _ in 0..<100 { if !api.changing && !api.running { break }; try await Task.sleep(for: .milliseconds(100)) }
        try check(!api.running && !store.externalServiceActive, "Restarted service did not release execution")
        let missing = LocalAPIController(directory: root.appendingPathComponent("missing"), executable: root.appendingPathComponent("missing-cli"), socketPath: socket)
        await missing.start(store: store)
        try check(!missing.running && !store.externalServiceActive && missing.error != nil, "Spawn failure left execution reserved")

        // Both an old successful response and a late transport error must be
        // ignored while a replacement service start owns the execution lock.
        for fail in [false, true] {
            await api.start(store: store)
            try check(api.running, "Service did not start before stale refresh test")
            let oldReply = transport.pause(fail: fail, markStale: true)
            var refreshFinished = false
            let refreshTask = Task { @MainActor in await api.refresh(); refreshFinished = true }
            try await wait("old status response", until: { oldReply.started })
            api.stop()
            try await wait("owned service stop", until: { !api.changing && !api.running })
            let newReply = transport.pause()
            var startFinished = false
            let startTask = Task { @MainActor in await api.start(store: store); startFinished = true }
            try await wait("replacement startup response", until: { newReply.started })
            oldReply.release()
            try await wait("old refresh completion", until: { refreshFinished })
            await refreshTask.value
            // Cover the previous 200 ms unbound stop poll after a new start.
            try await Task.sleep(for: .milliseconds(250))
            try check(api.changing && !api.running && store.externalServiceActive,
                      "Old stop/refresh unlocked replacement service start")
            try check(api.error == nil && api.activeJob.isEmpty && api.sessionModel.isEmpty && api.jobCount == 0,
                      "Old status/error polluted the replacement service")
            newReply.release()
            try await wait("replacement startup completion", until: { startFinished })
            await startTask.value
            try check(api.running && !api.changing, "Replacement service failed after stale response")
            await api.refresh()
            try check(api.error == nil && api.activeJob.isEmpty && api.sessionModel.isEmpty, "Fresh service status failed")
            api.stop()
            try await wait("replacement stop", until: { !api.changing && !api.running })
        }

        // A startup continuation has a defer and retry catch too: neither may
        // publish running=true nor release a newer generation's reservation.
        let oldStartReply = transport.pause()
        var oldStartFinished = false
        let oldStart = Task { @MainActor in await api.start(store: store); oldStartFinished = true }
        try await wait("old startup response", until: { oldStartReply.started })
        api.stop()
        try await wait("cancelled startup stop", until: { !api.changing && !api.running })
        let replacementReply = transport.pause()
        var replacementFinished = false
        let replacement = Task { @MainActor in await api.start(store: store); replacementFinished = true }
        try await wait("new startup response", until: { replacementReply.started })
        oldStartReply.release()
        try await wait("old startup completion", until: { oldStartFinished })
        await oldStart.value
        try check(api.changing && !api.running && store.externalServiceActive && api.error == nil,
                  "Old startup success/defer changed a new service generation")
        replacementReply.release()
        try await wait("new startup completion", until: { replacementFinished })
        await replacement.value
        try check(api.running && !api.changing, "Replacement start did not complete")
        api.stop()
        try await wait("final service stop", until: { !api.changing && !api.running })
        try check(!store.externalServiceActive && !FileManager.default.fileExists(atPath: socket), "Final stop left service ownership")
        print("PASS: App API start/status/stop/restart, execution ownership, duplicate socket isolation, failed launch recovery, stale status/error/startup isolation")
    }
}
