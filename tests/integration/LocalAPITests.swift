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
        let result: [String: Any]?
        init(fail: Bool, markStale: Bool, result: [String: Any]?) {
            self.fail = fail; self.markStale = markStale; self.result = result
        }
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
    private(set) var requests: [[String: Any]] = []
    init(executable: URL, socket: String) { self.executable = executable; self.socket = socket }
    func pause(fail: Bool = false, markStale: Bool = false, result: [String: Any]? = nil) -> Reply {
        let reply = Reply(fail: fail, markStale: markStale, result: result)
        pending.append(reply); replies.append(reply); return reply
    }
    func releaseAll() { for reply in replies { reply.release() } }
    func send(_ request: [String: Any]) async throws -> [String: Any] {
        requests.append(request)
        let reply = pending.isEmpty ? nil : pending.removeFirst()
        let payload = try JSONSerialization.data(withJSONObject: request)
        let data = try await LibraryTool.run(["rpc", socket, "{request}"], payload: payload, executable: executable)
        guard var raw = try JSONSerialization.jsonObject(with: data) as? [String: Any], raw["ok"] as? Bool == true else {
            throw NativeFailure(message: "Paused API transport received an invalid real response")
        }
        if let reply {
            await reply.wait()
            if reply.fail { throw NativeFailure(message: "old RPC failure") }
            if let result = reply.result { raw["result"] = result }
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

        // Read-only jobs UI uses protocol fixtures over the same real owned
        // service transport; no request is submitted and no model is opened.
        func job(_ id: String, state: String = "succeeded", output: String? = nil,
                 operation: String = "image.edit") -> [String: Any] {
            var value: [String: Any] = ["id": id, "state": state, "created_at": 1700000000.0,
                "request": ["model": "qwen-image-2.1", "operation": operation, "prompt": "Keep the scene.",
                            "output": root.appendingPathComponent("proposed-only.png").path]]
            if let output {
                value["result"] = ["schema_version": 1, "model": "qwen-image-2.1", "operation": operation,
                                   "output": output, "width": 512, "height": 512,
                                   "timings_seconds": ["request_wall": 2.5]]
            }
            if state == "failed" { value["error"] = "fixture failure" }
            return value
        }
        func page(_ values: [[String: Any]], total: Int, next: Int? = nil) -> [String: Any] {
            ["jobs": values, "total": total, "next_offset": next.map { $0 as Any } ?? NSNull()]
        }
        let imagePath = root.appendingPathComponent("image.png").path
        let image = try LocalAPIJob(job("image", output: imagePath))
        try check(image.isImage && image.outputURL?.path == imagePath && image.elapsedSeconds == 2.5,
                  "Successful image metadata was not available for preview/import")
        let video = try LocalAPIJob(job("video", output: root.appendingPathComponent("video.mp4").path, operation: "video.generate"))
        try check(video.outputURL != nil && !video.isImage, "Video result was mistaken for an image preview")
        for state in ["queued", "running", "cancelling", "failed", "cancelled", "interrupted"] {
            let value = try LocalAPIJob(job("incomplete", state: state, output: imagePath))
            try check(value.outputURL == nil && !value.isImage, "Incomplete job exposed a proposed/unpublished output")
        }
        for path in ["https://example.invalid/result.png", "relative.png", "/result\0.png"] {
            try check(try LocalAPIJob(job("bad-path", output: path)).outputURL == nil, "Nonlocal/invalid result path was trusted")
        }
        var proposed = job("proposed")
        try check(try LocalAPIJob(proposed).outputURL == nil, "request.output was mistaken for a successful result")
        proposed["result"] = ["schema_version": true, "output": imagePath, "model": "qwen-image-2.1", "operation": "image.edit"]
        try check(try LocalAPIJob(proposed).outputURL == nil, "Boolean result schema was accepted")
        await api.start(store: store)
        await api.refreshJobs()
        try check(api.running && api.jobs.isEmpty && api.jobsTotal == 0 && !api.jobsStale && api.jobsError == nil,
                  "Real empty service history did not decode as a confirmed empty page")
        let initialPage = page([job("image", output: imagePath), job("failed", state: "failed"),
                                job("video", output: root.appendingPathComponent("video.mp4").path, operation: "video.generate")], total: 13, next: 3)
        let initialReply = transport.pause(result: initialPage)
        let initialTask = Task { @MainActor in await api.refreshJobs(offset: 0) }
        try await wait("initial jobs response", until: { initialReply.started })
        initialReply.release(); await initialTask.value
        try check(api.jobs.count == 3 && api.jobsTotal == 13 && api.jobsNextOffset == 3 && !api.jobsStale && api.jobsUpdatedAt != nil,
                  "Jobs page metadata did not publish")
        try check(store.jobs.isEmpty, "API read-only history was inserted into creation history")
        let savedJobs = api.jobs, savedDate = api.jobsUpdatedAt
        let requestsBefore = transport.requests.count
        await api.refreshJobs(offset: -1); await api.refreshJobs(offset: Int.max)
        try check(transport.requests.count == requestsBefore && api.jobs == savedJobs && api.jobsStale,
                  "Invalid page bound sent an RPC or cleared cached jobs")
        let failedPage = transport.pause(fail: true)
        let failedTask = Task { @MainActor in await api.refreshJobs() }
        try await wait("failed jobs response", until: { failedPage.started })
        failedPage.release(); await failedTask.value
        try check(api.jobs == savedJobs && api.jobsUpdatedAt == savedDate && api.jobsStale && api.jobsError != nil && !api.jobsLoading,
                  "Disconnected history was presented as an empty/fresh page")
        let invalidPages: [[String: Any]] = [
            ["jobs": [], "total": true, "next_offset": NSNull()],
            ["jobs": [job("one")], "total": 2, "next_offset": 0],
            ["jobs": [job("duplicate"), job("duplicate")], "total": 2, "next_offset": NSNull()],
            ["jobs": [], "total": 0],
        ]
        for invalid in invalidPages {
            let reply = transport.pause(result: invalid)
            let task = Task { @MainActor in await api.refreshJobs(offset: 0) }
            try await wait("invalid jobs response", until: { reply.started })
            reply.release(); await task.value
            try check(api.jobs == savedJobs && api.jobsStale && api.jobsError != nil, "Malformed page erased or refreshed history")
        }
        // A slower page-zero reply must not overwrite a newer page, even if
        // both calls target the same service lifecycle.
        let slowPage = transport.pause(result: initialPage)
        let slowTask = Task { @MainActor in await api.refreshJobs(offset: 0) }
        try await wait("slow old page", until: { slowPage.started })
        let latestPage = transport.pause(result: page([job("latest", output: imagePath)], total: 11))
        let latestTask = Task { @MainActor in await api.refreshJobs(offset: 10) }
        try await wait("latest page", until: { latestPage.started })
        latestPage.release(); await latestTask.value
        slowPage.release(); await slowTask.value
        try check(api.jobs.map(\.id) == ["latest"] && api.jobsOffset == 10 && !api.jobsStale && api.jobsError == nil,
                  "Slow earlier page overwrote the latest page")
        for fail in [false, true] {
            let cache = api.jobs
            let oldPage = transport.pause(fail: fail, result: initialPage)
            let oldTask = Task { @MainActor in await api.refreshJobs(offset: 0) }
            try await wait("old lifecycle jobs", until: { oldPage.started })
            api.stop()
            try await wait("history service stop", until: { !api.changing && !api.running })
            try check(api.jobs == cache && api.jobsStale && !api.jobsLoading, "Stop erased history cache or left loading locked")
            await api.start(store: store)
            let fresh = transport.pause(result: page([], total: 0))
            let freshTask = Task { @MainActor in await api.refreshJobs(offset: 0) }
            try await wait("replacement history", until: { fresh.started })
            oldPage.release(); await oldTask.value
            try check(api.jobsLoading && api.jobs == cache && api.jobsError == nil,
                      "Old history success/error/defer polluted or unlocked the replacement page")
            fresh.release(); await freshTask.value
            try check(api.jobs.isEmpty && api.jobsTotal == 0 && api.jobsOffset == 0 && api.jobsNextOffset == nil && !api.jobsStale,
                      "Fresh confirmed empty page did not replace stale history")
        }
        api.stop()
        try await wait("history tests stop", until: { !api.changing && !api.running })

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
        print("PASS: App API lifecycle, execution ownership, history/result validation, safe pagination, retained stale history, slow-page and restarted-service isolation")
    }
}
