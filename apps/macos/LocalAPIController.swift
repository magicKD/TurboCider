import Foundation
import Combine
import Darwin

struct LocalAPIJob: Identifiable, Equatable {
    let id: String
    let state: String
    let createdAt: Date
    let model: String
    let operation: String
    let prompt: String
    let error: String?
    let phase: String?
    let outputURL: URL?
    let isImage: Bool
    let elapsedSeconds: Double?

    var stateTitle: String {
        ["queued": "排队中", "running": "执行中", "cancelling": "取消中", "succeeded": "已完成",
         "failed": "失败", "cancelled": "已取消", "interrupted": "已中断"][state] ?? state
    }
    init(_ value: [String: Any]) throws {
        guard let id = value["id"] as? String, !id.isEmpty, !id.contains("\0"),
              let state = value["state"] as? String,
              ["queued", "running", "cancelling", "succeeded", "failed", "cancelled", "interrupted"].contains(state),
              let timestamp = value["created_at"] as? NSNumber,
              CFGetTypeID(timestamp) != CFBooleanGetTypeID(), timestamp.doubleValue.isFinite,
              (0...253402300799).contains(timestamp.doubleValue) else {
            throw NativeFailure(message: "API 返回了无效任务记录。")
        }
        self.id = id; self.state = state; createdAt = Date(timeIntervalSince1970: timestamp.doubleValue)
        let request = value["request"] as? [String: Any] ?? [:]
        let result = value["result"] as? [String: Any] ?? [:]
        model = request["model"] as? String ?? result["model"] as? String ?? "未知模型"
        operation = result["operation"] as? String ?? request["operation"] as? String ?? ""
        let textInputs = request["inputs"] as? [[String: Any]] ?? []
        prompt = request["prompt"] as? String ?? textInputs.first(where: { $0["role"] as? String == "prompt" })?["text"] as? String ?? ""
        error = [value["error"] as? String, value["storage_error"] as? String].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: "\n").nilIfEmpty
        phase = (value["progress"] as? [String: Any])?["phase"] as? String
        let timings = result["timings_seconds"] as? [String: Any]
        if let seconds = timings?["request_wall"] as? NSNumber,
           CFGetTypeID(seconds) != CFBooleanGetTypeID(), seconds.doubleValue.isFinite, seconds.doubleValue >= 0 {
            elapsedSeconds = seconds.doubleValue
        } else { elapsedSeconds = nil }
        // Never open a queued/failed request's proposed output or a remote URL.
        // Result metadata is authoritative; request.output alone is insufficient.
        if state == "succeeded", let version = result["schema_version"] as? NSNumber,
           CFGetTypeID(version) != CFBooleanGetTypeID(), version.doubleValue == 1,
           let path = result["output"] as? String, path.hasPrefix("/"), !path.contains("\0"),
           let resultModel = result["model"] as? String, !resultModel.isEmpty,
           let resultOperation = result["operation"] as? String, !resultOperation.isEmpty {
            outputURL = URL(fileURLWithPath: path)
        } else { outputURL = nil }
        let imageExtensions = ["png", "jpg", "jpeg", "webp", "heic", "heif", "tif", "tiff", "bmp", "avif"]
        if let outputURL {
            isImage = operation.hasPrefix("image.") && imageExtensions.contains(outputURL.pathExtension.lowercased())
        } else { isImage = false }
    }
}

private extension String { var nilIfEmpty: String? { isEmpty ? nil : self } }

@MainActor final class LocalAPIController: ObservableObject {
    typealias RPCTransport = @MainActor ([String: Any]) async throws -> [String: Any]
    @Published private(set) var running = false
    @Published private(set) var changing = false
    @Published private(set) var status = "服务未启动"
    @Published private(set) var jobCount = 0
    @Published private(set) var activeJob = ""
    @Published private(set) var sessionModel = ""
    @Published private(set) var externalWorkerActive = false
    @Published var error: String?
    @Published private(set) var jobs: [LocalAPIJob] = []
    @Published private(set) var jobsTotal: Int?
    @Published private(set) var jobsOffset = 0
    @Published private(set) var jobsNextOffset: Int?
    @Published private(set) var jobsLoading = false
    @Published private(set) var jobsStale = true
    @Published private(set) var jobsError: String?
    @Published private(set) var jobsUpdatedAt: Date?
    static let jobsPageSize = 10
    let directory: URL
    let socketPath: String
    private let executable: URL
    private var process: Process?
    private var logHandle: FileHandle?
    private var lifecycleID = UUID()
    private var jobsRequestID = UUID()
    private weak var ownedStore: NativeJobStore?
    private let rpcTransport: RPCTransport?
    init(directory: URL, executable: URL? = nil, socketPath: String? = nil,
         rpcTransport: RPCTransport? = nil) {
        self.directory = directory.appendingPathComponent("api-service")
        self.socketPath = socketPath ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("turbocider-app-\(getuid()).sock").path
        self.executable = executable ?? Bundle.main.executableURL!.deletingLastPathComponent().appendingPathComponent("turbocider")
        self.rpcTransport = rpcTransport
    }
    func start(store: NativeJobStore) async {
        guard !changing, process == nil, !store.busy else { return }
        let generation = UUID(); lifecycleID = generation; ownedStore = store
        invalidateJobs()
        changing = true; error = nil; status = "正在启动…"; store.externalServiceActive = true
        defer {
            if lifecycleID == generation {
                changing = process != nil && !running
                if process == nil { store.externalServiceActive = false; ownedStore = nil }
            }
        }
        do {
            if store.canUnload { try await store.unload() }
            guard lifecycleID == generation else { return }
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let log = directory.appendingPathComponent("service.log")
            FileManager.default.createFile(atPath: log.path, contents: nil)
            let output = try FileHandle(forWritingTo: log)
            logHandle = output
            let child = Process(); child.executableURL = executable
            child.arguments = ["serve", socketPath, directory.appendingPathComponent("jobs").path]
            var environment = ProcessInfo.processInfo.environment
            environment["TURBOCIDER_SERVICE_PARENT_PID"] = String(getpid()); child.environment = environment
            child.standardOutput = output; child.standardError = output
            child.terminationHandler = { [weak self, weak store] child in
                Task { @MainActor in
                    guard let self, self.process === child else { return }
                    self.lifecycleID = UUID()
                    self.invalidateJobs()
                    self.running = false; self.changing = false; self.process = nil; self.activeJob = ""; store?.externalServiceActive = false
                    self.sessionModel = ""; self.externalWorkerActive = false; self.jobCount = 0; self.ownedStore = nil
                    self.status = child.terminationStatus == 0 ? "服务已停止" : "服务已退出"
                    if child.terminationStatus != 0 { self.error = (try? String(contentsOf: log, encoding: .utf8)).map { String($0.suffix(2000)) } }
                    try? self.logHandle?.close(); self.logHandle = nil
                }
            }
            process = child
            do { try child.run() } catch { process = nil; try? output.close(); logHandle = nil; throw error }
            for _ in 0..<50 {
                guard lifecycleID == generation, process === child else { return }
                guard child.isRunning else { throw NativeFailure(message: "服务未能启动，请查看日志。") }
                if FileManager.default.fileExists(atPath: socketPath) {
                    do {
                        let response = try await rpc(["action": "service_status"])
                        guard lifecycleID == generation, process === child, child.isRunning else { return }
                        let info = response["result"] as? [String: Any]
                        guard info?["pid"] as? Int == Int(child.processIdentifier) else { throw NativeFailure(message: "Socket 已由其他进程使用。") }
                        running = true; store.externalServiceActive = true; status = "仅当前用户可访问 · Unix socket"; return
                    } catch {
                        guard lifecycleID == generation, process === child else { return }
                    }
                }
                try await Task.sleep(for: .milliseconds(100))
            }
            child.terminate(); throw NativeFailure(message: "服务启动超时。")
        } catch {
            guard lifecycleID == generation else { return }
            if let process, process.isRunning { process.terminate() }
            self.error = error.localizedDescription; status = "启动失败"
        }
    }
    func stop() {
        lifecycleID = UUID(); running = false
        invalidateJobs()
        guard let process else {
            changing = false; ownedStore?.externalServiceActive = false; ownedStore = nil
            status = "服务已停止"; return
        }
        changing = true; status = "正在停止任务并释放模型…"
        if process.isRunning { process.terminate() }
        // The identity-bound termination handler owns final cleanup. A second
        // stop waiter must never unlock a later service start.
    }
    func terminateNow() { stop() }
    func refresh() async {
        guard running, let child = process else { return }
        let generation = lifecycleID
        do {
            let raw = try await rpc(["action": "service_status"])
            guard lifecycleID == generation, process === child, running, child.isRunning else { return }
            let result = raw["result"] as? [String: Any]
            guard result?["pid"] as? Int == Int(child.processIdentifier) else {
                throw NativeFailure(message: "Socket 已由其他进程使用。")
            }
            jobCount = result?["history_count"] as? Int ?? 0
            activeJob = result?["active_job"] as? String ?? ""; error = nil
            sessionModel = result?["session_model"] as? String ?? ""
            externalWorkerActive = result?["external_worker_active"] as? Bool ?? false
        } catch {
            guard lifecycleID == generation, process === child, running, child.isRunning else { return }
            self.error = error.localizedDescription
            jobsStale = true
        }
    }
    private func invalidateJobs() {
        jobsRequestID = UUID(); jobsLoading = false; jobsStale = true
    }
    private static func pageInteger(_ value: Any?, maximum: Int = Int(Int32.max)) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite, number.doubleValue == floor(number.doubleValue),
              number.doubleValue >= 0, number.doubleValue <= Double(maximum) else { return nil }
        return number.intValue
    }
    func refreshJobs(offset: Int? = nil) async {
        guard running, let child = process, child.isRunning else { jobsStale = true; return }
        let page = offset ?? jobsOffset
        guard (0...Int(Int32.max)).contains(page) else {
            jobsError = "API 历史页码超出支持范围。"; jobsStale = true; return
        }
        let generation = lifecycleID, requestID = UUID(); jobsRequestID = requestID
        jobsLoading = true
        defer {
            if lifecycleID == generation, process === child, jobsRequestID == requestID { jobsLoading = false }
        }
        do {
            let raw = try await rpc(["action": "jobs", "offset": page, "limit": Self.jobsPageSize])
            guard lifecycleID == generation, process === child, running, child.isRunning,
                  jobsRequestID == requestID else { return }
            guard let result = raw["result"] as? [String: Any],
                  let entries = result["jobs"] as? [[String: Any]], entries.count <= Self.jobsPageSize,
                  let total = Self.pageInteger(result["total"], maximum: 10000), entries.count <= total else {
                throw NativeFailure(message: "API 返回了无效任务分页。")
            }
            let next: Int?
            if result["next_offset"] is NSNull { next = nil }
            else {
                guard let value = Self.pageInteger(result["next_offset"]), value > page, value < total,
                      value == page + entries.count else { throw NativeFailure(message: "API 返回了无效下一页位置。") }
                next = value
            }
            let loaded = try entries.map(LocalAPIJob.init)
            guard Set(loaded.map(\.id)).count == loaded.count else {
                throw NativeFailure(message: "API 返回了重复任务记录。")
            }
            jobs = loaded; jobsTotal = total; jobsOffset = total == 0 ? 0 : page; jobsNextOffset = next
            jobsStale = false; jobsError = nil; jobsUpdatedAt = Date()
        } catch {
            guard lifecycleID == generation, process === child, running, child.isRunning,
                  jobsRequestID == requestID else { return }
            jobsError = error.localizedDescription; jobsStale = true
        }
    }
    private func rpc(_ value: [String: Any]) async throws -> [String: Any] {
        if let rpcTransport { return try await rpcTransport(value) }
        let data = try JSONSerialization.data(withJSONObject: value)
        let response = try await LibraryTool.run(["rpc", socketPath, "{request}"], payload: data, executable: executable)
        guard let raw = try JSONSerialization.jsonObject(with: response) as? [String: Any] else { throw NativeFailure(message: "服务返回了无效响应。") }
        guard raw["ok"] as? Bool == true else { throw NativeFailure(message: raw["error"] as? String ?? "服务返回了错误响应。") }
        return raw
    }
}
