import Foundation
import Combine
import Darwin

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
    let directory: URL
    let socketPath: String
    private let executable: URL
    private var process: Process?
    private var logHandle: FileHandle?
    private var lifecycleID = UUID()
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
