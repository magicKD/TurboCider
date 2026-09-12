import Foundation
import Combine
import Darwin

@MainActor final class LocalAPIController: ObservableObject {
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
    init(directory: URL, executable: URL? = nil, socketPath: String? = nil) {
        self.directory = directory.appendingPathComponent("api-service")
        self.socketPath = socketPath ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("turbocider-app-\(getuid()).sock").path
        self.executable = executable ?? Bundle.main.executableURL!.deletingLastPathComponent().appendingPathComponent("turbocider")
    }
    func start(store: NativeJobStore) async {
        guard !changing, process == nil, !store.busy else { return }
        changing = true; error = nil; status = "正在启动…"; store.externalServiceActive = true
        defer { changing = false; if process == nil { store.externalServiceActive = false } }
        do {
            if store.canUnload { try await store.unload() }
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
                    self.running = false; self.changing = false; self.process = nil; self.activeJob = ""; store?.externalServiceActive = false
                    self.sessionModel = ""; self.externalWorkerActive = false
                    self.status = child.terminationStatus == 0 ? "服务已停止" : "服务已退出"
                    if child.terminationStatus != 0 { self.error = (try? String(contentsOf: log, encoding: .utf8)).map { String($0.suffix(2000)) } }
                    try? self.logHandle?.close(); self.logHandle = nil
                }
            }
            process = child
            do { try child.run() } catch { process = nil; try? output.close(); logHandle = nil; throw error }
            for _ in 0..<50 {
                guard child.isRunning else { throw NativeFailure(message: "服务未能启动，请查看日志。") }
                if FileManager.default.fileExists(atPath: socketPath) {
                    do {
                        let response = try await rpc(["action": "service_status"])
                        let info = response["result"] as? [String: Any]
                        guard info?["pid"] as? Int == Int(child.processIdentifier) else { throw NativeFailure(message: "Socket 已由其他进程使用。") }
                        running = true; store.externalServiceActive = true; status = "仅当前用户可访问 · Unix socket"; return
                    } catch {}
                }
                try await Task.sleep(for: .milliseconds(100))
            }
            child.terminate(); throw NativeFailure(message: "服务启动超时。")
        } catch {
            if let process, process.isRunning { process.terminate() }
            self.error = error.localizedDescription; status = "启动失败"
        }
    }
    func stop() {
        guard let process, process.isRunning else { running = false; return }
        changing = true; status = "正在停止任务并释放模型…"; process.terminate()
        Task { [weak self] in
            while process.isRunning { try? await Task.sleep(for: .milliseconds(200)) }
            self?.changing = false
        }
    }
    func terminateNow() { process?.terminate() }
    func refresh() async {
        guard running else { return }
        do {
            let raw = try await rpc(["action": "service_status"])
            let result = raw["result"] as? [String: Any]
            jobCount = result?["history_count"] as? Int ?? 0
            activeJob = result?["active_job"] as? String ?? ""; error = nil
            sessionModel = result?["session_model"] as? String ?? ""
            externalWorkerActive = result?["external_worker_active"] as? Bool ?? false
        } catch { self.error = error.localizedDescription }
    }
    private func rpc(_ value: [String: Any]) async throws -> [String: Any] {
        let data = try JSONSerialization.data(withJSONObject: value)
        let response = try await LibraryTool.run(["rpc", socketPath, "{request}"], payload: data, executable: executable)
        guard let raw = try JSONSerialization.jsonObject(with: response) as? [String: Any] else { throw NativeFailure(message: "服务返回了无效响应。") }
        guard raw["ok"] as? Bool == true else { throw NativeFailure(message: raw["error"] as? String ?? "服务返回了错误响应。") }
        return raw
    }
}
