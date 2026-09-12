import Foundation
import Combine
import Darwin

enum LibraryTool {
    static func run(_ arguments: [String], payload: Data? = nil, executable: URL? = nil,
                    progress: @escaping @Sendable (LibraryDownloadEvent) -> Void = { _ in }) async throws -> Data {
        let worker = Task.detached(priority: .utility) { () throws -> Data in
            let fm = FileManager.default
            let directory = fm.temporaryDirectory.appendingPathComponent("tc-library-ui-\(UUID())")
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? fm.removeItem(at: directory) }
            let request = directory.appendingPathComponent("request.json")
            if let payload { try payload.write(to: request) }
            let output = directory.appendingPathComponent("output.json"), error = directory.appendingPathComponent("events.jsonl")
            fm.createFile(atPath: output.path, contents: nil); fm.createFile(atPath: error.path, contents: nil)
            let stdout = try FileHandle(forWritingTo: output), stderr = try FileHandle(forWritingTo: error)
            let reader = try FileHandle(forReadingFrom: error)
            defer { try? stdout.close(); try? stderr.close(); try? reader.close() }
            let process = Process()
            process.executableURL = executable ?? Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("turbocider-library")
            process.arguments = arguments.map { $0 == "{request}" ? request.path : $0 }
            process.standardOutput = stdout; process.standardError = stderr
            var environment = ProcessInfo.processInfo.environment
            environment["TURBOCIDER_LIBRARY_PARENT_PID"] = String(getpid()); process.environment = environment
            try Task.checkCancellation(); try process.run()
            defer { if process.isRunning { kill(process.processIdentifier, SIGKILL) } }
            var pending = Data(), failure = "模型库操作失败。"
            var cancelledAt: ContinuousClock.Instant?
            let start = ContinuousClock.now
            while true {
                if let data = try reader.read(upToCount: 256 * 1024), !data.isEmpty { pending.append(data) }
                while let end = pending.firstIndex(of: 0x0a) {
                    let line = Data(pending[..<end]); pending.removeSubrange(...end)
                    if let event = try? JSONDecoder().decode(LibraryDownloadEvent.self, from: line) { progress(event) }
                    else if let value = try? JSONSerialization.jsonObject(with: line) as? [String: Any], let message = value["error"] as? String { failure = message }
                    else if !line.isEmpty { failure = String(decoding: line.suffix(2000), as: UTF8.self) }
                }
                if pending.count > 64 * 1024 { pending = Data(pending.suffix(64 * 1024)) }
                if !process.isRunning { break }
                if Task.isCancelled, cancelledAt == nil { cancelledAt = .now; process.terminate() }
                if let cancelledAt, cancelledAt.duration(to: .now) > .seconds(5) { kill(process.processIdentifier, SIGKILL); throw CancellationError() }
                // Optional filesystem inspection must not hang behind a file provider.
                if arguments.first != "download", start.duration(to: .now) > .seconds(arguments.first == "plan" ? 180 : 20) {
                    process.terminate(); throw NativeFailure(message: "模型库操作超时，请检查目录访问权限或网络后重试。")
                }
                // A cancelled sleep must not skip the helper's cleanup grace period.
                if Task.isCancelled { await Task.detached { try? await Task.sleep(for: .milliseconds(100)) }.value }
                else { try? await Task.sleep(for: .milliseconds(100)) }
            }
            if Task.isCancelled || process.terminationStatus == 2 { throw CancellationError() }
            if !pending.isEmpty { failure = String(decoding: pending.suffix(2000), as: UTF8.self) }
            guard process.terminationStatus == 0 else { throw NativeFailure(message: failure) }
            return try Data(contentsOf: output)
        }
        return try await withTaskCancellationHandler { try await worker.value } onCancel: { worker.cancel() }
    }
    static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard raw?["ok"] as? Bool == true, let result = raw?["result"] else { throw NativeFailure(message: "模型库返回了无效结果。") }
        return try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: result, options: [.fragmentsAllowed]))
    }
}

@MainActor final class ModelLibraryController: ObservableObject {
    @Published private(set) var root = LibraryStore.defaultRoot.path
    @Published private(set) var installations: [LibraryInstallation] = []
    @Published private(set) var anePartitions: [LibraryANEPartition] = []
    @Published private(set) var loras: [LibraryLoRA] = []
    @Published private(set) var busy = false
    @Published private(set) var event: LibraryDownloadEvent?
    @Published var message: String?
    @Published private(set) var inspections: [String: InstallationInspection] = [:]
    private var operation: Task<Void, Never>?
    private weak var compilationStore: NativeJobStore?
    var canCancel: Bool { busy }

    func cancel() { operation?.cancel(); compilationStore?.cancel() }
    func inspection(modelID: String, path: String) -> InstallationInspection? { inspections[modelID + "\n" + path] }
    @discardableResult private func readInspection(modelID: String, path: String) async throws -> InstallationInspection {
        let data = try await LibraryTool.run(["inspect", modelID, path, "--root", root])
        let report = try LibraryTool.decode(InstallationInspection.self, from: data)
        inspections[modelID + "\n" + path] = report
        return report
    }
    func inspect(modelID: String, path: String) {
        perform { [self] in _ = try await readInspection(modelID: modelID, path: path) }
    }
    private func perform(_ work: @escaping @MainActor () async throws -> Void) {
        guard !busy else { return }
        busy = true; event = nil; message = nil
        operation = Task { [weak self] in
            defer { self?.busy = false; self?.operation = nil }
            do { try await work() }
            catch { self?.message = error is CancellationError ? "已取消；完整且已校验的文件可在重试时复用。" : error.localizedDescription }
        }
    }
    private func readIndex() async throws {
        let data = try await LibraryTool.run(["list", "--root", root])
        let index = try LibraryTool.decode(LibraryIndex.self, from: data)
        installations = index.installations
        loras = index.loras ?? []
        anePartitions = index.anePartitions ?? []
    }
    func refresh(studio: StudioState, migrate: Bool = false) {
        perform { [self] in
            let location = try LibraryTool.decode([String: String].self, from: await LibraryTool.run(["location"]))
            root = location["root"] ?? root
            if migrate {
                // Older App versions stored bindings beside test/job output. Rebuild
                // links from their resolved sources in the persistent library.
                if let path = studio.draft.modelPaths["z-image-turbo"], !path.isEmpty,
                   !path.hasPrefix(root + "/"),
                   FileManager.default.fileExists(atPath: URL(fileURLWithPath: path).appendingPathComponent("installation.json").path),
                   ZImageInstallation.splitDirectory(URL(fileURLWithPath: path)) != nil {
                    let installed = try ZImageInstallation.install(model: URL(fileURLWithPath: path), sharedText: nil,
                        directory: URL(fileURLWithPath: root).appendingPathComponent("bindings"))
                    studio.draft.modelPaths["z-image-turbo"] = installed.path
                    studio.save()
                }
                let data = try JSONEncoder().encode(["modelPaths": studio.draft.modelPaths])
                let response = try await LibraryTool.run(["import", "{request}", "--root", root], payload: data)
                if let raw = try JSONSerialization.jsonObject(with: response) as? [String: Any],
                   let result = raw["result"] as? [String: Any], let errors = result["errors"] as? [String: String], !errors.isEmpty {
                    message = errors.sorted { $0.key < $1.key }.map { "\($0.key)：\($0.value)" }.joined(separator: "\n")
                }
            }
            if migrate {
                for lora in studio.draft.loras where FileManager.default.isReadableFile(atPath: lora.path) {
                    _ = try await LibraryTool.run(["register-lora", studio.draft.modelID, lora.path, "--root", root])
                }
                if let path = studio.draft.modelPaths["z-image-turbo"],
                   let split = ZImageInstallation.splitDirectory(URL(fileURLWithPath: path)) {
                    let files = (try? FileManager.default.contentsOfDirectory(at: split.appendingPathComponent("loras"), includingPropertiesForKeys: nil)) ?? []
                    for file in files.sorted(by: { $0.path < $1.path }) where file.pathExtension == "safetensors" {
                        _ = try await LibraryTool.run(["register-lora", "z-image-turbo", file.path, "--root", root])
                    }
                }
            }
            if migrate, let config = studio.draft.acceleration {
                let paths = Set([config.manifest, config.sourceManifest] + (config.knownManifests ?? []))
                for path in paths.sorted() where !path.isEmpty {
                    // Older settings may retain another model's cache; infer the
                    // architecture from the manifest rather than the active draft.
                    do { _ = try await LibraryTool.run(["register-ane", "auto", path, "--root", root]) }
                    catch { message = "旧 ANE 分区未能登记：\(error.localizedDescription)" }
                }
            }
            try await readIndex()
            // Keep explicit draft choices. New installations fill missing entries only.
            for item in installations where (studio.draft.modelPaths[item.modelID] ?? "").isEmpty {
                studio.draft.modelPaths[item.modelID] = item.path
            }
        }
    }
    func configure(root url: URL, studio: StudioState) {
        perform { [self] in
            let result = try LibraryTool.decode([String: String].self, from: await LibraryTool.run(["configure", url.path]))
            root = result["root"] ?? url.path
            try await readIndex()
            message = "模型库目录已更新。已有模型文件保持原位，可通过文件夹选择或配置重新登记。"
        }
    }
    func remove(_ item: LibraryInstallation, studio: StudioState) {
        perform { [self] in
            _ = try await LibraryTool.run(["remove", item.id, "--root", root])
            if studio.draft.modelPaths[item.modelID] == item.path { studio.draft.modelPaths[item.modelID] = "" }
            try await readIndex(); message = "已移除登记，模型文件仍保留。"
        }
    }
    func importPaths(_ url: URL, studio: StudioState) {
        perform { [self] in
            let response = try await LibraryTool.run(["import", url.path, "--root", root])
            try await readIndex()
            let raw = try JSONSerialization.jsonObject(with: response) as? [String: Any]
            let result = raw?["result"] as? [String: Any]
            if let errors = result?["errors"] as? [String: String], !errors.isEmpty { message = errors.values.sorted().joined(separator: "\n") }
            if let imported = result?["installations"],
               let data = try? JSONSerialization.data(withJSONObject: imported),
               let items = try? JSONDecoder().decode([LibraryInstallation].self, from: data) {
                for item in items { studio.draft.modelPaths[item.modelID] = item.path }
            }
        }
    }
    func registerLoRA(_ url: URL, modelID: String) {
        perform { [self] in
            _ = try await LibraryTool.run(["register-lora", modelID, url.path, "--root", root])
            try await readIndex()
        }
    }
    func removeLoRA(_ item: LibraryLoRA) {
        perform { [self] in
            _ = try await LibraryTool.run(["remove-lora", item.id, "--root", root])
            try await readIndex()
        }
    }
    func discoverLoRAs(modelID: String, path: String) {
        perform { [self] in
            let base = URL(fileURLWithPath: path)
            let directories = [base.appendingPathComponent("loras"), base.appendingPathComponent("split_files/loras"), base.appendingPathComponent("models/loras")]
            for directory in directories {
                let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
                for file in files.sorted(by: { $0.path < $1.path }) where file.pathExtension == "safetensors" {
                    _ = try await LibraryTool.run(["register-lora", modelID, file.path, "--root", root])
                }
            }
            try await readIndex()
            message = "已扫描模型的 loras 目录。登记不自动启用；请确认 LoRA 与基础模型兼容。"
        }
    }
    func exportConfiguration(to url: URL, studio: StudioState) {
        do {
            let object: [String: Any] = ["schemaVersion": 1, "modelPaths": studio.draft.modelPaths,
                "loras": loras.map { ["modelID": $0.modelID, "path": $0.path] },
                "anePartitions": anePartitions.map { ["modelID": $0.modelID, "path": $0.path] }]
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
        } catch { message = error.localizedDescription }
    }
    func registerANE(_ url: URL, modelID: String, studio: StudioState? = nil, select: Bool = false) {
        perform { [self] in
            let data = try await LibraryTool.run(["register-ane", modelID, url.path, "--root", root])
            let item = try LibraryTool.decode(LibraryANEPartition.self, from: data)
            try await readIndex()
            if select, let studio { useANE(item, studio: studio) }
            message = "ANE 分区已登记 · \(item.capacity)。生成时复核基础权重与 LoRA。"
        }
    }
    func useANE(_ item: LibraryANEPartition, studio: StudioState) {
        if studio.draft.modelID != item.modelID { studio.selectModel(item.modelID) }
        var config = studio.draft.acceleration ?? StudioAcceleration()
        config.manifest = item.kind == "compiled" ? item.path : ""
        config.sourceManifest = item.kind == "source" ? item.path : item.sourceManifest ?? ""
        config.policy = "gpu_ane"
        config.automaticVersion = 1
        studio.draft.profilePath = ""
        studio.draft.acceleration = config
        studio.save()
    }
    func removeANE(_ item: LibraryANEPartition, studio: StudioState) {
        perform { [self] in
            _ = try await LibraryTool.run(["remove-ane", item.id, "--root", root])
            if var config = studio.draft.acceleration {
                if config.manifest == item.path { config.manifest = "" }
                if config.sourceManifest == item.path { config.sourceManifest = "" }
                config.knownManifests = config.knownManifests?.filter { $0 != item.path }
                studio.draft.acceleration = config
            }
            try await readIndex()
            message = "已移除 ANE 登记；源分区与编译文件保留。"
        }
    }
    func compileANE(_ item: LibraryANEPartition, store: NativeJobStore) {
        perform { [self] in
            compilationStore = store
            defer { compilationStore = nil }
            let request: [String: Any] = ["action": "compile", "model": item.modelID,
                "source_manifest": item.path, "cache": URL(fileURLWithPath: root).appendingPathComponent("ane-cache").path]
            let data = try await store.coreMLResources(JSONSerialization.data(withJSONObject: request))
            guard let report = try JSONSerialization.jsonObject(with: data) as? [String: Any], let path = report["manifest"] as? String else {
                throw NativeFailure(message: "ANE 编译结果缺少 manifest。")
            }
            _ = try await LibraryTool.run(["register-ane", item.modelID, path, "--root", root])
            try await readIndex()
            message = "编译完成并登记；已存在的兼容缓存可复用。"
        }
    }
    func preview(_ request: LibraryDownloadRequest, completion: @escaping (LibraryDownloadPlan) -> Void) {
        perform { [self] in
            let response = try await LibraryTool.run(["plan", "{request}", "--root", root], payload: JSONEncoder().encode(request))
            completion(try LibraryTool.decode(LibraryDownloadPlan.self, from: response))
        }
    }
    func download(_ request: LibraryDownloadRequest, studio: StudioState) {
        perform { [self] in
            let response = try await LibraryTool.run(["download", "{request}", "--root", root], payload: JSONEncoder().encode(request)) { [weak self] event in
                Task { @MainActor in self?.event = event }
            }
            let result = try LibraryTool.decode(LibraryInstallation.self, from: response)
            try await readIndex()
            studio.draft.modelPaths[result.modelID] = result.path
            message = "所选文件已下载并登记；正在检查执行器所需组件。"
            do {
                let report = try await readInspection(modelID: result.modelID, path: result.path)
                message = "所选文件已下载并登记。\(report.title)。在模型页查看检查详情。"
            } catch { message = "所选文件已下载并登记，安装检查未完成：\(error.localizedDescription)" }
        }
    }
}
