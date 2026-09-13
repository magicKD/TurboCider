import Foundation
import Darwin

private let outputLock = NSLock()
private func emit<T: Encodable>(_ value: T, to handle: FileHandle = .standardOutput) {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    guard var data = try? encoder.encode(value) else { return }
    data.append(0x0a)
    outputLock.lock(); defer { outputLock.unlock() }
    try? handle.write(contentsOf: data)
}
private struct Success<T: Encodable>: Encodable { let ok = true; let result: T }
private struct Failure: Encodable { let ok = false; let error: String }

@main struct ModelLibraryCLI {
    static func run(_ arguments: [String]) async throws {
        var args = arguments
        var root = LibraryStore.defaultRoot
        if let position = args.firstIndex(of: "--root") {
            guard position + 1 < args.count else { throw LibraryFailure(message: "--root requires a directory.") }
            root = URL(fileURLWithPath: args[position + 1], isDirectory: true)
            args.removeSubrange(position...(position + 1))
        }
        guard let command = args.first else { throw LibraryFailure(message: "Use library help for model-library commands.") }
        if command == "help" || command == "--help" {
            print("""
            turbocider library list [--root DIRECTORY]
            turbocider library register MODEL_ID DIRECTORY [--root DIRECTORY]
            turbocider library remove INSTALLATION_ID [--root DIRECTORY]
            turbocider library register-ane MODEL_ID MANIFEST.json [--root DIRECTORY]
            turbocider library remove-ane PARTITION_ID [--root DIRECTORY]
            turbocider library register-lora MODEL_ID FILE [--root DIRECTORY]
            turbocider library remove-lora LORA_ID [--root DIRECTORY]
            turbocider library plan REQUEST.json [--root DIRECTORY]
            turbocider library download REQUEST.json [--root DIRECTORY]
            turbocider library configure DIRECTORY
            turbocider library resolve MODEL_ID [--root DIRECTORY]
            turbocider library import CONFIG.json [--root DIRECTORY]
            turbocider library shared-text MODEL_DIRECTORY
            turbocider library inspect MODEL_ID MODEL_DIRECTORY

            REQUEST.json: {"modelID":"z-image-turbo","repository":"owner/repo",
              "provider":"modelscope","revision":"master","include":[],"components":{}}
            Provider defaults to ModelScope. Set HF_TOKEN or MODELSCOPE_API_TOKEN
            for gated repositories. plan reads metadata only; download installs files.
            remove forgets a registration and retains all model files.
            App and CLI share TURBOCIDER_MODEL_LIBRARY or the default Models directory.
            """)
            return
        }
        if command == "cache" {
            switch args.dropFirst().first ?? "help" {
            case "inventory" where args.count == 2: emit(Success(result: try TensorCache.standard.inventory()))
            case "settings" where args.count == 2: emit(Success(result: try TensorCacheSettings.read()))
            case "dump-directories" where args.count == 2: emit(Success(result: try DiagnosticTensorRegistry.read()))
            case "register-dumps" where args.count == 3, "forget-dumps" where args.count == 3:
                emit(Success(result: try DiagnosticTensorRegistry.update(URL(fileURLWithPath: args[2]), remove: args[1] == "forget-dumps")))
            case "retain" where args.count == 3:
                guard let days = Int(args[2]) else { throw LibraryFailure(message: "Retention requires days (0 means keep).") }
                let value = TensorCacheSettings(retentionDays: days); try value.save(); emit(Success(result: value))
            case "prune" where args.count == 3:
                guard let days = Int(args[2]) else { throw LibraryFailure(message: "Prune requires minimum age in days (0 means all recognized entries).") }
                emit(Success(result: try TensorCache.standard.prune(olderThanDays: days)))
            case "help": print("turbocider cache inventory | settings | retain DAYS | prune DAYS | dump-directories | register-dumps DIRECTORY | forget-dumps DIRECTORY\nManage native LTX conditioning and explicitly registered FLUX/Z-Image dump tensors. retain 0 disables automatic cleanup. prune 0 removes recognized entries. Registration is nonrecursive; forgetting retains files. No model weights or media are removed.")
            default: throw LibraryFailure(message: "Use turbocider cache help.")
            }
            return
        }
        let store = try LibraryStore(root: root)
        switch command {
        case "location" where args.count == 1:
            emit(Success(result: ["root": store.root.path]))
        case "configure" where args.count == 2:
            try LibrarySettings.configure(URL(fileURLWithPath: args[1], isDirectory: true))
            emit(Success(result: ["root": LibraryStore.defaultRoot.path]))
        case "resolve" where args.count == 2:
            guard let item = try store.read().installations.last(where: { $0.modelID == args[1] && FileManager.default.fileExists(atPath: $0.path) }) else {
                throw LibraryFailure(message: "No available registered installation for \(args[1]).")
            }
            emit(Success(result: item))
        case "shared-text" where args.count == 2:
            emit(Success(result: try SharedTextComponents.inspect(URL(fileURLWithPath: args[1]))))
        case "inspect" where args.count == 3:
            emit(Success(result: InstallationInspection.inspect(modelID: args[1], root: URL(fileURLWithPath: args[2]))))
        case "import" where args.count == 2:
            let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[1]))) as? [String: Any]
            guard let paths = raw?["modelPaths"] as? [String: String] else { throw LibraryFailure(message: "Configuration must contain modelPaths.") }
            var imported: [LibraryInstallation] = []; var errors: [String: String] = [:]
            for (model, path) in paths.sorted(by: { $0.key < $1.key }) where !path.isEmpty {
                do { imported.append(try store.register(modelID: model, path: URL(fileURLWithPath: path))) }
                catch { errors[model] = error.localizedDescription }
            }
            if let loras = raw?["loras"] as? [[String: String]] {
                for item in loras {
                    guard let model = item["modelID"], let path = item["path"] else {
                        errors["loras"] = "LoRA 需要 modelID 和 path。"; continue
                    }
                    do { _ = try store.registerLoRA(modelID: model, path: URL(fileURLWithPath: path)) }
                    catch { errors[path] = error.localizedDescription }
                }
            }
            if let partitions = raw?["anePartitions"] as? [[String: String]] {
                for item in partitions {
                    guard let model = item["modelID"], let path = item["path"] else {
                        errors["anePartitions"] = "ANE 分区需要 modelID 和 path。"; continue
                    }
                    do { _ = try store.registerANE(modelID: model, manifest: URL(fileURLWithPath: path)) }
                    catch { errors[path] = error.localizedDescription }
                }
            }
            struct ImportResult: Encodable { var installations: [LibraryInstallation]; var errors: [String: String] }
            emit(Success(result: ImportResult(installations: imported, errors: errors)))
        case "register-ane" where args.count == 3:
            emit(Success(result: try store.registerANE(modelID: args[1], manifest: URL(fileURLWithPath: args[2]))))
        case "remove-ane" where args.count == 2:
            try store.unregisterANE(id: args[1]); emit(Success(result: ["removed": args[1]]))
        case "register-lora" where args.count == 3:
            emit(Success(result: try store.registerLoRA(modelID: args[1], path: URL(fileURLWithPath: args[2]))))
        case "remove-lora" where args.count == 2:
            try store.unregisterLoRA(id: args[1]); emit(Success(result: ["removed": args[1]]))
        case "list" where args.count == 1:
            emit(Success(result: try store.read()))
        case "catalog" where args.count == 1:
            emit(Success(result: LibraryRecipe.all))
        case "register" where args.count == 3:
            emit(Success(result: try store.register(modelID: args[1], path: URL(fileURLWithPath: args[2]))))
        case "remove" where args.count == 2:
            try store.unregister(id: args[1]); emit(Success(result: ["removed": args[1], "files": "retained"]))
        case "plan" where args.count == 2, "download" where args.count == 2:
            let data = try Data(contentsOf: URL(fileURLWithPath: args[1]))
            let request = try JSONDecoder().decode(LibraryDownloadRequest.self, from: data)
            let token = ProcessInfo.processInfo.environment[request.provider == .modelscope ? "MODELSCOPE_API_TOKEN" : "HF_TOKEN"]
            let downloader = LibraryDownloader(store: store, client: HubClient(provider: request.provider, token: token))
            if command == "plan" { emit(Success(result: try await downloader.plan(request))) }
            else { emit(Success(result: try await downloader.install(request) { emit($0, to: .standardError) })) }
        default: throw LibraryFailure(message: "Invalid model-library arguments. Use: turbocider library help")
        }
    }

    static func main() async {
        let work = Task { try await run(Array(CommandLine.arguments.dropFirst())) }
        signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN)
        let sources = [SIGINT, SIGTERM].map { value -> DispatchSourceSignal in
            let source = DispatchSource.makeSignalSource(signal: value, queue: .global())
            source.setEventHandler { work.cancel() }; source.resume(); return source
        }
        let parent = ProcessInfo.processInfo.environment["TURBOCIDER_LIBRARY_PARENT_PID"].flatMap(Int32.init)
        let parentWatch = Task.detached {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                if let parent, getppid() != parent {
                    work.cancel()
                    try? await Task.sleep(for: .seconds(5))
                    exit(2)
                }
            }
        }
        defer { for source in sources { source.cancel() }; parentWatch.cancel() }
        do { try await work.value }
        catch {
            let cancelled = error is CancellationError || (error as? URLError)?.code == .cancelled
            emit(Failure(error: cancelled ? "Cancelled; completed verified files are reusable." : error.localizedDescription), to: .standardError)
            exit(cancelled ? 2 : 1)
        }
    }
}
