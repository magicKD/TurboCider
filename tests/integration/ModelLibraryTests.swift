import Foundation

@main struct ModelLibraryTests {
    @MainActor static func main() async throws {
        func check(_ value: Bool, _ message: String) throws {
            if !value { throw NativeFailure(message: message) }
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tc-model-switch-\(UUID())")
        defer { try? FileManager.default.removeItem(at: directory) }
        try await ModelLibraryRegistrationRegression.run(directory: directory)
        let studio = StudioState(directory: directory)
        studio.selectModel("z-image-turbo")
        studio.draft.loras = [StudioLoRA(path: "/fixture/z.safetensors", strength: 0.6)]
        studio.selectModel("flux2-klein-4b")
        try check(studio.draft.loras.isEmpty, "Z-Image LoRA leaked into FLUX")
        studio.selectModel("z-image-turbo")
        try check(studio.draft.loras.first?.strength == 0.6, "Model switch lost LoRA settings")
        studio.save()
        let restored = StudioState(directory: directory)
        restored.selectModel("flux2-klein-4b")
        restored.selectModel("z-image-turbo")
        try check(restored.draft.loras.first?.path == "/fixture/z.safetensors", "LoRA association did not survive restart")
        let fm = FileManager.default
        let source = directory.appendingPathComponent("source")
        for folder in ["models/diffusion_models", "models/vae", "text_encoder", "tokenizer"] {
            try fm.createDirectory(at: source.appendingPathComponent(folder), withIntermediateDirectories: true)
        }
        for file in ["models/diffusion_models/z_image_turbo_bf16.safetensors", "models/vae/ae.safetensors", "text_encoder/model.safetensors", "tokenizer/tokenizer.json"] {
            try Data("fixture".utf8).write(to: source.appendingPathComponent(file))
        }
        let oldParent = directory.appendingPathComponent("old-output")
        let old = try ZImageInstallation.install(model: source, sharedText: nil, directory: oldParent)
        let migrated = try ZImageInstallation.install(model: old, sharedText: nil, directory: directory.appendingPathComponent("library/bindings"))
        try fm.removeItem(at: oldParent)
        try check(ZImageInstallation.splitDirectory(migrated) != nil && ZImageInstallation.hasSharedText(migrated), "Migrated binding still depends on old output")
        try fm.removeItem(at: source.appendingPathComponent("models/diffusion_models/z_image_turbo_bf16.safetensors"))
        let inspection = InstallationInspection.inspect(modelID: "z-image-turbo", root: migrated)
        try check(!inspection.issues.isEmpty, "Broken weight link was not diagnosed")
        let catalog = StudioModel.catalog()
        guard let ltx = catalog.first(where: { $0.id == "ltx-2.5-distilled" }),
              let z = catalog.first(where: { $0.id == "z-image-turbo" }),
              let flux = catalog.first(where: { $0.id == "flux2-klein-4b" }),
              let h3 = catalog.first(where: { $0.id == "minimax-h3-turbo" }) else {
            throw NativeFailure(message: "Native catalog is incomplete")
        }
        try check(ltx.acceptsImageInputs &&
                  ltx.availableOperations == ["video.generate", "video.image"],
                  "LTX browser did not expose validated image-to-video")
        try check(!z.acceptsImageInputs && z.availableOperations == ["image.generate"], "Z-Image browser accepts images")
        try check(flux.acceptsImageInputs && h3.acceptsImageInputs, "Validated reference inputs are hidden")
        try check(ltx.canGenerateAudio && !z.canGenerateAudio &&
                  !flux.canGenerateAudio && h3.canGenerateAudio,
                  "Audio gate differs from executable model capabilities")
        try check(z.matchesLibrarySearch("Z-IMAGE turbo", path: ""), "Case-insensitive model search failed")
        try check(z.matchesLibrarySearch("文字", path: ""), "Capability search failed")
        try check(ltx.matchesLibrarySearch("视频", path: ""), "Video search failed")
        try check(z.matchesLibrarySearch("external Models", path: "/external/Models/z"), "Path search failed")
        try check(!z.matchesLibrarySearch("Z-Image nonexistent", path: ""), "Search words were not intersected")
        try check(catalog.allSatisfy { $0.matchesLibrarySearch(" \n ", path: "") }, "Blank search should show all entries")
        let unavailable = StudioModel(id: "fixture", name: "Fixture", executor: false, output: "video", operations: ["video.image"], default_steps: 1, default_frames: 1, default_width: 64, default_height: 64)
        try check(!unavailable.acceptsImageInputs && unavailable.availableOperations.isEmpty, "Disabled executor advertised operations")
        print("PASS: catalog capability gates, text/image distinction, name/path/multilingual search")
    }
}

/// The controller talks to a suspended metadata fixture: no helper process,
/// registry settings, weights, model catalog or inference is used by these cases.
@MainActor private enum ModelLibraryRegistrationRegression {
    @MainActor private final class Helper {
        var calls: [[String]] = []
        var indexes: [String: LibraryIndex] = [:]
        var holdCommand: String?
        var failInspection = false
        var failPaths = Set<String>()
        var failListOnce = false
        var maximumConcurrentCalls = 0
        private var concurrentCalls = 0
        private var held: CheckedContinuation<Void, Error>?
        var isHeld: Bool { held != nil }

        func release() { let continuation = held; held = nil; continuation?.resume(returning: ()) }
        private func response<T: Encodable>(_ result: T) throws -> Data {
            let value = try JSONSerialization.jsonObject(with: JSONEncoder().encode(result))
            return try JSONSerialization.data(withJSONObject: ["ok": true, "result": value])
        }
        func run(_ arguments: [String]) async throws -> Data {
            calls.append(arguments)
            concurrentCalls += 1; maximumConcurrentCalls = max(maximumConcurrentCalls, concurrentCalls)
            defer { concurrentCalls -= 1 }
            if arguments.first == holdCommand {
                holdCommand = nil
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in held = continuation }
            }
            try Task.checkCancellation()
            switch arguments[0] {
            case "inspect":
                if failInspection { throw NativeFailure(message: "fixture inspection failed") }
                return try response(InstallationInspection(modelID: arguments[1], path: arguments[2]))
            case "configure":
                return try response(["root": arguments[1]])
            case "register-lora":
                let modelID = arguments[1], path = arguments[2], root = arguments[4]
                if failPaths.contains(path) { throw NativeFailure(message: "fixture registration failed") }
                var index = indexes[root] ?? LibraryIndex()
                let item = index.loras?.first { $0.modelID == modelID && $0.path == path }
                    ?? LibraryLoRA(id: UUID().uuidString, modelID: modelID, path: path, name: URL(fileURLWithPath: path).lastPathComponent)
                if !(index.loras ?? []).contains(where: { $0.id == item.id }) { index.loras = (index.loras ?? []) + [item] }
                indexes[root] = index
                return try response(item)
            case "list":
                if failListOnce { failListOnce = false; throw NativeFailure(message: "fixture index read failed") }
                return try response(indexes[arguments[2]] ?? LibraryIndex())
            default: throw NativeFailure(message: "Unexpected fixture command: \(arguments)")
            }
        }
    }
    private static func check(_ value: Bool, _ message: String) throws {
        if !value { throw NativeFailure(message: message) }
    }
    private static func wait(_ label: String, until predicate: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while !predicate() {
            guard ContinuousClock.now < deadline else { throw NativeFailure(message: "Controller fixture timed out: \(label)") }
            try await Task.sleep(for: .milliseconds(1))
        }
    }
    static func run(directory: URL) async throws {
        let root = directory.appendingPathComponent("fixture-library")
        let a = URL(fileURLWithPath: "/fixture/a.safetensors"), b = URL(fileURLWithPath: "/fixture/b.safetensors")
        // A failed inspection must still drain additions, without concurrent writes.
        let helper = Helper(); helper.holdCommand = "inspect"; helper.failInspection = true
        defer { helper.release() }
        let controller = ModelLibraryController(root: root, metadataRunner: helper.run)
        controller.inspect(modelID: "qwen-image-2.1", path: "/fixture/model")
        try await wait("held inspection") { helper.isHeld }
        controller.registerLoRA(a, modelID: "qwen-image-2.1")
        controller.registerLoRA(URL(fileURLWithPath: "/fixture/sub/../a.safetensors"), modelID: "qwen-image-2.1")
        controller.registerLoRA(a, modelID: "flux2-klein-4b")
        controller.registerLoRA(b, modelID: "qwen-image-2.1")
        try check(helper.calls.count == 1 && controller.loras.isEmpty, "Busy controller dispatched or optimistically published a registration")
        helper.release()
        try await wait("inspection failure queue drain") { !controller.busy }
        let registrations = helper.calls.filter { $0.first == "register-lora" }
        try check(registrations.map { Array($0[1...2]) } == [
            ["qwen-image-2.1", a.path], ["flux2-klein-4b", a.path], ["qwen-image-2.1", b.path]
        ], "FIFO registration lost a model/path or failed to coalesce pending duplicates")
        try check(controller.loras.count == 3 && helper.calls.filter { $0.first == "list" }.count == 3 && helper.maximumConcurrentCalls == 1,
                  "Queued registrations were not serially published after index reads")
        let previousIDs = Set(controller.loras.map(\.id))
        controller.registerLoRA(a, modelID: "qwen-image-2.1")
        try await wait("already registered adapter") { !controller.busy }
        try check(Set(controller.loras.map(\.id)) == previousIDs, "Re-registering an existing adapter created duplicate metadata")

        // An active duplicate is coalesced too; failed register/list work cannot
        // prevent a later item from publishing or conceal the earlier error.
        let failures = Helper(); failures.holdCommand = "register-lora"; failures.failPaths = [a.path]; failures.failListOnce = true
        defer { failures.release() }
        let failedController = ModelLibraryController(root: root, metadataRunner: failures.run)
        failedController.registerLoRA(a, modelID: "qwen-image-2.1")
        try await wait("held registration") { failures.isHeld }
        failedController.registerLoRA(a, modelID: "qwen-image-2.1")
        failedController.registerLoRA(b, modelID: "qwen-image-2.1")
        let c = URL(fileURLWithPath: "/fixture/c.safetensors")
        failedController.registerLoRA(c, modelID: "qwen-image-2.1")
        failures.release()
        try await wait("failed registration/index queue drain") { !failedController.busy }
        try check(failures.calls.filter { $0.first == "register-lora" }.count == 3 && failedController.loras.count == 2,
                  "Active duplicate or failed queue item blocked later registrations")
        try check(failedController.message?.contains("fixture registration failed") == true && failedController.message?.contains("fixture index read failed") == true,
                  "A later success concealed registration or index failure")

        // Additions during configure follow its resulting root, never the old root.
        let switching = Helper(); switching.holdCommand = "configure"
        defer { switching.release() }
        let switchedController = ModelLibraryController(root: root, metadataRunner: switching.run)
        let studio = StudioState(directory: directory.appendingPathComponent("isolated-draft"), models: [])
        let nextRoot = directory.appendingPathComponent("next-library")
        switchedController.configure(root: nextRoot, studio: studio)
        try await wait("held configure") { switching.isHeld }
        switchedController.registerLoRA(a, modelID: "qwen-image-2.1")
        switching.release()
        try await wait("configure and registration") { !switchedController.busy }
        try check(switchedController.root == nextRoot.path && switchedController.loras.count == 1 && switching.indexes[root.path] == nil,
                  "Queued registration used the previous library after configure")
        try check(switching.calls.filter { $0.first == "register-lora" || $0.first == "list" }.allSatisfy { $0.last == nextRoot.path },
                  "Registration/index reads were split between library roots")

        // Cancel removes queued writes and the interrupted active write. Already
        // completed metadata remains; a fresh addition after cancellation works.
        let cancelling = Helper()
        defer { cancelling.release() }
        let cancelledController = ModelLibraryController(root: root, metadataRunner: cancelling.run)
        cancelledController.registerLoRA(c, modelID: "qwen-image-2.1")
        try await wait("initial completed registration") { !cancelledController.busy }
        cancelling.holdCommand = "register-lora"
        cancelledController.registerLoRA(a, modelID: "qwen-image-2.1")
        try await wait("held cancellable registration") { cancelling.isHeld }
        cancelledController.registerLoRA(b, modelID: "qwen-image-2.1")
        cancelledController.cancel(); cancelling.release()
        try await wait("cancelled registration") { !cancelledController.busy }
        try check(cancelledController.loras.map(\.path) == [c.path] && cancelling.calls.filter { $0.first == "register-lora" }.count == 2,
                  "Cancel dispatched pending additions or removed completed metadata")
        try check(cancelledController.message?.contains("待处理的 LoRA 登记") == true, "Cancelled queue did not explain unfinished registrations")
        cancelledController.registerLoRA(b, modelID: "qwen-image-2.1")
        try await wait("retry after cancellation") { !cancelledController.busy }
        try check(Set(cancelledController.loras.map(\.path)) == [b.path, c.path], "New registration after cancellation did not recover")
        print("PASS: queued LoRA registration FIFO, duplicate/model identity, failure/index visibility, root switching and cancellation")
    }
}
