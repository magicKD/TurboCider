import Foundation

@main struct StudioVariantTests {
    @MainActor static func main() throws {
        func check(_ value: Bool, _ message: String) throws {
            if !value { throw NativeFailure(message: message) }
        }
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("tc-studio-variants-\(UUID())").resolvingSymlinksInPath()
        let previous = getenv("TURBOCIDER_MODEL_LIBRARY").map { String(cString: $0) }
        setenv("TURBOCIDER_MODEL_LIBRARY", root.appendingPathComponent("library").path, 1)
        defer {
            if let previous { setenv("TURBOCIDER_MODEL_LIBRARY", previous, 1) }
            else { unsetenv("TURBOCIDER_MODEL_LIBRARY") }
            try? fm.removeItem(at: root)
        }
        func installation(_ id: String, filename: String) throws -> LibraryInstallation {
            let model = root.appendingPathComponent(id)
            let file = model.appendingPathComponent("split_files/diffusion_models/" + filename)
            try fm.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(repeating: 1, count: 16).write(to: file)
            return LibraryInstallation(id: id, modelID: "z-image-turbo", name: "Comfy-Org/z_image_turbo", path: model.path,
                                       managed: false, components: [:], createdAt: Date())
        }
        let bf16 = try installation("bf16", filename: "z_image_turbo_bf16.safetensors")
        let int8 = try installation("int8", filename: "z_image_turbo_int8_convrot.safetensors")
        let int8Streaming = AccelerationDiscovery.optimizationEnabled("z_image_int8_streaming")
        var migrated = StudioDraft()
        migrated.modelID = "z-image-turbo"
        migrated.modelPaths["z-image-turbo"] = int8.path
        migrated.residency = "streamed"
        migrated.prompt = "keep this prompt"
        try check(migrated.normalizeZImageResidency(systemJSON: "{}") && migrated.residency == "resident" &&
                  migrated.prompt == "keep this prompt", "Draft moved to an unsupported device did not restore resident loading")
        migrated.modelPaths["z-image-turbo"] = bf16.path
        migrated.residency = "streamed"
        try check(!migrated.normalizeZImageResidency(systemJSON: "{}") && migrated.residency == "streamed",
                  "Draft normalization disabled legacy BF16 streaming")
        for systemJSON in ["{}", #"{"optimization_profile":{"id":"legacy","z_image_int8_streaming":false}}"#,
                           #"{"optimization_profile":{"id":"m5pro24-v1","z_image_suffix_streaming":true}}"#] {
            try check(ZImageInstallation.requiresResident(variantID: "int8-convrot", systemJSON: systemJSON),
                      "INT8 streaming was enabled without the native feature flag")
            try check(!ZImageInstallation.requiresResident(variantID: "bf16", systemJSON: systemJSON),
                      "Device restriction removed legacy BF16 GPU streaming")
        }
        try check(!ZImageInstallation.requiresResident(variantID: "int8-convrot",
            systemJSON: #"{"optimization_profile":{"id":"m5pro24-v1","z_image_int8_streaming":true}}"#),
                  "Measured device did not expose INT8 streaming")
        let alias = root.appendingPathComponent("current-int8")
        try fm.createSymbolicLink(at: alias, withDestinationURL: URL(fileURLWithPath: int8.path))
        var other = bf16; other.modelID = "flux2-klein-4b"; other.path = root.appendingPathComponent("flux").path
        let options = ZImageInstallation.choices([bf16, int8, other], currentPath: alias.path)
        try check(options.count == 2 && options[0].title.hasPrefix("BF16") && options[1].title.hasPrefix("INT8"), "Installed versions were not labeled by their weights")
        try check(options[1].path == alias.path, "Current binding was duplicated or lost from the picker")
        let external = ZImageInstallation.choices([], currentPath: int8.path)
        try check(external.count == 1 && external[0].title.hasPrefix("INT8"), "Unregistered current installation disappeared")
        let secondBF16 = try installation("bf16-copy", filename: "z_image_turbo_bf16.safetensors")
        let copies = ZImageInstallation.choices([bf16, secondBF16], currentPath: bf16.path)
        try check(copies.count == 2 && copies[0].title != copies[1].title, "Duplicate versions cannot be distinguished")

        let library = try LibraryStore()
        func partition(_ installation: LibraryInstallation, filename: String) throws -> String {
            let directory = root.appendingPathComponent("partitions-" + installation.id)
            var artifacts: [String: [String: String]] = [:]
            for block in 0..<32 {
                let name = "block\(block).mlmodelc"
                try fm.createDirectory(at: directory.appendingPathComponent(name), withIntermediateDirectories: true)
                artifacts[String(block)] = ["int8_pc": name]
            }
            let raw: [String: Any] = ["schema_version": 2,
                "shape": ["K":3840, "N":3840, "input_mode":"enumerated", "buckets":[1056,1120,1536]],
                "source": ["checkpoint": installation.path + "/split_files/diffusion_models/" + filename, "checkpoint_bytes":16],
                "artifacts": artifacts]
            let manifest = directory.appendingPathComponent("manifest.json")
            try JSONSerialization.data(withJSONObject: raw).write(to: manifest)
            try library.registerANE(modelID: "z-image-turbo", manifest: manifest)
            return manifest.path
        }
        let bf16Manifest = try partition(bf16, filename: "z_image_turbo_bf16.safetensors")
        let int8Manifest = try partition(int8, filename: "z_image_turbo_int8_convrot.safetensors")
        let studio = StudioState(directory: root.appendingPathComponent("studio"))
        studio.selectModel("z-image-turbo")
        studio.draft.modelPaths["z-image-turbo"] = bf16.path
        studio.draft.prompt = "preserve this prompt"; studio.draft.steps = 8
        studio.draft.width = 768; studio.draft.height = 512
        studio.draft.seedText = "123"; studio.draft.randomSeed = false
        studio.draft.residency = "streamed"; studio.draft.zImageStreamingBudgetGiB = 10
        studio.draft.loras = [StudioLoRA(path: "/test/adapter.safetensors", strength: 0.5)]
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: bf16Manifest)
        studio.selectInstallation(modelID: "z-image-turbo", path: int8.path)
        try check(studio.draft.modelPath == int8.path && studio.draft.residency == (int8Streaming ? "streamed" : "resident"),
                  "INT8 residency did not follow native hardware support")
        try check(studio.draft.prompt == "preserve this prompt" && studio.draft.steps == 8 && studio.draft.width == 768 && studio.draft.height == 512 && studio.draft.seedText == "123" && !studio.draft.randomSeed, "Version switching reset generation parameters")
        try check(studio.draft.loras.count == 1 && studio.draft.loras[0].strength == 0.5, "Version switching lost LoRA settings")
        try check(studio.draft.usesANE && studio.draft.acceleration?.manifest.isEmpty == true && studio.draft.acceleration?.knownManifests?.contains(bf16Manifest) == true, "Version switching reused the previous checkpoint's selected partition")
        let restored = StudioState(directory: root.appendingPathComponent("studio"))
        try check(restored.draft.modelPath == int8.path && restored.draft.steps == 8 && restored.draft.usesANE, "Selected version did not persist")
        studio.draft.loras = []
        func match() -> AccelerationDiscovery.Match? {
            AccelerationDiscovery.find(modelPath: studio.draft.modelPath, preferred: studio.draft.acceleration?.manifest ?? "",
                minimumRows: 1120, modelID: "z-image-turbo", knownManifests: studio.draft.acceleration?.knownManifests ?? [])
        }
        try check(match()?.manifest == int8Manifest, "INT8 did not discover its own registered ANE partition")
        studio.draft.acceleration?.manifest = int8Manifest
        studio.selectInstallation(modelID: "z-image-turbo", path: bf16.path)
        try check(match()?.manifest == bf16Manifest && studio.draft.steps == 8, "Switching back did not select BF16's partition and retain steps")
        studio.draft.acceleration?.policy = "gpu"
        studio.selectInstallation(modelID: "z-image-turbo", path: int8.path)
        try check(!studio.draft.usesANE && studio.draft.accelerationHint.contains("INT8"), "GPU preference or INT8 label was lost")
        studio.draft.residency = "streamed"
        if int8Streaming {
            let streamed = try studio.draft.request(output: root.appendingPathComponent("streamed.png"))
            try check(streamed.residency == "streamed", "INT8 streaming request was not preserved")
        } else {
            do {
                _ = try studio.draft.request(output: root.appendingPathComponent("streamed.png"))
                throw NativeFailure(message: "Unsupported device accepted INT8 streaming")
            } catch { try check(error.localizedDescription.contains("Apple M5 Pro"), "Wrong INT8 device restriction") }
        }
        let nvfp4 = try installation("nvfp4", filename: "z_image_turbo_nvfp4.safetensors")
        var unsupported = studio.draft
        unsupported.modelPaths["z-image-turbo"] = nvfp4.path
        do {
            _ = try unsupported.request(output: root.appendingPathComponent("nvfp4.png"))
            throw NativeFailure(message: "NVFP4 accepted unsupported streaming")
        } catch { try check(error.localizedDescription.contains("仅支持常驻"), "Wrong NVFP4 residency validation") }
        studio.draft.residency = "resident"
        let request = try studio.draft.request(output: root.appendingPathComponent("output.png"))
        let result = #"{"checkpoint":"z_image_turbo_int8_convrot.safetensors","plan":{"execution":"gpu"}}"#
        let job = NativeJob(id: UUID(), createdAt: Date(), request: request, state: "succeeded", phase: "export",
                            completed: 1, total: 1, elapsed: 1, resultJSON: result, modelPath: int8.path)
        try check(job.routeSummary == "GPU · INT8 ConvRot", "INT8 result was mislabeled as BF16")
        print("PASS: version labels, aliases, external selection, parameter/LoRA persistence, BF16/INT8 ANE switching, residency validation and result precision")
    }
}
