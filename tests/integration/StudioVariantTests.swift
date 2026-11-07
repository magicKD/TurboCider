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
        var inactiveBudget = studio.draft
        inactiveBudget.qwen21WeightCacheBudget = "custom"; inactiveBudget.qwen21WeightCacheCustomMiB = -1
        try check(try inactiveBudget.request(output: root.appendingPathComponent("other.png")).qwen21_stream_weight_budget_bytes == nil,
            "Qwen cache settings affected another model")
        let result = #"{"checkpoint":"z_image_turbo_int8_convrot.safetensors","plan":{"execution":"gpu"}}"#
        let job = NativeJob(id: UUID(), createdAt: Date(), request: request, state: "succeeded", phase: "export",
                            completed: 1, total: 1, elapsed: 1, resultJSON: result, modelPath: int8.path)
        try check(job.routeSummary == "GPU · INT8 ConvRot", "INT8 result was mislabeled as BF16")
        let qwen = root.appendingPathComponent("qwen-turbo-convrot")
        for leaf in ["diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                     "text_encoders/qwen3vl_8b_int8_convrot.safetensors"] {
            let file = qwen.appendingPathComponent(leaf)
            try fm.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(repeating: 0, count: 16).write(to: file)
        }
        studio.selectModel("qwen-image-2.1")
        studio.draft.prompt = "preserve Qwen prompt"
        studio.draft.steps = 40; studio.draft.promptEnhance = true
        studio.draft.residency = "component_staged"; studio.draft.qwen21DiTCache = "conservative"
        studio.draft.loras = [StudioLoRA(path: "/test/base-adapter.safetensors")]
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: "/test/base.json")
        let turboInstall = LibraryInstallation(id: "turbo", modelID: "qwen-image-2.1", name: "Turbo INT8",
            path: qwen.path, managed: true, components: [:], createdAt: Date())
        studio.applyLibraryInstallation(turboInstall, replace: false)
        try check(studio.draft.hasQwen21TurboCheckpoint && studio.draft.hasQwen21ConvrotCheckpoint &&
            studio.draft.steps == 8 && studio.draft.residency == "streamed" && !studio.draft.usesANE &&
            !studio.draft.promptEnhance && studio.draft.qwen21DiTCache == "off",
            "A model download did not activate Turbo's eight-step streamed GPU recipe in the open draft")
        try check(studio.draft.prompt == "preserve Qwen prompt" && studio.draft.loras.count == 1 &&
            studio.draft.activeLoRAs.isEmpty, "Turbo selection lost prompt/adapter choices instead of disabling incompatible LoRA")
        if let adapter = studio.draft.loras.first {
            studio.setLoRAEnabled(adapter.id, enabled: true)
            try check(studio.draft.activeLoRAs.isEmpty, "Turbo draft accepted enabling a saved base adapter")
        }
        let qwenRequest = try studio.draft.request(output: root.appendingPathComponent("qwen.png"))
        try check(qwenRequest.model == "qwen-image-2.1" && qwenRequest.steps == 8 && qwenRequest.allow_approximation == true && qwenRequest.execution == "gpu" &&
            qwenRequest.residency == "streamed" && qwenRequest.qwen21_stream_weight_budget_bytes == nil,
            "ConvRot App request omitted approximate execution or authored Turbo steps")
        let oldDraft = try JSONDecoder().decode(StudioDraft.self, from: Data("{}".utf8))
        try check(oldDraft.qwen21WeightCacheBudget == "minimum" && oldDraft.qwen21WeightCacheCustomMiB == 3072,
            "A saved draft without cache settings lost the minimum-residency default")
        for preset in Qwen21WeightCacheBudget.allCases where preset != .custom {
            var cached = studio.draft; cached.qwen21WeightCacheBudget = preset.rawValue
            let request = try cached.request(output: root.appendingPathComponent("cached.png"))
            let expected = UInt64(preset.fixedMiB) << 20
            try check(request.qwen21_stream_weight_budget_bytes == (expected == 0 ? nil : expected),
                "A weight budget preset did not survive the App request")
            let v2 = NativeRequestV2(legacy: request)
            let roundTrip = try JSONDecoder().decode(NativeRequestV2.self, from: JSONEncoder().encode(v2))
            try check(roundTrip.execution.qwen21_stream_weight_budget_bytes == request.qwen21_stream_weight_budget_bytes,
                "V2 execution lost the streamed weight budget")
            try check(expected == 0 || roundTrip.execution.residency == "streamed",
                "V2 dropped the residency required by a positive cache budget")
        }
        var custom = studio.draft; custom.qwen21WeightCacheBudget = "custom"
        custom.qwen21WeightCacheCustomMiB = 3584
        let customRequest = try custom.request(output: root.appendingPathComponent("custom.png"))
        try check(customRequest.qwen21_stream_weight_budget_bytes == 3584 << 20,
            "Custom MiB budget was not converted exactly")
        let historyStudio = StudioState(directory: root.appendingPathComponent("history-studio"))
        let cacheJob = NativeJob(id: UUID(), createdAt: Date(), request: customRequest, state: "succeeded", phase: "complete",
            completed: 8, total: 8, elapsed: 1, resultJSON: "{}", modelPath: qwen.path)
        historyStudio.reuse(cacheJob)
        try check(historyStudio.draft.qwen21WeightCacheBudget == "custom" && historyStudio.draft.qwen21WeightCacheCustomMiB == 3584,
            "Reusing a historical job did not restore its custom cache budget")
        var minimumRequest = customRequest; minimumRequest.qwen21_stream_weight_budget_bytes = nil
        let minimumJob = NativeJob(id: UUID(), createdAt: Date(), request: minimumRequest, state: "succeeded", phase: "complete",
            completed: 8, total: 8, elapsed: 1, resultJSON: "{}", modelPath: qwen.path)
        historyStudio.reuse(minimumJob)
        try check(historyStudio.draft.qwen21WeightCacheBudget == "minimum", "A legacy job inherited a later job's cache budget")
        for invalid in [-1, 0, 1023, 16385, Int.max] {
            custom.qwen21WeightCacheCustomMiB = invalid
            do {
                _ = try custom.request(output: root.appendingPathComponent("invalid.png"))
                throw NativeFailure(message: "An invalid custom cache budget was accepted")
            } catch { try check(error.localizedDescription.contains("1024–16384"), "Wrong custom-budget validation") }
        }
        custom.qwen21WeightCacheBudget = "unknown"
        do {
            _ = try custom.request(output: root.appendingPathComponent("invalid.png"))
            throw NativeFailure(message: "An unknown cache preset was accepted")
        } catch { try check(error.localizedDescription.contains("档位无效"), "Wrong cache-preset validation") }
        custom.residency = "resident"
        try check(try custom.request(output: root.appendingPathComponent("resident.png")).qwen21_stream_weight_budget_bytes == nil,
            "An inactive streaming cache budget leaked into a resident request")
        studio.draft.qwen21WeightCacheBudget = "custom"; studio.draft.qwen21WeightCacheCustomMiB = 3584
        var wrongSteps = studio.draft; wrongSteps.steps = 40
        do {
            _ = try wrongSteps.request(output: root.appendingPathComponent("wrong.png"))
            throw NativeFailure(message: "Turbo checkpoint accepted base sampling steps")
        } catch { try check(error.localizedDescription.contains("固定 8 步"), "Wrong Turbo step validation") }
        var oversized = studio.draft; oversized.width = 768
        do {
            _ = try oversized.request(output: root.appendingPathComponent("large.png"))
            throw NativeFailure(message: "ConvRot streaming accepted an oversized canvas")
        } catch { try check(error.localizedDescription.contains("512×512"), "Wrong streaming canvas validation") }
        studio.draft.steps = 40
        studio.draft.residency = "component_staged"
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: "/test/stale.json")
        studio.save()
        let restoredTurbo = StudioState(directory: root.appendingPathComponent("studio"))
        try check(restoredTurbo.draft.steps == 8 && restoredTurbo.draft.residency == "streamed" &&
            !restoredTurbo.draft.usesANE && restoredTurbo.draft.prompt == "preserve Qwen prompt" &&
            restoredTurbo.draft.qwen21WeightCacheBudget == "custom" && restoredTurbo.draft.qwen21WeightCacheCustomMiB == 3584,
            "Restored Turbo draft kept incompatible base settings")
        restoredTurbo.changeOperation("image.edit")
        try check(restoredTurbo.draft.residency == "resident", "Image editing reused the text-only streamed route")
        restoredTurbo.changeOperation("image.generate")
        try check(restoredTurbo.draft.residency == "streamed", "Returning to text generation did not restore streaming")
        let base = root.appendingPathComponent("qwen-base")
        let baseDiT = base.appendingPathComponent("diffusion_models/qwen_image_2.1_bf16.safetensors")
        try fm.createDirectory(at: baseDiT.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 0, count: 16).write(to: baseDiT)
        let baseInstall = LibraryInstallation(id: "base", modelID: "qwen-image-2.1", name: "Base BF16",
            path: base.path, managed: false, components: [:], createdAt: Date())
        let qwenChoices = Qwen21Installation.choices([baseInstall, turboInstall], currentPath: qwen.path)
        try check(qwenChoices.count == 2 && qwenChoices.contains { $0.path == qwen.path && $0.title.contains("Turbo") && $0.title.contains("8 步") } &&
            Qwen21Installation.displayName(at: qwen.path) == "Qwen Image 2.1 Turbo" &&
            Qwen21Installation.displayName(at: base.path) == "Qwen Image 2.1",
            "Creation version picker lost Turbo/base identity")
        restoredTurbo.applyLibraryInstallation(baseInstall, replace: true)
        try check(!restoredTurbo.draft.hasQwen21TurboCheckpoint && restoredTurbo.draft.steps == 40 &&
            restoredTurbo.draft.residency == "component_staged" && restoredTurbo.draft.prompt == "preserve Qwen prompt",
            "Changing from a streamed Turbo draft to base retained its incompatible settings")
        let streamedRoute = #"{"plan":{"execution":"gpu","residency":"streamed"},"qwen_convrot_w8a8":{"encoder_enabled":true,"dit_enabled":true}}"#
        var qwenJob = NativeJob(id: UUID(), createdAt: Date(), request: qwenRequest, state: "succeeded", phase: "complete",
            completed: 8, total: 8, elapsed: 1, resultJSON: streamedRoute, modelPath: qwen.path)
        try check(qwenJob.routeSummary == "GPU · W8A8 ConvRot · 流式加载", "App mislabeled the measured W8A8 route")
        qwenJob.resultJSON = streamedRoute.replacingOccurrences(of: "true", with: "false")
        try check(qwenJob.routeSummary == "GPU · INT8 ConvRot · 浮点激活 · 流式加载", "App mislabeled float activation ConvRot")
        try Data(repeating: 0, count: 16).write(to: qwen.appendingPathComponent("diffusion_models/qwen_image_2.1_bf16.safetensors"))
        try check(!studio.draft.hasQwen21TurboCheckpoint && studio.draft.hasQwen21ConvrotCheckpoint,
            "App source priority differs from native when base and Turbo coexist")
        try check(!Qwen21Installation.isTurbo(at: qwen.path), "Turbo discovery ignores native checkpoint priority")
        print("PASS: version labels, aliases, external selection, parameter/LoRA persistence, BF16/INT8 ANE switching, residency validation and result precision")
    }
}
