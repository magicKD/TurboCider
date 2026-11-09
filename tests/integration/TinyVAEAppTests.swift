import Foundation
import AppKit
import Combine
import ImageIO

@main struct TinyVAEAppTests {
    @MainActor static func main() async throws {
        func check(_ condition: @autoclosure () -> Bool, _ reason: String) throws {
            if !condition() { throw NativeFailure(message: reason) }
        }
        var draft = try JSONDecoder().decode(StudioDraft.self, from: Data("{}".utf8))
        try check(
            !draft.tinyPreview && draft.tinyPreviewInterval == 2
                && draft.tinyVAEEncoder == "original", "old drafts must retain quality defaults")
        draft.modelID = "qwen-image-2.1"
        draft.tinyPreview = true
        draft.tinyVAEPaths["taeqi2_1"] = "/missing/tiny.safetensors"
        do {
            _ = try draft.tinyVAEConfiguration()
            throw NativeFailure(message: "accepted missing tiny weights")
        } catch let failure as NativeFailure {
            try check(failure.message.contains("请先"), "wrong missing-model error")
        }
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "tiny-app-\(UUID())")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let weights = temporary.appendingPathComponent("taeqi2_1.safetensors")
        try Data([0]).write(to: weights)
        draft.tinyVAEPaths["taeqi2_1"] = weights.path
        draft.tinyVAEEncoder = "tiny"
        draft.tinyVAEDecoder = "tiny"
        draft.operation = "image.edit"
        let config = try draft.tinyVAEConfiguration()
        try check(
            config?.encoder == "tiny" && config?.decoder == "tiny" && config?.interval == 2,
            "configuration lost selection")
        var legacy = NativeRequest(
            prompt: "test", output: temporary.appendingPathComponent("result.png").path)
        legacy.tiny_vae = config
        let v2 = NativeRequestV2(legacy: legacy)
        let roundtrip = try JSONDecoder().decode(
            NativeRequestV2.self, from: JSONEncoder().encode(v2))
        try check(roundtrip.parameters.tiny_vae == config, "schema 2 dropped tiny settings")
        let restored = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(draft))
        try check(
            restored.tinyVAEPaths == draft.tinyVAEPaths && restored.tinyPreview
                && restored.tinyVAEDecoder == "tiny", "draft persistence dropped settings")
        var wan = StudioDraft()
        wan.modelID = "wan2.2-ti2v-5b-fastwan"
        wan.modelPaths[wan.modelID] = temporary.path
        try check(wan.wan22TinyVAEDecoder == "tiny", "Wan must default to Super decoding")
        let migratedWan = try JSONDecoder().decode(StudioDraft.self, from: Data("{}".utf8))
        try check(migratedWan.wan22TinyVAEDecoder == "tiny", "older drafts must default to Super")
        do {
            _ = try wan.tinyVAEConfiguration()
            throw NativeFailure(message: "Wan accepted missing Super weights")
        } catch let failure as NativeFailure {
            try check(failure.message.contains("不会自动回退"), "Wan missing-weight error must be explicit")
        }
        let wanVAE = temporary.appendingPathComponent("vae")
        try FileManager.default.createDirectory(at: wanVAE, withIntermediateDirectories: false)
        let superWeights = wanVAE.appendingPathComponent("taew2_2_super.safetensors")
        try Data([0]).write(to: superWeights)
        let wanConfig = try wan.tinyVAEConfiguration()
        try check(wanConfig?.path == superWeights.path && wanConfig?.encoder == "original" &&
                  wanConfig?.decoder == "tiny" && wanConfig?.preview == false,
                  "Wan Super must replace only final video decoding")
        let resources = temporary.appendingPathComponent("resources/tiny-vae")
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        let bundledSuper = resources.appendingPathComponent("taew2_2_super.safetensors")
        try Data([1]).write(to: bundledSuper)
        try check(wan.resolveTinyVAEPath(resources: resources.deletingLastPathComponent()) == bundledSuper.path,
                  "Wan did not discover the packaged Super weights")
        wan.wan22TinyVAEDecoder = "original"
        let wanFull = try wan.tinyVAEConfiguration()
        try check(wanFull == nil, "Wan must allow explicitly selecting the full VAE")
        wan.wan22TinyVAEDecoder = "tiny"
        let restoredWan = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(wan))
        try check(restoredWan.wan22TinyVAEDecoder == "tiny", "Wan decoder preference did not persist")
        draft.streaming.selection = .tier12
        do {
            _ = try draft.tinyVAEConfiguration()
            throw NativeFailure(message: "accepted uncalibrated streaming")
        } catch let failure as NativeFailure {
            try check(failure.message.contains("内存档位"), "wrong streaming error")
        }
        let good = NativeEvent(
            sequence: 1, phase: "preview", completed: 2, total: 6, elapsed_seconds: 1,
            preview_path: TinyPreviewPath.expected(output: legacy.output))
        try check(
            TinyPreviewPath.valid(good, output: legacy.output, steps: 6),
            "valid preview path rejected")
        var bad = good
        bad.preview_path = "/other/private.png"
        try check(
            !TinyPreviewPath.valid(bad, output: legacy.output, steps: 6),
            "unrelated preview path accepted")
        try check(
            !TinyPreviewPath.valid(good, output: legacy.output, steps: 4),
            "preview wrong step count accepted")
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 512, pixelsHigh: 512,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        memset(bitmap.bitmapData!, 255, bitmap.bytesPerRow * bitmap.pixelsHigh)
        let frame = bitmap.representation(using: .png, properties: [:])!
        try check(TinyPreviewFrame.matches(frame, width: 512, height: 512),
                  "full-size 512 preview rejected")
        try check(!TinyPreviewFrame.matches(frame, width: 511, height: 512)
                    && !TinyPreviewFrame.matches(Data([0]), width: 512, height: 512),
                  "wrong-size or invalid preview accepted")
        print(
            "Tiny VAE App defaults, persistence, request conversion, memory guard and preview correlation passed"
        )
        let args = CommandLine.arguments
        if args.count == 2 && args[1] == "--bundled" {
            guard let asset = TinyVAEAsset.all.first(where: { $0.id == "taeqi2_1" }),
                  let bundledPath = asset.bundledPath() else {
                throw NativeFailure(message: "bundle fixture is missing TAEQI2.1")
            }
            let library = ModelLibraryController(root: temporary.appendingPathComponent("clean-library"))
            let studio = StudioState(directory: temporary.appendingPathComponent("clean-studio"))
            studio.draft.modelID = asset.modelID
            studio.draft.tinyVAEPaths[asset.id] = weights.path
            try check(library.tinyVAEPath(asset) == nil, "bundle fixture has a downloaded installation")
            library.useTinyVAE(asset, studio: studio)
            studio.draft.tinyPreview = true
            studio.draft.tinyVAEDecoder = "tiny"
            let selected = try studio.draft.tinyVAEConfiguration()
            try check(studio.draft.tinyVAEPaths[asset.id] == nil && selected?.path == bundledPath
                && selected?.preview == true && selected?.decoder == "tiny",
                "bundled selection did not resolve from a clean library")
            print("PASS App bundled TAEQI2.1 discovery, selection and request binding with an empty library")
            return
        }
        if args.count == 3 && args[1] == "--download" {
            let library = ModelLibraryController(root: URL(fileURLWithPath: args[2]))
            let studio = StudioState(directory: temporary.appendingPathComponent("download-ui"))
            for asset in TinyVAEAsset.all {
                let originalModel = studio.draft.modelID
                library.downloadTinyVAE(asset, source: .mirror)
                let started = ContinuousClock.now
                while library.busy {
                    try check(
                        started.duration(to: .now) < .seconds(30),
                        "App download helper did not finish")
                    try await Task.sleep(for: .milliseconds(30))
                }
                guard let path = library.tinyVAEPath(asset) else {
                    throw NativeFailure(
                        message: library.message ?? "App download produced no installation")
                }
                library.useTinyVAE(asset, studio: studio)
                try check(
                    studio.draft.modelID == originalModel
                        && studio.draft.tinyVAEPaths[asset.id] == path,
                    "selecting tiny changed base model or lost the file")
                studio.draft.modelID = asset.modelID
                studio.draft.tinyPreview = true
                let selected = try studio.draft.tinyVAEConfiguration()
                try check(
                    selected?.path == path && selected?.preview == true,
                    "download selection did not enter native request")
            }
            try check(
                library.installations.filter { $0.modelID.hasPrefix("tiny-vae-") }.count == 2,
                "cached download duplicated installs")
            print(
                "PASS App Tiny VAE download helper, verified installs, cached reuse, selection and native request binding"
            )
            return
        }
        let dmd2 = args.count == 7 && args[1] == "--gpu-dmd2"
        let cachedConvrot = args.count == 4 && args[1] == "--gpu-convrot-cache"
        let streamedConvrot = cachedConvrot || (args.count == 4 && args[1] == "--gpu-convrot")
        guard dmd2 || streamedConvrot || (args.count == 6 && ["--gpu", "--gpu-base"].contains(args[1])) else { return }
        let authoredBase = args[1] == "--gpu-base"
        let acceptanceEnvironment = ProcessInfo.processInfo.environment
        let selectedAcceptanceMode = acceptanceEnvironment["TURBOCIDER_TINY_ACCEPTANCE_MODE"]
        let acceptancePrompt = acceptanceEnvironment["TURBOCIDER_TINY_ACCEPTANCE_PROMPT"]
        let acceptanceBaseline = acceptanceEnvironment["TURBOCIDER_TINY_ACCEPTANCE_BASELINE_PNG"]
        if selectedAcceptanceMode != nil || acceptancePrompt != nil || acceptanceBaseline != nil {
            try check(streamedConvrot,
                "Tiny acceptance overrides require explicit --gpu-convrot or --gpu-convrot-cache")
        }
        if let acceptancePrompt {
            try check(!acceptancePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "Tiny acceptance prompt override must be nonempty")
        }
        let modelID = streamedConvrot ? "qwen-image-2.1" : args[2]
        let model = URL(fileURLWithPath: args[streamedConvrot ? 2 : 3]).standardizedFileURL
        let output = URL(fileURLWithPath: args[streamedConvrot ? 3 : 5]).standardizedFileURL
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let streamedStudio: StudioState?
        let tiny: String
        if streamedConvrot {
            let contents = try FileManager.default.contentsOfDirectory(atPath: output.path)
            try check(contents.isEmpty, "ConvRot acceptance requires an empty output directory")
            let studio = StudioState(directory: output)
            let library = ModelLibraryController(root: output.appendingPathComponent("library"))
            studio.selectModel(modelID)
            let registration = output.appendingPathComponent("model-import.json")
            try JSONEncoder().encode(["modelPaths": [modelID: model.path]]).write(to: registration, options: .withoutOverwriting)
            library.importPaths(registration, studio: studio)
            let started = ContinuousClock.now
            while library.busy {
                try check(started.duration(to: .now) < .seconds(30), "Model library import did not finish")
                try await Task.sleep(for: .milliseconds(30))
            }
            try check(library.installations.contains { $0.modelID == modelID && $0.path == model.path } &&
                studio.draft.hasQwen21TurboCheckpoint && studio.draft.hasQwen21ConvrotCheckpoint &&
                studio.draft.steps == 8 && studio.draft.residency == "streamed" && !studio.draft.usesANE,
                "Fresh model registration did not activate the streamed Turbo recipe: \(library.message ?? "")")
            guard let asset = TinyVAEAsset.all.first(where: { $0.id == "taeqi2_1" }),
                  let bundledPath = asset.bundledPath() else {
                throw NativeFailure(message: "ConvRot acceptance requires the bundled TAEQI2.1")
            }
            library.useTinyVAE(asset, studio: studio)
            if cachedConvrot { studio.draft.qwen21WeightCacheBudget = "gib2" }
            tiny = bundledPath
            streamedStudio = studio
        } else {
            tiny = URL(fileURLWithPath: args[4]).standardizedFileURL.path
            streamedStudio = nil
        }
        let store = NativeJobStore(directory: output)
        var reports: [[String: Any]] = []
        var baseline: Data?
        var modes = authoredBase ? ["tiny-final"] : ["baseline", "preview", "cancel", "tiny-final"]
        if modelID == "qwen-image-2.1" && !authoredBase && !streamedConvrot { modes.append("tiny-edit") }
        if let selectedAcceptanceMode {
            try check(modes.contains(selectedAcceptanceMode), "unknown existing Tiny acceptance mode")
            modes = [selectedAcceptanceMode]
        }
        if selectedAcceptanceMode == "preview" {
            guard let acceptanceBaseline else {
                throw NativeFailure(message: "preview-only acceptance requires an explicit baseline PNG")
            }
            let data = try Data(contentsOf: URL(fileURLWithPath: acceptanceBaseline))
            try check(TinyPreviewFrame.matches(data, width: 512, height: 512),
                "preview-only acceptance baseline is not a valid full-size PNG")
            baseline = data
        }
        for mode in modes {
            var request = NativeRequest(
                prompt:
                    acceptancePrompt ?? "A red ceramic teapot on a wooden table, warm sunlight, detailed photography.",
                output: output.appendingPathComponent("\(mode).png").path)
            request.model = modelID
            request.operation = "image.generate"
            request.width = 512
            request.height = 512
            request.steps = modelID == "qwen-image-2.1" ? 8 : 4
            request.audio = false
            request.residency = modelID == "qwen-image-2.1" ? "component_staged" : "resident"
            if dmd2 {
                try check(modelID == "qwen-image-2.1", "DMD2 acceptance is Qwen-only")
                request.steps = 2
                request.qwen21_sampler = "dmd2"
                request.loras = [NativeLoRA(path: args[6], strength: 1, role: "transformer")]
                request.lora_strategy = "inference_time"
                request.allow_approximation = true
            }
            if mode != "baseline" {
                request.tiny_vae = NativeTinyVAEConfig(
                    path: tiny, decoder: mode == "tiny-final" ? "tiny" : "original",
                    preview: true, interval: mode == "tiny-final" ? 3 : 2)
                if mode == "tiny-final" { request.allow_approximation = true }
            }
            if mode == "tiny-edit" {
                request.operation = "image.edit"
                request.inputs = [
                    NativeInput(
                        kind: "image", role: "reference",
                        path: output.appendingPathComponent("baseline.png").path)
                ]
                request.prompt =
                    "Change the teapot in <image1> to blue. Preserve its shape and the background."
                request.tiny_vae?.encoder = "tiny"
                request.allow_approximation = true
            }
            if authoredBase {
                var settings = StudioDraft()
                settings.modelID = modelID
                settings.modelPaths[modelID] = model.path
                settings.prompt = request.prompt
                settings.operation = "image.generate"
                settings.width = 512
                settings.height = 512
                settings.steps = 20
                settings.audio = false
                settings.residency = "component_staged"
                settings.acceleration = StudioAcceleration(policy: "gpu")
                settings.tinyVAEPaths[settings.tinyArchitecture] = tiny
                settings.tinyVAEDecoder = "tiny"
                settings.tinyPreview = true
                request = try settings.request(output: URL(fileURLWithPath: request.output))
                try check(
                    request.allow_approximation == true && request.tiny_vae?.decoder == "tiny"
                        && request.steps == 20,
                    "ordinary Creation draft lost tiny settings/approximation permission")
            }
            if let studio = streamedStudio {
                studio.draft.prompt = request.prompt
                studio.draft.randomSeed = false
                studio.draft.seedText = "73"
                studio.draft.tinyPreview = mode != "baseline"
                studio.draft.tinyPreviewInterval = mode == "tiny-final" ? 3 : 2
                studio.draft.tinyVAEDecoder = mode == "tiny-final" ? "tiny" : "original"
                let resolved = try await store.resolveAcceleration(studio.draft)
                let pair = try resolved.publicStreamingRequest(output: URL(fileURLWithPath: request.output))
                request = pair.legacy
                try check(pair.v2 == nil && request.residency == "streamed" && request.execution == "gpu" &&
                    request.allow_approximation == true && request.steps == 8 && request.seed == 73 &&
                    request.qwen21_stream_weight_budget_bytes == (cachedConvrot ? UInt64(2) << 30 : nil) &&
                    (mode == "baseline" ? request.tiny_vae == nil : request.tiny_vae?.path == tiny),
                    "Creation request lost its ConvRot streaming or bundled preview settings")
            }
            var previews: [Int] = [], phases = Set<String>(), cancelled = false
            var phaseTransitions: [String] = [], stateTransitions: [String] = []
            var invalidPreviewFrame = false
            let progress = store.$jobs.sink { jobs in
                if let job = jobs.first(where: { $0.request.output == request.output }) {
                    phases.insert(job.phase)
                    if phaseTransitions.last != job.phase {
                        phaseTransitions.append(job.phase)
                        print("PHASE \(mode) \(job.phase)")
                    }
                    if stateTransitions.last != job.state {
                        stateTransitions.append(job.state)
                        print("STATE \(mode) \(job.state)")
                    }
                }
            }
            let observation = store.$livePreview.sink { data in
                guard let data else { return }
                if TinyPreviewFrame.matches(data, width: request.width, height: request.height) {
                    previews.append(store.livePreviewStep)
                    print("PREVIEW \(mode) \(store.livePreviewStep)")
                } else {
                    invalidPreviewFrame = true
                }
                if mode == "cancel", !cancelled {
                    cancelled = true
                    store.cancel()
                }
            }
            if streamedConvrot {
                try JSONEncoder().encode(request).write(to: output.appendingPathComponent("\(mode)-request.json"),
                    options: .withoutOverwriting)
                print("RUNTIME \(NativeEngine.runtimeBuildIdentity())")
                print("BUNDLE_EXECUTABLE \(Bundle.main.executableURL?.path ?? "missing")")
            }
            print("START \(modelID) \(mode)")
            do {
                let job = try await store.generate(modelURL: model, request: request)
                try check(mode != "cancel", "cancelled job succeeded")
                try check(job.state == "succeeded", "job did not succeed")
                let data = try Data(contentsOf: URL(fileURLWithPath: request.output))
                guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                    let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
                else { throw NativeFailure(message: "unreadable result") }
                try check(image.width == 512 && image.height == 512, "output geometry mismatch")
                if mode == "baseline" { baseline = data }
                if mode == "preview" { try check(data == baseline, "preview changed final image") }
                if mode != "baseline" {
                    try check(!previews.isEmpty && !invalidPreviewFrame, "App did not display valid full-size live previews")
                }
                if mode == "tiny-edit" {
                    try check(
                        phases.contains("tiny_vae_encode"), "tiny reference encoder not invoked")
                }
                if streamedConvrot {
                    try check(!phases.contains("memory_wait") && !phases.contains("memory_ready"),
                        "ConvRot GPU generation must start without heuristic memory waiting")
                    try check(job.routeSummary?.contains("ConvRot") == true &&
                        job.routeSummary?.contains("流式加载") == true &&
                        phases.contains("denoise") && (mode != "tiny-final" || phases.contains("tiny_vae_decode")),
                        "App progress or history lost the measured ConvRot route")
                    if let receipt = job.resultJSON {
                        try Data(receipt.utf8).write(to: output.appendingPathComponent("\(mode)-native.json"),
                            options: .withoutOverwriting)
                        let policyResult = try JSONSerialization.jsonObject(with: Data(receipt.utf8)) as? [String: Any]
                        let policyStreaming = policyResult?["qwen_convrot_streaming"] as? [String: Any]
                        try check(["encoder", "denoiser"].allSatisfy {
                            (policyStreaming?[$0] as? [String: Any])?["allocation_policy"] as? String == "system_managed"
                        }, "ConvRot mandatory weights did not use the system allocator policy")
                        if cachedConvrot {
                            let raw = try JSONSerialization.jsonObject(with: Data(receipt.utf8)) as? [String: Any]
                            let streaming = raw?["qwen_convrot_streaming"] as? [String: Any]
                            let denoiser = streaming?["denoiser"] as? [String: Any]
                            try check((streaming?["requested_weight_budget_bytes"] as? NSNumber)?.uint64Value == UInt64(2) << 30 &&
                                (((denoiser?["prefix"] as? NSNumber)?.intValue ?? 0) > 0 ||
                                 (denoiser?["cache_limited_by_memory"] as? Bool) == true),
                                "Packaged acceptance lost the budget or failed to explain its physical-headroom fallback")
                        } else {
                            let raw = try JSONSerialization.jsonObject(with: Data(receipt.utf8)) as? [String: Any]
                            let streaming = raw?["qwen_convrot_streaming"] as? [String: Any]
                            try check((streaming?["requested_weight_budget_bytes"] as? NSNumber)?.uint64Value == 0,
                                "Packaged acceptance unexpectedly retained optional weight cache")
                        }
                    } else {
                        throw NativeFailure(message: "ConvRot acceptance is missing its actual native receipt")
                    }
                    let intermediate = previews.filter { $0 < request.steps }
                    let expected = mode == "baseline" ? [] : mode == "tiny-final" ? [3, 6] : [2, 4, 6]
                    try check(intermediate == expected, "App missed or reordered live previews: \(previews)")
                }
                reports.append([
                    "mode": mode, "state": job.state, "preview_steps": previews,
                    "phase_count": phases.count, "phases": phases.sorted(), "phase_transitions": phaseTransitions,
                    "state_transitions": stateTransitions, "route_summary": job.routeSummary ?? "",
                    "runtime_build_id": NativeEngine.runtimeBuildIdentity(),
                    "bundle_executable": Bundle.main.executableURL?.path ?? "",
                    "baseline_png": acceptanceBaseline ?? "generated-baseline.png",
                    "preview_unchanged": mode == "preview" ? data == baseline : true,
                ])
            } catch {
                guard mode == "cancel", cancelled else { throw error }
                try check(
                    !FileManager.default.fileExists(atPath: request.output),
                    "cancelled output was published")
                if streamedConvrot {
                    try check(error is CancellationError && store.jobs.first?.state == "cancelled",
                        "Cancelled ConvRot generation did not finish as a cancelled job")
                }
                reports.append(["mode": mode, "state": "cancelled", "preview_steps": previews,
                    "phases": phases.sorted(), "phase_transitions": phaseTransitions,
                    "state_transitions": stateTransitions])
            }
            observation.cancel()
            progress.cancel()
            try check(
                store.livePreview == nil && !store.busy,
                "preview or busy state leaked after terminal result")
        }
        if streamedConvrot {
            let restored = NativeJobStore(directory: output)
            let expectedSuccesses = modes.filter { $0 != "cancel" }.count
            let expectedCancellations = modes.filter { $0 == "cancel" }.count
            try check(restored.jobs.filter { $0.state == "succeeded" && $0.hasOutput }.count == expectedSuccesses &&
                restored.jobs.filter { $0.state == "cancelled" }.count == expectedCancellations,
                "Fresh App profile lost ConvRot success or cancellation history")
            try await store.unload()
            try check(store.loadedPath == nil, "ConvRot acceptance did not release the App session")
        }
        try JSONSerialization.data(withJSONObject: reports, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("acceptance.json"))
        print("Tiny VAE App GPU acceptance passed: \(reports)")
    }
}
