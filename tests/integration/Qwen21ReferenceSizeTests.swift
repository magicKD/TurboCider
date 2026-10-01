import AppKit
import Foundation

// CPU-only SDK/draft/state contracts. Catalog metadata is queried by request(),
// but no NativeEngine instance, native plan, weights or inference are used.
@main struct Qwen21ReferenceSizeTests {
    static func check(_ value: @autoclosure () throws -> Bool, _ reason: String) throws {
        guard try value() else { throw NativeFailure(message: reason) }
    }
    static func bytes<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }
    static func object<T: Encodable>(_ value: T) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: bytes(value)) as! [String: Any]
    }

    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-reference-encoding-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = try JSONDecoder().decode(StudioModel.self, from: Data(#"{"id":"qwen-image-2.1","name":"CPU fixture","executor":true,"output":"image","operations":["image.generate","image.edit"],"default_steps":40,"default_frames":1,"default_width":512,"default_height":512,"inputs":["image","text"],"max_images":10,"supports_lora":true,"lora_strategies":["inference_time"]}"#.utf8))
        let image = root.appendingPathComponent("reference.png")
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 64, pixelsHigh: 96,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        memset(bitmap.bitmapData!, 255, bitmap.bytesPerRow * bitmap.pixelsHigh)
        try bitmap.representation(using: .png, properties: [:])!.write(to: image)
        var draft = StudioDraft()
        draft.modelID = model.id; draft.modelPaths[model.id] = root.path
        draft.operation = "image.edit"; draft.prompt = "Preserve the referenced person."
        draft.steps = 25; draft.acceleration = StudioAcceleration(policy: "gpu")
        draft.assets = (0..<3).map { _ in StudioAsset(path: image.path, name: image.lastPathComponent, width: 64, height: 96) }
        draft.initImageID = draft.assets.first!.id
        let output = root.appendingPathComponent("never-generated.png")
        try compatibilityAndWire(draft, output: output)
        try eligibility(draft, model: model, root: root, output: output)
        try stateRecovery(draft, model: model, root: root, output: output)
        try await playground(draft, model: model, image: image, root: root, output: output)
        print("PASS Qwen reference encoding: SDK v1/v2, old draft/history, explicit 1024/512, supported Base/r128 gates, parameter recovery/persistence/import/reuse/model reset, canvas lock and independent Playground role/submission contracts (CPU only)")
    }

    static func compatibilityAndWire(_ draft: StudioDraft, output: URL) throws {
        let empty = try JSONDecoder().decode(StudioDraft.self, from: Data("{}".utf8))
        try check(empty.qwen21ReferenceSize == 1024, "Old drafts lost standard 1024 default")
        var fields = try object(draft); fields.removeValue(forKey: "qwen21ReferenceSize")
        let oldDraft = try JSONDecoder().decode(StudioDraft.self, from: JSONSerialization.data(withJSONObject: fields))
        try check(oldDraft.qwen21ReferenceSize == 1024 && oldDraft.assets == draft.assets, "Old draft migration changed reference content")
        let standard = try draft.request(output: output)
        try check(standard.qwen21_reference_size == 1024, "Qwen standard omitted its explicit encoding size")
        var oldWire = try object(standard); oldWire.removeValue(forKey: "qwen21_reference_size")
        let oldRequest = try JSONDecoder().decode(NativeRequest.self, from: JSONSerialization.data(withJSONObject: oldWire))
        try check(oldRequest.qwen21_reference_size == nil, "Old SDK request unexpectedly enabled a resize")
        var fast = standard; fast.qwen21_reference_size = 512; fast.allow_approximation = true
        let v1 = try object(fast)
        try check(v1["qwen21_reference_size"] as? Int == 512, "v1 did not encode top-level reference size")
        let decoded = try JSONDecoder().decode(NativeRequest.self, from: bytes(fast))
        try check(decoded.qwen21_reference_size == 512, "v1 reference size did not roundtrip")
        let v2 = NativeRequestV2(legacy: fast)
        let v2JSON = try object(v2)
        try check((v2JSON["parameters"] as? [String: Any])?["qwen21_reference_size"] as? Int == 512 &&
                  (v2JSON["execution"] as? [String: Any])?["qwen21_reference_size"] == nil &&
                  v2JSON["qwen21_reference_size"] == nil, "v2 reference size is outside parameters")
        try check(try JSONDecoder().decode(NativeRequestV2.self, from: bytes(v2)).parameters.qwen21_reference_size == 512,
                  "v2 reference size did not roundtrip")
        let oldV2 = NativeRequestV2(legacy: oldRequest)
        try check(try JSONDecoder().decode(NativeRequestV2.self, from: bytes(oldV2)).parameters.qwen21_reference_size == nil,
                  "Old v2 optional reference field did not decode")
        var other = StudioDraft()
        other.modelPaths[other.modelID] = draft.modelPath
        let otherRequest = try other.request(output: output)
        try check(otherRequest.qwen21_reference_size == nil && (try object(otherRequest))["qwen21_reference_size"] == nil,
                  "Non-Qwen request leaked a reference encoding setting")
    }

    static func eligibility(_ original: StudioDraft, model: StudioModel, root: URL, output: URL) throws {
        var base = original; base.qwen21ReferenceSize = 512
        for count in 1...3 {
            for steps in [20, 25, 40] {
                var candidate = base; candidate.assets = Array(original.assets.prefix(count)); candidate.steps = steps
                try candidate.validate(model: model)
                let request = try candidate.request(output: output)
                try check(request.qwen21_reference_size == 512 && request.allow_approximation == true &&
                          request.steps == steps && request.inputs?.map(\.path) == candidate.assets.map(\.path) &&
                          request.width == 512 && request.height == 512 && request.execution == "gpu" &&
                          request.qwen21_dit_cache == "off", "Eligible Base request lost its explicit domain or input order")
            }
        }
        let ordinary = root.appendingPathComponent("ordinary.safetensors")
        let r128 = root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
        let r256 = root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors")
        for url in [ordinary, r128, r256] { try Data("CPU presence fixture, not weights".utf8).write(to: url) }
        let invalid: [(String, (inout StudioDraft) -> Void)] = [
            ("generate", { $0.operation = "image.generate" }),
            ("transform", { $0.operation = "image.transform" }),
            ("zero refs", { $0.assets = [] }),
            ("four refs", { $0.assets.append($0.assets[0]) }),
            ("canvas", { $0.width = 768 }),
            ("PE", { $0.promptEnhance = true }),
            ("DiT", { $0.qwen21DiTCache = "conservative" }),
            ("ANE", { $0.acceleration?.policy = "gpu_ane" }),
            ("auto", { $0.acceleration?.policy = "auto" }),
            ("profile", { $0.profilePath = "profile.json" }),
            ("streaming", { $0.streaming.selection = .tier8 }),
            ("nineteen steps", { $0.steps = 19 }),
            ("forty one steps", { $0.steps = 41 }),
            ("ordinary LoRA", { $0.loras = [StudioLoRA(path: ordinary.path)] }),
            ("r256", { $0.loras = [StudioLoRA(path: r256.path)]; $0.steps = 6 }),
            ("invalid size", { $0.qwen21ReferenceSize = 256 })
        ]
        for (label, mutate) in invalid {
            var candidate = base; mutate(&candidate)
            let before = try bytes(candidate)
            try check(candidate.qwen21ReferenceSizeIssue != nil, "\(label) has no visible reference-size reason")
            var rejected = false
            do { try candidate.validate(model: model) } catch { rejected = true }
            try check(rejected && (try bytes(candidate)) == before, "\(label) bypassed validation or silently normalized parameters")
        }
        var turbo = base; turbo.loras = [StudioLoRA(path: r128.path)]; turbo.steps = 6
        turbo.loraStrategy = "inference_time"
        let request = try turbo.request(output: output)
        try check(request.qwen21_reference_size == 512 && request.steps == 6 && request.allow_approximation == true &&
                  request.lora_strategy == "inference_time" && request.loras?.first?.strength == 1,
                  "Qualified r128 request lost six-step approximate inference-time contract")
        for mutate: (inout StudioDraft) -> Void in [
            { $0.loras[0].strength = 0.9 }, { $0.loras[0].role = "text_encoder" },
            { $0.steps = 25 }, { $0.loraStrategy = "disk_premerge" },
            { $0.loras.append(StudioLoRA(path: ordinary.path)) }
        ] {
            var candidate = turbo; mutate(&candidate)
            try check(candidate.qwen21ReferenceSizeIssue != nil, "Invalid r128 setting had no recovery reason")
        }
        var portrait = base; portrait.assets[0].original = StudioAssetOriginal(path: original.assets[0].path,
            name: "original.png", width: 512, height: 768)
        let suggested = EditingCanvasSizing.suggestion(for: portrait, preset: .original)
        try check(!suggested.canApply && suggested.blockedReason?.contains("快速 512") == true,
                  "Output canvas helper bypassed the fast reference encoding size lock")
    }

    @MainActor static func stateRecovery(_ initial: StudioDraft, model: StudioModel, root: URL, output: URL) throws {
        let directory = root.appendingPathComponent("studio")
        let studio = StudioState(directory: directory, models: [model])
        studio.draft = initial
        try check(studio.setQwen21ReferenceSize(512), "Studio rejected eligible fast encoding")
        let expected = studio.draft
        let restored = StudioState(directory: directory, models: [model])
        try check(restored.draft.qwen21ReferenceSize == 512, "Fast encoding was not persisted")
        studio.draft.qwen21DiTCache = "balanced"
        var recovered = studio.draft; recovered.qwen21ReferenceSize = 1024
        try check(studio.setQwen21ReferenceSize(1024) && (try bytes(studio.draft)) == (try bytes(recovered)),
                  "Restore standard changed prompt/assets/steps/cache/LoRA instead of only encoding size")
        let before = try bytes(studio.draft)
        try check(!studio.setQwen21ReferenceSize(512) && (try bytes(studio.draft)) == before,
                  "Invalid selector choice silently changed the draft")
        let config = root.appendingPathComponent("configuration.json")
        try bytes(expected).write(to: config)
        try studio.importConfiguration(from: config)
        try check(studio.draft.qwen21ReferenceSize == 512 && studio.draft.assets == expected.assets,
                  "Exported App configuration lost reference encoding or ordered inputs")
        let request = try expected.request(output: output)
        var job = NativeJob(id: UUID(), createdAt: Date(), request: request, state: "succeeded", phase: "complete",
                            completed: 25, total: 25, elapsed: 0)
        job.modelPath = root.path
        let historical = try JSONDecoder().decode(NativeJob.self, from: bytes(job))
        studio.reuse(historical)
        try check(studio.draft.qwen21ReferenceSize == 512 && studio.draft.assets.map(\.path) == expected.assets.map(\.path),
                  "History reuse lost encoding size or input order")
        var legacyRequest = request; legacyRequest.qwen21_reference_size = nil
        let legacy = NativeJob(id: UUID(), createdAt: Date(), request: legacyRequest, state: "succeeded", phase: "complete",
                              completed: 25, total: 25, elapsed: 0)
        studio.reuse(legacy)
        try check(studio.draft.qwen21ReferenceSize == 1024, "Old history reused a stale fast encoding choice")
        studio.draft.qwen21ReferenceSize = 512
        studio.selectModel(model.id)
        try check(studio.draft.qwen21ReferenceSize == 1024, "Model reset retained a stale fast encoding choice")
    }

    @MainActor static func playground(_ initial: StudioDraft, model: StudioModel, image: URL, root: URL, output: URL) async throws {
        var copied = initial; copied.qwen21ReferenceSize = 512
        let beforeCreation = try bytes(copied)
        let directory = root.appendingPathComponent("playground")
        let state = PlaygroundState(directory: directory, initialSettings: copied, models: [model])
        try check(!state.setQwen21ReferenceSize(512), "Empty Playground roles allowed fast encoding")
        let imported = await state.importFiles([image, image])
        try check(imported, "Cannot import isolated outfit roles")
        let roles = state.orderedAssets
        let draft = try state.generationDraft()
        let request = try draft.request(output: output)
        try check(request.qwen21_reference_size == 512 && request.inputs?.map(\.path) == roles.map(\.path) &&
                  request.prompt.contains("<image1> as the person reference") && request.prompt.contains("<image2> as the clothing reference"),
                  "Playground submission lost encoding size or actual template role/prompt pipeline")
        state.recordSubmission(draft, seed: request.seed, template: .outfit)
        let reopened = PlaygroundState(directory: directory, initialSettings: StudioDraft(), models: [model])
        try check(reopened.settings.qwen21ReferenceSize == 512 && reopened.orderedAssets == roles &&
                  reopened.current.lastSeed == request.seed, "Playground submission settings did not persist")
        try check(state.setQwen21ReferenceSize(1024) && state.orderedAssets == roles,
                  "Playground standard recovery changed role references")
        copied.qwen21DiTCache = "balanced"
        try check(state.syncSettings(from: copied) && state.referenceEncodingDraft.qwen21ReferenceSizeIssue != nil &&
                  state.generationBlocker != nil, "Incompatible synchronized fast settings had no blocker")
        try check(state.setQwen21ReferenceSize(1024) && state.settings.qwen21DiTCache == "balanced" && state.generationBlocker == nil,
                  "Playground recovery disabled cache or failed to recover generation")
        state.selectTemplate(.identity)
        try check(state.settings.qwen21ReferenceSize == 512 && state.orderedAssets.isEmpty,
                  "Encoding recovery leaked to the other independent template")
        copied.qwen21DiTCache = "off"
        try check(try bytes(copied) == beforeCreation, "Playground changed Creation settings")
    }
}
