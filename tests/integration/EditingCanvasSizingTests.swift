import Foundation

// Metadata-only canvas contracts. No image reads, native planning or inference.
// Added during the test freeze; this source has not been executed.
@main struct EditingCanvasSizingTests {
    static func check(_ value: @autoclosure () -> Bool, _ reason: String) throws {
        guard value() else { throw NativeFailure(message: reason) }
    }

    static func main() throws {
        var draft = StudioDraft()
        draft.modelID = "qwen-image-2.1"; draft.operation = "image.edit"
        let reference = StudioAsset(path: "/derivative.png", name: "Reference", width: 512, height: 352,
            original: StudioAssetOriginal(path: "/original.png", name: "Original", width: 1000, height: 666),
            preparation: .fit512)
        draft.assets = [reference]
        let original = EditingCanvasSizing.suggestion(for: draft, preset: .original)
        try check(original.canApply && original.originalDimensions == EditingCanvasDimensions(width: 1000, height: 666) &&
                  original.dimensions == EditingCanvasDimensions(width: 992, height: 672),
                  "Matching used the derivative or failed Qwen 32-pixel alignment")
        let automatic = EditingCanvasSizing.suggestion(for: draft, preset: .automatic768)
        try check(automatic.canApply && automatic.dimensions == EditingCanvasDimensions(width: 768, height: 512),
                  "Automatic fit cropped/squared the original or used prepared dimensions")
        let before = try JSONEncoder().encode(draft)
        for preset in EditingCanvasPreset.allCases { _ = EditingCanvasSizing.suggestion(for: draft, preset: preset) }
        let after = try JSONEncoder().encode(draft)
        // Stable comparison tolerates JSON dictionary key ordering.
        let decoder = JSONDecoder()
        let unchanged = try decoder.decode(StudioDraft.self, from: after)
        let old = try decoder.decode(StudioDraft.self, from: before)
        try check(unchanged.assets == old.assets && unchanged.width == old.width && unchanged.height == old.height &&
                  unchanged.loras == old.loras && unchanged.qwen21DiTCache == old.qwen21DiTCache,
                  "A pure canvas suggestion changed draft settings")

        draft.modelID = "flux2-klein-4b"; draft.operation = "image.transform"
        let selected = StudioAsset(path: "/selected.png", name: "Selected", width: 1050, height: 706)
        draft.assets.append(selected); draft.initImageID = selected.id
        let transform = EditingCanvasSizing.suggestion(for: draft, preset: .original)
        try check(transform.canApply && transform.assetID == selected.id &&
                  transform.dimensions == EditingCanvasDimensions(width: 1056, height: 704),
                  "FLUX transform did not use selected init-image or 16-pixel alignment")
        let inactive = EditingCanvasSizing.suggestion(for: draft, preset: .original, assetID: reference.id)
        try check(!inactive.canApply && inactive.blockedReason != nil, "Inactive transform input was offered without an explanation")
        draft.modelID = "flux2-klein-9b"; draft.operation = "image.edit"
        try check(EditingCanvasSizing.suggestion(for: draft, preset: .automatic1024).canApply,
                  "Native FLUX 9B editing was excluded")

        draft.modelID = "qwen-image-2.1"
        draft.assets = [StudioAsset(path: "/square.png", name: "Square", width: 1024, height: 1024)]
        for path in ["/ordinary.safetensors", "/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"] {
            draft.loras = [StudioLoRA(path: path)]
            let blocked = EditingCanvasSizing.suggestion(for: draft, preset: .original)
            let allowed = EditingCanvasSizing.suggestion(for: draft, preset: .automatic512)
            try check(!blocked.canApply && blocked.blockedReason?.contains("512×512") == true &&
                      blocked.dimensions == EditingCanvasDimensions(width: 1024, height: 1024),
                      "Qwen LoRA lock did not preserve/explain the rejected candidate")
            try check(allowed.canApply && allowed.dimensions == EditingCanvasDimensions(width: 512, height: 512) &&
                      draft.loras[0].enabled, "A compatible canvas disabled LoRA or was rejected")
        }
        draft.loras = []
        for mode in ["conservative", "balanced", "fast"] {
            draft.qwen21DiTCache = mode
            try check(!EditingCanvasSizing.suggestion(for: draft, preset: .automatic768).canApply &&
                      EditingCanvasSizing.suggestion(for: draft, preset: .automatic512).canApply &&
                      draft.qwen21DiTCache == mode, "Canvas fitting silently disabled or bypassed a DiT cache lock")
        }
        draft.qwen21DiTCache = "off"
        draft.assets = [StudioAsset(path: "/near-min.png", name: "Near minimum", width: 63, height: 512)]
        try check(EditingCanvasSizing.suggestion(for: draft, preset: .original).dimensions ==
                  EditingCanvasDimensions(width: 64, height: 512), "Original matching rejected a valid nearest-aligned dimension")
        try check(!EditingCanvasSizing.suggestion(for: draft, preset: .automatic512).canApply,
                  "Automatic fit enlarged a short side below the minimum")
        draft.assets = [StudioAsset(path: "/small.png", name: "Small", width: 500, height: 500)]
        try check(EditingCanvasSizing.suggestion(for: draft, preset: .automatic512).dimensions ==
                  EditingCanvasDimensions(width: 480, height: 480), "Automatic alignment enlarged a small reference")
        draft.assets = [StudioAsset(path: "/square.png", name: "Square", width: 1024, height: 1024)]
        draft.loras = [StudioLoRA(path: "/disabled.safetensors", enabled: false)]
        try check(EditingCanvasSizing.suggestion(for: draft, preset: .automatic1024).canApply,
                  "Disabled LoRA incorrectly locked the canvas")
        draft.loras = []; draft.assets = [reference]
        draft.qwen21DiTCache = "fast"
        draft.loras = [StudioLoRA(path: "/ordinary.safetensors")]
        let combined = EditingCanvasSizing.suggestion(for: draft, preset: .automatic512)
        try check(combined.blockedReason?.contains("LoRA与DiT 缓存") == true,
                  "Combined restrictions only explained one setting")
        draft.loras = []
        let rectangle = EditingCanvasSizing.suggestion(for: draft, preset: .automatic512)
        try check(!rectangle.canApply && rectangle.blockedReason?.contains("512×512") == true,
                  "Cache lock silently squared a non-square reference")

        draft.qwen21DiTCache = "off"
        for (width, height, preset) in [(1, 4000, EditingCanvasPreset.automatic1024),
                                       (8192, 64, .automatic512), (0, 512, .original),
                                       (Int.max, Int.max, .original), (4096, 4096, .original)] {
            draft.assets = [StudioAsset(path: "/edge.png", name: "Edge", width: width, height: height)]
            let result = EditingCanvasSizing.suggestion(for: draft, preset: preset)
            try check(!result.canApply && result.blockedReason != nil, "Invalid/aspect/area bounds lacked an actionable blocked reason")
        }
        draft.assets = [StudioAsset(path: "/max.png", name: "Max", width: 2048, height: 4096)]
        try check(EditingCanvasSizing.suggestion(for: draft, preset: .original).canApply, "Qwen's 8MP boundary was excluded")
        draft.modelID = "flux2-klein-4b"
        try check(!EditingCanvasSizing.suggestion(for: draft, preset: .original).canApply &&
                  EditingCanvasSizing.suggestion(for: draft, preset: .automatic1024).canApply,
                  "FLUX max dimension did not offer a valid automatic alternative")
        for operation in ["video.image", "video.reference", "image.generate"] {
            draft.operation = operation
            let result = EditingCanvasSizing.suggestion(for: draft, preset: .automatic512)
            try check(!result.canApply && result.blockedReason != nil, "Non-edit mode was offered a reference-matching canvas")
        }
        draft.modelID = "qwen-image-2.1"; draft.operation = "image.transform"
        try check(!EditingCanvasSizing.suggestion(for: draft, preset: .automatic512).canApply,
                  "Sizing invented Qwen image.transform capability")
        print("PASS editing canvas metadata/original selection, alignment/bounds, active reference selection and LoRA/cache locks (no inference)")
    }
}
