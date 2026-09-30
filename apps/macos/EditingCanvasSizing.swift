import Foundation

/// Output-canvas choices, independent of input-file preparation and Qwen's
/// internal reference encoding size. Original means the retained pixel baseline.
enum EditingCanvasPreset: String, CaseIterable, Identifiable, Sendable {
    case original, automatic512, automatic768, automatic1024
    var id: String { rawValue }
    var title: String {
        switch self {
        case .original: return "匹配参考原图尺寸"
        case .automatic512: return "自动适合 512"
        case .automatic768: return "自动适合 768"
        case .automatic1024: return "自动适合 1024"
        }
    }
    var detail: String {
        switch self {
        case .original: return "使用保留的原始副本尺寸，按模型要求对齐输出宽高。"
        default: return "按参考原图比例缩小到最长边范围内，再按模型要求对齐；不修改参考图文件。"
        }
    }
    fileprivate var longestSide: Int? {
        switch self {
        case .original: return nil
        case .automatic512: return 512
        case .automatic768: return 768
        case .automatic1024: return 1024
        }
    }
}

struct EditingCanvasDimensions: Equatable, Sendable {
    let width: Int
    let height: Int
}

struct EditingCanvasSuggestion: Equatable, Sendable {
    let preset: EditingCanvasPreset
    let assetID: UUID?
    let originalDimensions: EditingCanvasDimensions?
    /// A candidate may be present even when the active LoRA/cache blocks it.
    /// UI must use canApply, not just the presence of dimensions, to enable Apply.
    let dimensions: EditingCanvasDimensions?
    let blockedReason: String?
    let detail: String
    var canApply: Bool { dimensions != nil && blockedReason == nil }

    fileprivate func blocking(_ reason: String) -> Self {
        Self(preset: preset, assetID: assetID, originalDimensions: originalDimensions,
             dimensions: dimensions, blockedReason: reason, detail: detail)
    }
}

/// Pure metadata arithmetic: no image decoding, model loading or native calls.
/// Bounds mirror native/runtime/plan.cpp and the FLUX/Qwen module validators.
enum EditingCanvasSizing {
    static func suggestion(for draft: StudioDraft, preset: EditingCanvasPreset,
                           assetID: UUID? = nil) -> EditingCanvasSuggestion {
        func blocked(_ reason: String) -> EditingCanvasSuggestion {
            EditingCanvasSuggestion(preset: preset, assetID: assetID, originalDimensions: nil,
                                    dimensions: nil, blockedReason: reason, detail: preset.detail)
        }
        guard ["image.edit", "image.transform"].contains(draft.operation) else {
            return blocked("画布匹配仅用于图片编辑，请先选择编辑方式并添加参考图。")
        }
        let multiple: Int, maximum: Int, maximumPixels: Int
        switch draft.modelID {
        case "qwen-image-2.1":
            guard draft.operation == "image.edit" else {
                return blocked("Qwen 2.1 使用图片编辑方式，不支持单图修改路径。")
            }
            multiple = 32; maximum = 4096; maximumPixels = 8_388_608
        case "flux2-klein-4b", "flux2-klein-9b":
            multiple = 16; maximum = 2048; maximumPixels = 2048 * 2048
        default:
            return blocked("当前模型未开放可匹配参考图的图片编辑画布。")
        }
        let active = draft.activeAssets
        let asset = assetID.flatMap { id in active.first { $0.id == id } } ?? (assetID == nil ? active.first : nil)
        guard let asset else {
            return blocked(assetID == nil ? "请先添加参考图；单图修改需要选择原图。" : "这张图片不在当前编辑输入中，请重新选择参考图。")
        }
        let source = asset.originalImage
        let original = EditingCanvasDimensions(width: source.width, height: source.height)
        func result(_ dimensions: EditingCanvasDimensions? = nil, reason: String? = nil,
                    detail: String = "") -> EditingCanvasSuggestion {
            EditingCanvasSuggestion(preset: preset, assetID: asset.id, originalDimensions: original,
                                    dimensions: dimensions, blockedReason: reason,
                                    detail: detail.isEmpty ? preset.detail : detail)
        }
        guard source.width > 0, source.height > 0,
              Double(source.width) * Double(source.height) <= 80_000_000 else {
            return result(reason: "参考原图尺寸无效或超过导入上限，请重新导入图片。")
        }
        let scale = preset.longestSide.map {
            min(1.0, Double($0) / Double(max(source.width, source.height)))
        } ?? 1.0
        let idealWidth = Double(source.width) * scale
        let idealHeight = Double(source.height) * scale
        // A very thin image cannot fit this bound with both sides >=64 while
        // retaining its proportions. Do not silently stretch or square it.
        guard preset == .original || (idealWidth >= 64 && idealHeight >= 64) else {
            return result(reason: "按原图比例计算后短边小于模型要求的 64 像素；请选更大的范围或手动设置画布。")
        }
        func aligned(_ value: Double, original: Int) -> Int {
            // Automatic fit never enlarges an already small reference. Exact
            // original matching uses the nearest supported model dimensions.
            let nearest = Int((value / Double(multiple)).rounded()) * multiple
            return preset == .original ? nearest : min(nearest, original / multiple * multiple)
        }
        let dimensions = EditingCanvasDimensions(width: aligned(idealWidth, original: source.width),
                                                 height: aligned(idealHeight, original: source.height))
        let description = "参考原图 \(source.width)×\(source.height) → 输出画布 \(dimensions.width)×\(dimensions.height)（\(multiple) 倍数对齐）。参考图文件与模型内部参考编码尺寸保持当前设置；对齐会轻微改变比例。"
        guard (64...maximum).contains(dimensions.width), (64...maximum).contains(dimensions.height) else {
            return result(dimensions, reason: "对齐后尺寸超出当前模型的 64–\(maximum) 范围；请选择自动适合或手动设置画布。", detail: description)
        }
        guard dimensions.width * dimensions.height <= maximumPixels else {
            return result(dimensions, reason: "对齐后画布超过 Qwen 2.1 的 8 百万像素上限；请选择自动适合。", detail: description)
        }
        if draft.modelID == "qwen-image-2.1", dimensions.width != 512 || dimensions.height != 512 {
            var restrictions: [String] = []
            if !draft.activeLoRAs.isEmpty { restrictions.append(draft.hasQwen21TurboAdapter ? "六步 Viggle LoRA" : "普通 Qwen LoRA") }
            if draft.qwen21DiTCache != "off" { restrictions.append("DiT 缓存") }
            if !restrictions.isEmpty {
                let names = restrictions.joined(separator: "与")
                return result(dimensions, reason: "\(names)当前限定 512×512，不能应用 \(dimensions.width)×\(dimensions.height)。请保留兼容画布，或自行关闭上述选项后再匹配。", detail: description)
            }
        }
        return result(dimensions, detail: description)
    }
}

@MainActor extension StudioState {
    func editingCanvasSuggestion(preset: EditingCanvasPreset, assetID: UUID? = nil) -> EditingCanvasSuggestion {
        let value = EditingCanvasSizing.suggestion(for: draft, preset: preset, assetID: assetID)
        if importing { return value.blocking("参考图正在处理，请等待完成后再调整输出画布。") }
        guard let model = models.first(where: { $0.id == draft.modelID }), model.supports(draft.operation), !model.isVideo else {
            return value.blocking("当前模型或操作尚未开放此图片编辑画布。")
        }
        return value
    }

    @discardableResult
    func applyEditingCanvas(preset: EditingCanvasPreset, assetID: UUID? = nil) -> Bool {
        let value = editingCanvasSuggestion(preset: preset, assetID: assetID)
        guard value.canApply, let dimensions = value.dimensions else {
            message = value.blockedReason ?? "无法建议画布尺寸，请检查参考图。"
            return false
        }
        // Apply to the latest draft, never to a cached suggestion from a sheet.
        // No LoRA/cache/step/prompt/asset mutation and no image-file rewrite.
        var updated = draft
        updated.width = dimensions.width; updated.height = dimensions.height
        draft = updated
        message = "已设置输出画布为 \(dimensions.width)×\(dimensions.height)。参考图文件、提示词、LoRA 与缓存设置未改变。"
        save()
        return true
    }
}
