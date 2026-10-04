import Foundation

/// UI choices for shared input preparation. This changes an input file, never the output
/// canvas or Qwen's independent internal reference_size setting.
enum ReferenceImagePreparation: String, Codable, CaseIterable, Identifiable, Sendable {
    case original, automatic, fit512, portrait512, landscape512
    var id: String { rawValue }
    var title: String {
        switch self {
        case .original: return "原图"
        case .automatic: return "自动缩小"
        case .fit512: return "适合 512×512"
        case .portrait512: return "适合 512×768"
        case .landscape512: return "适合 768×512"
        }
    }
    var detail: String {
        switch self {
        case .original: return "恢复保留的原始副本，不改变尺寸。"
        case .automatic: return "最长边不超过 1024，保持比例，不裁剪、不放大。"
        case .fit512: return "等比缩小到 512×512 范围内，不裁剪、不补边、不放大。"
        case .portrait512: return "等比缩小到 512×768 范围内，不裁剪、不补边、不放大。"
        case .landscape512: return "等比缩小到 768×512 范围内，不裁剪、不补边、不放大。"
        }
    }
    func dimensions(width: Int, height: Int) throws -> (width: Int, height: Int) {
        guard width > 0, height > 0, Double(width) * Double(height) <= 80_000_000 else {
            throw ReferenceImagePreparationFailure(message: "图片尺寸无效或超过 8000 万像素。")
        }
        let bounds: (Int, Int)
        switch self {
        case .original: return (width, height)
        case .automatic: bounds = (1024, 1024)
        case .fit512: bounds = (512, 512)
        case .portrait512: bounds = (512, 768)
        case .landscape512: bounds = (768, 512)
        }
        let scale = min(1.0, min(Double(bounds.0) / Double(width), Double(bounds.1) / Double(height)))
        return (max(1, min(width, bounds.0, Int((Double(width) * scale).rounded()))),
                max(1, min(height, bounds.1, Int((Double(height) * scale).rounded()))))
    }
}

private struct ReferenceImagePreparationFailure: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct PreparedReferenceImage: Sendable {
    let originalWidth: Int
    let originalHeight: Int
    let width: Int
    let height: Int
    /// Nil means restoration of the original bytes; no derivative is written.
    let png: Data?
}

/// The native CPU renderer also backs the CLI/RPC image_prepare operation.
/// Called from the importer actor; no model or main-actor image work.
enum ReferenceImagePreparationRenderer {
    static func render(source url: URL, preset: ReferenceImagePreparation) throws -> PreparedReferenceImage {
        try Task.checkCancellation()
        let result = try NativeEngine.prepareReferenceImage(sourcePath: url.path, preset: preset.rawValue)
        try Task.checkCancellation()
        return PreparedReferenceImage(originalWidth: result.metadata.original_width,
                                      originalHeight: result.metadata.original_height,
                                      width: result.metadata.width, height: result.metadata.height,
                                      png: result.png)
    }
}
