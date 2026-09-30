import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// App-side input preparation. This changes an input file, never the output
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

/// ImageIO decoding and bitmap drawing only. Called from the importer actor,
/// with no model runtime, Metal context or main-actor image work.
enum ReferenceImagePreparationRenderer {
    static func render(source url: URL, preset: ReferenceImagePreparation) throws -> PreparedReferenceImage {
        try Task.checkCancellation()
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetCount(source) == 1,
              let info = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let rawWidth = info[kCGImagePropertyPixelWidth] as? Int,
              let rawHeight = info[kCGImagePropertyPixelHeight] as? Int else {
            throw ReferenceImagePreparationFailure(message: "无法读取原始图片，请重新导入单帧图片。")
        }
        let orientation = info[kCGImagePropertyOrientation] as? Int ?? 1
        let rotated = (5...8).contains(orientation)
        let width = rotated ? rawHeight : rawWidth, height = rotated ? rawWidth : rawHeight
        let target = try preset.dimensions(width: width, height: height)
        // A preset that already fits keeps original bytes (color space, bit
        // depth and EXIF included). Orientation is still reflected in metadata.
        let keepsOriginal = target.width == width && target.height == height
        let maximum = keepsOriginal ? min(128, max(width, height)) : max(target.width, target.height)
        guard let decoded = CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximum,
            kCGImageSourceShouldCacheImmediately: true
        ] as CFDictionary) else {
            throw ReferenceImagePreparationFailure(message: "原始图片无法解码，请重新导入。")
        }
        try Task.checkCancellation()
        if keepsOriginal {
            return PreparedReferenceImage(originalWidth: width, originalHeight: height,
                                          width: width, height: height, png: nil)
        }
        // Exact integer fit dimensions make UI predictions match the PNG.
        // Premultiplied RGBA preserves transparency through the resampling.
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: target.width, height: target.height,
                  bitsPerComponent: 8, bytesPerRow: target.width * 4, space: space,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else {
            throw ReferenceImagePreparationFailure(message: "无法准备图片处理缓冲区。")
        }
        context.interpolationQuality = .high
        context.draw(decoded, in: CGRect(x: 0, y: 0, width: CGFloat(target.width), height: CGFloat(target.height)))
        guard let image = context.makeImage() else {
            throw ReferenceImagePreparationFailure(message: "无法生成处理后的图片。")
        }
        try Task.checkCancellation()
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw ReferenceImagePreparationFailure(message: "无法创建 PNG 副本。")
        }
        // Pixels are already oriented. Do not carry the original EXIF transform.
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: 1] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw ReferenceImagePreparationFailure(message: "无法编码 PNG 副本。")
        }
        try Task.checkCancellation()
        return PreparedReferenceImage(originalWidth: width, originalHeight: height,
                                      width: target.width, height: target.height, png: data as Data)
    }
}
