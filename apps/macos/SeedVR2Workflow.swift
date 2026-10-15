import Foundation

/// App-only output shortcuts. The native request still owns exact dimensions,
/// source frame/FPS checks and memory admission; these never trim the input.
enum SeedVR2OutputPreset: String, CaseIterable, Identifiable {
    case scale2, scale4, long512, long1024
    var id: String { rawValue }
    var title: String {
        switch self {
        case .scale2: return "2×"
        case .scale4: return "4×"
        case .long512: return "长边 512"
        case .long1024: return "长边 1024"
        }
    }
    func dimensions(source: StudioAsset) throws -> (width: Int, height: Int) {
        guard source.width > 0, source.height > 0,
              Double(source.width) * Double(source.height) <= 80_000_000 else {
            throw NativeFailure(message: "源媒体尺寸无效，请重新导入图片或视频。")
        }
        let width: Int, height: Int
        switch self {
        case .scale2, .scale4:
            let scale = self == .scale2 ? 2 : 4
            // Check before multiplication, including malformed legacy metadata.
            guard source.width <= 2048 / scale, source.height <= 2048 / scale else {
                throw NativeFailure(message: "此倍率会超过当前 2048 像素单边上限；可先选择长边 512 或 1024 测试。")
            }
            width = source.width * scale / 16 * 16
            height = source.height * scale / 16 * 16
        case .long512, .long1024:
            let edge = self == .long512 ? 512 : 1024
            let ratio = Double(edge) / Double(max(source.width, source.height))
            width = Int(Double(source.width) * ratio) / 16 * 16
            height = Int(Double(source.height) * ratio) / 16 * 16
        }
        guard (64...2048).contains(width), (64...2048).contains(height) else {
            throw NativeFailure(message: "该比例的短边不足 64 像素，请在画幅设置中指定有效输出尺寸。")
        }
        return (width, height)
    }
}
