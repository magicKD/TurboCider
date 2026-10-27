import Foundation

/// Model capabilities are explicit: the existing Real-ESRGAN executor is image-only.
enum UpscaleModelChoice: String, CaseIterable, Identifiable {
    case seedvr2 = "seedvr2-3b", realesrgan2 = "real-esrgan-x2plus", realesrgan4 = "real-esrgan-x4plus"
    var id: String { rawValue }
    var title: String {
        switch self {
        case .seedvr2: return "SeedVR2 3B · 图片 / 视频"
        case .realesrgan2: return "Real-ESRGAN x2plus · 仅图片"
        case .realesrgan4: return "Real-ESRGAN x4plus · 仅图片"
        }
    }
    var supportsVideo: Bool { self == .seedvr2 }
    func isConfigured(in draft: StudioDraft) -> Bool {
        if self == .seedvr2 { return !(draft.modelPaths[rawValue] ?? "").isEmpty }
        guard let variant = UpscaleVariant.from(modelID: rawValue) else { return false }
        let path = draft.upscaleVariant == variant ? draft.upscaleModelPath
            : draft.upscaleModelPaths[variant.rawValue] ?? ""
        return !path.isEmpty
    }
}

struct SeedVR2ProgressPresentation {
    let phase: String
    let completed: Int
    let total: Int
    var stage: Int {
        switch phase {
        case "seedvr2_encode_progress", "seedvr2_vae_encode": return 1
        case "seedvr2_dit", "seedvr2_dit_stage_load": return 2
        case "seedvr2_decode_progress", "seedvr2_vae_decode", "seedvr2_vae_export_frame": return 3
        case "export", "complete": return 4
        default: return 0
        }
    }
    static let stages = ["准备 / 导入", "VAE 编码", "模型恢复", "VAE 解码", "保存"]
    var title: String { Self.stages[stage] }
    var detail: String {
        let count = "\(max(0,min(completed,total))) / \(max(0,total))"
        switch phase {
        case "seedvr2_encode_progress": return "已编码 \(count) 帧"
        case "seedvr2_decode_progress": return "已解码 \(count) 帧"
        case "seedvr2_dit": return "已完成 \(count) 个 Transformer 块（单步恢复不等于瞬时完成）"
        case "seedvr2_dit_stage_load": return "分阶段驻留：正在重新加载推理权重"
        case "seedvr2_video_import": return "已导入 \(count) 帧"
        case "export": return "正在校验并保存完整媒体"
        default: return "正在准备模型和源媒体"
        }
    }
    var fraction: Double? {
        guard total > 0, ["seedvr2_encode_progress", "seedvr2_decode_progress", "seedvr2_dit", "seedvr2_video_import"].contains(phase) else { return nil }
        return min(1,max(0,Double(completed)/Double(total)))
    }
    func remainingStageSeconds(elapsed: Double) -> Double? {
        guard ["seedvr2_encode_progress", "seedvr2_decode_progress"].contains(phase),
              completed >= 5, completed < total, elapsed.isFinite, elapsed > 0 else { return nil }
        return elapsed / Double(completed) * Double(total-completed)
    }
}

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
