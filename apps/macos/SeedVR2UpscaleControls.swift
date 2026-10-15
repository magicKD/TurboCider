import SwiftUI

struct SeedVR2UpscaleControls: View {
    @ObservedObject var studio: StudioState
    var locked: Bool
    var manageModels: () -> Void
    private var source: StudioAsset? { studio.draft.activeAssets.first }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("SeedVR2 图像 / 视频超分", systemImage: "arrow.up.left.and.arrow.down.right")
                    .font(.headline)
                Spacer()
                Button("管理 SeedVR2 模型", action: manageModels)
                    .disabled(locked).accessibilityIdentifier("seedvr2ManageModel")
            }
            Text("原始 FP16 模型 · 单步 GPU 恢复 · 无需提示词。原媒体保留，图片另存 PNG，视频另存无音轨 MP4。")
                .font(.caption).foregroundStyle(.secondary)
            if let source {
                HStack {
                    Text("源媒体 \(source.width) × \(source.height)")
                    if source.mediaKind == "video" {
                        Text("\(source.frames ?? studio.draft.frames) 帧 · \(source.fps ?? studio.draft.fps) FPS")
                    }
                    Text("→ \(studio.draft.width) × \(studio.draft.height)")
                }.font(.caption).monospacedDigit()
            }
            HStack(spacing: 8) {
                Text("输出尺寸").font(.caption)
                ForEach(SeedVR2OutputPreset.allCases) { preset in
                    Button(preset.title) { studio.applySeedVR2OutputPreset(preset) }
                        .disabled(locked || source == nil)
                        .accessibilityIdentifier("seedvr2Output-\(preset.rawValue)")
                }
                Spacer(minLength: 0)
                Button("省内存建议") { studio.applySeedVR2MemoryAdvice() }
                    .disabled(locked).accessibilityIdentifier("seedvr2QuickMemoryAdvice")
            }
            Text(studio.draft.modelPath.isEmpty
                 ? "尚未配置模型：到模型中心导入已有 SeedVR2 DiT 和 VAE。不会自动下载模型。"
                 : "首次建议用长边 512 / 1024 测试。视频需为 2–121 帧、整数源 FPS；不自动截帧或补帧。右侧可调驻留，详细 VAE 选项在模型中心的加速设置中。")
                .font(.caption).foregroundStyle(studio.draft.modelPath.isEmpty ? Color.orange : Color.secondary)
        }.padding(12)
            .background(Color.accentColor.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
            .accessibilityIdentifier("seedvr2Workspace")
    }
}
