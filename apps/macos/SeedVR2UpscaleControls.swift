import SwiftUI

struct UpscaleWorkspaceHeader: View {
    @Binding var kind: String
    @Binding var modelID: String
    var locked: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 16) {
                Text("超分").font(.title2.weight(.semibold))
                Picker("媒体", selection: $kind) {
                    Text("图片").tag("image"); Text("视频").tag("video")
                }.pickerStyle(.segmented).frame(width: 120).labelsHidden()
                Picker("超分模型", selection: $modelID) {
                    ForEach(UpscaleModelChoice.allCases.filter { kind != "video" || $0.supportsVideo }) {
                        Text($0.title).tag($0.rawValue)
                    }
                }.frame(maxWidth: 370).accessibilityIdentifier("upscaleModelChoice")
                Spacer(minLength: 0)
            }.disabled(locked)
            Text(kind == "video"
                 ? "当前可用视频超分模型：SeedVR2。保留完整帧数与源 FPS；暂不保留音轨。"
                 : "选择模型恢复细节并放大图片。SeedVR2 也支持视频；Real-ESRGAN 为图片专用。")
                .font(.caption).foregroundStyle(.secondary)
        }.padding(.bottom,8).accessibilityIdentifier("upscaleWorkspace")
    }
}

struct SeedVR2UpscaleControls: View {
    @ObservedObject var studio: StudioState
    var locked: Bool
    var manageModels: () -> Void
    private var source: StudioAsset? { studio.draft.activeAssets.first }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("SeedVR2 恢复设置", systemImage: "slider.horizontal.3")
                    .font(.headline)
                Spacer()
                Button("管理模型", action: manageModels)
                    .disabled(locked).accessibilityIdentifier("seedvr2ManageModel")
            }
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
            }
            HStack {
                Text("运行偏好").font(.caption)
                Spacer()
                Button("速度优先") { studio.applySeedVR2SpeedSettings() }
                    .disabled(locked).accessibilityIdentifier("seedvr2SpeedSettings")
                Button("省内存") { studio.applySeedVR2MemoryAdvice() }
                    .disabled(locked).accessibilityIdentifier("seedvr2QuickMemoryAdvice")
            }
            let options = studio.draft.seedvr2VAE ?? NativeSeedVR2VAEConfig()
            Text("当前：\(studio.draft.residency == "resident" ? "常驻权重" : "分阶段驻留") · \(options.encoder == "streaming" ? "逐帧导入" : "缓冲导入") / \(options.decoder == "streaming" ? "逐帧导出" : "缓冲导出") · 缓存 \(SeedVR2ResourceAdvice.gib(options.allocator_cache_bytes))")
                .font(.caption).foregroundStyle(.secondary)
                .help("速度优先使用完整空间、4 GiB 缓存和分帧历史；省内存采用分阶段驻留、1 GiB 缓存及视频流式读写，可能更慢。两者都不减帧、不降低恢复分辨率。")
                .accessibilityIdentifier("seedvr2CurrentConfiguration")
            Text(studio.draft.modelPath.isEmpty
                 ? "尚未配置模型：到模型中心导入已有 SeedVR2 DiT 和 VAE。不会自动下载模型。"
                 : "省内存可能更慢，不降低画质。视频需 2–121 帧、整数 FPS；原媒体保留，输出为无音轨 MP4。")
                .font(.caption).foregroundStyle(studio.draft.modelPath.isEmpty ? Color.orange : Color.secondary)
        }.padding(12)
            .background(Color.accentColor.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
            .accessibilityIdentifier("seedvr2Workspace")
    }
}

struct SeedVR2RunProgressView: View {
    let job: NativeJob
    var cancel: () -> Void
    var body: some View {
        let info = SeedVR2ProgressPresentation(phase: job.phase, completed: job.completed, total: job.total)
        TimelineView(.periodic(from: .now, by: 1)) { context in
            let elapsed = max(job.elapsed, context.date.timeIntervalSince(job.createdAt))
            let stageElapsed = max(0, elapsed - (job.phaseStartedElapsed ?? job.elapsed))
            VStack(alignment: .leading, spacing: 7) {
                HStack {
                    Text("\(info.title) · 阶段 \(info.stage+1) / 5").font(.headline)
                    Spacer()
                    Text(String(format: "已用时 %.0f 秒",elapsed)).monospacedDigit()
                    Button(job.state == "cancelling" ? "正在安全停止…" : "取消", action: cancel)
                        .disabled(job.state == "cancelling")
                }
                HStack(spacing: 10) {
                    ForEach(Array(SeedVR2ProgressPresentation.stages.enumerated()), id: \.offset) { index, title in
                        Label(title, systemImage: index < info.stage ? "checkmark.circle.fill" : index == info.stage ? "circle.inset.filled" : "circle")
                            .foregroundStyle(index == info.stage ? Color.accentColor : Color.secondary)
                    }
                }.font(.caption)
                if let fraction = info.fraction { ProgressView(value: fraction) }
                else { ProgressView().controlSize(.small) }
                HStack {
                    Text(info.detail)
                    Spacer()
                    if let seconds = info.remainingStageSeconds(elapsed: stageElapsed) {
                        Text("本阶段约剩 \(Int(ceil(seconds))) 秒（估算）").monospacedDigit()
                    }
                }.font(.caption)
                Text("\(job.request.width) × \(job.request.height) · \(job.request.frames) 帧 · \(job.request.fps ?? 24) FPS。编码、推理、解码分别计数，进度条为当前阶段。")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }.accessibilityIdentifier("seedvr2RunProgress")
    }
}
