import SwiftUI

struct UpscaleWorkspaceHeader: View {
    @Binding var kind: String
    var locked: Bool
    var body: some View {
        HStack(spacing: 16) {
            Text("超分").font(.title2.weight(.semibold))
            Picker("媒体", selection: $kind) {
                Text("图片").tag("image")
                Text("视频").tag("video")
            }.pickerStyle(.segmented).frame(width: 120).labelsHidden()
                .disabled(locked).accessibilityIdentifier("upscaleMediaKind")
            Spacer(minLength: 0)
        }.padding(.bottom, 12).accessibilityIdentifier("upscaleWorkspace")
    }
}

/// Both executors share the same inspector picker, so changing models keeps
/// the source and workspace in place instead of navigating to the model hub.
struct UpscaleModelPicker: View {
    @ObservedObject var studio: StudioState
    @Binding var modelID: String
    var kind: String
    var locked: Bool

    var body: some View {
        Picker("模型", selection: $modelID) {
            ForEach(UpscaleModelChoice.available(for: kind, models: studio.models)) { choice in
                let configured = choice.isConfigured(in: studio.draft)
                let name = choice.pickerTitle
                Label("\(name) · \(configured ? "已配置" : "未配置")",
                      systemImage: configured ? "checkmark.circle.fill" : "arrow.down.circle")
                    .tag(choice.rawValue)
            }
        }.disabled(locked).accessibilityIdentifier("upscaleModelChoice")
    }
}

struct SeedVR2UpscaleControls: View {
    @ObservedObject var studio: StudioState
    var locked: Bool
    private var source: StudioAsset? { studio.draft.activeAssets.first }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 8) {
                Text("输出尺寸").font(.subheadline.weight(.medium))
                if let source {
                    Text("\(source.width) × \(source.height) → \(studio.draft.width) × \(studio.draft.height)")
                        .font(.caption.monospacedDigit())
                }
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 6) {
                    ForEach(SeedVR2OutputPreset.allCases) { preset in
                        Button { studio.applySeedVR2OutputPreset(preset) } label: {
                            Text(preset.title).frame(maxWidth: .infinity)
                        }
                        .disabled(source == nil)
                        .accessibilityIdentifier("seedvr2Output-\(preset.rawValue)")
                    }
                }
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("运行偏好").font(.subheadline.weight(.medium))
                HStack {
                    Button("速度优先") { studio.applySeedVR2SpeedSettings() }
                        .accessibilityIdentifier("seedvr2SpeedSettings")
                    Button("省内存") { studio.applySeedVR2MemoryAdvice() }
                        .accessibilityIdentifier("seedvr2QuickMemoryAdvice")
                }
                let options = studio.draft.seedvr2VAE ?? NativeSeedVR2VAEConfig()
                let residency = studio.draft.residency == "resident" ? "常驻权重" : "分阶段驻留"
                Text("\(residency) · 缓存 \(SeedVR2ResourceAdvice.gib(options.allocator_cache_bytes))")
                    .font(.caption).foregroundStyle(.secondary)
                    .accessibilityIdentifier("seedvr2CurrentConfiguration")
                if studio.creationKind == "video" {
                    Text("\(options.encoder == "streaming" ? "逐帧导入" : "缓冲导入") · \(options.decoder == "streaming" ? "逐帧导出" : "缓冲导出")")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("视频需 2–121 帧和整数 FPS，输出为无音轨 MP4。").font(.caption).foregroundStyle(.secondary)
                }
            }
            DisclosureGroup("高级设置") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        TextField("宽度", value: $studio.draft.width, format: .number)
                            .accessibilityIdentifier("width")
                        Text("×")
                        TextField("高度", value: $studio.draft.height, format: .number)
                            .accessibilityIdentifier("height")
                    }.textFieldStyle(.roundedBorder)
                    Text("宽高为 64–2048 内的 16 倍数。").font(.caption).foregroundStyle(.secondary)
                    Picker("权重驻留", selection: $studio.draft.residency) {
                        Text("常驻").tag("resident")
                        Text("分阶段").tag("component_staged")
                    }.accessibilityIdentifier("seedvr2Residency")
                    Toggle("随机种子", isOn: $studio.draft.randomSeed)
                    if !studio.draft.randomSeed {
                        TextField("种子", text: $studio.draft.seedText).textFieldStyle(.roundedBorder)
                    }
                }.padding(.top, 6)
            }
        }.disabled(locked).accessibilityIdentifier("seedvr2Workspace")
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
