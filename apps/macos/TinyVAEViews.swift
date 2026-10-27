import SwiftUI
import AppKit

/// Shared controls for Creation and Playground; request state stays in StudioDraft.
struct TinyVAEControls: View {
    @Binding var draft: StudioDraft
    var locked = false
    var body: some View {
        if draft.supportsTinyVAE {
            DisclosureGroup("Tiny VAE 与实时预览") {
                VStack(alignment: .leading, spacing: 9) {
                    HStack {
                        Text(
                            draft.resolvedTinyVAEPath.map {
                                URL(fileURLWithPath: $0).lastPathComponent
                            } ?? "尚未安装 \(draft.tinyArchitecture)"
                        )
                        .font(.caption).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        Button("选择模型…") {
                            let panel = NSOpenPanel()
                            panel.canChooseDirectories = false
                            panel.allowsMultipleSelection = false
                            panel.message =
                                "选择 \(draft.tinyArchitecture).safetensors（原生权重，不需要 Python）"
                            if panel.runModal() == .OK, let url = panel.url {
                                draft.tinyVAEPaths[draft.tinyArchitecture] = url.path
                            }
                        }.accessibilityIdentifier("tinyVAEChooseModel")
                    }
                    if draft.modelID == "qwen-image-2.1" {
                        Picker("参考图编码", selection: $draft.tinyVAEEncoder) {
                            Text("原始 VAE（质量优先）").tag("original")
                            Text("Tiny（近似、速度优先）").tag("tiny")
                        }.accessibilityIdentifier("tinyVAEEncoder")
                    }
                    Picker("最终图解码", selection: $draft.tinyVAEDecoder) {
                        Text("原始 VAE（质量优先）").tag("original")
                        Text("Tiny（近似、速度优先）").tag("tiny")
                    }.accessibilityIdentifier("tinyVAEDecoder")
                    Toggle("实时采样预览", isOn: $draft.tinyPreview).accessibilityIdentifier(
                        "tinyPreviewEnabled")
                    if draft.tinyPreview {
                        Stepper(
                            "每 \(draft.tinyPreviewInterval) 步预览", value: $draft.tinyPreviewInterval,
                            in: 1...50
                        )
                        .accessibilityIdentifier("tinyPreviewInterval")
                    }
                    Text(
                        "预览独立于最终解码，默认每 2 步更新，并显示最后一步。预览不会改变采样结果；Tiny 编码／最终解码会损失细节。Z-Image 文生图不调用参考图编码器。"
                    )
                    .font(.caption2).foregroundStyle(.secondary)
                    if draft.streaming.selection != .off {
                        Text("使用 Tiny VAE 或预览前，请关闭未标定的流式内存档位。")
                            .font(.caption2).foregroundStyle(.orange)
                    }
                }.padding(.top, 8)
            }.disabled(locked)
        }
    }
}

struct TinyLivePreview: View {
    let data: Data
    let step: Int
    let total: Int
    var body: some View {
        VStack(spacing: 8) {
            if let image = NSImage(data: data) { Image(nsImage: image).resizable().scaledToFit() }
            Text("实时预览 · \(step)/\(total) 步 · 非最终结果").font(.caption).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, maxHeight: .infinity).accessibilityIdentifier(
            "tinyLivePreview")
    }
}
