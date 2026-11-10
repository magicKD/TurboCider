import SwiftUI

struct TinyVAELibraryView: View {
    @ObservedObject var library: ModelLibraryController
    @ObservedObject var studio: StudioState
    let locked: Bool
    @State private var source = TinyVAESource.mirror
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("LTX 视频 VAE").font(.title2.weight(.semibold))
                Text("在创作页的高级参数中选择视频解码器。Conv VAE 随 LTX 基础模型安装；PrunaVAED 可单独下载。")
                    .font(.callout).foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 10) {
                    Label("Conv VAE · 默认", systemImage: "film").font(.headline)
                    Text("使用 LTX 原始视频解码器，随基础模型下载。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("使用 Conv VAE") {
                        studio.draft.ltxVideoVAE = "conv"; studio.save()
                    }.disabled(locked || studio.imageInputsBusy)
                        .accessibilityIdentifier("useLTXConvVAE")
                }.padding(18).frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Label("PrunaVAED · 更快解码", systemImage: "film").font(.headline)
                        Spacer()
                        Text(ByteCountFormatter.string(fromByteCount: PrunaVAEAsset.model.size, countStyle: .file))
                            .font(.caption).monospacedDigit()
                    }
                    Text("ModelScope · PrunaAI/PrunaVAED").font(.caption).foregroundStyle(.secondary)
                    Text("剪枝视频解码器，可缩短解码时间；细节与原始 VAE 有差异。图生视频的输入编码仍使用原始 VAE。")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        if library.prunaVAEPath != nil || studio.draft.resolvedPrunaVAEPath != nil {
                            Label("已可用", systemImage: "checkmark.circle").font(.caption).foregroundStyle(.green)
                            Button("用于 LTX 创作") { library.usePrunaVAE(studio: studio) }
                                .disabled(library.busy || locked || studio.imageInputsBusy)
                                .accessibilityIdentifier("usePrunaVAE")
                        }
                        Spacer()
                        Button(library.prunaVAEPath == nil ? "从 ModelScope 下载" : "校验／复用") {
                            library.downloadPrunaVAE(studio: studio)
                        }.disabled(library.busy || locked || studio.imageInputsBusy)
                            .accessibilityIdentifier("downloadPrunaVAE")
                    }
                }.padding(18).background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
                Divider()
                Text("Tiny VAE · 编码、解码与实时预览").font(.title2.weight(.semibold))
                Text("官方 safetensors，下载后可直接用于原生运行，不需要 Python 或权重转换。下载不会切换基础模型或覆盖原始 VAE。")
                    .font(.callout).foregroundStyle(.secondary)
                Picker("下载来源", selection: $source) {
                    ForEach(TinyVAESource.allCases) { Text($0.title).tag($0) }
                }.pickerStyle(.segmented).frame(maxWidth: 460).accessibilityIdentifier(
                    "tinyVAEDownloadSource")
                Text("镜像仅提供传输；固定官方版本、大小和 SHA256 校验。公共下载不发送登录凭据。")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(TinyVAEAsset.all) { asset in
                    let installed = library.tinyVAEPath(asset) != nil
                    let bundled = asset.bundledPath() != nil
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Image(systemName: "photo.stack").foregroundStyle(.tint)
                            Text(asset.title).font(.headline)
                            Spacer()
                            Text(
                                ByteCountFormatter.string(
                                    fromByteCount: asset.size, countStyle: .file)
                            ).font(.caption).monospacedDigit()
                        }
                        Text(asset.repository).font(.caption).foregroundStyle(.secondary)
                            .textSelection(.enabled)
                        Text(
                            asset.id == "taef1"
                                ? "Z-Image 的 RGB Tiny VAE；现有文生图使用解码器与实时预览。"
                                : "Qwen Image 2.1 的 RGBA Tiny VAE；支持参考图编码、最终解码与实时预览。"
                        )
                        .font(.caption).foregroundStyle(.secondary)
                        HStack {
                            if installed || bundled {
                                Label(installed ? "已安装 · SHA256 已校验" : "App 内置",
                                      systemImage: "checkmark.circle").font(
                                    .caption
                                ).foregroundStyle(.green)
                                Spacer()
                                Button("用于创作") {
                                    library.useTinyVAE(asset, studio: studio)
                                }.disabled(library.busy || locked || studio.imageInputsBusy)
                                    .accessibilityIdentifier("useTinyVAE-\(asset.id)")
                            } else {
                                Spacer()
                            }
                            if installed || !bundled {
                                Button(installed ? "校验／复用" : "下载") {
                                    library.downloadTinyVAE(asset, source: source)
                                }.disabled(library.busy || locked).accessibilityIdentifier(
                                    "downloadTinyVAE-\(asset.id)")
                            }
                        }
                    }.padding(18).background(
                        Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
                }
                if let event = library.event {
                    ProgressView(
                        value: Double(event.completedBytes), total: Double(max(1, event.totalBytes))
                    )
                    Text(
                        "\(event.phase) · \(ByteCountFormatter.string(fromByteCount: event.completedBytes, countStyle: .file)) / \(ByteCountFormatter.string(fromByteCount: event.totalBytes, countStyle: .file))"
                    )
                    .font(.caption).monospacedDigit()
                }
                Text("Tiny 最终编解码是近似选项，会损失细节；仅预览不会改变最终采样结果。")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(22).frame(maxWidth: .infinity, alignment: .leading)
        }.accessibilityIdentifier("tinyVAELibrary")
    }
}
