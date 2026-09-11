import SwiftUI
import AppKit

struct ModelDownloadView: View {
    let model: StudioModel
    @ObservedObject var library: ModelLibraryController
    @ObservedObject var studio: StudioState
    @Environment(\.dismiss) private var dismiss
    @State private var provider: HubProvider = .modelscope
    @State private var repository = ""
    @State private var revision = ""
    @State private var components: [String: LibraryComponent] = [:]
    @State private var plan: LibraryDownloadPlan?
    @State private var selectedPaths: Set<String> = []
    @State private var inspectingShared = false
    @State private var localError: String?
    @State private var variantID = "bf16"
    @State private var textPrecision = "bf16"
    private var isComfy: Bool { model.id == "z-image-turbo" && repository == ZImageVariant.repository }
    private var variant: ZImageVariant { ZImageVariant.all.first { $0.id == variantID } ?? ZImageVariant.all[0] }
    private var estimatedWeightBytes: Int64 {
        variant.weightBytes + 335_304_388 + (textPrecision == "q4" ? 3_049_229_147 : textPrecision == "q8" ? 4_865_887_491 : 8_044_982_048)
    }
    private var compatibleSelection: Bool { !isComfy || (variant.runnable && (textPrecision == "bf16" || !components.isEmpty)) }
    private var recipe: LibraryRecipe? { LibraryRecipe.all.first { $0.modelID == model.id } }
    private var busy: Bool { library.busy || inspectingShared }
    private var fingerprint: String { "\(provider.rawValue)|\(repository)|\(revision)|\(variantID)|\(textPrecision)|\(components.sorted { $0.key < $1.key }.map { $0.value.path }.joined(separator: "|"))" }
    private func request(include: [String]? = nil) -> LibraryDownloadRequest {
        if isComfy {
            return LibraryDownloadRequest(modelID: model.id, repository: repository, provider: provider,
                revision: revision.isEmpty ? nil : revision, name: "\(model.name) · \(variantID) · text \(textPrecision)",
                include: include ?? variant.include(sharedText: !components.isEmpty), components: components,
                supplements: components.isEmpty ? [ZImageVariant.tokenizerSource] : [])
        }
        return LibraryDownloadRequest(modelID: model.id, repository: repository.trimmingCharacters(in: .whitespacesAndNewlines), provider: provider,
            revision: revision.isEmpty ? nil : revision, name: model.name,
            include: include ?? (repository == recipe?.repository ? recipe?.include ?? [] : []), components: components,
            supplements: repository == recipe?.repository ? recipe?.supplements ?? [] : [])
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack { Text("下载 \(model.name)").font(.title2.weight(.semibold)); Spacer(); Button("完成") { dismiss() }.disabled(busy) }
            Text("先查看所需文件和空间，再开始下载。文件统一保存在模型库。").foregroundStyle(.secondary)
            Form {
                Picker("来源", selection: $provider) { Text("ModelScope（默认）").tag(HubProvider.modelscope); Text("Hugging Face").tag(HubProvider.huggingface) }
                    .accessibilityIdentifier("downloadProvider")
                TextField("仓库 owner/name", text: $repository).accessibilityIdentifier("downloadRepository")
                TextField("版本（留空使用来源默认版本）", text: $revision)
                LabeledContent("保存目录", value: library.root).font(.caption).textSelection(.enabled)
            }.disabled(busy)
            if isComfy {
                GroupBox("模型版本与统一内存") {
                    VStack(alignment: .leading, spacing: 8) {
                        Picker("图像模型", selection: $variantID) {
                            ForEach(ZImageVariant.all) { Text($0.title).tag($0.id) }
                        }.accessibilityIdentifier("downloadModelVariant")
                        Text(variant.note).font(.caption)
                        Picker("Qwen3-4B 文本编码器", selection: $textPrecision) {
                            Text("BF16 · 下载或复用本地").tag("bf16")
                            Text("MLX Q4 · 选择本地转换组件").tag("q4")
                            Text("MLX Q8 · 选择本地转换组件").tag("q8")
                        }.accessibilityIdentifier("downloadTextPrecision")
                        Text("本机统一内存 \(bytes(Int64(ProcessInfo.processInfo.physicalMemory)))。权重大小不等于运行峰值，需额外预留激活、VAE、系统内存；量化并不保证更快。").font(.caption).foregroundStyle(.secondary)
                        Text("所选三组件权重合计约 \(bytes(estimatedWeightBytes))（磁盘估算，非最低内存要求；已有组件可共享）。").font(.caption).foregroundStyle(.secondary)
                        if textPrecision != "bf16" {
                            Text("先用 tools/convert/qwen3_affine.py 转换为 \(textPrecision.uppercased())，再在下方选择组件。FP4/FP8 mixed 和 GGUF 不等于 MLX affine Q4/Q8。").font(.caption)
                        }
                    }.disabled(busy)
                }
            }
            if let recipe { Text(recipe.preparation).font(.caption).foregroundStyle(.secondary) }
            if let url = URL(string: "\(provider.endpoint.absoluteString)/\(repository)") {
                Link("查看来源与模型许可证", destination: url).font(.caption)
            }
            if repository == recipe?.repository {
                ForEach(recipe?.supplements ?? [], id: \.repository) { source in
                    Link("补充组件来源与许可证：\(source.repository)", destination: provider.endpoint.appendingPathComponent(source.repository)).font(.caption)
                }
            }
            if isComfy && components.isEmpty {
                Text("此 Comfy 仓库不含 tokenizer，将从 Tongyi-MAI/Z-Image-Turbo 补充；来源与版本会写入下载记录。").font(.caption).foregroundStyle(.secondary)
            }
            if model.id == "z-image-turbo" {
                GroupBox("共享文本组件") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(components.isEmpty ? "已有 FLUX.2 Klein 4B 等兼容模型时，可复用编码器和 tokenizer。" : "已检查 Qwen3-4B 结构，将链接本地组件并跳过对应下载。")
                            .font(.caption).foregroundStyle(.secondary)
                        if let path = components["text_encoder"]?.path { Text(path).font(.caption2).textSelection(.enabled) }
                        HStack {
                            Button("选择已有文本模型…", action: selectShared).accessibilityIdentifier("chooseSharedText")
                            if !components.isEmpty { Button("改为下载文本组件") { components = [:] } }
                            if inspectingShared { ProgressView().controlSize(.small) }
                        }.disabled(busy)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            HStack {
                Button("预览下载") {
                    library.preview(request()) { value in plan = value; selectedPaths = Set(value.files.map(\.path)) }
                }.disabled(busy || repository.isEmpty || !compatibleSelection).accessibilityIdentifier("previewDownload")
                if busy { ProgressView().controlSize(.small); Button("取消") { library.cancel() }.disabled(inspectingShared) }
                Spacer()
                Button("开始下载") { library.download(request(include: selectedPaths.sorted()), studio: studio) }
                    .buttonStyle(.borderedProminent).disabled(busy || !compatibleSelection || plan == nil || selectedPaths.isEmpty || library.event?.phase == "complete")
                    .accessibilityIdentifier("startDownload")
            }
            if let plan {
                Text("预览：需下载 \(bytes(plan.downloadBytes)) · 已缓存 \(bytes(plan.cachedBytes)) · \(plan.files.count) 个文件")
                    .font(.callout.weight(.medium))
                Text("取消勾选文件可能使模型不完整。共享组件不计入下载大小。来源版本在正式下载时会重新检查。")
                    .font(.caption).foregroundStyle(.secondary)
                List(plan.files, id: \.path) { file in
                    Toggle(isOn: Binding(get: { selectedPaths.contains(file.path) }, set: { if $0 { selectedPaths.insert(file.path) } else { selectedPaths.remove(file.path) } })) {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(file.path).font(.caption)
                                if let source = file.sourceRepository { Text(source).font(.caption2).foregroundStyle(.secondary) }
                            }
                            Spacer(); Text(bytes(file.size)).font(.caption).monospacedDigit()
                        }
                    }.toggleStyle(.checkbox).disabled(busy)
                }.frame(minHeight: 130, maxHeight: 220)
            }
            if let event = library.event {
                ProgressView(value: Double(event.completedBytes), total: Double(max(1, event.totalBytes)))
                Text("\(phase(event.phase)) · \(event.completedFiles)/\(event.totalFiles) 文件 · \(bytes(event.completedBytes))/\(bytes(event.totalBytes))").font(.caption).monospacedDigit()
                Text(event.file).font(.caption2).lineLimit(2).textSelection(.enabled)
            }
            if let error = localError ?? library.message { Text(error).font(.caption).textSelection(.enabled) }
        }.padding(24).frame(width: 660)
            .onAppear { repository = model.id == "z-image-turbo" ? ZImageVariant.repository : recipe?.repository ?? ""; library.message = nil }
            .onChange(of: textPrecision) { _, _ in components = [:] }
            .onChange(of: fingerprint) { _, _ in plan = nil; selectedPaths = [] }
            .interactiveDismissDisabled(busy)
    }
    private func bytes(_ value: Int64) -> String { ByteCountFormatter.string(fromByteCount: value, countStyle: .file) }
    private func phase(_ value: String) -> String {
        ["resolving": "检查来源", "downloading": "下载中", "verifying": "校验中", "installed_file": "文件就绪", "complete": "已完成"][value] ?? value
    }
    private func selectShared() {
        let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
        panel.message = "选择包含 text_encoder 和 tokenizer 的 Qwen3-4B 模型目录。"
        if let path = studio.draft.modelPaths["flux2-klein-4b"], !path.isEmpty { panel.directoryURL = URL(fileURLWithPath: path) }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        inspectingShared = true; localError = nil
        Task {
            defer { inspectingShared = false }
            do {
                let checked = try LibraryTool.decode([String: LibraryComponent].self, from: await LibraryTool.run(["shared-text", url.path]))
                if isComfy {
                    let data = try Data(contentsOf: url.appendingPathComponent("text_encoder/config.json"))
                    let config = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    let quant = config?["quantization"] as? [String: Any]
                    let valid = textPrecision == "bf16" ? quant == nil :
                        (quant?["bits"] as? Int == (textPrecision == "q4" ? 4 : 8) && quant?["group_size"] as? Int == 32 &&
                         quant?["mode"] as? String == "affine" && config?["turbocider_dense_embedding"] as? Bool == true)
                    guard valid else {
                        throw LibraryFailure(message: "请选择由 qwen3_affine.py 生成的对应精度组件（group 32、保留 BF16 embedding）。")
                    }
                }
                components = checked
            }
            catch { localError = error.localizedDescription }
        }
    }
}
