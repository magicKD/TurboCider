import SwiftUI
import AppKit

/// Preview is the only network action before the user starts a download.
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
    @State private var presetID = "bf16"
    @State private var componentKind = "all"
    private var recipe: LibraryRecipe? { LibraryRecipe.all.first { $0.modelID == model.id } }
    private var presets: [LibraryVariant] { recipe?.downloadVariants ?? [] }
    private var preset: LibraryVariant? { presets.first { $0.id == presetID } ?? presets.first }
    private var isPreset: Bool { model.id != "z-image-turbo" && repository == preset?.repository }
    private var isComfy: Bool { model.id == "z-image-turbo" && repository == ZImageVariant.repository }
    private var variant: ZImageVariant { ZImageVariant.all.first { $0.id == variantID } ?? ZImageVariant.all[0] }
    private var estimatedWeightBytes: Int64 {
        variant.weightBytes + 335_304_388 + (textPrecision == "q4" ? 3_049_229_147 : textPrecision == "q8" ? 4_865_887_491 : 8_044_982_048)
    }
    private var compatibleSelection: Bool { !isComfy || (variant.runnable && (textPrecision == "bf16" || !components.isEmpty)) }
    private var busy: Bool { library.busy || inspectingShared }
    private var fingerprint: String {
        "\(provider.rawValue)|\(repository)|\(revision)|\(variantID)|\(textPrecision)|\(presetID)|\(componentKind)|\(components.sorted { $0.key < $1.key }.map { $0.value.path }.joined(separator: "|"))"
    }
    private func request(include: [String]? = nil) throws -> LibraryDownloadRequest {
        if isComfy {
            let base = LibraryDownloadRequest(modelID: model.id, repository: repository, provider: provider,
                revision: revision.isEmpty ? nil : revision, name: "\(model.name) · \(variantID) · text \(textPrecision)",
                include: variant.include(sharedText: !components.isEmpty), components: components,
                supplements: components.isEmpty ? [ZImageVariant.tokenizerSource] : [])
            return try include.map { try base.subset(include: $0) } ?? base
        }
        if isPreset, let preset {
            var base = try preset.request(modelID: model.id, provider: provider, componentKind: componentKind == "all" ? nil : componentKind)
            base.revision = revision.isEmpty ? nil : revision
            return try include.map { try base.subset(include: $0) } ?? base
        }
        return LibraryDownloadRequest(modelID: model.id, repository: repository.trimmingCharacters(in: .whitespacesAndNewlines), provider: provider,
            revision: revision.isEmpty ? nil : revision, name: model.name, include: include ?? [], components: components)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                ModelBrandBadge(modelID: model.id, size: 44)
                VStack(alignment: .leading, spacing: 4) {
                    Text("下载 \(model.displayName)").font(.title2.weight(.semibold))
                    Text("选择版本与组件，先预览再下载。").font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                Button("完成") { dismiss() }.disabled(busy)
            }.padding(22)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("下载来源").font(.headline)
                        Picker("来源", selection: $provider) {
                            Text("ModelScope · 默认").tag(HubProvider.modelscope)
                            Text("Hugging Face").tag(HubProvider.huggingface)
                        }.pickerStyle(.segmented).accessibilityIdentifier("downloadProvider")
                        Text("预览会核对主仓库与配套组件。若来源缺少所需文件，可切换另一来源；不会自动下载替代版本。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if presets.count > 1 && model.id != "z-image-turbo" {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("模型版本").font(.headline)
                            Picker("精度与格式", selection: $presetID) {
                                ForEach(presets) { Text($0.title).tag($0.id) }
                            }.accessibilityIdentifier("downloadQuantization")
                            if let preset {
                                Label(preset.runnable ? (preset.id == "bf16" ? "原生 GPU · 生成与参考编辑" : "实验性近似文生图") : "可下载 · 当前 App 暂不支持运行",
                                      systemImage: preset.runnable ? "checkmark.circle" : "info.circle")
                                    .font(.caption.weight(.medium)).foregroundStyle(preset.runnable ? Color.accentColor : Color.orange)
                                Text(preset.preparation).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    if isPreset, let preset, !preset.components.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("下载组件").font(.headline)
                                Spacer()
                                Picker("下载范围", selection: $componentKind) {
                                    Text("完整模型").tag("all")
                                    Text("仅 DiT").tag("dit")
                                    Text("仅 Encoder + Tokenizer").tag("encoder")
                                    Text("仅 VAE").tag("vae")
                                }.labelsHidden().frame(width: 230).accessibilityIdentifier("downloadComponentScope")
                            }
                            ForEach(Array(preset.components.filter { componentKind == "all" || $0.kind == componentKind || (componentKind == "encoder" && $0.kind == "tokenizer") }.enumerated()), id: \.offset) { _, component in
                                HStack(alignment: .top, spacing: 10) {
                                    Image(systemName: component.kind == "dit" ? "square.stack.3d.up" : component.kind == "vae" ? "photo.stack" : "text.viewfinder")
                                        .foregroundStyle(.tint).frame(width: 22)
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(component.title).font(.callout.weight(.medium))
                                        Text(component.repository).font(.caption2).foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    Text("\(component.format) · \(component.precision)").font(.caption).foregroundStyle(.secondary)
                                }.padding(10).background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 10))
                            }
                            Text("仅下载单个组件不会替换当前创作模型。完整模型需要兼容的 DiT、Encoder、VAE 与 Tokenizer。").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if isComfy { zImageOptions }
                    if model.id == "z-image-turbo" { sharedTextOptions }
                    if presets.count <= 1, let recipe { Text(recipe.preparation).font(.caption).foregroundStyle(.secondary) }
                    DisclosureGroup("仓库与版本") {
                        VStack(alignment: .leading, spacing: 10) {
                            TextField("仓库 owner/name", text: $repository).textFieldStyle(.roundedBorder).accessibilityIdentifier("downloadRepository")
                            TextField("版本 · 留空使用来源默认版本", text: $revision).textFieldStyle(.roundedBorder)
                            Text("修改仓库后将使用该仓库的自定义文件选择；请核对模型兼容性。").font(.caption).foregroundStyle(.secondary)
                        }.padding(.top, 10)
                    }
                    Link("查看来源与模型许可证 · \(repository)", destination: repositoryURL(repository)).font(.caption)
                    if isPreset {
                        ForEach((try? request())?.supplements ?? [], id: \.repository) { source in
                            Link("配套组件 · \(source.repository)", destination: repositoryURL(source.repository)).font(.caption)
                        }
                    }
                    Text("保存到：\(library.root)").font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
                    if let plan {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("下载预览 · \(plan.files.count) 个文件").font(.headline)
                            Text("取消勾选可仅保存部分组件；不完整的安装不会自动用于创作。").font(.caption).foregroundStyle(.secondary)
                            ForEach(plan.files, id: \.path) { file in
                                Toggle(isOn: Binding(get: { selectedPaths.contains(file.path) }, set: {
                                    if $0 { selectedPaths.insert(file.path) } else { selectedPaths.remove(file.path) }
                                })) {
                                    HStack(alignment: .top) {
                                        VStack(alignment: .leading, spacing: 3) {
                                            Text(file.installationPath).font(.caption).textSelection(.enabled)
                                            if let source = file.sourceRepository { Text(source).font(.caption2).foregroundStyle(.secondary) }
                                        }
                                        Spacer(); Text(bytes(file.size)).font(.caption).monospacedDigit()
                                    }
                                }.toggleStyle(.checkbox).padding(.vertical, 4)
                            }
                        }
                    }
                }.padding(22).disabled(busy)
            }
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                if let event = library.event {
                    ProgressView(value: Double(event.completedBytes), total: Double(max(1, event.totalBytes)))
                    Text("\(phase(event.phase)) · \(event.completedFiles)/\(event.totalFiles) 文件 · \(bytes(event.completedBytes))/\(bytes(event.totalBytes))")
                        .font(.caption).monospacedDigit()
                    if !event.file.isEmpty { Text(event.file).font(.caption2).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle) }
                }
                if let error = localError ?? library.message { Text(error).font(.caption).foregroundStyle(.secondary).lineLimit(3).textSelection(.enabled) }
                HStack {
                    Button("预览下载") { preview() }.disabled(busy || repository.isEmpty || !compatibleSelection).accessibilityIdentifier("previewDownload")
                    if busy { ProgressView().controlSize(.small); Button("取消") { library.cancel() }.disabled(inspectingShared) }
                    Spacer()
                    if let plan {
                        VStack(alignment: .trailing, spacing: 2) {
                            Text(bytes(plan.files.filter { selectedPaths.contains($0.path) }.reduce(0) { $0 + $1.size })).font(.callout.weight(.semibold))
                            Text("完整预览缓存 \(bytes(plan.cachedBytes))").font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    Button("开始下载") { startDownload() }.buttonStyle(.borderedProminent)
                        .disabled(busy || !compatibleSelection || plan == nil || selectedPaths.isEmpty)
                        .accessibilityIdentifier("startDownload")
                }
            }.padding(18)
        }.frame(width: 680, height: 700)
            .onAppear {
                presetID = presets.first?.id ?? "default"
                repository = model.id == "z-image-turbo" ? ZImageVariant.repository : preset?.repository ?? ""
                library.message = nil
            }
            .onChange(of: presetID) { _, _ in
                // Z-Image keeps its independent Comfy precision controls.
                guard model.id != "z-image-turbo" else { return }
                repository = preset?.repository ?? ""; revision = ""
            }
            .onChange(of: textPrecision) { _, _ in components = [:] }
            .onChange(of: fingerprint) { _, _ in plan = nil; selectedPaths = []; localError = nil }
            .interactiveDismissDisabled(busy)
    }
    private func repositoryURL(_ repository: String) -> URL {
        provider.endpoint.appendingPathComponent(provider == .modelscope ? "models/" + repository : repository)
    }
    private func preview() {
        localError = nil
        do { library.preview(try request()) { value in plan = value; selectedPaths = Set(value.files.map(\.path)) } }
        catch { localError = error.localizedDescription }
    }
    private func startDownload() {
        localError = nil
        do { library.download(try request(include: selectedPaths.sorted()), studio: studio) }
        catch { localError = error.localizedDescription }
    }
    private var zImageOptions: some View {
        GroupBox("模型版本与统一内存") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("图像模型", selection: $variantID) { ForEach(ZImageVariant.all) { Text($0.title).tag($0.id) } }
                    .accessibilityIdentifier("downloadModelVariant")
                Text(variant.note).font(.caption)
                Picker("Qwen3-4B 文本编码器", selection: $textPrecision) {
                    Text("BF16 · 下载或复用本地").tag("bf16")
                    Text("MLX Q4 · 选择本地转换组件").tag("q4")
                    Text("MLX Q8 · 选择本地转换组件").tag("q8")
                }.accessibilityIdentifier("downloadTextPrecision")
                Text("本机统一内存 \(bytes(Int64(ProcessInfo.processInfo.physicalMemory)))；三组件权重约 \(bytes(estimatedWeightBytes))。运行需另预留激活、VAE 和系统内存。").font(.caption).foregroundStyle(.secondary)
                if textPrecision != "bf16" { Text("先用 tools/convert/qwen3_affine.py 转换为对应精度，再选择本地组件。GGUF 与 MLX affine Q4/Q8 不通用。").font(.caption) }
            }
        }
    }
    private var sharedTextOptions: some View {
        GroupBox("共享文本组件") {
            VStack(alignment: .leading, spacing: 8) {
                Text(components.isEmpty ? "可复用已有兼容 Qwen3-4B 编码器与 Tokenizer，避免重复下载。" : "组件结构检查通过，将链接本地文件并跳过对应下载。")
                    .font(.caption).foregroundStyle(.secondary)
                if let path = components["text_encoder"]?.path { Text(path).font(.caption2).textSelection(.enabled) }
                HStack {
                    Button("选择已有文本模型…", action: selectShared).accessibilityIdentifier("chooseSharedText")
                    if !components.isEmpty { Button("改为下载文本组件") { components = [:] } }
                    if inspectingShared { ProgressView().controlSize(.small) }
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
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
