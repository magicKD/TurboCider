import SwiftUI
import AppKit

/// Browsing a model never changes the active creation draft or unloads a session.
struct ModelLibraryView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    @ObservedObject var library: ModelLibraryController
    let chooseModel: (String) -> Void
    let loadModel: (String) -> Void
    let importConfiguration: () -> Void
    let operationName: (String) -> String
    @State private var query = ""
    @State private var category = "all"
    @State private var selection: String?
    @State private var downloadModel: StudioModel?

    private var matches: [StudioModel] {
        studio.models.filter { model in
            let path = studio.draft.modelPaths[model.id] ?? ""
            return model.matchesLibrarySearch(query, path: path) &&
                (category != "local" || !path.isEmpty) &&
                (category != "image" || !model.isVideo) &&
                (category != "video" || model.isVideo)
        }
    }
    private var selected: StudioModel? {
        matches.first { $0.id == (selection ?? studio.draft.modelID) } ?? matches.first
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("模型库").font(.largeTitle.weight(.semibold))
                    Text("浏览模型能力，管理本地文件与运行会话。").foregroundStyle(.secondary)
                }
                Spacer()
                Button(action: importConfiguration) { Label("导入配置", systemImage: "square.and.arrow.down") }
                    .disabled(store.busy || studio.importing).accessibilityIdentifier("importConfiguration")
            }
            sessionBanner
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("统一模型目录 · \(library.installations.count) 个登记").font(.caption.weight(.medium))
                    Text(library.root).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
                }
                Spacer()
                Menu("管理目录") {
                    Button("更换模型库目录…", action: chooseLibraryRoot)
                    Button("导入模型与 LoRA 配置…", action: importPaths)
                    Button("导出模型与 LoRA 配置…", action: exportPaths)
                    Button("在 Finder 中打开") { NSWorkspace.shared.open(URL(fileURLWithPath: library.root)) }
                    Button("刷新并同步已有路径") { library.refresh(studio: studio, migrate: true) }
                }.disabled(library.busy || store.busy).accessibilityIdentifier("manageModelLibrary")
            }
            if library.busy {
                HStack { ProgressView().controlSize(.small); Text(library.event == nil ? "正在检查模型库…" : "模型下载进行中").font(.caption); Spacer(); Button("取消") { library.cancel() } }
            }
            if let message = library.message { Text(message).font(.caption).foregroundStyle(.secondary).lineLimit(3).textSelection(.enabled) }
            HStack(spacing: 12) {
                HStack {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("搜索名称、模型 ID 或路径", text: $query).textFieldStyle(.plain)
                        .accessibilityIdentifier("modelSearch")
                    if !query.isEmpty { Button { query = "" } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).help("清空搜索") }
                }.padding(9).background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 8))
                Picker("筛选", selection: $category) {
                    Text("全部").tag("all"); Text("已登记").tag("local")
                    Text("图像").tag("image"); Text("视频").tag("video")
                }.labelsHidden().frame(width: 120).accessibilityIdentifier("modelFilter")
            }
            HStack(alignment: .top, spacing: 0) {
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(matches) { item in
                            Button { selection = item.id } label: { modelRow(item) }.buttonStyle(.plain)
                                .accessibilityIdentifier("modelRow-\(item.id)")
                        }
                    }.padding(6)
                }.frame(width: 250)
                Divider()
                ScrollView {
                    if let selected { details(selected).padding(22) }
                    else {
                        ContentUnavailableView("没有匹配的模型", systemImage: "magnifyingglass", description: Text("试试其他名称，或切换到“全部”。")).padding(30)
                    }
                }.frame(maxWidth: .infinity)
            }.background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
        }.padding(24)
            .sheet(item: $downloadModel) { item in ModelDownloadView(model: item, library: library, studio: studio) }
            .onChange(of: studio.draft.modelPaths) { _, _ in library.refresh(studio: studio, migrate: true) }
    }

    private var sessionBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: store.loadedModelID == nil ? "memorychip" : "memorychip.fill").font(.title2).foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 4) {
                Text(store.loadedModelID.flatMap { id in studio.models.first { $0.id == id }?.name } ?? "没有打开的模型会话").font(.headline)
                Text(store.sessionState).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button("释放内存") { Task { do { try await store.unload() } catch { studio.message = error.localizedDescription } } }
                .disabled(!store.canUnload).accessibilityIdentifier("unloadModel")
        }.padding(16).background(Color.accentColor.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
    }

    private func modelRow(_ item: StudioModel) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: item.isVideo ? "film" : "photo").font(.title3).frame(width: 28, height: 30)
            VStack(alignment: .leading, spacing: 5) {
                Text(item.name).font(.callout.weight(.medium)).multilineTextAlignment(.leading)
                Text(item.isVideo ? "视频生成" : "图像生成").font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 5) {
                    if studio.draft.modelID == item.id { Text("当前创作").foregroundStyle(.tint) }
                    if store.loadedModelID == item.id { Text("会话已打开").foregroundStyle(.green) }
                    else { Text((studio.draft.modelPaths[item.id] ?? "").isEmpty ? "未登记" : "已登记").foregroundStyle(.secondary) }
                }.font(.caption2)
            }
            Spacer(minLength: 0)
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(selected?.id == item.id ? Color.accentColor.opacity(0.12) : .clear, in: RoundedRectangle(cornerRadius: 8))
            .contentShape(Rectangle())
    }

    private func details(_ item: StudioModel) -> some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text(item.name).font(.title2.weight(.semibold))
                Text(item.id).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                Text(item.availableOperations.map(operationName).joined(separator: " · ")).font(.callout)
                HStack {
                    Label(item.acceptsImageInputs ? "文字与图片输入" : "仅文字输入", systemImage: item.acceptsImageInputs ? "photo.on.rectangle" : "text.alignleft")
                    if item.supports_lora == true { Text("LoRA") }
                    if item.supports_gpu_ane == true { Text("GPU + ANE") }
                }.font(.caption).foregroundStyle(.secondary)
            }
            if item.executor {
                VStack(alignment: .leading, spacing: 10) {
                    Text("本地文件").font(.headline)
                    let path = studio.draft.modelPaths[item.id] ?? ""
                    Text(path.isEmpty ? "尚未登记模型文件夹" : path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    HStack {
                        Button("选择文件夹…") { chooseModel(item.id) }.disabled(store.busy || library.busy).accessibilityIdentifier("chooseModel")
                        Button("下载模型…") { downloadModel = item }.disabled(library.busy).accessibilityIdentifier("downloadModel")
                        if !path.isEmpty {
                            Button("在 Finder 中显示") { NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: path) }
                            Button("检查安装") { library.inspect(modelID: item.id, path: path) }
                                .disabled(library.busy).accessibilityIdentifier("inspectInstallation")
                        }
                    }
                    if let report = library.inspection(modelID: item.id, path: path) { inspectionView(report) }
                    if item.id == "z-image-turbo" {
                        Text("支持 Comfy 和 Diffusers 目录；可复用 FLUX.2 Klein 4B 的文本编码器与 tokenizer。").font(.caption).foregroundStyle(.secondary)
                    }
                    if !item.acceptsImageInputs {
                        Text("此执行器当前只接受提示词，创作页不会启用图片输入。").font(.caption).foregroundStyle(.secondary)
                    }
                }
                if item.supports_lora == true { loraLibrary(item) }
                let registered = library.installations.filter { $0.modelID == item.id }
                if !registered.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("已登记的安装").font(.headline)
                        ForEach(registered) { installation in
                            VStack(alignment: .leading, spacing: 5) {
                                HStack {
                                    Text(installation.name).font(.callout.weight(.medium)); Spacer()
                                    Text(installation.managed ? "托管下载" : "外部目录").font(.caption2).foregroundStyle(.secondary)
                                }
                                Text(installation.path).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
                                HStack {
                                    Button("使用此安装") { studio.draft.modelPaths[item.id] = installation.path; studio.selectModel(item.id) }
                                        .disabled(store.busy || library.busy || studio.draft.modelPaths[item.id] == installation.path)
                                    Button("移除登记") { library.remove(installation, studio: studio) }.disabled(store.busy || library.busy)
                                        .help("仅移除登记，保留模型文件。")
                                }
                            }.padding(10).background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
                Divider()
                HStack {
                    Button(studio.draft.modelID == item.id ? "当前创作模型" : "用于创作") { studio.selectModel(item.id) }
                        .buttonStyle(.borderedProminent).disabled(studio.draft.modelID == item.id || store.busy || studio.importing)
                    Button("加载权重") { loadModel(item.id) }
                        .disabled(store.busy || (studio.draft.modelPaths[item.id] ?? "").isEmpty).accessibilityIdentifier("loadModel")
                }
                Text("加载与预热可能需要准备提示词和加速分区。会话打开后按需加载权重；分阶段模式会释放已完成阶段的权重。").font(.caption).foregroundStyle(.secondary)
                if studio.draft.modelID == item.id {
                    DisclosureGroup("加速与编译缓存") { AccelerationView(store: store, studio: studio).padding(.top, 12) }
                } else {
                    Text("设为创作模型后，可配置此模型的 GPU / ANE 加速。").font(.caption).foregroundStyle(.secondary)
                }
            } else { Text("此模型执行器尚未开放。").foregroundStyle(.secondary) }
            if store.loadedModelID == item.id, let report = store.sessionReport {
                Text("当前会话最近采样").font(.headline)
                RunInsightsView(json: report)
                if item.id == "z-image-turbo" {
                    Text("Z-Image 在文本编码后释放编码器权重，保留编码结果供相同提示词复用。释放会话也会清除此内存缓存。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                DisclosureGroup("会话资源报告") { Text(report).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
            }
            if let message = studio.message { Text(message).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private func inspectionView(_ report: InstallationInspection) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(report.title, systemImage: report.status == "files_present" ? "checkmark.circle" : "exclamationmark.circle")
                .font(.callout.weight(.medium))
            Text("已检查 \(report.checkedWeightFiles) 个权重文件 · \(ByteCountFormatter.string(fromByteCount: Int64(clamping: report.referencedWeightBytes), countStyle: .file)) 引用大小")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(Array(report.issues.enumerated()), id: \.offset) { _, issue in
                Text("\(issue.path)：\(issue.message)").font(.caption).textSelection(.enabled)
            }
            ForEach(report.preparation, id: \.self) { Text($0).font(.caption).textSelection(.enabled) }
            Text("检查于 \(report.checkedAt.formatted(date: .omitted, time: .shortened))。仅检查所需文件、索引和张量字节范围；实际加载仍会验证权重内容。共享文件的引用大小不代表新增磁盘占用。")
                .font(.caption2).foregroundStyle(.secondary)
        }.padding(12).background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
    }
    private func chooseLibraryRoot() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
        panel.message = "选择 App 和 CLI 共用的模型目录。更换目录不会移动或删除已有模型文件。"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        library.configure(root: url, studio: studio)
    }
    private func loraLibrary(_ model: StudioModel) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("LoRA 模型库").font(.headline)
                Spacer()
                Button("登记文件…") {
                    let panel = NSOpenPanel(); panel.canChooseDirectories = false
                    if panel.runModal() == .OK, let url = panel.url { library.registerLoRA(url, modelID: model.id) }
                }
                Button("扫描模型目录") { library.discoverLoRAs(modelID: model.id, path: studio.draft.modelPaths[model.id] ?? "") }
                    .disabled((studio.draft.modelPaths[model.id] ?? "").isEmpty)
            }
            Text("按基础模型登记外部文件，不复制权重。扫描 loras 目录后，选择需要启用的文件。").font(.caption).foregroundStyle(.secondary)
            ForEach(library.loras.filter { $0.modelID == model.id }) { item in
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.name).font(.callout)
                    Text(item.path).font(.caption2).textSelection(.enabled)
                    HStack {
                        if !FileManager.default.isReadableFile(atPath: item.path) { Text("文件不可读取，请重新登记").foregroundStyle(.red) }
                        Button("添加到创作") {
                            if studio.draft.modelID != model.id { studio.selectModel(model.id) }
                            studio.draft.loras.append(StudioLoRA(path: item.path))
                        }.disabled(!FileManager.default.isReadableFile(atPath: item.path) || (studio.draft.modelID == model.id && (studio.draft.loras.count >= 8 || studio.draft.loras.contains { $0.path == item.path })))
                        Button("移除登记") { library.removeLoRA(item) }
                    }
                }
            }
        }.disabled(library.busy || store.busy)
    }
    private func exportPaths() {
        let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "turbocider-models.json"
        if panel.runModal() == .OK, let url = panel.url { library.exportConfiguration(to: url, studio: studio) }
    }
    private func importPaths() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        panel.message = "选择包含 modelPaths 的 JSON 文件；只登记模型路径，不改变提示词和生成参数。"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        library.importPaths(url, studio: studio)
    }
}
