import SwiftUI
import AppKit

/// Browsing a model never changes the active creation draft or unloads a session.
struct ModelLibraryView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    @ObservedObject var library: ModelLibraryController
    @Binding var selection: String?
    let chooseModel: (String) -> Void
    let loadModel: (String) -> Void
    let importConfiguration: () -> Void
    let operationName: (String) -> String
    @State private var query = ""
    @State private var category = ModelHubCategory.all
    @State private var scope = ModelHubScope.discover
    @State private var showsLoRAs = false
    @State private var downloadModel: StudioModel?
    @State private var componentModel: StudioModel?
    @State private var replacementPaths: [String: String] = [:]
    @State private var componentName = ""

    private var hub: ModelHubPresentation {
        let metadata = Dictionary(uniqueKeysWithValues: LibraryRecipe.all.map { recipe in
            (recipe.modelID, recipe.downloadVariants.map { $0.title + " " + $0.repository + " " + $0.components.map { $0.format + " " + $0.precision }.joined(separator: " ") })
        })
        return ModelHubPresentation.make(models: studio.models, installations: library.installations,
            modelPaths: studio.draft.modelPaths, upscalePaths: studio.draft.upscaleModelPaths,
            selectionID: selection ?? studio.draft.modelID, query: query, category: category, scope: scope,
            inspections: library.inspections, searchMetadata: metadata)
    }
    private var selected: StudioModel? { hub.selectedEntry?.model }
    private var selectedUpscaler: UpscaleVariant? { hub.selectedEntry?.upscaler }
    private var componentKey: String {
        guard let selected else { return "" }
        return selected.id + "\n" + (studio.draft.modelPaths[selected.id] ?? "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            sessionBanner
            Picker("管理内容", selection: $showsLoRAs) {
                Text("基础模型").tag(false)
                Text("LoRA · \(library.loras.count)").tag(true)
            }.pickerStyle(.segmented).labelsHidden().frame(width: 250).accessibilityIdentifier("modelLibrarySection")
            if library.busy || library.message != nil {
                HStack(spacing: 8) {
                    if library.busy { ProgressView().controlSize(.small) }
                    Text(library.message ?? (library.event == nil ? "正在检查模型库…" : "模型下载进行中"))
                        .font(.caption).foregroundStyle(.secondary).lineLimit(2).textSelection(.enabled)
                    Spacer(minLength: 0)
                    if library.busy { Button("取消") { library.cancel() } }
                }.padding(.horizontal, 4)
            }
            if showsLoRAs {
                ModelLoRALibraryView(store: store, studio: studio, library: library)
            } else {
            HStack(spacing: 12) {
                Picker("浏览范围", selection: $scope) {
                    Text("发现").tag(ModelHubScope.discover)
                    Text("本机模型").tag(ModelHubScope.local)
                }.pickerStyle(.segmented).labelsHidden().frame(width: 170).accessibilityIdentifier("modelHubScope")
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("搜索模型、格式或路径", text: $query).textFieldStyle(.plain)
                        .accessibilityIdentifier("modelSearch")
                    if !query.isEmpty {
                        Button { query = "" } label: { Image(systemName: "xmark.circle.fill") }
                            .buttonStyle(.plain).help("清空搜索")
                    }
                }.padding(10).background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 10))
                Picker("模型类型", selection: $category) {
                    Text("全部类型").tag(ModelHubCategory.all)
                    Text("图像").tag(ModelHubCategory.image)
                    Text("视频").tag(ModelHubCategory.video)
                    Text("超分").tag(ModelHubCategory.upscale)
                }.labelsHidden().frame(width: 110).accessibilityIdentifier("modelFilter")
            }
            HStack(alignment: .top, spacing: 0) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(scope == .discover ? "兼容模型" : "本机模型").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        Spacer(); Text("\(hub.entries.count)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }.padding(.horizontal, 14).padding(.top, 16)
                    ScrollView {
                        LazyVStack(spacing: 6) {
                            ForEach(hub.entries) { entry in
                                Button { selection = entry.id } label: { hubRow(entry) }
                                    .buttonStyle(.plain).accessibilityIdentifier("modelRow-\(entry.id)")
                            }
                            if hub.entries.isEmpty {
                                VStack(spacing: 10) {
                                    Image(systemName: "magnifyingglass").font(.title2)
                                    Text("没有匹配的模型").font(.callout.weight(.medium))
                                    Text(scope == .local ? "导入本地模型，或到“发现”查看下载选项。" : "试试其他名称或格式。")
                                        .font(.caption).multilineTextAlignment(.center)
                                    if scope == .local { Button("浏览可用模型") { scope = .discover; query = ""; category = .all } }
                                }.foregroundStyle(.secondary).padding(20)
                            }
                        }.padding(8)
                    }
                }.frame(width: 250)
                Divider()
                ScrollView {
                    if let variant = selectedUpscaler { upscalerDetails(variant).padding(22) }
                    else if let selected { details(selected).padding(22) }
                    else { ContentUnavailableView("找到适合你的模型", systemImage: "square.stack.3d.up", description: Text("浏览兼容模型，或管理本机已有文件。")).padding(30) }
                }.id(hub.selectedID).frame(maxWidth: .infinity)
            }.background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 16))
                .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.primary.opacity(0.06)))
            }
        }.padding(24)
            .task {
                if let selection, UpscaleVariant.from(modelID: selection) != nil { category = .upscale }
                library.refresh(studio: studio)
            }
            .onChange(of: store.busy) { _, busy in if !busy { library.refresh(studio: studio) } }
            .onChange(of: componentKey) { _, _ in refreshComponents() }
            .onChange(of: showsLoRAs) { _, _ in refreshComponents() }
            .onChange(of: library.busy) { _, busy in if !busy { refreshComponents() } }
            .sheet(item: $downloadModel) { item in ModelDownloadView(model: item, library: library, studio: studio) }
            .sheet(item: $componentModel) { item in componentEditor(item) }
            .onChange(of: studio.draft.upscaleModelPaths) { _, _ in library.refresh(studio: studio, migrate: true) }
            .onChange(of: studio.draft.modelPaths) { _, _ in library.refresh(studio: studio, migrate: true) }
    }
    private var header: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                Text("模型中心").font(.largeTitle.weight(.semibold))
                Text("找到合适的模型，按组件管理本机文件。").foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    hubTag("\(library.installations.count) 个本机安装", symbol: "internaldrive")
                    hubTag("\(ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory)) 统一内存", symbol: "memorychip")
                }.padding(.top, 4)
            }
            Spacer(minLength: 0)
            Menu {
                Button("更换模型库目录…", action: chooseLibraryRoot)
                Button("导入模型路径配置…", action: importPaths)
                Button("导出模型路径配置…", action: exportPaths)
                Button("导入完整创作配置…", action: importConfiguration)
                Divider()
                Button("在 Finder 中打开模型库") { NSWorkspace.shared.open(URL(fileURLWithPath: library.root)) }
                Button("刷新并同步已有路径") { library.refresh(studio: studio, migrate: true) }
            } label: { Label("管理目录", systemImage: "folder.badge.gearshape") }
                .fixedSize()
                .disabled(library.busy || store.busy || studio.imageInputsBusy)
                .accessibilityIdentifier("manageModelLibrary").help(library.root)
        }
    }
    private func hubTag(_ title: String, symbol: String? = nil, tint: Color = .secondary) -> some View {
        HStack(spacing: 4) {
            if let symbol { Image(systemName: symbol) }
            Text(title)
        }.font(.caption).foregroundStyle(tint).padding(.horizontal, 9).padding(.vertical, 5)
            .background(tint.opacity(0.08), in: Capsule())
    }
    private func refreshComponents() {
        guard !showsLoRAs, let selected, !library.busy else { return }
        let path = studio.draft.modelPaths[selected.id] ?? ""
        if !path.isEmpty { library.inspectComponents(modelID: selected.id, path: path) }
    }
    private func hubRow(_ entry: ModelHubEntry) -> some View {
        HStack(alignment: .top, spacing: 10) {
            ModelBrandBadge(modelID: entry.id, size: 38)
            VStack(alignment: .leading, spacing: 5) {
                Text(entry.title).font(.callout.weight(.semibold)).multilineTextAlignment(.leading).lineLimit(2)
                Text(entry.model?.isVideo == true ? "视频生成" : entry.upscaler != nil ? "图像超分" : "图像生成")
                    .font(.caption).foregroundStyle(.secondary)
                if entry.localState == .missing { Label("文件缺失或无效", systemImage: "exclamationmark.circle").font(.caption2).foregroundStyle(.orange) }
                else if entry.installationCount > 0 { Text("\(entry.installationCount) 个本机安装").font(.caption2).foregroundStyle(.tint) }
                else if !entry.localPaths.isEmpty { Text("本机路径已配置").font(.caption2).foregroundStyle(.secondary) }
            }
            Spacer(minLength: 0)
            if hub.selectedID == entry.id { Image(systemName: "chevron.right").font(.caption.weight(.semibold)).foregroundStyle(.tint) }
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(hub.selectedID == entry.id ? Color.accentColor.opacity(0.10) : Color.clear, in: RoundedRectangle(cornerRadius: 12))
            .contentShape(Rectangle())
    }

    private func upscalerDetails(_ variant: UpscaleVariant) -> some View {
        let path = studio.draft.upscaleModelPaths[variant.rawValue] ?? ""
        let exists = !path.isEmpty && FileManager.default.fileExists(atPath: path)
        return VStack(alignment: .leading, spacing: 16) {
            Text(variant.title).font(.title2)
            Text("把图片的宽和高各放大 \(variant.scale) 倍。原图保留，结果另存为 PNG。")
            Text(exists ? "本地模型已配置" : "需要下载或导入模型")
                .foregroundStyle(exists ? Color.green : Color.secondary)
                .accessibilityIdentifier("upscaleLibraryStatus")
            HStack {
                Link("下载模型（ZIP · 约 30 MB）", destination: variant.downloadURL)
                Link("转换与处理说明", destination: UpscaleVariant.processingURL)
            }
            Text("1. 点击下载，在浏览器中保存 ZIP。\n2. 解压后导入 .mlpackage。\n3. App 校验倍率并由 Core ML 原生编译，无需 Python。")
                .font(.callout).foregroundStyle(.secondary)
            Text("首次加载或预加载会准备模型。GPU 启动较快；可在创作参数中选择 ANE 优先。")
                .font(.caption).foregroundStyle(.secondary)
            Button("导入并校验本地模型…") { chooseUpscaler(variant) }
                .disabled(library.busy || store.busy).accessibilityIdentifier("importUpscaleModel")
            if !path.isEmpty {
                Text(path).font(.caption).textSelection(.enabled).accessibilityIdentifier("upscaleLibraryPath")
                if !exists { Text("文件已移动或不可访问，请重新导入。").foregroundStyle(.orange) }
                Button("在 Finder 中显示") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
                    .disabled(!exists)
            }
            Divider()
            Text("已登记路径").font(.headline)
            ForEach(library.installations.filter { $0.modelID == variant.modelID }) { item in
                VStack(alignment: .leading, spacing: 6) {
                    Text(item.path).font(.caption).textSelection(.enabled)
                    HStack {
                        Button(item.path == path ? "当前路径" : "使用此路径") { library.registerUpscaler(URL(fileURLWithPath: item.path), variant: variant, studio: studio) }
                            .disabled(item.path == path || !FileManager.default.fileExists(atPath: item.path))
                        Button("移除登记") { library.remove(item, studio: studio) }
                    }.disabled(library.busy || store.busy)
                }
            }
            Text("登记外部路径不会复制权重；移除登记也不会删除模型文件。路径随统一模型配置导入和导出。")
                .font(.caption).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private func chooseUpscaler(_ variant: UpscaleVariant) {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.message = "选择解压后的 \(variant.rawValue) Core ML 模型。"
        let completion: (NSApplication.ModalResponse) -> Void = { response in
            if response == .OK, let url = panel.url { library.registerUpscaler(url, variant: variant, studio: studio) }
        }
        if let window = NSApp.windows.first(where: { $0.isVisible && !($0 is NSPanel) }) {
            panel.beginSheetModal(for: window, completionHandler: completion)
        } else { panel.begin(completionHandler: completion) }
    }

    private var sessionBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "circle.fill").font(.system(size: 7)).foregroundStyle(store.loadedModelID == nil ? Color.secondary : Color.green)
            Text(store.loadedModelID.flatMap { id in studio.models.first { $0.id == id }?.name }
                ?? store.upscaleReady.map { "超分会话 · ×\($0.scale)" } ?? "本机运行")
                .font(.caption.weight(.medium))
            Text(store.sessionState).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            Spacer()
            if store.busy { Button("取消任务") { store.cancel() }.accessibilityIdentifier("cancelModelOperation") }
            Button("释放内存") { Task { do { try await store.unload() } catch { studio.message = error.localizedDescription } } }
                .disabled(!store.canUnload).accessibilityIdentifier("unloadModel")
        }.padding(.horizontal, 14).padding(.vertical, 10)
            .background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 10))
    }

    private func details(_ item: StudioModel) -> some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(alignment: .top, spacing: 14) {
                ModelBrandBadge(modelID: item.id, size: 64)
                VStack(alignment: .leading, spacing: 7) {
                    Text(item.displayName).font(.title2.weight(.semibold))
                    Text(ModelBrandBadge.publisher(item.id)).font(.callout).foregroundStyle(.secondary)
                    Text(item.id).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                }
                Spacer(minLength: 0)
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 6) { capabilityTags(item) }
                VStack(alignment: .leading, spacing: 6) { capabilityTags(item) }
            }
            if let recipe = LibraryRecipe.all.first(where: { $0.modelID == item.id }) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label("下载版本", systemImage: "arrow.down.circle").font(.headline)
                        Spacer()
                        Text(recipe.downloadVariants.count > 1 ? "\(recipe.downloadVariants.count) 个选项" : "完整组件")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text(recipe.downloadVariants.count > 1 ? "\(recipe.downloadVariants.map(\.title).joined(separator: " · "))" : recipe.repository)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(3)
                    Text("ModelScope 优先 · 可切换 Hugging Face · 下载前预览文件与大小")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding(14).frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.accentColor.opacity(0.045), in: RoundedRectangle(cornerRadius: 12))
            }
            if item.executor {
                VStack(alignment: .leading, spacing: 10) {
                    Text("当前安装").font(.headline)
                    let path = studio.draft.modelPaths[item.id] ?? ""
                    Text(path.isEmpty ? "尚未登记模型文件夹" : path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    HStack {
                        Button("选择文件夹…") { chooseModel(item.id) }.disabled(store.busy || library.busy).accessibilityIdentifier("chooseModel")
                        Button("下载模型…") { downloadModel = item }.buttonStyle(.borderedProminent).disabled(library.busy || LibraryRecipe.all.first(where: { $0.modelID == item.id }) == nil).accessibilityIdentifier("downloadModel")
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
                componentSection(item)
                if item.supports_lora == true {
                    HStack {
                        Label("LoRA · \(library.loras.filter { $0.modelID == item.id }.count) 个登记", systemImage: "slider.horizontal.3")
                        Spacer()
                        Button("管理 LoRA") { showsLoRAs = true }.accessibilityIdentifier("manageLoRAs")
                    }
                }
                if ["z-image-turbo", "flux2-klein-4b"].contains(item.id) {
                    DisclosureGroup("ANE 分区与编译缓存") { ANELibraryView(modelID: item.id, library: library, studio: studio, store: store).padding(.top, 12) }
                }
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
                                if item.id == "z-image-turbo", let variant = ZImageInstallation.variant(URL(fileURLWithPath: installation.path)) {
                                    Text(variant.title).font(.caption.weight(.medium))
                                }
                                Text(installation.path).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
                                HStack {
                                    Button("使用此安装") { studio.selectInstallation(modelID: item.id, path: installation.path) }
                                        .disabled(store.busy || store.resolvingAcceleration || studio.imageInputsBusy || library.busy || studio.draft.modelPaths[item.id] == installation.path || library.inspection(modelID: item.id, path: installation.path)?.status == "incomplete")
                                    if ["z-image-turbo", "flux2-klein-4b", "flux2-klein-9b"].contains(item.id) {
                                        Button("校验文件") {
                                            Task {
                                                do { try await store.verifyModelSources(modelURL: URL(fileURLWithPath: installation.path), modelID: item.id) }
                                                catch { studio.message = error.localizedDescription }
                                            }
                                        }
                                        .disabled(store.busy || store.requiresProcessRestart || store.externalServiceActive || store.resolvingAcceleration || studio.imageInputsBusy || library.busy)
                                        .help("读取模型文件并核对内容身份；首次校验可能需要一些时间。校验不会授予加速预设资格。")
                                    }
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
                        .buttonStyle(.borderedProminent).disabled(studio.draft.modelID == item.id || store.busy || studio.imageInputsBusy)
                    Button("加载权重") { loadModel(item.id) }
                        .disabled(store.busy || library.busy || (studio.draft.modelPaths[item.id] ?? "").isEmpty || library.inspection(modelID: item.id, path: studio.draft.modelPaths[item.id] ?? "")?.status == "incomplete").accessibilityIdentifier("loadModel")
                }
                Text("加载与预热可能需要准备提示词和加速分区。会话打开后按需加载权重；分阶段模式会释放已完成阶段的权重。").font(.caption).foregroundStyle(.secondary)
                if studio.draft.modelID == item.id {
                    DisclosureGroup("加速与编译缓存") { AccelerationView(store: store, studio: studio, library: library).padding(.top, 12) }
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
    @ViewBuilder private func capabilityTags(_ item: StudioModel) -> some View {
        hubTag(item.isVideo ? "视频" : "图像", symbol: item.isVideo ? "film" : "photo", tint: .accentColor)
        hubTag(item.acceptsImageInputs ? "参考图编辑" : "文字输入", symbol: item.acceptsImageInputs ? "photo.on.rectangle" : "text.alignleft")
        if item.supports_lora == true { hubTag("LoRA") }
        if item.supports_gpu_ane == true { hubTag("GPU + ANE") }
    }
    private func componentSection(_ model: StudioModel) -> some View {
        let path = studio.draft.modelPaths[model.id] ?? ""
        let inventory = library.components(modelID: model.id, path: path)
        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("模型组件").font(.headline)
                Spacer()
                Button { library.inspectComponents(modelID: model.id, path: path, force: true) } label: { Image(systemName: "arrow.clockwise") }
                    .disabled(path.isEmpty || library.busy).help("刷新组件状态").accessibilityIdentifier("refreshModelComponents")
            }
            if inventory?.supported == false || !["qwen-image-2.1", "z-image-turbo", "flux2-klein-4b", "flux2-klein-9b"].contains(model.id) {
                Text(inventory?.detail ?? "此模型请使用完整安装目录管理。").font(.caption).foregroundStyle(.secondary)
            } else {
                ForEach(["encoder", "dit", "vae"], id: \.self) { kind in
                    componentCard(kind, component: inventory?.components.first { $0.kind == kind }, model: model, path: path)
                }
                Text("Tokenizer 继承当前安装的配套词表。替换组件只建立新的文件链接，原安装与权重保留。").font(.caption).foregroundStyle(.secondary)
            }
            if let error = library.componentError(modelID: model.id, path: path) { Text(error).font(.caption).foregroundStyle(.orange) }
        }
    }
    private func componentCard(_ kind: String, component: LibraryResolvedComponent?, model: StudioModel, path: String) -> some View {
        let names = ["encoder": "Encoder · 文本与参考编码", "dit": "DiT · 图像生成", "vae": "VAE · 图像编解码"]
        let symbols = ["encoder": "text.viewfinder", "dit": "square.stack.3d.up", "vae": "photo.stack"]
        return HStack(alignment: .top, spacing: 12) {
            Image(systemName: symbols[kind] ?? "cpu").font(.title3).foregroundStyle(.tint)
                .frame(width: 38, height: 38).background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
            VStack(alignment: .leading, spacing: 5) {
                Text(names[kind] ?? kind).font(.callout.weight(.semibold))
                if let component {
                    Text(component.status == "present" ? "\(component.format.isEmpty ? "目录" : component.format.uppercased()) · \(component.precision) · 已找到文件" : "缺少组件")
                        .font(.caption).foregroundStyle(component.status == "present" ? Color.secondary : Color.orange)
                    if !component.path.isEmpty { Text(component.path).font(.caption2).foregroundStyle(.secondary).lineLimit(2).truncationMode(.middle).textSelection(.enabled).help(component.path) }
                } else { Text(path.isEmpty ? "导入完整模型后可分别管理" : "正在读取组件信息…").font(.caption).foregroundStyle(.secondary) }
            }
            Spacer(minLength: 0)
            VStack(alignment: .trailing, spacing: 8) {
                Button("替换…") { beginComponentEdit(model) }
                    .disabled(path.isEmpty || component == nil || library.busy || store.busy)
                    .accessibilityIdentifier("replaceComponent-\(kind)")
                if let component, !component.path.isEmpty {
                    Button { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: component.path)]) } label: { Image(systemName: "folder") }
                        .buttonStyle(.plain).help("在 Finder 中显示组件")
                }
            }
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.055)))
    }
    private func beginComponentEdit(_ model: StudioModel) {
        replacementPaths = [:]; componentName = model.displayName + " · 自定义组件"
        componentModel = model
    }
    private func componentEditor(_ model: StudioModel) -> some View {
        let path = studio.draft.modelPaths[model.id] ?? ""
        let inventory = library.components(modelID: model.id, path: path)
        return VStack(alignment: .leading, spacing: 16) {
            HStack { Text("配置模型组件").font(.title2.weight(.semibold)); Spacer(); Button("完成") { componentModel = nil }.disabled(library.busy) }
            Text("从当前安装继承其余组件，选择需要替换的部分。检查通过后保存为独立安装并用于创作。").foregroundStyle(.secondary)
            TextField("安装名称", text: $componentName).textFieldStyle(.roundedBorder).disabled(library.busy)
            ForEach(["encoder", "dit", "vae"], id: \.self) { kind in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(["encoder": "Encoder", "dit": "DiT", "vae": "VAE"][kind] ?? kind).font(.headline)
                        Spacer()
                        Button("选择文件或目录…") { chooseComponent(kind, model: model) }.accessibilityIdentifier("chooseComponent-\(kind)")
                        if replacementPaths[kind] != nil { Button("恢复继承") { replacementPaths.removeValue(forKey: kind) } }
                    }
                    Text(replacementPaths[kind] ?? inventory?.components.first(where: { $0.kind == kind })?.path ?? "未找到组件")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(3).textSelection(.enabled)
                }.padding(12).background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 10)).disabled(library.busy)
            }
            Text("保留当前安装的 Tokenizer。请使用同一模型家族的组件；完整加载仍会验证张量与模型结构。").font(.caption).foregroundStyle(.secondary)
            if let message = library.message { Text(message).font(.caption).textSelection(.enabled) }
            HStack {
                if library.busy { ProgressView().controlSize(.small); Button("取消") { library.cancel() } }
                Spacer()
                Button("检查并保存组合") {
                    library.composeComponents(.init(modelID: model.id, basePath: path, name: componentName, components: replacementPaths), studio: studio) { componentModel = nil }
                }.buttonStyle(.borderedProminent).disabled(library.busy || store.busy || replacementPaths.isEmpty)
                    .accessibilityIdentifier("saveComponentComposition")
            }
        }.padding(24).frame(width: 640).interactiveDismissDisabled(library.busy)
    }
    private func chooseComponent(_ kind: String, model: StudioModel) {
        let panel = NSOpenPanel(); panel.canChooseFiles = true; panel.canChooseDirectories = true
        panel.message = "选择与 \(model.displayName) 兼容的 \(kind.uppercased()) 文件或组件目录；Tokenizer 继承原安装。"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        replacementPaths[kind] = url.path
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

/// Publisher logos are bundled locally; a missing resource keeps the native fallback.
struct ModelBrandBadge: View {
    let modelID: String
    var size: CGFloat = 40
    private static let brandKeys = ["qwen", "flux", "z-image", "ltx", "minimax", "wan", "llada"]
    private static let logos: [String: NSImage] = Dictionary(uniqueKeysWithValues:
        brandKeys.compactMap { key -> (String, NSImage)? in
            for ext in ["png", "jpg"] {
                if let url = Bundle.main.url(forResource: key, withExtension: ext, subdirectory: "ModelBrands"),
                   let image = NSImage(contentsOf: url) { return (key, image) }
            }
            return nil
        })
    private var logo: NSImage? {
        guard let key = Self.brandKeys.first(where: { modelID.hasPrefix($0) }) else { return nil }
        return Self.logos[key]
    }
    private var tint: Color {
        if modelID.hasPrefix("qwen") { return .indigo }
        if modelID.hasPrefix("flux") { return .primary }
        if modelID.hasPrefix("z-image") { return .purple }
        if modelID.hasPrefix("ltx") { return .blue }
        if modelID.hasPrefix("minimax") { return .orange }
        if modelID.hasPrefix("wan") { return .cyan }
        if modelID.hasPrefix("llada") { return .blue }
        return .teal
    }
    private var glyph: String {
        if modelID.hasPrefix("qwen") { return "Q" }
        if modelID.hasPrefix("flux") { return "F" }
        if modelID.hasPrefix("z-image") { return "Z" }
        if modelID.hasPrefix("ltx") { return "L" }
        if modelID.hasPrefix("minimax") { return "M" }
        if modelID.hasPrefix("wan") { return "W" }
        if modelID.hasPrefix("llada") { return "L" }
        return "↑"
    }
    static func publisher(_ id: String) -> String {
        if id.hasPrefix("qwen") { return "Qwen · Alibaba" }
        if id.hasPrefix("flux") { return "Black Forest Labs" }
        if id.hasPrefix("z-image") { return "Tongyi-MAI" }
        if id.hasPrefix("ltx") { return "Lightricks" }
        if id.hasPrefix("minimax") { return "MiniMax" }
        if id.hasPrefix("wan") { return "Wan · Alibaba" }
        if id.hasPrefix("llada") { return "inclusionAI" }
        return id.hasPrefix("real-esrgan") ? "Real-ESRGAN" : "模型"
    }
    var body: some View {
        Group {
            if let logo {
                Image(nsImage: logo).resizable().interpolation(.high).scaledToFit()
                    .padding(size * 0.10).background(Color.white)
            } else {
                Text(glyph).font(.system(size: size * 0.5, weight: .heavy, design: .rounded)).foregroundStyle(tint)
            }
        }
            .frame(width: size, height: size)
            .background(tint.opacity(0.08).gradient, in: RoundedRectangle(cornerRadius: size * 0.26))
            .clipShape(RoundedRectangle(cornerRadius: size * 0.26))
            .overlay(RoundedRectangle(cornerRadius: size * 0.26).stroke(tint.opacity(0.14)))
            .accessibilityHidden(true)
    }
}
