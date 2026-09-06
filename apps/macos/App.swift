import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers

@main
struct TurboCiderNativeApp: App {
    @StateObject private var store: NativeJobStore
    @StateObject private var studio: StudioState
    @NSApplicationDelegateAdaptor(StudioAppDelegate.self) private var delegate
    init() {
        let directory = ProcessInfo.processInfo.environment["TURBOCIDER_NATIVE_STATE"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCiderNative")
        _store = StateObject(wrappedValue: NativeJobStore(directory: directory))
        _studio = StateObject(wrappedValue: StudioState(directory: directory))
    }
    var body: some Scene {
        WindowGroup("TurboCider") {
            StudioView(store: store, studio: studio).frame(minWidth: 980, minHeight: 700)
                .onAppear { delegate.store = store; delegate.studio = studio }
        }
        .commands {
            CommandGroup(replacing: .newItem) { Button("新建创作") { studio.newDraft() }.keyboardShortcut("n") }
            CommandGroup(after: .pasteboard) {
                Button("粘贴图片到创作") { Task { await studio.pasteImage() } }.keyboardShortcut("v", modifiers: [.command, .shift])
            }
        }
        Settings { VStack(alignment: .leading, spacing: 14) {
            Text("本地运行").font(.title2)
            Text("当前 App 使用嵌入式模型会话。退出时生成会停止；所有素材与结果保存在本机。")
            Button("打开数据文件夹") { NSWorkspace.shared.open(store.directory) }
            Text("模型权重与原始图片不会随“释放内存”删除。").foregroundStyle(.secondary)
        }.padding(24).frame(width: 440) }
    }
}
@MainActor final class StudioAppDelegate: NSObject, NSApplicationDelegate {
    weak var store: NativeJobStore?
    weak var studio: StudioState?
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        studio?.save()
        guard store?.busy == true else { return .terminateNow }
        let alert = NSAlert(); alert.messageText = "任务仍在进行中"; alert.informativeText = "退出会中断本机任务。草稿与历史记录会保留。"
        alert.addButton(withTitle: "继续运行"); alert.addButton(withTitle: "退出并中断")
        return alert.runModal() == .alertFirstButtonReturn ? .terminateCancel : .terminateNow
    }
}
private enum StudioPage: String, CaseIterable, Identifiable {
    case studio = "创作", library = "素材库", tasks = "任务", models = "模型"
    var id: String { rawValue }
    var symbol: String { switch self { case .studio: return "sparkles"; case .library: return "photo.on.rectangle"; case .tasks: return "clock"; case .models: return "cpu" } }
}
private let ciderAccent = Color(red: 0.96, green: 0.70, blue: 0.37)
private func operationName(_ value: String) -> String {
    ["image.generate": "文生图", "image.transform": "单图修改", "image.edit": "参考编辑"][value] ?? value
}
private func stateName(_ value: String) -> String {
    ["preparing": "准备中", "running": "生成中", "cancelling": "正在安全停止", "succeeded": "已完成", "failed": "失败", "cancelled": "已取消", "interrupted": "已中断"][value] ?? value
}
private func phaseName(_ value: String) -> String {
    if value == "denoise" { return "采样" }
    if value.contains("vae") || value.contains("decode") { return "图像编解码" }
    if value.contains("load") { return "加载权重" }
    if value.contains("text") || value.contains("qwen") { return "文本编码" }
    return ["prepare": "准备", "export": "保存结果", "complete": "完成", "image_encode": "编码参考图"][value] ?? value
}

struct StudioView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    @State private var page: StudioPage? = .studio
    @State private var selected: UUID?
    @State private var inspector = true
    @State private var dropping = false
    @State private var compareOriginal = false
    @State private var submitting = false
    private var selectedJob: NativeJob? { store.jobs.first { $0.id == selected } ?? store.jobs.first { $0.state == "succeeded" } }
    private var model: StudioModel? { studio.models.first { $0.id == studio.draft.modelID } }
    private var inputSummary: String { "\(studio.draft.activeAssets.count) 张输入 → 1 张图像" }
    var body: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 18) {
                HStack { Image(systemName: "sparkles").foregroundStyle(ciderAccent); Text("TurboCider").font(.title3.weight(.semibold)) }.padding(.horizontal, 12).padding(.top, 16)
                List(StudioPage.allCases, selection: $page) { item in Label(item.rawValue, systemImage: item.symbol).tag(item) }.listStyle(.sidebar)
                VStack(alignment: .leading, spacing: 5) {
                    Label("本机运行", systemImage: "circle.fill").foregroundStyle(.green).font(.caption)
                    Text(store.sessionState).font(.caption).foregroundStyle(.secondary)
                }.padding(12)
                SettingsLink { Label("设置", systemImage: "gearshape") }.buttonStyle(.plain).padding(12)
            }.navigationSplitViewColumnWidth(min: 170, ideal: 185, max: 230)
        } detail: {
            Group {
                switch page ?? .studio {
                case .studio: workspace
                case .models: modelsPage
                case .tasks: tasksPage
                case .library: libraryPage
                }
            }.background(Color(nsColor: .windowBackgroundColor))
        }
        .tint(ciderAccent)
        .toolbar {
            ToolbarItem(placement: .automatic) { Text("本地创作").foregroundStyle(.secondary) }
            ToolbarItem(placement: .automatic) { Text(studio.saved ? "草稿已保存" : "草稿尚未保存").font(.caption).foregroundStyle(.secondary) }
            ToolbarItem { Button { studio.newDraft(); selected = nil; page = .studio } label: { Label("新建创作", systemImage: "square.and.pencil") } }
            ToolbarItem { Button { inspector.toggle() } label: { Label("显示参数", systemImage: "sidebar.right") } }
        }
        .onDisappear { studio.save() }
    }
    private var workspace: some View {
        HStack(spacing: 0) {
            VStack(spacing: 12) {
                HStack {
                    Picker("创作方式", selection: Binding(get: { studio.draft.operation }, set: { studio.changeOperation($0); compareOriginal = false })) {
                        ForEach(model?.operations ?? [], id: \.self) { Text(operationName($0)).tag($0) }
                    }.pickerStyle(.segmented).frame(maxWidth: 380).accessibilityIdentifier("operation")
                    Spacer()
                    Text("STUDIO").font(.caption2).tracking(2).foregroundStyle(.secondary)
                }
                mediaStage.frame(maxWidth: .infinity, maxHeight: .infinity)
                if !store.jobs.filter({ $0.state == "succeeded" }).isEmpty { resultStrip }
                composer
                runStatus
            }.padding(18).frame(minWidth: 480)
            if inspector {
                Divider()
                ScrollView { inspectorContents.padding(18) }.frame(width: 270).background(Color(nsColor: .controlBackgroundColor))
            }
        }
    }
    private var mediaStage: some View {
        VStack(spacing: 10) {
            if let job = selectedJob, job.state == "succeeded" {
                HStack {
                    Text(store.activeJob == nil ? "生成结果" : "上一结果 · 新任务进行中").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if job.request.inputs?.first != nil { Toggle("查看原图", isOn: $compareOriginal).toggleStyle(.button).font(.caption) }
                }
                MediaPreview(path: compareOriginal ? (job.request.inputs?.first?.path ?? job.request.output) : job.request.output)
                    .frame(maxWidth: .infinity, maxHeight: .infinity).accessibilityIdentifier("generatedImage")
                HStack(spacing: 12) {
                    Text("\(job.request.width) × \(job.request.height) · 种子 \(job.request.seed)").font(.caption).monospacedDigit().foregroundStyle(.secondary)
                    Spacer()
                    Button("编辑此图") { Task {
                        let previous = Set(studio.draft.assets.map(\.id))
                        await studio.addFiles([URL(fileURLWithPath: job.request.output)])
                        if let added = studio.draft.assets.first(where: { !previous.contains($0.id) }) {
                            studio.changeOperation("image.transform"); studio.draft.initImageID = added.id
                        }
                    } }.disabled(studio.importing)
                    Menu {
                        Button("另存为…") { exportResult(job.request.output) }
                        Button("在 Finder 中显示") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: job.request.output)]) }
                        Button("复用参数") { studio.reuse(job) }
                    } label: { Image(systemName: "ellipsis.circle") }.menuStyle(.borderlessButton).frame(width: 26)
                }
            } else {
                Image(systemName: "photo.on.rectangle.angled").font(.system(size: 42, weight: .ultraLight)).foregroundStyle(.secondary)
                Text("把想法变成画面").font(.title2.weight(.medium))
                Text("输入提示词，或添加图片开始修改\n所有生成都在这台 Mac 上完成").font(.callout).multilineTextAlignment(.center).foregroundStyle(.secondary)
            }
        }.padding(12).background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 12))
    }
    private var resultStrip: some View {
        ScrollView(.horizontal) { LazyHStack(spacing: 8) {
            ForEach(store.jobs.filter { $0.state == "succeeded" }) { job in
                Button { selected = job.id; compareOriginal = false } label: {
                    MediaPreview(path: job.request.output, maxPixel: 140).frame(width: 60, height: 48)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(selectedJob?.id == job.id ? ciderAccent : .clear, lineWidth: 2))
                }.buttonStyle(.plain).help("种子 \(job.request.seed) · \(operationName(job.request.operation ?? "image.generate"))")
            }
        }.padding(2) }.frame(height: 52)
    }
    private var composer: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button(action: chooseImages) { Label("添加图片", systemImage: "plus") }.accessibilityIdentifier("addImages")
                Button { Task { await studio.pasteImage() } } label: { Label("粘贴图片", systemImage: "doc.on.clipboard") }.accessibilityIdentifier("pasteImages")
                if studio.canUndoAssets { Button { studio.undoAssetChange() } label: { Image(systemName: "arrow.uturn.backward") }.help("撤销素材修改") }
                Spacer()
                if studio.importing { ProgressView().controlSize(.small) }
                Text("\(studio.draft.assets.count) / 8").font(.caption).foregroundStyle(.secondary)
            }.disabled(studio.importing)
            if !studio.draft.assets.isEmpty { inputStrip }
            if studio.draft.assets.count > studio.draft.activeAssets.count {
                Text("\(studio.draft.assets.count - studio.draft.activeAssets.count) 张图片已保留，未参与当前模式。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            PromptEditor(text: $studio.draft.prompt) { Task { await studio.pasteImage() } }
                .frame(height: 72).accessibilityIdentifier("prompt")
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(inputSummary).font(.caption)
                    Text(studio.draft.randomSeed ? "每次随机 · 本次 \(studio.lastSeed.map(String.init) ?? "待确定")" : "固定种子 \(studio.draft.seedText)").font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                Button(action: generate) { Label(store.busy ? "正在运行" : "生成图像", systemImage: "sparkles").padding(.horizontal, 8).padding(.vertical, 4) }
                    .buttonStyle(.borderedProminent).foregroundStyle(Color(red: 0.13, green: 0.09, blue: 0.04))
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(store.busy || submitting || studio.importing || model?.executor != true)
                    .accessibilityIdentifier("generate")
            }
        }.padding(14).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(dropping ? ciderAccent : Color.primary.opacity(0.12), lineWidth: 1))
            .onDrop(of: [.fileURL, .image], isTargeted: $dropping) { providers in Task { await studio.importProviders(providers) }; return true }
    }
    private var inputStrip: some View {
        ScrollView(.horizontal) { HStack(alignment: .top, spacing: 10) {
            ForEach(Array(studio.draft.assets.enumerated()), id: \.element.id) { index, asset in
                VStack(alignment: .leading, spacing: 4) {
                    Button { if studio.draft.operation == "image.transform" { studio.draft.initImageID = asset.id } } label: {
                        MediaPreview(path: asset.path, maxPixel: 160).frame(width: 78, height: 54).clipped()
                            .overlay(RoundedRectangle(cornerRadius: 5).stroke(studio.draft.activeAssets.contains(asset) ? ciderAccent : .clear, lineWidth: 2))
                    }.buttonStyle(.plain).accessibilityLabel("参考图 \(index + 1)，\(asset.name)，点击选为原图")
                    HStack(spacing: 4) {
                        Text(studio.draft.operation == "image.transform" && studio.draft.initImageID == asset.id ? "原图" : "参考 \(index + 1)").font(.caption2)
                        Menu {
                            Button("设为原图") { studio.changeOperation("image.transform"); studio.draft.initImageID = asset.id }
                            Button("向前移动") { studio.move(asset.id, offset: -1) }.disabled(index == 0)
                            Button("向后移动") { studio.move(asset.id, offset: 1) }.disabled(index == studio.draft.assets.count - 1)
                            Button("移除", role: .destructive) { studio.remove(asset.id) }
                        } label: { Image(systemName: "ellipsis") }.menuStyle(.borderlessButton).frame(width: 22)
                    }
                }.help("\(asset.name) · \(asset.width) × \(asset.height)")
            }
        }.padding(2) }.frame(height: 82)
    }
    private var inspectorContents: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("生成参数").font(.headline)
            VStack(alignment: .leading, spacing: 8) {
                Text(model?.name ?? "FLUX.2 Klein 4B").font(.subheadline.weight(.medium))
                Label(studio.draft.acceleration?.policy == "gpu_ane" ? "GPU + ANE · INT8 混合" : (studio.draft.acceleration?.policy ?? "auto") == "auto" ? "本机自动适配 · GPU / ANE" : "本地 BF16 · Metal GPU", systemImage: "checkmark.circle").font(.caption).foregroundStyle(.secondary)
                Button(studio.draft.modelPath.isEmpty ? "选择模型…" : "管理模型") { page = .models }
            }
            Divider()
            Text("输出尺寸").font(.subheadline)
            HStack { TextField("宽", value: $studio.draft.width, format: .number).accessibilityIdentifier("width"); Text("×"); TextField("高", value: $studio.draft.height, format: .number).accessibilityIdentifier("height") }.textFieldStyle(.roundedBorder)
            HStack { ForEach([256, 512, 768], id: \.self) { size in Button("\(size)") { studio.draft.width = size; studio.draft.height = size }.font(.caption) } }
            if studio.draft.operation == "image.transform" {
                Divider()
                HStack { Text("原图保留强度"); Spacer(); Text(studio.draft.strength, format: .number.precision(.fractionLength(2))).monospacedDigit() }.font(.caption)
                Slider(value: $studio.draft.strength, in: 0...1, step: 0.05).accessibilityLabel("原图保留强度")
                Text("值越大，保留原图越多；1 仅进行 VAE 重建，0 使用完整采样。实际采样步数随强度变化。").font(.caption2).foregroundStyle(.secondary)
            }
            Divider()
            Text("种子").font(.subheadline)
            Toggle("每次生成随机", isOn: $studio.draft.randomSeed).toggleStyle(.switch).controlSize(.small).accessibilityIdentifier("randomSeed")
            HStack {
                TextField("42", text: $studio.draft.seedText).textFieldStyle(.roundedBorder).disabled(studio.draft.randomSeed).accessibilityIdentifier("seed")
                Button { studio.draft.seedText = String(Int.random(in: 0...2147483647)); studio.draft.randomSeed = false } label: { Image(systemName: "dice") }.help("随机一次并固定种子")
            }
            DisclosureGroup("高级参数") {
                VStack(alignment: .leading, spacing: 12) {
                    TextField("采样步数", value: $studio.draft.steps, format: .number).textFieldStyle(.roundedBorder).accessibilityIdentifier("steps")
                    Text("1–50 步，推荐 4 步").font(.caption2).foregroundStyle(.secondary)
                    Toggle("动态文本长度", isOn: $studio.draft.dynamicText).controlSize(.small)
                    Picker("模型驻留", selection: $studio.draft.residency) { Text("保留图像权重").tag("resident"); Text("分阶段释放").tag("component_staged") }
                    Button(studio.draft.profilePath.isEmpty ? "选择加速配置…" : "更换加速配置…", action: chooseProfile)
                    if !studio.draft.profilePath.isEmpty {
                        Text(URL(fileURLWithPath: studio.draft.profilePath).lastPathComponent).font(.caption2)
                        Button("恢复默认 GPU") { studio.draft.profilePath = ""; if studio.draft.acceleration != nil { studio.draft.acceleration?.policy = "gpu" } }
                    }
                    Text("加速配置由引擎校验。本次请求的真实执行计划可在任务详情中查看。").font(.caption2).foregroundStyle(.secondary)
                }.padding(.top, 12)
            }
            Spacer(minLength: 8)
            Text("一次生成一个结果\n图片、提示词与参数均保存在本机。").font(.caption2).foregroundStyle(.secondary)
        }
    }
    private var runStatus: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let job = store.activeJob {
                ProgressView(value: Double(job.completed), total: Double(max(1, job.total))).tint(ciderAccent)
                HStack {
                    Text("\(stateName(job.state)) · \(phaseName(job.phase)) \(job.completed)/\(job.total)")
                    Spacer()
                    if job.phase == "denoise", let speed = job.secondsPerStep { Text(String(format: "%.2f 秒/步", speed)).monospacedDigit() }
                    Text(String(format: "%.1f 秒", job.elapsed)).monospacedDigit()
                    Button("取消") { store.cancel() }.disabled(job.state == "cancelling")
                }.font(.caption)
                if job.phase == "denoise", let speed = job.secondsPerStep, job.completed < job.total {
                    Text("采样阶段预计还需约 \(Int(ceil(speed * Double(job.total - job.completed)))) 秒，图像解码另计。")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            } else if store.busy { HStack { ProgressView().controlSize(.small); Text(store.sessionState).font(.caption); Spacer(); Button("取消") { store.cancel() } } }
            else if let job = store.jobs.first {
                Text("\(stateName(job.state)) · 共用时 \(String(format: "%.1f", job.elapsed)) 秒 · 种子 \(job.request.seed)").font(.caption).foregroundStyle(.secondary)
            }
            if let error = studio.message ?? store.storageError {
                HStack(alignment: .top) { Text(error).font(.caption).textSelection(.enabled); Spacer(); Button { studio.message = nil } label: { Image(systemName: "xmark") } }
                    .foregroundStyle(.secondary).accessibilityIdentifier("statusMessage")
            }
        }
    }
    private var modelsPage: some View {
        ScrollView { VStack(alignment: .leading, spacing: 22) {
            Text("模型中心").font(.largeTitle.weight(.medium))
            Text("选择本地模型，按需加载，空闲时释放内存。文件与内存分开管理。").foregroundStyle(.secondary)
            ForEach(studio.models) { item in
                VStack(alignment: .leading, spacing: 14) {
                    HStack { Label(item.name, systemImage: "cpu").font(.title3); Spacer(); Text(item.executor ? "已接入" : "暂不可用").font(.caption).foregroundStyle(.secondary) }
                    if item.executor {
                        Text(studio.draft.modelPaths[item.id] ?? "尚未选择模型文件夹").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        HStack {
                            Button("选择模型文件夹…") { chooseModel(item.id) }.disabled(store.busy).accessibilityIdentifier("chooseModel")
                            Button("Load · 加载") { loadModel(item.id) }.disabled(store.busy || (studio.draft.modelPaths[item.id] ?? "").isEmpty).accessibilityIdentifier("loadModel")
                            Button("Unload · 释放内存") { Task { do { try await store.unload() } catch { studio.message = error.localizedDescription } } }.disabled(!store.canUnload).accessibilityIdentifier("unloadModel")
                        }
                        Text(store.loadedModelID == item.id ? store.sessionState : "未加载").font(.callout)
                        Text("加载当前提示词、图像权重与所选加速分区；预热还会执行一次完整计算。模型路径失效会报错，不会自动下载。").font(.caption).foregroundStyle(.secondary)
                    } else { Text("视频执行器尚未通过正式验收，当前版本优先支持 FLUX 图像生成与编辑。").font(.callout).foregroundStyle(.secondary) }
                }.padding(20).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            }
            AccelerationView(store: store, studio: studio)
            if let report = store.resourceReport { DisclosureGroup("最近资源报告") { Text(report).font(.system(.caption, design: .monospaced)).textSelection(.enabled) } }
            runStatus
        }.padding(28) }
    }
    private var tasksPage: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("任务与历史").font(.largeTitle.weight(.medium))
            Text("每次一个结果；运行中可编辑下一份草稿。").foregroundStyle(.secondary)
            List(store.jobs) { job in
                DisclosureGroup {
                    Text(job.request.prompt).textSelection(.enabled)
                    if let error = job.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
                    HStack {
                        Button("复用参数") { studio.reuse(job); page = .studio }
                        if job.state == "succeeded" { Button("查看结果") { selected = job.id; page = .studio } }
                    }
                    if let json = job.resultJSON { DisclosureGroup("执行与性能详情") { Text(json).font(.system(.caption, design: .monospaced)).textSelection(.enabled) } }
                } label: {
                    HStack { VStack(alignment: .leading) { Text(job.request.prompt).lineLimit(1); Text("\(operationName(job.request.operation ?? "image.generate")) · \(job.request.width)×\(job.request.height) · 种子 \(job.request.seed)").font(.caption).foregroundStyle(.secondary) }; Spacer(); Text(stateName(job.state)).font(.caption) }
                }.padding(.vertical, 6)
            }.listStyle(.inset)
            runStatus
        }.padding(28)
    }
    private var libraryPage: some View {
        ScrollView { VStack(alignment: .leading, spacing: 18) {
            Text("素材库").font(.largeTitle.weight(.medium))
            Text("每一张结果，都可以成为下一次创作的输入。").foregroundStyle(.secondary)
            if store.jobs.filter({ $0.state == "succeeded" }).isEmpty { ContentUnavailableView("还没有生成结果", systemImage: "photo", description: Text("在创作中生成第一张图像。")) }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 180))], spacing: 18) {
                ForEach(store.jobs.filter { $0.state == "succeeded" }) { job in
                    VStack(alignment: .leading, spacing: 8) {
                        Button { selected = job.id; compareOriginal = false; page = .studio } label: { MediaPreview(path: job.request.output, maxPixel: 500).frame(height: 180) }.buttonStyle(.plain)
                        Text(job.request.prompt).font(.caption).lineLimit(2)
                        Text("种子 \(job.request.seed)").font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
        }.padding(28) }
    }
    private func chooseModel(_ id: String) {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = "选择包含 transformer、text_encoder、vae 和 tokenizer 的 FLUX 模型目录。"
        if panel.runModal() == .OK, let url = panel.url { studio.draft.modelPaths[id] = url.path; studio.save() }
    }
    private func chooseProfile() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        if panel.runModal() == .OK, let url = panel.url { studio.draft.profilePath = url.path; var acceleration = studio.draft.acceleration ?? StudioAcceleration(); acceleration.policy = "profile"; studio.draft.acceleration = acceleration }
    }
    private func chooseImages() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { let urls = panel.urls; Task { await studio.addFiles(urls) } }
    }
    private func loadModel(_ id: String) {
        guard let path = studio.draft.modelPaths[id], !path.isEmpty else { return }
        studio.message = nil
        Task { do { let request = try studio.draft.request(output: store.directory.appendingPathComponent("unused-prepare.png")); try await store.prepare(modelURL: URL(fileURLWithPath: path), request: request, warmup: false) } catch { studio.message = error is CancellationError ? "加载已取消" : error.localizedDescription } }
    }
    private func generate() {
        guard !store.busy, !submitting, !studio.importing else { return }
        do {
            let request = try studio.draft.request(output: store.directory.appendingPathComponent("outputs/\(UUID().uuidString).png"))
            let modelURL = URL(fileURLWithPath: studio.draft.modelPath)
            studio.lastSeed = request.seed; studio.message = nil; studio.save(); submitting = true
            Task {
                defer { submitting = false }
                do { let job = try await store.generate(modelURL: modelURL, request: request); selected = job.id; compareOriginal = false }
                catch { studio.message = error is CancellationError ? "生成已取消，草稿与原图已保留。" : error.localizedDescription }
            }
        } catch { studio.message = error.localizedDescription }
    }
    private func exportResult(_ path: String) {
        let panel = NSSavePanel(); panel.allowedContentTypes = [.png]; panel.nameFieldStringValue = URL(fileURLWithPath: path).lastPathComponent
        if panel.runModal() == .OK, let destination = panel.url {
            Task {
                do { try await Task.detached { try Data(contentsOf: URL(fileURLWithPath: path)).write(to: destination, options: .atomic) }.value }
                catch { studio.message = error.localizedDescription }
            }
        }
    }
}
