import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers

@main
struct TurboCiderNativeApp: App {
    @StateObject private var store: NativeJobStore
    @StateObject private var studio: StudioState
    @StateObject private var api: LocalAPIController
    @NSApplicationDelegateAdaptor(StudioAppDelegate.self) private var delegate
    init() {
        let directory = ProcessInfo.processInfo.environment["TURBOCIDER_NATIVE_STATE"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCiderNative")
        if ProcessInfo.processInfo.environment["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] == nil {
            setenv("TURBOCIDER_LTX_CONDITIONING_CACHE_DIR", TensorCache.sharedRoot.path, 0)
        }
        _store = StateObject(wrappedValue: NativeJobStore(directory: directory))
        _studio = StateObject(wrappedValue: StudioState(directory: directory))
        _api = StateObject(wrappedValue: LocalAPIController(directory: directory))
    }
    var body: some Scene {
        WindowGroup("TurboCider") {
            StudioView(store: store, studio: studio, api: api).frame(minWidth: 980, minHeight: 700)
                .onAppear { delegate.store = store; delegate.studio = studio; delegate.api = api }
        }
        .commands {
            CommandGroup(replacing: .newItem) { Button("新建创作") { studio.newDraft() }.keyboardShortcut("n") }
            CommandGroup(after: .pasteboard) {
                Button("粘贴图片到创作") { Task { await studio.pasteImage() } }.keyboardShortcut("v", modifiers: [.command, .shift])
            }
        }
        Settings { ScrollView { VStack(alignment: .leading, spacing: 14) {
            Text("本地运行").font(.title2)
            Text("当前 App 使用嵌入式模型会话。退出时生成会停止；所有素材与结果保存在本机。")
            Button("打开数据文件夹") { NSWorkspace.shared.open(store.directory) }
            Text("模型权重与原始图片不会随“释放内存”删除。").foregroundStyle(.secondary)
            Divider()
            TensorCacheView(store: store)
        }.padding(24) }.frame(width: 540, height: 650) }
    }
}
@MainActor final class StudioAppDelegate: NSObject, NSApplicationDelegate {
    weak var store: NativeJobStore?
    weak var studio: StudioState?
    weak var api: LocalAPIController?
    func applicationWillTerminate(_ notification: Notification) { api?.terminateNow() }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        studio?.save()
        guard store?.busy == true || api?.running == true else { return .terminateNow }
        let alert = NSAlert(); alert.messageText = "任务仍在进行中"; alert.informativeText = "退出会中断本机任务。草稿与历史记录会保留。"
        alert.addButton(withTitle: "继续运行"); alert.addButton(withTitle: "退出并中断")
        return alert.runModal() == .alertFirstButtonReturn ? .terminateCancel : .terminateNow
    }
}
private enum StudioPage: String, CaseIterable, Identifiable {
    case studio = "创作", library = "素材库", tasks = "任务", models = "模型", api = "本地 API"
    var id: String { rawValue }
    var symbol: String { switch self { case .studio: return "sparkles"; case .library: return "photo.on.rectangle"; case .tasks: return "clock"; case .models: return "cpu"; case .api: return "network" } }
}
private let ciderAccent = Color(red: 0.02, green: 0.70, blue: 0.64)
private func operationName(_ value: String) -> String {
    ["image.generate": "文生图", "image.transform": "单图修改", "image.edit": "参考编辑",
     "video.generate": "文生视频", "video.image": "图生视频", "video.keyframes": "关键帧视频",
     "video.reference": "参考视频"][value] ?? value
}
private func stateName(_ value: String) -> String {
    ["preparing": "准备中", "running": "生成中", "finalizing": "正在保存结果", "cancelling": "正在安全停止", "succeeded": "已完成", "failed": "失败", "cancelled": "已取消", "interrupted": "已中断"][value] ?? value
}
private func phaseName(_ value: String) -> String {
    if value == "pack_z_image_suffix" { return "整理 GPU 权重" }
    if value == "denoise" { return "采样" }
    if value.contains("text_cache_hit") { return "复用文本编码" }
    if value.contains("vae") || value.contains("decode") { return "图像编解码" }
    if value.contains("load") { return "加载权重" }
    if value.contains("text") || value.contains("qwen") { return "文本编码" }
    return ["prepare": "准备", "export": "保存结果", "complete": "完成", "image_encode": "编码参考图"][value] ?? value
}
private struct StudioOutputPreview: View {
    let path: String
    var maxPixel = 1600
    private var video: Bool { ["mp4", "mov", "m4v"].contains(URL(fileURLWithPath: path).pathExtension.lowercased()) }
    var body: some View {
        if video { SafeVideoPreview(path: path) }
        else { MediaPreview(path: path, maxPixel: maxPixel) }
    }
}

private struct StudioResultThumbnail: View {
    let path: String
    private var video: Bool { ["mp4", "mov", "m4v"].contains(URL(fileURLWithPath: path).pathExtension.lowercased()) }
    var body: some View {
        if video { Image(systemName: "film").font(.title2).frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.primary.opacity(0.06)) }
        else { MediaPreview(path: path, maxPixel: 140) }
    }
}

struct StudioView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    @ObservedObject var api: LocalAPIController
    @StateObject private var library = ModelLibraryController()
    @StateObject private var tensorCache = TensorCacheController()
    @State private var page: StudioPage? = .studio
    @State private var selected: UUID?
    @State private var selectedTasks: Set<UUID> = []
    @State private var resultSelection = HistorySelection()
    @State private var inspector = true
    @State private var dropping = false
    @State private var compareOriginal = false
    @State private var submitting = false
    private var selectedJob: NativeJob? { store.jobs.first { $0.id == selected && $0.hasOutput } ?? store.jobs.first { $0.hasOutput } }
    private var outputJobs: [NativeJob] { store.jobs.filter { $0.hasOutput } }
    private var outputIDs: [UUID] { outputJobs.map(\.id) }
    private var model: StudioModel? { studio.models.first { $0.id == studio.draft.modelID } }
    private var loraStrategyHint: String {
        let selected = studio.draft.loraStrategy == "auto"
            ? model?.default_lora_strategy : studio.draft.loraStrategy
        switch selected {
        case "disk_premerge":
            return "使用离线准备并通过来源校验的预融合模型；App 不执行磁盘融合。"
        case "in_memory_merge":
            return "加载时把 delta 融合到内存，不写完整 merged checkpoint。"
        case "inference_time":
            return "推理请求期间加载独立 LoRA；不会长期保存融合后的模型权重。"
        default:
            return "自动选择当前模型已声明的默认 LoRA 策略。"
        }
    }
    private var inputSummary: String {
        let output = model?.isVideo == true ? "视频" : "图像"
        return "\(studio.draft.activeAssets.count) 张输入 → 1 个\(output)"
    }
    var body: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    if let mark = NSImage(named: "LogoMark") { Image(nsImage: mark).resizable().scaledToFit().frame(width: 32, height: 32) }
                    else { Image(systemName: "sparkles").foregroundStyle(ciderAccent) }
                    Text("TurboCider").font(.title3.weight(.semibold))
                }.padding(.horizontal, 12).padding(.top, 16)
                List(StudioPage.allCases, selection: $page) { item in Label(item.rawValue, systemImage: item.symbol).tag(item) }.listStyle(.sidebar)
                VStack(alignment: .leading, spacing: 5) {
                    Label("本机运行", systemImage: "circle.fill").foregroundStyle(.green).font(.caption)
                    Text(api.running ? "API 服务运行中" : store.sessionState).font(.caption).foregroundStyle(.secondary)
                }.padding(12)
                ResourceMonitorView().padding(.horizontal, 12)
                SettingsLink { Label("设置", systemImage: "gearshape") }.buttonStyle(.plain).padding(12)
            }.navigationSplitViewColumnWidth(min: 170, ideal: 185, max: 230)
        } detail: {
            Group {
                switch page ?? .studio {
                case .studio: workspace
                case .models: modelsPage
                case .tasks: tasksPage
                case .library: libraryPage
                case .api: LocalAPIView(api: api, store: store)
                }
            }.background(Color(nsColor: .windowBackgroundColor))
        }
        .tint(ciderAccent)
        .toolbar {
            ToolbarItem(placement: .automatic) { Text("本地创作").foregroundStyle(.secondary) }
            ToolbarItem(placement: .automatic) { Text(studio.saved ? "草稿已保存" : "草稿尚未保存").font(.caption).foregroundStyle(.secondary) }
            ToolbarItem { Button { studio.newDraft(); selected = nil; resultSelection.clear(); page = .studio } label: { Label("新建创作", systemImage: "square.and.pencil") } }
            ToolbarItem { Button { inspector.toggle() } label: { Label("显示参数", systemImage: "sidebar.right") } }
        }
        .onDisappear { studio.save() }
        .onChange(of: store.deletableJobIDs) { _, ids in selectedTasks.formIntersection(ids) }
        .onChange(of: outputIDs) { _, ids in resultSelection.retain(Set(ids)) }
        .task { library.refresh(studio: studio, migrate: true) }
        .task(id: studio.streamingQueryKey) {
            await studio.refreshStreamingOptions()
        }
        .task {
            while !Task.isCancelled {
                await tensorCache.automaticSweep(store: store)
                do { try await Task.sleep(for: .seconds(3600)) } catch { break }
            }
        }
    }
    private var workspace: some View {
        HStack(spacing: 0) {
            VStack(spacing: 12) {
                if api.running { HStack { Label("本地 API 正在接收任务", systemImage: "network"); Spacer(); Button("管理服务") { page = .api } }.font(.callout).padding(10).background(ciderAccent.opacity(0.1), in: RoundedRectangle(cornerRadius: 8)) }
                HStack {
                    Picker("创作类型", selection: Binding(get: { studio.creationKind }, set: { studio.changeCreationKind($0); compareOriginal = false })) {
                        Label("图片生成", systemImage: "photo").tag("image")
                        Label("视频生成", systemImage: "video").tag("video")
                    }.pickerStyle(.segmented).frame(maxWidth: 300)
                        .disabled(store.busy || studio.importing).accessibilityIdentifier("creationKind")
                    Spacer()
                    Button { page = .models } label: { Label("模型中心", systemImage: "square.stack.3d.up") }
                }
                HStack {
                    Picker("创作方式", selection: Binding(get: { studio.draft.operation }, set: { studio.changeOperation($0); compareOriginal = false })) {
                        ForEach(studio.creationOperations, id: \.self) { Text(operationName($0)).tag($0) }
                    }.pickerStyle(.segmented).frame(maxWidth: 440).disabled(store.busy || studio.importing).accessibilityIdentifier("operation")
                    Spacer()
                    Text("STUDIO").font(.caption2).tracking(2).foregroundStyle(.secondary)
                }
                mediaStage.frame(maxWidth: .infinity, maxHeight: .infinity)
                if !store.jobs.filter({ $0.hasOutput }).isEmpty { resultStrip }
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
            if let job = selectedJob, job.hasOutput {
                HStack {
                    Text(store.activeJob == nil ? "生成结果" : "上一结果 · 新任务进行中").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if job.request.inputs?.first != nil { Toggle("查看原图", isOn: $compareOriginal).toggleStyle(.button).font(.caption) }
                }
                StudioOutputPreview(path: compareOriginal ? (job.request.inputs?.first?.path ?? job.request.output) : job.request.output)
                    .frame(maxWidth: .infinity, maxHeight: .infinity).accessibilityIdentifier("generatedImage")
                if let route = job.routeSummary { Text("实际路径：\(route)").font(.caption2).foregroundStyle(.secondary) }
                HStack(spacing: 12) {
                    Text("\(job.request.width) × \(job.request.height) · 种子 \(job.request.seed)").font(.caption).monospacedDigit().foregroundStyle(.secondary)
                    Spacer()
                    if URL(fileURLWithPath: job.request.output).pathExtension.lowercased() == "png" { Button("编辑此图") { Task {
                        let previous = Set(studio.draft.assets.map(\.id))
                        await studio.addFiles([URL(fileURLWithPath: job.request.output)])
                        if let added = studio.draft.assets.first(where: { !previous.contains($0.id) }) {
                            studio.changeOperation("image.transform"); studio.draft.initImageID = added.id
                        }
                    } }.disabled(studio.importing) }
                    Menu {
                        Button("另存为…") { exportResult(job.request.output) }
                        Button("在 Finder 中显示") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: job.request.output)]) }
                        Button("复用参数") { studio.reuse(job) }
                        Button(resultDeletionTitle(for: job), role: .destructive) { trashResults(resultDeletionIDs(for: job)) }.disabled(store.busy)
                    } label: { Image(systemName: "ellipsis.circle") }.menuStyle(.borderlessButton).frame(width: 26)
                }
            } else {
                Image(systemName: "photo.on.rectangle.angled").font(.system(size: 42, weight: .ultraLight)).foregroundStyle(.secondary)
                Text("把想法变成画面").font(.title2.weight(.medium))
                Text(studio.supportsImageInputs ? "输入提示词，或添加图片开始创作\n所有生成都在这台 Mac 上完成" : "输入提示词开始创作\n所有生成都在这台 Mac 上完成").font(.callout).multilineTextAlignment(.center).foregroundStyle(.secondary)
            }
        }.padding(12).background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 12))
    }
    private var resultStrip: some View {
        VStack(alignment: .leading, spacing: 6) {
            if resultSelection.ids.count > 1 { resultSelectionControls }
            ScrollView(.horizontal) { LazyHStack(spacing: 8) {
                ForEach(outputJobs) { job in
                    resultThumbnail(job, height: 48).frame(width: 60)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(resultSelection.ids.contains(job.id) || (resultSelection.ids.isEmpty && selectedJob?.id == job.id) ? ciderAccent : .clear, lineWidth: 2))
                        .contextMenu {
                            Button(resultDeletionTitle(for: job), role: .destructive) { trashResults(resultDeletionIDs(for: job)) }.disabled(store.busy)
                        }.help("种子 \(job.request.seed) · Ctrl/⌘ 点击多选，Shift 点击连续选择")
                }
            }.padding(2) }.frame(height: 52)
        }
    }
    private var composer: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button(action: chooseImages) { Label("添加图片", systemImage: "plus") }.disabled(!studio.supportsImageInputs).accessibilityIdentifier("addImages")
                Button { Task { await studio.pasteImage() } } label: { Label("粘贴图片", systemImage: "doc.on.clipboard") }.disabled(!studio.supportsImageInputs).accessibilityIdentifier("pasteImages")
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
            PromptCapacityView(studio: studio, busy: store.busy || submitting)
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(inputSummary).font(.caption)
                    Text(studio.draft.randomSeed ? "每次随机 · 本次 \(studio.lastSeed.map(String.init) ?? "待确定")" : "固定种子 \(studio.draft.seedText)").font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                Button(action: generate) { Label(store.busy ? "正在运行" : (model?.isVideo == true ? "生成视频" : "生成图像"), systemImage: "sparkles").padding(.horizontal, 8).padding(.vertical, 4) }
                    .buttonStyle(.borderedProminent).foregroundStyle(Color(red: 0.13, green: 0.09, blue: 0.04))
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(store.busy || api.running || api.changing || submitting || studio.importing ||
                              model?.executor != true || !studio.selectedStreamingTargetAvailable)
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
                    Button { if ["image.transform", "video.image"].contains(studio.draft.operation) { studio.draft.initImageID = asset.id } } label: {
                        MediaPreview(path: asset.path, maxPixel: 160).frame(width: 78, height: 54).clipped()
                            .overlay(RoundedRectangle(cornerRadius: 5).stroke(studio.draft.activeAssets.contains(asset) ? ciderAccent : .clear, lineWidth: 2))
                    }.buttonStyle(.plain).accessibilityLabel("参考图 \(index + 1)，\(asset.name)，点击选为原图")
                    HStack(spacing: 4) {
                        Text(["image.transform", "video.image"].contains(studio.draft.operation) && studio.draft.initImageID == asset.id ? "原图" : "参考 \(index + 1)").font(.caption2)
                        Menu {
                            Button("设为原图") {
                                let operation = studio.draft.modelID == "ltx-2.5-distilled" ? "video.image" : "image.transform"
                                studio.changeOperation(operation); studio.draft.initImageID = asset.id
                            }
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
                Picker("模型", selection: Binding(get: { studio.draft.modelID }, set: { studio.changeModel($0) })) {
                    ForEach(studio.creationModels) { item in
                        let downloaded = !(studio.draft.modelPaths[item.id] ?? "").isEmpty
                        Label("\(item.name) · \(downloaded ? "已下载" : "未下载")",
                              systemImage: downloaded ? "checkmark.circle.fill" : "arrow.down.circle")
                            .tag(item.id)
                    }
                }.disabled(store.busy || studio.importing).accessibilityIdentifier("studioModel")
                if studio.draft.modelID == "z-image-turbo" {
                    let versions = ZImageInstallation.choices(library.installations, currentPath: studio.draft.modelPath)
                    if !versions.isEmpty {
                        Picker("权重版本", selection: Binding(get: { studio.draft.modelPath }, set: {
                            studio.selectInstallation(modelID: "z-image-turbo", path: $0)
                        })) {
                            ForEach(versions) { version in Text(version.title).tag(version.path) }
                        }
                        .disabled(store.busy || submitting || studio.importing || library.busy || api.running || api.changing)
                        .accessibilityIdentifier("zImageWeightVersion")
                        Text("选择已安装的版本；可在模型中心下载其他版本。")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                Text(studio.draft.accelerationHint).font(.caption).foregroundStyle(.secondary)
                Button(studio.draft.modelPath.isEmpty ? "选择模型…" : "管理模型") { page = .models }
            }
            Divider()
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("采样步数")
                    TextField("采样步数", value: $studio.draft.steps, format: .number)
                        .textFieldStyle(.roundedBorder).accessibilityIdentifier("steps")
                    Button("重置") { studio.draft.steps = model?.default_steps ?? 4 }
                }
                Text(studio.draft.modelID.hasPrefix("z-image-turbo") ? "1–50 步，默认 9 步。其他步数的画质与加速收益需自行验证。" : "1–50 步，默认 \(model?.default_steps ?? 4) 步。")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Text("输出尺寸").font(.subheadline)
            HStack { TextField("宽", value: $studio.draft.width, format: .number).accessibilityIdentifier("width"); Text("×"); TextField("高", value: $studio.draft.height, format: .number).accessibilityIdentifier("height") }.textFieldStyle(.roundedBorder)
            if studio.draft.modelID == "ltx-2.5-distilled" {
                HStack {
                    Button("5 秒 · 480p 桶") {
                        studio.draft.width = 768; studio.draft.height = 448
                        studio.draft.frames = 121; studio.draft.fps = 24
                        studio.draft.steps = 11
                    }.font(.caption)
                    Button("5 秒 · 720p 桶") {
                        studio.draft.width = 1280; studio.draft.height = 704
                        studio.draft.frames = 121; studio.draft.fps = 24
                        studio.draft.steps = 11
                    }.font(.caption)
                }
                Text("LTX 会使用 64 的倍数；480/720 高度分别向下对齐为 448/704。121 帧约 5.04 秒。")
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                HStack { ForEach([256, 512, 768, 1024], id: \.self) { size in Button("\(size)") { studio.draft.width = size; studio.draft.height = size }.font(.caption) } }
            }
            if model?.isVideo == true {
                Divider()
                Text("视频参数").font(.subheadline)
                HStack {
                    TextField("帧数", value: $studio.draft.frames, format: .number).textFieldStyle(.roundedBorder)
                    TextField("FPS", value: $studio.draft.fps, format: .number).textFieldStyle(.roundedBorder)
                }
                if model?.canGenerateAudio == true {
                    Toggle("生成音频", isOn: $studio.draft.audio).controlSize(.small)
                } else { Text("当前执行器输出无音轨视频。").font(.caption).foregroundStyle(.secondary) }
            }
            if ["image.transform", "video.image"].contains(studio.draft.operation) {
                Divider()
                HStack { Text(studio.draft.operation == "video.image" ? "首帧保留强度" : "原图保留强度"); Spacer(); Text(studio.draft.strength, format: .number.precision(.fractionLength(2))).monospacedDigit() }.font(.caption)
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
            VStack(alignment: .leading, spacing: 8) {
                Text("计算设备").font(.caption)
                Toggle("GPU", isOn: .constant(true)).toggleStyle(.checkbox).disabled(true)
                Toggle("额外启用 ANE", isOn: Binding(get: { studio.draft.usesANE }, set: { studio.setANEEnabled($0) }))
                    .toggleStyle(.checkbox).disabled(store.busy || submitting || model?.supports_gpu_ane != true || studio.draft.zImageVariant?.id == "nvfp4").accessibilityIdentifier("enableANE")
                Text(studio.draft.usesANE ? "生成前检查匹配分区。ANE 不保证更快；首次加载较慢，长文本可优先使用 GPU。" : "默认只使用 GPU。").font(.caption2).foregroundStyle(.secondary)
                if studio.draft.usesANE { Button("管理 ANE 分区与缓存") { page = .models } }
                if studio.draft.usesANE, let status = store.accelerationStatus { Text(status).font(.caption2).foregroundStyle(.secondary) }
            }
            if model?.supports_lora == true {
                Divider()
                HStack {
                    Text("LoRA 独立文件").font(.caption); Spacer()
                    Menu("模型库") {
                        ForEach(library.loras.filter { $0.modelID == studio.draft.modelID }) { item in
                            Button(item.name) { studio.draft.loras.append(StudioLoRA(path: item.path)) }
                                .disabled(studio.draft.loras.count >= 8 || studio.draft.loras.contains { $0.path == item.path } || !FileManager.default.isReadableFile(atPath: item.path))
                        }
                    }.disabled(!library.loras.contains { $0.modelID == studio.draft.modelID })
                    Button("添加…", action: chooseLoRA)
                }
                ForEach(studio.draft.loras.indices, id: \.self) { index in
                    VStack(alignment: .leading, spacing: 6) {
                        Toggle(isOn: $studio.draft.loras[index].enabled) { Text(URL(fileURLWithPath: studio.draft.loras[index].path).lastPathComponent).font(.caption2).lineLimit(1) }.toggleStyle(.checkbox).accessibilityIdentifier("loraEnabled-\(index)")
                        HStack {
                            Picker("角色", selection: $studio.draft.loras[index].role) {
                                Text("Transformer").tag("transformer")
                                if studio.draft.modelID.hasPrefix("flux2-") { Text("Text Encoder").tag("text_encoder") }
                                if studio.draft.modelID == "ltx-2.5-distilled" { Text("Refiner").tag("refiner") }
                            }.labelsHidden()
                            Text("强度").font(.caption2)
                            TextField("强度", value: $studio.draft.loras[index].strength, format: .number).textFieldStyle(.roundedBorder).frame(width: 64).disabled(!studio.draft.loras[index].enabled).accessibilityIdentifier("loraStrength-\(index)")
                            Button { studio.draft.loras.remove(at: index) } label: { Image(systemName: "trash") }
                        }
                    }
                }
                Text("强度默认 1.0；取消勾选会保留文件和强度。可输入 −8 到 8。").font(.caption2).foregroundStyle(.secondary)
                Picker("LoRA 执行策略", selection: $studio.draft.loraStrategy) {
                    Text("自动").tag("auto")
                    ForEach(model?.lora_strategies ?? [], id: \.self) { strategy in
                        Text(strategy == "disk_premerge" ? "离线预融合" :
                             strategy == "in_memory_merge" ? "内存融合" :
                             strategy == "inference_time" ? "推理时融合" : strategy).tag(strategy)
                    }
                }.disabled(studio.draft.activeLoRAs.isEmpty)
                Text(loraStrategyHint).font(.caption2).foregroundStyle(.secondary)
                Text(model?.runtime_lora == true ? "运行时按文件身份缓存并应用，不复制整份 checkpoint。" : "此模型要求 LoRA 对应的预融合 checkpoint 与 provenance manifest。")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            DisclosureGroup("高级参数") {
                VStack(alignment: .leading, spacing: 12) {
                    Toggle("动态文本长度", isOn: $studio.draft.dynamicText).controlSize(.small)
                    if studio.draft.publicStreamingModel {
                        Picker("流式加载", selection: Binding(
                            get: { studio.draft.streaming.selection },
                            set: { studio.setStreamingSelection($0) })) {
                            ForEach(StudioStreamingSelection.allCases) { value in
                                let option = studio.streamingOption(for: value)
                                let unavailable = value != .off && option?.status != "available"
                                Text(value.label + (option?.status == "candidate" ? "（待验证）" : unavailable ? "（不可用）" : ""))
                                    .tag(value)
                                    .disabled(unavailable)
                            }
                        }
                        .disabled(store.busy || submitting)
                        .accessibilityIdentifier("publicStreamingSelection")
                        if studio.streamingOptionsLoading {
                            Text("正在检查当前模型与本机物理内存的可用档位…")
                                .font(.caption2).foregroundStyle(.secondary)
                        } else if studio.draft.streaming.selection != .off,
                                  let option = studio.streamingOption(for: studio.draft.streaming.selection) {
                            let selection = studio.draft.streaming.selection
                            let detail = option.status == "available"
                                ? "已校验本地模型，生成前会再次确认。"
                                : option.status == "candidate"
                                    ? "请先选择本地模型，完成档位验证。"
                                    : "当前模型或任务尚无匹配的已验证档位。"
                            Text("\(selection.label)：\(detail)")
                                .font(.caption2)
                                .foregroundStyle(option.status == "available" ? .green : .orange)
                        } else if studio.draft.streaming.selection == .off {
                            let recommendation = studio.recommendedStreamingSelection
                            if recommendation != .off {
                                Text("当前可用档位：\(recommendation.label)；Off 使用默认加载方式。")
                                    .font(.caption2).foregroundStyle(.secondary)
                            } else {
                                Text(studio.streamingOptions?.targets.contains(where: { $0.status == "candidate" }) == true
                                     ? "已有候选档位，请先选择本地模型完成验证。"
                                     : "暂无经过验证的流式档位。Off 使用默认加载方式。")
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                        if let error = studio.streamingOptionsError {
                            Text(error).font(.caption2).foregroundStyle(.orange)
                        }
                    }
                    if studio.draft.modelID == "ltx-2.5-distilled" {
                        Picker("LTX 后端", selection: $studio.draft.ltxBackend) {
                            Text("自动（C/Metal）").tag("auto")
                            Text("C/Metal").tag("c_metal")
                            Text("C++/MLX（实验）").tag("cpp_mlx")
                        }
                        Toggle("GPU 快速 A/V 调度", isOn: $studio.draft.ltxFastAV)
                            .controlSize(.small)
                        Text("Fast A/V 会并行音频分支并合并命令缓冲区；已通过 97 帧 latent 一致性门禁。")
                            .font(.caption2).foregroundStyle(.secondary)
                        Toggle("Video Attention 批处理（实验）",
                               isOn: $studio.draft.ltxVideoAttentionBatch)
                            .controlSize(.small)
                            .disabled(studio.draft.ltxAccelerationMode != "quality")
                        Text("严格等价；当前 97 帧 ABBA 的 denoise 约快 1.1%，但总耗时收益不稳定。Sol 模式会自动关闭。")
                            .font(.caption2).foregroundStyle(.secondary)
                        Picker("LTX 加速模式", selection: $studio.draft.ltxAccelerationMode) {
                            Text("画质优先（Dense）").tag("quality")
                            Text("Stage-2 Sol 近似").tag("sol")
                            Text("Sol + 256 行文本（最快候选）").tag("fast_approx")
                        }
                        .disabled(store.busy || submitting)
                        Text("近似模式只修改 Stage-2；需要多 prompt/seed 质量回归，720p 自动限制为画质优先。")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    if studio.draft.modelID == "z-image-turbo", studio.draft.zImageRequiresResident, !studio.draft.usesPublicStreaming {
                        LabeledContent("模型驻留", value: "常驻")
                        Text(studio.draft.zImageVariant?.id == "int8-convrot"
                             ? "INT8 流式加载仅在 Apple M5 Pro、24 GiB 内存的机器上启用；当前设备使用常驻加载。"
                             : "此权重版本使用常驻加载。")
                            .font(.caption2).foregroundStyle(.secondary)
                    } else if studio.draft.modelID == "z-image-turbo" && !studio.draft.usesPublicStreaming {
                        Picker("模型驻留", selection: $studio.draft.residency) {
                            Text("常驻").tag("resident")
                            Text("流式加载（实验）").tag("streamed")
                        }.disabled(store.busy || submitting).accessibilityIdentifier("zImageResidency")
                        if studio.draft.residency == "streamed" {
                            Picker("采样内存预算", selection: $studio.draft.zImageStreamingBudgetGiB) {
                                ForEach([6, 8, 10, 12], id: \.self) { Text("\($0) GiB").tag($0) }
                            }.disabled(store.busy || submitting)
                            Text(AccelerationDiscovery.optimizationEnabled("z_image_suffix_streaming")
                                 ? "支持 Comfy BF16 / INT8 ConvRot，不支持 LoRA。本机已启用 M5 ANE 流式优化：只加载 GPU 负责的 MLP 权重，首次准备需要整理权重。预算不包含 ANE，也不是整个应用的内存硬上限。"
                                 : "支持 Comfy BF16、纯 GPU、不使用 LoRA。提前读取下一层以降低内存占用；速度取决于磁盘。预算用于采样阶段规划，并非整个应用的内存硬上限。")
                                .font(.caption2).foregroundStyle(.secondary)
                            if studio.draft.zImageVariant?.id == "int8-convrot" {
                                Text("INT8 权重能全部放入预算时会自动保留，避免重复读取。")
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    } else if studio.draft.modelID == "z-image-turbo-gguf" {
                        LabeledContent("模型驻留", value: "常驻")
                            .font(.caption)
                    } else if !studio.draft.usesPublicStreaming {
                        Picker("模型驻留", selection: $studio.draft.residency) { Text("保留图像权重").tag("resident"); Text("分阶段释放").tag("component_staged") }
                    }
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
                HStack(alignment: .firstTextBaseline) {
                    Text(phaseName(job.phase)).font(.title3.weight(.semibold))
                    if job.phase == "denoise" { Text("\(job.completed) / \(job.total) 步已完成").font(.title2.weight(.semibold)).monospacedDigit().accessibilityIdentifier("denoiseStepProgress") }
                    Spacer()
                    Text(stateName(job.state)).font(.callout).foregroundStyle(.secondary)
                }
                Text(store.actualRoute.map { "实际路径：\($0)" } ?? "实际路径：准备后确认").font(.caption2).foregroundStyle(.secondary)
                ProgressView(value: Double(job.completed), total: Double(max(1, job.total))).tint(ciderAccent)
                HStack {
                    Text("\(stateName(job.state)) · \(phaseName(job.phase)) \(job.completed)/\(job.total)")
                    Spacer()
                    if job.phase == "denoise", let speed = job.secondsPerStep { Text(String(format: "%.2f 秒/步", speed)).monospacedDigit() }
                    Text(String(format: "%.1f 秒", job.elapsed)).monospacedDigit()
                    Button("取消") { store.cancel() }.disabled(job.state == "cancelling")
                }.font(.callout)
                if job.phase == "denoise", let speed = job.secondsPerStep, job.completed < job.total {
                    Text("采样阶段预计还需约 \(Int(ceil(speed * Double(job.total - job.completed)))) 秒，图像解码另计。")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            } else if store.busy { HStack { ProgressView().controlSize(.small); Text(store.sessionState).font(.caption); Spacer(); Button("取消") { store.cancel() } } }
            else if submitting || store.resolvingAcceleration {
                HStack { ProgressView().controlSize(.small); Text(store.accelerationStatus ?? "正在准备生成请求…").font(.callout); Spacer() }
            }
            else if let job = store.jobs.first {
                Text("\(stateName(job.state)) · 共用时 \(String(format: "%.1f", job.elapsed)) 秒 · 种子 \(job.request.seed)").font(.caption).foregroundStyle(.secondary)
                if let json = job.resultJSON {
                    Text(RunInsights(json: json).cacheLabel).font(.callout).foregroundStyle(.secondary)
                }
            }
            if let error = studio.message ?? store.storageError {
                HStack(alignment: .top) { Text(error).font(.caption).textSelection(.enabled); Spacer(); Button { studio.message = nil } label: { Image(systemName: "xmark") } }
                    .foregroundStyle(.secondary).accessibilityIdentifier("statusMessage")
            }
        }
    }
    private var modelsPage: some View {
        ModelLibraryView(store: store, studio: studio, library: library, chooseModel: chooseModel,
                         loadModel: loadModel, importConfiguration: importConfiguration,
                         operationName: operationName)
    }
    private var tasksPage: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("任务与历史").font(.largeTitle.weight(.medium))
            Text("删除任务记录保留结果文件；删除图片会移到废纸篓。").foregroundStyle(.secondary)
            HStack {
                Button(selectedTasks == store.deletableJobIDs && !selectedTasks.isEmpty ? "取消全选" : "全选") {
                    selectedTasks = selectedTasks == store.deletableJobIDs ? [] : store.deletableJobIDs
                }.disabled(store.deletableJobIDs.isEmpty).accessibilityIdentifier("selectAllTasks")
                Text("已选择 \(selectedTasks.count) 项").font(.callout).foregroundStyle(.secondary)
                Button("删除所选任务", role: .destructive) { deleteTasks(selectedTasks) }
                    .disabled(selectedTasks.isEmpty).accessibilityIdentifier("deleteSelectedTasks")
                Spacer()
                Button("清空任务历史", role: .destructive) {
                    do { try store.clearHistory(); selectedTasks = [] }
                    catch { studio.message = error.localizedDescription }
                }.disabled(store.deletableJobIDs.isEmpty).accessibilityIdentifier("clearTaskHistory")
            }
            if store.jobs.contains(where: { !store.deletableJobIDs.contains($0.id) }) {
                Text("进行中的任务会保留，完成或取消后可删除。").font(.caption).foregroundStyle(.secondary)
            }
            if !store.deletedJobs.isEmpty { Button("撤销删除 \(store.deletedJobs.count) 项任务") { do { try store.undoDeleteJob() } catch { studio.message = error.localizedDescription } }.accessibilityIdentifier("undoDeleteJob") }
            List(store.jobs) { job in
                DisclosureGroup {
                    Text(job.request.prompt).textSelection(.enabled)
                    if let json = job.resultJSON { RunInsightsView(json: json) }
                    if let error = job.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
                    HStack {
                        Button("复用参数") { studio.reuse(job); page = .studio }
                        if job.hasOutput {
                            Button("查看结果") { showResult(job) }
                            Button("删除图片", role: .destructive) { trashResult(job) }.disabled(store.busy)
                        }
                        if job.outputDeleted == true { Text("图片已删除").foregroundStyle(.secondary) }
                        Button("删除任务记录", role: .destructive) { deleteTasks([job.id]) }
                            .disabled(!store.deletableJobIDs.contains(job.id)).accessibilityIdentifier("deleteJob")
                    }.buttonStyle(.borderless).accessibilityElement(children: .contain)
                    if let json = job.resultJSON { DisclosureGroup("执行与性能详情") { Text(json).font(.system(.caption, design: .monospaced)).textSelection(.enabled) } }
                } label: {
                    HStack(spacing: 10) {
                        Toggle("选择任务：\(job.request.prompt)", isOn: Binding(
                            get: { selectedTasks.contains(job.id) },
                            set: { if $0 { selectedTasks.insert(job.id) } else { selectedTasks.remove(job.id) } }
                        )).toggleStyle(.checkbox).labelsHidden()
                            .disabled(!store.deletableJobIDs.contains(job.id))
                            .accessibilityIdentifier("selectTask-\(job.id)")
                        VStack(alignment: .leading) { Text(job.request.prompt).lineLimit(1); Text("\(operationName(job.request.operation ?? "image.generate")) · \(job.request.width)×\(job.request.height) · 种子 \(job.request.seed)").font(.caption).foregroundStyle(.secondary) }; Spacer(); Text(stateName(job.state)).font(.caption)
                    }
                }.padding(.vertical, 6)
            }.listStyle(.inset)
            runStatus
        }.padding(28)
    }
    private var libraryPage: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("素材库").font(.largeTitle.weight(.medium))
            Text("单击选择，Ctrl/⌘ 点击多选，Shift 点击连续选择；双击查看结果。").foregroundStyle(.secondary)
            resultSelectionControls
            ScrollView {
                if outputJobs.isEmpty { ContentUnavailableView("还没有生成结果", systemImage: "photo", description: Text("在创作中生成第一张图像。")) }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 180))], spacing: 18) {
                    ForEach(outputJobs) { job in
                        VStack(alignment: .leading, spacing: 8) {
                            resultThumbnail(job, height: 180)
                            Text(job.request.prompt).font(.caption).lineLimit(2)
                            HStack {
                                Text("种子 \(job.request.seed)").font(.caption2).foregroundStyle(.secondary)
                                Spacer()
                                Button { showResult(job) } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }.help("查看结果")
                                Button { trashResults(resultDeletionIDs(for: job)) } label: { Image(systemName: "trash") }
                                    .help(resultDeletionTitle(for: job)).disabled(store.busy).accessibilityIdentifier("trashOutput")
                            }
                        }.padding(8)
                            .background(resultSelection.ids.contains(job.id) ? ciderAccent.opacity(0.10) : .clear, in: RoundedRectangle(cornerRadius: 10))
                            .overlay(RoundedRectangle(cornerRadius: 10).stroke(resultSelection.ids.contains(job.id) ? ciderAccent : .clear, lineWidth: 2))
                            .contextMenu {
                                Button("查看结果") { showResult(job) }
                                Button(resultDeletionTitle(for: job), role: .destructive) { trashResults(resultDeletionIDs(for: job)) }.disabled(store.busy)
                            }
                    }
                }.padding(2)
            }
            runStatus
        }.padding(28)
    }
    private var resultSelectionControls: some View {
        HStack {
            Text("已选择 \(resultSelection.ids.count) 项").font(.callout).foregroundStyle(.secondary)
            Button("全选") { resultSelection.selectAll(outputIDs) }.disabled(outputIDs.isEmpty).accessibilityIdentifier("selectAllOutputs")
            Button("取消选择") { resultSelection.clear() }.disabled(resultSelection.ids.isEmpty)
            Spacer()
            Button("删除所选图片", role: .destructive) { trashResults(resultSelection.ids) }
                .disabled(store.busy || resultSelection.ids.isEmpty).help("将所选结果移到废纸篓")
                .accessibilityIdentifier("trashSelectedOutputs")
        }
    }
    private func resultThumbnail(_ job: NativeJob, height: CGFloat) -> some View {
        StudioResultThumbnail(path: job.request.output).frame(height: height).frame(maxWidth: .infinity)
            .accessibilityHidden(true)
            .overlay(alignment: .topTrailing) {
                if resultSelection.ids.contains(job.id) {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(.white, ciderAccent).padding(4).allowsHitTesting(false)
                }
            }
            .overlay {
                ResultSelectionTarget(label: "选择图片：\(job.request.prompt)，种子 \(job.request.seed)", selected: resultSelection.ids.contains(job.id)) { flags, clicks in
                    let toggle = !flags.intersection([.control, .command]).isEmpty
                    resultSelection.select(job.id, orderedIDs: outputIDs, toggle: toggle, range: flags.contains(.shift))
                    if resultSelection.ids.contains(job.id) { selected = job.id; compareOriginal = false }
                    if clicks == 2 && !toggle && !flags.contains(.shift) { showResult(job) }
                }
            }
    }
    private func showResult(_ job: NativeJob) {
        selected = job.id; compareOriginal = false
        resultSelection.select(job.id, orderedIDs: outputIDs)
        page = .studio
    }
    private func resultDeletionIDs(for job: NativeJob) -> Set<UUID> {
        resultSelection.ids.contains(job.id) ? resultSelection.ids : [job.id]
    }
    private func resultDeletionTitle(for job: NativeJob) -> String {
        let count = resultDeletionIDs(for: job).count
        return count > 1 ? "删除所选 \(count) 个结果（移到废纸篓）" : "删除图片（移到废纸篓）"
    }
    private func deleteTasks(_ ids: Set<UUID>) {
        do { try store.deleteJobs(ids); selectedTasks.subtract(ids) }
        catch { studio.message = error.localizedDescription }
    }
    private func chooseModel(_ id: String) {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = "选择与 \(id) 对应的本地模型目录。TurboCider 会在打开时校验所需权重。"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if id == "z-image-turbo" {
            var shared: URL?
            if ZImageInstallation.needsSharedText(url) {
                let textPanel = NSOpenPanel(); textPanel.canChooseDirectories = true; textPanel.canChooseFiles = false
                textPanel.message = "此模型缺少文本组件。选择包含 text_encoder 和 tokenizer 的 Qwen3-4B 模型目录，例如 FLUX.2-klein-4B。"
                if let path = studio.draft.modelPaths["flux2-klein-4b"], !path.isEmpty { textPanel.directoryURL = URL(fileURLWithPath: path) }
                guard textPanel.runModal() == .OK, let selected = textPanel.url else { return }
                shared = selected
            }
            do { try studio.installZImage(model: url, sharedText: shared) }
            catch { studio.message = error.localizedDescription }
        } else { studio.draft.modelPaths[id] = url.path; studio.selectModel(id); studio.save() }
    }
    private func importConfiguration() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        panel.message = "选择 TurboCider App 生成配置，恢复模型、LoRA、提示词和加速设置。"
        if panel.runModal() == .OK, let url = panel.url {
            do { try studio.importConfiguration(from: url); page = .studio }
            catch { studio.message = "无法导入生成配置：\(error.localizedDescription)" }
        }
    }
    private func chooseProfile() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        if panel.runModal() == .OK, let url = panel.url { studio.draft.profilePath = url.path; var acceleration = studio.draft.acceleration ?? StudioAcceleration(); acceleration.policy = "profile"; studio.draft.acceleration = acceleration }
    }
    private func chooseImages() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { let urls = panel.urls; Task { await studio.addFiles(urls) } }
    }
    private func chooseLoRA() {
        guard studio.draft.loras.count < 8 else { studio.message = "最多可配置 8 个 LoRA 文件。"; return }
        let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.canChooseFiles = true
        panel.allowsMultipleSelection = false; panel.message = "选择独立的 .safetensors LoRA 文件。文件不会复制或合并进基础 checkpoint。"
        panel.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
        if let split = ZImageInstallation.splitDirectory(URL(fileURLWithPath: studio.draft.modelPath)) {
            panel.directoryURL = split.appendingPathComponent("loras")
        }
        if panel.runModal() == .OK, let url = panel.url {
            studio.draft.loras.append(StudioLoRA(path: url.path))
            library.registerLoRA(url, modelID: studio.draft.modelID)
        }
    }
    private func loadModel(_ id: String) {
        guard !api.running, !api.changing else { studio.message = "请先停止本地 API 服务，再加载 App 创作会话。"; return }
        guard let path = studio.draft.modelPaths[id], !path.isEmpty else { return }
        studio.message = nil
        if studio.draft.modelID != id { studio.selectModel(id) }
        let snapshot = studio.draft
        Task { do {
            let resolved = try await store.resolveAcceleration(snapshot)
            studio.rememberAcceleration(resolved)
            if resolved.usesPublicStreaming {
                throw NativeFailure(message: "public 流式加载会在生成时解析并锁定档位；请直接点击生成。")
            }
            let output = store.directory.appendingPathComponent("unused-prepare.png")
            let request = try await Task.detached { try resolved.request(output: output) }.value
            try await store.prepare(modelURL: URL(fileURLWithPath: path), request: request, warmup: false)
        } catch { studio.message = error is CancellationError ? "加载已取消" : error.localizedDescription } }
    }
    private func trashResult(_ job: NativeJob) {
        trashResults([job.id])
    }
    private func trashResults(_ ids: Set<UUID>) {
        do {
            try store.trashOutputs(ids)
            resultSelection.retain(Set(outputIDs))
            if let selected, !outputIDs.contains(selected) { self.selected = nil }
            compareOriginal = false
        }
        catch { studio.message = error.localizedDescription }
    }
    private func generate() {
        guard !store.busy, !api.running, !api.changing, !submitting, !studio.importing else { return }
        let snapshot = studio.draft
        let ext = model?.isVideo == true ? "mp4" : "png"
        let output = store.directory.appendingPathComponent("outputs/\(UUID().uuidString).\(ext)")
        submitting = true; studio.message = nil
        Task {
            defer { submitting = false }
            do {
                let resolved = try await store.resolveAcceleration(snapshot)
                studio.rememberAcceleration(resolved)
                let pair = try await Task.detached {
                    try resolved.publicStreamingRequest(output: output)
                }.value
                studio.lastSeed = pair.legacy.seed; studio.save()
                let job = try await store.generate(
                    modelURL: URL(fileURLWithPath: resolved.modelPath),
                    request: pair.legacy,
                    streamingRequest: pair.v2)
                selected = job.id; compareOriginal = false
                resultSelection.select(job.id, orderedIDs: outputIDs)
            } catch { studio.message = error is CancellationError ? "生成已取消，草稿与原图已保留。" : error.localizedDescription }
        }
    }
    private func exportResult(_ path: String) {
        let panel = NSSavePanel(); panel.allowedContentTypes = URL(fileURLWithPath: path).pathExtension.lowercased() == "png" ? [.png] : [.mpeg4Movie]; panel.nameFieldStringValue = URL(fileURLWithPath: path).lastPathComponent
        if panel.runModal() == .OK, let destination = panel.url {
            Task {
                do { try await Task.detached { try Data(contentsOf: URL(fileURLWithPath: path)).write(to: destination, options: .atomic) }.value }
                catch { studio.message = error.localizedDescription }
            }
        }
    }
}
