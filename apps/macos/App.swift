import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers

@main
struct TurboCiderNativeApp: App {
    @StateObject private var store: NativeJobStore
    @StateObject private var studio: StudioState
    @StateObject private var api: LocalAPIController
    @State private var submitting = false
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
            StudioView(store: store, studio: studio, api: api, submitting: $submitting).frame(minWidth: 980, minHeight: 700)
                .onAppear { delegate.store = store; delegate.studio = studio; delegate.api = api }
        }
        .commands {
            CommandGroup(replacing: .newItem) { Button("新建创作") { studio.newDraft() }.keyboardShortcut("n")
                .disabled(store.busy || submitting || studio.importing) }
            CommandGroup(after: .pasteboard) {
                Button("粘贴图片到创作") { Task { await studio.pasteImage() } }.keyboardShortcut("v", modifiers: [.command, .shift])
                    .disabled((submitting && !store.busy) || studio.importing)
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
    ["image.upscale": "图片超分", "image.generate": "文生图", "image.transform": "单图修改", "image.edit": "参考编辑",
     "video.generate": "文生视频", "video.image": "图生视频", "video.keyframes": "关键帧视频",
     "video.reference": "参考视频"][value] ?? value
}
private func stateName(_ value: String) -> String {
    ["preparing": "准备中", "running": "生成中", "cleanup_pending": "等待进程清理", "finalizing": "正在保存结果", "cancelling": "正在安全停止", "succeeded": "已完成", "failed": "失败", "cancelled": "已取消", "interrupted": "已中断"][value] ?? value
}
private func phaseName(_ value: String) -> String {
    if value == "pack_z_image_suffix" { return "整理 GPU 权重" }
    if value == "upscale" { return "图像超分" }
    if value == "denoise" { return "采样" }
    if ["image_encode", "vision_encode", "reference_vae_encode"].contains(value) { return "编码参考图" }
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
    @State private var upscaleSourcePath = ""
    @State private var imageUpscaling = false
    @State private var librarySelection: String?
    @Binding var submitting: Bool
    @State private var annotationAsset: StudioAsset?
    private var selectedJob: NativeJob? { store.jobs.first { $0.id == selected && $0.hasOutput } ?? store.jobs.first { $0.hasOutput } }
    private var outputJobs: [NativeJob] { store.jobs.filter { $0.hasOutput } }
    private var outputIDs: [UUID] { outputJobs.map(\.id) }
    private var model: StudioModel? { studio.models.first { $0.id == studio.draft.modelID } }
    private var preparingSubmission: Bool { submitting && !store.busy }
    private var assetControlsLocked: Bool { preparingSubmission || studio.importing }
    private var hasDetailedRunStatus: Bool {
        if store.activeJob != nil || store.busy || store.workerCleanupPending || submitting || store.resolvingAcceleration ||
            studio.message != nil || store.storageError != nil || studio.draft.zImageStreamingConflict() != nil { return true }
        guard let job = store.jobs.first else { return false }
        return (job.state == "failed" || job.state == "interrupted") && job.error != nil
    }
    private var referenceImportLimit: Int {
        studio.draft.qwen21TurboLoRA != nil && studio.draft.operation == "image.edit"
            ? 3 : studio.imageImportLimit
    }
    private var workflowOperations: [String] {
        if imageUpscaling { return ["image.generate", "image.transform", "image.edit"] }
        if studio.draft.modelID == "qwen-image-2.1" { return ["image.generate", "image.edit"] }
        return studio.creationOperations
    }
    private var generationBlocker: String? {
        guard let model, model.executor else { return "请选择可运行的本地模型。" }
        if studio.draft.modelPath.isEmpty { return "请先选择本地模型文件夹。" }
        if !model.supports(studio.draft.operation) { return "当前模型不支持此创作方式，请重新选择。" }
        if studio.draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "请输入画面描述或编辑指令。" }
        let count = studio.draft.activeAssets.count
        if studio.draft.modelID == "qwen-image-2.1", !studio.draft.activeLoRAs.isEmpty, studio.draft.qwen21TurboLoRA == nil {
            return "Qwen 2.1 请选择单个已支持的 Viggle 六步 r128 / r256 LoRA。"
        }
        if let adapter = studio.draft.qwen21TurboLoRA {
            if adapter.role != "transformer" || adapter.strength != 1 {
                return "六步 LoRA 需要 Transformer 角色、强度 1；请应用 GPU 预设。"
            }
            if studio.draft.width != 512 || studio.draft.height != 512 || studio.draft.steps != 6 {
                return "六步 LoRA 需要 512×512、6 步；请在参数中应用 GPU 预设。"
            }
            if studio.draft.operation == "image.edit", count > 3 { return "六步 LoRA 最多使用 3 张参考图，请移除多余图片或关闭 LoRA。" }
        }
        if ["image.transform", "video.image"].contains(studio.draft.operation), count != 1 { return "请添加并选择一张原图。" }
        if ["image.edit", "video.reference", "video.keyframes"].contains(studio.draft.operation), count == 0 { return "请添加至少一张参考图。" }
        if count > (model.max_images ?? 8) { return "当前模型最多支持 \(model.max_images ?? 8) 张输入图，请移除多余图片。" }
        if !studio.selectedStreamingTargetAvailable { return "当前流式档位不可用，请在参数中选择可用档位。" }
        return nil
    }
    private var workflowDescription: String {
        switch studio.draft.operation {
        case "image.transform": return "选择一张原图，描述想修改的内容。"
        case "image.edit": return "添加 1–\(studio.draft.qwen21TurboLoRA != nil ? 3 : (model?.max_images ?? 8)) 张参考图；提示词中的图片编号与下方顺序一致。"
        case "video.image": return "选择首帧图片，描述镜头与动作。"
        case "video.keyframes": return "按顺序添加首帧与尾帧，描述画面如何变化。"
        case "video.reference": return "添加参考图，描述人物、场景和动作。"
        default: return model?.isVideo == true ? "描述场景、镜头与动作，在本机生成视频。" : "描述主体、场景与风格，在本机生成图片。"
        }
    }
    private var promptPlaceholder: String {
        studio.draft.activeAssets.isEmpty ? "描述你想创作的画面…" : "描述修改内容，以及需要保留的主体、构图或细节…"
    }
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
                case .api: LocalAPIView(api: api, store: store).disabled(preparingSubmission)
                }
            }.background(Color(nsColor: .windowBackgroundColor))
        }
        .tint(ciderAccent)
        .toolbar {
            ToolbarItem(placement: .automatic) { Text("本地创作").foregroundStyle(.secondary) }
            ToolbarItem(placement: .automatic) { Text(studio.saved ? "草稿已保存" : "草稿尚未保存").font(.caption).foregroundStyle(.secondary) }
            ToolbarItem { Button { studio.newDraft(); selected = nil; resultSelection.clear(); imageUpscaling = false; page = .studio } label: { Label("新建创作", systemImage: "square.and.pencil") }.disabled(store.busy || assetControlsLocked) }
            ToolbarItem { Button { inspector.toggle() } label: { Label(inspector ? "隐藏参数" : "显示参数", systemImage: "sidebar.right") }.help(inspector ? "收起生成参数" : "展开模型与生成参数") }
        }
        .onDisappear { studio.save() }
        .onChange(of: store.deletableJobIDs) { _, ids in selectedTasks.formIntersection(ids) }
        .onChange(of: outputIDs) { _, ids in resultSelection.retain(Set(ids)) }
        .onChange(of: library.installationGeneration) { _, _ in studio.invalidateStreamingInstallation() }
        .sheet(item: $annotationAsset) { asset in Qwen21AnnotationEditor(asset: asset, studio: studio) }
        .task { library.refresh(studio: studio, migrate: true) }
        .task(id: "\(page == .studio && imageUpscaling):\(studio.draft.upscaleAutoPreload):\(studio.draft.upscaleAfterGeneration):\(studio.draft.upscaleModelPath):\(studio.draft.upscaleCompute.rawValue)") {
            do { try await Task.sleep(for: .milliseconds(350)) } catch { return }
            guard !preparingSubmission, studio.draft.upscaleAutoPreload, (page == .studio && imageUpscaling || studio.draft.upscaleAfterGeneration),
                  (try? ImageUpscaler.validateModelURL(URL(fileURLWithPath: studio.draft.upscaleModelPath))) != nil,
                  !studio.draft.upscaleModelPath.isEmpty, !store.busy, !api.running, !api.changing else { return }
            do {
                try await store.preloadUpscaler(modelURL: URL(fileURLWithPath: studio.draft.upscaleModelPath), compute: studio.draft.upscaleCompute)
                if let info = store.upscaleReady { studio.rememberUpscaleModel(info) }
            } catch { if !(error is CancellationError) { studio.message = error.localizedDescription } }
        }
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
        VStack(spacing: 0) {
            creationHeader.padding(.horizontal, 18).padding(.top, 18)
            if imageUpscaling {
                ImageUpscaleView(store: store, studio: studio, showSettings: inspector, sourcePath: $upscaleSourcePath,
                                 manageModels: { librarySelection = studio.draft.upscaleVariant.modelID; page = .models }) { job in
                    selected = job.id; compareOriginal = false; resultSelection.select(job.id, orderedIDs: outputIDs)
                }
            } else {
                generationWorkspace
            }
        }
    }
    private var creationHeader: some View {
        VStack(spacing: 12) {
                HStack {
                    Picker("创作类型", selection: Binding(get: { imageUpscaling ? "image" : studio.creationKind }, set: {
                        guard $0 != (imageUpscaling ? "image" : studio.creationKind) else { return }
                        imageUpscaling = false; studio.changeCreationKind($0); compareOriginal = false
                    })) {
                        Label("图片生成", systemImage: "photo").tag("image")
                        Label("视频生成", systemImage: "video").tag("video")
                    }.pickerStyle(.segmented).frame(maxWidth: 300)
                        .disabled(store.busy || assetControlsLocked).accessibilityIdentifier("creationKind")
                    Spacer()
                    Button { page = .models } label: { Label("模型中心", systemImage: "square.stack.3d.up") }
                }
                HStack {
                    Picker("创作方式", selection: Binding(get: { imageUpscaling ? "image.transform" : studio.draft.operation }, set: {
                        guard $0 != (imageUpscaling ? "image.transform" : studio.draft.operation) else { return }
                        imageUpscaling = false
                        studio.changeOperation(studio.draft.modelID == "qwen-image-2.1" && $0 == "image.transform" ? "image.edit" : $0)
                        compareOriginal = false
                    })) {
                        ForEach(workflowOperations, id: \.self) { operation in
                            Text(imageUpscaling && operation == "image.transform" ? "图像超分"
                                 : studio.draft.modelID == "qwen-image-2.1" && operation == "image.edit" ? "图片编辑"
                                 : operationName(operation)).tag(operation)
                        }
                    }.pickerStyle(.segmented).frame(maxWidth: 440).disabled(store.busy || assetControlsLocked).accessibilityIdentifier("operation")
                    Spacer()
                    VStack(alignment: .trailing, spacing: 3) {
                        Text(model?.name ?? "请选择模型").font(.caption).lineLimit(1)
                        Text(studio.draft.usesANE && studio.draft.qwen21TurboLoRA == nil ? "GPU + ANE" : "GPU").font(.caption2).foregroundStyle(.secondary)
                    }.help(studio.draft.modelPath)
                }
            if imageUpscaling || studio.draft.operation == "image.transform" {
                Picker("单图处理", selection: Binding(get: { imageUpscaling }, set: {
                    imageUpscaling = $0
                    if !$0 { studio.changeOperation(model?.supports("image.transform") == true ? "image.transform" : "image.edit") }
                    else if upscaleSourcePath.isEmpty { upscaleSourcePath = studio.draft.activeAssets.first?.path ?? "" }
                })) {
                    Text("修改内容").tag(false)
                    Text("图像超分").tag(true)
                }.pickerStyle(.segmented).frame(maxWidth: 300)
                    .disabled(store.busy || assetControlsLocked).accessibilityIdentifier("singleImageTool")
            }
            if !imageUpscaling {
                Text(workflowDescription).font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
    private var generationWorkspace: some View {
        HStack(spacing: 0) {
            VStack(spacing: 12) {
                if api.running { HStack { Label("本地 API 正在接收任务", systemImage: "network"); Spacer(); Button("管理服务") { page = .api } }.font(.callout).padding(10).background(ciderAccent.opacity(0.1), in: RoundedRectangle(cornerRadius: 8)) }
                mediaStage.frame(maxWidth: .infinity, minHeight: selectedJob == nil ? 180 : (hasDetailedRunStatus ? 200 : 240), maxHeight: .infinity)
                    .layoutPriority(selectedJob == nil ? 0 : 1)
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
                let metadata = resultMetadata(job)
                HStack {
                    Text(store.activeJob == nil ? "生成结果" : "上一结果 · 新任务进行中").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if job.request.inputs?.first != nil { Toggle("查看原图", isOn: $compareOriginal).toggleStyle(.button).font(.caption) }
                }
                StudioOutputPreview(path: compareOriginal ? (job.request.inputs?.first?.path ?? job.request.output) : job.request.output)
                    .frame(maxWidth: .infinity, minHeight: hasDetailedRunStatus ? 100 : 140, maxHeight: .infinity)
                    .layoutPriority(1).accessibilityIdentifier("generatedImage")
                HStack(spacing: 12) {
                    Text(metadata).font(.caption2).monospacedDigit().foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.tail).frame(maxWidth: .infinity, alignment: .leading).help(metadata)
                    if URL(fileURLWithPath: job.request.output).pathExtension.lowercased() == "png" {
                        Button("编辑此图") { Task {
                            if await studio.editResult(job) {
                                imageUpscaling = false; compareOriginal = false; page = .studio
                            }
                        } }
                            .disabled(store.busy || assetControlsLocked)
                            .help("以这张结果开始单图编辑，保留对应模型。")
                            .accessibilityIdentifier("editResult")
                    }
                    Menu {
                        if URL(fileURLWithPath: job.request.output).pathExtension.lowercased() == "png" {
                            Button("单独超分此图…") { upscaleSourcePath = job.request.output; imageUpscaling = true; page = .studio }
                        }
                        Button("另存为…") { exportResult(job.request.output) }
                        Button("在 Finder 中显示") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: job.request.output)]) }
                        Button("复用参数") { reuseParameters(job) }.disabled(store.busy || assetControlsLocked)
                        Button(resultDeletionTitle(for: job), role: .destructive) { trashResults(resultDeletionIDs(for: job)) }.disabled(store.busy)
                    } label: { Image(systemName: "ellipsis.circle") }.menuStyle(.borderlessButton).frame(width: 26)
                }
            } else if let asset = studio.draft.activeAssets.first {
                Text(studio.draft.activeAssets.count == 1 ? "原图预览" : "参考 1 预览 · 共 \(studio.draft.activeAssets.count) 张")
                    .font(.caption).foregroundStyle(.secondary)
                MediaPreview(path: asset.path).frame(maxWidth: .infinity, maxHeight: .infinity)
                Text("\(asset.name) · \(asset.width) × \(asset.height)").font(.caption).foregroundStyle(.secondary).lineLimit(1)
            } else {
                Image(systemName: "photo.on.rectangle.angled").font(.system(size: 42, weight: .ultraLight)).foregroundStyle(.secondary)
                Text(["image.transform", "image.edit", "video.image", "video.reference", "video.keyframes"].contains(studio.draft.operation) ? "添加图片，开始编辑" : "把想法变成画面").font(.title2.weight(.medium))
                Text(workflowDescription).font(.callout).multilineTextAlignment(.center).foregroundStyle(.secondary)
                if ["image.transform", "image.edit", "video.image", "video.reference", "video.keyframes"].contains(studio.draft.operation) {
                    Button(action: chooseImages) { Label("选择图片…", systemImage: "plus") }
                        .disabled(assetControlsLocked || !studio.supportsImageInputs)
                }
            }
        }.padding(12).background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 12))
    }
    private func resultMetadata(_ job: NativeJob) -> String {
        if compareOriginal { return "输入原图" }
        let dimensions = "\(job.request.width) × \(job.request.height)"
        let detail = job.request.operation == "image.upscale" ? "超分结果" : "种子 \(job.request.seed)"
        return "\(dimensions) · \(detail)" + (job.routeSummary.map { " · \($0)" } ?? "")
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
            ScrollView { composerInputs }
                .frame(minHeight: hasDetailedRunStatus ? 80 : (selectedJob != nil ? 120 : (studio.draft.assets.isEmpty ? 120 : 180)),
                       maxHeight: selectedJob == nil ? 280 : 200)
            generationControls
        }.padding(14).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(dropping ? ciderAccent : Color.primary.opacity(0.12), lineWidth: 1))
            .onDrop(of: [.fileURL, .image], isTargeted: $dropping) { providers in
                guard !assetControlsLocked, studio.supportsImageInputs else { return false }
                Task { await studio.importProviders(providers) }; return true
            }
    }
    private var composerInputs: some View {
        VStack(alignment: .leading, spacing: 10) {
            if store.busy {
                Label("当前任务使用已提交的参数；下方修改用于下一次生成。", systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if studio.supportsImageInputs || !studio.draft.assets.isEmpty {
            HStack {
                Button(action: chooseImages) { Label("添加图片", systemImage: "plus") }.disabled(!studio.supportsImageInputs || studio.draft.assets.count >= referenceImportLimit).accessibilityIdentifier("addImages")
                Button { Task { await studio.pasteImage() } } label: { Label("粘贴图片", systemImage: "doc.on.clipboard") }.disabled(!studio.supportsImageInputs || studio.draft.assets.count >= referenceImportLimit).accessibilityIdentifier("pasteImages")
                if studio.canUndoAssets { Button { studio.undoAssetChange() } label: { Image(systemName: "arrow.uturn.backward") }.help("撤销素材修改") }
                Spacer()
                if studio.importing { ProgressView().controlSize(.small) }
                Text("已添加 \(studio.draft.assets.count) / \(referenceImportLimit)").font(.caption).foregroundStyle(.secondary)
            }.disabled(assetControlsLocked)
            }
            if !studio.draft.assets.isEmpty { inputStrip }
            if studio.draft.assets.count > studio.draft.activeAssets.count {
                HStack {
                    Text("\(studio.draft.assets.count - studio.draft.activeAssets.count) 张图片已保留，未参与当前模式。")
                    if studio.draft.operation == "image.generate", model?.supports("image.edit") == true {
                        Button("用于图片编辑") { studio.changeOperation("image.edit") }.disabled(store.busy || assetControlsLocked)
                    }
                }.font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Text(studio.draft.activeAssets.isEmpty ? "画面描述" : "编辑指令").font(.subheadline.weight(.medium))
                Spacer()
                Text("⌘ Return 生成").font(.caption2).foregroundStyle(.secondary)
            }
            PromptEditor(text: $studio.draft.prompt, placeholder: promptPlaceholder, editable: !preparingSubmission) { Task { await studio.pasteImage() } }
                .frame(height: 78).accessibilityIdentifier("prompt")
            if studio.draft.modelID == "qwen-image-2.1" {
                HStack {
                    Menu("替换为 Qwen 2.1 示例提示词…") {
                        ForEach(Qwen21PromptExample.allCases, id: \.self) { example in
                            Button(example.title) { studio.applyQwen21Example(example) }
                                .disabled(studio.draft.assets.count < example.referenceCount)
                        }
                    }.disabled(studio.importing || store.busy || submitting)
                    Text("支持透明 PNG；蒙版作为有序参考图输入。")
                        .font(.caption2).foregroundStyle(.secondary)
                }

            }
            PromptCapacityView(studio: studio, busy: store.busy || submitting)
        }
    }
    private var generationControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("生成画布 \(studio.draft.width) × \(studio.draft.height)").font(.caption).monospacedDigit()
                    Text(studio.draft.upscaleAfterGeneration && model?.isVideo != true
                         ? "超分 ×\(studio.draft.upscaleVariant.scale) → \(studio.draft.width * studio.draft.upscaleVariant.scale) × \(studio.draft.height * studio.draft.upscaleVariant.scale)" : inputSummary).font(.caption2).foregroundStyle(.secondary)
                    Text(studio.draft.randomSeed ? "每次随机 · 本次 \(studio.lastSeed.map(String.init) ?? "待确定")" : "固定种子 \(studio.draft.seedText)").font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                if model?.isVideo != true {
                    Picker("生成后", selection: Binding(get: { studio.generationUpscaleVariant }, set: { studio.selectGenerationUpscale($0) })) {
                        Text("不超分").tag(Optional<UpscaleVariant>.none)
                        Text("超分 ×2").tag(Optional(UpscaleVariant.x2plus))
                        Text("超分 ×4").tag(Optional(UpscaleVariant.x4plus))
                    }
                    .labelsHidden().frame(width: 125)
                    .help("选择是否在生图完成后放大图片，原图始终保留。")
                    .disabled(store.busy || submitting || api.running || api.changing || studio.importing)
                    .accessibilityLabel("生成后超分")
                    .accessibilityIdentifier("generationUpscale")
                }
                Button(action: generate) { Label(store.busy ? "正在运行" : (model?.isVideo == true ? "生成视频" : "生成图像"), systemImage: "sparkles").padding(.horizontal, 8).padding(.vertical, 4) }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(store.busy || api.running || api.changing || submitting || studio.importing ||
                              model?.executor != true || !studio.selectedStreamingTargetAvailable || generationBlocker != nil)
                    .accessibilityIdentifier("generate")
            }
            if !store.busy, !submitting, let blocker = generationBlocker {
                HStack(alignment: .top) {
                    Label(blocker, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if studio.draft.modelPath.isEmpty { Button("选择模型") { page = .models }.font(.caption) }
                    else if studio.supportsImageInputs, studio.draft.operation != "image.generate", studio.draft.activeAssets.isEmpty {
                        Button("添加图片", action: chooseImages).font(.caption).disabled(assetControlsLocked)
                    }
                }.accessibilityIdentifier("generationRequirement")
            }
        }
    }
    private var promptEnhancementSettings: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("蒙版示例：参考 1 为原图，参考 2 为白色编辑区 / 黑色保留区蒙版。圈选示例直接使用带标注的图片。示例会替换提示词与参数，不是自动 prompt rewriting。")
                .font(.caption2).foregroundStyle(.secondary)
            Toggle("原生 PE 提示词增强（实验性）", isOn: $studio.draft.promptEnhance)
                .disabled(store.busy || submitting)
                .accessibilityIdentifier("qwen21PromptEnhance")
            if studio.draft.promptEnhance {
                if studio.draft.operation == "image.edit" {
                    Toggle("允许 PE-I2I FP32 视觉（未通过编辑质量验收）",
                           isOn: $studio.draft.promptEnhanceEditExperimental)
                        .disabled(store.busy || submitting)
                        .accessibilityIdentifier("qwen21PromptEnhanceEditExperimental")
                    Text("默认关闭。使用 PE-I2I 模型及 8 位参考图；PNG 已完成字节对齐，JPEG 解码仍有差异。编辑可能改变未指定区域或细节。")
                        .font(.caption2).foregroundStyle(.orange)
                }
                HStack {
                    TextField(studio.draft.operation == "image.edit" ? "PE-I2I 安装目录" : "PE-T2I 安装目录",
                              text: $studio.draft.promptEnhancerPath)
                    Button("选择…") {
                        let panel = NSOpenPanel()
                        panel.canChooseDirectories = true; panel.canChooseFiles = false
                        panel.allowsMultipleSelection = false
                        if panel.runModal() == .OK, let url = panel.url {
                            studio.draft.promptEnhancerPath = url.path
                        }
                    }
                }.disabled(store.busy || submitting)
                Text(studio.draft.operation == "image.edit"
                     ? "无 Python 推理；实验性 PE-I2I 可能耗时数十分钟。保留下方显式画布尺寸，不自动采用 PE 推荐比例。"
                     : "无 Python 推理；当前增强可能耗时数分钟。保留下方显式画布尺寸，不自动采用 PE 推荐比例。")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
    private var inputStrip: some View {
        ScrollView(.horizontal) { HStack(alignment: .top, spacing: 10) {
            ForEach(Array(studio.draft.assets.enumerated()), id: \.element.id) { index, asset in
                VStack(alignment: .leading, spacing: 4) {
                    ZStack(alignment: .topTrailing) {
                        inputAssetPreview(asset)
                            .frame(width: 104, height: 64).clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(studio.draft.activeAssets.contains(asset) ? ciderAccent : Color.primary.opacity(0.12), lineWidth: 2))
                        Button { studio.remove(asset.id) } label: {
                            Image(systemName: "xmark").font(.system(size: 9, weight: .bold))
                                .frame(width: 20, height: 20).background(.regularMaterial, in: Circle())
                        }.buttonStyle(.plain).padding(4).help("移除 \(asset.name)")
                            .accessibilityLabel("移除参考图 \(index + 1)").accessibilityIdentifier("removeReference-\(index)")
                    }
                    HStack(spacing: 4) {
                        Text(["image.transform", "video.image"].contains(studio.draft.operation) && studio.draft.initImageID == asset.id ? "原图 · 已选择" : "参考 \(index + 1)")
                            .font(.caption2.weight(.medium))
                        Spacer(minLength: 0)
                        Menu {
                            if studio.draft.modelID == "qwen-image-2.1" {
                                Button("圈选 / 涂抹 / 独立蒙版…") { annotationAsset = asset }
                                    .disabled(store.busy || assetControlsLocked)
                            }
                            Button(model?.supports("image.edit") == true && model?.supports("image.transform") != true ? "仅编辑这张" : "设为原图") { studio.useOnlyAssetForEditing(asset.id) }
                                .disabled(store.busy)
                            Divider()
                            Button("向前移动") { studio.move(asset.id, offset: -1) }.disabled(index == 0)
                            Button("向后移动") { studio.move(asset.id, offset: 1) }.disabled(index == studio.draft.assets.count - 1)
                            Button("移除", role: .destructive) { studio.remove(asset.id) }
                        } label: { Image(systemName: "ellipsis") }.menuStyle(.borderlessButton).frame(width: 18)
                            .accessibilityLabel("参考图 \(index + 1) 操作")
                    }
                    Text(asset.name).font(.caption2).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                    HStack(spacing: 4) {
                        Text(studio.draft.activeAssets.contains(asset) ? "参与生成" : "已保留").font(.caption2).foregroundStyle(studio.draft.activeAssets.contains(asset) ? ciderAccent : .secondary)
                        Spacer(minLength: 0)
                        Button { studio.move(asset.id, offset: -1) } label: { Image(systemName: "chevron.left") }
                            .disabled(index == 0).help("将参考 \(index + 1) 向前移动")
                            .accessibilityLabel("参考 \(index + 1) 向前移动")
                        Button { studio.move(asset.id, offset: 1) } label: { Image(systemName: "chevron.right") }
                            .disabled(index == studio.draft.assets.count - 1).help("将参考 \(index + 1) 向后移动")
                            .accessibilityLabel("参考 \(index + 1) 向后移动")
                    }.buttonStyle(.borderless).font(.caption2)
                }.frame(width: 104).help("\(asset.name) · \(asset.width) × \(asset.height)").disabled(assetControlsLocked)
            }
        }.padding(2) }.frame(height: 126)
    }
    @ViewBuilder private func inputAssetPreview(_ asset: StudioAsset) -> some View {
        if ["image.transform", "video.image"].contains(studio.draft.operation) {
            Button { studio.draft.initImageID = asset.id } label: { MediaPreview(path: asset.path, maxPixel: 208) }
                .buttonStyle(.plain).accessibilityLabel("\(asset.name)，点击选为原图")
        } else {
            MediaPreview(path: asset.path, maxPixel: 208).accessibilityLabel(asset.name)
        }
    }
    private var inspectorContents: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("生成参数").font(.headline)
            VStack(alignment: .leading, spacing: 8) {
                Picker("模型", selection: Binding(get: { studio.draft.modelID }, set: { studio.changeModel($0) })) {
                    ForEach(studio.creationModels) { item in
                        let downloaded = !(studio.draft.modelPaths[item.id] ?? "").isEmpty
                        Label("\(item.name) · \(downloaded ? "已配置" : "未配置")",
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
                    Button("重置") { studio.draft.steps = studio.draft.qwen21TurboLoRA != nil ? 6 : (model?.default_steps ?? 4) }
                }
                Text(studio.draft.qwen21TurboLoRA != nil ? "此 Viggle 适配器使用固定 6 步采样。"
                     : studio.draft.modelID.hasPrefix("z-image-turbo") ? "1–50 步，默认 \(model?.default_steps ?? 8) 步。其他步数的画质与加速收益需自行验证。" : "1–50 步，默认 \(model?.default_steps ?? 4) 步。")
                    .font(.caption2).foregroundStyle(.secondary)
                if studio.draft.modelID == "qwen-image-2.1", studio.draft.activeLoRAs.isEmpty, studio.draft.steps == 6 {
                    Button("基础模型改用 40 步") { studio.draft.steps = 40 }.font(.caption)
                    Text("当前未启用六步 LoRA；基础模型推荐使用 40 步。")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            Text("输出尺寸").font(.subheadline)
            HStack(alignment: .bottom) {
                VStack(alignment: .leading, spacing: 4) { Text("宽度").font(.caption2).foregroundStyle(.secondary); TextField("宽", value: $studio.draft.width, format: .number).accessibilityIdentifier("width") }
                Text("×").padding(.bottom, 5)
                VStack(alignment: .leading, spacing: 4) { Text("高度").font(.caption2).foregroundStyle(.secondary); TextField("高", value: $studio.draft.height, format: .number).accessibilityIdentifier("height") }
                Button { let width = studio.draft.width; studio.draft.width = studio.draft.height; studio.draft.height = width } label: { Image(systemName: "arrow.left.arrow.right") }
                    .disabled(studio.draft.width == studio.draft.height).help("交换宽高").accessibilityLabel("交换宽高").padding(.bottom, 3)
            }.textFieldStyle(.roundedBorder)
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
                if studio.draft.modelID == "qwen-image-2.1" {
                    Menu("官方 2K 比例（未完成验证）") {
                        ForEach(Qwen21CanvasPreset.recommended) { preset in
                            Button(preset.title) {
                                studio.draft.width = preset.width
                                studio.draft.height = preset.height
                            }
                        }
                    }.accessibilityIdentifier("qwen21CanvasPresets")
                    Text("当前回归测试聚焦 512 × 512。2K 内存占用高且未完成质量验证；实验性 GPU + ANE 仅支持 512 × 512。")
                        .font(.caption2).foregroundStyle(.secondary)
                }
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
                Toggle("额外启用 ANE", isOn: Binding(get: { studio.draft.usesANE && studio.draft.qwen21TurboLoRA == nil }, set: { studio.setANEEnabled($0) }))
                    .toggleStyle(.checkbox).disabled(store.busy || submitting || model?.supports_gpu_ane != true || studio.draft.zImageVariant?.id == "nvfp4" || studio.draft.qwen21TurboLoRA != nil).accessibilityIdentifier("enableANE")
                Text(studio.draft.qwen21TurboLoRA != nil ? "六步 LoRA 使用纯 GPU，基础模型的 ANE 分区不包含适配器。" : (studio.draft.usesANE ? "生成前检查匹配分区。ANE 不保证更快；首次加载较慢，长文本可优先使用 GPU。" : "默认只使用 GPU。")).font(.caption2).foregroundStyle(.secondary)
                if studio.draft.usesANE, studio.draft.qwen21TurboLoRA == nil { Button("管理 ANE 分区与缓存") { page = .models } }
                if studio.draft.usesANE, studio.draft.qwen21TurboLoRA == nil, let status = store.accelerationStatus { Text(status).font(.caption2).foregroundStyle(.secondary) }
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
                ForEach($studio.draft.loras) { $lora in
                    let index = studio.draft.loras.firstIndex(where: { $0.id == lora.id }) ?? 0
                    VStack(alignment: .leading, spacing: 6) {
                        Toggle(isOn: $lora.enabled) { Text(URL(fileURLWithPath: lora.path).lastPathComponent).font(.caption2).lineLimit(1).help(lora.path) }.toggleStyle(.checkbox).accessibilityIdentifier("loraEnabled-\(index)")
                        HStack {
                            Picker("角色", selection: $lora.role) {
                                Text("Transformer").tag("transformer")
                                if studio.draft.modelID.hasPrefix("flux2-") { Text("Text Encoder").tag("text_encoder") }
                                if studio.draft.modelID == "ltx-2.5-distilled" { Text("Refiner").tag("refiner") }
                            }.labelsHidden()
                            Text("强度").font(.caption2)
                            TextField("强度", value: $lora.strength, format: .number).textFieldStyle(.roundedBorder).frame(width: 64).disabled(!lora.enabled).accessibilityIdentifier("loraStrength-\(index)")
                            Button { studio.draft.loras.removeAll { $0.id == lora.id } } label: { Image(systemName: "trash") }
                                .help("移除 LoRA").accessibilityLabel("移除 \(URL(fileURLWithPath: lora.path).lastPathComponent)")
                        }
                    }
                }
                Text(studio.draft.modelID == "qwen-image-2.1" ? "六步 Viggle 使用单个适配器，强度固定为 1；关闭后保留文件。" : "强度默认 1.0；取消勾选会保留文件和强度。可输入 −8 到 8。").font(.caption2).foregroundStyle(.secondary)
                if studio.draft.qwen21TurboLoRA != nil {
                    Button("应用 512×512 / 6 步 GPU 预设") { studio.applyQwen21TurboPreset() }
                        .disabled(store.busy || assetControlsLocked)
                        .accessibilityIdentifier("qwen21TurboPreset")
                    Text("已识别六步 Viggle 适配器；预设保留提示词和参考图。")
                        .font(.caption2).foregroundStyle(.secondary)
                }
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
            if studio.draft.modelID == "qwen-image-2.1" {
                Divider()
                Text("提示词增强").font(.headline)
                promptEnhancementSettings
            }
            if model?.isVideo != true {
                Divider()
                UpscaleSettingsView(store: store, studio: studio,
                                    locked: store.busy || submitting || api.running || api.changing,
                                    showsVariantPicker: false, manageModels: { librarySelection = studio.draft.upscaleVariant.modelID; page = .models })
            }
            DisclosureGroup("高级参数") {
                VStack(alignment: .leading, spacing: 12) {
                    Toggle("动态文本长度", isOn: $studio.draft.dynamicText).controlSize(.small)
                    if studio.draft.publicStreamingModel {
                        Picker("流式内存档位", selection: Binding(
                            get: { studio.draft.streaming.selection },
                            set: { studio.setStreamingSelection($0) })) {
                            ForEach(StudioStreamingSelection.allCases) { value in
                                let option = studio.streamingOption(for: value)
                                let unavailable = value != .off && option?.status != "available"
                                Text(value.label + (option?.status == "candidate" ? "（待验证）" : unavailable ? "（不可用）" : option?.release_channel == "public-calibrated" ? "（已校准）" : ""))
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
                            if studio.draft.modelID == "z-image-turbo", option.status != "available" {
                                Button("恢复常驻加载，保留当前设置") { studio.useZImageResidentLoading() }
                                    .disabled(store.busy || submitting)
                                    .accessibilityIdentifier("recoverUnavailableStreamingTarget")
                            }
                            if option.release_channel == "public-calibrated" {
                                Text("此档位已针对匹配的模型、设备和任务校准内存用量；预算不是内存硬上限，也不保证没有交换或一定更快。")
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        } else if studio.draft.streaming.selection == .off {
                            let recommendation = studio.recommendedStreamingSelection
                            if studio.draft.modelID == "z-image-turbo", studio.draft.residency == "streamed" {
                                Text("当前使用下方的实验流式设置；未启用校准档位。")
                                    .font(.caption2).foregroundStyle(.secondary)
                            } else if recommendation != .off {
                                Text("当前可用档位：\(recommendation.label)；未选择档位时使用下方加载方式。")
                                    .font(.caption2).foregroundStyle(.secondary)
                            } else {
                                Text(studio.streamingOptions?.targets.contains(where: { $0.status == "candidate" }) == true
                                     ? "已有候选档位，请先选择本地模型完成验证。"
                                     : "暂无经过验证的流式档位，使用下方加载方式。")
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
                    if studio.draft.modelID == "z-image-turbo", let profile = StudioLocalStreamingProfile.bundled {
                        Button("应用本机实验流式配置（512×512 · 9 步）") {
                            studio.applyLocalStreamingProfile(profile)
                        }
                        .disabled(store.busy || submitting || profile.conflict(draft: studio.draft) != nil)
                        .accessibilityIdentifier("localExperimentalStreamingProfile")
                        Text(profile.conflict(draft: studio.draft)
                             ?? "仅本机实验：M4 Pro / 48 GiB、BF16、纯 GPU、无 LoRA。采样预算 10 GiB，不是应用内存上限；尚未通过正式容量认证。")
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
        }.disabled(assetControlsLocked)
    }
    @ViewBuilder private var runStatus: some View {
        if hasDetailedRunStatus {
            ScrollView(.vertical) {
                runStatusContents.frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
            }
            .frame(minHeight: 90, idealHeight: 120, maxHeight: 120)
            .layoutPriority(2)
        } else {
            runStatusContents
        }
    }
    private var runStatusContents: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let job = store.activeJob {
                HStack(alignment: .firstTextBaseline) {
                    Text(phaseName(job.phase)).font(.title3.weight(.semibold))
                    if job.phase == "denoise" { Text("\(job.completed) / \(job.total) 步已完成").font(.title2.weight(.semibold)).monospacedDigit().accessibilityIdentifier("denoiseStepProgress") }
                    Spacer()
                    Text(stateName(job.state)).font(.callout).foregroundStyle(.secondary)
                }
                Text(store.actualRoute.map { "实际路径：\($0)" } ?? "实际路径：准备后确认").font(.caption2).foregroundStyle(.secondary)
                if job.total > 0 {
                    ProgressView(value: Double(min(job.completed, job.total)), total: Double(job.total)).tint(ciderAccent)
                } else {
                    ProgressView().controlSize(.small)
                }
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
            } else if store.workerCleanupPending { HStack { Text(store.sessionState).font(.caption); Spacer(); Button("检查清理状态") { Task { await store.refreshWorkerCleanup() } } } }
            else if store.busy { HStack { ProgressView().controlSize(.small); Text(store.sessionState).font(.caption); Spacer(); Button("取消") { store.cancel() } } }
            else if submitting || store.resolvingAcceleration {
                HStack { ProgressView().controlSize(.small); Text(store.accelerationStatus ?? "正在准备生成请求…").font(.callout); Spacer() }
            }
            else if let job = store.jobs.first {
                Text("\(stateName(job.state)) · 共用时 \(String(format: "%.1f", job.elapsed)) 秒 · 种子 \(job.request.seed)").font(.caption).foregroundStyle(.secondary)
                if job.state == "failed" || job.state == "interrupted", let error = job.error {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Label(job.state == "failed" ? "生成失败" : "任务已中断", systemImage: "exclamationmark.triangle.fill").font(.callout.weight(.medium))
                            Spacer(minLength: 8)
                            Button("恢复这次参数") { reuseParameters(job) }.font(.caption).disabled(assetControlsLocked)
                        }
                        Text(error).font(.caption).textSelection(.enabled)
                    }.foregroundStyle(.red).padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.07), in: RoundedRectangle(cornerRadius: 8))
                        .accessibilityIdentifier("generationFailure")
                }
                if let json = job.resultJSON {
                    Text(RunInsights(json: json).cacheLabel).font(.callout).foregroundStyle(.secondary)
                }
            }
            if let conflict = studio.draft.zImageStreamingConflict() {
                VStack(alignment: .leading, spacing: 6) {
                    Text(conflict).font(.caption).textSelection(.enabled)
                    Button("切换常驻加载，保留当前设置") { studio.useZImageResidentLoading() }
                        .disabled(store.busy || submitting)
                        .accessibilityIdentifier("fixZImageStreamingConflict")
                    Text("常驻加载会使用更多内存。此问题与提示词内容或长度无关。")
                        .font(.caption2).foregroundStyle(.secondary)
                }.accessibilityIdentifier("zImageStreamingConflict")
            }
            if let message = studio.message {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "info.circle").foregroundStyle(ciderAccent)
                    Text(message).font(.caption).textSelection(.enabled)
                    Spacer(minLength: 0)
                    Button { studio.message = nil } label: { Image(systemName: "xmark") }
                        .buttonStyle(.plain).help("关闭提示").accessibilityLabel("关闭提示")
                }.padding(10).background(ciderAccent.opacity(0.07), in: RoundedRectangle(cornerRadius: 8))
                    .accessibilityIdentifier("statusMessage")
            }
            if let error = store.storageError {
                Label(error, systemImage: "externaldrive.badge.exclamationmark")
                    .font(.caption).foregroundStyle(.red).textSelection(.enabled)
                    .accessibilityIdentifier("storageError")
            }
        }
    }
    private var modelsPage: some View {
        ModelLibraryView(store: store, studio: studio, library: library, selection: $librarySelection, chooseModel: chooseModel,
                         loadModel: loadModel, importConfiguration: importConfiguration,
                         operationName: operationName)
            .disabled(preparingSubmission)
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
                        Button("复用参数") { reuseParameters(job) }.disabled(assetControlsLocked)
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
        selected = job.id; imageUpscaling = false; compareOriginal = false
        resultSelection.select(job.id, orderedIDs: outputIDs)
        page = .studio
    }
    private func reuseParameters(_ job: NativeJob) {
        guard !assetControlsLocked else { return }
        studio.reuse(job)
        imageUpscaling = false; compareOriginal = false; page = .studio
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
            do {
                try studio.importConfiguration(from: url)
                imageUpscaling = false; compareOriginal = false; page = .studio
            }
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
        if let conflict = studio.draft.zImageStreamingConflict() { studio.message = conflict; return }
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
        if let conflict = studio.draft.zImageStreamingConflict() { studio.message = conflict; return }
        let snapshot = studio.draft
        if snapshot.upscaleAfterGeneration && model?.isVideo != true {
            do { try ImageUpscaler.validateModelURL(URL(fileURLWithPath: snapshot.upscaleModelPath)) }
            catch { studio.message = error.localizedDescription; return }
        }
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
                if snapshot.upscaleAfterGeneration && ext == "png" {
                    let upscaled = try await store.upscale(source: output, modelURL: URL(fileURLWithPath: snapshot.upscaleModelPath), compute: snapshot.upscaleCompute)
                    if let info = store.upscaleReady { studio.rememberUpscaleModel(info) }
                    selected = upscaled.id; resultSelection.select(upscaled.id, orderedIDs: outputIDs)
                }
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
