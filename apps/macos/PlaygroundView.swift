import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct PlaygroundView: View {
    @ObservedObject var state: PlaygroundState
    @ObservedObject var creator: StudioState
    @ObservedObject var store: NativeJobStore
    @ObservedObject var api: LocalAPIController
    @Binding var submitting: Bool
    let showCreation: () -> Void
    let continueInCreation: (NativeJob) -> Void
    @State private var selectedResultID: UUID?
    @State private var showReference = false
    @State private var showPrompt = false

    private var controlsLocked: Bool { state.importing || state.importTask != nil || submitting || store.busy || api.running || api.changing }
    private var executionLabel: String {
        if !state.settings.profilePath.isEmpty || state.settings.acceleration?.policy == "profile" { return "设备配置" }
        return state.settings.usesANE ? "GPU + Core ML" : "GPU"
    }
    private var workflowJobs: [NativeJob] {
        store.jobs.filter { $0.workflowID == state.template.workflowID }.sorted { $0.createdAt > $1.createdAt }
    }
    private var resultJobs: [NativeJob] { workflowJobs.filter(\.hasOutput) }
    private var selectedJob: NativeJob? {
        if let id = selectedResultID, let job = resultJobs.first(where: { $0.id == id }) { return job }
        if let id = state.current.lastJobID, let job = resultJobs.first(where: { $0.id == id }) { return job }
        return resultJobs.first
    }
    private var ownedActiveJob: NativeJob? {
        guard let job = store.activeJob, job.workflowID == state.template.workflowID else { return nil }
        return job
    }
    private var submitBlocker: String? {
        if store.requiresProcessRestart { return "引擎需要重启 App 后才能继续生成。Playground 草稿已保留。" }
        if let error = store.storageError { return "任务历史保存失败：\(error)" }
        if api.running || api.changing { return "本地 API 正在占用引擎，请先停止 API 服务。" }
        return state.generationBlocker
    }
    private var statusText: String? {
        if let job = ownedActiveJob {
            return "\(state.template.title) · \(job.phase) · \(job.completed)/\(job.total) · \(Int(job.elapsed)) 秒"
        }
        if submitting { return "正在检查参数与执行资源…" }
        if state.importing || state.importTask != nil { return "正在处理参考图，原始副本与之前的任务输入会保留…" }
        if store.workerCleanupPending { return "正在等待工作进程清理，完成后可继续生成。" }
        if store.busy { return "引擎正在执行其他任务，完成后可生成。" }
        return state.storageError ?? submitBlocker ?? state.message
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            GeometryReader { geometry in
                if geometry.size.width >= 700 {
                    HStack(alignment: .top, spacing: 18) {
                        ScrollView { inputs.padding(.vertical, 14) }.frame(width: 300)
                        results.padding(.vertical, 14).frame(maxWidth: .infinity, maxHeight: .infinity)
                    }.padding(.horizontal, 18)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 18) {
                            inputs
                            results.frame(height: 380)
                        }.padding(16)
                    }
                }
            }
            Divider()
            footer
        }
        .onChange(of: state.template) { _, _ in selectedResultID = nil; showReference = false }
        .accessibilityIdentifier("playgroundWorkspace")
    }

    private var header: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Playground").font(.title2.weight(.semibold))
                Text("独立图片编辑工作区").font(.caption).foregroundStyle(.secondary)
            }
            Picker("模板", selection: Binding(get: { state.template }, set: { state.selectTemplate($0) })) {
                ForEach(PlaygroundTemplate.allCases) { template in Text(template.title).tag(template) }
            }.pickerStyle(.segmented).frame(maxWidth: 250).disabled(controlsLocked)
                .accessibilityIdentifier("playgroundTemplatePicker")
            Spacer(minLength: 8)
            Text(state.saved ? "已保存" : "未保存").font(.caption).foregroundStyle(.secondary)
                .accessibilityIdentifier("playgroundSavedState")
            Button("创作页", systemImage: "arrow.up.right") { showCreation() }
                .disabled((submitting && !store.busy) || state.importing || state.importTask != nil)
                .accessibilityIdentifier("playgroundShowCreation")
        }.padding(.horizontal, 18).padding(.vertical, 14)
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(state.template.detail).font(.callout).foregroundStyle(.secondary)
            configuration
            ForEach(state.template.roles) { role in roleCard(role) }
            Button { chooseFiles(replacing: nil) } label: {
                Label("添加图片，或拖到这里按空槽顺序填写", systemImage: "plus.rectangle.on.rectangle")
                    .font(.caption).frame(maxWidth: .infinity).padding(12)
                    .background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 9))
                    .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(Color.secondary.opacity(0.3), style: StrokeStyle(lineWidth: 1, dash: [4])))
            }.buttonStyle(.plain).disabled(controlsLocked)
                .onDrop(of: [UTType.fileURL.identifier, UTType.image.identifier], isTargeted: nil) { providers in
                    acceptDrop(providers, replacing: nil)
                }.accessibilityIdentifier("playgroundReferenceGroup")
            VStack(alignment: .leading, spacing: 6) {
                Text("附加指令").font(.callout.weight(.semibold))
                TextEditor(text: Binding(get: { state.instruction }, set: { state.setInstruction($0) }))
                    .font(.body).frame(height: 82).scrollContentBackground(.hidden)
                    .padding(6).background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                    .disabled(controlsLocked).accessibilityIdentifier("playgroundInstruction")
                Text("人物始终为 <image1>；服装或场景为 <image2>。参考图是视觉引导。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            DisclosureGroup("查看实际提示词", isExpanded: $showPrompt) {
                Text(state.prompt).font(.caption).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 6)
            }.font(.caption).accessibilityIdentifier("playgroundActualPrompt")
        }
    }

    private var configuration: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(state.model?.displayName ?? state.settings.modelID).font(.callout.weight(.semibold))
            Text("\(state.settings.width)×\(state.settings.height) · \(state.settings.steps) 步 · \(executionLabel)")
                .font(.caption).foregroundStyle(.secondary)
            Text(state.settings.randomSeed ? "Seed：随机" : "Seed：\(state.settings.seedText)")
                .font(.caption).foregroundStyle(.secondary)
            if let seed = state.current.lastSeed {
                Text("上次实际 Seed：\(seed)").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            if !state.settings.activeLoRAs.isEmpty {
                Text("LoRA：\(state.settings.activeLoRAs.count) 个\(state.settings.hasQwen21TurboAdapter ? " · 六步 Viggle" : "")")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(state.settings.activeLoRAs) { lora in
                    Text("\(URL(fileURLWithPath: lora.path).deletingPathExtension().lastPathComponent) · 强度 \(lora.strength, specifier: "%.2f")")
                        .font(.caption2).foregroundStyle(.secondary).lineLimit(2).help(lora.path)
                }
            }
            if state.settings.modelID == "qwen-image-2.1" {
                Text("DiT 缓存：\(Qwen21DiTCacheMode(rawValue: state.settings.qwen21DiTCache)?.title ?? state.settings.qwen21DiTCache)")
                    .font(.caption).foregroundStyle(.secondary)
                Qwen21ReferenceEncodingSettings(draft: state.referenceEncodingDraft, locked: controlsLocked,
                    accessibilityPrefix: "playground", setSize: { state.setQwen21ReferenceSize($0) })
            }
            Button("从创作页同步模型与参数") { state.syncSettings(from: creator.draft) }
                .disabled(controlsLocked || creator.imageInputsBusy).accessibilityIdentifier("playgroundSyncSettings")
            Text("同步保留模板图片和指令；参数不兼容时会提示原因。")
                .font(.caption).foregroundStyle(.secondary)
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 10))
    }

    private func roleCard(_ role: PlaygroundRole) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(role.title).font(.callout.weight(.semibold))
                Text(state.template.requiredRoles.contains(role) ? "必填" : "可选").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if let number = state.imageNumber(for: role) {
                    Text("<image\(number)>").font(.caption.monospaced()).foregroundStyle(.secondary)
                }
            }
            HStack(alignment: .center, spacing: 10) {
                Button { chooseFiles(replacing: role) } label: {
                    if let asset = state.asset(for: role) {
                        MediaPreview(path: asset.path, maxPixel: 180).frame(width: 78, height: 76)
                    } else {
                        Image(systemName: role == .person ? "person.crop.rectangle" : "photo.badge.plus")
                            .font(.title2).foregroundStyle(.secondary).frame(width: 78, height: 76)
                    }
                }.buttonStyle(.plain).background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                    .clipShape(RoundedRectangle(cornerRadius: 8)).disabled(controlsLocked)
                VStack(alignment: .leading, spacing: 6) {
                    if let asset = state.asset(for: role) {
                        Text(asset.name).font(.caption).lineLimit(2)
                        Text("输入 \(asset.width)×\(asset.height) · 原图 \(asset.originalImage.width)×\(asset.originalImage.height)")
                            .font(.caption2).foregroundStyle(.secondary)
                        HStack(spacing: 10) {
                            Button("替换") { chooseFiles(replacing: role) }
                            Button("移除") { state.remove(role) }
                        }.font(.caption).disabled(controlsLocked)
                    } else {
                        Text(role.detail).font(.caption).foregroundStyle(.secondary)
                        Button("选择图片") { chooseFiles(replacing: role) }.font(.caption).disabled(controlsLocked)
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            if let asset = state.asset(for: role) {
                Menu("输入文件尺寸 · \((asset.preparation ?? .original).title)") {
                    ForEach(ReferenceImagePreparation.allCases) { preset in
                        Button(preset.title) {
                            guard !controlsLocked else { return }
                            state.startImport { state in await state.prepare(role, preset: preset) }
                        }
                    }
                }.font(.caption).disabled(controlsLocked)
                    .accessibilityIdentifier("playgroundPrepare.\(role.rawValue)")
            }
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.primary.opacity(0.1), lineWidth: 1))
            .onDrop(of: [UTType.fileURL.identifier, UTType.image.identifier], isTargeted: nil) { providers in
                acceptDrop(providers, replacing: role)
            }.accessibilityIdentifier("playgroundRole.\(role.rawValue)")
    }

    private var results: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(showReference || selectedJob == nil ? "人物参考" : "生成结果").font(.callout.weight(.semibold))
                Spacer()
                if selectedJob != nil, state.asset(for: .person) != nil {
                    Button(showReference ? "查看生成结果" : "查看人物参考") { showReference.toggle() }
                        .font(.caption).accessibilityIdentifier("playgroundPreviewSource")
                }
            }
            ZStack {
                if !showReference, let job = selectedJob {
                    InteractiveMediaPreview(path: job.request.output)
                } else if let asset = state.asset(for: .person) {
                    InteractiveMediaPreview(path: asset.path)
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "person.crop.rectangle").font(.largeTitle).foregroundStyle(.secondary)
                        Text("添加人物参考，开始\(state.template.title)").font(.callout).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }.frame(minHeight: 180, maxHeight: .infinity)
                .background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 12))
                .clipShape(RoundedRectangle(cornerRadius: 12)).accessibilityIdentifier("playgroundCanvas")
            if !showReference, let job = selectedJob {
                Text("\(job.request.width)×\(job.request.height) · Seed \(job.request.seed) · \(Int(job.elapsed)) 秒")
                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                if job.request.model == "qwen-image-2.1" {
                    Text("模型参考编码：\(job.request.qwen21_reference_size ?? 1024)\((job.request.qwen21_reference_size ?? 1024) == 512 ? " · 近似" : " · 标准")")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                HStack(spacing: 10) {
                    Button("用作人物参考") {
                        guard !controlsLocked else { return }
                        if let task = state.startImport({ state in await state.useResultAsPerson(job) }) {
                            Task {
                                if await task.value, state.template.workflowID == job.workflowID { showReference = true }
                            }
                        }
                    }.disabled(controlsLocked).accessibilityIdentifier("playgroundUseResultAsPerson")
                    Button("在创作中继续", systemImage: "arrow.up.right") { continueInCreation(job) }
                        .disabled(controlsLocked || creator.imageInputsBusy).accessibilityIdentifier("playgroundContinueInCreation")
                }.font(.caption)
            }
            if !resultJobs.isEmpty {
                Text("\(state.template.title)历史 · 点击只切换预览").font(.caption).foregroundStyle(.secondary)
                ScrollView(.horizontal) {
                    LazyHStack(spacing: 8) {
                        ForEach(resultJobs) { job in
                            Button { selectedResultID = job.id; showReference = false } label: {
                                MediaPreview(path: job.request.output, maxPixel: 160).frame(width: 56, height: 56)
                                    .clipShape(RoundedRectangle(cornerRadius: 7))
                                    .overlay(RoundedRectangle(cornerRadius: 7).stroke(selectedJob?.id == job.id ? Color.accentColor : Color.secondary.opacity(0.2), lineWidth: selectedJob?.id == job.id ? 2 : 1))
                            }.buttonStyle(.plain).help("Seed \(job.request.seed) · \(job.request.width)×\(job.request.height)")
                                .accessibilityLabel("\(state.template.title)结果，Seed \(job.request.seed)")
                        }
                    }.padding(2)
                }.frame(height: 62).accessibilityIdentifier("playgroundResultHistory")
            }
        }
    }

    private var footer: some View {
        HStack(alignment: .center, spacing: 14) {
            VStack(alignment: .leading, spacing: 5) {
                if let text = statusText {
                    ScrollView {
                        Text(text).font(.caption).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    }.frame(height: 40).accessibilityIdentifier("playgroundStatus")
                }
                if let job = ownedActiveJob {
                    ProgressView(value: Double(job.completed), total: Double(max(1, job.total)))
                }
                if state.storageError != nil {
                    HStack {
                        Button("重新读取保存文件") { state.reload() }
                        Button("重试保存") { state.save() }
                        Button("显示保存文件") { NSWorkspace.shared.activateFileViewerSelecting([state.fileURL]) }
                    }.font(.caption).disabled(controlsLocked)
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
            if state.importTask != nil {
                Button("取消图片处理", role: .destructive) { state.cancelImport() }
                    .accessibilityIdentifier("playgroundCancelImport")
            }
            if ownedActiveJob != nil || (state.generationTask != nil && submitting) {
                Button("取消生成", role: .destructive) {
                    state.generationTask?.cancel()
                    if ownedActiveJob != nil || state.generationOwnsStore { store.cancel() }
                }.accessibilityIdentifier("playgroundCancelGeneration")
            }
            Button("生成 · \(state.template.title)", systemImage: "sparkles") { generate() }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(controlsLocked || submitBlocker != nil || creator.imageInputsBusy)
                .accessibilityIdentifier("playgroundGenerate")
        }.padding(.horizontal, 18).padding(.vertical, 10)
    }

    private func chooseFiles(replacing role: PlaygroundRole?) {
        guard !controlsLocked else { return }
        let template = state.template
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]; panel.canChooseDirectories = false; panel.allowsMultipleSelection = role == nil
        panel.message = role.map { "选择\($0.title)参考图；替换不会覆盖历史输入。" } ?? "按人物、服装或场景的空槽顺序添加图片。"
        guard panel.runModal() == .OK else { return }
        let urls = panel.urls
        guard !controlsLocked, state.template == template else { return }
        state.startImport { state in await state.importFiles(urls, replacing: role) }
    }
    private func acceptDrop(_ providers: [NSItemProvider], replacing role: PlaygroundRole?) -> Bool {
        guard !controlsLocked, !providers.isEmpty else { return false }
        return state.startImport { state in await state.importProviders(providers, replacing: role) } != nil
    }
    private func generate() {
        guard !controlsLocked, !creator.imageInputsBusy, submitBlocker == nil else { return }
        let snapshot: StudioDraft
        do {
            state.save()
            snapshot = try state.generationDraft()
        } catch { state.message = error.localizedDescription; return }
        let workflow = state.template
        let output = store.directory.appendingPathComponent("outputs/\(UUID().uuidString).png")
        submitting = true; state.message = nil
        state.generationTask = Task {
            defer { submitting = false; state.generationTask = nil; state.generationOwnsStore = false }
            do {
                let resolved = try await store.resolveAcceleration(snapshot)
                try Task.checkCancellation()
                let requestTask = Task.detached { try resolved.publicStreamingRequest(output: output) }
                let pair = try await withTaskCancellationHandler {
                    try await requestTask.value
                } onCancel: { requestTask.cancel() }
                try Task.checkCancellation()
                guard !api.running, !api.changing, !store.busy else { throw NativeFailure(message: "引擎状态已改变，请稍后重新生成。") }
                state.recordSubmission(resolved, seed: pair.legacy.seed, template: workflow)
                guard state.storageError == nil else { throw NativeFailure(message: state.storageError ?? "Playground 保存失败。") }
                try Task.checkCancellation()
                state.generationOwnsStore = true
                let job = try await store.generate(modelURL: URL(fileURLWithPath: resolved.modelPath),
                    request: pair.legacy, streamingRequest: pair.v2, workflowID: workflow.workflowID,
                    inputAssets: resolved.activeAssets)
                state.recordResult(job, template: workflow)
                if state.template == workflow { selectedResultID = job.id; showReference = false }
            } catch {
                state.message = error is CancellationError ? "生成已取消，模板参数与参考图已保留，可重新生成。"
                    : "\(error.localizedDescription)\n模板参数与参考图已保留，可调整后重新生成。"
            }
        }
    }
}
