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
    @AppStorage("playgroundInspectorVisible") private var inspectorVisible = true

    private var controlsLocked: Bool { state.importing || state.importTask != nil || submitting || store.busy || api.running || api.changing }
    private var referenceTitle: String { "\(state.template.title(for: state.template.primaryRole))参考" }
    private var executionLabel: String {
        state.settings.executionDeviceLabel
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
                HStack(spacing: 0) {
                    workspace
                    if inspectorVisible {
                        Divider()
                        ScrollView { configuration.padding(14) }
                            .frame(width: min(280, max(240, geometry.size.width * 0.32)))
                            .background(Color.primary.opacity(0.015))
                            .accessibilityIdentifier("playgroundInspector")
                    }
                }
            }
            Divider()
            footer
        }
        .onChange(of: state.template) { _, _ in selectedResultID = nil; showReference = false }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("playgroundWorkspace")
    }

    private var workspace: some View {
        GeometryReader { geometry in
            if geometry.size.width >= 760 {
                HStack(alignment: .top, spacing: 18) {
                    ScrollView { inputs.padding(.vertical, 14) }.frame(width: 280)
                    results.padding(.vertical, 14).frame(maxWidth: .infinity, maxHeight: .infinity)
                }.padding(.horizontal, 18)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        // Result metadata, actions and thumbnails need their
                        // own height in this scrolling column. A small fixed
                        // canvas alone would let them overlap the role cards.
                        results.frame(height: max(resultJobs.isEmpty ? 240 : 420,
                                                  min(460, geometry.size.height * 0.7)))
                        inputs
                    }.padding(16)
                }
            }
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Playground").font(.title2.weight(.semibold))
                Text("独立图片工作区").font(.caption).foregroundStyle(.secondary)
            }
            Picker("模板", selection: Binding(get: { state.template }, set: { state.selectTemplate($0) })) {
                ForEach(PlaygroundTemplate.allCases) { template in
                    Text(template.title).tag(template).accessibilityIdentifier("playgroundTemplate.\(template.rawValue)")
                }
            }.pickerStyle(.menu).labelsHidden().frame(width: 174).disabled(controlsLocked)
                .help("选择 Playground 模板")
                .accessibilityIdentifier("playgroundTemplatePicker")
            Spacer(minLength: 8)
            Text(state.saved ? "已保存" : "未保存").font(.caption).foregroundStyle(.secondary)
                .accessibilityIdentifier("playgroundSavedState")
            Button { inspectorVisible.toggle() } label: {
                Label(inspectorVisible ? "收起设置" : "生成设置", systemImage: "sidebar.right")
            }.help(inspectorVisible ? "收起生成设置" : "展开生成设置")
                .accessibilityIdentifier("playgroundToggleInspector")
                .accessibilityValue(inspectorVisible ? "已展开" : "已收起")
            Button("创作页", systemImage: "arrow.up.right") { showCreation() }
                .disabled((submitting && !store.busy) || state.importing || state.importTask != nil)
                .accessibilityIdentifier("playgroundShowCreation")
        }.padding(.horizontal, 18).padding(.vertical, 14)
    }

    private var inputs: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(state.template.detail).font(.callout).foregroundStyle(.secondary)
            Text("\(state.settings.width)×\(state.settings.height) · \(state.settings.steps) 步 · \(executionLabel)")
                .font(.caption).foregroundStyle(.secondary)
                .accessibilityIdentifier("playgroundConfigurationSummary")
            if state.template == .outpaint {
                VStack(alignment: .leading, spacing: 6) {
                    Text("扩图范围（试验）").font(.callout.weight(.semibold))
                    Picker("扩图倍率", selection: Binding(get: { state.outpaintExpansion }, set: { state.setOutpaintExpansion($0) })) {
                        Text("1.25×").tag(1.25)
                        Text("1.5×").tag(1.5)
                        Text("2×").tag(2.0)
                    }.pickerStyle(.segmented).disabled(controlsLocked)
                        .accessibilityIdentifier("playgroundOutpaintExpansion")
                    Text("倍率仅引导提示词，模型可能无法按该倍率扩展视野；原图细节也可能变化。实际输出尺寸仍使用当前画布设置。")
                        .font(.caption).foregroundStyle(.secondary)
                        .accessibilityIdentifier("playgroundOutpaintHint")
                }
            }
            if state.template == .transparent {
                Text("输出真正 RGBA PNG，效果取决于模型；可通过透明棋盘背景查看透明区域。")
                    .font(.caption).foregroundStyle(.secondary)
                    .accessibilityIdentifier("playgroundTransparencyHint")
            }
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
                Text(state.template.instructionTitle).font(.callout.weight(.semibold))
                TextEditor(text: Binding(get: { state.instruction }, set: { state.setInstruction($0) }))
                    .font(.body).frame(height: 82).scrollContentBackground(.hidden)
                    .padding(6).background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                    .disabled(controlsLocked).accessibilityIdentifier("playgroundInstruction")
                Text(state.template.referenceHint)
                    .font(.caption).foregroundStyle(.secondary)
                    .accessibilityIdentifier("playgroundReferenceHint")
            }
            DisclosureGroup("查看实际提示词", isExpanded: $showPrompt) {
                Text(state.prompt).font(.caption).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 6)
            }.font(.caption).accessibilityIdentifier("playgroundActualPrompt")
        }
    }

    private var configuration: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("生成设置").font(.headline)
            Text("仅应用于“\(state.template.title)”，自动保存；不会修改创作页。")
                .font(.caption).foregroundStyle(.secondary)
            modelSettings
            Divider()
            canvasSettings
            Divider()
            samplingSettings
            Divider()
            executionSettings
            Divider()
            loraSettings
            if state.settings.modelID == "qwen-image-2.1" {
                Divider()
                ditCacheSettings
                Divider()
                if state.referenceEncodingDraft.operation == "image.edit" {
                    Qwen21ReferenceEncodingSettings(draft: state.referenceEncodingDraft, locked: controlsLocked,
                        accessibilityPrefix: "playground", setSize: { state.setQwen21ReferenceSize($0) })
                } else {
                    Text("文生图不使用参考编码。").font(.caption2).foregroundStyle(.secondary)
                        .accessibilityIdentifier("playgroundReferenceEncodingNotApplicable")
                }
            }
            Button("从创作页同步模型与参数") { state.syncSettings(from: creator.draft) }
                .disabled(controlsLocked || creator.imageInputsBusy).accessibilityIdentifier("playgroundSyncSettings")
            Text("同步保留模板图片和指令；参数不兼容时会提示原因。")
                .font(.caption).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, alignment: .leading)
            .disabled(controlsLocked || !state.canEditSettings)
    }

    private func settingsBinding<Value>(_ keyPath: WritableKeyPath<StudioDraft, Value>) -> Binding<Value> {
        Binding(get: { state.settings[keyPath: keyPath] }, set: { value in
            guard !controlsLocked else { return }
            state.updateSettings { $0[keyPath: keyPath] = value }
        })
    }

    private var modelSettings: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("模型", selection: Binding(get: { state.settings.modelID }, set: { state.selectModel($0) })) {
                if state.settings.modelID != "qwen-image-2.1" {
                    Text("\(state.model?.displayName ?? state.settings.modelID)（不适用）").tag(state.settings.modelID)
                }
                ForEach(state.models.filter { $0.id == "qwen-image-2.1" && $0.supports("image.edit") }) { model in
                    Text(model.displayName).tag(model.id)
                }
            }.accessibilityIdentifier("playgroundModel")
            HStack {
                Text("本地模型目录").font(.caption)
                Spacer()
                Button("选择…") { chooseModelDirectory() }.accessibilityIdentifier("playgroundChooseModelDirectory")
            }
            TextField("Qwen-Image-2.1 目录", text: Binding(get: { state.settings.modelPath }, set: { path in
                guard !controlsLocked else { return }
                state.updateSettings { $0.modelPaths[$0.modelID] = path }
            }))
                .textFieldStyle(.roundedBorder).font(.caption).help(state.settings.modelPath)
                .accessibilityIdentifier("playgroundModelPath")
        }
    }

    private var canvasSettings: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("输出画布").font(.subheadline.weight(.medium))
                Spacer()
                Menu("常用比例") {
                    canvasPreset("1:1 · 512×512", width: 512, height: 512)
                    canvasPreset("2:3 · 512×768", width: 512, height: 768)
                    canvasPreset("3:2 · 768×512", width: 768, height: 512)
                    canvasPreset("1:1 · 768×768", width: 768, height: 768)
                    canvasPreset("1:1 · 1024×1024", width: 1024, height: 1024)
                }.accessibilityIdentifier("playgroundCanvasPresets")
            }
            HStack {
                TextField("宽", value: settingsBinding(\.width), format: .number)
                    .accessibilityIdentifier("playgroundWidth")
                Text("×").foregroundStyle(.secondary)
                TextField("高", value: settingsBinding(\.height), format: .number)
                    .accessibilityIdentifier("playgroundHeight")
            }.textFieldStyle(.roundedBorder)
            Text("新模板默认 512×512；可手动修改宽高。输出画布与参考图文件尺寸、模型编码尺寸分别设置。")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }
    private func canvasPreset(_ title: String, width: Int, height: Int) -> some View {
        Button(title) { state.updateSettings { $0.width = width; $0.height = height } }
            .accessibilityIdentifier("playgroundCanvasPreset.\(width)x\(height)")
    }

    private var samplingSettings: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("采样步数").font(.caption)
                TextField("步数", value: settingsBinding(\.steps), format: .number)
                    .textFieldStyle(.roundedBorder).accessibilityIdentifier("playgroundSteps")
                Menu("预设") {
                    Button("25 步") { state.updateSettings { $0.steps = 25 } }
                    Button("40 步") { state.updateSettings { $0.steps = 40 } }
                }
            }
            Text(state.settings.hasQwen21TurboAdapter ? "六步 Viggle：6 步、强度 1，可使用下方快速预设。" : "基础模型和普通 LoRA 建议 20–40 步；较多步数更慢。")
                .font(.caption2).foregroundStyle(.secondary)
            Toggle("随机 Seed", isOn: settingsBinding(\.randomSeed)).toggleStyle(.checkbox)
                .accessibilityIdentifier("playgroundRandomSeed")
            HStack {
                Text("Seed").font(.caption)
                TextField("42", text: settingsBinding(\.seedText)).textFieldStyle(.roundedBorder)
                    .disabled(state.settings.randomSeed).accessibilityIdentifier("playgroundSeed")
                Button {
                    state.updateSettings { $0.seedText = String(Int.random(in: 0...2147483647)); $0.randomSeed = false }
                } label: { Image(systemName: "dice") }.help("随机一次并固定种子")
                    .accessibilityIdentifier("playgroundRollSeed")
            }
            if let seed = state.current.lastSeed {
                Text("上次实际 Seed：\(seed)").font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
            }
        }
    }

    private var executionSettings: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack { Text("执行设备").font(.caption); Spacer(); Text(executionLabel).font(.caption.weight(.medium)) }
            if let notice = state.settings.aneConfigurationNotice {
                Text(notice).font(.caption2).foregroundStyle(.orange)
                    .accessibilityIdentifier("playgroundANEConfigurationNotice")
            }
            Picker("内存管理", selection: settingsBinding(\.residency)) {
                Text("分阶段加载（推荐）").tag("component_staged")
                Text("全部常驻").tag("resident")
                if !["component_staged", "resident"].contains(state.settings.residency) {
                    Text("原配置：\(state.settings.residency)").tag(state.settings.residency)
                }
            }.accessibilityIdentifier("playgroundResidency")
            Text("分阶段加载会在编码完成后释放 encoder，再加载 DiT，最后解码，降低峰值内存。")
                .font(.caption2).foregroundStyle(.secondary)
            if !state.settings.profilePath.isEmpty || (state.settings.acceleration?.policy ?? "gpu") != "gpu" {
                Text("同步的设备配置已保留。纯 GPU 支持当前快速参考编码与 DiT 缓存组合。")
                    .font(.caption2).foregroundStyle(.secondary)
                Button("使用纯 GPU") { state.updateSettings { $0.profilePath = ""; $0.acceleration = StudioAcceleration() } }
                    .accessibilityIdentifier("playgroundUseGPU")
            }
            if state.settings.upscaleAfterGeneration {
                Text("Playground 不执行生成后自动超分。")
                    .font(.caption2).foregroundStyle(.orange)
                Button("关闭自动超分") { state.updateSettings { $0.upscaleAfterGeneration = false } }
                    .accessibilityIdentifier("playgroundDisableAutoUpscale")
            }
            if state.settings.promptEnhance {
                Toggle("提示词增强（实验性）", isOn: settingsBinding(\.promptEnhance))
                    .toggleStyle(.checkbox).accessibilityIdentifier("playgroundPromptEnhance")
            }
            if state.settings.usesPublicStreaming {
                Text("已保留同步的流式卸载配置；部分加速选项需要关闭。")
                    .font(.caption2).foregroundStyle(.secondary)
                Button("关闭流式卸载") { state.updateSettings { $0.streaming = StudioStreamingState() } }
                    .accessibilityIdentifier("playgroundDisableStreaming")
            }
        }
    }

    private var loraSettings: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("LoRA").font(.subheadline.weight(.medium)); Spacer()
                Button("添加…") { chooseLoRA() }.disabled(state.settings.loras.count >= 8)
                    .accessibilityIdentifier("playgroundAddLoRA")
            }
            ForEach(state.settings.loras) { lora in loraRow(lora) }
            Text("普通 LoRA 强度默认 1，可输入 −8 到 8。取消勾选会保留文件；参数不兼容时会提示原因。")
                .font(.caption2).foregroundStyle(.secondary)
            if !state.settings.activeLoRAs.isEmpty && !state.settings.hasQwen21TurboAdapter &&
                (state.settings.width != 512 || state.settings.height != 512) {
                Text("普通 LoRA 当前需要 512×512 输出；恢复该画布或关闭 LoRA 后可生成。")
                    .font(.caption2).foregroundStyle(.orange)
                Button("恢复 512×512 画布") { state.updateSettings { $0.width = 512; $0.height = 512 } }
                    .accessibilityIdentifier("playgroundRestoreLoRACanvas")
            }
            if state.settings.qwen21TurboLoRA != nil {
                if let issue = state.settings.qwen21TurboConfigurationIssues.first {
                    Text(issue).font(.caption2).foregroundStyle(.orange)
                        .accessibilityIdentifier("playgroundTurboUnavailable")
                }
                Button("应用 512×512 / 6 步 GPU 预设") { state.applyTurboPreset() }
                    .accessibilityIdentifier("playgroundTurboPreset")
            }
            Picker("LoRA 策略", selection: settingsBinding(\.loraStrategy)) {
                Text("自动").tag("auto")
                Text("运行时加载").tag("inference_time")
                if !["auto", "inference_time"].contains(state.settings.loraStrategy) {
                    Text("\(state.settings.loraStrategy)（不适用）").tag(state.settings.loraStrategy)
                }
            }.accessibilityIdentifier("playgroundLoRAStrategy")
        }
    }
    private func loraBinding<Value>(_ id: UUID, _ keyPath: WritableKeyPath<StudioLoRA, Value>, fallback: Value) -> Binding<Value> {
        Binding(get: { state.settings.loras.first(where: { $0.id == id })?[keyPath: keyPath] ?? fallback }, set: { value in
            guard !controlsLocked else { return }
            state.updateLoRA(id) { $0[keyPath: keyPath] = value }
        })
    }
    private func loraRow(_ lora: StudioLoRA) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle(isOn: loraBinding(lora.id, \.enabled, fallback: false)) {
                Text(URL(fileURLWithPath: lora.path).lastPathComponent).font(.caption).lineLimit(2).help(lora.path)
            }.toggleStyle(.checkbox).accessibilityIdentifier("playgroundLoRAEnabled.\(lora.id)")
            HStack {
                Text("强度").font(.caption2)
                TextField("强度", value: loraBinding(lora.id, \.strength, fallback: 1), format: .number)
                    .textFieldStyle(.roundedBorder).disabled(!lora.enabled)
                    .accessibilityIdentifier("playgroundLoRAStrength.\(lora.id)")
                Button { state.removeLoRA(lora.id) } label: { Image(systemName: "trash") }
                    .help("移除 LoRA").accessibilityIdentifier("playgroundRemoveLoRA.\(lora.id)")
            }
            if lora.role != "transformer" {
                Button("将角色设为 Transformer") { state.updateLoRA(lora.id) { $0.role = "transformer" } }
                    .font(.caption).accessibilityIdentifier("playgroundLoRARole.\(lora.id)")
            }
        }.padding(8).background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
    }

    private var ditCacheSettings: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("DiT 缓存（实验性）").font(.subheadline.weight(.medium))
            Picker("缓存档位", selection: settingsBinding(\.qwen21DiTCache)) {
                ForEach(Qwen21DiTCacheMode.allCases) { mode in Text(mode.title).tag(mode.rawValue) }
            }.disabled(state.referenceEncodingDraft.qwen21DiTCacheUnavailableReason != nil)
                .accessibilityIdentifier("playgroundDiTCacheMode")
            if let reason = state.referenceEncodingDraft.qwen21DiTCacheUnavailableReason {
                Text(reason).font(.caption2).foregroundStyle(.secondary).accessibilityIdentifier("playgroundDiTCacheUnavailable")
            }
            if let mode = Qwen21DiTCacheMode(rawValue: state.settings.qwen21DiTCache) {
                Text(mode.detail).font(.caption2).foregroundStyle(.secondary)
            }
            if state.settings.qwen21DiTCache != "off" {
                Button("关闭 DiT 缓存") { state.updateSettings { $0.qwen21DiTCache = "off" } }
                    .font(.caption).accessibilityIdentifier("playgroundDisableDiTCache")
            }
            Text("近似复用采样中间层，可能改变细节；默认关闭。与编辑参考图前缀缓存不同。")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func chooseModelDirectory() {
        guard !controlsLocked, state.canEditSettings else { return }
        let template = state.template
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = "选择本机 Qwen-Image-2.1 模型目录。不会下载模型。"
        guard panel.runModal() == .OK, let url = panel.url, !controlsLocked, state.template == template else { return }
        state.updateSettings { $0.modelPaths[$0.modelID] = url.path }
    }
    private func chooseLoRA() {
        guard !controlsLocked, state.canEditSettings else { return }
        let template = state.template
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
        panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        panel.message = "选择本机 LoRA 文件；六步 Viggle 可通过预设应用其采样参数。"
        guard panel.runModal() == .OK, let url = panel.url, !controlsLocked, state.template == template else { return }
        state.addLoRA(url.path)
    }

    private func roleCard(_ role: PlaygroundRole) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(state.template.title(for: role)).font(.callout.weight(.semibold))
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
                    .accessibilityIdentifier("playgroundRoleImage.\(role.rawValue)")
                VStack(alignment: .leading, spacing: 6) {
                    if let asset = state.asset(for: role) {
                        Text(asset.name).font(.caption).lineLimit(2)
                        Text("输入 \(asset.width)×\(asset.height) · 原图 \(asset.originalImage.width)×\(asset.originalImage.height)")
                            .font(.caption2).foregroundStyle(.secondary)
                        HStack(spacing: 10) {
                            Button("替换") { chooseFiles(replacing: role) }
                                .accessibilityIdentifier("playgroundReplace.\(role.rawValue)")
                            Button("移除") { state.remove(role) }
                                .accessibilityIdentifier("playgroundRemove.\(role.rawValue)")
                        }.font(.caption).disabled(controlsLocked)
                    } else {
                        Text(state.template.detail(for: role)).font(.caption).foregroundStyle(.secondary)
                        Button("选择图片") { chooseFiles(replacing: role) }.font(.caption).disabled(controlsLocked)
                            .accessibilityIdentifier("playgroundChoose.\(role.rawValue)")
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
            }.accessibilityElement(children: .contain)
            .accessibilityIdentifier("playgroundRole.\(role.rawValue)")
    }

    private var results: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(showReference || selectedJob == nil ? (state.previewAsset == nil ? "生成预览" : referenceTitle) : "生成结果")
                    .font(.callout.weight(.semibold))
                Spacer()
                if selectedJob != nil, state.previewAsset != nil {
                    Button(showReference ? "查看生成结果" : "查看\(referenceTitle)") { showReference.toggle() }
                        .font(.caption).accessibilityIdentifier("playgroundPreviewSource")
                }
            }
            ZStack {
                if !showReference, let job = selectedJob {
                    InteractiveMediaPreview(path: job.request.output)
                } else if let asset = state.previewAsset {
                    InteractiveMediaPreview(path: asset.path)
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "photo.on.rectangle").font(.largeTitle).foregroundStyle(.secondary)
                        Text(state.template == .transparent ? "输入描述生成透明图片，或添加原图提取主体。"
                             : "添加\(referenceTitle)，开始\(state.template.title)")
                            .font(.callout).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }.frame(minHeight: 180, maxHeight: .infinity)
                .background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 12))
                .clipShape(RoundedRectangle(cornerRadius: 12)).accessibilityIdentifier("playgroundCanvas")
            if !showReference, let job = selectedJob {
                Text("\(job.request.width)×\(job.request.height) · Seed \(job.request.seed) · \(Int(job.elapsed)) 秒")
                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                if job.request.model == "qwen-image-2.1", job.request.operation == "image.edit" {
                    Text("模型参考编码：\(job.request.qwen21_reference_size ?? 1024)\((job.request.qwen21_reference_size ?? 1024) == 512 ? " · 近似" : " · 标准")")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                HStack(spacing: 10) {
                    Button("用作\(referenceTitle)") {
                        guard !controlsLocked else { return }
                        if let task = state.startImport({ state in await state.useResultAsReference(job) }) {
                            Task {
                                if await task.value, state.template.workflowID == job.workflowID { showReference = true }
                            }
                        }
                    }.disabled(controlsLocked).accessibilityIdentifier("playgroundUseResultAsReference")
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
        panel.message = role.map { "选择\(template.title(for: $0))参考图；替换不会覆盖历史输入。" }
            ?? "按\(template.roles.map { template.title(for: $0) }.joined(separator: "、"))的空槽顺序添加图片。"
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
                    request: pair.legacy, streamingRequest: pair.v2,
                    runtimeOptions: try resolved.runtimeOptions(store: store.directory), workflowID: workflow.workflowID,
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
