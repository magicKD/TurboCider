import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Local metadata management. Browsing and importing never switch the draft's
/// base model; only the explicitly labelled activation buttons do so.
@MainActor struct ModelLoRALibraryView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    @ObservedObject var library: ModelLibraryController
    @State private var modelFilter = ""
    @State private var query = ""
    @State private var notice: String?
    @State private var refreshID = 0
    @State private var checkingFiles = false
    @State private var fileStates: [String: LoRALocalFileState] = [:]

    private var configurationLocked: Bool { store.busy || studio.imageInputsBusy }
    private var supportedModels: [StudioModel] {
        studio.models.filter { $0.supports_lora == true }.sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
    }
    private var filterIDs: [String] {
        Set(supportedModels.map(\.id) + library.loras.map(\.modelID)).sorted { modelTitle($0).localizedStandardCompare(modelTitle($1)) == .orderedAscending }
    }
    private func modelTitle(_ id: String) -> String { studio.models.first { $0.id == id }?.displayName ?? id }
    private func normalized(_ path: String) -> String { URL(fileURLWithPath: path).standardizedFileURL.path }
    private func sameFile(_ first: String, _ second: String) -> Bool {
        if normalized(first) == normalized(second) { return true }
        guard let a = fileStates[first]?.resolvedPath, let b = fileStates[second]?.resolvedPath else { return false }
        return a == b
    }
    private var allRows: [LoRALibraryRow] {
        var result = library.loras.map { LoRALibraryRow(id: "registered-" + $0.id, modelID: $0.modelID, name: $0.name, path: $0.path, registration: $0) }
        for lora in studio.draft.loras where !result.contains(where: { $0.modelID == studio.draft.modelID && sameFile($0.path, lora.path) }) {
            result.append(.init(id: "draft-" + lora.id.uuidString, modelID: studio.draft.modelID,
                name: URL(fileURLWithPath: lora.path).lastPathComponent, path: lora.path, registration: nil))
        }
        return result.sorted {
            if $0.modelID != $1.modelID { return modelTitle($0.modelID).localizedStandardCompare(modelTitle($1.modelID)) == .orderedAscending }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
    }
    private var rows: [LoRALibraryRow] {
        let words = query.split(whereSeparator: \.isWhitespace).map(String.init)
        return allRows.filter { item in
            (modelFilter.isEmpty || item.modelID == modelFilter) && words.allSatisfy { word in
                [item.name, item.path, item.modelID, modelTitle(item.modelID)].contains { $0.localizedCaseInsensitiveContains(word) }
            }
        }
    }
    private var pathsToCheck: [String] {
        Array(Set(library.loras.map(\.path) + studio.draft.loras.map(\.path) + studio.draft.modelLoRAs.values.flatMap { $0.map(\.path) })).sorted()
    }
    private var fileCheckKey: String { library.root + "\n" + String(refreshID) + "\n" + pathsToCheck.joined(separator: "\n") }
    private func selectedLoRA(_ row: LoRALibraryRow) -> StudioLoRA? {
        guard row.modelID == studio.draft.modelID else { return nil }
        return studio.draft.loras.first { sameFile($0.path, row.path) }
    }
    private func fileReady(_ row: LoRALibraryRow) -> Bool { !checkingFiles && fileStates[row.path]?.readable == true }
    private func supportsActivation(_ row: LoRALibraryRow) -> Bool {
        guard let model = studio.models.first(where: { $0.id == row.modelID }) else { return false }
        return model.executor && model.supports_lora == true && model.runtime_lora != false
    }
    private func replacesQwenLoRA(_ row: LoRALibraryRow) -> Bool {
        guard row.modelID == "qwen-image-2.1" else { return false }
        let choices = studio.draft.modelID == row.modelID ? studio.draft.loras : studio.draft.modelLoRAs[row.modelID] ?? []
        return choices.contains { $0.enabled && !sameFile($0.path, row.path) }
    }
    private func activationTitle(_ row: LoRALibraryRow) -> String {
        if studio.draft.modelID != row.modelID {
            return "切换到 \(modelTitle(row.modelID)) 并" + (replacesQwenLoRA(row) ? "替换启用" : "启用")
        }
        return replacesQwenLoRA(row) ? "替换当前并启用" : "启用到当前创作"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "slider.horizontal.3").font(.title2).foregroundStyle(.tint)
                    .frame(width: 42, height: 42).background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                VStack(alignment: .leading, spacing: 5) {
                    Text("LoRA 管理").font(.title2.weight(.semibold))
                    Text("本地导入、按基础模型登记，需要时再启用。文件保留原位，不下载或合并权重。")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                Menu {
                    ForEach(supportedModels) { model in
                        Button("登记到 \(model.displayName)…") { chooseFiles(modelID: model.id) }
                    }
                } label: { Label("导入 LoRA…", systemImage: "plus") }
                    .disabled(configurationLocked || supportedModels.isEmpty)
                    .accessibilityIdentifier("loraLibraryImport")
                    .help("先选择文件所属的基础模型，再多选本地 .safetensors；登记本身不会启用或切换模型。")
            }
            HStack(spacing: 10) {
                Text("当前创作").font(.caption).foregroundStyle(.secondary)
                Text(modelTitle(studio.draft.modelID)).font(.callout.weight(.medium))
                Text("\(studio.draft.activeLoRAs.count) 个启用").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Spacer(minLength: 0)
                Button("仅看当前模型") { modelFilter = studio.draft.modelID; query = "" }
                    .disabled(!filterIDs.contains(studio.draft.modelID))
            }.padding(12).background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 10))
            HStack(spacing: 10) {
                Picker("基础模型", selection: $modelFilter) {
                    Text("全部基础模型").tag("")
                    ForEach(filterIDs, id: \.self) { id in Text(modelTitle(id)).tag(id) }
                }.frame(maxWidth: 240).accessibilityIdentifier("loraLibraryModelFilter")
                HStack(spacing: 7) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("搜索名称、模型或路径", text: $query).textFieldStyle(.plain)
                        .accessibilityIdentifier("loraLibrarySearch")
                    if !query.isEmpty {
                        Button { query = "" } label: { Image(systemName: "xmark.circle.fill") }
                            .buttonStyle(.plain).help("清空搜索")
                    }
                }.padding(8).background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 8))
                Button { refreshID += 1; library.refresh(studio: studio) } label: { Image(systemName: "arrow.clockwise") }
                    .disabled(library.busy).help("刷新登记与本地文件状态")
                    .accessibilityLabel("刷新 LoRA 模型库").accessibilityIdentifier("loraLibraryRefresh")
            }
            HStack(spacing: 10) {
                Button { scanCurrentModel() } label: { Label("扫描当前基础模型目录", systemImage: "folder.badge.magnifyingglass") }
                    .disabled(library.busy || configurationLocked || studio.draft.modelPath.isEmpty || !supportedModels.contains { $0.id == studio.draft.modelID })
                    .accessibilityIdentifier("loraLibraryScan")
                Text("只查模型目录内的 lora / loras 文件夹，不递归扫描磁盘。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if configurationLocked {
                Label(store.busy ? "任务运行中，完成后可修改 LoRA 配置。" : "图片导入中，完成或取消导入后可修改 LoRA。", systemImage: "lock")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let notice {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "info.circle")
                    Text(notice).textSelection(.enabled)
                    Spacer(minLength: 0)
                    Button { self.notice = nil } label: { Image(systemName: "xmark") }.buttonStyle(.plain).accessibilityLabel("关闭 LoRA 提示")
                }.font(.caption).foregroundStyle(.secondary).padding(10)
                    .background(Color.accentColor.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
            }
            HStack {
                Text("\(rows.count) 个文件").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                if checkingFiles { ProgressView().controlSize(.small); Text("检查本地文件…").font(.caption).foregroundStyle(.secondary) }
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if rows.isEmpty { emptyState }
                    ForEach(rows) { item in row(item) }
                }.padding(.bottom, 8)
            }.frame(maxHeight: .infinity)
            Label("模型归属是登记信息，不是兼容认证。实际加载仍会检查适配器结构、投影、精度及运行路线。", systemImage: "info.circle")
                .font(.caption).foregroundStyle(.secondary)
        }.task(id: fileCheckKey) { await checkFiles() }
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: query.isEmpty ? "slider.horizontal.3" : "magnifyingglass").font(.largeTitle).foregroundStyle(.secondary)
            Text(query.isEmpty ? "这里还没有 LoRA" : "没有匹配的 LoRA").font(.headline)
            Text(query.isEmpty ? "从右上角导入本地文件，或扫描当前基础模型的 LoRA 目录。" : "试试其他名称、路径或基础模型筛选。")
                .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
            if !modelFilter.isEmpty || !query.isEmpty { Button("查看全部") { modelFilter = ""; query = "" } }
        }.padding(30).frame(maxWidth: .infinity)
    }
    private func pill(_ title: String, color: Color = .secondary) -> some View {
        Text(title).font(.caption2.weight(.medium)).foregroundStyle(color).padding(.horizontal, 7).padding(.vertical, 4)
            .background(color.opacity(0.09), in: Capsule())
    }
    private func row(_ item: LoRALibraryRow) -> some View {
        let selected = selectedLoRA(item)
        let state = fileStates[item.path]
        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "doc.badge.gearshape").font(.title3).foregroundStyle(.secondary).frame(width: 28, height: 28)
                VStack(alignment: .leading, spacing: 5) {
                    Text(item.name).font(.callout.weight(.semibold)).textSelection(.enabled)
                    Text(modelTitle(item.modelID)).font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                pill(item.registration == nil ? "创作中 · 未登记" : "已登记")
                if selected?.enabled == true { pill("当前启用", color: .accentColor) }
                else if selected != nil { pill("当前关闭") }
                if let state, !state.readable { pill(state.status, color: .red) }
                Menu {
                    Button("在 Finder 中显示") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: item.path)]) }
                        .disabled(state?.readable != true)
                    Button("复制路径") {
                        NSPasteboard.general.clearContents(); NSPasteboard.general.setString(item.path, forType: .string)
                    }
                    if let registered = item.registration {
                        Divider()
                        Button("移除登记（保留文件）") { removeRegistration(registered) }
                            .disabled(library.busy || configurationLocked)
                    } else {
                        Button("登记到此基础模型") { library.registerLoRA(URL(fileURLWithPath: item.path), modelID: item.modelID) }
                            .disabled(configurationLocked || !fileReady(item))
                    }
                } label: { Image(systemName: "ellipsis") }
                    .menuStyle(.borderlessButton).fixedSize().accessibilityLabel("管理 \(item.name)")
            }
            HStack(alignment: .top, spacing: 8) {
                Text(item.path).font(.caption2.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                    .lineLimit(2).help(item.path)
                Spacer(minLength: 0)
                if let bytes = state?.bytes { Text(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)).font(.caption2).foregroundStyle(.secondary) }
            }
            if let selected {
                HStack(spacing: 12) {
                    Toggle(selected.enabled ? "用于当前创作" : (replacesQwenLoRA(item) ? "替换当前并启用" : "用于当前创作"), isOn: Binding(
                        get: { selectedLoRA(item)?.enabled ?? false },
                        set: { if $0 { activate(item) } else { disable(item) } }))
                        .toggleStyle(.checkbox).disabled(configurationLocked || (!selected.enabled && (!fileReady(item) || !supportsActivation(item))))
                        .accessibilityIdentifier("loraLibraryEnabled-\(item.id)")
                    Spacer(minLength: 0)
                    Button("从创作移除") { removeFromDraft(item) }.disabled(configurationLocked)
                        .help("保留登记和原文件，移除当前创作中的选择。")
                }
                if selected.enabled { editor(selected, item: item) }
            } else {
                HStack {
                    Button(activationTitle(item)) { activate(item) }
                        .disabled(configurationLocked || !fileReady(item) || !supportsActivation(item) || targetAtCapacity(item))
                        .accessibilityIdentifier("loraLibraryActivate-\(item.id)")
                        .help(targetAtCapacity(item) ? "此基础模型已配置 8 个 LoRA，请先从其创作配置移除一个。" : "启用才会改变创作配置；跨模型操作会切换基础模型与生成参数。")
                    if item.modelID == studio.draft.modelID && targetAtCapacity(item) { Text("当前已配置 8 个，请先移除一个。").font(.caption).foregroundStyle(.secondary) }
                    Spacer(minLength: 0)
                }
            }
            if !supportsActivation(item) {
                Text("此基础模型没有独立 LoRA 运行入口，或需要离线预融合 checkpoint；登记文件不会自动改写模型。")
                    .font(.caption).foregroundStyle(.secondary)
            } else if item.modelID == "qwen-image-2.1" {
                Text(StudioDraft.qwen21TurboAdapterNames.contains(URL(fileURLWithPath: item.path).lastPathComponent)
                     ? "Viggle 六步模式 · 启用后应用 512×512、6 步、强度 1 的快速设置。Qwen 每次启用一个适配器。"
                     : "普通 Qwen LoRA 使用单个 Transformer 适配器、纯 GPU、20–40 步；与六步 Viggle 使用不同采样。")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }.padding(14).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(selected?.enabled == true ? Color.accentColor.opacity(0.25) : Color.primary.opacity(0.06)))
    }
    private func editor(_ selected: StudioLoRA, item: LoRALibraryRow) -> some View {
        let turbo = studio.draft.qwen21TurboLoRA?.id == selected.id
        return HStack(spacing: 10) {
            Text("强度").font(.caption).foregroundStyle(.secondary)
            TextField("强度", value: Binding(get: { selectedLoRA(item)?.strength ?? 1 }, set: { setStrength($0, item: item) }), format: .number)
                .textFieldStyle(.roundedBorder).frame(width: 70).disabled(configurationLocked || turbo)
                .accessibilityIdentifier("loraLibraryStrength-\(item.id)").help(turbo ? "六步 Viggle 强度固定为 1。" : "支持 −8 到 8；1 为适配器原始强度。")
            if turbo { Text("六步模式固定为 1").font(.caption).foregroundStyle(.secondary) }
            else {
                Button("重置为 1") { setStrength(1, item: item) }.disabled(configurationLocked || selected.strength == 1)
                Text("−8 … 8").font(.caption2).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Text(selected.role == "transformer" ? "Transformer" : selected.role).font(.caption).foregroundStyle(.secondary)
        }
    }
    private func targetAtCapacity(_ item: LoRALibraryRow) -> Bool {
        let choices = studio.draft.modelID == item.modelID ? studio.draft.loras : studio.draft.modelLoRAs[item.modelID] ?? []
        return choices.count >= 8 && !choices.contains { sameFile($0.path, item.path) }
    }
    private func activate(_ item: LoRALibraryRow) {
        guard !configurationLocked, fileReady(item), supportsActivation(item), !targetAtCapacity(item) else { return }
        let switched = studio.draft.modelID != item.modelID
        if switched { studio.changeModel(item.modelID) }
        guard studio.draft.modelID == item.modelID, !configurationLocked else { return }
        if item.modelID == "qwen-image-2.1" {
            for other in studio.draft.activeLoRAs where !sameFile(other.path, item.path) { studio.setLoRAEnabled(other.id, enabled: false) }
        }
        if let existing = selectedLoRA(item) { studio.setLoRAEnabled(existing.id, enabled: true) }
        else { studio.addLoRA(item.path) }
        guard selectedLoRA(item)?.enabled == true else { notice = studio.message ?? "未能启用此 LoRA，请检查当前配置。"; return }
        notice = (switched ? "已切换到 \(modelTitle(item.modelID)) 并启用 " : "已启用 ") + item.name + "。" + (studio.draft.modelPath.isEmpty ? "基础模型文件尚未配置，请先选择完整安装。" : "登记归属不代替实际加载兼容检查。")
    }
    private func disable(_ item: LoRALibraryRow) {
        guard !configurationLocked, let selected = selectedLoRA(item) else { return }
        studio.setLoRAEnabled(selected.id, enabled: false)
    }
    private func removeFromDraft(_ item: LoRALibraryRow) {
        guard !configurationLocked, let selected = selectedLoRA(item) else { return }
        studio.removeLoRA(selected.id)
        notice = "已从当前创作移除 \(item.name)，登记和原文件保留。"
    }
    private func setStrength(_ value: Double, item: LoRALibraryRow) {
        guard !configurationLocked, let selected = selectedLoRA(item), selected.enabled,
              studio.draft.qwen21TurboLoRA?.id != selected.id else { return }
        guard value.isFinite, (-8...8).contains(value) else { notice = "LoRA 强度需要为 −8 到 8 的有限数值。"; return }
        guard let index = studio.draft.loras.firstIndex(where: { $0.id == selected.id }) else { return }
        studio.draft.loras[index].strength = value; studio.save()
    }
    private func removeRegistration(_ item: LibraryLoRA) {
        guard !library.busy, !configurationLocked else { return }
        library.removeLoRA(item)
        notice = "已请求移除登记：\(item.name)。原文件与当前创作中的启用设置保留。"
    }
    private func chooseFiles(modelID: String) {
        guard !configurationLocked else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
        panel.allowsMultipleSelection = true; panel.canChooseDirectories = false
        panel.prompt = "登记 LoRA"
        panel.message = "登记到 \(modelTitle(modelID))。可多选 .safetensors；不复制文件，也不自动启用或切换基础模型。"
        guard panel.runModal() == .OK, !configurationLocked else { return }
        let files = panel.urls.filter { $0.pathExtension == "safetensors" }
        guard !files.isEmpty else { notice = "请选择 .safetensors LoRA 文件。"; return }
        for file in files { library.registerLoRA(file, modelID: modelID) }
        notice = "已请求将 \(files.count) 个文件登记到 \(modelTitle(modelID))；请查看模型库操作结果。当前创作配置保持不变。"
    }
    private func scanCurrentModel() {
        guard !library.busy, !configurationLocked, !studio.draft.modelPath.isEmpty,
              supportedModels.contains(where: { $0.id == studio.draft.modelID }) else { return }
        library.discoverLoRAs(modelID: studio.draft.modelID, path: studio.draft.modelPath)
    }
    private func checkFiles() async {
        let key = fileCheckKey, paths = pathsToCheck
        checkingFiles = true
        let result = await Task.detached(priority: .utility) { () -> [String: LoRALocalFileState] in
            var states: [String: LoRALocalFileState] = [:]
            for path in paths {
                if Task.isCancelled { break }
                let file = URL(fileURLWithPath: path)
                do {
                    let values = try file.resolvingSymlinksInPath().resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                    let readable = values.isRegularFile == true && FileManager.default.isReadableFile(atPath: path)
                    states[path] = .init(readable: readable, status: readable ? "文件可读取" : "文件不可读取", bytes: values.fileSize.map(Int64.init), resolvedPath: readable ? file.resolvingSymlinksInPath().path : nil)
                } catch { states[path] = .init(readable: false, status: "文件缺失或不可访问", bytes: nil, resolvedPath: nil) }
            }
            return states
        }.value
        guard !Task.isCancelled, key == fileCheckKey else { return }
        fileStates = result; checkingFiles = false
    }
}

private struct LoRALibraryRow: Identifiable {
    var id: String
    var modelID: String
    var name: String
    var path: String
    var registration: LibraryLoRA?
}

private struct LoRALocalFileState: Sendable {
    var readable: Bool
    var status: String
    var bytes: Int64?
    var resolvedPath: String?
}
