import AppKit
import Combine
import Foundation
import UniformTypeIdentifiers

private struct PlaygroundWorkflowCatalog: Decodable {
    struct Role: Decodable {
        let id: String
        let title: String
        let detail: String
        let required: Bool
    }
    struct Mode: Decodable {
        let id: String
        let default_instruction: String?
    }
    struct Workflow: Decodable {
        let id: String
        let title: String
        let detail: String
        let default_instruction: String
        let roles: [Role]
        let modes: [Mode]
    }
    let schema_version: Int
    let model: String
    let workflows: [Workflow]
    // A metadata-only C ABI query: no engine, planning, file access or weights.
    static let shared: PlaygroundWorkflowCatalog? = {
        guard let value = try? JSONDecoder().decode(Self.self, from: NativeEngine.workflows()),
              value.schema_version == 1, value.model == "qwen-image-2.1",
              Set(value.workflows.map(\.id)).count == value.workflows.count else { return nil }
        return value
    }()
}

enum PlaygroundTemplate: String, Codable, CaseIterable, Identifiable, Sendable {
    case outfit, identity, face, outpaint, transparent
    var id: String { rawValue }
    var workflowID: String { "playground.\(rawValue)" }
    fileprivate var definition: PlaygroundWorkflowCatalog.Workflow? {
        PlaygroundWorkflowCatalog.shared?.workflows.first { $0.id == workflowID }
    }
    var title: String { definition?.title ?? rawValue }
    var detail: String { definition?.detail ?? "工作流目录不可用，请重新安装完整 App。" }
    var roles: [PlaygroundRole] { definition?.roles.compactMap { PlaygroundRole(rawValue: $0.id) } ?? [] }
    var requiredRoles: [PlaygroundRole] {
        definition?.roles.filter(\.required).compactMap { PlaygroundRole(rawValue: $0.id) } ?? []
    }
    var defaultInstruction: String { definition?.default_instruction ?? "" }
    var extractionInstruction: String { definition?.modes.first { $0.id == "extract" }?.default_instruction ?? defaultInstruction }
    func title(for role: PlaygroundRole) -> String { definition?.roles.first { $0.id == role.rawValue }?.title ?? role.title }
    func detail(for role: PlaygroundRole) -> String { definition?.roles.first { $0.id == role.rawValue }?.detail ?? role.detail }
    var primaryRole: PlaygroundRole {
        switch self { case .outpaint, .transparent: return .source; case .face: return .target; default: return .person }
    }
    var instructionTitle: String { self == .transparent ? "主体描述或附加要求" : "附加指令" }
    var referenceHint: String {
        switch self {
        case .outfit: return "人物为 <image1>，服装为 <image2>。"
        case .identity: return "人物为 <image1>；可选场景为 <image2>。"
        case .face: return "身份参考为 <image1>，要修改脸部的目标图为 <image2>。"
        case .outpaint: return "原图为 <image1>。在当前画布中扩展视野，AI 会重新构图，原图细节可能变化。"
        case .transparent: return "不添加原图时按描述生成透明图；添加原图后提取其中的主体。"
        }
    }
}

enum PlaygroundRole: String, Codable, Identifiable, Sendable {
    case person, clothing, scene, source, target
    var id: String { rawValue }
    private var definition: PlaygroundWorkflowCatalog.Role? {
        PlaygroundWorkflowCatalog.shared?.workflows.flatMap(\.roles).first { $0.id == rawValue }
    }
    var title: String { definition?.title ?? rawValue }
    var detail: String { definition?.detail ?? "选择参考图片" }
}

struct PlaygroundTemplateDraft: Codable, Sendable {
    var settings: StudioDraft
    var roleAssets: [String: StudioAsset] = [:]
    var instruction: String
    var lastSeed: Int?
    var expansion: Double? = nil
    var lastJobID: UUID?
}

struct PlaygroundDocument: Codable, Sendable {
    var schemaVersion = 2
    var selectedTemplate = PlaygroundTemplate.outfit
    var templates: [String: PlaygroundTemplateDraft]
}

/// An independent draft and importer. All slots use immutable copies; replacing
/// a binding never rewrites or deletes inputs referenced by older jobs.
@MainActor final class PlaygroundState: ObservableObject {
    @Published private(set) var document: PlaygroundDocument
    @Published private(set) var importing = false
    @Published private(set) var saved = true
    @Published private(set) var storageError: String?
    @Published var message: String?
    // The submission outlives the SwiftUI page when the user visits another
    // workspace. Keep its cancellation handle with the persistent state object.
    @Published var generationTask: Task<Void, Never>?
    @Published var generationOwnsStore = false
    @Published private(set) var importTask: Task<Bool, Never>?
    let models: [StudioModel]
    let importer: StudioAssetImporter
    let fileURL: URL
    private var revision = UUID()
    private var saveTask: Task<Void, Never>?
    private var loadBlocked = false
    private var importTaskID: UUID?

    init(directory: URL, initialSettings: StudioDraft, models: [StudioModel]) {
        self.models = models
        fileURL = directory.appendingPathComponent("playground.json")
        importer = StudioAssetImporter(directory: directory.appendingPathComponent("playground-inputs"))
        let settings = Self.parameterSettings(initialSettings)
        document = PlaygroundDocument(templates: Dictionary(uniqueKeysWithValues: PlaygroundTemplate.allCases.map {
            ($0.rawValue, PlaygroundTemplateDraft(settings: settings, instruction: $0.defaultInstruction))
        }))
        readSavedDocument()
    }

    var template: PlaygroundTemplate { document.selectedTemplate }
    var current: PlaygroundTemplateDraft { document.templates[template.rawValue]! }
    var settings: StudioDraft { current.settings }
    var instruction: String { current.instruction }
    var model: StudioModel? { models.first { $0.id == settings.modelID } }
    var orderedAssets: [StudioAsset] { template.roles.compactMap { current.roleAssets[$0.rawValue] } }
    func asset(for role: PlaygroundRole) -> StudioAsset? { current.roleAssets[role.rawValue] }
    func imageNumber(for role: PlaygroundRole) -> Int? {
        // Display the template's expected position even if the required person
        // slot is still empty. Submission itself requires that slot to be filled.
        guard asset(for: role) != nil, let index = template.roles.firstIndex(of: role) else { return nil }
        return index + 1
    }
    var previewAsset: StudioAsset? { asset(for: template.primaryRole) }
    var outpaintExpansion: Double { current.expansion ?? 1.5 }
    @discardableResult func setOutpaintExpansion(_ value: Double) -> Bool {
        guard template == .outpaint, !importing, importTask == nil, generationTask == nil,
              [1.25, 1.5, 2.0].contains(value) else { return false }
        var updated = document; updated.templates[template.rawValue]?.expansion = value
        commit(updated); return true
    }
    private struct CompiledWorkflow: Decodable {
        let operation: String
        let prompt: String
    }
    private func compileWorkflow(preview: Bool = false) throws -> CompiledWorkflow {
        guard template.definition != nil else {
            throw NativeFailure(message: "工作流目录不可用，请重新安装完整 App。")
        }
        var paths = Dictionary(uniqueKeysWithValues: template.roles.compactMap { role in
            asset(for: role).map { (role.rawValue, $0.path) }
        })
        // Missing required images have placeholder paths only in the prompt
        // preview. Submission always composes again using actual bound assets.
        if preview {
            for role in template.requiredRoles where paths[role.rawValue] == nil {
                paths[role.rawValue] = "/reference/\(role.rawValue)"
            }
        }
        var input: [String: Any] = ["workflow_id": template.workflowID, "role_paths": paths,
            "instruction": instruction,
            "request": ["schema_version": 1, "model": settings.modelID]]
        if template == .outpaint { input["expansion"] = outpaintExpansion }
        return try JSONDecoder().decode(CompiledWorkflow.self,
            from: NativeEngine.workflowRequest(input: JSONSerialization.data(withJSONObject: input)))
    }
    var prompt: String {
        (try? compileWorkflow(preview: true).prompt) ?? "工作流提示词不可用，请检查模型与参数。"
    }
    var generationBlocker: String? {
        do { _ = try generationDraft(); return nil }
        catch { return error.localizedDescription }
    }
    var referenceEncodingDraft: StudioDraft {
        var draft = settings
        draft.operation = template == .transparent && orderedAssets.isEmpty ? "image.generate" : "image.edit"
        draft.assets = orderedAssets
        return draft
    }
    @discardableResult func setQwen21ReferenceSize(_ size: Int) -> Bool {
        guard !importing, importTask == nil, generationTask == nil, [1024, 512].contains(size) else { return false }
        if size == 512, let reason = referenceEncodingDraft.qwen21FastReferenceUnavailableReason {
            message = reason; return false
        }
        var updated = document
        updated.templates[template.rawValue]?.settings.qwen21ReferenceSize = size
        commit(updated)
        return true
    }

    func generationDraft() throws -> StudioDraft {
        guard storageError == nil, !loadBlocked else {
            throw NativeFailure(message: "Playground 配置尚未成功读取或保存，请先处理保存提示。")
        }
        guard !importing, importTask == nil else { throw NativeFailure(message: "请等待参考图处理完成。") }
        guard let model, model.id == "qwen-image-2.1", model.output == "image", model.supports("image.edit") else {
            throw NativeFailure(message: "这些工作流使用 Qwen Image 2.1，请从创作页同步该模型与参数。")
        }
        guard let definition = template.definition,
              definition.roles.count == template.roles.count else {
            throw NativeFailure(message: "工作流目录不完整，请重新安装完整 App。")
        }
        guard !settings.modelPath.isEmpty else {
            throw NativeFailure(message: "尚未选择本地模型目录，请在创作页选择后同步模型与参数。")
        }
        guard !settings.upscaleAfterGeneration else {
            throw NativeFailure(message: "Playground 首版不执行生成后自动超分。请在创作页关闭自动超分后，再同步模型与参数。")
        }
        for role in template.requiredRoles where asset(for: role) == nil {
            throw NativeFailure(message: "请添加\(template.title(for: role))参考图。")
        }
        let workflow = try compileWorkflow()
        var draft = settings
        draft.operation = workflow.operation; draft.assets = orderedAssets; draft.initImageID = orderedAssets.first?.id
        draft.prompt = workflow.prompt
        // Explicit local registry; never call the default Native catalog in a
        // View body and never normalize incompatible LoRA/cache parameters.
        try draft.validate(models: models)
        return draft
    }

    func selectTemplate(_ value: PlaygroundTemplate) {
        guard !importing, importTask == nil, value != template else { return }
        var updated = document; updated.selectedTemplate = value
        commit(updated); message = nil
    }
    func setInstruction(_ value: String) {
        guard !importing, importTask == nil else { return }
        var updated = document; updated.templates[template.rawValue]?.instruction = value
        commit(updated, immediate: false)
    }
    @discardableResult func syncSettings(from draft: StudioDraft) -> Bool {
        guard !importing, importTask == nil else { message = "请等待参考图处理完成。"; return false }
        guard let model = models.first(where: { $0.id == draft.modelID }), model.id == "qwen-image-2.1",
              model.output == "image", model.supports("image.edit") else {
            message = "请在创作页选择 Qwen Image 2.1 后同步模型与参数。"; return false
        }
        var updated = document
        updated.templates[template.rawValue]?.settings = Self.parameterSettings(draft)
        commit(updated)
        message = "已同步模型与生成参数，按图片编辑执行。模板图片与附加指令已保留。"
        return true
    }
    func remove(_ role: PlaygroundRole) {
        guard !importing, importTask == nil, template.roles.contains(role), asset(for: role) != nil else { return }
        var updated = document; updated.templates[template.rawValue]?.roleAssets.removeValue(forKey: role.rawValue)
        if template == .transparent, role == .source, current.instruction == template.extractionInstruction {
            updated.templates[template.rawValue]?.instruction = template.defaultInstruction
        }
        commit(updated); message = "已移除\(role.title)绑定，历史任务的输入副本仍保留。"
    }

    /// Own UI imports across page changes and clear the handle after cleanup.
    /// Its ID cannot clear a newer retry's handle.
    @discardableResult func startImport(
        _ operation: @escaping @MainActor (PlaygroundState) async -> Bool
    ) -> Task<Bool, Never>? {
        guard !importing, importTask == nil, !loadBlocked else { return nil }
        let id = UUID(); importTaskID = id
        let task = Task { [weak self] in
            defer {
                if self?.importTaskID == id { self?.importTask = nil; self?.importTaskID = nil }
            }
            guard !Task.isCancelled, let self else { return false }
            return await operation(self)
        }
        importTask = task
        return task
    }
    func cancelImport() { importTask?.cancel() }

    @discardableResult func importFiles(_ urls: [URL], replacing role: PlaygroundRole? = nil) async -> Bool {
        await importImages(count: urls.count, replacing: role) { index in
            try await self.importer.importFile(urls[index])
        }
    }
    @discardableResult func importProviders(_ providers: [NSItemProvider], replacing role: PlaygroundRole? = nil) async -> Bool {
        await importImages(count: providers.count, replacing: role) { index in
            let provider = providers[index]
            let type = provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) ? UTType.fileURL.identifier
                : provider.registeredTypeIdentifiers.first { UTType($0)?.conforms(to: .image) == true }
            guard let type else { throw NativeFailure(message: "请拖入图片文件。") }
            let data = try await ItemProviderDataLoader.load(provider: provider, typeIdentifier: type)
            try Task.checkCancellation()
            if type == UTType.fileURL.identifier {
                guard let url = URL(dataRepresentation: data, relativeTo: nil) else { throw NativeFailure(message: "无法读取图片路径。") }
                return try await self.importer.importFile(url)
            }
            return try await self.importer.importData(data)
        }
    }
    private func importImages(count: Int, replacing role: PlaygroundRole?,
                              load: (Int) async throws -> StudioAsset) async -> Bool {
        guard !importing, !loadBlocked, count > 0 else { return false }
        let destinations: [PlaygroundRole]
        do { destinations = try destinationRoles(count: count, replacing: role) }
        catch { message = error.localizedDescription; return false }
        let context = revision, workflow = template
        importing = true; defer { importing = false }
        var staged: [StudioAsset] = []
        do {
            for index in 0..<count {
                try Task.checkCancellation()
                let asset = try await load(index)
                staged.append(asset)
                try validateContext(context, template: workflow)
            }
            var updated = document
            for (destination, asset) in zip(destinations, staged) {
                updated.templates[workflow.rawValue]?.roleAssets[destination.rawValue] = asset
            }
            if workflow == .transparent, asset(for: .source) == nil,
               current.instruction == workflow.defaultInstruction {
                updated.templates[workflow.rawValue]?.instruction = workflow.extractionInstruction
            }
            commit(updated)
            message = "已添加\(destinations.map(\.title).joined(separator: "、"))参考。\(template.referenceHint)"
            return true
        } catch {
            await importer.discard(staged)
            message = error is CancellationError ? "图片导入已取消，之前的参考图已保留。" : error.localizedDescription
            return false
        }
    }
    private func destinationRoles(count: Int, replacing role: PlaygroundRole?) throws -> [PlaygroundRole] {
        if let role {
            guard template.roles.contains(role) else { throw NativeFailure(message: "当前模板没有此参考角色。") }
            guard count == 1 else { throw NativeFailure(message: "每个角色只接受一张图片；多图请使用“添加参考图”，按空位顺序导入。") }
            return [role]
        }
        let roles = template.roles.filter { asset(for: $0) == nil }
        guard count <= roles.count else {
            throw NativeFailure(message: "本次有 \(count) 张图片，但只有 \(roles.count) 个可填位置。请减少图片；已填角色请点该位置替换。")
        }
        return Array(roles.prefix(count))
    }

    @discardableResult func prepare(_ role: PlaygroundRole, preset: ReferenceImagePreparation) async -> Bool {
        guard !importing, !loadBlocked, let asset = asset(for: role) else { return false }
        let context = revision, workflow = template
        importing = true; defer { importing = false }
        var staged: [StudioAssetImporter.Preparation] = []
        do {
            let prepared = try await importer.prepare(asset: asset, preset: preset)
            staged.append(prepared)
            try validateContext(context, template: workflow)
            var updated = document; updated.templates[workflow.rawValue]?.roleAssets[role.rawValue] = prepared.asset
            commit(updated)
            message = "\(role.title)参考已设为\(preset.title)，输出画布与模型内部参考编码尺寸未改变。"
            return true
        } catch {
            await importer.discardPreparations(staged)
            message = error is CancellationError ? "参考图处理已取消。" : error.localizedDescription
            return false
        }
    }
    private func validateContext(_ context: UUID, template: PlaygroundTemplate) throws {
        try Task.checkCancellation()
        guard revision == context, self.template == template else {
            throw NativeFailure(message: "模板或参数已经改变，本次图片处理已取消；之前的参考图未修改。")
        }
    }

    func recordSubmission(_ draft: StudioDraft, seed: Int, template: PlaygroundTemplate) {
        var updated = document
        updated.templates[template.rawValue]?.settings = Self.parameterSettings(draft)
        updated.templates[template.rawValue]?.lastSeed = seed
        commit(updated)
    }
    func recordResult(_ job: NativeJob, template: PlaygroundTemplate) {
        var updated = document; updated.templates[template.rawValue]?.lastJobID = job.id
        commit(updated)
    }
    @discardableResult func useResultAsReference(_ job: NativeJob) async -> Bool {
        guard job.hasOutput, ["image.edit", "image.generate"].contains(job.request.operation), job.workflowID == template.workflowID else {
            message = "请先选择当前模板的有效生成图片。"; return false
        }
        return await importFiles([URL(fileURLWithPath: job.request.output)], replacing: template.primaryRole)
    }
    // Retain the original state API for existing clients/tests.
    @discardableResult func useResultAsPerson(_ job: NativeJob) async -> Bool {
        guard template.primaryRole == .person else { return false }
        return await useResultAsReference(job)
    }

    func reload() {
        guard !importing, importTask == nil else { return }
        saveTask?.cancel(); readSavedDocument()
    }
    private func readSavedDocument() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        do {
            var decoded = try JSONDecoder().decode(PlaygroundDocument.self, from: Data(contentsOf: fileURL))
            let legacyTemplates: [PlaygroundTemplate] = [.outfit, .identity]
            guard [1, 2].contains(decoded.schemaVersion), PlaygroundWorkflowCatalog.shared != nil else {
                throw NativeFailure(message: "Playground 保存文件版本或工作流目录不可用。")
            }
            let expected = decoded.schemaVersion == 1 ? legacyTemplates : PlaygroundTemplate.allCases
            guard Set(decoded.templates.keys) == Set(expected.map(\.rawValue)),
                  expected.contains(decoded.selectedTemplate) else {
                throw NativeFailure(message: "Playground 保存文件模板不完整或无法识别。")
            }
            for template in expected {
                let entry = decoded.templates[template.rawValue]!, assets = entry.roleAssets
                guard Set(assets.keys).isSubset(of: Set(template.roles.map(\.rawValue))),
                      Set(assets.values.map(\.id)).count == assets.count,
                      entry.expansion == nil || (template == .outpaint && [1.25, 1.5, 2.0].contains(entry.expansion!)) else {
                    throw NativeFailure(message: "Playground 参考图角色、标识或扩图倍率不一致。")
                }
            }
            let migrated = decoded.schemaVersion == 1
            if migrated {
                let settings = Self.parameterSettings(decoded.templates[decoded.selectedTemplate.rawValue]!.settings)
                for template in PlaygroundTemplate.allCases where decoded.templates[template.rawValue] == nil {
                    decoded.templates[template.rawValue] = PlaygroundTemplateDraft(settings: settings, instruction: template.defaultInstruction)
                }
                decoded.schemaVersion = 2
            }
            document = decoded; revision = UUID(); loadBlocked = false; storageError = nil; saved = !migrated
            message = nil
        } catch {
            loadBlocked = true; saved = false
            storageError = "无法恢复 Playground：\(error.localizedDescription)。保存已暂停，原文件不会被覆盖；修复文件后可重新读取。"
            message = storageError
        }
    }
    func save() {
        saveTask?.cancel()
        guard !loadBlocked else { saved = false; return }
        do {
            try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(document).write(to: fileURL, options: .atomic)
            saved = true; storageError = nil
        } catch {
            saved = false; storageError = "Playground 保存失败：\(error.localizedDescription)"
        }
    }
    private func commit(_ updated: PlaygroundDocument, immediate: Bool = true) {
        revision = UUID(); document = updated; saved = false
        saveTask?.cancel()
        if immediate { save() }
        else {
            saveTask = Task { [weak self] in
                do { try await Task.sleep(for: .milliseconds(350)) } catch { return }
                self?.save()
            }
        }
    }
    private static func parameterSettings(_ value: StudioDraft) -> StudioDraft {
        var result = value
        result.operation = "image.edit"; result.assets = []; result.initImageID = nil; result.prompt = ""
        return result
    }
}
