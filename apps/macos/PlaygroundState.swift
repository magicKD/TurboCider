import AppKit
import Combine
import Foundation
import UniformTypeIdentifiers

enum PlaygroundTemplate: String, Codable, CaseIterable, Identifiable, Sendable {
    case outfit, identity
    var id: String { rawValue }
    var workflowID: String { "playground.\(rawValue)" }
    var title: String { self == .outfit ? "换装" : "人物一致性" }
    var detail: String {
        self == .outfit ? "保留人物身份，用第二张参考图更换服装。"
            : "保留人物外观，可加入第二张图片作为场景参考。"
    }
    var roles: [PlaygroundRole] { self == .outfit ? [.person, .clothing] : [.person, .scene] }
    var requiredRoles: [PlaygroundRole] { self == .outfit ? roles : [.person] }
    var defaultInstruction: String {
        self == .outfit ? "服装自然贴合，保留人物的脸、发型和姿态。" : "保持人物的脸、发型和服装一致，画面自然。"
    }
}

enum PlaygroundRole: String, Codable, Identifiable, Sendable {
    case person, clothing, scene
    var id: String { rawValue }
    var title: String {
        switch self { case .person: return "人物"; case .clothing: return "服装"; case .scene: return "场景" }
    }
    var detail: String {
        switch self {
        case .person: return "人物身份与外观的参考"
        case .clothing: return "要穿上的服装参考"
        case .scene: return "可选的背景与构图参考"
        }
    }
}

struct PlaygroundTemplateDraft: Codable, Sendable {
    var settings: StudioDraft
    var roleAssets: [String: StudioAsset] = [:]
    var instruction: String
    var lastSeed: Int?
    var lastJobID: UUID?
}

struct PlaygroundDocument: Codable, Sendable {
    var schemaVersion = 1
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
    let models: [StudioModel]
    let importer: StudioAssetImporter
    let fileURL: URL
    private var revision = UUID()
    private var saveTask: Task<Void, Never>?
    private var loadBlocked = false

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
    var prompt: String {
        let base: String
        switch template {
        case .outfit:
            base = "Use <image1> as the person reference and <image2> as the clothing reference. Dress the person from <image1> in the clothing from <image2>. Preserve the person's facial identity, hairstyle, body proportions and pose. Keep the original scene unless instructed otherwise. Make the clothing fit naturally. Do not reproduce the clothing reference as a separate image. Output one finished image."
        case .identity:
            base = "Use <image1> as the person identity reference. Preserve the person's facial identity, hairstyle, body proportions and clothing. "
                + (asset(for: .scene) == nil ? "Keep a natural, coherent scene. " : "Use <image2> as the scene and composition reference, placing the person from <image1> into that scene naturally. ")
                + "Output one finished image."
        }
        return base + "\n\nAdditional instruction:\n" + instruction.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    var generationBlocker: String? {
        do { _ = try generationDraft(); return nil }
        catch { return error.localizedDescription }
    }

    func generationDraft() throws -> StudioDraft {
        guard storageError == nil, !loadBlocked else {
            throw NativeFailure(message: "Playground 配置尚未成功读取或保存，请先处理保存提示。")
        }
        guard !importing else { throw NativeFailure(message: "请等待参考图处理完成。") }
        guard let model, model.output == "image", model.supports("image.edit") else {
            throw NativeFailure(message: "请从创作页同步一个已开放图片编辑的本地模型。")
        }
        guard !settings.modelPath.isEmpty else {
            throw NativeFailure(message: "尚未选择本地模型目录，请在创作页选择后同步模型与参数。")
        }
        guard !settings.upscaleAfterGeneration else {
            throw NativeFailure(message: "Playground 首版不执行生成后自动超分。请在创作页关闭自动超分后，再同步模型与参数。")
        }
        for role in template.requiredRoles where asset(for: role) == nil {
            throw NativeFailure(message: "请添加\(role.title)参考图。")
        }
        var draft = settings
        draft.operation = "image.edit"; draft.assets = orderedAssets; draft.initImageID = orderedAssets.first?.id
        draft.prompt = prompt
        // Explicit local registry; never call the default Native catalog in a
        // View body and never normalize incompatible LoRA/cache parameters.
        try draft.validate(models: models)
        return draft
    }

    func selectTemplate(_ value: PlaygroundTemplate) {
        guard !importing, value != template else { return }
        var updated = document; updated.selectedTemplate = value
        commit(updated); message = nil
    }
    func setInstruction(_ value: String) {
        guard !importing else { return }
        var updated = document; updated.templates[template.rawValue]?.instruction = value
        commit(updated, immediate: false)
    }
    @discardableResult func syncSettings(from draft: StudioDraft) -> Bool {
        guard !importing else { message = "请等待参考图处理完成。"; return false }
        guard let model = models.first(where: { $0.id == draft.modelID }), model.output == "image", model.supports("image.edit") else {
            message = "当前创作模型未开放图片编辑，请先在创作页选择支持参考编辑的模型。"; return false
        }
        var updated = document
        updated.templates[template.rawValue]?.settings = Self.parameterSettings(draft)
        commit(updated)
        message = "已同步模型与生成参数，按图片编辑执行。模板图片与附加指令已保留。"
        return true
    }
    func remove(_ role: PlaygroundRole) {
        guard !importing, template.roles.contains(role), asset(for: role) != nil else { return }
        var updated = document; updated.templates[template.rawValue]?.roleAssets.removeValue(forKey: role.rawValue)
        commit(updated); message = "已移除\(role.title)绑定，历史任务的输入副本仍保留。"
    }

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
            let data: Data = try await withCheckedThrowingContinuation { continuation in
                provider.loadDataRepresentation(forTypeIdentifier: type) { data, error in
                    if let data { continuation.resume(returning: data) }
                    else { continuation.resume(throwing: error ?? NativeFailure(message: "无法读取拖入的图片。")) }
                }
            }
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
            commit(updated)
            message = "已添加\(destinations.map(\.title).joined(separator: "、"))参考；人物始终为第一张。"
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
    @discardableResult func useResultAsPerson(_ job: NativeJob) async -> Bool {
        guard job.hasOutput, job.request.operation == "image.edit", job.workflowID == template.workflowID else {
            message = "请先选择当前模板的有效生成图片。"; return false
        }
        return await importFiles([URL(fileURLWithPath: job.request.output)], replacing: .person)
    }

    func reload() {
        guard !importing else { return }
        saveTask?.cancel(); readSavedDocument()
    }
    private func readSavedDocument() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        do {
            let decoded = try JSONDecoder().decode(PlaygroundDocument.self, from: Data(contentsOf: fileURL))
            guard decoded.schemaVersion == 1, PlaygroundTemplate.allCases.allSatisfy({ decoded.templates[$0.rawValue] != nil }) else {
                throw NativeFailure(message: "Playground 保存文件版本或模板不完整。")
            }
            for template in PlaygroundTemplate.allCases {
                let assets = decoded.templates[template.rawValue]!.roleAssets
                guard Set(assets.keys).isSubset(of: Set(template.roles.map(\.rawValue))),
                      Set(assets.values.map(\.id)).count == assets.count else {
                    throw NativeFailure(message: "Playground 参考图角色或标识不一致。")
                }
            }
            document = decoded; revision = UUID(); loadBlocked = false; storageError = nil; saved = true
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
