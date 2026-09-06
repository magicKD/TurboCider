import AppKit
import Combine
import ImageIO
import UniformTypeIdentifiers

struct StudioModel: Decodable, Identifiable {
    let id: String
    let name: String
    let executor: Bool
    let output: String
    let operations: [String]
    let default_steps: Int
    let default_frames: Int
    let default_width: Int
    let default_height: Int
    static func catalog() -> [StudioModel] {
        struct Registry: Decodable { var models: [StudioModel] }
        return (try? JSONDecoder().decode(Registry.self, from: Data(NativeEngine.models().utf8)).models) ?? []
    }
}

struct StudioAsset: Codable, Identifiable, Equatable, Sendable {
    var id = UUID()
    var path: String
    var name: String
    var width: Int
    var height: Int
}
struct StudioAcceleration: Codable, Sendable {
    var policy = "auto"
    var automaticVersion: Int? = 1
    var manifest = ""
    var sourceManifest = ""
    var compileGPU: Bool?
    var coreMLStorage: String?
    var coreMLCache: String?
    var exportPython: String?
    var exportPythonPath: String?
    var exportProfile: String?
}
struct StudioDraft: Codable, Sendable {
    var modelID = "flux2-klein-4b"
    var modelPaths: [String: String] = [:]
    var operation = "image.generate"
    var prompt = "日落时分的山间湖泊，暖金色的光线落在山脊上，宁静的水面，电影感构图。"
    var width = 512
    var height = 512
    var steps = 4
    var seedText = "42"
    var randomSeed = false
    var strength = 0.75
    var dynamicText = true
    var residency = "resident"
    var profilePath = ""
    var acceleration: StudioAcceleration?
    var assets: [StudioAsset] = []
    var initImageID: UUID?
    var modelPath: String { modelPaths[modelID] ?? "" }
    var activeAssets: [StudioAsset] {
        switch operation {
        case "image.transform": return assets.filter { $0.id == initImageID }
        case "image.edit": return assets
        default: return []
        }
    }
    func validate() throws {
        guard !modelPath.isEmpty else { throw NativeFailure(message: "请先在模型中心选择 FLUX 模型文件夹。") }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw NativeFailure(message: "请输入描述画面或修改方式的提示词。") }
        guard ["image.generate", "image.transform", "image.edit"].contains(operation) else { throw NativeFailure(message: "此操作尚未开放。") }
        guard (64...2048).contains(width), (64...2048).contains(height), width % 16 == 0, height % 16 == 0 else { throw NativeFailure(message: "宽高需为 64–2048 之间的 16 倍数。") }
        guard (1...50).contains(steps) else { throw NativeFailure(message: "采样步数需为 1–50，推荐 4 步。") }
        if !randomSeed { _ = try fixedSeed() }
        guard strength.isFinite, (0...1).contains(strength) else { throw NativeFailure(message: "图像强度需为 0–1。") }
        if operation == "image.transform" && activeAssets.count != 1 { throw NativeFailure(message: "请选择一张原图。") }
        if operation == "image.edit" && !(1...8).contains(activeAssets.count) { throw NativeFailure(message: "参考编辑需要 1–8 张有序参考图。") }
        for asset in activeAssets where !FileManager.default.fileExists(atPath: asset.path) { throw NativeFailure(message: "找不到素材：\(asset.name)。请重新添加。") }
    }
    func fixedSeed() throws -> Int {
        guard let value = Int(seedText), (0...2147483647).contains(value) else { throw NativeFailure(message: "种子需为 0–2147483647 的整数。") }
        return value
    }
    func request(output: URL, random: () -> Int = { Int.random(in: 0...2147483647) }) throws -> NativeRequest {
        try validate()
        var request = NativeRequest(prompt: prompt, output: output.path)
        request.model = modelID; request.operation = operation
        request.width = width; request.height = height; request.steps = steps
        request.seed = randomSeed ? random() : try fixedSeed()
        request.frames = 1; request.dynamic_text = dynamicText; request.residency = residency
        request.profile = profilePath.isEmpty ? nil : profilePath
        request.compile_gpu = acceleration?.compileGPU
        let acceleration = self.acceleration ?? (profilePath.isEmpty ? StudioAcceleration() : StudioAcceleration(policy: "profile"))
        if acceleration.policy != "profile" {
            request.profile = nil; request.execution = acceleration.policy
            if acceleration.policy == "auto" {
                request.ane_manifest = AccelerationDiscovery.find(modelPath: modelPath, preferred: acceleration.manifest, cache: acceleration.coreMLCache.map { URL(fileURLWithPath: $0) })?.manifest
                request.allow_approximation = true
            }
            if acceleration.policy == "gpu_ane" {
                guard !acceleration.manifest.isEmpty else { throw NativeFailure(message: "请在模型中心选择已编译的分区 manifest，或先预编译本地源分区。") }
                request.ane_manifest = acceleration.manifest; request.allow_approximation = true
            }
        }
        request.inputs = activeAssets.map { NativeInput(kind: "image", role: operation == "image.transform" ? "init_image" : "reference", path: $0.path, strength: operation == "image.transform" ? strength : nil) }
        return request
    }
}

/// Serial import keeps provider order stable. Source files are never removed;
/// a draft removes bindings only, so in-flight jobs keep their immutable inputs.
actor StudioAssetImporter {
    let directory: URL
    init(directory: URL) { self.directory = directory }
    func importFile(_ url: URL) throws -> StudioAsset {
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else { throw NativeFailure(message: "无法读取图片：\(url.lastPathComponent)") }
        return try stage(source: source, name: url.lastPathComponent, sourceURL: url, data: nil)
    }
    func importData(_ data: Data) throws -> StudioAsset {
        guard let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary) else { throw NativeFailure(message: "剪贴板不包含可解码的图片。") }
        return try stage(source: source, name: "粘贴图片", sourceURL: nil, data: data)
    }
    func discard(_ assets: [StudioAsset]) {
        for asset in assets where URL(fileURLWithPath: asset.path).deletingLastPathComponent().standardizedFileURL == directory.standardizedFileURL {
            try? FileManager.default.removeItem(atPath: asset.path)
        }
    }
    private func stage(source: CGImageSource, name: String, sourceURL: URL?, data: Data?) throws -> StudioAsset {
        guard CGImageSourceGetCount(source) == 1 else { throw NativeFailure(message: "\(name)：请先导出动画或多页图片中的单帧。") }
        guard let info = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = info[kCGImagePropertyPixelWidth] as? Int, let height = info[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, Double(width) * Double(height) <= 80_000_000 else { throw NativeFailure(message: "图片过大或尺寸无效；首版导入上限为 8000 万像素。") }
        guard CGImageSourceCreateThumbnailAtIndex(source, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 128] as CFDictionary) != nil else { throw NativeFailure(message: "图片已损坏：\(name)") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let ext = CGImageSourceGetType(source).flatMap { UTType($0 as String)?.preferredFilenameExtension } ?? "img"
        let target = directory.appendingPathComponent("\(UUID().uuidString).\(ext)")
        if let sourceURL { try FileManager.default.copyItem(at: sourceURL, to: target) }
        else if let data { try data.write(to: target, options: .atomic) }
        let orientation = info[kCGImagePropertyOrientation] as? Int ?? 1
        return StudioAsset(path: target.path, name: name, width: orientation >= 5 ? height : width, height: orientation >= 5 ? width : height)
    }
}

@MainActor
final class StudioState: ObservableObject {
    @Published var draft = StudioDraft() { didSet { scheduleSave() } }
    @Published var message: String?
    @Published var importing = false
    @Published var saved = true
    @Published var lastSeed: Int?
    let models = StudioModel.catalog()
    let importer: StudioAssetImporter
    private let file: URL
    private var saveTask: Task<Void, Never>?
    private var undoAssets: [([StudioAsset], UUID?)] = []
    init(directory: URL) {
        file = directory.appendingPathComponent("studio-draft.json")
        importer = StudioAssetImporter(directory: directory.appendingPathComponent("inputs"))
        if FileManager.default.fileExists(atPath: file.path) {
            do { draft = try JSONDecoder().decode(StudioDraft.self, from: Data(contentsOf: file)) }
            catch { message = "无法恢复草稿：\(error.localizedDescription)" }
        }
        if draft.acceleration?.automaticVersion == nil, draft.profilePath.isEmpty,
           draft.acceleration == nil || draft.acceleration?.policy == "gpu" {
            var configuration = draft.acceleration ?? StudioAcceleration()
            configuration.policy = "auto"; configuration.automaticVersion = 1; draft.acceleration = configuration
        }
        if draft.modelPath.isEmpty {
            draft.modelPaths["flux2-klein-4b"] = ProcessInfo.processInfo.environment["TURBOCIDER_FLUX_MODEL"]
                ?? UserDefaults.standard.string(forKey: "modelPath.flux2-klein-4b")
                ?? UserDefaults.standard.string(forKey: "TurboCiderNativeModelPath") ?? ""
        }
    }
    private func scheduleSave() {
        saved = false; saveTask?.cancel()
        saveTask = Task { [weak self] in
            do { try await Task.sleep(for: .milliseconds(350)) } catch { return }
            self?.save()
        }
    }
    func save() {
        saveTask?.cancel()
        do {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(draft).write(to: file, options: .atomic); saved = true
        } catch { saved = false; message = "草稿保存失败：\(error.localizedDescription)" }
    }
    func changeOperation(_ operation: String) {
        message = nil
        draft.operation = operation
        if draft.initImageID == nil { draft.initImageID = draft.assets.first?.id }
    }
    func rememberAssets() { undoAssets.append((draft.assets, draft.initImageID)); undoAssets = Array(undoAssets.suffix(20)) }
    func remove(_ id: UUID) { rememberAssets(); draft.assets.removeAll { $0.id == id }; if draft.initImageID == id { draft.initImageID = draft.assets.first?.id } }
    func move(_ id: UUID, offset: Int) {
        guard let index = draft.assets.firstIndex(where: { $0.id == id }), draft.assets.indices.contains(index + offset) else { return }
        rememberAssets(); draft.assets.swapAt(index, index + offset)
        message = "参考图顺序已更新，请核对提示词中的图片编号。"
    }
    var canUndoAssets: Bool { !undoAssets.isEmpty }
    func undoAssetChange() { if let previous = undoAssets.popLast() { draft.assets = previous.0; draft.initImageID = previous.1 } }
    func addFiles(_ urls: [URL]) async {
        guard !importing else { return }
        importing = true; defer { importing = false }
        var staged: [StudioAsset] = []
        do {
            guard draft.assets.count + urls.count <= 8 else { throw NativeFailure(message: "最多保留 8 张输入图片。本次未导入任何图片，请减少选择后重试。") }
            for url in urls { staged.append(try await importer.importFile(url)) }
            attach(staged)
        } catch { await importer.discard(staged); message = error.localizedDescription }
    }
    func pasteImage(from board: NSPasteboard = .general) async {
        let urls = (board.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
        if !urls.isEmpty { await addFiles(urls); return }
        // One representation per pasteboard item, even when TIFF and PNG coexist.
        let images = (board.pasteboardItems ?? []).compactMap { $0.data(forType: .png) ?? $0.data(forType: .tiff) }
        guard !images.isEmpty else { message = "剪贴板中没有图片。可复制图片或 Finder 中的图片文件后粘贴。"; return }
        guard !importing else { return }
        importing = true; defer { importing = false }
        var staged: [StudioAsset] = []
        do {
            guard draft.assets.count + images.count <= 8 else { throw NativeFailure(message: "粘贴后超过 8 张图片，本次未导入。") }
            for image in images { staged.append(try await importer.importData(image)) }
            attach(staged)
        } catch { await importer.discard(staged); message = error.localizedDescription }
    }
    func importProviders(_ providers: [NSItemProvider]) async {
        guard !importing else { return }
        importing = true; defer { importing = false }
        var staged: [StudioAsset] = []
        do {
            guard draft.assets.count + providers.count <= 8 else { throw NativeFailure(message: "最多 8 张图片，请减少选择后重试。") }
            for provider in providers {
                let type = provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) ? UTType.fileURL.identifier
                    : provider.registeredTypeIdentifiers.first(where: { UTType($0)?.conforms(to: .image) == true })
                guard let type else { throw NativeFailure(message: "拖入的内容不是图片文件。") }
                let data: Data = try await withCheckedThrowingContinuation { continuation in
                    provider.loadDataRepresentation(forTypeIdentifier: type) { data, error in
                        if let data { continuation.resume(returning: data) }
                        else { continuation.resume(throwing: error ?? NativeFailure(message: "无法读取拖入图片。")) }
                    }
                }
                if type == UTType.fileURL.identifier, let url = URL(dataRepresentation: data, relativeTo: nil) { staged.append(try await importer.importFile(url)) }
                else { staged.append(try await importer.importData(data)) }
            }
            attach(staged)
        } catch { await importer.discard(staged); message = error.localizedDescription }
    }
    private func attach(_ assets: [StudioAsset]) {
        rememberAssets(); draft.assets.append(contentsOf: assets)
        if draft.initImageID == nil { draft.initImageID = assets.first?.id }
        message = draft.operation == "image.generate" ? "图片已保留。请选择「单图修改」或「参考编辑」让它们参与生成。" : nil
    }
    func reuse(_ job: NativeJob) {
        guard !importing else { message = "请等待素材导入完成。"; return }
        let request = job.request
        draft.modelID = request.model
        if let path = job.modelPath { draft.modelPaths[request.model] = path }
        draft.prompt = request.prompt; draft.width = request.width; draft.height = request.height
        draft.steps = request.steps; draft.seedText = String(request.seed); draft.randomSeed = false
        draft.operation = request.operation ?? "image.generate"
        draft.residency = request.residency ?? "resident"; draft.profilePath = request.profile ?? ""
        draft.acceleration = StudioAcceleration(policy: request.profile == nil ? request.execution : "profile", manifest: request.ane_manifest ?? "", sourceManifest: draft.acceleration?.sourceManifest ?? "", compileGPU: request.compile_gpu)
        draft.dynamicText = request.dynamic_text
        draft.assets = (request.inputs ?? []).map { StudioAsset(path: $0.path, name: URL(fileURLWithPath: $0.path).lastPathComponent, width: 0, height: 0) }
        draft.initImageID = draft.assets.first?.id; draft.strength = request.inputs?.first?.strength ?? 0.75
    }
    func newDraft() { guard !importing else { message = "请等待素材导入完成。"; return }; let paths = draft.modelPaths; draft = StudioDraft(); draft.modelPaths = paths; undoAssets = []; lastSeed = nil }
}
