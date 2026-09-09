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
    var default_fps: Int? = nil
    var default_audio: Bool? = nil
    var default_residency: String? = nil
    var executor_operations: [String]? = nil
    var inputs: [String]? = nil
    var max_images: Int? = nil
    var supports_lora: Bool? = nil
    var runtime_lora: Bool? = nil
    var supports_gpu_ane: Bool? = nil
    var lora_mode: String? = nil
    var lora_strategies: [String]? = nil
    var default_lora_strategy: String? = nil
    var isVideo: Bool { output == "video" }
    func supports(_ operation: String) -> Bool {
        executor && (executor_operations ?? operations).contains(operation)
    }
    static func catalog() -> [StudioModel] {
        struct Registry: Decodable { var models: [StudioModel] }
        return (try? JSONDecoder().decode(Registry.self, from: Data(NativeEngine.models().utf8)).models) ?? []
    }
}

struct StudioLoRA: Codable, Sendable, Identifiable, Equatable {
    var id = UUID()
    var path: String
    var strength: Double = 1.0
    var role: String = "transformer"
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
    var frames = 1
    var fps = 24
    var audio = false
    var seedText = "42"
    var randomSeed = false
    var strength = 0.75
    var dynamicText = true
    var residency = "resident"
    var profilePath = ""
    var acceleration: StudioAcceleration?
    var loraStrategy = "auto"
    var assets: [StudioAsset] = []
    var loras: [StudioLoRA] = []
    var initImageID: UUID?
    init() {}
    private enum CodingKeys: String, CodingKey {
        case modelID, modelPaths, operation, prompt, width, height, steps, frames, fps, audio
        case seedText, randomSeed, strength, dynamicText, residency, profilePath, acceleration
        case assets, loras, initImageID, loraStrategy
    }
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try c.decodeIfPresent(String.self, forKey: .modelID) ?? modelID
        modelPaths = try c.decodeIfPresent([String: String].self, forKey: .modelPaths) ?? modelPaths
        // Migrate saved drafts, not runtime model aliases. Preserve the old
        // path entry as user data, and never overwrite an explicit Wan path.
        let legacyWanID = "fastmetal-1.3b-qad"
        let wanID = "wan2.1-1.3b-qad"
        if modelID == legacyWanID { modelID = wanID }
        if modelPaths[wanID] == nil, let path = modelPaths[legacyWanID] {
            modelPaths[wanID] = path
        }
        operation = try c.decodeIfPresent(String.self, forKey: .operation) ?? operation
        prompt = try c.decodeIfPresent(String.self, forKey: .prompt) ?? prompt
        width = try c.decodeIfPresent(Int.self, forKey: .width) ?? width
        height = try c.decodeIfPresent(Int.self, forKey: .height) ?? height
        steps = try c.decodeIfPresent(Int.self, forKey: .steps) ?? steps
        frames = try c.decodeIfPresent(Int.self, forKey: .frames) ?? frames
        fps = try c.decodeIfPresent(Int.self, forKey: .fps) ?? fps
        audio = try c.decodeIfPresent(Bool.self, forKey: .audio) ?? audio
        seedText = try c.decodeIfPresent(String.self, forKey: .seedText) ?? seedText
        randomSeed = try c.decodeIfPresent(Bool.self, forKey: .randomSeed) ?? randomSeed
        strength = try c.decodeIfPresent(Double.self, forKey: .strength) ?? strength
        dynamicText = try c.decodeIfPresent(Bool.self, forKey: .dynamicText) ?? dynamicText
        residency = try c.decodeIfPresent(String.self, forKey: .residency) ?? residency
        profilePath = try c.decodeIfPresent(String.self, forKey: .profilePath) ?? profilePath
        acceleration = try c.decodeIfPresent(StudioAcceleration.self, forKey: .acceleration)
        loraStrategy = try c.decodeIfPresent(String.self, forKey: .loraStrategy) ?? loraStrategy
        assets = try c.decodeIfPresent([StudioAsset].self, forKey: .assets) ?? assets
        loras = try c.decodeIfPresent([StudioLoRA].self, forKey: .loras) ?? loras
        initImageID = try c.decodeIfPresent(UUID.self, forKey: .initImageID)
    }
    var modelPath: String { modelPaths[modelID] ?? "" }
    var accelerationHint: String {
        let policy = acceleration?.policy ?? (profilePath.isEmpty ? "auto" : "profile")
        if policy == "gpu" { return "GPU · BF16，按所选融合设置运行" }
        if policy == "profile" { return "设备配置 · 运行时校验" }
        if policy == "gpu_ane" { return "手动混合 · 需匹配实际 token 容量，可能不比 GPU 快" }
        if modelID == "flux2-klein-4b", operation == "image.generate", width == height, [512, 1024].contains(width), steps == 4, residency == "resident" {
            return "\(width) 文生图 · 候选 GPU + Core ML；最终按机型、文本长度与 \(width == 512 ? 1088 : 4160) 分区复核"
        }
        if modelID == "z-image-turbo", operation == "image.generate",
           width == 1024, height == 1024, steps == 9, residency == "resident" {
            return "1024 文生图 · M4 Max 64 GB 候选 a4096 / 4128-token GPU + Core ML 路线"
        }
        return "当前任务 · GPU；尚无匹配此尺寸、操作与步数的混合收益验证"
    }
    func coreMLResourceRequest(_ action: String, kind: String? = nil) -> [String: Any] {
        let config = acceleration ?? StudioAcceleration()
        var result: [String: Any] = ["action": action, "model": modelID,
                                     "model_root": modelPath]
        for (key, value) in [("manifest", config.manifest), ("source_manifest", config.sourceManifest),
                             ("storage", config.coreMLStorage ?? ""), ("cache", config.coreMLCache ?? "")] where !value.isEmpty {
            result[key] = value
        }
        // Runtime resource operations never forward developer toolchain settings.
        if let kind { result["kind"] = kind }
        return result
    }
    var activeAssets: [StudioAsset] {
        switch operation {
        case "image.transform": return assets.filter { $0.id == initImageID }
        case "image.edit": return assets
        case "video.image": return assets.filter { $0.id == initImageID }
        case "video.reference", "video.keyframes": return assets
        default: return []
        }
    }
    func validate(models: [StudioModel] = StudioModel.catalog()) throws {
        guard let model = models.first(where: { $0.id == modelID }), model.supports(operation) else {
            throw NativeFailure(message: "当前模型不支持此创作方式，请重新选择模型或操作。")
        }
        try validate(model: model)
    }
    func validate(model: StudioModel) throws {
        guard !modelPath.isEmpty else { throw NativeFailure(message: "请先在模型中心选择模型文件夹。") }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw NativeFailure(message: "请输入描述画面或修改方式的提示词。") }
        guard model.supports(operation) else { throw NativeFailure(message: "当前模型不支持“\(operation)”操作。") }
        guard (64...2048).contains(width), (64...2048).contains(height), width % 16 == 0, height % 16 == 0 else { throw NativeFailure(message: "宽高需为 64–2048 之间的 16 倍数。") }
        guard (1...50).contains(steps) else { throw NativeFailure(message: "采样步数需为 1–50，推荐 4 步。") }
        guard frames >= 1 && frames <= 362 else { throw NativeFailure(message: "帧数超出支持范围。") }
        guard (1...120).contains(fps) else { throw NativeFailure(message: "帧率超出支持范围。") }
        if !randomSeed { _ = try fixedSeed() }
        guard strength.isFinite, (0...1).contains(strength) else { throw NativeFailure(message: "图像强度需为 0–1。") }
        if operation == "image.transform" && activeAssets.count != 1 { throw NativeFailure(message: "请选择一张原图。") }
        if operation == "image.edit" && !(1...8).contains(activeAssets.count) { throw NativeFailure(message: "参考编辑需要 1–8 张有序参考图。") }
        if operation == "video.image" && activeAssets.count != 1 { throw NativeFailure(message: "视频首帧模式需要一张首帧图片。") }
        if operation == "video.keyframes" && !(1...2).contains(activeAssets.count) { throw NativeFailure(message: "关键帧模式需要一张首帧，或首尾两张图片。") }
        if !activeAssets.isEmpty {
            guard model.inputs?.contains("image") != false else {
                throw NativeFailure(message: "当前模型不支持图片输入。")
            }
        }
        if let max = model.max_images, activeAssets.count > max { throw NativeFailure(message: "当前模型最多接受 \(max) 张输入图片。") }
        if !loras.isEmpty && model.supports_lora != true { throw NativeFailure(message: "当前模型不支持 LoRA。") }
        if modelID == "z-image-turbo-gguf" && residency != "resident" {
            throw NativeFailure(message: "GGUF 当前只支持原生 MLX 常驻模式，请将模型驻留改为 resident。")
        }
        guard ["auto", "disk_premerge", "in_memory_merge", "inference_time"].contains(loraStrategy) else {
            throw NativeFailure(message: "不支持的 LoRA 执行策略：\(loraStrategy)")
        }
        if loras.isEmpty && loraStrategy != "auto" {
            throw NativeFailure(message: "选择 LoRA 执行策略前请先添加 LoRA 文件。")
        }
        if !loras.isEmpty, let supported = model.lora_strategies,
           loraStrategy != "auto" && !supported.contains(loraStrategy) {
            throw NativeFailure(message: "当前模型不支持 LoRA 执行策略：\(loraStrategy)")
        }
        for lora in loras {
            guard FileManager.default.fileExists(atPath: lora.path) else { throw NativeFailure(message: "找不到 LoRA 文件：\(lora.path)") }
            guard lora.strength.isFinite, (-8...8).contains(lora.strength) else { throw NativeFailure(message: "LoRA 强度需为 -8–8。") }
            guard ["transformer", "text_encoder", "refiner"].contains(lora.role) else { throw NativeFailure(message: "不支持的 LoRA 角色。") }
        }
        for asset in activeAssets where !FileManager.default.fileExists(atPath: asset.path) { throw NativeFailure(message: "找不到素材：\(asset.name)。请重新添加。") }
    }
    func fixedSeed() throws -> Int {
        guard let value = Int(seedText), (0...2147483647).contains(value) else { throw NativeFailure(message: "种子需为 0–2147483647 的整数。") }
        return value
    }
    func request(output: URL, random: () -> Int = { Int.random(in: 0...2147483647) }) throws -> NativeRequest {
        let model = StudioModel.catalog().first { $0.id == modelID }
        guard let model, model.executor else { throw NativeFailure(message: "当前模型没有可用执行器。") }
        try validate(model: model)
        var request = NativeRequest(prompt: prompt, output: output.path)
        request.model = modelID; request.operation = operation
        request.width = width; request.height = height; request.steps = steps
        request.seed = randomSeed ? random() : try fixedSeed()
        request.frames = frames; request.fps = fps; request.audio = audio
        request.dynamic_text = dynamicText; request.residency = residency
        request.lora_strategy = loraStrategy
        request.profile = profilePath.isEmpty ? nil : profilePath
        let acceleration = self.acceleration ?? (profilePath.isEmpty ? StudioAcceleration() : StudioAcceleration(policy: "profile"))
        request.compile_gpu = acceleration.policy == "gpu" && modelID.hasPrefix("flux2-")
            ? acceleration.compileGPU : nil
        if acceleration.policy != "profile" {
            request.profile = nil
            request.execution = acceleration.policy == "gpu_ane" && model.supports_gpu_ane != true
                ? "gpu" : acceleration.policy
            if acceleration.policy == "auto" {
                let requiredRows = AccelerationDiscovery.automaticBucket(
                    modelID: modelID, operation: operation, width: width, height: height,
                    steps: steps, residency: residency, hasInputs: !activeAssets.isEmpty)
                request.ane_manifest = requiredRows.flatMap {
                    AccelerationDiscovery.find(
                        modelPath: modelPath, preferred: acceleration.manifest,
                        cache: acceleration.coreMLCache.map { URL(fileURLWithPath: $0) },
                        minimumRows: (width / 16) * (height / 16) + 1,
                        requiredRows: $0, enforceAutomaticPolicy: true,
                        modelID: modelID, loras: loras)?.manifest
                }
                request.allow_approximation = true
            }
            let imageLoRA = !loras.isEmpty &&
                (modelID.hasPrefix("flux2-") || modelID == "z-image-turbo" ||
                 modelID == "z-image-turbo-gguf")
            // Automatic/profile selection must not guess that a base artifact
            // contains the active adapter. Explicit GPU+ANE is allowed only if
            // the selected manifest declares this exact adapter set; native
            // loading then verifies SHA-256 in addition to this App preflight.
            let loraManifestMatches = imageLoRA && loraStrategy != "inference_time" &&
                acceleration.policy == "gpu_ane" &&
                AccelerationDiscovery.manifestBinds(manifest: acceleration.manifest,
                                                    loras: loras)
            let loraRequiresBaseGPU = imageLoRA && !loraManifestMatches
            if acceleration.policy == "gpu_ane" && model.supports_gpu_ane == true && !loraRequiresBaseGPU {
                guard !acceleration.manifest.isEmpty else { throw NativeFailure(message: "请在模型中心选择已编译的分区 manifest，或先预编译本地源分区。") }
                request.ane_manifest = acceleration.manifest; request.allow_approximation = true
            }
            // A base Core ML artifact does not contain an active LoRA delta.
            // Keep App requests safe and usable by selecting the native GPU path
            // before the manifest guard; direct native gpu_ane requests remain
            // fail-closed in the model executor.
            if loraRequiresBaseGPU {
                request.execution = "gpu"
                request.ane_manifest = nil
            }
        }
        request.inputs = activeAssets.enumerated().map { index, asset in
            let role: String
            switch operation {
            case "image.transform": role = "init_image"
            case "video.image": role = "first_frame"
            case "video.keyframes": role = index == 0 ? "first_frame" : "last_frame"
            default: role = "reference"
            }
            return NativeInput(kind: "image", role: role, path: asset.path,
                               strength: (operation == "image.transform" || operation == "video.image") ? strength : nil)
        }
        request.loras = loras.isEmpty ? nil : loras.map { NativeLoRA(path: $0.path, strength: $0.strength, role: $0.role) }
        if modelID == "wan2.1-1.3b-qad" && !loras.isEmpty { request.execution = "gpu" }
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
    let models: [StudioModel]
    let importer: StudioAssetImporter
    private let file: URL
    private var saveTask: Task<Void, Never>?
    private var undoAssets: [([StudioAsset], UUID?)] = []
    init(directory: URL, models: [StudioModel] = StudioModel.catalog()) {
        self.models = models
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
    func changeModel(_ id: String) {
        guard !importing, let model = models.first(where: { $0.id == id }), model.executor else { return }
        if draft.modelID != id { selectModel(id); return }
        if !model.supports(draft.operation) {
            draft.operation = (model.executor_operations ?? model.operations).first ?? ""
        }
        draft.steps = model.default_steps
        message = nil
    }
    func changeOperation(_ operation: String) {
        guard models.first(where: { $0.id == draft.modelID })?.supports(operation) == true else {
            message = "当前模型不支持此创作方式。"; return
        }
        message = nil
        draft.operation = operation
        if draft.initImageID == nil { draft.initImageID = draft.assets.first?.id }
    }
    func selectModel(_ id: String) {
        guard let model = models.first(where: { $0.id == id }), model.executor else { return }
        message = nil
        draft.modelID = id
        draft.operation = (model.executor_operations ?? model.operations).first ??
            (model.isVideo ? "video.generate" : "image.generate")
        draft.width = model.default_width; draft.height = model.default_height
        draft.steps = model.default_steps; draft.frames = model.default_frames
        draft.fps = model.default_fps ?? (model.isVideo ? 24 : 1)
        draft.audio = model.default_audio ?? false
        draft.residency = model.default_residency ?? "resident"
        draft.profilePath = ""
        draft.acceleration = StudioAcceleration(policy: "auto")
        draft.loras = []
        draft.loraStrategy = "auto"
        if !model.operations.contains(where: { $0 != "image.generate" && $0 != "video.generate" }) {
            draft.assets.removeAll(); draft.initImageID = nil
        }
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
    var supportsImageInputs: Bool {
        guard let model = models.first(where: { $0.id == draft.modelID }), model.executor, model.inputs?.contains("image") != false, (model.max_images ?? 8) > 0 else { return false }
        return (model.executor_operations ?? model.operations).contains {
            ["image.transform", "image.edit", "video.image", "video.reference", "video.keyframes"].contains($0)
        }
    }
    private func validateImageImport() -> Bool {
        guard supportsImageInputs else { message = "当前模型未开放图片输入。已有素材会继续保留。"; return false }
        return true
    }
    func addFiles(_ urls: [URL]) async {
        guard validateImageImport() else { return }
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
        guard validateImageImport() else { return }
        let urls = (board.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
        if !urls.isEmpty { await addFiles(urls); return }
        // One representation per pasteboard item, even when TIFF and PNG coexist.
        // Some pasteboards expose item-level data only through the board-level
        // accessor (notably unique/headless boards), so preserve that single
        // image as a fallback without importing both representations.
        let itemImages = (board.pasteboardItems ?? []).compactMap {
            $0.data(forType: .png) ?? $0.data(forType: .tiff)
        }
        var images = itemImages
        if images.isEmpty, let image = board.readObjects(forClasses: [NSImage.self], options: nil)?.first as? NSImage,
           let data = image.tiffRepresentation {
            images = [data]
        }
        if images.isEmpty, let data = board.data(forType: .png) ?? board.data(forType: .tiff) {
            images = [data]
        }
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
        guard validateImageImport() else { return }
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
        draft.steps = request.steps; draft.frames = request.frames; draft.fps = request.fps ?? 24
        draft.audio = request.audio ?? false; draft.seedText = String(request.seed); draft.randomSeed = false
        draft.operation = request.operation ?? "image.generate"
        draft.residency = request.residency ?? "resident"; draft.profilePath = request.profile ?? ""
        draft.acceleration = StudioAcceleration(policy: request.profile == nil ? request.execution : "profile", manifest: request.ane_manifest ?? "", sourceManifest: draft.acceleration?.sourceManifest ?? "", compileGPU: request.compile_gpu)
        draft.dynamicText = request.dynamic_text
        draft.assets = (request.inputs ?? []).map { StudioAsset(path: $0.path, name: URL(fileURLWithPath: $0.path).lastPathComponent, width: 0, height: 0) }
        draft.loras = (request.loras ?? []).map { StudioLoRA(path: $0.path, strength: $0.strength, role: $0.role) }
        draft.loraStrategy = request.lora_strategy ?? "auto"
        draft.initImageID = draft.assets.first?.id; draft.strength = request.inputs?.first?.strength ?? 0.75
    }
    func newDraft() { guard !importing else { message = "请等待素材导入完成。"; return }; let paths = draft.modelPaths; draft = StudioDraft(); draft.modelPaths = paths; undoAssets = []; lastSeed = nil }
}
