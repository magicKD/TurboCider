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
    var audio_output: Bool? = nil
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
    // Legacy H3 exposes audio through its executable default; newer modules
    // also report audio_output independently from upstream candidate support.
    var canGenerateAudio: Bool { executor && isVideo && (audio_output == true || default_audio == true) }
    var availableOperations: [String] { executor ? (executor_operations ?? operations) : [] }
    var acceptsImageInputs: Bool {
        inputs?.contains("image") != false && (max_images ?? 8) > 0 &&
        availableOperations.contains { $0 != "image.generate" && $0 != "video.generate" }
    }
    func matchesLibrarySearch(_ query: String, path: String) -> Bool {
        let words = query.split(whereSeparator: { $0.isWhitespace })
        let fields = ([name, id, output, path, isVideo ? "视频 video" : "图像 image",
                       acceptsImageInputs ? "图片输入 image input" : "文字输入 text input"] + availableOperations).joined(separator: " ")
        return words.allSatisfy { fields.localizedCaseInsensitiveContains(String($0)) }
    }
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
    var enabled: Bool = true
    init(id: UUID = UUID(), path: String, strength: Double = 1.0, role: String = "transformer", enabled: Bool = true) {
        self.id = id; self.path = path; self.strength = strength; self.role = role; self.enabled = enabled
    }
    private enum CodingKeys: String, CodingKey { case id, path, strength, role, enabled }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        path = try c.decode(String.self, forKey: .path)
        strength = try c.decodeIfPresent(Double.self, forKey: .strength) ?? 1.0
        role = try c.decodeIfPresent(String.self, forKey: .role) ?? "transformer"
        enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
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
    var policy = "gpu"
    var automaticVersion: Int? = 1
    var manifest = ""
    var sourceManifest = ""
    var knownManifests: [String]?
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
    var ltxBackend = "auto"
    var ltxFastAV = true
    var ltxVideoAttentionBatch = false
    var ltxAccelerationMode = "quality"
    var seedText = "42"
    var randomSeed = false
    var strength = 0.75
    var dynamicText = true
    var promptEnhance = false
    var promptEnhanceEditExperimental = false
    var promptEnhancerPath = ""
    var residency = "resident"
    var zImageStreamingBudgetGiB = 10
    var profilePath = ""
    var acceleration: StudioAcceleration?
    var loraStrategy = "auto"
    var assets: [StudioAsset] = []
    var loras: [StudioLoRA] = []
    var modelLoRAs: [String: [StudioLoRA]] = [:]
    var initImageID: UUID?
    init() {}
    private enum CodingKeys: String, CodingKey {
        case modelID, modelPaths, operation, prompt, width, height, steps, frames, fps, audio, ltxBackend, ltxFastAV, ltxVideoAttentionBatch, ltxAccelerationMode
        case seedText, randomSeed, strength, dynamicText, promptEnhance, promptEnhanceEditExperimental, promptEnhancerPath, residency, zImageStreamingBudgetGiB, profilePath, acceleration
        case assets, loras, initImageID, loraStrategy, modelLoRAs
    }
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try c.decodeIfPresent(String.self, forKey: .modelID) ?? modelID
        modelPaths = try c.decodeIfPresent([String: String].self, forKey: .modelPaths) ?? modelPaths
        operation = try c.decodeIfPresent(String.self, forKey: .operation) ?? operation
        prompt = try c.decodeIfPresent(String.self, forKey: .prompt) ?? prompt
        width = try c.decodeIfPresent(Int.self, forKey: .width) ?? width
        height = try c.decodeIfPresent(Int.self, forKey: .height) ?? height
        steps = try c.decodeIfPresent(Int.self, forKey: .steps) ?? steps
        frames = try c.decodeIfPresent(Int.self, forKey: .frames) ?? frames
        fps = try c.decodeIfPresent(Int.self, forKey: .fps) ?? fps
        audio = try c.decodeIfPresent(Bool.self, forKey: .audio) ?? audio
        ltxBackend = try c.decodeIfPresent(String.self, forKey: .ltxBackend) ?? ltxBackend
        ltxFastAV = try c.decodeIfPresent(Bool.self, forKey: .ltxFastAV) ?? ltxFastAV
        ltxVideoAttentionBatch = try c.decodeIfPresent(
            Bool.self, forKey: .ltxVideoAttentionBatch) ?? ltxVideoAttentionBatch
        ltxAccelerationMode = try c.decodeIfPresent(
            String.self, forKey: .ltxAccelerationMode) ?? ltxAccelerationMode
        seedText = try c.decodeIfPresent(String.self, forKey: .seedText) ?? seedText
        randomSeed = try c.decodeIfPresent(Bool.self, forKey: .randomSeed) ?? randomSeed
        strength = try c.decodeIfPresent(Double.self, forKey: .strength) ?? strength
        dynamicText = try c.decodeIfPresent(Bool.self, forKey: .dynamicText) ?? dynamicText
        promptEnhance = try c.decodeIfPresent(Bool.self, forKey: .promptEnhance) ?? promptEnhance
        promptEnhanceEditExperimental = try c.decodeIfPresent(Bool.self, forKey: .promptEnhanceEditExperimental) ?? false
        promptEnhancerPath = try c.decodeIfPresent(String.self, forKey: .promptEnhancerPath) ?? promptEnhancerPath
        residency = try c.decodeIfPresent(String.self, forKey: .residency) ?? residency
        zImageStreamingBudgetGiB = try c.decodeIfPresent(Int.self, forKey: .zImageStreamingBudgetGiB) ?? 10
        profilePath = try c.decodeIfPresent(String.self, forKey: .profilePath) ?? profilePath
        acceleration = try c.decodeIfPresent(StudioAcceleration.self, forKey: .acceleration)
        loraStrategy = try c.decodeIfPresent(String.self, forKey: .loraStrategy) ?? loraStrategy
        assets = try c.decodeIfPresent([StudioAsset].self, forKey: .assets) ?? assets
        loras = try c.decodeIfPresent([StudioLoRA].self, forKey: .loras) ?? loras
        modelLoRAs = try c.decodeIfPresent([String: [StudioLoRA]].self, forKey: .modelLoRAs) ?? [:]
        initImageID = try c.decodeIfPresent(UUID.self, forKey: .initImageID)
    }
    var activeLoRAs: [StudioLoRA] { loras.filter(\.enabled) }
    var usesANE: Bool { acceleration?.policy == "gpu_ane" }
    var modelPath: String { modelPaths[modelID] ?? "" }
    var zImageVariant: ZImageVariant? {
        modelID == "z-image-turbo" && !modelPath.isEmpty
            ? ZImageInstallation.variant(URL(fileURLWithPath: modelPath)) : nil
    }
    var zImageRequiresResident: Bool {
        ZImageInstallation.requiresResident(variantID: zImageVariant?.id)
    }
    @discardableResult
    mutating func normalizeZImageResidency(systemJSON: String = NativeEngine.system()) -> Bool {
        guard residency == "streamed",
              ZImageInstallation.requiresResident(variantID: zImageVariant?.id, systemJSON: systemJSON) else { return false }
        residency = "resident"
        return true
    }
    var accelerationHint: String {
        let policy = acceleration?.policy ?? (profilePath.isEmpty ? "gpu" : "profile")
        if policy == "gpu", let variant = zImageVariant, variant.id != "bf16" {
            return "GPU · \(variant.title)"
        }
        if policy == "gpu" { return "GPU · BF16，按所选融合设置运行" }
        if policy == "profile" { return "设备配置 · 运行时校验" }
        if policy == "gpu_ane" { return "手动混合 · 需匹配实际 token 容量，可能不比 GPU 快" }
        if modelID == "flux2-klein-4b", operation == "image.generate", width == height, [512, 1024].contains(width), steps == 4, residency == "resident" {
            return "\(width) 文生图 · 候选 GPU + Core ML；最终按机型、文本长度与 \(width == 512 ? 1088 : 4160) 分区复核"
        }
        if modelID == "z-image-turbo", operation == "image.generate",
           width == 512, height == 512, steps == 9 {
            return "512 文生图 · 自动使用 GPU；可导入本机验证过的 GPU + ANE 生成配置"
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
        if modelID == "ltx-2.5-distilled" {
            guard ["auto", "c_metal", "cpp_mlx"].contains(ltxBackend) else {
                throw NativeFailure(message: "LTX 后端选择无效。")
            }
            guard ["quality", "sol", "fast_approx"].contains(ltxAccelerationMode) else {
                throw NativeFailure(message: "LTX 加速模式无效。")
            }
            guard width % 64 == 0, height % 64 == 0 else {
                throw NativeFailure(message: "LTX 2.5 的宽高必须是 64 的倍数。")
            }
            guard steps == 11 else {
                throw NativeFailure(message: "LTX 2.5 Distilled 固定使用 8+3，共 11 步。")
            }
            guard frames >= 9, frames % 8 == 1 else {
                throw NativeFailure(message: "LTX 2.5 帧数必须满足 8n+1，例如 97 或 121 帧。")
            }
            guard fps == 24 else {
                throw NativeFailure(message: "LTX 2.5 Distilled 当前固定使用 24 FPS。")
            }
            if audio && residency == "streamed" {
                throw NativeFailure(message: "LTX 带音频输出暂不支持 streamed 驻留，请选择常驻或分阶段释放。")
            }
            if ltxBackend == "cpp_mlx" && operation == "video.image" {
                throw NativeFailure(message: "LTX C++/MLX 当前只支持文生视频；图生视频请选择 C/Metal。")
            }
            if residency == "streamed" && operation == "video.image" {
                throw NativeFailure(message: "LTX 图生视频不支持 streamed 驻留，请选择分阶段释放或常驻。")
            }
            if ltxAccelerationMode != "quality" {
                guard ltxBackend != "cpp_mlx" else {
                    throw NativeFailure(message: "LTX Sol 近似加速当前只支持 C/Metal。")
                }
                let latentFrames = (frames - 1) / 8 + 1
                let stage2Rows = latentFrames * (width / 32) * (height / 32)
                guard stage2Rows <= 4096 else {
                    throw NativeFailure(message: "LTX Sol 近似加速当前最多支持 4096 个 Stage-2 视频 token；720p 请使用画质优先模式。")
                }
            }
        }
        guard !audio || model.canGenerateAudio else { throw NativeFailure(message: "当前执行器尚未开放音频输出，请关闭音频。") }
        let maxDimension = modelID == "qwen-image-2.1" ? 4096 : 2048
        let dimensionMultiple = modelID == "qwen-image-2.1" ? 32 : 16
        guard (64...maxDimension).contains(width), (64...maxDimension).contains(height), width % dimensionMultiple == 0, height % dimensionMultiple == 0 else { throw NativeFailure(message: "宽高需为 64–\(maxDimension) 之间的 \(dimensionMultiple) 倍数。") }
        if modelID == "qwen-image-2.1", width * height > 8_388_608 {
            throw NativeFailure(message: "Qwen Image 2.1 当前画布上限为 8 百万像素。")
        }
        guard (1...50).contains(steps) else { throw NativeFailure(message: "采样步数需为 1–50，当前模型默认 \(model.default_steps) 步。") }
        if ["z-image-turbo", "z-image-turbo-gguf"].contains(modelID) {
            guard (residency == "resident" || (modelID == "z-image-turbo" && residency == "streamed")), frames == 1, !audio else {
                throw NativeFailure(message: "Z-Image-Turbo 支持常驻或 BF16 / INT8 流式加载，每次生成单张图片。")
            }
            if residency == "streamed" {
                guard !zImageRequiresResident else {
                    throw NativeFailure(message: zImageVariant?.id == "int8-convrot"
                        ? "INT8 流式加载仅在 Apple M5 Pro、24 GiB 内存的机器上启用；当前设备仅支持常驻加载。"
                        : "当前权重版本仅支持常驻加载，请将模型驻留改为常驻。")
                }
                guard activeLoRAs.isEmpty, profilePath.isEmpty,
                      ["gpu", "gpu_ane"].contains(acceleration?.policy ?? "gpu") else {
                    throw NativeFailure(message: "流式加载支持 BF16 / INT8，暂不支持 LoRA，请明确选择 GPU 或 GPU+ANE。")
                }
                guard acceleration?.policy != "gpu_ane" ||
                      AccelerationDiscovery.optimizationEnabled("z_image_suffix_streaming") else {
                    throw NativeFailure(message: "ANE 流式优化目前仅在已验证的 M5 Pro 24 GiB 上启用；此设备请使用纯 GPU 流式加载。")
                }
                guard [6, 8, 10, 12].contains(zImageStreamingBudgetGiB) else {
                    throw NativeFailure(message: "请选择 6、8、10 或 12 GiB 的流式内存预算。")
                }
            }
            guard activeLoRAs.allSatisfy({ $0.role == "transformer" }) else {
                throw NativeFailure(message: "Z-Image LoRA 仅支持 transformer 角色。")
            }
        }
        guard frames >= 1 && frames <= 362 else { throw NativeFailure(message: "帧数超出支持范围。") }
        guard (1...120).contains(fps) else { throw NativeFailure(message: "帧率超出支持范围。") }
        if !randomSeed { _ = try fixedSeed() }
        guard strength.isFinite, (0...1).contains(strength) else { throw NativeFailure(message: "图像强度需为 0–1。") }
        if operation == "image.transform" && activeAssets.count != 1 { throw NativeFailure(message: "请选择一张原图。") }
        let editLimit = model.max_images ?? 8
        if operation == "image.edit" && (activeAssets.isEmpty || activeAssets.count > editLimit) {
            throw NativeFailure(message: "参考编辑需要 1–\(editLimit) 张有序参考图。")
        }
        if operation == "video.image" && activeAssets.count != 1 { throw NativeFailure(message: "视频首帧模式需要一张首帧图片。") }
        if operation == "video.keyframes" && !(1...2).contains(activeAssets.count) { throw NativeFailure(message: "关键帧模式需要一张首帧，或首尾两张图片。") }
        if !activeAssets.isEmpty {
            guard model.inputs?.contains("image") != false else {
                throw NativeFailure(message: "当前模型不支持图片输入。")
            }
        }
        if let max = model.max_images, activeAssets.count > max { throw NativeFailure(message: "当前模型最多接受 \(max) 张输入图片。") }
        if !activeLoRAs.isEmpty && model.supports_lora != true { throw NativeFailure(message: "当前模型不支持 LoRA。") }
        if modelID == "z-image-turbo-gguf" && residency != "resident" {
            throw NativeFailure(message: "GGUF 当前只支持原生 MLX 常驻模式，请将模型驻留改为 resident。")
        }
        guard ["auto", "disk_premerge", "in_memory_merge", "inference_time"].contains(loraStrategy) else {
            throw NativeFailure(message: "不支持的 LoRA 执行策略：\(loraStrategy)")
        }
        if loras.isEmpty && loraStrategy != "auto" {
            throw NativeFailure(message: "选择 LoRA 执行策略前请先添加 LoRA 文件。")
        }
        if !activeLoRAs.isEmpty, let supported = model.lora_strategies,
           loraStrategy != "auto" && !supported.contains(loraStrategy) {
            throw NativeFailure(message: "当前模型不支持 LoRA 执行策略：\(loraStrategy)")
        }
        for lora in activeLoRAs {
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
        if modelID == "qwen-image-2.1" && promptEnhance {
            guard operation == "image.generate" || (operation == "image.edit" && promptEnhanceEditExperimental) else {
                throw NativeFailure(message: "PE-I2I 需要显式开启实验性 FP32 视觉；编辑质量尚未通过验收。")
            }
            guard FileManager.default.fileExists(atPath: URL(fileURLWithPath: promptEnhancerPath).appendingPathComponent("system_prompt.txt").path),
                  FileManager.default.fileExists(atPath: URL(fileURLWithPath: promptEnhancerPath).appendingPathComponent("tokenizer.json").path) else {
                throw NativeFailure(message: operation == "image.edit"
                    ? "请选择完整的 Qwen-Image-2.1 PE-I2I 模型安装目录。"
                    : "请选择完整的 Qwen-Image-2.1 PE-T2I 模型安装目录。")
            }
        }
        var request = NativeRequest(prompt: prompt, output: output.path)
        request.model = modelID; request.operation = operation
        request.width = width; request.height = height; request.steps = steps
        request.seed = randomSeed ? random() : try fixedSeed()
        request.frames = frames; request.fps = fps; request.audio = audio
        if modelID == "ltx-2.5-distilled" {
            request.ltx_backend = ltxBackend
            request.ltx_fast_av = ltxFastAV
            request.ltx_video_attention_batch =
                ltxAccelerationMode == "quality" && ltxVideoAttentionBatch
            if ltxAccelerationMode != "quality" {
                request.allow_approximation = true
                request.ltx_sol_stage2 = true
                request.ltx_sol_tau = 1.0
                request.ltx_sol_dense_edge_blocks = 1
                request.ltx_sol_dense_edge_steps = 0
                if ltxAccelerationMode == "fast_approx" {
                    request.ltx_stage2_text_rows = 256
                }
            }
        }
        request.dynamic_text = dynamicText; request.residency = residency
        request.prompt_enhance = modelID == "qwen-image-2.1" && promptEnhance
        request.prompt_enhance_edit_experimental = modelID == "qwen-image-2.1" &&
            operation == "image.edit" && promptEnhance && promptEnhanceEditExperimental
        request.prompt_enhancer_path = modelID == "qwen-image-2.1" && !promptEnhancerPath.isEmpty ? promptEnhancerPath : nil
        if modelID == "z-image-turbo", residency == "streamed" {
            request.memory_budget_bytes = UInt64(zImageStreamingBudgetGiB) << 30
        }
        request.lora_strategy = activeLoRAs.isEmpty ? "auto" : loraStrategy
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
                        modelID: modelID, loras: activeLoRAs)?.manifest
                }
                request.allow_approximation = true
            }
            let imageLoRA = !activeLoRAs.isEmpty &&
                (modelID.hasPrefix("flux2-") || modelID == "z-image-turbo" ||
                 modelID == "z-image-turbo-gguf")
            // Automatic/profile selection must not guess that a base artifact
            // contains the active adapter. Explicit GPU+ANE is allowed only if
            // the selected manifest declares this exact adapter set; native
            // loading then verifies SHA-256 in addition to this App preflight.
            let loraManifestMatches = imageLoRA && loraStrategy != "inference_time" &&
                acceleration.policy == "gpu_ane" &&
                AccelerationDiscovery.manifestBinds(manifest: acceleration.manifest,
                                                    loras: activeLoRAs)
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
        request.loras = activeLoRAs.isEmpty ? nil : activeLoRAs.map { NativeLoRA(path: $0.path, strength: $0.strength, role: $0.role) }
        if modelID == "wan2.1-1.3b-qad" && !activeLoRAs.isEmpty { request.execution = "gpu" }
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

enum Qwen21AnnotationTool: String, CaseIterable { case ellipse, brush }
enum Qwen21AnnotationOutput: String { case annotatedImage, separateMask }
struct Qwen21AnnotationStroke {
    var tool: Qwen21AnnotationTool
    // Coordinates are normalized to the displayed image, top-left origin.
    var points: [CGPoint]
    var width: Double = 0.012
}
enum Qwen21AnnotationRenderer {
    static func render(source: URL, strokes: [Qwen21AnnotationStroke],
                       output: Qwen21AnnotationOutput = .annotatedImage) throws -> Data {
        guard !strokes.isEmpty, strokes.count <= 100,
              strokes.allSatisfy({ stroke in
                  (1...4096).contains(stroke.points.count) && stroke.width.isFinite &&
                  (0.002...0.1).contains(stroke.width) &&
                  (stroke.tool != .ellipse || (stroke.points.count >= 2 &&
                    stroke.points.first!.x != stroke.points.last!.x && stroke.points.first!.y != stroke.points.last!.y)) &&
                  stroke.points.allSatisfy { $0.x.isFinite && $0.y.isFinite && (0...1).contains($0.x) && (0...1).contains($0.y) }
              }) else { throw NativeFailure(message: "无效或过多的标注，请减少笔画后重试。") }
        guard let source = CGImageSourceCreateWithURL(source as CFURL, nil),
              CGImageSourceGetCount(source) == 1,
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 2048
              ] as CFDictionary),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: image.width, height: image.height,
                bitsPerComponent: 8, bytesPerRow: image.width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        else { throw NativeFailure(message: "无法读取标注原图。") }
        let w = CGFloat(image.width), h = CGFloat(image.height)
        let bounds = CGRect(x: 0, y: 0, width: w, height: h)
        if output == .separateMask {
            // A mask is an opaque visual reference, never the source's alpha.
            context.setFillColor(CGColor(colorSpace: space, components: [0, 0, 0, 1])!)
            context.fill(bounds)
        } else { context.draw(image, in: bounds) }
        let ink = CGColor(colorSpace: space, components: output == .separateMask ? [1, 1, 1, 1] : [1, 0, 0, 1])!
        context.setStrokeColor(ink)
        context.setFillColor(ink)
        context.setLineCap(.round); context.setLineJoin(.round)
        for stroke in strokes {
            let points = stroke.points.map { CGPoint(x: $0.x * w, y: (1 - $0.y) * h) }
            let thickness = CGFloat(stroke.width) * min(w, h)
            context.setLineWidth(thickness)
            if stroke.tool == .ellipse {
                guard let a = points.first, let b = points.last, a != b else { continue }
                let rect = CGRect(x: min(a.x, b.x), y: min(a.y, b.y),
                                  width: abs(a.x-b.x), height: abs(a.y-b.y))
                if output == .separateMask { context.fillEllipse(in: rect) }
                else { context.strokeEllipse(in: rect) }
            } else if points.allSatisfy({ $0 == points[0] }) {
                context.fillEllipse(in: CGRect(x: points[0].x-thickness/2, y: points[0].y-thickness/2,
                                               width: thickness, height: thickness))
            } else {
                context.beginPath(); context.addLines(between: points); context.strokePath()
            }
        }
        let data = NSMutableData()
        guard let result = context.makeImage(),
              let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { throw NativeFailure(message: "无法创建标注 PNG。") }
        CGImageDestinationAddImage(destination, result, nil)
        guard CGImageDestinationFinalize(destination) else { throw NativeFailure(message: "保存标注失败。") }
        return data as Data
    }
}

struct Qwen21CanvasPreset: Identifiable {
    let id: String
    let width: Int
    let height: Int
    var title: String { "\(id) · \(width) × \(height)" }
    // Exact aligned sizes recommended by the official Qwen-Image-2.1 README.
    static let recommended: [Qwen21CanvasPreset] = [
        .init(id: "1:1", width: 2048, height: 2048),
        .init(id: "4:3", width: 2400, height: 1792),
        .init(id: "3:4", width: 1792, height: 2400),
        .init(id: "3:2", width: 2528, height: 1696),
        .init(id: "2:3", width: 1696, height: 2528),
        .init(id: "16:9", width: 2752, height: 1536),
        .init(id: "9:16", width: 1536, height: 2752)
    ]
}

enum Qwen21PromptExample: String, CaseIterable {
    case transparent, extraction, rgbaEdit, maskEdit, annotatedEdit
    var title: String {
        switch self {
        case .transparent: return "透明文生图"
        case .extraction: return "提取主体（透明背景）"
        case .rgbaEdit: return "RGBA 图编辑"
        case .maskEdit: return "单独蒙版引导编辑"
        case .annotatedEdit: return "圈选 / 涂抹引导编辑"
        }
    }
    var referenceCount: Int { self == .transparent ? 0 : (self == .maskEdit ? 2 : 1) }
    var prompt: String {
        switch self {
        case .transparent:
            return "This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent."
        case .extraction:
            return "Extract the main foreground subject from <image1>. Preserve its appearance and details. Remove the background and output an RGBA image with a transparent background."
        case .rgbaEdit:
            return "Change the foreground subject in <image1> to cobalt blue. Preserve its shape and details. Keep the transparent background and output an RGBA image with an alpha channel."
        case .maskEdit:
            return "Use <image2> as a spatial mask for <image1>: white marks the area to edit and black marks the area to preserve. Change the object inside the white region to matte red. Keep the rest of <image1> unchanged. Do not include the mask in the output."
        case .annotatedEdit:
            return "Change the object marked by the circle or painted annotation in <image1> to matte red. Remove the annotation from the final image. Keep the unmarked objects and background unchanged."
        }
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
        if draft.profilePath.isEmpty, draft.acceleration == nil || draft.acceleration?.policy == "auto" {
            var configuration = draft.acceleration ?? StudioAcceleration()
            configuration.policy = "gpu"; configuration.automaticVersion = 1; draft.acceleration = configuration
        }
        if draft.modelPath.isEmpty {
            draft.modelPaths["flux2-klein-4b"] = ProcessInfo.processInfo.environment["TURBOCIDER_FLUX_MODEL"]
                ?? UserDefaults.standard.string(forKey: "modelPath.flux2-klein-4b")
                ?? UserDefaults.standard.string(forKey: "TurboCiderNativeModelPath") ?? ""
        }
        if draft.normalizeZImageResidency() {
            message = "此设备未启用当前权重的流式加载，已恢复常驻。"
            save()
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
    func installZImage(model: URL, sharedText: URL?) throws {
        let installed = try ZImageInstallation.install(model: model, sharedText: sharedText,
            directory: LibraryStore.defaultRoot.appendingPathComponent("bindings"))
        selectModel("z-image-turbo")
        draft.modelPaths["z-image-turbo"] = installed.path
        save()
    }
    func importConfiguration(from url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let fields = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              fields["modelID"] is String, fields["modelPaths"] is [String: String] else {
            throw NativeFailure(message: "请选择含 modelID 和 modelPaths 的 App 生成配置。")
        }
        let imported = try JSONDecoder().decode(StudioDraft.self, from: data)
        try imported.validate(models: models)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: imported.modelPath, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw NativeFailure(message: "配置中的模型目录不存在，请先选择本地模型。")
        }
        // Preserve registrations for other models; replace only this creation draft.
        var selected = imported
        selected.modelPaths = draft.modelPaths.merging(imported.modelPaths.filter { !$0.value.isEmpty }) { _, incoming in incoming }
        draft = selected
        message = nil
        save()
    }
    func setANEEnabled(_ enabled: Bool) {
        var config = draft.acceleration ?? StudioAcceleration()
        config.policy = enabled ? "gpu_ane" : "gpu"
        config.automaticVersion = 1
        draft.profilePath = ""
        draft.acceleration = config
    }
    func rememberAcceleration(_ resolved: StudioDraft) {
        guard draft.modelID == resolved.modelID, draft.activeLoRAs == resolved.activeLoRAs,
              draft.acceleration?.policy == resolved.acceleration?.policy else { return }
        draft.acceleration = resolved.acceleration
        save()
    }
    func preparationRequest(modelID: String, output: URL) throws -> NativeRequest {
        if draft.modelID != modelID { selectModel(modelID) }
        return try draft.request(output: output)
    }
    func changeModel(_ id: String) {
        guard !importing, let model = models.first(where: { $0.id == id }), model.executor else { return }
        if draft.modelID != id {
            let operation = draft.operation
            selectModel(id)
            if model.supports(operation) { draft.operation = operation }
            return
        }
        if !model.supports(draft.operation) {
            draft.operation = (model.executor_operations ?? model.operations).first ?? ""
        }
        message = nil
    }
    var creationKind: String { draft.operation.hasPrefix("video.") ? "video" : "image" }
    var creationModels: [StudioModel] {
        models.filter { model in model.availableOperations.contains { $0.hasPrefix(creationKind + ".") } }
    }
    var creationOperations: [String] {
        let operations = creationModels.flatMap(\.availableOperations)
        let preferred = ["image.generate", "image.transform", "image.edit", "video.generate", "video.image", "video.keyframes", "video.reference"]
        return preferred.filter { operations.contains($0) } + Set(operations).subtracting(preferred).sorted()
    }
    func changeCreationKind(_ kind: String) {
        guard ["image", "video"].contains(kind), kind != creationKind else { return }
        changeOperation(kind + ".generate")
    }
    func changeOperation(_ operation: String) {
        guard !importing else { return }
        var switchedModel: StudioModel? = nil
        if models.first(where: { $0.id == draft.modelID })?.supports(operation) != true {
            let candidates = models.filter { $0.supports(operation) }
            guard let candidate = candidates.first(where: { !(draft.modelPaths[$0.id] ?? "").isEmpty }) ?? candidates.first else {
                message = "暂时没有支持此创作方式的执行器，请前往模型中心查看。"; return
            }
            let assets = draft.assets
            let initImageID = draft.initImageID
            selectModel(candidate.id)
            draft.assets = assets
            draft.initImageID = initImageID
            switchedModel = candidate
        }
        message = switchedModel.map { "已切换至 \($0.name)，模型参数与加速配置已重置，已恢复该模型的 LoRA 选择。" + (draft.modelPath.isEmpty ? "请在模型中心配置模型文件。" : "请核对生成参数。") }
        draft.operation = operation
        if draft.initImageID == nil { draft.initImageID = draft.assets.first?.id }
    }
    func selectInstallation(modelID: String, path: String) {
        guard !importing, !path.isEmpty else { return }
        if draft.modelID != modelID { selectModel(modelID) }
        guard draft.modelID == modelID, draft.modelPath != path else { return }
        draft.modelPaths[modelID] = path
        draft.profilePath = ""
        var config = draft.acceleration ?? StudioAcceleration()
        // Keep candidates for switching back; resolution checks the new checkpoint
        // and active LoRAs before accepting any previous or registered partition.
        config.knownManifests = Array(Set((config.knownManifests ?? []) +
            [config.manifest, config.sourceManifest])).filter { !$0.isEmpty }.sorted()
        config.manifest = ""; config.sourceManifest = ""; config.compileGPU = nil
        if config.policy == "profile" || draft.zImageVariant?.id == "nvfp4" { config.policy = "gpu" }
        draft.acceleration = config
        draft.normalizeZImageResidency()
        let title = draft.zImageVariant?.title ?? "所选安装"
        message = "已切换至 \(title)。" + (draft.usesANE ? "生成时自动匹配此版本的 ANE 分区。" : "")
        save()
    }
    func selectModel(_ id: String) {
        guard let model = models.first(where: { $0.id == id }), model.executor else { return }
        message = nil
        draft.modelLoRAs[draft.modelID] = draft.loras
        draft.modelID = id
        draft.operation = (model.executor_operations ?? model.operations).first ??
            (model.isVideo ? "video.generate" : "image.generate")
        draft.width = model.default_width; draft.height = model.default_height
        if id == "z-image-turbo" || id == "qwen-image-2.1" { draft.width = 512; draft.height = 512 }
        draft.steps = model.default_steps; draft.frames = model.default_frames
        draft.fps = model.default_fps ?? (model.isVideo ? 24 : 1)
        draft.audio = model.default_audio ?? false
        draft.ltxBackend = "auto"; draft.ltxFastAV = true
        draft.ltxVideoAttentionBatch = false
        draft.ltxAccelerationMode = "quality"
        draft.residency = model.default_residency ?? "resident"
        draft.profilePath = ""
        draft.acceleration = StudioAcceleration(policy: "gpu")
        draft.loras = draft.modelLoRAs[id] ?? []
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
    // Keep the existing eight-asset staging area for smaller-input models,
    // while allowing the complete ten-reference Qwen21 input contract.
    var imageImportLimit: Int {
        max(8, models.first(where: { $0.id == draft.modelID })?.max_images ?? 8)
    }
    func applyQwen21Example(_ example: Qwen21PromptExample) {
        guard draft.modelID == "qwen-image-2.1", !importing else { return }
        guard draft.assets.count >= example.referenceCount else {
            message = "请先添加至少 \(example.referenceCount) 张图片；蒙版示例中第一张为原图、第二张为白色编辑区蒙版。"
            return
        }
        // Output canvas is independent of the native reference encoding size.
        // Keep examples inside the maintained 512-square regression scope.
        let size = (512, 512)
        draft.operation = example.referenceCount == 0 ? "image.generate" : "image.edit"
        draft.prompt = example.prompt
        draft.width = size.0; draft.height = size.1; draft.steps = 40
        draft.profilePath = ""
        var acceleration = draft.acceleration ?? StudioAcceleration()
        acceleration.policy = "gpu"
        draft.acceleration = acceleration
        message = "已替换为可编辑的示例提示词，使用 GPU / 40 步 / \(size.0)×\(size.1)。这不是模型提示词重写；蒙版与标注作为视觉参考，不保证逐像素锁定未编辑区。"
    }
    func annotateQwen21Asset(_ id: UUID, strokes: [Qwen21AnnotationStroke],
                            output: Qwen21AnnotationOutput = .annotatedImage) async -> Bool {
        guard draft.modelID == "qwen-image-2.1", !importing,
              let original = draft.assets.first(where: { $0.id == id }) else { return false }
        if output == .separateMask && draft.assets.count >= imageImportLimit {
            message = "独立蒙版需要一个参考图位置；请先移除一张图片（最多 10 张，包含蒙版）。"
            return false
        }
        importing = true; defer { importing = false }
        do {
            let data = try Qwen21AnnotationRenderer.render(source: URL(fileURLWithPath: original.path), strokes: strokes, output: output)
            var annotated = try await importer.importData(data)
            guard draft.modelID == "qwen-image-2.1", let index = draft.assets.firstIndex(where: { $0.id == id }),
                  output != .separateMask || draft.assets.count < imageImportLimit else {
                await importer.discard([annotated]); return false
            }
            annotated.name = original.name + (output == .separateMask ? " · 黑白蒙版" : " · 标注")
            rememberAssets()
            if output == .separateMask {
                // Append, don't insert: existing <imageN> references stay valid.
                draft.assets.append(annotated)
                message = "已添加 <image\(draft.assets.count)> 作为 <image\(index + 1)> 的黑白蒙版：白色编辑、黑色保留。请在提示词中引用这两个编号；属于视觉引导，不保证逐像素锁定。原图与提示词未修改，可撤销。"
            } else {
                draft.assets[index] = annotated
                if draft.initImageID == id { draft.initImageID = annotated.id }
                message = "已在参考 \(index + 1) 使用标注副本；原文件未改动，可撤销。请在提示词中说明圈选 / 涂抹区域的修改，并要求移除标注。"
            }
            draft.operation = "image.edit"
            return true
        } catch {
            message = error.localizedDescription
            return false
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
            guard draft.assets.count + urls.count <= imageImportLimit else { throw NativeFailure(message: "最多保留 \(imageImportLimit) 张输入图片。本次未导入任何图片，请减少选择后重试。") }
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
            guard draft.assets.count + images.count <= imageImportLimit else { throw NativeFailure(message: "粘贴后超过 \(imageImportLimit) 张图片，本次未导入。") }
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
            guard draft.assets.count + providers.count <= imageImportLimit else { throw NativeFailure(message: "最多 \(imageImportLimit) 张图片，请减少选择后重试。") }
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
        if request.model == "z-image-turbo", let budget = request.memory_budget_bytes {
            draft.zImageStreamingBudgetGiB = Int(min(budget >> 30, 12))
        }
        draft.acceleration = StudioAcceleration(policy: request.profile == nil ? request.execution : "profile", manifest: request.ane_manifest ?? "", sourceManifest: draft.acceleration?.sourceManifest ?? "", compileGPU: request.compile_gpu)
        draft.dynamicText = request.dynamic_text
        draft.promptEnhance = request.prompt_enhance ?? false
        draft.promptEnhanceEditExperimental = request.prompt_enhance_edit_experimental ?? false
        draft.promptEnhancerPath = request.prompt_enhancer_path ?? ""
        draft.assets = (request.inputs ?? []).map { StudioAsset(path: $0.path, name: URL(fileURLWithPath: $0.path).lastPathComponent, width: 0, height: 0) }
        draft.loras = (request.loras ?? []).map { StudioLoRA(path: $0.path, strength: $0.strength, role: $0.role) }
        draft.loraStrategy = request.lora_strategy ?? "auto"
        draft.initImageID = draft.assets.first?.id; draft.strength = request.inputs?.first?.strength ?? 0.75
    }
    func newDraft() { guard !importing else { message = "请等待素材导入完成。"; return }; let paths = draft.modelPaths; draft = StudioDraft(); draft.modelPaths = paths; undoAssets = []; lastSeed = nil }
}
