import Foundation

/// A precision is a concrete storage/runtime contract, not just a bit count.
struct ZImageVariant: Identifiable, Sendable {
    var id: String
    var title: String
    var filename: String
    var weightBytes: Int64
    var runnable: Bool
    var note: String
    static let all: [Self] = [
        .init(id: "bf16", title: "BF16 · 原始精度", filename: "z_image_turbo_bf16.safetensors", weightBytes: 12_309_866_400, runnable: true, note: "质量基线；权重约 12.31 GB。"),
        .init(id: "int8-convrot", title: "INT8 ConvRot · 低内存", filename: "z_image_turbo_int8_convrot.safetensors", weightBytes: 6_201_001_296, runnable: true, note: "原生旋转 + MLX Q8 运算；速度和质量需按任务实测。"),
        .init(id: "nvfp4", title: "NVFP4 · 实验性 GPU W4A16", filename: "z_image_turbo_nvfp4.safetensors", weightBytes: 4_509_509_600, runnable: true, note: "原生重排 Comfy NVFP4 权重，MLX W4A16（BF16 激活），不是 NVIDIA W4A4。仅 GPU；不支持此版本的 LoRA / ANE。质量变化可能更明显。")
    ]
    static let repository = "Comfy-Org/z_image_turbo"
    var path: String { "split_files/diffusion_models/" + filename }
    func include(sharedText: Bool) -> [String] {
        [path, "split_files/vae/ae.safetensors"] + (sharedText ? [] : ["split_files/text_encoders/qwen_3_4b.safetensors", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"])
    }
    static let tokenizerSource = LibrarySupplement(repository: "Tongyi-MAI/Z-Image-Turbo", include: ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"])
}

/// A catalog entry describes source bytes and the current runtime contract
/// separately. Download availability is not proof of executable compatibility.
struct LibraryVariantComponent: Codable, Sendable {
    var kind: String
    var title: String
    var format: String
    var precision: String
    var repository: String
    var include: [String]
    var note: String
}

struct LibraryVariant: Codable, Identifiable, Sendable {
    var id: String
    var title: String
    var repository: String
    var include: [String]
    var supplements: [LibrarySupplement] = []
    var destinations: [String: String] = [:]
    var components: [LibraryVariantComponent] = []
    var runnable: Bool
    var preparation: String

    func request(modelID: String, provider: HubProvider = .modelscope, componentKind: String? = nil) throws -> LibraryDownloadRequest {
        let request = LibraryDownloadRequest(modelID: modelID, repository: repository, provider: provider,
            name: title, include: include, supplements: supplements, destinations: destinations)
        guard let componentKind else { return request }
        let kinds: [String]
        switch componentKind {
        case "encoder", "text_encoder": kinds = ["text_encoder", "tokenizer"]
        case "dit", "vae": kinds = [componentKind]
        default: throw LibraryFailure(message: "Unknown downloadable component: \(componentKind)")
        }
        return try request.subset(include: components.filter { kinds.contains($0.kind) }.flatMap(\.include))
    }
}

struct LibraryRecipe: Codable, Sendable {
    var modelID: String
    var repository: String
    var include: [String]
    var preparation: String
    var supplements: [LibrarySupplement] = []
    var variants: [LibraryVariant] = []
    /// Every existing recipe remains selectable even without a precision catalog.
    var downloadVariants: [LibraryVariant] {
        if !variants.isEmpty { return variants }
        let baseline = LibraryVariant(id: "default", title: "默认配套文件", repository: repository,
            include: include, supplements: supplements, components: defaultComponents,
            runnable: true, preparation: preparation)
        guard modelID == "z-image-turbo" else { return [baseline] }
        return [baseline] + ZImageVariant.all.map { variant in
            let encoder = "split_files/text_encoders/qwen_3_4b.safetensors"
            let vae = "split_files/vae/ae.safetensors"
            return LibraryVariant(id: "comfy-" + variant.id, title: variant.title, repository: ZImageVariant.repository,
                include: variant.include(sharedText: false), supplements: [ZImageVariant.tokenizerSource],
                components: [
                    .init(kind: "dit", title: "Z-Image DiT", format: "safetensors", precision: variant.title,
                          repository: ZImageVariant.repository, include: [variant.path], note: variant.note),
                    .init(kind: "text_encoder", title: "Qwen3-4B 编码器", format: "safetensors", precision: "BF16",
                          repository: ZImageVariant.repository, include: [encoder], note: "与 Z-Image 配套。"),
                    .init(kind: "vae", title: "VAE", format: "safetensors", precision: "BF16",
                          repository: ZImageVariant.repository, include: [vae], note: "16-channel Z-Image VAE。"),
                    .init(kind: "tokenizer", title: "Tokenizer", format: "JSON", precision: "原始词表",
                          repository: ZImageVariant.tokenizerSource.repository, include: ZImageVariant.tokenizerSource.include,
                          note: "与 Qwen3-4B 编码器配套。")
                ], runnable: variant.runnable, preparation: variant.note)
        }
    }
    private var defaultComponents: [LibraryVariantComponent] {
        guard ["z-image-turbo", "flux2-klein-4b", "flux2-klein-9b", "ltx-2.5-distilled", "wan2.1-1.3b-qad"].contains(modelID) else { return [] }
        let groups: [(String, String, [String])] = [
            ("dit", "DiT", include.filter { $0.hasPrefix("transformer/") || $0.hasPrefix("diffusion_models/") || $0.hasPrefix("mlx_dit.") }),
            ("text_encoder", "Encoder", include.filter { $0.hasPrefix("text_encoder/") || $0.hasPrefix("text_encoders/") }),
            ("vae", "VAE", include.filter { $0.hasPrefix("vae/") }),
            ("tokenizer", "Tokenizer", include.filter { $0.hasPrefix("tokenizer/") || $0.hasSuffix("/tokenizer.json") })
        ]
        return groups.flatMap { kind, title, paths in
            // Keep supplementary provenance visible even for a single-file
            // tokenizer whose installation path resembles the primary layout.
            Dictionary(grouping: paths) { path in
                supplements.first(where: { $0.include.contains(path) })?.repository ?? repository
            }.map { source, files in
                LibraryVariantComponent(kind: kind, title: title, format: kind == "tokenizer" ? "JSON" : "仓库原始格式",
                    precision: kind == "tokenizer" ? "原始词表" : "配方原始精度", repository: source, include: files.sorted(), note: preparation)
            }.sorted { $0.repository < $1.repository }
        }
    }

    private static let qwenRepository = "Comfy-Org/Qwen-Image-2.1"
    private static let qwenTokenizer = "processor/tokenizer.json"
    private static let qwenEncoder = "text_encoders/qwen3vl_8b_bf16.safetensors"
    private static let qwenVAE = "vae/qwen_image_2.1_vae_bf16.safetensors"
    private static let qwenBF16 = "diffusion_models/qwen_image_2.1_bf16.safetensors"
    private static let qwenTokenizerSource = LibrarySupplement(repository: "Qwen/Qwen-Image-2.1", include: [qwenTokenizer])
    private static let qwenSharedComponents: [LibraryVariantComponent] = [
        .init(kind: "text_encoder", title: "Qwen3-VL 文本与图像编码器", format: "safetensors", precision: "BF16",
              repository: qwenRepository, include: [qwenEncoder], note: "保留完整视觉编码器；GGUF DiT 的编辑 / LoRA 运行仍未准入。"),
        .init(kind: "vae", title: "Qwen Image 2.1 VAE", format: "safetensors", precision: "BF16",
              repository: qwenRepository, include: [qwenVAE], note: "与 Qwen Image 2.1 配套；不能用旧版 Qwen Image VAE 替代。"),
        .init(kind: "tokenizer", title: "Tokenizer", format: "JSON", precision: "原始词表",
              repository: "Qwen/Qwen-Image-2.1", include: [qwenTokenizer], note: "与编码器配套，从原始发布仓库补齐。")
    ]
    private static let qwenVariants: [LibraryVariant] = {
        let baseline = LibraryVariant(id: "bf16", title: "BF16 · 推荐", repository: qwenRepository,
            include: [qwenBF16, qwenEncoder, qwenVAE, qwenTokenizer], supplements: [qwenTokenizerSource],
            components: [.init(kind: "dit", title: "Qwen Image 2.1 DiT", format: "safetensors", precision: "BF16",
                repository: qwenRepository, include: [qwenBF16], note: "原生 GPU 基线，支持已验收的生成、编辑与 LoRA 路径。")]
                + qwenSharedComponents,
            runnable: true, preparation: "原生 GPU；支持生成、1–3 张参考图编辑及已支持的 LoRA。ANE 属单独配置的实验路径，不由下载格式自动启用。请确认 Qwen 研究用途许可。")
        // Exact upstream filenames verified in unsloth/Qwen-Image-2.1-GGUF.
        // Q4_K_M is currently the only filename selected by the native loader.
        let quants = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q6_K_XL", "Q8_0"]
        return [baseline] + quants.map { quant in
            let filename = "qwen-image-2.1-\(quant).gguf"
            let supported = quant == "Q4_K_M"
            let note = supported
                ? "实验性近似文生图；需明确允许近似。当前不支持 GGUF 编辑、LoRA、提示词增强或编码器 ANE。GPU 为常规入口；不会自动启用实验 ANE。"
                : "仅下载 · 运行暂不支持。当前原生入口只识别 Q4_K_M，不能通过重命名文件获得兼容性。"
            return LibraryVariant(id: "unsloth-\(quant.lowercased())", title: "Unsloth GGUF \(quant) · \(supported ? "实验文生图" : "仅下载")",
                repository: "unsloth/Qwen-Image-2.1-GGUF",
                include: [filename, qwenEncoder, qwenVAE, qwenTokenizer],
                supplements: [.init(repository: qwenRepository, include: [qwenEncoder, qwenVAE]), qwenTokenizerSource],
                destinations: [filename: "diffusion_models/" + filename],
                components: [.init(kind: "dit", title: "Qwen Image 2.1 DiT", format: "GGUF", precision: quant,
                    repository: "unsloth/Qwen-Image-2.1-GGUF", include: [filename], note: note)] + qwenSharedComponents,
                runnable: supported, preparation: note + " GGUF 文件仅含 DiT；此配方补齐 BF16 编码器、VAE 和 Tokenizer，不下载第二份 DiT。默认优先 ModelScope；任一配套来源无镜像时可切换 Hugging Face。请确认 Qwen 研究用途许可。")
        }
    }()
    static let all: [LibraryRecipe] = [
        .init(modelID: "seedvr2-3b", repository: "Comfy-Org/SeedVR2",
              include: ["diffusion_models/seedvr2_3b_fp16.safetensors", "vae/seedvr2_ema_vae_fp16.safetensors"],
              preparation: "原生图像 / 视频超分；原始 FP16 权重、BF16 DiT、FP16 因果 VAE。无需文本编码器。先使用 GPU；GPU+ANE 和成对速度验收仍在进行。视频需填写完整源帧数和整数源 FPS，当前输出无音轨。"),
        .init(modelID: "qwen-image-2.1", repository: "Comfy-Org/Qwen-Image-2.1",
              include: [qwenBF16, qwenEncoder, qwenVAE, qwenTokenizer],
              preparation: qwenVariants[0].preparation,
              supplements: [qwenTokenizerSource], variants: qwenVariants),
        .init(modelID: "z-image-turbo", repository: "Tongyi-MAI/Z-Image-Turbo",
              include: ["model_index.json", "scheduler/", "transformer/", "vae/", "text_encoder/", "tokenizer/"],
              preparation: "完整 Diffusers 目录可直接加载；可选择兼容 Qwen3-4B 文本组件避免重复下载。"),
        .init(modelID: "flux2-klein-4b", repository: "black-forest-labs/FLUX.2-klein-4B",
              include: ["model_index.json", "scheduler/", "transformer/", "vae/", "text_encoder/", "tokenizer/"],
              preparation: "完整目录下载后先使用 GPU；ANE 分区需离线导出，然后在加速面板中导入并原生编译。"),
        .init(modelID: "flux2-klein-9b", repository: "black-forest-labs/FLUX.2-klein-9B",
              include: ["model_index.json", "scheduler/", "transformer/", "vae/", "text_encoder/", "tokenizer/"],
              preparation: "需要在来源网站接受模型条款。9B 当前使用 GPU，需要更多统一内存。"),
        .init(modelID: "ltx-2.5-distilled", repository: "comfyicu/LTX-2.5",
              include: ["diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
                        "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "gemma4-12b-ltx-v1/tokenizer.json"],
              preparation: "权重来自 comfyicu，tokenizer 单独来自 mlx-community/ltx-2.5-mlx；不下载第二份编码器。当前仅文生视频、无音轨。任一仓库没有 ModelScope 镜像时，请选择 Hugging Face。",
              supplements: [.init(repository: "mlx-community/ltx-2.5-mlx", include: ["gemma4-12b-ltx-v1/tokenizer.json"])]),
        .init(modelID: "minimax-h3-turbo", repository: "MiniMaxAI/MiniMax-H3", include: ["FL2VA/"],
              preparation: "默认仅下载 FL2VA（文生视频、首尾帧）。基础权重还需配套 LightX2V Turbo LoRA，并通过离线 prepare-lora 工具生成预融合模型；参考视频另需 Ref2VA 的独立模型与 Turbo 来源记录。"),
        .init(modelID: "wan2.1-1.3b-qad", repository: "FastVideo/FastMetal-1.3B-QAD", include: ["model_index.json", "mlx_dit.json", "mlx_dit.safetensors", "scheduler/", "text_encoder/", "tokenizer/", "vae/"],
              preparation: "原生 Wan：保留基础 mlx_dit 权重，并离线准备 vae/taew2_1.safetensors。运行时不需要 Python；下载源中的原始 VAE 不等于已转换的 TAEHV。")
    ]
}
