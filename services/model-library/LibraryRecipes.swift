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
    var modelScopePins: [HubFile]? = nil

    func request(modelID: String, provider: HubProvider = .modelscope, componentKind: String? = nil) throws -> LibraryDownloadRequest {
        if modelScopePins != nil && provider != .modelscope {
            throw LibraryFailure(message: "此版本使用 ModelScope 固定权重，请选择 ModelScope。")
        }
        let request = LibraryDownloadRequest(modelID: modelID, repository: repository, provider: provider,
            name: title, include: include, supplements: supplements, destinations: destinations,
            pinnedVariantID: modelScopePins == nil ? nil : id)
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
    private static let qwenTurboConvRot: LibraryVariant = {
        let dit = "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"
        let encoder = "text_encoders/qwen3vl_8b_int8_convrot.safetensors"
        let companions = "5dc8a8352312860d67e07cc9bee310dc7ed3e55c"
        // Per-file ModelScope identities from the verified download receipts.
        // The native loader consumes tokenizer.json directly from processor/.
        let pins = [
            HubFile(path: dit, size: 7_256_787_829,
                sha256: "8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c",
                revision: "8c798285d642b81a1af415b70f2e444e809820d7"),
            HubFile(path: encoder, size: 9_350_798_360,
                sha256: "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f",
                revision: companions),
            HubFile(path: qwenVAE, size: 675_509_688,
                sha256: "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9",
                revision: companions),
            HubFile(path: qwenTokenizer, size: 11_422_654,
                sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                revision: "c491be04a88fa35fbd86443f78f9c3002a503677", sourceRepository: qwenTokenizerSource.repository),
        ]
        return LibraryVariant(id: "turbo-int8-convrot", title: "Turbo · 8 步 · 双 INT8 ConvRot",
            repository: qwenRepository, include: pins.map(\.path), components: [
                .init(kind: "dit", title: "Qwen Image 2.1 Turbo DiT", format: "safetensors", precision: "INT8 ConvRot",
                    repository: qwenRepository, include: [dit], note: "使用 Turbo 固定 8 步调度。"),
                .init(kind: "text_encoder", title: "Qwen3-VL-8B 编码器", format: "safetensors", precision: "INT8 ConvRot",
                    repository: qwenRepository, include: [encoder], note: "与 Turbo DiT 配套。"),
            ] + qwenSharedComponents.filter { $0.kind != "text_encoder" }, runnable: true,
            preparation: "Turbo 固定 8 步。支持原生 GPU 流式文生图，使用 INT8 ConvRot DiT 和编码器、原始 VAE 与官方 Tokenizer。请确认 Qwen 研究用途许可。",
            modelScopePins: pins)
    }()
    static let ltxAudioPin = HubFile(path: "vae/ltx-2.5-audio-vae-bf16.safetensors", size: 364866540,
        sha256: "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
        revision: "5a5da253b8bd7ae649a8c7acdec41e6c1a1cffda")
    private static let ltxINT8Variant: LibraryVariant = {
        let pins = [
            HubFile(path: "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", size: 21504034224,
                sha256: "c4279eeff115cbeaca494bd2183e7d768c38fe85a184dc6afbb7159157c44334",
                revision: "4b205d041cd006b4b268e5edfd19cdbcaed6732f"),
            HubFile(path: "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", size: 15372969374,
                sha256: "6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f",
                revision: "a14f34a9a04bc0adf394b608773c94899cd9723b"),
            HubFile(path: "vae/ltx-2.5-video-vae-conv-bf16.safetensors", size: 1452269922,
                sha256: "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d",
                revision: "65c57ea625e4b1da955f35f5a7d379c067554811"),
            ltxAudioPin,
            HubFile(path: "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", size: 995778752,
                sha256: "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
                revision: "4a7b2baed1b4366885c3c47f8208c0a78a85ddfa"),
            HubFile(path: "gemma4-12b-ltx-v1/tokenizer.json", size: 32169626,
                sha256: "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
                revision: "9677e48834a7ab2582daaa86f615b6fde846f8f1", sourceRepository: "mlx-community/ltx-2.5-mlx"),
        ]
        let components = [
            ("dit", "LTX 2.5 Distilled", "INT8 ConvRot", "diffusion_models/"),
            ("text_encoder", "Gemma 4 12B", "INT8 ConvRot", "text_encoders/"),
            ("vae", "视频 / 音频 VAE 与 Vocoder", "BF16", "vae/"),
            ("upscaler", "空间放大模型", "BF16", "latent_upscale_models/"),
            ("tokenizer", "Tokenizer", "JSON", "gemma4-12b-ltx-v1/")
        ].map { kind, title, precision, prefix in
            LibraryVariantComponent(kind: kind, title: title, format: kind == "tokenizer" ? "JSON" : "safetensors",
                precision: precision, repository: kind == "tokenizer" ? "mlx-community/ltx-2.5-mlx" : "Lightricks/LTX-2.5",
                include: pins.map(\.path).filter { $0.hasPrefix(prefix) }, note: "LTX 2.5 原始配套文件。")
        }
        return LibraryVariant(id: "distilled-int8-convrot", title: "Distilled · INT8 ConvRot",
            repository: "Lightricks/LTX-2.5", include: pins.map(\.path),
            components: components, runnable: true,
            preparation: "ModelScope 固定版本与 SHA256 校验。INT8 DiT 和 Gemma；M5 及以上、macOS 26.4 及以上自动使用 W8A8，其余设备保留 BF16 激活。包含视频 VAE、音频 VAE / Vocoder 和空间放大模型，可同步输出音视频。",
            modelScopePins: pins)
    }()
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
        return [baseline, qwenTurboConvRot] + quants.map { quant in
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
        .init(modelID: "ltx-2.5-distilled", repository: "Lightricks/LTX-2.5",
              include: ["diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
                        "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "gemma4-12b-ltx-v1/tokenizer.json"],
              preparation: ltxINT8Variant.preparation,
              supplements: [.init(repository: "mlx-community/ltx-2.5-mlx", include: ["gemma4-12b-ltx-v1/tokenizer.json"])], variants: [ltxINT8Variant]),
        .init(modelID: "minimax-h3-turbo", repository: "MiniMaxAI/MiniMax-H3", include: ["FL2VA/"],
              preparation: "默认仅下载 FL2VA（文生视频、首尾帧）。基础权重还需配套 LightX2V Turbo LoRA，并通过离线 prepare-lora 工具生成预融合模型；参考视频另需 Ref2VA 的独立模型与 Turbo 来源记录。"),
        .init(modelID: "wan2.1-1.3b-qad", repository: "FastVideo/FastMetal-1.3B-QAD", include: ["model_index.json", "mlx_dit.json", "mlx_dit.safetensors", "scheduler/", "text_encoder/", "tokenizer/", "vae/"],
              preparation: "原生 Wan：保留基础 mlx_dit 权重，并离线准备 vae/taew2_1.safetensors。运行时不需要 Python；下载源中的原始 VAE 不等于已转换的 TAEHV。")
    ]
}
