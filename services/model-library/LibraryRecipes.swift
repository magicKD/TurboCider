import Foundation

struct LibraryRecipe: Codable, Sendable {
    var modelID: String
    var repository: String
    var include: [String]
    var preparation: String
    var supplements: [LibrarySupplement] = []
    static let all: [LibraryRecipe] = [
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
