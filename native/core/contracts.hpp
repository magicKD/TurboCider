#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tc {
struct InputAsset {
    std::string kind, role, path, text, audio_path;
    bool include_audio = true;
    float strength = 0.75f;
};
struct LoRAAsset {
    std::string path;
    float strength = 1.0f;
    std::string role = "transformer";
};
struct Request {
    std::string model = "flux2-klein-4b";
    std::string operation = "image.generate";
    std::string prompt, output, execution = "gpu", dump, noise_path, ane_manifest;
    // Optional encoder-only Core ML manifest.  This is intentionally separate
    // from `ane_manifest`, which selects a denoiser/DiT partition.
    std::string encoder_ane_manifest;
    std::string profile, residency = "resident", quantized_cache;
    std::string ltx_backend = "auto";
    std::string profile_identity;
    std::string model_variant = "auto";
    // Optional native Qwen3.5 PE-T2I prompt enhancer installation. Empty
    // keeps the normal Comfy Qwen21 text path; PE edit/vision is not implied.
    std::string prompt_enhancer_path;
    bool prompt_enhance = false;
    // Explicit opt-in only: native PE-I2I with experimental FP32 vision.
    // Does not qualify BF16 PE vision or edit-image quality.
    bool prompt_enhance_edit_experimental = false;
    std::string lora_strategy = "auto";
    bool vsa = false;
    // Preserve the JSON decimal through top-k computation. A float-rounded
    // 0.9 retains 29 of 280 tiles after ceil(), while FastVideo's double
    // parameter correctly retains 28.
    double vsa_sparsity = 0.9;
    int vsa_tile_size = 64;
    std::string vsa_prefix_mode = "exempt";
    int vsa_dense_first_n_steps = 0;
    std::vector<int> vsa_dense_layers;
    std::string vsa_impl = "auto";
    uint64_t memory_budget_bytes = 0, allocator_cache_bytes = 512ull << 20;
    int warmup_iterations = 0;
    std::vector<InputAsset> inputs;
    std::vector<LoRAAsset> loras;
    int width = 512, height = 512, steps = 4, frames = 1, fps = 24;
    uint64_t seed = 42;
    bool compile_gpu = false;
    bool dynamic_text = true, allow_approximation = false, audio = true;
    bool ltx_fast_av = true;
    // Strictly equivalent Video attention command batching. This is opt-in
    // because its full-request gain is device/schedule sensitive.
    bool ltx_video_attention_batch = false;
    // LTX quality-preserving GPU defaults stay dense.  These fields expose
    // the ltx-mac approximate fast path explicitly; the parser rejects it
    // unless allow_approximation is true.
    bool ltx_sol_stage1 = false;
    bool ltx_sol_stage2 = false;
    double ltx_sol_tau = 0.5;
    int ltx_sparse_mode = 0;
    int ltx_sparse_radius = 1;
    int ltx_sparse_anchor_stride = 0;
    int ltx_sparse_tokens_per_frame = 0;
    int ltx_sparse_keep_blocks = 0;
    int ltx_sol_dense_edge_blocks = 1;
    int ltx_sol_dense_edge_steps = 1;
    int ltx_stage2_text_rows = 0;
    // Reserved legacy GGUF hint. Native backends reject unsupported offload;
    // this never selects or launches an external inference engine.
    bool streaming_offload = false;
};
struct Stage {
    std::string id;
    std::vector<std::string> dependencies;
    int iterations = 1;
};
struct Recipe {
    std::string model;
    std::vector<Stage> stages;
    bool executable = false;
};
} // namespace tc
