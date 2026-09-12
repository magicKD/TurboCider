#include "../runtime/session.hpp"

#include <algorithm>
#include <cmath>

namespace tc {
std::unique_ptr<ModelSession> create_h3_mlx(const std::filesystem::path &);
std::unique_ptr<ModelSession> create_h3_mlx_vsa(const std::filesystem::path &);

namespace {
constexpr const char *dense_id = "minimax-h3-fasth3-mlx-int6";
constexpr const char *vsa_id = "minimax-h3-fasth3-mlx-int6-vsa";

Recipe recipe(const char *model) {
    return Recipe{
        model,
        {{"text_encode", {}},
         {"av_denoise", {"text_encode"}, 4},
         {"video_decode", {"av_denoise"}},
         {"audio_decode", {"av_denoise"}},
         {"mux", {"video_decode", "audio_decode"}}},
        true,
    };
}

void validate_common(const Request &r, const char *model) {
    require(r.model_variant == "auto" || r.model_variant == model,
            "model_variant does not match FastH3 MLX INT6 profile");
    require(r.operation == "video.generate",
            "FastH3 MLX INT6 currently supports video.generate only");
    require(r.inputs.empty(), "FastH3 MLX T2VA does not accept media inputs");
    require(r.loras.empty(),
            "FastH3 MLX INT6 LoRA is not part of the validated profile");
    require(r.steps == 4, "FastH3 MLX INT6 requires exactly four steps");
    require(r.fps == 24, "FastH3 MLX INT6 requires 24 fps");
    require(r.width % 32 == 0 && r.height % 32 == 0 &&
                int64_t(r.width) * r.height <= 768 * 1344,
            "FastH3 MLX canvas must be multiples of 32 within 768*1344 pixels");
    require(r.frames >= 22 && r.frames <= 362 && (r.frames - 5) % 17 == 0,
            "FastH3 MLX frames must be 5+17n, between 22 and 362");
    require(r.execution == "gpu" || r.execution == "auto",
            "FastH3 MLX INT6 supports gpu or auto execution only");
    require(r.residency == "component_staged" || r.residency == "resident",
            "FastH3 MLX uses component-staged or resident phase ownership");
    require(r.quantized_cache.empty(),
            "FastH3 MLX INT6 does not use the legacy C/Metal quantized cache");
    if (!r.noise_path.empty())
        require(std::filesystem::path(r.noise_path).extension() == ".safetensors",
                "FastH3 MLX noise fixtures must be safetensors files");
}

void validate_vsa(const Request &r) {
    require(std::isfinite(r.vsa_sparsity) && r.vsa_sparsity >= 0.0 &&
                r.vsa_sparsity < 1.0,
            "FastH3 VSA sparsity must be in [0, 1)");
    require(r.vsa_tile_size == 64 || r.vsa_tile_size == 256,
            "FastH3 VSA tile_size must be 64 or 256");
    require(r.vsa_prefix_mode == "exempt" || r.vsa_prefix_mode == "compete",
            "FastH3 VSA prefix_mode must be exempt or compete");
    require(r.vsa_dense_first_n_steps >= 0 && r.vsa_dense_first_n_steps <= 4,
            "FastH3 VSA dense_first_n_steps must be 0...4");
    require(r.vsa_impl == "auto" || r.vsa_impl == "reference" ||
                r.vsa_impl == "simd",
            "FastH3 VSA impl must be auto, reference, or simd");
    for (int layer : r.vsa_dense_layers)
        require(layer >= 0 && layer < 50,
                "FastH3 VSA dense layer indices must be 0...49");
    auto layers = r.vsa_dense_layers;
    std::sort(layers.begin(), layers.end());
    require(std::adjacent_find(layers.begin(), layers.end()) == layers.end(),
            "FastH3 VSA dense layer indices must be unique");
}

ModelDescriptor descriptor(bool vsa) {
    ModelDescriptor d;
    d.id = vsa ? vsa_id : dense_id;
    d.name = vsa ? "MiniMax H3 FastH3 MLX INT6 VSA"
                 : "MiniMax H3 FastH3 MLX INT6";
    d.executable = true;
    d.operations = {"video.generate"};
    d.inputs = {"text"};
    d.output = "video";
    d.steps = 4;
    d.frames = 124;
    d.width = 832;
    d.height = 480;
    d.fps = 24;
    d.default_audio = true;
    d.default_residency = "component_staged";
    d.backend = "mlx_cpp_metal";
    d.runtime_dependency = "ModelScope FastH3 student + MLX C++";
    d.parallel_strategy = vsa
        ? "streamed Qwen3-VL + INT6/g64 VSA DiT + staged full H3 VAEs"
        : "streamed Qwen3-VL + INT6/g64 dense DiT + staged full H3 VAEs";
    d.audio_output = true;
    d.native_audio_output_candidate = true;
    d.native_audio_vae_candidate = true;
    d.audio_capability = "full-h3-audio-vae-32khz-stereo";
    d.executor_operations = {"video.generate"};
    d.candidate_limitations = vsa
        ? std::vector<std::string>{
              "T2VA only; requires ModelScope VSA gate checkpoint",
              "reference gather+SDPA is implemented; SIMD kernel qualification remains pending",
              "FastVideo ABBA and perceptual quality gates remain required",
              "dense and old C/Metal profiles remain unchanged"}
        : std::vector<std::string>{
              "T2VA only; FL2VA/Ref2VA/keyframes/LoRA/VSA require another profile",
              "FastVideo ABBA performance and final perceptual quality gates remain required",
              "old minimax-h3-turbo C/Metal module remains the compatibility default"};
    return d;
}
} // namespace

ModelModule h3_mlx_module() {
    return {
        dense_id,
        [] { return recipe(dense_id); },
        [](const Request &r) {
            validate_common(r, dense_id);
            require(!r.vsa, "use minimax-h3-fasth3-mlx-int6-vsa for VSA");
            require(r.vsa_sparsity == 0.9 && r.vsa_tile_size == 64 &&
                        r.vsa_prefix_mode == "exempt" &&
                        r.vsa_dense_first_n_steps == 0 &&
                        r.vsa_dense_layers.empty() && r.vsa_impl == "auto",
                    "VSA parameters are not accepted by the dense FastH3 profile");
        },
        create_h3_mlx,
        [] { return descriptor(false); },
    };
}

ModelModule h3_mlx_vsa_module() {
    return {
        vsa_id,
        [] { return recipe(vsa_id); },
        [](const Request &r) {
            validate_common(r, vsa_id);
            validate_vsa(r);
            require(r.vsa, "VSA profile requires parameters.vsa=true");
        },
        create_h3_mlx_vsa,
        [] { return descriptor(true); },
    };
}
} // namespace tc
