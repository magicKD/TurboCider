#include "bridge.hpp"

#include "../../media/audio.hpp"
#include "../../media/video.hpp"
#include "../../models/h3_mlx/audio_vae.hpp"
#include "../../models/h3_mlx/conditioner.hpp"
#include "../../models/h3_mlx/pipeline.hpp"
#include "../../models/h3_mlx/prompt_cache.hpp"
#include "../../models/h3_mlx/video_vae.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>

namespace tc {
namespace {

using h3_mlx::AudioVAE;
using h3_mlx::Conditioner;
using h3_mlx::DenoiseOptions;
using h3_mlx::DiTConfig;
using h3_mlx::Pipeline;
using h3_mlx::VideoVAE;

double seconds_since(Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

std::filesystem::path required_directory(
    const std::vector<std::filesystem::path> &candidates,
    const std::string &label) {
    for (const auto &candidate : candidates)
        if (std::filesystem::is_directory(candidate))
            return std::filesystem::absolute(candidate).lexically_normal();
    throw std::runtime_error("FastH3 MLX " + label + " directory is missing");
}

std::filesystem::path component_path(const std::filesystem::path &root,
                                     const std::string &name) {
    const auto parent = root.parent_path();
    return required_directory({root / name,
                               root / "FastH3-ModelScope" / name,
                               parent / "FastH3-ModelScope" / name}, name);
}

std::filesystem::path checkpoint_path(const std::filesystem::path &root, bool vsa) {
    const auto parent = root.parent_path();
    std::vector<std::filesystem::path> candidates;
    if (vsa) {
        candidates = {root / "vsa-int6", root / "int6-vsa",
                      root / "vsa-int6" / "int6", root / "int6-vsa" / "int6",
                      root / "FastH3-MLX" / "vsa-int6",
                      root / "FastH3-MLX" / "vsa-int6" / "int6",
                      parent / "FastH3-MLX" / "vsa-int6",
                      parent / "FastH3-MLX" / "vsa-int6" / "int6",
                      root / ".." / "FastH3-MLX" / "vsa-int6"};
    } else {
        candidates = {root / "int6", root / "FastH3-MLX" / "int6",
                      parent / "FastH3-MLX" / "int6",
                      root / ".." / "FastH3-MLX" / "int6"};
    }
    for (const auto &candidate : candidates) {
        const auto absolute = std::filesystem::absolute(candidate).lexically_normal();
        if (std::filesystem::is_regular_file(absolute / "mlx_h3_dit.json") &&
            std::filesystem::is_regular_file(absolute / "mlx_h3_dit.safetensors"))
            return absolute;
    }
    throw std::runtime_error(
        vsa ? "FastH3 MLX VSA INT6 checkpoint is missing; expected vsa-int6/ beside the ModelScope tree"
            : "FastH3 MLX INT6 checkpoint is missing; expected int6/ beside the ModelScope tree");
}

std::filesystem::path prompt_cache_root(const std::filesystem::path &model_root) {
    if (const char *configured = std::getenv("TURBOCIDER_H3_PROMPT_CACHE_DIR"))
        if (*configured) return std::filesystem::absolute(configured);
    if (const char *home = std::getenv("HOME"))
        if (*home)
            return std::filesystem::path(home) / "Library" / "Caches" /
                   "TurboCider" / "h3-prompt-cache";
    return model_root / ".turbocider-prompt-cache";
}

void load_noise_fixture(const std::filesystem::path &path,
                        DenoiseOptions &options) {
    require(std::filesystem::is_regular_file(path),
            "FastH3 MLX noise fixture is missing: " + path.string());
    auto loaded = mx::load_safetensors(path.string()).first;
    auto video = loaded.find("video_noise");
    auto audio = loaded.find("audio_noise");
    require(video != loaded.end() && audio != loaded.end(),
            "FastH3 MLX noise fixture must contain video_noise and audio_noise");
    options.video_noise = mx::astype(video->second, mx::float32);
    options.audio_noise = mx::astype(audio->second, mx::float32);
    mx::eval(*options.video_noise, *options.audio_noise);
}

void require_modelscope_provenance(const std::filesystem::path &root,
                                   bool vsa) {
    const auto manifest = root / "modelscope_download.json";
    require(std::filesystem::is_regular_file(manifest),
            "FastH3 MLX requires modelscope_download.json provenance");
    auto value = read_json(manifest);
    require(string_value(value, @"model_id") ==
                (vsa ? "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
                     : "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree") &&
                string_value(value, @"endpoint") == "https://modelscope.cn",
            "FastH3 MLX checkpoint provenance is not the approved ModelScope repository");
}

std::vector<uint8_t> video_bytes(const Tensor &pixels, int frames, int height,
                                 int width) {
    require(pixels.ndim() == 5 && pixels.shape() ==
                mx::Shape{1, 3, frames, height, width},
            "FastH3 MLX video decoder returned an unexpected shape");
    auto value = mx::contiguous(pixels);
    mx::eval(value);
    const float *source = value.data<float>();
    std::vector<uint8_t> output(static_cast<size_t>(frames) * height * width * 3);
    for (int frame = 0; frame < frames; ++frame) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t pixel =
                    (static_cast<size_t>(frame) * height * width +
                     static_cast<size_t>(y) * width + x) * 3;
                for (int channel = 0; channel < 3; ++channel) {
                    const size_t index =
                        (static_cast<size_t>(channel) * frames + frame) *
                            static_cast<size_t>(height) * width +
                        static_cast<size_t>(y) * width + x;
                    output[pixel + static_cast<size_t>(channel)] =
                        static_cast<uint8_t>(std::lround(
                            std::clamp(source[index], 0.f, 1.f) * 255.f));
                }
            }
        }
    }
    return output;
}

std::vector<float> audio_interleaved(const Tensor &waveform) {
    require(waveform.ndim() == 3 && waveform.shape(1) == 1 &&
                waveform.shape(0) == 2,
            "FastH3 MLX audio decoder returned an unexpected shape");
    auto value = mx::contiguous(waveform);
    mx::eval(value);
    const float *source = value.data<float>();
    const size_t samples = static_cast<size_t>(value.shape(2));
    std::vector<float> output(samples * 2);
    for (size_t sample = 0; sample < samples; ++sample) {
        output[sample * 2] = source[sample];
        output[sample * 2 + 1] = source[samples + sample];
    }
    return output;
}

class H3MLXSession final : public ModelSession {
    std::filesystem::path root_;
    std::filesystem::path text_encoder_;
    std::filesystem::path tokenizer_;
    std::filesystem::path video_vae_root_;
    std::filesystem::path audio_vae_root_;
    std::filesystem::path checkpoint_root_;
    bool resolved_vsa_ = false;
    std::unique_ptr<Conditioner> conditioner_;
    Pipeline pipeline_;

    void resolve_components(bool vsa) {
        if (!text_encoder_.empty() && resolved_vsa_ == vsa) return;
        if (!text_encoder_.empty()) unload();
        root_ = std::filesystem::absolute(root_).lexically_normal();
        require_modelscope_provenance(root_, vsa);
        text_encoder_ = component_path(root_, "text_encoder");
        tokenizer_ = component_path(root_, "tokenizer");
        video_vae_root_ = component_path(root_, "vae");
        audio_vae_root_ = component_path(root_, "audio_vae");
        checkpoint_root_ = checkpoint_path(root_, vsa);
        resolved_vsa_ = vsa;
        conditioner_ = std::make_unique<Conditioner>(text_encoder_, tokenizer_);
    }

  public:
    explicit H3MLXSession(const std::filesystem::path &root) : root_(root) {
        require(std::filesystem::is_directory(root_),
                "FastH3 ModelScope root is missing: " + root_.string());
    }

    bool uses_parent_mlx() const override { return true; }

    void unload() override {
        pipeline_.unload();
        conditioner_.reset();
        resolved_vsa_ = false;
        text_encoder_.clear();
        tokenizer_.clear();
        video_vae_root_.clear();
        audio_vae_root_.clear();
        checkpoint_root_.clear();
        mx::clear_cache();
    }

    RunResult generate(const Request &request, const Event &event,
                       std::atomic<bool> &cancelled) override {
        require(request.model == "minimax-h3-fasth3-mlx-int6" ||
                    request.model == "minimax-h3-fasth3-mlx-int6-vsa",
                "request model differs from FastH3 MLX session");
        const bool use_vsa = request.model == "minimax-h3-fasth3-mlx-int6-vsa";
        auto plan = make_plan(request);
        require(!request.prompt.empty() && !request.output.empty(),
                "FastH3 MLX requires prompt and output");
        require(std::filesystem::path(request.output).extension() == ".mp4",
                "FastH3 MLX output must be .mp4");
        checkpoint(cancelled);
        resolve_components(use_vsa);

        const auto cache_root = prompt_cache_root(root_);
        const auto cache_identity = h3_mlx::prompt_cache_identity(
            text_encoder_, tokenizer_, request.prompt);
        const bool cache_hit =
            std::filesystem::is_regular_file(cache_root / (cache_identity + ".safetensors")) &&
            std::filesystem::is_regular_file(cache_root / (cache_identity + ".json"));

        const auto request_started = Clock::now();
        mx::reset_peak_memory();
        const auto condition_started = Clock::now();
        auto conditioning = conditioner_->encode_prompt_cached(
            request.prompt, cache_root, event, cancelled);
        const double condition_seconds = seconds_since(condition_started);
        const uint64_t condition_peak_bytes = mx::get_peak_memory();
        checkpoint(cancelled);

        const auto denoise_started = Clock::now();
        mx::reset_peak_memory();
        pipeline_.load(checkpoint_root_, event, cancelled);
        const uint64_t denoise_load_peak_bytes = mx::get_peak_memory();
        const DiTConfig dit_config = pipeline_.checkpoint().config();
        const std::string checkpoint_sha =
            pipeline_.checkpoint().identity().source_sha256;
        DenoiseOptions options;
        options.width = request.width;
        options.height = request.height;
        options.frames = request.frames;
        options.fps = request.fps;
        options.steps = request.steps;
        options.seed = request.seed;
        options.vsa.enabled = use_vsa;
        options.vsa.sparsity = request.vsa_sparsity;
        options.vsa.tile_size = request.vsa_tile_size;
        options.vsa.prefix_mode = h3_mlx::vsa_prefix_mode(request.vsa_prefix_mode);
        options.vsa.dense_first_n_steps = request.vsa_dense_first_n_steps;
        options.vsa.dense_layers = request.vsa_dense_layers;
        options.vsa.implementation = h3_mlx::vsa_implementation(request.vsa_impl);
        if (!request.noise_path.empty())
            load_noise_fixture(request.noise_path, options);
        auto denoised = pipeline_.denoise(
            conditioning.hidden_states, conditioning.token_tags, options,
            event, cancelled);
        mx::eval(denoised.video_rows, denoised.audio_rows);
        auto video_latents = h3_mlx::unpatchify_video_rows(
            denoised.video_rows, denoised.layout, dit_config);
        auto audio_latents = h3_mlx::unpack_audio_rows(
            denoised.audio_rows, denoised.layout, dit_config);
        mx::eval(video_latents, audio_latents);
        if (!request.dump.empty()) {
            const auto dump_root = std::filesystem::absolute(request.dump);
            std::filesystem::create_directories(dump_root);
            Tensor token_tags(conditioning.token_tags.data(),
                              {int(conditioning.token_tags.size())}, mx::int32);
            mx::save_safetensors(
                (dump_root / "h3_mlx_e2e.safetensors").string(),
                {{"conditioning", mx::astype(conditioning.hidden_states, mx::float32)},
                 {"token_tags", token_tags},
                 {"video_rows", mx::astype(denoised.video_rows, mx::float32)},
                 {"audio_rows", mx::astype(denoised.audio_rows, mx::float32)},
                 {"video_latents", mx::astype(video_latents, mx::float32)},
                 {"audio_latents", mx::astype(audio_latents, mx::float32)}});
        }
        const double denoise_seconds = seconds_since(denoise_started);
        const auto denoise_metrics = denoised.metrics;
        const uint64_t denoise_peak_bytes =
            std::max(denoise_load_peak_bytes, denoise_metrics.peak_bytes);
        pipeline_.unload();
        mx::clear_cache();
        checkpoint(cancelled);

        std::vector<uint8_t> rgb;
        mx::reset_peak_memory();
        const auto video_started = Clock::now();
        {
            VideoVAE video_vae;
            video_vae.load(video_vae_root_, event, cancelled);
            require(video_vae.config().latent_channels == dit_config.latent_channels,
                    "FastH3 DiT/video VAE latent channels do not match");
            auto decoded = video_vae.decode(
                video_vae.denormalize_latents(video_latents), request.frames,
                request.height, request.width, true, event, cancelled);
            rgb = video_bytes(video_vae.denormalize_pixels(decoded),
                              request.frames, request.height, request.width);
            video_vae.unload();
        }
        const double video_seconds = seconds_since(video_started);
        const uint64_t video_peak_bytes = mx::get_peak_memory();
        mx::clear_cache();
        checkpoint(cancelled);

        std::vector<float> audio;
        double audio_seconds = 0.0;
        uint64_t audio_peak_bytes = 0;
        if (request.audio) {
            mx::reset_peak_memory();
            const auto audio_started = Clock::now();
            AudioVAE audio_vae;
            audio_vae.load(audio_vae_root_, event, cancelled);
            auto decoded = audio_vae.decode(
                audio_vae.denormalize_latents(audio_latents), event, cancelled);
            audio = audio_interleaved(decoded);
            audio_vae.unload();
            audio_seconds = seconds_since(audio_started);
            audio_peak_bytes = mx::get_peak_memory();
            mx::clear_cache();
        }
        checkpoint(cancelled);

        const uint64_t overall_peak_bytes = std::max(
            {condition_peak_bytes, denoise_peak_bytes,
             video_peak_bytes, audio_peak_bytes});
        const uint64_t active_bytes = mx::get_active_memory();

        const auto output = std::filesystem::absolute(request.output);
        const auto video_only = output.string() + ".h3-mlx-video-only.mp4";
        const auto export_started = Clock::now();
        try {
            write_video_rgb24(video_only, rgb.data(), request.frames,
                              request.width, request.height, request.fps);
            if (request.audio) {
                require(!audio.empty(), "FastH3 audio decoder produced no samples");
                mux_video_with_audio(video_only, output, audio.data(),
                                     audio.size() / 2, AudioVAE::sampling_rate, 2);
                std::error_code ignored;
                std::filesystem::remove(video_only, ignored);
            } else {
                std::filesystem::create_directories(output.parent_path());
                std::filesystem::rename(video_only, output);
            }
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove(video_only, ignored);
            throw;
        }
        const double export_seconds = seconds_since(export_started);

        NSDictionary *value = @{
            @"schema_version": @1,
            @"model": @(request.model.c_str()),
            @"profile": @(request.model.c_str()),
            @"backend": @"mlx_cpp_metal",
            @"precision": @"int6_g64_bf16_activation",
            @"checkpoint": @(checkpoint_root_.string().c_str()),
            @"checkpoint_sha256": @(checkpoint_sha.c_str()),
            @"schedule_id": @"fasth3-v0.2-v12-a3-4step",
            @"decoder": @"full-h3-vae",
            @"prompt_cache_hit": @(cache_hit),
            @"noise_fixture": request.noise_path.empty()
                ? [NSNull null] : @(std::filesystem::absolute(request.noise_path).string().c_str()),
            @"tensor_dump": request.dump.empty()
                ? [NSNull null] : @(std::filesystem::absolute(request.dump).string().c_str()),
            @"text_tokens": @(conditioning.tokens),
            @"valid_text_tokens": @(conditioning.tokens),
            @"timings_seconds": @{
                @"request_wall": @(seconds_since(request_started)),
                @"text_encode": @(condition_seconds),
                @"denoise": @(denoise_seconds),
                @"video_decode": @(video_seconds),
                @"audio_decode": @(audio_seconds),
                @"mux": @(export_seconds),
            },
            @"mlx_peak_bytes": @(overall_peak_bytes),
            @"mlx_active_bytes": @(active_bytes),
            @"memory": @{
                @"condition_peak_bytes": @(condition_peak_bytes),
                @"denoise_peak_bytes": @(denoise_peak_bytes),
                @"video_decode_peak_bytes": @(video_peak_bytes),
                @"audio_decode_peak_bytes": @(audio_peak_bytes),
                @"scope": @"MLX allocator; excludes process RSS and media encoders",
            },
            @"quant": @{
                @"bits": @6,
                @"group_size": @64,
                @"dq_gemm_floor_m": @(denoise_metrics.affine_dq_gemm_min_rows),
                @"qmm_calls": @(denoise_metrics.quantized_matmul_calls),
                @"dq_gemm_calls": @(denoise_metrics.dequantized_gemm_calls),
            },
            @"vsa": denoise_metrics.vsa ? @{
                @"enabled": @YES,
                @"configured_sparsity": @(denoise_metrics.vsa->configured_sparsity),
                @"achieved_sparsity": @(denoise_metrics.vsa->achieved_sparsity),
                @"video_keep": @(denoise_metrics.vsa->video_keep),
                @"tile_size": @(denoise_metrics.vsa->tile_size),
                @"num_prefix_tiles": @(denoise_metrics.vsa->num_prefix_tiles),
                @"num_video_tiles": @(denoise_metrics.vsa->num_video_tiles),
                @"attention_calls": @(denoise_metrics.vsa->attention_calls),
                @"sparse_calls": @(denoise_metrics.vsa->sparse_calls),
                @"prefix_mode": @(denoise_metrics.vsa->prefix_mode.c_str()),
                @"implementation": @(denoise_metrics.vsa->implementation.c_str()),
                @"dense_fallback_reason": denoise_metrics.vsa->dense_fallback_reason.empty()
                    ? (id)[NSNull null] : @(denoise_metrics.vsa->dense_fallback_reason.c_str()),
            } : @{
                @"enabled": @NO,
                @"configured_sparsity": @0.0,
                @"achieved_sparsity": @0.0,
            },
            @"output": @{
                @"width": @(request.width), @"height": @(request.height),
                @"frames": @(request.frames), @"fps": @(request.fps),
                @"audio_rate": @(AudioVAE::sampling_rate), @"audio_channels": @2,
            },
        };
        auto result = native_run_result(value, request, plan);
        result.backend = "mlx_cpp_metal";
        result.precision = "int6_g64_bf16_activation";
        result.checkpoint = checkpoint_root_.string();
        result.prompt_cache_hit = cache_hit;
        result.actual_steps = 4;
        return result;
    }
};

} // namespace

std::unique_ptr<ModelSession> create_h3_mlx(const std::filesystem::path &root) {
    return std::make_unique<H3MLXSession>(root);
}

std::unique_ptr<ModelSession> create_h3_mlx_vsa(const std::filesystem::path &root) {
    return std::make_unique<H3MLXSession>(root);
}
} // namespace tc
