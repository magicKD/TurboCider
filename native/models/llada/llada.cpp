#include "llada.hpp"

#include "../flux2/flux_vae_ops.hpp"
#include "../../backends/coreml.hpp"
#include "../../media/image.hpp"
#include "../../platform/apple/platform.hpp"
#include "../../runtime/residency.hpp"

namespace tc {
namespace {

bool has_safetensors(const std::filesystem::path &directory) {
    if (!std::filesystem::is_directory(directory))
        return false;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
        if (!entry.is_symlink() && entry.is_regular_file() &&
            entry.path().extension() == ".safetensors")
            return true;
    return false;
}

int padded_rows(int value) {
    return ((value + 31) / 32) * 32;
}

Tensor initial_noise(const Request &request) {
    const int height = request.height / 16;
    const int width = request.width / 16;
    Tensor noise = mx::zeros({1, 128, height, width}, mx::float32);
    if (request.noise_path.empty()) {
        noise = mx::random::normal({1, 128, height, width}, mx::float32,
                                   0.f, 1.f, mx::random::key(request.seed));
    } else {
        require(std::filesystem::is_regular_file(request.noise_path),
                "LLaDA initial noise file missing: " + request.noise_path);
        auto loaded = mx::load_safetensors(request.noise_path);
        auto found = loaded.first.find("noise");
        if (found == loaded.first.end())
            found = loaded.first.find("tensor");
        if (found == loaded.first.end())
            found = loaded.first.find("latent_tensor");
        require(found != loaded.first.end(),
                "LLaDA initial noise must contain noise, tensor, or latent_tensor");
        noise = found->second;
        require(noise.ndim() == 4 && noise.shape(0) == 1 &&
                    noise.shape(1) == 128 && noise.shape(2) == height &&
                    noise.shape(3) == width,
                "LLaDA initial noise shape must be [1,128,H/16,W/16]");
    }
    // Match the official pipeline's observable BF16 rounding before the
    // sampler state is promoted back to FP32.
    return mx::astype(mx::astype(noise, mx::bfloat16), mx::float32);
}

Tensor scheduler_noise(const Request &request, int step, const mx::Shape &shape) {
    // A reference dump may place the exact stochastic scheduler samples next
    // to initial_noise.safetensors.  This is an acceptance-only input path; a
    // normal request generates deterministic per-step noise from the request
    // seed without depending on PyTorch's process-global RNG.
    if (!request.noise_path.empty()) {
        auto path = std::filesystem::path(request.noise_path).parent_path() /
                    ("scheduler_noise_" + std::to_string(step + 1) +
                     ".safetensors");
        if (std::filesystem::is_regular_file(path)) {
            auto loaded = mx::load_safetensors(path.string());
            auto found = loaded.first.find("noise");
            if (found == loaded.first.end())
                found = loaded.first.find("tensor");
            require(found != loaded.first.end(),
                    "LLaDA scheduler noise must contain noise or tensor: " +
                        path.string());
            require(found->second.shape() == shape,
                    "LLaDA scheduler noise geometry mismatch: " + path.string());
            return mx::astype(found->second, mx::float32);
        }
    }
    constexpr uint64_t kStepStride = 0x9e3779b97f4a7c15ULL;
    return mx::random::normal(shape, mx::float32, 0.f, 1.f,
                              mx::random::key(request.seed +
                                              kStepStride * uint64_t(step + 1)));
}

Tensor stochastic_step(const Tensor &sample, const Tensor &model_output,
                       float sigma, float sigma_next,
                       const std::optional<Tensor> &noise) {
    auto x0 = sample - Tensor(sigma, mx::float32) * model_output;
    if (sigma_next == 0.f)
        return x0;
    require(noise.has_value(), "LLaDA stochastic scheduler noise is missing");
    return Tensor(1.f - sigma_next, mx::float32) * x0 +
           Tensor(sigma_next, mx::float32) * *noise;
}

std::vector<float> sigmas(int steps) {
    require(steps > 0, "LLaDA requires at least one denoising step");
    // The reference pipeline explicitly supplies a uniform pre-shift grid
    // [1, 1-1/N, ..., 1/N] to FlowMatchEulerDiscreteScheduler.  Its scheduler
    // then applies the configured shift=3 transform:
    //     shifted = 3*sigma / (1 + 2*sigma)
    // and appends a terminal zero.  Keep the same schedule here instead of
    // silently using a hand-written approximation.
    std::vector<float> schedule;
    schedule.reserve(size_t(steps + 1));
    for (int index = 0; index < steps; ++index) {
        const float raw = 1.f - float(index) / float(steps);
        schedule.push_back(3.f * raw / (1.f + 2.f * raw));
    }
    schedule.push_back(0.f);
    return schedule;
}

Tensor decode_latent(const Tensor &latent, Weights &vae, int height, int width,
                     const Event &event, std::atomic<bool> &cancelled,
                     const std::string &dump) {
    const int h = height / 16;
    const int w = width / 16;
    auto packed = mx::astype(latent, mx::bfloat16);
    packed = packed * mx::sqrt(mx::reshape(vae.at("bn.running_var"),
                                           {1, 128, 1, 1}) +
                               Tensor(1e-4f, mx::bfloat16)) +
             mx::reshape(vae.at("bn.running_mean"), {1, 128, 1, 1});
    auto raw = mx::reshape(
        mx::transpose(mx::reshape(packed, {1, 32, 2, 2, h, w}),
                      {0, 1, 4, 2, 5, 3}),
        {1, 32, h * 2, w * 2});
    return flux_vae_decode_raw(raw, vae, height, width, event, cancelled, dump);
}

class LLaDAImageNative final : public ModelSession {
    std::filesystem::path root_;
    Tokenizer tokenizer_;
    Weights text_encoder_;
    Weights queryformer_;
    Weights text_projection_;
    Weights transformer_;
    Weights vae_;
    std::unique_ptr<HybridSession> hybrid_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)>
        hybrid_gpu_graph_;
    int hybrid_gpu_mlp_start_ = -1;
    std::optional<LLaDAConditioning> conditioning_;
    std::string cached_prompt_;

    bool prepare_conditioning(const Request &request, const Event &event,
                              std::atomic<bool> &cancelled) {
        if (conditioning_ && cached_prompt_ == request.prompt) {
            event("llada_text_cache_hit", 1, 1);
            return true;
        }
        conditioning_.reset();
        mx::clear_cache();
        auto tokens = tokenizer_.llada_image_prompt(request.prompt);
        conditioning_ = llada_encode_text(root_, tokens, text_encoder_, queryformer_,
                                          text_projection_, event, cancelled,
                                          request.dump);
        cached_prompt_ = request.prompt;
        return false;
    }

    RunResult run(const Request &requested, const Event &event,
                  std::atomic<bool> &cancelled, bool warmup,
                  bool load_only = false) {
        auto request = requested;
        if (request.execution == "auto")
            request.execution = "gpu";
        require(request.operation == "image.generate",
                "native LLaDA currently supports image.generate while editing is being ported");
        require(request.execution == "gpu" || request.execution == "gpu_ane",
                "native LLaDA execution must be gpu or gpu_ane");
        require(!request.prompt.empty() &&
                    (warmup || load_only || !request.output.empty()),
                "LLaDA prompt and output are required");
        require(warmup || load_only ||
                    std::filesystem::path(request.output).extension() == ".png",
                "LLaDA output must be a .png file");
        require(request.width % 16 == 0 && request.height % 16 == 0,
                "LLaDA dimensions must be multiples of 16");
        require(request.steps == 4, "LLaDA-Image-Turbo requires 4 steps");
        require(request.loras.empty(), "native LLaDA LoRA is not implemented yet");
        request.compile_gpu = request.execution == "gpu" &&
                              !std::getenv("TURBOCIDER_LLADA_EAGER_BLOCKS");

        auto begin = Clock::now();
        auto plan = make_plan(request);
        ResidencyPolicy::validate_budget(plan, device_info().physical_memory);
        mx::reset_peak_memory();
        mx::set_cache_limit(request.allocator_cache_bytes);
        auto text_start = Clock::now();
        const bool prompt_hit = prepare_conditioning(request, event, cancelled);
        if (std::getenv("TURBOCIDER_LLADA_REFERENCE_CONDITIONING")) {
            require(!request.noise_path.empty(),
                    "reference conditioning requires a diagnostic noise path");
            auto path = std::filesystem::path(request.noise_path).parent_path() /
                        "conditioning.safetensors";
            require(std::filesystem::is_regular_file(path),
                    "LLaDA reference conditioning is missing: " + path.string());
            auto loaded = mx::load_safetensors(path.string());
            auto found = loaded.first.find("conditioning");
            if (found == loaded.first.end())
                found = loaded.first.find("tensor");
            require(found != loaded.first.end(),
                    "LLaDA reference conditioning must contain conditioning or tensor");
            auto features = found->second;
            if (features.ndim() == 3) {
                require(features.shape(0) == 1,
                        "LLaDA reference conditioning batch must be one");
                features = mx::squeeze(features, 0);
            }
            require(features.ndim() == 2 && features.shape(1) == 2560,
                    "LLaDA reference conditioning geometry mismatch");
            conditioning_->features = mx::astype(features, mx::bfloat16);
            conditioning_->total_tokens = conditioning_->features.shape(0);
            mx::eval(conditioning_->features);
            event("llada_reference_conditioning", 1, 1);
        }
        const double text_seconds =
            std::chrono::duration<double>(Clock::now() - text_start).count();

        // The language model is substantially larger than the image
        // transformer.  Retaining both is useful for interactive low-resolution
        // prompt iteration, but leaves too little headroom for a 1024-square
        // denoiser on a 64 GB machine.  Large requests keep the computed
        // conditioning resident and evict only the re-loadable text weights.
        const uint64_t pixel_count =
            uint64_t(request.width) * uint64_t(request.height);
        const bool retain_text_weights =
            pixel_count <= uint64_t(512 * 512) &&
            device_info().physical_memory >= (64ull << 30) &&
            (!request.memory_budget_bytes ||
             request.memory_budget_bytes >= (48ull << 30));
        if (!retain_text_weights) {
            text_encoder_.clear();
            queryformer_.clear();
            text_projection_.clear();
            mx::clear_cache();
        }

        load(event, cancelled);
        auto hybrid_start = Clock::now();
        if (request.execution == "gpu_ane") {
            require(request.allow_approximation,
                    "LLaDA GPU+ANE requires allow_approximation=true");
            require(!request.ane_manifest.empty(),
                    "LLaDA GPU+ANE requires a compiled Core ML manifest");
            const int image_rows =
                padded_rows((request.height / 16) * (request.width / 16));
            const int caption_rows = padded_rows(conditioning_->total_tokens);
            const int rows = image_rows + caption_rows;
            const auto checkpoint =
                root_ / "transformer/diffusion_pytorch_model.safetensors.index.json";
            if (!hybrid_ || hybrid_->manifest != request.ane_manifest ||
                hybrid_->rows < rows) {
                hybrid_ = std::make_unique<HybridSession>(
                    request.ane_manifest, root_, rows, event, cancelled,
                    request.warmup_iterations, checkpoint);
            }
            require(hybrid_->hidden == 3840 && hybrid_->block_count == 32 &&
                        hybrid_->mlp_width == 10240 &&
                        hybrid_->ane_mlp_start == 0 &&
                        hybrid_->ane_mlp_end > 0 &&
                        hybrid_->ane_mlp_end < hybrid_->mlp_width,
                    "LLaDA Core ML manifest has incompatible FFN geometry");
            if (!hybrid_gpu_graph_ ||
                hybrid_gpu_mlp_start_ != hybrid_->ane_mlp_end) {
                hybrid_gpu_graph_ = make_llada_hybrid_gpu_graph(
                    hybrid_->mlp_width, hybrid_->ane_mlp_end);
                hybrid_gpu_mlp_start_ = hybrid_->ane_mlp_end;
            }
        } else {
            hybrid_.reset();
            hybrid_gpu_graph_ = {};
            hybrid_gpu_mlp_start_ = -1;
        }
        const double hybrid_seconds =
            std::chrono::duration<double>(Clock::now() - hybrid_start).count();
        if (load_only) {
            RunResult result;
            result.prepared = true;
            result.request = request;
            result.plan = std::move(plan);
            result.selection = request.execution == "gpu_ane"
                                   ? "gpu_ane: in-process native MLX/Core ML LLaDA"
                                   : (retain_text_weights
                                          ? "gpu: in-process native C++/MLX LLaDA; resident text"
                                          : "gpu: in-process native C++/MLX LLaDA; staged text");
            result.backend = request.execution == "gpu_ane"
                                 ? "mlx_cpp_metal+coreml"
                                 : "mlx_cpp_metal";
            result.precision = request.execution == "gpu_ane"
                                   ? "bf16_gpu+int8_mlp_fp16_io"
                                   : "bf16";
            result.prompt_cache_hit = prompt_hit;
            result.text_tokens = conditioning_->prompt_tokens;
            result.valid_text_tokens = conditioning_->prompt_tokens;
            result.total_tokens = conditioning_->total_tokens +
                                  (request.height / 16) * (request.width / 16);
            result.timings.wall =
                std::chrono::duration<double>(Clock::now() - begin).count();
            result.timings.text = text_seconds;
            result.timings.hybrid = hybrid_seconds;
            result.active_bytes = mx::get_active_memory();
            result.peak_bytes = mx::get_peak_memory();
            if (hybrid_)
                result.hybrid = hybrid_->metrics();
            return result;
        }
        auto latent = initial_noise(request);
        mx::eval(latent);
        auto dump = [&](const std::string &name, const Tensor &value) {
            if (request.dump.empty())
                return;
            std::filesystem::create_directories(request.dump);
            mx::save_safetensors(
                (std::filesystem::path(request.dump) / (name + ".safetensors")).string(),
                {{"tensor", value}});
        };
        dump("llada_conditioning", conditioning_->features);
        dump("llada_initial_latent", latent);

        const bool compile_blocks = request.compile_gpu;
        auto schedule = sigmas(request.steps);
        auto denoise_start = Clock::now();
        auto context = llada_prepare_transformer_context(
            conditioning_->features, transformer_, event, cancelled, request.dump);
        for (int step = 0; step < request.steps; ++step) {
            checkpoint(cancelled);
            event("denoise", step, request.steps);
            auto noise = llada_transformer(
                latent, context, schedule[step],
                request.height, request.width, transformer_, event, cancelled,
                hybrid_.get(), hybrid_ ? &hybrid_gpu_graph_ : nullptr,
                compile_blocks, request.dump, step);
            mx::eval(noise);
            dump("llada_noise_" + std::to_string(step), noise);
            std::optional<Tensor> step_noise;
            if (schedule[step + 1] > 0.f)
                step_noise = scheduler_noise(request, step, latent.shape());
            latent = stochastic_step(latent, noise, schedule[step],
                                     schedule[step + 1], step_noise);
            mx::eval(latent);
            dump("llada_latent_" + std::to_string(step + 1), latent);
            require(mx::all(mx::isfinite(latent)).item<bool>(),
                    "nonfinite LLaDA latent");
            event("denoise", step + 1, request.steps);
        }
        const double denoise_seconds =
            std::chrono::duration<double>(Clock::now() - denoise_start).count();

        auto decode_start = Clock::now();
        auto pixels = decode_latent(latent, vae_, request.height, request.width,
                                    event, cancelled, request.dump);
        mx::eval(pixels);
        require(mx::all(mx::isfinite(pixels)).item<bool>(),
                "nonfinite LLaDA decoded pixels");
        const double decode_seconds =
            std::chrono::duration<double>(Clock::now() - decode_start).count();
        if (!warmup) {
            event("export", 0, 1);
            checkpoint(cancelled);
            save_png(pixels, request.output);
            event("export", 1, 1);
        }

        RunResult result;
        result.request = request;
        result.plan = std::move(plan);
        result.selection = request.execution == "gpu_ane"
                               ? "gpu_ane: in-process native MLX/Core ML LLaDA"
                               : (retain_text_weights
                                      ? "gpu: in-process native C++/MLX LLaDA; resident text"
                                      : "gpu: in-process native C++/MLX LLaDA; staged text");
        result.backend = request.execution == "gpu_ane"
                             ? "mlx_cpp_metal+coreml"
                             : "mlx_cpp_metal";
        result.precision = request.execution == "gpu_ane"
                               ? "bf16_gpu+int8_mlp_fp16_io"
                               : "bf16";
        result.warmup = warmup;
        result.prompt_cache_hit = prompt_hit;
        result.text_tokens = conditioning_->prompt_tokens;
        result.valid_text_tokens = conditioning_->prompt_tokens;
        result.total_tokens = conditioning_->total_tokens +
                              (request.height / 16) * (request.width / 16);
        result.actual_steps = request.steps;
        result.timings.wall =
            std::chrono::duration<double>(Clock::now() - begin).count();
        result.timings.text = text_seconds;
        result.timings.hybrid = hybrid_seconds;
        result.timings.denoise = denoise_seconds;
        result.timings.decode = decode_seconds;
        result.active_bytes = mx::get_active_memory();
        result.peak_bytes = mx::get_peak_memory();
        if (hybrid_)
            result.hybrid = hybrid_->metrics();
        return result;
    }

  public:
    explicit LLaDAImageNative(const std::filesystem::path &root)
        : root_(std::filesystem::absolute(root)), tokenizer_(root_ / "tokenizer") {
        require(std::filesystem::is_directory(root_), "LLaDA model directory is missing");
        for (const char *component : {"text_encoder", "queryformer", "text_projection",
                                      "transformer", "vae", "tokenizer"})
            require(std::filesystem::is_directory(root_ / component),
                    "LLaDA component is missing: " + std::string(component));
        require(has_safetensors(root_ / "text_encoder"),
                "LLaDA text encoder safetensors are missing");
        require(has_safetensors(root_ / "transformer"),
                "LLaDA transformer safetensors are missing");
        require(has_safetensors(root_ / "vae"),
                "LLaDA VAE safetensors are missing");
    }

    LoadResult load(const Event &event, std::atomic<bool> &cancelled) override {
        if (transformer_.bytes() == 0) {
            event("load_llada_transformer", 0, 1);
            transformer_.load(root_ / "transformer", event, cancelled);
            normalize_llada_transformer(transformer_);
            transformer_.erase_prefix("semantic_embedder");
            transformer_.erase_prefix("sigvq_embedder");
            transformer_.erase_prefix("sigvq_refiner");
            transformer_.erase("sigvq_pad_token");
            transformer_.materialize();
            event("load_llada_transformer", 1, 1);
        }
        if (vae_.bytes() == 0) {
            event("load_llada_vae", 0, 1);
            vae_.load(root_ / "vae", event, cancelled);
            vae_.materialize();
            event("load_llada_vae", 1, 1);
        }
        return {uint64_t(text_encoder_.bytes() + queryformer_.bytes() +
                         text_projection_.bytes() + transformer_.bytes() +
                         vae_.bytes()),
                mx::get_active_memory()};
    }

    void unload() override {
        hybrid_.reset();
        hybrid_gpu_graph_ = {};
        hybrid_gpu_mlp_start_ = -1;
        conditioning_.reset();
        cached_prompt_.clear();
        text_encoder_.clear();
        queryformer_.clear();
        text_projection_.clear();
        transformer_.clear();
        vae_.clear();
        mx::clear_cache();
    }

    RunResult prepare(const Request &request, bool warmup, const Event &event,
                      std::atomic<bool> &cancelled) override {
        return run(request, event, cancelled, warmup, !warmup);
    }

    RunResult generate(const Request &request, const Event &event,
                       std::atomic<bool> &cancelled) override {
        return run(request, event, cancelled, false);
    }
};

} // namespace

std::unique_ptr<ModelSession> create_llada_image_native(
    const std::filesystem::path &root) {
    return std::make_unique<LLaDAImageNative>(root);
}

} // namespace tc
