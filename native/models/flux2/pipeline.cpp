#include "flux.hpp"
#include "../../platform/apple/platform.hpp"
#include "../../runtime/residency.hpp"
#include "../../media/image.hpp"
#include "../../backends/coreml.hpp"
#include <cmath>
namespace tc {
Flux::Flux(const std::filesystem::path &root) : root_(root), tokenizer_(root / "tokenizer") {
    validate_flux_configuration(root);
}
Flux::~Flux() = default;
LoadResult Flux::load(const Event &event, std::atomic<bool> &cancelled) {
    // Load image weights only: Qwen is intentionally staged during prompt encoding.
    require(device_info().physical_memory >= (16ull << 30),
            "insufficient memory for BF16 image weights");
    transformer_.load(root_ / "transformer", event, cancelled);
    vae_.load(root_ / "vae", event, cancelled);
    event("prepare_image_weights", 0, 2);
    transformer_.materialize();
    event("prepare_image_weights", 1, 2);
    vae_.materialize();
    event("prepare_image_weights", 2, 2);
    checkpoint(cancelled);
    return {transformer_.bytes() + vae_.bytes(), mx::get_active_memory()};
}
void Flux::unload() {
    hybrid_.reset();
    cached_conditioning_.reset();
    cached_prompt_.clear();
    transformer_.clear();
    vae_.clear();
}
bool Flux::conditioning(const Request &r, const Tokens &tokens, const Event &event,
                        std::atomic<bool> &cancelled) {
    bool hit = cached_conditioning_.has_value() && cached_prompt_ == r.prompt &&
               cached_dynamic_ == r.dynamic_text;
    if (hit) {
        event("text_cache_hit", 1, 1);
        return true;
    }
    // Keep preloaded image weights on machines with enough conservative headroom.
    // Smaller devices and explicit low budgets retain the staged text policy.
    bool retain =
        ResidencyPolicy::for_request(r, device_info().physical_memory).retain_images_during_text;
    if (!retain) {
        hybrid_.reset();
        transformer_.clear();
        vae_.clear();
    }
    cached_conditioning_.reset();
    mx::clear_cache();
    auto encoded = encode(tokens, event, cancelled);
    checkpoint(cancelled);
    cached_conditioning_ = encoded;
    cached_prompt_ = r.prompt;
    cached_dynamic_ = r.dynamic_text;
    mx::clear_cache();
    return false;
}
std::string Flux::select_acceleration(Request &r, int count, const Event &event,
                                      std::atomic<bool> &cancelled) {
    const bool automatic = r.execution == "auto";
    if (automatic) {
        r.execution = "gpu";
        if (!r.allow_approximation || r.ane_manifest.empty()) {
            hybrid_.reset();
            return "gpu: no opted-in compatible local partition";
        }
        auto system = device_info();
        // Automatic selection is limited to the device on which this partition policy was measured.
        if (system.gpu != "Apple M4 Pro" || device_info().physical_memory != (48ull << 30)) {
            hybrid_.reset();
            return "gpu: automatic hybrid policy not validated on this hardware";
        }
        uint64_t estimate = (16ull << 30) + uint64_t(r.width) * r.height * 8192;
        if (device_info().physical_memory < estimate + (4ull << 30) ||
            (r.memory_budget_bytes && r.memory_budget_bytes < estimate)) {
            hybrid_.reset();
            return "gpu: hybrid memory budget unavailable";
        }
        r.execution = "gpu_ane";
    }
    if (r.execution != "gpu_ane") {
        hybrid_.reset();
        return "gpu: explicitly selected";
    }
    try {
        checkpoint(cancelled);
        if (!hybrid_ || hybrid_->manifest != r.ane_manifest)
            hybrid_ = std::make_unique<HybridSession>(r.ane_manifest, root_, count, event,
                                                      cancelled, r.warmup_iterations);
        require(count <= hybrid_->rows, "Core ML token bucket cannot serve this request");
        return automatic ? "gpu_ane: hardware, checkpoint and token bucket matched"
                         : "gpu_ane: explicitly selected";
    } catch (const Cancelled &) {
        throw;
    } catch (const std::exception &error) {
        if (!automatic)
            throw;
        hybrid_.reset();
        mx::clear_cache();
        r.execution = "gpu";
        event("acceleration_gpu_fallback", 1, 1);
        return std::string("gpu: ") + error.what();
    }
}
RunResult Flux::prepare(const Request &requested, bool warmup, const Event &event,
                        std::atomic<bool> &cancelled) {
    if (warmup)
        return run(requested, event, cancelled, true);
    auto r = requested;
    auto begin = Clock::now();
    auto plan = make_plan(r);
    require(r.model == "flux2-klein-4b" && !r.prompt.empty(), "FLUX preparation requires a prompt");
    ResidencyPolicy::validate_budget(plan, device_info().physical_memory);
    mx::set_cache_limit(r.allocator_cache_bytes);
    auto tokens = tokenizer_.prompt(r.prompt, r.dynamic_text);
    bool hit = conditioning(r, tokens, event, cancelled);
    load(event, cancelled);
    int count = int(tokens.ids.size()) + (r.width / 16) * (r.height / 16);
    for (auto &input : r.inputs)
        if (r.operation == "image.edit") {
            auto image = load_image_tensor(input.path, r.width, r.height, true);
            count += (image.shape(1) / 16) * (image.shape(2) / 16);
        }
    auto selection = select_acceleration(r, count, event, cancelled);
    RunResult result;
    result.prepared = true;
    result.selection = selection;
    result.prompt_cache_hit = hit;
    result.request = r;
    result.text_tokens = int(tokens.ids.size());
    result.total_tokens = count;
    result.timings.wall = std::chrono::duration<double>(Clock::now() - begin).count();
    result.active_bytes = mx::get_active_memory();
    if (hybrid_)
        result.hybrid = hybrid_->metrics();
    return result;
}
RunResult Flux::generate(const Request &r, const Event &event, std::atomic<bool> &cancelled) {
    return run(r, event, cancelled, false);
}
RunResult Flux::run(const Request &requested, const Event &event, std::atomic<bool> &cancelled,
                    bool warmup) {
    auto r = requested;
    auto begin = Clock::now();
    auto plan = make_plan(r);
    require(r.model == "flux2-klein-4b", "model executor unavailable; see static acceptance plan");
    require(!r.prompt.empty() && (warmup || !r.output.empty()), "prompt and output are required");
    require(warmup || std::filesystem::path(r.output).extension() == ".png",
            "native image output must be .png");
    auto physical = device_info().physical_memory;
    ResidencyPolicy::validate_budget(plan, physical);
    auto residency = ResidencyPolicy::for_request(r, physical);
    mx::reset_peak_memory();
    mx::set_cache_limit(r.allocator_cache_bytes);
    auto tokens = tokenizer_.prompt(r.prompt, r.dynamic_text);
    if (r.execution != "gpu_ane" && r.execution != "auto")
        hybrid_.reset();
    auto dump = [&](const std::string &name, const Tensor &a) {
        if (!r.dump.empty()) {
            std::filesystem::create_directories(r.dump);
            mx::save_safetensors((std::filesystem::path(r.dump) / (name + ".safetensors")).string(),
                                 {{"tensor", a}});
        }
    };
    auto text_start = Clock::now();
    bool prompt_hit = conditioning(r, tokens, event, cancelled);
    double text_s = std::chrono::duration<double>(Clock::now() - text_start).count();
    std::optional<Tensor> reference_latents, clean_latents;
    std::vector<float> reference_ids;
    auto image_start = Clock::now();
    if (!r.inputs.empty()) {
        vae_.load(root_ / "vae", event, cancelled);
        for (size_t index = 0; index < r.inputs.size(); ++index) {
            checkpoint(cancelled);
            auto image = load_image_tensor(r.inputs[index].path, r.width, r.height,
                                           r.operation == "image.edit");
            dump("input_image_" + std::to_string(index), image);
            auto encoded = encode_image(image, event, cancelled);
            dump("image_latent_" + std::to_string(index), encoded);
            if (r.operation == "image.transform")
                clean_latents = encoded;
            else {
                reference_latents =
                    reference_latents ? mx::concatenate({*reference_latents, encoded}, 1) : encoded;
                int h = image.shape(1) / 16, w = image.shape(2) / 16;
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                        reference_ids.push_back(float(10 + 10 * index));
                        reference_ids.push_back(float(y));
                        reference_ids.push_back(float(x));
                        reference_ids.push_back(0);
                    }
            }
        }
    }
    if (residency.release_before_denoise) {
        vae_.clear();
        mx::clear_cache();
    }
    double image_s = std::chrono::duration<double>(Clock::now() - image_start).count();
    int actual_tokens = int(tokens.ids.size()) + (r.width / 16) * (r.height / 16) +
                        (reference_latents ? reference_latents->shape(1) : 0);
    require(actual_tokens <= 20000, "request exceeds native token workspace budget");
    auto hybrid_start = Clock::now();
    auto selection = select_acceleration(r, actual_tokens, event, cancelled);
    plan = make_plan(r);
    double hybrid_s = std::chrono::duration<double>(Clock::now() - hybrid_start).count();
    auto text = *cached_conditioning_;
    dump("conditioning", text);
    checkpoint(cancelled);
    transformer_.load(root_ / "transformer", event, cancelled);
    auto z = mx::astype(mx::random::normal({1, 128, r.height / 16, r.width / 16}, mx::float32, 0, 1,
                                           mx::random::key(r.seed)),
                        mx::bfloat16);
    z = mx::transpose(mx::reshape(z, {1, 128, (r.height / 16) * (r.width / 16)}), {0, 2, 1});
    mx::eval(z);
    dump("initial_latent", z);
    auto sigmas = flux_gpu_sigmas(z.shape(1), r.steps);
    int start_step = 0;
    if (clean_latents && r.inputs[0].strength > 0) {
        start_step = std::max(1, int(r.steps * r.inputs[0].strength));
        auto sigma = Tensor(sigmas[start_step]);
        z = (Tensor(1.f) - sigma) * (*clean_latents) + sigma * z;
        mx::eval(z);
        dump("conditioned_initial_latent", z);
    }
    auto dit_start = Clock::now();
    for (int i = start_step; i < r.steps; ++i) {
        checkpoint(cancelled);
        event("denoise", i, r.steps);
        auto model_input = reference_latents ? mx::concatenate({z, *reference_latents}, 1) : z;
        auto noise = denoise(model_input, text, sigmas[i], r.height, r.width, event, cancelled,
                             reference_ids, r.compile_gpu);
        if (reference_latents)
            noise = slice_axis(noise, 1, 0, z.shape(1));
        mx::eval(noise);
        dump("noise_" + std::to_string(i), noise);
        z = euler_step(z, noise, sigmas[i + 1] - sigmas[i]);
        mx::eval(z);
        dump("latent_" + std::to_string(i), z);
        require(mx::all(mx::isfinite(z)).item<bool>(), "nonfinite latent");
        event("denoise", i + 1, r.steps);
    }
    double dit_s = std::chrono::duration<double>(Clock::now() - dit_start).count();
    std::optional<HybridMetrics> hybrid_metrics;
    if (hybrid_)
        hybrid_metrics = hybrid_->metrics();
    checkpoint(cancelled);
    if (residency.release_after_denoise) {
        transformer_.clear();
        hybrid_.reset();
        mx::clear_cache();
    }
    vae_.load(root_ / "vae", event, cancelled);
    auto decode_start = Clock::now();
    auto pixels = decode(z, r.height, r.width, event, cancelled, r.dump);
    dump("pixels_nhwc", pixels);
    require(mx::all(mx::isfinite(pixels)).item<bool>(), "nonfinite decoded pixels");
    double decode_s = std::chrono::duration<double>(Clock::now() - decode_start).count();
    checkpoint(cancelled);
    if (!warmup) {
        event("export", 0, 1);
        checkpoint(cancelled);
        save_png(pixels, r.output);
        event("export", 1, 1);
    } else
        event("warmup_complete", 1, 1);

    if (residency.release_after_decode) {
        vae_.clear();
        mx::clear_cache();
    }
    double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    RunResult result;
    result.selection = selection;
    result.warmup = warmup;
    result.request = r;
    result.plan = std::move(plan);
    result.reference_tokens = reference_latents ? reference_latents->shape(1) : 0;
    result.actual_steps = r.steps - start_step;
    result.text_tokens = int(tokens.ids.size());
    result.valid_text_tokens = tokens.valid;
    result.prompt_cache_hit = prompt_hit;
    result.timings = {seconds, text_s, image_s, hybrid_s, dit_s, decode_s};
    result.peak_bytes = mx::get_peak_memory();
    result.active_bytes = mx::get_active_memory();
    result.hybrid = hybrid_metrics;
    return result;
}
} // namespace tc
