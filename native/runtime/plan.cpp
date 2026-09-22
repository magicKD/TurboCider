#include "session.hpp"
#include <set>
#include <cmath>
#include <algorithm>
namespace tc {
namespace {
bool supported_lora_strategy(const std::string &strategy) {
    return strategy == "auto" || strategy == "disk_premerge" ||
           strategy == "in_memory_merge" || strategy == "inference_time";
}
} // namespace

std::string effective_lora_strategy(const Request &request) {
    require(supported_lora_strategy(request.lora_strategy),
            "unknown LoRA strategy: " + request.lora_strategy);
    if (request.loras.empty()) {
        require(request.lora_strategy == "auto",
                "lora_strategy requires at least one LoRA adapter");
        return "none";
    }
    auto descriptor = module_for(request.model).describe();
    require(descriptor.supports_lora,
            "selected model does not support LoRA adapters");
    auto strategy = request.lora_strategy == "auto"
        ? descriptor.default_lora_strategy : request.lora_strategy;
    require(!strategy.empty(),
            "selected model does not declare a default LoRA strategy");
    require(std::find(descriptor.lora_strategies.begin(),
                      descriptor.lora_strategies.end(), strategy) !=
                descriptor.lora_strategies.end(),
            "LoRA strategy " + strategy + " is not supported by " + request.model);
    return strategy;
}

void validate_recipe(const Recipe &r) {
    std::set<std::string> done;
    for (auto &s : r.stages) {
        require(!s.id.empty() && s.iterations > 0, "invalid recipe stage");
        require(!done.count(s.id), "duplicate recipe stage: " + s.id);
        for (auto &dep : s.dependencies)
            require(done.count(dep), "unsatisfied recipe dependency: " + dep);
        done.insert(s.id);
    }
    require(!done.empty(), "empty recipe");
}
std::vector<float> flux_sigmas(int tokens, int steps) {
    require(tokens > 0 && steps >= 1 && steps <= 50, "invalid Flux schedule");
    double mu;
    double m200 = .00016927 * tokens + .45666666;
    if (tokens > 4300)
        mu = m200;
    else {
        double m10 = 8.73809524e-5 * tokens + 1.89833333;
        double a = (m200 - m10) / 190.;
        mu = a * steps + m200 - 200 * a;
    }
    std::vector<float> s;
    for (int i = 0; i < steps; ++i) {
        double t = 1. - double(i) / steps;
        s.push_back(float(std::exp(mu) / (std::exp(mu) + 1. / t - 1.)));
    }
    s.push_back(0);
    return s;
}
ExecutionPlan make_plan(const Request &requested) {
    Request r = requested;
    const auto lora_strategy = effective_lora_strategy(r);
    if (lora_strategy != "none")
        r.lora_strategy = lora_strategy;
    auto recipe = model_recipe(r.model);
    validate_recipe(recipe);
    const int max_dimension = r.model == "qwen-image-2.1" ? 4096 : 2048;
    require(r.width >= 64 && r.height >= 64 && r.width <= max_dimension && r.height <= max_dimension,
            "dimensions must be 64..." + std::to_string(max_dimension));
    require(r.steps >= 1 && r.steps <= 50, "steps must be 1...50");
    require(r.execution == "gpu" || r.execution == "auto" || r.execution == "gpu_ane",
            "unknown execution mode");
    bool hybrid = r.execution == "gpu_ane";
    if (hybrid) {
        require(r.allow_approximation,
                "gpu_ane requires allow_approximation=true");
        require(!r.ane_manifest.empty(), "gpu_ane requires an explicit ANE manifest or profile");
    }
    module_for(r.model).validate(r);
    require((!r.prompt_enhance && !r.prompt_enhance_edit_experimental && r.prompt_enhancer_path.empty()) || r.model == "qwen-image-2.1",
            "native prompt enhancement is currently supported only for Qwen21");
    require(!r.prompt_enhance_edit_experimental ||
                (r.prompt_enhance && r.operation == "image.edit" && !r.inputs.empty()),
            "experimental PE-I2I requires prompt_enhance and image.edit references");
    if (r.prompt_enhance) {
        require((r.operation == "image.generate" && r.inputs.empty()) || r.prompt_enhance_edit_experimental,
                "Qwen21 PE edit requires explicit prompt_enhance_edit_experimental=true (FP32 vision; not quality-qualified)");
        require(!r.prompt_enhancer_path.empty(), "prompt_enhance requires prompt_enhancer_path");
        recipe.stages.insert(recipe.stages.begin(), {"prompt_enhance", {}});
        for (auto &stage : recipe.stages)
            if (stage.id == "text_encode") stage.dependencies.push_back("prompt_enhance");
    }
    if (r.model == "ltx-2.5-distilled" && !r.audio) {
        recipe.stages.erase(std::remove_if(recipe.stages.begin(), recipe.stages.end(),
                                           [](const Stage &stage) {
                                               return stage.id == "audio_vae_vocoder" ||
                                                      stage.id == "mux";
                                           }),
                            recipe.stages.end());
        recipe.stages.push_back({"export", {"video_vae"}});
    }
    if (r.model.starts_with("flux2-klein-") && !r.inputs.empty()) {
        recipe.stages.insert(recipe.stages.begin() + 1, {"image_encode", {}});
        for (auto &stage : recipe.stages)
            if (stage.id == "denoise")
                stage.dependencies.push_back("image_encode");
    }
    if (r.model == "qwen-image-2.1" && !r.inputs.empty()) {
        recipe.stages.insert(recipe.stages.begin(), {"vision_encode", {}});
        recipe.stages.insert(recipe.stages.begin() + 2, {"reference_vae_encode", {}});
        for (auto &stage : recipe.stages) {
            if (stage.id == "text_encode") stage.dependencies.push_back("vision_encode");
            if (stage.id == "denoise") stage.dependencies.push_back("reference_vae_encode");
        }
    }
    if (r.model == "ltx-2.5-distilled" && r.operation == "video.image") {
        recipe.stages.insert(recipe.stages.begin(), {"first_frame_vae_encode", {}});
        for (auto &stage : recipe.stages)
            if (stage.id == "av_stage1")
                stage.dependencies.push_back("first_frame_vae_encode");
    }
    /* LTX audio finalization and first-frame I2V are native Session paths.
     * Request-time validation remains fail-closed for missing/invalid audio
     * assets, while the optional MLX denoiser still rejects I2V explicitly. */
    validate_recipe(recipe);
    ExecutionPlan plan{r, recipe, {}};
    if (r.model == "flux2-klein-4b")
        plan.memory_estimate_bytes =
            ((hybrid ? 16ull : 12ull) << 30) + uint64_t(r.width) * r.height * 8192;
    else if (r.model == "flux2-klein-9b")
        plan.memory_estimate_bytes = (28ull << 30) + uint64_t(r.width) * r.height * 12288;
    else if (r.model == "ltx-2.5-distilled") {
        const uint64_t geometry =
            uint64_t(r.width) * r.height * r.frames * 128;
        if (r.residency == "streamed")
            plan.memory_estimate_bytes = (26ull << 30) + geometry;
        else
            plan.memory_estimate_bytes = (36ull << 30) + geometry;
    }
    else if (r.model == "minimax-h3-vdn") {
        const uint64_t geometry =
            uint64_t(r.width) * r.height * r.frames * 80;
        plan.memory_estimate_bytes =
            (r.residency == "streamed" ? (28ull << 30) : (36ull << 30)) +
            geometry;
    }
    else if (r.model == "minimax-h3-turbo" ||
             r.model.starts_with("minimax-h3-fasth3-mlx-int6"))
        plan.memory_estimate_bytes = (32ull << 30) +
            uint64_t(r.width) * r.height * r.frames * 64;
    else if (r.model == "wan2.1-1.3b-qad")
        // Native full-video peaks exceed the historical Python-worker estimate.
        // Account for staged UMT5 plus DiT/decoder activations, with headroom;
        // this remains a heuristic, not an allocator-enforced limit.
        plan.memory_estimate_bytes = (12ull << 30) +
            uint64_t(r.width) * r.height * r.frames * (hybrid ? 1024 : 1408);
    else if (r.model == "qwen-image-2.1") {
        // Each approximately 1024-square reference contributes 4096 latent
        // tokens. Its BF16 prefix K/V alone occupies 2 GiB across 32 layers
        // (2 * 32 * 4096 tokens * 4096 hidden * 2 bytes). Add 512 MiB per
        // reference for working tensors, beyond the canvas/base allowance.
        // The previous 512-MiB-only term underestimated the measured ten-ref
        // peak (44,596,678,034 MLX bytes at 512-square) by over 13 GiB.
        // Still a heuristic: input aspect ratios and allocator lifetimes vary.
        constexpr uint64_t reference_prefix_bytes = 2ull * 32 * 4096 * 4096 * 2;
        // A 2048-square one-step probe peaked at 62,259,019,410 MLX bytes;
        // include the observed VAE/DiT working-set growth so high-resolution
        // plans do not under-report unified-memory pressure. This is not a cap.
        plan.memory_estimate_bytes = ((r.residency == "resident" ? 40ull : 22ull) << 30) +
                                     (hybrid ? (10ull << 30) : 0) + uint64_t(r.width) * r.height * 10240 +
                                     uint64_t(r.inputs.size()) * (reference_prefix_bytes + (512ull << 20));
    }
    else if (r.model == "z-image-turbo")
        plan.memory_estimate_bytes = r.residency == "streamed"
            ? std::max<uint64_t>(10ull << 30, r.memory_budget_bytes)
            : (25ull << 30) + uint64_t(r.width) * r.height * 8192;
    else if (r.model == "z-image-turbo-gguf")
        plan.memory_estimate_bytes = (16ull << 30) +
            uint64_t(r.width) * r.height * 4096;
    else if (r.model == "llada-image-turbo") {
        const uint64_t pixels = uint64_t(r.width) * uint64_t(r.height);
        if (r.operation == "image.generate" && r.execution != "gpu_ane") {
            // Native text-to-image keeps text weights resident for interactive
            // <=512-square work and stages them out before larger denoising
            // requests.  These estimates include observed allocator peaks plus
            // geometry-dependent activation headroom; validate_budget adds a
            // further 4 GiB system margin.
            plan.memory_estimate_bytes =
                (pixels <= uint64_t(512 * 512) ? (46ull << 30) : (34ull << 30)) +
                pixels * 12288;
        } else if (r.operation == "image.generate") {
            // Native hybrid follows the same staged-text policy as native GPU.
            // Core ML allocations are not visible to MLX, so reserve additional
            // fixed headroom without carrying the old full PyTorch worker budget.
            plan.memory_estimate_bytes =
                (pixels <= uint64_t(512 * 512) ? (50ull << 30) : (40ull << 30)) +
                pixels * 12288;
        } else {
            plan.memory_estimate_bytes = (52ull << 30) + pixels * 12288;
        }
    }
    return plan;
}
} // namespace tc
