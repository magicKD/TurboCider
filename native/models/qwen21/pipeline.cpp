#include "pipeline.hpp"
#include "conditioning.hpp"
#include "transformer.hpp"
#include "vae.hpp"
#include "scheduler.hpp"
#include "pe_generation.hpp"
#include "../../media/image.hpp"
#include "../../runtime/residency.hpp"
#include "../../platform/apple/platform.hpp"
#include <mlx/random.h>
#include <fstream>

namespace tc::qwen21 {
namespace {
void emit(const Event &event, const std::string &phase, int step, int total) {
    if (event) event(phase, step, total);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string read_utf8_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "cannot read Qwen35 PE system prompt: " + path.string());
    std::string value((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    require(!value.empty() && value.size() <= 1024 * 1024, "invalid Qwen35 PE system prompt");
    return value;
}
}
Session::Session(const std::filesystem::path &root) : root_(root) {
    for (const char *relative : {"diffusion_models/qwen_image_2.1_bf16.safetensors",
                                "text_encoders/qwen3vl_8b_bf16.safetensors",
                                "vae/qwen_image_2.1_vae_bf16.safetensors", "processor/tokenizer.json"})
        require(std::filesystem::is_regular_file(root / relative), std::string("missing Qwen Image 2.1 asset: ") + relative);
}
LoadResult Session::load(const Event &event, std::atomic<bool> &cancelled) {
    checkpoint(cancelled);
    if (!transformer_.bytes()) {
        emit(event, "load_qwen21_transformer", 0, 1);
        transformer_.load_file(root_ / "diffusion_models/qwen_image_2.1_bf16.safetensors");
        transformer_.materialize();
        emit(event, "load_qwen21_transformer", 1, 1);
    }
    checkpoint(cancelled);
    if (!vae_.bytes()) {
        emit(event, "load_qwen21_vae", 0, 1);
        vae_.load_file(root_ / "vae/qwen_image_2.1_vae_bf16.safetensors");
        vae_.materialize();
        emit(event, "load_qwen21_vae", 1, 1);
    }
    return {uint64_t(transformer_.bytes() + vae_.bytes()), mx::get_active_memory()};
}
void Session::unload() {
    hybrid_mlp_.reset();
    hybrid_.reset();
    hybrid_manifest_.clear();
    cached_text_.reset();
    cached_prompt_.clear();
    transformer_.clear();
    vae_.clear();
    mx::clear_cache();
}
RunResult Session::prepare(const Request &request, bool warmup, const Event &event, std::atomic<bool> &cancelled) {
    return run(request, event, cancelled, warmup, !warmup);
}
RunResult Session::generate(const Request &request, const Event &event, std::atomic<bool> &cancelled) {
    return run(request, event, cancelled, false, false);
}
RunResult Session::run(const Request &requested, const Event &event, std::atomic<bool> &cancelled,
                       bool warmup, bool prepare_only) try {
    auto start = Clock::now();
    Request r = requested;
    require(r.model == "qwen-image-2.1", "Qwen21 session received another model id");
    const std::string original_prompt = r.prompt;
    const bool hybrid_requested = r.execution == "gpu_ane";
    auto plan = make_plan(r);
    require(!r.prompt.empty(), "Qwen21 requires a prompt");
    require(warmup || prepare_only || (!r.output.empty() && std::filesystem::path(r.output).extension() == ".png"),
            "Qwen21 requires a .png output");
    ResidencyPolicy::validate_budget(plan, device_info().physical_memory);
    if (!hybrid_requested) {
        hybrid_mlp_.reset(); hybrid_.reset(); hybrid_manifest_.clear();
        r.execution = "gpu"; // automatic selection remains conservative
    }
    plan.request = r;
    checkpoint(cancelled);
    // Include prompt enhancement in whole-request allocator accounting and
    // apply the requested cache policy before allocating PE tensors.
    mx::reset_peak_memory();
    mx::set_cache_limit(r.allocator_cache_bytes);
    double prompt_enhance_seconds = 0.;
    int prompt_enhance_tokens = 0;
    bool prompt_enhance_chunked_prefill = false;
    std::string enhanced_ratio, enhanced_ratio_follow;
    if (r.prompt_enhance) {
        auto enhance_start = Clock::now();
        require(r.inputs.empty() || r.prompt_enhance_edit_experimental,
                "Qwen35 PE edit requires the experimental FP32 image route");
        const auto pe_root = std::filesystem::path(r.prompt_enhancer_path);
        require(std::filesystem::is_regular_file(pe_root / "system_prompt.txt") &&
                    std::filesystem::is_regular_file(pe_root / "tokenizer.json"),
                "Qwen35 PE installation requires system_prompt.txt and tokenizer.json");
        // PE weights must not overlap the diffusion checkpoint on staged runs.
        if (r.residency == "component_staged") {
            hybrid_mlp_.reset();
            transformer_.clear(); vae_.clear(); mx::clear_cache();
        }
        Weights pe_weights;
        pe_weights.load(pe_root, [&](const std::string &, int current, int total) {
            emit(event, "load_qwen35_pe", current, total);
        }, cancelled);
        Tokenizer pe_tokenizer(pe_root);
        const auto system_prompt = read_utf8_file(pe_root / "system_prompt.txt");
        qwen21::pe::TextGenerationResult pe;
        if (r.prompt_enhance_edit_experimental) {
            require(pe_weights.has("model.visual.patch_embed.proj.weight"),
                    "PE-I2I installation is missing Qwen3.5 visual weights");
            std::vector<Tensor> pe_images;
            for (const auto &input : r.inputs) {
                checkpoint(cancelled);
                pe_images.push_back(load_pe_image_tensor(input.path));
            }
            pe = qwen21::pe::generate_edit_images(pe_weights, pe_tokenizer, system_prompt,
                r.prompt, pe_images, qwen21::pe::SamplingProfile::for_edit(), r.seed,
                event, cancelled, {}, true);
        } else {
            qwen21::pe::SamplingProfile profile;
            pe = qwen21::pe::generate_text(pe_weights, pe_tokenizer, system_prompt,
                r.prompt, profile, r.seed, event, cancelled);
        }
        require(pe.complete(), "Qwen35 PE did not produce a complete rewritten prompt");
        r.prompt = pe.rewrite.positive_prompt;
        enhanced_ratio = pe.rewrite.wh_ratio;
        enhanced_ratio_follow = pe.rewrite.ratio_follow;
        prompt_enhance_tokens = pe.generated_tokens;
        prompt_enhance_chunked_prefill = pe.chunked_prefill;
        prompt_enhance_seconds = seconds(enhance_start);
        pe_weights.clear();
        mx::clear_cache();
    }
    plan.request = r;
    auto dump = [&](const std::string &name, const Tensor &tensor) {
        if (r.dump.empty()) return;
        std::filesystem::create_directories(r.dump);
        mx::save_safetensors((std::filesystem::path(r.dump) / (name + ".safetensors")).string(), {{"tensor", tensor}});
    };
    std::vector<Tensor> images;
    for (const auto &input : r.inputs) {
        checkpoint(cancelled);
        auto pixels = resize_reference(load_rgba_image_tensor(input.path));
        images.push_back(pixels);
    }
    const bool hit = images.empty() && cached_text_ && cached_prompt_ == r.prompt;
    Tensor text(0.f);
    std::vector<int> slots;
    auto text_start = Clock::now();
    if (hit) text = *cached_text_;
    else {
        // Staged requests release the DiT; resident requests keep its packed
        // suffix too, including when encoding a different prompt.
        if (r.residency == "component_staged") {
            hybrid_mlp_.reset();
            transformer_.clear(); vae_.clear(); mx::clear_cache();
        }
        Weights weights;
        emit(event, "load_qwen21_text", 0, 1);
        weights.load_file(root_ / "text_encoders/qwen3vl_8b_bf16.safetensors");
        emit(event, "load_qwen21_text", 1, 1);
        Tokenizer tokenizer(root_ / "processor");
        auto tokens = tokenizer.raw_bounded(reference_prompt_template(r.prompt, images.size()), Tokenizer::qwen21_limit);
        std::vector<VisualReference> refs;
        VisionEncoder vision(weights);
        for (const auto &pixels : images) {
            checkpoint(cancelled);
            auto alpha = slice_axis(pixels, 3, 3, 4);
            auto input = vision_input(slice_axis(pixels, 3, 0, 3) * alpha + (1.f - alpha));
            refs.push_back({vision.encode(input.patches, input.grid_height, input.grid_width, event, cancelled),
                            input.grid_height, input.grid_width});
        }
        auto assembled = assemble_prompt(tokens, weights.at("model.embed_tokens.weight"), refs);
        TextConfig config;
        config.final_norm = false; // official checkpoint's pre-final-RMSNorm hidden state
        TextEncoder encoder(weights, config);
        text = assembled.retain(encoder.encode_embeddings(assembled.embeddings, assembled.positions,
            assembled.embeddings.shape(1), event, cancelled, assembled.deepstack_deltas));
        mx::eval(text);
        slots = assembled.image_slots;
        if (images.empty()) { cached_text_ = text; cached_prompt_ = r.prompt; }
    }
    mx::clear_cache();
    double text_seconds = seconds(text_start);
    dump("qwen21_text", text);
    std::vector<ReferenceLatents> references;
    auto image_start = Clock::now();
    if (!images.empty()) {
        Weights weights;
        weights.load_file(root_ / "vae/qwen_image_2.1_vae_bf16.safetensors");
        VAE encoder(weights);
        for (size_t i = 0; i < images.size(); ++i) {
            checkpoint(cancelled);
            auto latent = encoder.encode(mx::transpose(images[i] * 2.f - 1.f, {0, 3, 1, 2}), event, cancelled);
            latent = mx::astype(mx::reshape(mx::transpose(latent, {0, 2, 3, 1}), {1, -1, 64}), mx::bfloat16);
            mx::eval(latent);
            references.push_back({latent, {images[i].shape(1) / 16, images[i].shape(2) / 16, slots[i]}});
            dump("qwen21_reference_" + std::to_string(i), latent);
        }
    }
    images.clear(); mx::clear_cache();
    double image_seconds = seconds(image_start);
    load(event, cancelled);
    auto hybrid_start = Clock::now();
    if (hybrid_requested) {
        auto manifest = std::filesystem::canonical(r.ane_manifest);
        auto identity = manifest.string() + ":" + sha256_file(manifest);
        if (!hybrid_ || hybrid_manifest_ != identity) {
            hybrid_mlp_.reset();
            hybrid_.reset();
            hybrid_ = std::make_unique<HybridSession>(
                manifest, root_, r.width / 16 * (r.height / 16),
                event, cancelled, 1,
                root_ / "diffusion_models/qwen_image_2.1_bf16.safetensors",
                std::vector<LoRAAsset>{}, r.width / 16 * (r.height / 16), 32);
            hybrid_manifest_ = identity;
        }
        require(hybrid_->hidden == 4096 && hybrid_->mlp_width == 12288 &&
                    hybrid_->ane_mlp_start == 0 && hybrid_->ane_mlp_end == 4096 &&
                    hybrid_->block_count == 32 && hybrid_->checkpoint_sha_verified &&
                    hybrid_->tensor_layout == "qwen21" && hybrid_->export_variant == "fp16" &&
                    hybrid_->output_scale == 1.f,
                "Qwen21 gpu_ane manifest is not a verified 32-block FP16 partition");
        if (!hybrid_mlp_) hybrid_mlp_ = std::make_unique<HybridMLP>(transformer_, *hybrid_);
    }
    RunResult result;
    result.original_prompt = original_prompt;
    result.enhanced_prompt = r.prompt_enhance ? r.prompt : "";
    result.enhanced_wh_ratio = enhanced_ratio;
    result.enhanced_ratio_follow = enhanced_ratio_follow;
    result.prompt_enhance_tokens = prompt_enhance_tokens;
    result.prompt_enhance_chunked_prefill = prompt_enhance_chunked_prefill;
    result.prompt_enhance_seconds = prompt_enhance_seconds;
    result.request = r; result.plan = std::move(plan);
    result.prepared = prepare_only; result.warmup = warmup; result.prompt_cache_hit = hit;
    result.selection = "gpu: native Qwen Image 2.1 with request-owned prefix KV cache";
    result.backend = "mlx_cpp_metal"; result.precision = "bf16";
    if (hybrid_requested) {
        result.backend = "mlx_cpp_metal+coreml";
        result.precision = "bf16_gpu+fp16_mlp_fp16_io";
        result.selection = "gpu_ane experimental: BF16 GPU suffix + FP16 Core ML CPU/ANE prefix; runtime placement not guaranteed";
    }
    result.timings.hybrid = hybrid_requested ? seconds(hybrid_start) : 0;
    result.checkpoint = "qwen_image_2.1_bf16.safetensors";
    result.text_tokens = result.valid_text_tokens = text.shape(1);
    for (const auto &ref : references) result.reference_tokens += ref.latents.shape(1);
    result.total_tokens = result.text_tokens + result.reference_tokens + r.height / 16 * (r.width / 16);
    result.timings.text = text_seconds; result.timings.image = image_seconds;
    emit(event, hybrid_requested ? "route_gpu_ane" : "route_gpu", 1, 1);
    if (!prepare_only) {
        auto schedule = sigmas(r.width, r.height, r.steps);
        mx::eval(schedule);
        auto latents = mx::astype(mx::random::normal({1, r.height / 16 * (r.width / 16), 64},
            mx::float32, mx::random::key(r.seed)), mx::bfloat16);
        if (!r.noise_path.empty()) {
            auto [values, metadata] = mx::load_safetensors(r.noise_path);
            require(values.count("tensor") || values.count("initial"), "Qwen21 noise file requires tensor or initial");
            const auto &noise = values.at(values.count("tensor") ? "tensor" : "initial");
            require(noise.shape() == latents.shape(), "Qwen21 noise tensor shape mismatch");
            latents = mx::astype(noise, mx::bfloat16);
        }
        dump("qwen21_initial", latents);
        auto dit_start = Clock::now();
        {
            Transformer dit(transformer_);
            if (hybrid_requested) {
                dit.set_decode_mlp([&](int block, const Tensor &input) {
                    checkpoint(cancelled);
                    return (*hybrid_mlp_)(block, input);
                });
            }
            // The App records the first denoise event as the progress/time
            // origin. Emit it before sampling so step one is not subtracted.
            emit(event, "denoise", 0, r.steps);
            for (int step = 0; step < r.steps; ++step) {
                checkpoint(cancelled);
                auto noise = dit.forward(latents, text, schedule.data<float>()[step], r.height / 16, r.width / 16,
                                         true, nullptr, references);
                latents = latents + noise * Tensor(schedule.data<float>()[step+1] - schedule.data<float>()[step], latents.dtype());
                mx::eval(latents);
                require(mx::all(mx::isfinite(latents)).item<bool>(), "nonfinite Qwen21 latent");
                emit(event, "denoise", step + 1, r.steps);
            }
        }
        result.timings.denoise = seconds(dit_start); result.actual_steps = r.steps;
        dump("qwen21_latents", latents);
        if (r.residency == "component_staged") { hybrid_mlp_.reset(); transformer_.clear(); mx::clear_cache(); }
        checkpoint(cancelled);
        auto decode_start = Clock::now();
        VAE decoder(vae_);
        auto spatial = mx::transpose(mx::reshape(latents, {1, r.height / 16, r.width / 16, 64}), {0, 3, 1, 2});
        auto pixels = mx::transpose(decoder.decode(spatial, event, cancelled), {0, 2, 3, 1});
        mx::eval(pixels);
        require(mx::all(mx::isfinite(pixels)).item<bool>(), "nonfinite Qwen21 pixels");
        dump("qwen21_pixels", pixels);
        result.timings.decode = seconds(decode_start);
        if (!warmup) {
            checkpoint(cancelled);
            emit(event, "export", 0, 1); save_rgba_png(pixels, r.output); emit(event, "export", 1, 1);
        }
    }
    if (hybrid_requested) result.hybrid = hybrid_->metrics(); // session-cumulative, including preparation
    if (!prepare_only && r.residency == "component_staged") {
        hybrid_mlp_.reset(); hybrid_.reset(); hybrid_manifest_.clear();
        transformer_.clear(); vae_.clear(); mx::clear_cache();
    }
    result.timings.wall = seconds(start);
    result.active_bytes = mx::get_active_memory(); result.peak_bytes = mx::get_peak_memory();
    return result;
} catch (...) {
    // Local prefix state is already destroyed; no partial cache survives a retry.
    try { mx::synchronize(); } catch (...) {}
    hybrid_mlp_.reset(); hybrid_.reset(); hybrid_manifest_.clear();
    throw;
}
} // namespace tc::qwen21
