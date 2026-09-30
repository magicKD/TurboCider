#include "pipeline.hpp"
#include "diagnostic_options.hpp"
#include "viggle_adapter.hpp"
#include "conditioning.hpp"
#include "transformer.hpp"
#include "vae.hpp"
#include "scheduler.hpp"
#include "pe_generation.hpp"
#include "../../media/image.hpp"
#include "../../runtime/residency.hpp"
#include "../../platform/apple/platform.hpp"
#include <mlx/random.h>
#include <bit>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string_view>

namespace tc::qwen21 {
namespace {
void emit(const Event &event, const std::string &phase, int step, int total) {
    if (event) event(phase, step, total);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
size_t runtime_ane_budget(uint64_t physical, uint64_t active) {
    constexpr uint64_t system_margin = uint64_t(4) << 30;
    constexpr uint64_t optional_cap = uint64_t(2) << 30;
    // Share the same optional-tier allowance between FFN and QKV. Subtract
    // before adding active memory so even an invalidly large reading cannot
    // wrap around and appear to leave headroom.
    const uint64_t headroom = physical > system_margin ? physical - system_margin : 0;
    return std::min(optional_cap, active < headroom ? headroom - active : uint64_t(0));
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
void Session::clear_prefix_cache() {
    cached_prefix_transformer_.reset();
    cached_prefix_runtime_.clear();
    cached_prefix_sigma_ = -1.f;
}
void Session::clear_prefix_snapshot() {
    cached_prefix_snapshot_.reset();
    cached_snapshot_runtime_.clear();
}
void Session::unload() {
    runtime_ffn_.reset(); runtime_manifest_.clear();
    runtime_qkv_.reset(); qkv_manifest_.clear();
    clear_prefix_cache();
    clear_prefix_snapshot();
    fused_qkv_weights_.clear();
    hybrid_mlp_.reset();
    hybrid_.reset();
    hybrid_manifest_.clear();
    hybrid_runtime_options_.clear();
    cached_text_.reset();
    cached_prompt_.clear();
    cached_edit_.reset();
    transformer_.clear();
    vae_.clear();
    active_lora_identity_.clear();
    lora_applied_projections_ = 0;
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
    const bool runtime_requested = r.hybrid_mlp_mode == "runtime";
    const bool qkv_requested = r.hybrid_mlp_mode == "runtime_qkv";
    const bool hybrid_requested = r.execution == "gpu_ane" && !runtime_requested && !qkv_requested;
    if (!runtime_requested) { runtime_ffn_.reset(); runtime_manifest_.clear(); }
    if (!qkv_requested) { runtime_qkv_.reset(); qkv_manifest_.clear(); }
    const bool rectangular_w8a8 = option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_RECT_W8A8_DIAGNOSTIC"));
    const bool lora_base_ane = qwen21::lora_base_ane(r);
    const bool gate_up_ane = qwen21::gate_up_ane(r);
    const bool fused_lora_ane = qwen21::fused_lora_ane(r);
    const bool fused_qkv = option_enabled(std::getenv("TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC"));
    const char *tiled_prefill_flag = std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC");
    const int tiled_prefill_layers = tiled_prefill_layer_count(tiled_prefill_flag ? tiled_prefill_flag : "0");
    require(tiled_prefill_layers >= 0, "invalid Qwen21 tiled prefill option");
    const bool tiled_prefill = tiled_prefill_layers > 0;
    const bool tiled_prefix_reuse = option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC"));
    const bool prefix_target_only = option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC"));
    const bool last_target_only = option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC"));
    const bool reuse_final_ffn = option_enabled(std::getenv("TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN")) &&
        r.steps >= 3;
    const bool hybrid_reuse_ffn = option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC")) && r.steps == 5;
    const bool hybrid_reuse_last16 = option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC")) && r.steps == 5;
    const bool hybrid_half_reuse = option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC")) && r.steps == 5;
    const bool half_reuse_ffn = (option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN")) && r.steps >= 4) || hybrid_half_reuse;
    const bool profile_steps = option_enabled(std::getenv("TURBOCIDER_QWEN21_PROFILE_STEPS"));
    const bool db_cache = option_enabled(std::getenv("TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC"));
    const float db_threshold = db_cache_threshold(std::getenv("TURBOCIDER_QWEN21_DBCACHE_THRESHOLD"));
    const int db_max_consecutive = db_cache_max_consecutive(std::getenv(
        "TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE"));
    const bool resident_prefix = option_enabled(std::getenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV")) &&
        r.residency == "resident" && r.width == 512 && r.height == 512 &&
        r.steps >= 2 && r.loras.empty() && !r.prompt_enhance;
    if (!resident_prefix) clear_prefix_cache();
    const char *snapshot_option = std::getenv("TURBOCIDER_QWEN21_PREFIX_SNAPSHOT");
    require(binary_option_or_unset(snapshot_option), "Qwen21 prefix snapshot accepts only 0 or 1");
    // Keep one bounded bank for repeated edits. Set the option to 0 for the
    // original prefill path. Target-only first-step kernels can round slightly
    // differently from full prefill, particularly with BF16 runtime LoRA.
    const bool snapshot_enabled = !snapshot_option || option_enabled(snapshot_option);
    // Reuse only the ordinary GPU editing path. Keep experimental routing,
    // altered reference geometry and approximation caches out of this bank.
    const bool snapshot_route = snapshot_enabled && !resident_prefix &&
        !hybrid_requested && !runtime_requested && !qkv_requested &&
        !fused_qkv && !option_enabled(std::getenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE")) &&
        !option_enabled(std::getenv("TURBOCIDER_QWEN21_METAL_QK_ROPE")) &&
        (!std::getenv("TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION") ||
         std::string_view(std::getenv("TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION")) == "0") &&
        !tiled_prefill && !tiled_prefix_reuse && !prefix_target_only &&
        !last_target_only && !reuse_final_ffn && !hybrid_reuse_ffn &&
        !hybrid_reuse_last16 && !half_reuse_ffn && !db_cache &&
        !option_enabled(std::getenv("TURBOCIDER_QWEN21_VIGGLE_LORA_FP16")) &&
        !option_enabled(std::getenv("TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS")) &&
        !option_enabled(std::getenv("TURBOCIDER_QWEN21_PROFILE_GPU_OPS")) &&
        !option_enabled(std::getenv("TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS")) &&
        !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
        r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3 &&
        r.qwen21_reference_size == 1024 && r.width == 512 && r.height == 512 &&
        r.steps >= 2 && !r.prompt_enhance && r.dump.empty() &&
        (r.residency == "resident" || r.residency == "component_staged");
    if (!snapshot_route) clear_prefix_snapshot();
    // Preparation does not enter the sampling block. Still honor a newly
    // lowered cache budget before loading any weights for prepare(false).
    if (cached_prefix_snapshot_) {
        const uint64_t physical = device_info().physical_memory;
        const uint64_t limit = r.memory_budget_bytes
            ? std::min(physical, r.memory_budget_bytes) : physical;
        if (cached_prefix_snapshot_->bytes > std::min(uint64_t(8) << 30, limit / 8))
            clear_prefix_snapshot();
    }
    // A cached Transformer holds a pointer to the fused-weight bank. Destroy
    // it before changing the bank when a resident request toggles the flag.
    if (!fused_qkv && !fused_qkv_weights_.empty()) {
        clear_prefix_cache();
        fused_qkv_weights_.clear();
    }
    auto plan = make_plan(r);
    require(!r.prompt.empty(), "Qwen21 requires a prompt");
    require(warmup || prepare_only || (!r.output.empty() && std::filesystem::path(r.output).extension() == ".png"),
            "Qwen21 requires a .png output");
    ResidencyPolicy::validate_budget(plan, device_info().physical_memory);
    if (!hybrid_requested) {
        hybrid_mlp_.reset(); hybrid_.reset(); hybrid_manifest_.clear(); hybrid_runtime_options_.clear();
        if (!runtime_requested && !qkv_requested) r.execution = "gpu"; // automatic selection remains conservative
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
            clear_prefix_cache();
            fused_qkv_weights_.clear();
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
    // The one-entry edit cache belongs to this Session, including staged GPU
    // sessions. It holds completed conditioning, not encoder/DiT weights. A
    // path/mtime-only key would reuse stale visual conditions after an image
    // is overwritten in place; verify the ordered file bytes on each request.
    std::vector<std::string> image_sha256;
    const bool cache_edit_conditions = r.residency == "resident" ||
        r.residency == "component_staged";
    if (cache_edit_conditions) {
        image_sha256.reserve(r.inputs.size());
        for (const auto &input : r.inputs) {
            checkpoint(cancelled);
            image_sha256.push_back(sha256_file(input.path));
        }
    } else cached_edit_.reset();
    const bool edit_hit = cached_edit_ && !r.inputs.empty() &&
        cached_edit_->prompt == r.prompt &&
        cached_edit_->reference_size == r.qwen21_reference_size &&
        cached_edit_->image_sha256 == image_sha256;
    std::vector<Tensor> images;
    if (!edit_hit) for (const auto &input : r.inputs) {
        checkpoint(cancelled);
        images.push_back(resize_reference(load_rgba_image_tensor(input.path),
                                          r.qwen21_reference_size));
    }
    const bool hit = edit_hit || (r.inputs.empty() && cached_text_ && cached_prompt_ == r.prompt);
    if (!hit) { clear_prefix_cache(); clear_prefix_snapshot(); }
    Tensor text(0.f);
    std::vector<int> slots;
    auto text_start = Clock::now();
    if (edit_hit) { text = cached_edit_->text; slots = cached_edit_->image_slots; }
    else if (hit) text = *cached_text_;
    else {
        // Staged requests release the DiT; resident requests keep its packed
        // suffix too, including when encoding a different prompt.
        if (r.residency == "component_staged") {
            hybrid_mlp_.reset();
            clear_prefix_cache();
            fused_qkv_weights_.clear();
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
        if (r.inputs.empty()) { cached_text_ = text; cached_prompt_ = r.prompt; }
    }
    mx::clear_cache();
    double text_seconds = seconds(text_start);
    dump("qwen21_text", text);
    std::vector<ReferenceLatents> references;
    auto image_start = Clock::now();
    if (edit_hit) {
        references = cached_edit_->reference_latents;
        for (size_t i = 0; i < references.size(); ++i)
            dump("qwen21_reference_" + std::to_string(i), references[i].latents);
    } else if (!images.empty()) {
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
    if (!edit_hit && !r.inputs.empty() && cache_edit_conditions) {
        // Do not publish a cache entry if a reference changed during the
        // decode/encode pass. Hashing again also catches same-path overwrites
        // whose mtime or file size were preserved.
        for (size_t i = 0; i < r.inputs.size(); ++i) {
            checkpoint(cancelled);
            require(sha256_file(r.inputs[i].path) == image_sha256[i],
                    "Qwen21 reference changed while encoding conditioning");
        }
        cached_edit_ = CachedEditCondition{r.prompt, r.qwen21_reference_size,
                                          std::move(image_sha256), text, slots, references};
    }
    images.clear(); mx::clear_cache();
    double image_seconds = seconds(image_start);
    // Keep the distilled student as separate low-rank matrices. A BF16
    // in-memory/disk merge loses the update's small values. Revalidate the
    // pinned downloaded asset when its path/size/mtime changes; staged runs
    // reload and bind it on every request after the text encoder is released.
    std::string lora_identity;
    if (!r.loras.empty()) {
        auto path = std::filesystem::canonical(r.loras[0].path);
        require(std::filesystem::is_regular_file(path), "Qwen21 LoRA is not a regular file");
        lora_identity = path.string() + ":" + std::to_string(std::filesystem::file_size(path)) +
            ":" + std::to_string(static_cast<long long>(
                      std::filesystem::last_write_time(path).time_since_epoch().count())) +
            ":" + std::to_string(std::bit_cast<uint32_t>(r.loras[0].strength));
        // Alternate adapters are not pinned by filename, so a same-size,
        // same-mtime replacement must still invalidate the resident MLX
        // binding. The original Viggle path keeps its fast warm-request ABI.
        if (fused_lora_ane || runtime_requested) lora_identity += ":" + sha256_file(path);
    }
    const bool bind_lora = !r.loras.empty() &&
        (active_lora_identity_ != lora_identity || !transformer_.bytes());
    if (active_lora_identity_ != lora_identity) {
        if (runtime_ffn_) runtime_ffn_->drain();
        hybrid_mlp_.reset();
        clear_prefix_cache();
        clear_prefix_snapshot();
        fused_qkv_weights_.clear();
        transformer_.clear();
        active_lora_identity_.clear();
        lora_applied_projections_ = 0;
    }
    // The experimental QKV layout replaces (rather than duplicates) its
    // three source weights. If the resident Session returns to the regular
    // GPU path, reload the original checkpoint before constructing a DiT.
    if (!fused_qkv && transformer_.has("transformer_blocks.0.attn.qkv_packed.weight")) {
        hybrid_mlp_.reset();
        clear_prefix_cache();
        fused_qkv_weights_.clear();
        transformer_.clear();
    }
    if (bind_lora) {
        const auto *adapter = viggle_v021_adapter(
            std::filesystem::path(r.loras[0].path).filename().string());
        require(fused_lora_ane || runtime_requested ||
                    (adapter && sha256_file(r.loras[0].path) == adapter->sha256),
                "Viggle v0.2.1 r128/r256 LoRA hash does not match the pinned adapter");
    }
    load(event, cancelled);
    const bool lora_fp16 = option_enabled(std::getenv("TURBOCIDER_QWEN21_VIGGLE_LORA_FP16"));
    transformer_.set_runtime_lora_fp16(lora_fp16);
    if (bind_lora) {
        lora_applied_projections_ = transformer_.apply_loras(r.loras, "transformer", event, cancelled, true);
        require((fused_lora_ane || runtime_requested) ? lora_applied_projections_ > 0 : lora_applied_projections_ == 227,
                "Qwen21 LoRA did not bind transformer projections");
        active_lora_identity_ = lora_identity;
    }
    if (fused_qkv && fused_qkv_weights_.empty()) {
        clear_prefix_cache();
        fused_qkv_weights_.reserve(32);
        for (int block = 0; block < 32; ++block) {
            checkpoint(cancelled);
            const auto stem = "transformer_blocks." + std::to_string(block) + ".attn.";
            const auto name = stem + "to_";
            require(!transformer_.has(name + "q.bias") &&
                        !transformer_.has(name + "k.bias") &&
                        !transformer_.has(name + "v.bias"),
                    "Qwen21 fused QKV kernel needs bias-free checkpoint projections");
            const auto target = stem + "qkv_packed.weight";
            transformer_.fuse_keys(target, {name + "q.weight", name + "k.weight",
                                            name + "v.weight"}, 0);
            const auto &packed = transformer_.at(target);
            mx::eval(packed);
            fused_qkv_weights_.push_back(packed);
        }
        mx::clear_cache();
    }
    auto hybrid_start = Clock::now();
    if (runtime_requested) {
        auto manifest = std::filesystem::canonical(r.ane_manifest);
        const std::string identity = manifest.string() + ":" + sha256_file(manifest) + ":" +
            (std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS") ? std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS") : "auto");
        if (!runtime_ffn_ || !runtime_ffn_->available() || runtime_manifest_ != identity ||
            (!r.loras.empty() && !runtime_ffn_->supports_lora_inputs())) {
            runtime_ffn_.reset();
            const size_t budget = runtime_ane_budget(device_info().physical_memory,
                                                     mx::get_active_memory());
            runtime_ffn_ = std::make_unique<ane::HybridFfn>(manifest, 4096, 12288, budget, cancelled,
                                                         !r.loras.empty());
            runtime_manifest_ = identity;
        }
        runtime_ffn_->begin_request(active_lora_identity_);
    }
    if (qkv_requested) {
        for (int block = 0; block < 32; ++block) {
            const auto stem = "transformer_blocks." + std::to_string(block) + ".attn.to_";
            for (const char *projection : {"q", "k", "v"}) {
                const auto name = stem + projection;
                require(!transformer_.has(name + ".bias") &&
                            transformer_.at(name + ".weight").shape() == mx::Shape{4096, 4096} &&
                            (transformer_.at(name + ".weight").dtype() == mx::bfloat16 ||
                             transformer_.at(name + ".weight").dtype() == mx::float16),
                        "Qwen21 runtime QKV requires bias-free dense [4096,4096] checkpoint projections");
            }
        }
        auto manifest = std::filesystem::canonical(r.ane_manifest);
        const std::string identity = manifest.string() + ":" + sha256_file(manifest) + ":" +
            (std::getenv("TURBOCIDER_RUNTIME_ANE_QKV_CHUNKS") ?
                std::getenv("TURBOCIDER_RUNTIME_ANE_QKV_CHUNKS") : "1");
        if (!runtime_qkv_ || !runtime_qkv_->available() || qkv_manifest_ != identity) {
            runtime_qkv_.reset();
            const size_t budget = runtime_ane_budget(device_info().physical_memory,
                                                     mx::get_active_memory());
            runtime_qkv_ = std::make_unique<ane::HybridQkv>(manifest, budget, cancelled);
            qkv_manifest_ = identity;
        }
        runtime_qkv_->begin_request();
    }
    if (hybrid_requested) {
        auto manifest = std::filesystem::canonical(r.ane_manifest);
        auto identity = manifest.string() + ":" + sha256_file(manifest);
        if (!hybrid_ || hybrid_manifest_ != identity) {
            hybrid_mlp_.reset();
            hybrid_.reset();
            const int graph_rows = rectangular_w8a8 ? 1024 : r.width / 16 * (r.height / 16);
            hybrid_ = std::make_unique<HybridSession>(
                manifest, root_, graph_rows,
                event, cancelled, 1,
                root_ / "diffusion_models/qwen_image_2.1_bf16.safetensors",
                std::vector<LoRAAsset>{}, graph_rows, 32);
            hybrid_manifest_ = identity;
            hybrid_runtime_options_.clear();
        }
        const std::string runtime_options =
            (r.qwen21_w8a8 ? "w8a8" : "fp16") +
            std::string(r.qwen21_gpu_w8a16 ? ":w8a16" : ":bf16") +
            (r.qwen21_gpu_full_ffn_blocks.empty() ? ":full" : ":fallback357");
        // Keep the established 4096-channel path and permit the separately
        // calibrated 6144-channel W8A8 experiment only by explicit manifest.
        const bool supported_partition = r.qwen21_w8a8
            ? (hybrid_->ane_mlp_end == 4096 ||
               (((r.width == 512 && r.height == 512) || rectangular_w8a8) &&
                hybrid_->ane_mlp_end == 6144))
            : hybrid_->ane_mlp_end == 4096;
        // Full FFN coverage on edits was visually checked only for the
        // 6144-channel W8A8 candidate. Preserve the calibrated 4096 edit
        // route's 3/5/7 GPU fallback rather than silently broadening it.
        const bool supported_edit_coverage = r.operation != "image.edit" ||
            !r.qwen21_gpu_full_ffn_blocks.empty() ||
            (r.qwen21_w8a8 && hybrid_->ane_mlp_end == 6144);
        require(supported_edit_coverage,
                "Qwen21 32-layer W8A8 edit requires a 6144-channel manifest; use GPU fallback 3,5,7 for the 4096-channel route");
        if (r.operation == "image.edit" && r.qwen21_reference_size == 1024)
            require(r.qwen21_w8a8 && hybrid_->ane_mlp_end == 6144 &&
                        r.qwen21_gpu_full_ffn_blocks.empty(),
                    "Qwen21 full-reference W8A8 diagnostic requires a 6144-channel, 32-layer manifest");
        if (lora_base_ane)
            require(r.qwen21_w8a8 && hybrid_->ane_mlp_end == 6144 &&
                        r.qwen21_gpu_full_ffn_blocks.empty(),
                    "Qwen21 runtime LoRA/base ANE needs a 6144-channel full-coverage base manifest");
        require((fused_lora_ane ? (hybrid_->mlp_output_kind == "fused_lora" &&
                                     hybrid_->output_channels == 4096 + 6144 &&
                                     hybrid_->a8_graph == "sq_v1_both") :
                gate_up_ane ? (hybrid_->mlp_output_kind == "gate_up" &&
                                   hybrid_->output_channels == 12288 &&
                                   hybrid_->a8_graph == "sq_v1_input") :
                               (hybrid_->mlp_output_kind.empty() &&
                                hybrid_->output_channels == 4096)),
                "Qwen21 alternate FFN manifest requires its matching explicit mode");
        if (r.operation == "image.edit" && r.qwen21_reference_size == 512)
            require(r.qwen21_w8a8 && hybrid_->ane_mlp_end == 6144 &&
                        r.qwen21_gpu_full_ffn_blocks.empty(),
                    "Qwen21 512px-reference W8A8 edit requires a 6144-channel, 32-layer manifest");
        if (rectangular_w8a8)
            require(r.qwen21_w8a8 && hybrid_->ane_mlp_end == 6144 &&
                        r.qwen21_gpu_full_ffn_blocks.empty(),
                    "Qwen21 rectangular W8A8 requires the full-coverage 6144-channel manifest");
        require(hybrid_->hidden == 4096 && hybrid_->mlp_width == 12288 &&
                    hybrid_->ane_mlp_start == 0 && supported_partition &&
                    hybrid_->block_count == 32 && hybrid_->checkpoint_sha_verified &&
                    hybrid_->rows == (rectangular_w8a8 ? 1024 :
                        r.width / 16 * (r.height / 16)) &&
                    hybrid_->tensor_layout == "qwen21" &&
                    (r.qwen21_w8a8
                        ? hybrid_->export_variant == "int8_pc" &&
                          hybrid_->activation_precision == "int8" &&
                          hybrid_->a8_graph == (gate_up_ane ? "sq_v1_input" : "sq_v1_both") &&
                          hybrid_->projected_weight_granularity == "per_tensor"
                        : hybrid_->export_variant == "fp16" &&
                          hybrid_->activation_precision == "fp16") &&
                    hybrid_->output_scale == 1.f,
                "Qwen21 gpu_ane manifest precision/rows are not a verified 32-block partition");
        if (!hybrid_mlp_ || hybrid_runtime_options_ != runtime_options) {
            hybrid_mlp_.reset();
            hybrid_mlp_ = std::make_unique<HybridMLP>(
                transformer_, *hybrid_, r.qwen21_gpu_w8a16, r.qwen21_gpu_full_ffn_blocks);
            hybrid_runtime_options_ = runtime_options;
        }
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
    result.lora_applied_projections = lora_applied_projections_;
    result.prepared = prepare_only; result.warmup = warmup; result.prompt_cache_hit = hit;
    result.selection = resident_prefix
        ? "gpu: native Qwen Image 2.1 with experimental resident prefix KV cache"
        : "gpu: native Qwen Image 2.1 with request-owned prefix KV cache";
    result.db_cache_enabled = db_cache;
    result.db_cache_threshold = db_cache ? db_threshold : 0.f;
    result.db_cache_max_consecutive = db_cache ? db_max_consecutive : 0;
    const char *profile_segments = std::getenv("TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS");
    if (profile_segments && std::string_view(profile_segments) == "1")
        result.selection += "; diagnostic synchronized block-0 prefill attention segments";
    if (!r.loras.empty()) {
        const auto *adapter = viggle_v021_adapter(
            std::filesystem::path(r.loras[0].path).filename().string());
        result.selection += (fused_lora_ane || runtime_requested || !adapter)
            ? "; experimental runtime LoRA with Viggle six-step schedule; adapter quality unqualified"
            : "; Viggle v0.2.1 " + std::string(adapter->rank) + " runtime LoRA; six-step student schedule";
    }
    const char *norm_rope = std::getenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE");
    if (fused_qkv)
        result.selection += "; diagnostic Metal fused QKV projection, Q/K norm and RoPE";
    const char *local_references = std::getenv("TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION");
    if (local_references && std::string_view(local_references) == "1")
        result.selection += "; experimental reference-local prefill attention";
    if (local_references && std::string_view(local_references) == "2")
        result.selection += "; experimental last-reference-local prefill attention";
    if (local_references && std::string_view(local_references) == "3")
        result.selection += "; diagnostic last-16-block reference-local prefill attention";
    if (reuse_final_ffn)
        result.selection += "; experimental final-step cached GPU FFN approximation";
    if (hybrid_reuse_ffn)
        result.selection += "; diagnostic final-step cached hybrid FFN approximation";
    if (hybrid_reuse_last16)
        result.selection += "; diagnostic final-step cached last-16 hybrid FFN approximation";
    if (half_reuse_ffn)
        result.selection += hybrid_half_reuse
            ? "; diagnostic penultimate-step even-layer hybrid FFN reuse"
            : "; experimental penultimate-step even-layer FFN reuse";
    result.backend = "mlx_cpp_metal"; result.precision = "bf16";
    if (hybrid_requested) {
        result.backend = "mlx_cpp_metal+coreml";
        result.precision = r.qwen21_w8a8
            ? (r.qwen21_gpu_w8a16 ? "w8a16_gpu+w8a8_mlp_fp16_io" : "bf16_gpu+w8a8_mlp_fp16_io")
            : "bf16_gpu+fp16_mlp_fp16_io";
        result.selection = r.qwen21_w8a8
            ? "gpu_ane explicit experimental: W8A8 Core ML prefix + GPU FFN suffix, placement unverified"
            : "gpu_ane experimental: BF16 GPU suffix + FP16 Core ML CPU/ANE prefix; runtime placement not guaranteed";
        if (fused_qkv)
            result.selection += "; diagnostic Metal fused QKV projection, Q/K norm and RoPE";
        if (hybrid_reuse_ffn)
            result.selection += "; diagnostic final-step cached hybrid FFN approximation";
        if (hybrid_reuse_last16)
            result.selection += "; diagnostic final-step cached last-16 hybrid FFN approximation";
        if (hybrid_half_reuse)
            result.selection += "; diagnostic penultimate-step even-layer hybrid FFN reuse";
        if (local_references && std::string_view(local_references) == "3")
            result.selection += "; diagnostic last-16-block reference-local prefill attention";
        if (tiled_prefill)
            result.selection += "; diagnostic first-step last " + std::to_string(tiled_prefill_layers) +
                " layers 1024-row tiled W8A8 FFN";
        if (rectangular_w8a8)
            result.selection += "; diagnostic 1536-row decode FFN in two 1024-row W8A8 tiles";
        if (fused_lora_ane && r.loras.empty())
            result.selection += "; experimental reusable frozen-base fused FFN with zero runtime LoRA input";
        else if (lora_base_ane)
            result.selection += fused_lora_ane
                ? "; diagnostic fused base W8A8 ANE FFN with pre-SiLU runtime LoRA and GPU down LoRA"
                : gate_up_ane
                ? "; diagnostic base W8A8 ANE gate/up, GPU pre-SiLU LoRA and complete BF16 down projection"
                : "; diagnostic runtime LoRA on GPU FFN suffix only, base W8A8 ANE prefix unchanged";
        else if (gate_up_ane)
            result.selection += "; diagnostic base W8A8 ANE gate/up, GPU SiLU and BF16 down projection";
        if (tiled_prefix_reuse)
            result.selection += "; diagnostic repeated tiled-prefill prefix KV";
        if (prefix_target_only)
            result.selection += "; diagnostic lossy target-only W8A8 prefix hit";
    }
    if (last_target_only)
        result.selection += "; diagnostic final prefill block target-only output";
    if (db_cache)
        result.selection += "; diagnostic decode DBCache (front 8, back 0, warmup 8)";
    if (runtime_requested) {
        result.backend = "mlx_cpp_metal+coreml_runtime_weight";
        result.precision = "bf16_gpu+runtime_fp16_ffn_bf16_io";
        result.selection = "gpu_ane explicit runtime-weight token-row FFN; base-only weight slots with optional GPU LoRA activation corrections; physical placement unverified";
        if (!r.loras.empty()) result.selection += "; six-step student schedule; alternate adapter quality unqualified";
    }
    if (qkv_requested) {
        result.backend = "mlx_cpp_metal+coreml_runtime_qkv";
        result.precision = "bf16_gpu+runtime_fp16_qkv_bf16_io";
        result.selection = "gpu_ane explicit diagnostic runtime-weight QKV token-row projection + original GPU FFN; physical placement and whole-request gain unverified";
    }
    // Hybrid route descriptions replace the initial GPU description. Keep
    // these GPU-kernel receipts after those replacements, on every route, and
    // do not label base-only requests as using a LoRA approximation.
    if (lora_fp16 && !r.loras.empty())
        result.selection += "; experimental FP16 low-rank LoRA matmuls";
    if (norm_rope && std::string_view(norm_rope) == "1")
        result.selection += "; experimental fused Metal Q/K norm-RoPE";
    result.timings.hybrid = (hybrid_requested || runtime_requested || qkv_requested) ? seconds(hybrid_start) : 0;
    result.checkpoint = "qwen_image_2.1_bf16.safetensors";
    result.text_tokens = result.valid_text_tokens = text.shape(1);
    for (const auto &ref : references) result.reference_tokens += ref.latents.shape(1);
    result.total_tokens = result.text_tokens + result.reference_tokens + r.height / 16 * (r.width / 16);
    result.timings.text = text_seconds; result.timings.image = image_seconds;
    emit(event, (hybrid_requested || runtime_requested || qkv_requested) ? "route_gpu_ane" : "route_gpu", 1, 1);
    const uint64_t coreml_calls_before = hybrid_requested ? hybrid_->metrics().runtime_calls : 0;
    std::optional<Transformer::PrefixSnapshot> pending_snapshot;
    std::string pending_snapshot_runtime;
    if (!prepare_only) {
        auto schedule = r.loras.empty() ? sigmas(r.width, r.height, r.steps) :
                                      viggle_v021_sigmas(r.width, r.height);
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
        // A 32-layer BF16 K/V bank costs roughly 512 KiB per prefix token.
        // Keep at most one bank, within 8 GiB and 1/8 of physical memory.
        const uint64_t prefix_tokens = uint64_t(text.shape(1)) + result.reference_tokens;
        const uint64_t prefix_bytes = prefix_tokens * 2ULL * 32 * 32 * 128 * 2;
        const uint64_t physical_bytes = device_info().physical_memory;
        const bool retain_prefix = resident_prefix &&
            prefix_bytes <= std::min(uint64_t(8) << 30, physical_bytes / 8) &&
            mx::get_active_memory() < physical_bytes - std::min(physical_bytes, prefix_bytes + (uint64_t(8) << 30));
        if (!retain_prefix) clear_prefix_cache();
        const float first_sigma = schedule.data<float>()[0];
        const std::string prefix_runtime = std::to_string(r.steps) + ":" +
            std::to_string(r.width) + ":" + std::to_string(r.height) + ":" +
            active_lora_identity_ + ":" + (lora_fp16 ? "fp16" : "fp32") + ":" +
            (hybrid_requested ? hybrid_manifest_ + hybrid_runtime_options_ :
             runtime_requested ? runtime_manifest_ : qkv_requested ? qkv_manifest_ : "gpu") + ":" +
            (fused_qkv ? "fused-qkv" : "ordinary-qkv") + ":" +
            (norm_rope ? std::string(norm_rope) : "") + ":" +
            (std::getenv("TURBOCIDER_QWEN21_METAL_QK_ROPE") ?
                std::getenv("TURBOCIDER_QWEN21_METAL_QK_ROPE") : "") + ":" +
            (local_references ? local_references : "") + ":" +
            (tiled_prefill ? "tiled-prefill-" + std::to_string(tiled_prefill_layers) : "exact-prefill") + ":" +
            (tiled_prefix_reuse ? "tiled-prefix-tail-reuse" : "no-tiled-prefix-reuse") + ":" +
            (prefix_target_only ? "prefix-hit-target-only" : "prefix-hit-matched-tile") + ":" +
            (last_target_only ? "last-target-only" : "full-last-block") + ":" +
            (reuse_final_ffn ? "final-ffn-reuse" : "no-ffn-reuse") + ":" +
            (hybrid_reuse_ffn ? "hybrid-final-ffn-reuse" :
             hybrid_reuse_last16 ? "hybrid-final-last16-reuse" : "no-hybrid-ffn-reuse") + ":" +
            (hybrid_half_reuse ? "hybrid-half-penultimate-reuse" :
             half_reuse_ffn ? "half-penultimate-reuse" : "no-half-reuse");
        if (cached_prefix_transformer_ &&
            (cached_prefix_runtime_ != prefix_runtime || cached_prefix_sigma_ != first_sigma ||
             !cached_prefix_transformer_->prefix_matches(text, r.height / 16, r.width / 16, references)))
            clear_prefix_cache();
        const bool prefix_hit = retain_prefix && bool(cached_prefix_transformer_);
        std::unique_ptr<Transformer> request_dit;
        if (!retain_prefix) request_dit = std::make_unique<Transformer>(
            transformer_, TransformerConfig{}, fused_qkv ? &fused_qkv_weights_ : nullptr);
        else if (!cached_prefix_transformer_)
            cached_prefix_transformer_ = std::make_unique<Transformer>(
                transformer_, TransformerConfig{}, fused_qkv ? &fused_qkv_weights_ : nullptr);
        Transformer &dit = retain_prefix ? *cached_prefix_transformer_ : *request_dit;
        const uint64_t snapshot_memory_limit = r.memory_budget_bytes
            ? std::min(physical_bytes, r.memory_budget_bytes) : physical_bytes;
        const uint64_t snapshot_budget = std::min(uint64_t(8) << 30, snapshot_memory_limit / 8);
        const bool snapshot_allowed = snapshot_route && prefix_bytes <= snapshot_budget;
        const auto checkpoint_path = root_ / "diffusion_models/qwen_image_2.1_bf16.safetensors";
        const std::string snapshot_runtime = snapshot_allowed ? prefix_runtime + ":" +
            std::to_string(std::filesystem::file_size(checkpoint_path)) + ":" +
            std::to_string(static_cast<long long>(
                std::filesystem::last_write_time(checkpoint_path).time_since_epoch().count())) : "";
        if (!snapshot_allowed || cached_snapshot_runtime_ != snapshot_runtime)
            clear_prefix_snapshot();
        const bool snapshot_hit = cached_prefix_snapshot_ &&
            dit.import_prefix_snapshot(*cached_prefix_snapshot_, text,
                                       r.height / 16, r.width / 16, references);
        if (!snapshot_hit) clear_prefix_snapshot();
        if (snapshot_allowed)
            result.selection += snapshot_hit ? "; edit prefix KV snapshot hit" : "; edit prefix KV snapshot miss";
        dit.configure_db_cache(db_cache, db_threshold, r.steps, db_max_consecutive);
        result.selection += retain_prefix ?
            (prefix_hit ? "; experimental resident prefix KV hit" : "; experimental resident prefix KV miss") : "";
        auto dit_start = Clock::now();
        {
            // Keep the original fused gate/up GPU FFN on both full-GPU
            // scheduler probes and the GPU row head. With LoRA use the same
            // native projection path as the ordinary GPU, never raw matrices
            // that would omit the adapter contributions.
            std::function<std::vector<Tensor>(const std::vector<Tensor> &)> runtime_gpu;
            if (runtime_requested) runtime_gpu = mx::compile([](const std::vector<Tensor> &a) {
                auto gu = mx::split(mx::matmul(a[0], mx::transpose(a[1])), 2, -1);
                return std::vector<Tensor>{mx::matmul(silu(gu[0]) * gu[1], mx::transpose(a[2]))};
            });
            std::vector<std::vector<Tensor>> runtime_weights;
            using RuntimeFunction = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;
            RuntimeFunction gpu_qkv;
            std::vector<RuntimeFunction> runtime_lora_gpu, runtime_lora_gate_up, runtime_lora_down_add;
            if (runtime_requested) {
                for (int block = 0; block < 32; ++block) {
                    const auto p = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
                    auto gu = mx::split(transformer_.at(p + "gate_up.weight"), 2, 0);
                    runtime_weights.push_back({gu[0], gu[1], transformer_.at(p + "out.weight")});
                    mx::eval(runtime_weights.back());
                    if (transformer_.has_runtime_loras()) {
                        // The ordinary GPU Transformer compiles this same
                        // projection/low-rank arithmetic inside its blocks.
                        // Preserve fusion across the new FFN boundary too.
                        // Closures are request-local: never reuse captures
                        // after a different adapter has rebound the weights.
                        runtime_lora_gpu.push_back(mx::compile([this, p](const std::vector<Tensor> &a) {
                            auto gu = mx::split(transformer_.project(a[0], p + "gate_up"), 2, -1);
                            return std::vector<Tensor>{transformer_.project(silu(gu[0]) * gu[1], p + "out")};
                        }));
                        runtime_lora_gate_up.push_back(mx::compile([this, p](const std::vector<Tensor> &a) {
                            // Request each logical half directly. Separate
                            // gate/up adapters otherwise pad both corrections
                            // with zeros to the full fused width, add them,
                            // then split again. Keep the same per-half FP32
                            // accumulation and final rounding; fused/stacked
                            // adapters still intersect both requested ranges.
                            auto gate = transformer_.lora_delta_slice(a[0], p + "gate_up",
                                0, 12288, 0, 4096);
                            auto up = transformer_.lora_delta_slice(a[0], p + "gate_up",
                                12288, 24576, 0, 4096);
                            return std::vector<Tensor>{mx::contiguous(gate), mx::contiguous(up)};
                        }));
                        runtime_lora_down_add.push_back(mx::compile([this, p](const std::vector<Tensor> &a) {
                            // Keep the existing BF16 delta rounding, FP32 add
                            // and final cast; only enlarge the GPU graph.
                            auto delta = transformer_.lora_delta_slice(a[0], p + "out", 0, 4096, 0, 12288);
                            return std::vector<Tensor>{mx::astype(mx::astype(a[1], mx::float32) +
                                mx::astype(delta, mx::float32), a[1].dtype())};
                        }));
                    }
                }
                dit.set_plan_mlp([&](int block, int rows) {
                    checkpoint(cancelled);
                    const auto plan = runtime_ffn_->plan_block(block, rows);
                    if (plan.mode == ane::RowScheduler::Mode::HybridUntimed)
                        return Transformer::MLPPlan::SplitUntimed;
                    return plan.split() ? Transformer::MLPPlan::Split :
                        plan.mode == ane::RowScheduler::Mode::GpuProbe ? Transformer::MLPPlan::GpuProbe
                                                                     : Transformer::MLPPlan::Gpu;
                });
                dit.set_observe_mlp([&](int block, int rows, double seconds) {
                    runtime_ffn_->observe_block(block, rows, seconds);
                });
                dit.set_stage_mlp([&](int block, int rows) {
                    checkpoint(cancelled);
                    runtime_ffn_->stage(block, rows, runtime_weights.at(block));
                });
                auto run_ffn = [&](int block, const Tensor &input) {
                    const auto p = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
                    ane::HybridFfn::Adapter adapter{
                        [&](const Tensor &x) {
                            auto gu = runtime_lora_gate_up.at(block)({x});
                            return std::make_pair(gu[0], gu[1]);
                        },
                        [&](const Tensor &h, const Tensor &base) {
                            return runtime_lora_down_add.at(block)({h, base})[0];
                        }};
                    return runtime_ffn_->run(block, input, [&](const Tensor &x) {
                        if (transformer_.has_runtime_loras()) return runtime_lora_gpu.at(block)({x})[0];
                        return runtime_gpu({x, transformer_.at(p + "gate_up.weight"),
                                              transformer_.at(p + "out.weight")})[0];
                    }, cancelled, transformer_.has_runtime_loras() ? &adapter : nullptr);
                };
                dit.set_prefill_mlp(run_ffn);
                dit.set_decode_mlp(run_ffn);
            }
            if (qkv_requested) {
                // Keep three checkpoint matrices separate on the GPU and in
                // the runtime slot. Neither stage nor fallback builds a
                // resident packed bank or changes norm/RoPE/attention/FFN.
                gpu_qkv = mx::compile([](const std::vector<Tensor> &a) {
                    return std::vector<Tensor>{mx::concatenate({
                        mx::matmul(a[0], mx::transpose(a[1])),
                        mx::matmul(a[0], mx::transpose(a[2])),
                        mx::matmul(a[0], mx::transpose(a[3]))}, -1)};
                });
                dit.set_plan_qkv([&](int block, int rows) {
                    checkpoint(cancelled);
                    const auto plan = runtime_qkv_->plan_block(block, rows);
                    using Mode = ane::QkvScheduler::Mode;
                    if (plan.mode == Mode::Hybrid) return Transformer::QKVPlan::HybridTimed;
                    if (plan.mode == Mode::HybridUntimed) return Transformer::QKVPlan::Hybrid;
                    return plan.mode == Mode::GpuProbe ? Transformer::QKVPlan::GpuProbe
                                                        : Transformer::QKVPlan::Gpu;
                });
                dit.set_observe_qkv([&](int block, int rows, double seconds) {
                    runtime_qkv_->observe_block(block, rows, seconds);
                });
                dit.set_stage_qkv([&](int block, int rows) {
                    checkpoint(cancelled);
                    const auto p = "transformer_blocks." + std::to_string(block) + ".attn.to_";
                    runtime_qkv_->stage(block, rows, {transformer_.at(p + "q.weight"),
                        transformer_.at(p + "k.weight"), transformer_.at(p + "v.weight")});
                });
                dit.set_project_qkv([&](int block, const Tensor &input) {
                    return runtime_qkv_->run(block, input, [&](int gpu_block, const Tensor &x) {
                        const auto p = "transformer_blocks." + std::to_string(gpu_block) + ".attn.to_";
                        return gpu_qkv({x, transformer_.at(p + "q.weight"),
                            transformer_.at(p + "k.weight"), transformer_.at(p + "v.weight")})[0];
                    }, cancelled);
                });
                // QKV callbacks are cleared before gpu_qkv leaves this scope.
            }
            if (tiled_prefill)
                dit.set_prefill_mlp([this, &cancelled, last_target_only](int block, const Tensor &input) {
                    checkpoint(cancelled);
                    require(bool(hybrid_mlp_), "Qwen21 tiled prefill lost its Core ML session");
                    return last_target_only && input.shape(1) == hybrid_->rows
                        ? (*hybrid_mlp_)(block, input)
                        : hybrid_mlp_->tiled_sequence(block, input);
                }, 32 - tiled_prefill_layers,
                    tiled_prefix_reuse && retain_prefix && !prefix_target_only);
            // The App records the first denoise event as the progress/time
            // origin. Emit it before sampling so step one is not subtracted.
            emit(event, "denoise", 0, r.steps);
            for (int step = 0; step < r.steps; ++step) {
                checkpoint(cancelled);
                dit.set_db_cache_step(step);
                if (step == 0 && prefix_hit && tiled_prefix_reuse)
                    dit.set_decode_mlp([this, &cancelled, &dit, last_target_only,
                                        prefix_target_only, prefix_tokens](int block,
                                                                                   const Tensor &input) {
                        checkpoint(cancelled);
                        require(bool(hybrid_mlp_), "Qwen21 cached tiled prefill lost its Core ML session");
                        // An aligned prefix has no partial tile to reconstruct:
                        // its 1,024 target rows already form the original tile.
                        if (prefix_target_only || prefix_tokens % hybrid_->rows == 0 ||
                            (last_target_only && block == 31))
                            return (*hybrid_mlp_)(block, input);
                        return hybrid_mlp_->tiled_target_with_prefix_tail(
                            block, input, dit.prefill_tile_tail(block));
                    }, 32 - tiled_prefill_layers);
                // On a cross-request prefix-KV hit the first step reuses the
                // prefix too, but its target FFN must remain full BF16 GPU.
                // Install the hybrid callback only for actual decode steps.
                if (hybrid_requested && step == 1)
                    dit.set_decode_mlp([this, &cancelled, rectangular_w8a8](int block, const Tensor &input) {
                        checkpoint(cancelled);
                        require(bool(hybrid_mlp_), "Qwen21 hybrid callback lost its Core ML session");
                        return rectangular_w8a8
                            ? hybrid_mlp_->tiled_sequence(block, input)
                            : (*hybrid_mlp_)(block, input);
                    });
                const auto step_started = profile_steps ? Clock::now() : Clock::time_point{};
                const auto prediction_before = profile_steps && hybrid_requested
                    ? hybrid_->metrics().prediction_seconds : 0.;
                if (reuse_final_ffn || hybrid_reuse_ffn || hybrid_reuse_last16)
                    dit.set_ffn_cache_mode(half_reuse_ffn && step == r.steps - 3 ? Transformer::FFNCacheMode::Capture :
                                           half_reuse_ffn && step == r.steps - 2 ? Transformer::FFNCacheMode::ReuseEvenAndCapture :
                                           !half_reuse_ffn && step == r.steps - 2 ? Transformer::FFNCacheMode::Capture :
                                           step == r.steps - 1 ? (hybrid_reuse_last16 ? Transformer::FFNCacheMode::ReuseLast16 :
                                                                  Transformer::FFNCacheMode::Reuse) :
                                           Transformer::FFNCacheMode::Off);
                auto noise = dit.forward(latents, text, schedule.data<float>()[step], r.height / 16, r.width / 16,
                                         true, nullptr, references);
                latents = latents + noise * Tensor(schedule.data<float>()[step+1] - schedule.data<float>()[step], latents.dtype());
                const auto graph_seconds = profile_steps ? seconds(step_started) : 0.;
                mx::eval(latents);
                require(mx::all(mx::isfinite(latents)).item<bool>(), "nonfinite Qwen21 latent");
                if (profile_steps) {
                    const auto elapsed = seconds(step_started);
                    const auto prediction = hybrid_requested
                        ? hybrid_->metrics().prediction_seconds - prediction_before : 0.;
                    std::cerr << "{\"qwen21_step\":" << step
                              << ",\"phase\":\"" << (step == 0 && !prefix_hit && !snapshot_hit ? "prefill" : "decode")
                              << "\",\"seconds\":" << elapsed
                              << ",\"graph_build_seconds\":" << graph_seconds
                              << ",\"evaluate_seconds\":" << elapsed - graph_seconds
                              << ",\"prefix_cache_hit\":" << (prefix_hit || snapshot_hit ? "true" : "false")
                              << ",\"coreml_prediction_api_seconds\":" << prediction
                              << ",\"reference_tokens\":" << result.reference_tokens
                              << ",\"hybrid\":" << (hybrid_requested ? "true" : "false")
                              << "}" << std::endl;
                }
                emit(event, "denoise", step + 1, r.steps);
            }
            if (hybrid_requested) dit.set_decode_mlp({});
            if (tiled_prefill) dit.set_prefill_mlp({});
            if (runtime_requested) {
                dit.set_plan_mlp({}); dit.set_stage_mlp({});
                dit.set_observe_mlp({});
                dit.set_decode_mlp({}); dit.set_prefill_mlp({});
                runtime_ffn_->drain();
            }
            if (qkv_requested) {
                dit.set_stage_qkv({}); dit.set_project_qkv({});
                dit.set_plan_qkv({}); dit.set_observe_qkv({});
                runtime_qkv_->drain();
            }
            if (retain_prefix) {
                cached_prefix_runtime_ = prefix_runtime;
                cached_prefix_sigma_ = first_sigma;
            }
        }
        result.db_cache_steps = dit.db_cached_steps();
        result.timings.denoise = seconds(dit_start); result.actual_steps = r.steps;
        if (r.qwen21_w8a8) {
            const W8A8CallBudget budget{
                .steps = r.steps,
                .decode_layers = int(32 - r.qwen21_gpu_full_ffn_blocks.size()),
                .decode_tiles = rectangular_w8a8 ? 2 : 1,
                .db_cached_steps = result.db_cache_steps,
                .db_skipped_layers = 32 - Transformer::db_front_blocks - Transformer::db_back_blocks,
                .final_reuse_layers = hybrid_reuse_ffn ? 32 : hybrid_reuse_last16 ? 16 : 0,
                .penultimate_reuse_layers = hybrid_half_reuse ? 16 : 0,
                .tiled_prefill_layers = tiled_prefill_layers,
                .prefix_tokens = prefix_tokens,
                .total_tokens = uint64_t(result.total_tokens),
                .tile_rows = hybrid_->rows,
                .prefix_hit = prefix_hit,
                .tiled_prefix_reuse = tiled_prefix_reuse,
                .prefix_target_only = prefix_target_only,
                .last_target_only = last_target_only,
            };
            require(hybrid_->metrics().runtime_calls - coreml_calls_before ==
                        expected_w8a8_calls(budget),
                    "Qwen21 W8A8 did not execute the required FFN layer coverage");
        }
        dump("qwen21_latents", latents);
        if (snapshot_allowed && !snapshot_hit &&
            mx::get_active_memory() < snapshot_memory_limit -
                std::min(snapshot_memory_limit, prefix_bytes + (uint64_t(8) << 30))) {
            checkpoint(cancelled);
            pending_snapshot = dit.export_prefix_snapshot(snapshot_budget);
            if (pending_snapshot) pending_snapshot_runtime = snapshot_runtime;
        }
        if (r.residency == "component_staged") {
            // Compiled block functions and prefix tensors can retain weights
            // after the session map is cleared. Destroy their owner before
            // entering VAE decoding so staged residency releases the DiT.
            request_dit.reset();
            hybrid_mlp_.reset(); clear_prefix_cache();
            fused_qkv_weights_.clear(); transformer_.clear();
            mx::synchronize(); mx::clear_cache();
        }
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
    if (runtime_requested) {
        result.hybrid = runtime_ffn_->metrics();
        if (!runtime_ffn_->available()) result.selection += "; GPU fallback: " + runtime_ffn_->reason();
    }
    if (qkv_requested) {
        result.qkv = runtime_qkv_->metrics();
        if (!runtime_qkv_->available())
            result.selection += "; GPU fallback: " + result.qkv->failure_reason;
    }
    if (!prepare_only && r.residency == "component_staged") {
        hybrid_mlp_.reset(); hybrid_.reset(); hybrid_manifest_.clear(); hybrid_runtime_options_.clear();
        clear_prefix_cache(); fused_qkv_weights_.clear();
        transformer_.clear(); vae_.clear();
        mx::synchronize(); mx::clear_cache();
    }
    // Publish reusable values only after a successful request. Do not add a
    // cancellation throw after PNG export: publication already completed.
    if (cancelled.load()) clear_prefix_snapshot();
    else if (pending_snapshot) {
        cached_prefix_snapshot_ = std::move(pending_snapshot);
        cached_snapshot_runtime_ = std::move(pending_snapshot_runtime);
    }
    result.timings.wall = seconds(start);
    result.active_bytes = mx::get_active_memory(); result.peak_bytes = mx::get_peak_memory();
    return result;
} catch (...) {
    if (runtime_ffn_) runtime_ffn_->drain();
    if (runtime_qkv_) runtime_qkv_->drain();
    try { mx::synchronize(); } catch (...) {}
    clear_prefix_cache();
    clear_prefix_snapshot();
    fused_qkv_weights_.clear();
    hybrid_mlp_.reset(); hybrid_.reset(); hybrid_manifest_.clear(); hybrid_runtime_options_.clear();
    if (transformer_.has("transformer_blocks.0.attn.qkv_packed.weight"))
        transformer_.clear(); // A cancelled pack may have replaced only some layers.
    if (requested.residency == "component_staged") {
        // Cancellation must honor the same small idle footprint as a finished
        // staged request. Fully evaluated conditioning remains safe to reuse.
        transformer_.clear(); vae_.clear();
        active_lora_identity_.clear(); lora_applied_projections_ = 0;
        mx::clear_cache();
    }
    throw;
}
} // namespace tc::qwen21
