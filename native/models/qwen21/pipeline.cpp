#include "pipeline.hpp"
#include "diagnostic_options.hpp"
#include "conditioning.hpp"
#include "transformer.hpp"
#include "vae.hpp"
#include "scheduler.hpp"
#include "pe_generation.hpp"
#include "runtime_ffn_graphs.hpp"
#include "runtime_gpu_layer_config.hpp"
#include "../../media/image.hpp"
#include "../../runtime/residency.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include "../../backends/ane_backend.hpp"
#include "../../platform/apple/platform.hpp"
#include <mlx/random.h>
#include <bit>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string_view>
#include <map>
#include <tuple>

namespace tc::qwen21 {
namespace {
void emit(const Event &event, const std::string &phase, int step, int total) {
    if (event) event(phase, step, total);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
size_t runtime_ane_budget(uint64_t physical, uint64_t active, uint64_t resident_optional = 0) {
    constexpr uint64_t system_margin = uint64_t(4) << 30;
    constexpr uint64_t optional_cap = uint64_t(2) << 30;
    // Share the same optional-tier allowance between FFN and QKV. Subtract
    // before adding active memory so even an invalidly large reading cannot
    // wrap around and appear to leave headroom.
    const uint64_t headroom = physical > system_margin ? physical - system_margin : 0;
    const uint64_t remaining_optional=resident_optional<optional_cap ? optional_cap-resident_optional : 0;
    return std::min(remaining_optional, active < headroom ? headroom - active : uint64_t(0));
}
std::string runtime_ffn_identity(const std::filesystem::path &manifest) {
    return manifest.string()+":"+sha256_file(manifest)+":"+
        (std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS") ? std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS") : "auto")+
        ane::HybridFfn::executor_configuration_identity();
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
    transformer_source_=component_source(root,"diffusion_models/qwen_image_2.1_bf16.safetensors",
        "diffusion_models/qwen-image-2.1-Q4_K_M.gguf");
    encoder_source_=component_source(root,"text_encoders/qwen3vl_8b_bf16.safetensors",
        "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf");
    for (const char *relative : {"vae/qwen_image_2.1_vae_bf16.safetensors", "processor/tokenizer.json"})
        require(std::filesystem::is_regular_file(root / relative), std::string("missing Qwen Image 2.1 asset: ") + relative);
}
LoadResult Session::load(const Event &event, std::atomic<bool> &cancelled) {
    checkpoint(cancelled);
    if(transformer_gguf_)transformer_gguf_->bank->check_unchanged();
    if (!transformer_.bytes()) {
        emit(event, "load_qwen21_transformer", 0, 1);
        if(transformer_source_.extension()==".gguf")
            transformer_gguf_=load_gguf_component(transformer_source_,transformer_,false,event,cancelled);
        else transformer_.load_file(transformer_source_);
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
void Session::prepare_transformer(const Request &r,const Event &event,std::atomic<bool> &cancelled,
                                 bool experimental_adapter,bool fused_qkv,bool lora_fp16) {
    std::string identity;
    std::shared_ptr<const streaming::SourceLease> adapter_lease;
    if(!r.loras.empty()) {
        auto path=std::filesystem::canonical(r.loras[0].path);
        require(std::filesystem::is_regular_file(path),"Qwen21 LoRA is not a regular file");
        identity=path.string()+":"+std::to_string(std::filesystem::file_size(path))+":"+
            std::to_string(static_cast<long long>(std::filesystem::last_write_time(path).time_since_epoch().count()))+":"+
            std::to_string(std::bit_cast<uint32_t>(r.loras[0].strength));
        if(experimental_adapter) {
            streaming::SourceFileIdentity source;source.logical_id="qwen21-adapter";source.path=r.loras[0].path;
            // Full hash on the first capture; subsequent requests may consume
            // ONLY native process proofs for the same dev/ino/size/mtime/ctime.
            // Caller digests/metadata are never imported as proof. The same
            // held descriptors feed the first actual low-rank source binding.
            adapter_lease=streaming::SourceLease::capture_verified({source},&cancelled);
            require(adapter_lease->file("qwen21-adapter").canonical_target_path==path,
                    "Qwen21 adapter alias changed during identity capture");
            identity+=":"+adapter_lease->file("qwen21-adapter").content_digest;
        }
    }
    const bool bind=!r.loras.empty() && (active_lora_identity_!=identity || !transformer_.bytes());
    if(active_lora_identity_!=identity) {
        if(runtime_ffn_)runtime_ffn_->drain();
        hybrid_mlp_.reset();clear_prefix_cache();fused_qkv_weights_.clear();transformer_.clear();
        active_lora_identity_.clear();lora_applied_projections_=0;
    }
    if(!fused_qkv && transformer_.has("transformer_blocks.0.attn.qkv_packed.weight")) {
        hybrid_mlp_.reset();clear_prefix_cache();fused_qkv_weights_.clear();transformer_.clear();
    }
    if(bind)require(experimental_adapter || sha256_file(r.loras[0].path)==
        "2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803",
        "Viggle v0.2.1 r256 LoRA hash does not match the pinned adapter");
    load(event,cancelled);transformer_.set_runtime_lora_fp16(lora_fp16);
    if(bind) {
        lora_applied_projections_=adapter_lease ? transformer_.apply_loras_leased(r.loras,"transformer",event,cancelled,true,
            adapter_lease,{"qwen21-adapter"}) : transformer_.apply_loras(r.loras,"transformer",event,cancelled,true);
        require(experimental_adapter ? lora_applied_projections_>0 : lora_applied_projections_==227,
            "Qwen21 LoRA did not bind transformer projections");
        active_lora_identity_=std::move(identity);
    }
    if(adapter_lease) {
        adapter_lease->revalidate_after_drain();
        if(option_enabled(std::getenv("TURBOCIDER_QWEN21_PROFILE_STEPS")))
            std::cerr<<"{\"qwen_lora_source_verification\":{\"bytes_read\":"<<adapter_lease->verification_bytes_read()
                <<",\"native_cache_hits\":"<<adapter_lease->verification_cache_hits()<<",\"source_files\":"<<adapter_lease->file_count()
                <<",\"bound_this_request\":"<<(bind?"true":"false")
                <<",\"scope\":\"native generation-bound full SHA256; held-fd/path revalidation; bound A/B materialized before successful leased bind\"}}\n";
    }
}
void Session::unload() {
    encoder_runtime_.reset(); encoder_runtime_identity_.clear();
    encoder_weights_.reset();encoder_weight_identity_.clear();encoder_weight_loads_=0;
    encoder_gguf_.reset();
    runtime_ffn_.reset(); runtime_manifest_.clear();
    runtime_qkv_.reset(); qkv_manifest_.clear();
    clear_prefix_cache();
    fused_qkv_weights_.clear();
    hybrid_mlp_.reset();
    hybrid_.reset();
    hybrid_manifest_.clear();
    hybrid_runtime_options_.clear();
    conditioning_cache_.clear();
    transformer_.clear();
    transformer_gguf_.reset();
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
    require(binary_option_or_unset(std::getenv("TURBOCIDER_QWEN21_LORA_BF16_AB")),"Qwen joint BF16 A/B requires0 or1");
    const bool joint_ab=joint_bf16_lora_ab(r);
    const auto ffn_phase=runtime_ffn_phase(std::getenv("TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE"));
    require(ffn_phase!=RuntimeFfnPhase::Invalid,"Qwen runtime FFN phase requires all, prefill or decode");
    const auto prefill_gpu_layers=configured_prefill_gpu_layers(r);
    const auto ffn_phase_identity=runtime_ffn_phase_identity(ffn_phase)+prefill_gpu_layer_identity(prefill_gpu_layers);
    const int student_reuse_layers=student_ffn_reuse_layers(std::getenv("TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE"));
    require(student_reuse_layers>=0,"Qwen student final FFN reuse requires 0,1,16 or32");
    const bool student_ffn_reuse=student_final_ffn_reuse(r);
    const char *b_epilogue_flag=std::getenv("TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE");
    require(binary_option_or_unset(b_epilogue_flag),"Qwen fused B epilogue requires0 or1");
    const bool b_epilogue=(option_enabled(b_epilogue_flag) || joint_ab) && !r.loras.empty();
    if(transformer_.runtime_lora_b_epilogue()!=b_epilogue) {
        if(runtime_ffn_)runtime_ffn_->drain();
        mx::synchronize();transformer_.set_runtime_lora_b_epilogue(b_epilogue);
    }
    const auto *bf16_rank_flag=std::getenv("TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS");
    require(binary_option_or_unset(bf16_rank_flag),"Qwen BF16 operand/F32 ranks require 0 or 1");
    const bool bf16_operand_ranks=(option_enabled(bf16_rank_flag) || joint_ab) && !r.loras.empty();
    if(transformer_.runtime_lora_bf16_fp32_ranks()!=bf16_operand_ranks) {
        if(runtime_ffn_)runtime_ffn_->drain();
        mx::synchronize();transformer_.set_runtime_lora_bf16_fp32_ranks(bf16_operand_ranks);
    }
    const char *share_rank_flag=std::getenv("TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS");
    require(binary_option_or_unset(share_rank_flag),"Qwen shared LoRA ranks require 0 or 1");
    const bool share_lora_ranks=option_enabled(share_rank_flag) && runtime_requested && !r.loras.empty();
    const char *down_rank_flag=std::getenv("TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS");
    require(binary_option_or_unset(down_rank_flag),"Qwen split down ranks require 0 or 1");
    const bool split_down_ranks=option_enabled(down_rank_flag) && runtime_requested && !r.loras.empty();
    const bool qkv_requested = r.hybrid_mlp_mode == "runtime_qkv";
    const bool hybrid_requested = r.execution == "gpu_ane" && !runtime_requested && !qkv_requested;
    const char *encoder_weights_flag=std::getenv("TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS");
    require(binary_option_or_unset(encoder_weights_flag),"Qwen encoder weight retention requires 0 or 1");
    const bool retain_encoder_weights=option_enabled(encoder_weights_flag);
    const bool gguf_transformer=transformer_source_.extension()==".gguf";
    const bool gguf_encoder=encoder_source_.extension()==".gguf";
    const char *shared_down_flag=std::getenv("TURBOCIDER_QWEN21_GGUF_SHARED_DOWN");
    require(binary_option_or_unset(shared_down_flag),"Qwen GGUF shared down requires0 or1");
    const bool shared_down=option_enabled(shared_down_flag);
    require(!shared_down || (gguf_transformer && r.allow_approximation && r.residency=="resident" &&
        r.width==512 && r.height==512 && !r.streaming.active() && !r.memory_constrained.enabled),
        "Qwen GGUF shared down requires approximate resident512 mixed-K source");
    if(gguf_transformer || gguf_encoder)
        require(r.operation=="image.generate" && r.inputs.empty() &&
            (r.execution=="gpu" || (gguf_transformer && r.execution=="gpu_ane" && runtime_requested &&
                ane::configured_backend().preferred==ane::BackendPreference::Private && ane::private_channel_count(12288)>0 &&
                !ane::configured_fp32_channel_join())) &&
            r.allow_approximation && r.loras.empty() && (r.execution!="gpu" || r.ane_manifest.empty()) && r.encoder_ane_manifest.empty() &&
            !r.prompt_enhance && !r.qwen21_gpu_w8a16 && !r.qwen21_w8a8,
            "Qwen21 mixed GGUF requires approximate GPU/fixed Private FP16 channel base generation; encoder hybrid/edit/LoRA qualification pending");
    const bool compile_encoder_gpu=compiled_encoder_gpu(r);
    if(retain_encoder_weights)
        require(r.residency=="resident" && r.width==512 && r.height==512 && !r.prompt_enhance &&
            !r.memory_constrained.enabled && !r.streaming.active() && !r.memory_budget_bytes,
            "Qwen encoder weight retention requires unconstrained resident 512px execution without prompt enhancement");
    std::optional<int> encoder_channels;
    if(const char *raw=std::getenv("TURBOCIDER_QWEN21_ENCODER_ANE_CHANNELS")) {
        const int channels=ane::parse_private_channel_count(12288,raw);
        require(channels>=0,"Qwen encoder channel override requires a fixed aligned width, not auto");
        encoder_channels=channels;
    }
    if(!retain_encoder_weights && encoder_weights_) {
        if(encoder_runtime_)encoder_runtime_->drain();
        encoder_weights_.reset();encoder_weight_identity_.clear();
        encoder_gguf_.reset();
    }
    const char *encoder_retain_flag=std::getenv("TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME");
    require(binary_option_or_unset(encoder_retain_flag),"Qwen encoder runtime retention requires 0 or 1");
    const bool retain_encoder_runtime=option_enabled(encoder_retain_flag) && !r.encoder_ane_manifest.empty();
    if(retain_encoder_runtime)
        require(r.residency=="resident" && r.width==512 && r.height==512 &&
            !r.memory_constrained.enabled && !r.streaming.active() && !r.memory_budget_bytes,
            "Qwen encoder runtime retention requires unconstrained resident 512px execution");
    if(!retain_encoder_runtime) {encoder_runtime_.reset();encoder_runtime_identity_.clear();}
    if (!runtime_requested) { runtime_ffn_.reset(); runtime_manifest_.clear(); }
    if (!qkv_requested) { runtime_qkv_.reset(); qkv_manifest_.clear(); }
    auto upcoming_encoder_growth=[&] {
        uint64_t bytes=0;
        auto charge=[&](uint64_t value) {
            require(value<=UINT64_MAX-bytes,"Qwen encoder retention growth overflow");bytes+=value;
        };
        if(!transformer_.bytes())charge(std::filesystem::file_size(transformer_source_));
        if(!vae_.bytes())charge(std::filesystem::file_size(root_/"vae/qwen_image_2.1_vae_bf16.safetensors"));
        if(!r.loras.empty() && active_lora_identity_.empty())charge(std::filesystem::file_size(r.loras.front().path));
        if(r.execution=="gpu_ane" && !(runtime_requested && runtime_ffn_ && runtime_ffn_->available()))charge(uint64_t(2)<<30);
        return bytes;
    };
    EncoderWeightResidencyMetrics encoder_weight_metrics;
    encoder_weight_metrics.enabled=retain_encoder_weights;
    std::optional<EncoderSourceGeneration> encoder_source;
    if(retain_encoder_weights) {
        encoder_source=encoder_source_generation(encoder_source_);
        if(encoder_weights_ && encoder_weight_identity_!=encoder_source->identity) {
            encoder_runtime_.reset();encoder_runtime_identity_.clear();
            encoder_weights_.reset();encoder_weight_identity_.clear();
            encoder_gguf_.reset();
        }
        if(encoder_weights_) {
            const auto observed=ane::observe_runtime_memory(mx::get_active_memory());
            const auto decision=admit_encoder_weights(observed,encoder_weights_->bytes(),upcoming_encoder_growth());
            if(!decision.allowed()) {
                encoder_weight_metrics.decline_reason=ane::memory_denial_reason(decision.denial,observed);
                encoder_runtime_.reset();encoder_runtime_identity_.clear();
                encoder_weights_.reset();encoder_weight_identity_.clear();
                encoder_gguf_.reset();
            }
        }
    }
    const bool rectangular_w8a8 = option_enabled(std::getenv(
        "TURBOCIDER_QWEN21_RECT_W8A8_DIAGNOSTIC"));
    const bool lora_base_ane = qwen21::lora_base_ane(r);
    const bool gate_up_ane = qwen21::gate_up_ane(r);
    const bool fused_lora_ane = qwen21::fused_lora_ane(r);
    const bool fused_qkv = option_enabled(std::getenv("TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC"));
    const bool lora_fp16 = option_enabled(std::getenv("TURBOCIDER_QWEN21_VIGGLE_LORA_FP16"));
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
    // This opt-in profile already commits to overlapping these resident
    // components. Admit against actual DiT/VAE/adapter payload, not a forecast.
    // Ordinary/request-local source requests retain their original ordering.
    if(retain_encoder_weights)prepare_transformer(r,event,cancelled,fused_lora_ane || runtime_requested,fused_qkv,lora_fp16);
    if(retain_encoder_weights && runtime_requested && ane::private_channel_count(12288)>=0) {
        // Fixed-share construction is independent of text rows. Observe its
        // real arena before retaining the source; do not guess a 2GiB payload.
        // Automatic calibration still waits for its model-supplied workload.
        auto manifest=std::filesystem::canonical(r.ane_manifest);
        const auto identity=runtime_ffn_identity(manifest)+
            (bf16_operand_ranks?":bf16-operands-f32-ranks-v1":"")+(b_epilogue?":bf16-b-fused-epilogue-v1":"")+ffn_phase_identity;
        if(!runtime_ffn_ || !runtime_ffn_->usable_configuration() || runtime_manifest_!=identity ||
            (runtime_ffn_->available() && !r.loras.empty() && !runtime_ffn_->supports_lora_inputs())) {
            runtime_ffn_.reset();
            const size_t budget=runtime_ane_budget(device_info().physical_memory,mx::get_active_memory(),
                encoder_runtime_ ? encoder_runtime_->metrics().runtime_weight_estimated_bytes : 0);
            runtime_ffn_=std::make_unique<ane::HybridFfn>(manifest,4096,12288,budget,cancelled,!r.loras.empty());
            runtime_manifest_=identity;
        }
        runtime_ffn_->set_gpu_layers(prefill_gpu_layers);
    }
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
    // The one-entry edit cache is request-local to this resident Session. A
    // path/mtime-only key would reuse stale visual conditions after an image
    // is overwritten in place; verify the ordered file bytes on each request.
    std::vector<std::string> image_sha256;
    if (r.residency == "resident") {
        image_sha256.reserve(r.inputs.size());
        for (const auto &input : r.inputs) {
            checkpoint(cancelled);
            image_sha256.push_back(sha256_file(input.path));
        }
    } else conditioning_cache_.edit.reset();
    std::string encoder_identity=r.encoder_ane_manifest.empty() ? "gpu" :
        std::filesystem::canonical(r.encoder_ane_manifest).string()+":"+sha256_file(r.encoder_ane_manifest)+
        ane::HybridFfn::executor_configuration_identity()+(retain_encoder_runtime ? ":retained-encoder-executor-v1" : ":request-encoder-executor-v1");
    if(encoder_channels)encoder_identity+=":encoder-channels-"+std::to_string(*encoder_channels);
    if(encoder_source)encoder_identity+=":retained-source-"+encoder_source->identity;
    if(compile_encoder_gpu)encoder_identity+=":compiled-gpu-blocks-v3-canonical-native-gqa";
    if(encoder_runtime_identity_!=encoder_identity) {encoder_runtime_.reset();encoder_runtime_identity_.clear();}
    conditioning_cache_.select_encoder(encoder_identity);
    const bool edit_hit = !r.inputs.empty() && conditioning_cache_.edit_hit(
        r.prompt, r.qwen21_reference_size, image_sha256);
    std::vector<Tensor> images;
    if (!edit_hit) for (const auto &input : r.inputs) {
        checkpoint(cancelled);
        images.push_back(resize_reference(load_rgba_image_tensor(input.path),
                                          r.qwen21_reference_size));
    }
    const bool hit = edit_hit || (r.inputs.empty() && conditioning_cache_.text_hit(r.prompt));
    if(hit && encoder_runtime_) {
        encoder_runtime_->begin_request(); // resident admission also on condition-cache hits
        if(!encoder_runtime_->available()) {encoder_runtime_.reset();encoder_runtime_identity_.clear();}
    }
    if (!hit) clear_prefix_cache();
    Tensor text(0.f);
    std::vector<int> slots;
    std::optional<HybridMetrics> encoder_metrics; // this request only; never replayed by a cache hit
    std::optional<EncoderRuntimeReuseMetrics> encoder_reuse;
    if(!r.encoder_ane_manifest.empty()) {
        encoder_reuse.emplace();encoder_reuse->enabled=retain_encoder_runtime;
    }
    auto text_start = Clock::now();
    if (edit_hit) { text = conditioning_cache_.edit->text; slots = conditioning_cache_.edit->image_slots; }
    else if (hit) text = *conditioning_cache_.text;
    else {
        // Staged requests release the DiT; resident requests keep its packed
        // suffix too, including when encoding a different prompt.
        if (r.residency == "component_staged") {
            hybrid_mlp_.reset();
            clear_prefix_cache();
            fused_qkv_weights_.clear();
            transformer_.clear(); vae_.clear(); mx::clear_cache();
        }
        std::unique_ptr<Weights> request_encoder_weights;
        std::unique_ptr<GgufComponent> request_encoder_gguf;
        encoder_weight_metrics.reused=bool(encoder_weights_);
        if(!encoder_weights_) {
            request_encoder_weights=std::make_unique<Weights>();
            emit(event, "load_qwen21_text", 0, 1);
            if(gguf_encoder)request_encoder_gguf=load_gguf_component(encoder_source_,*request_encoder_weights,true,event,cancelled);
            else request_encoder_weights->load_file(encoder_source_);
            ++encoder_weight_loads_;
            emit(event, "load_qwen21_text", 1, 1);
        }
        Weights &weights=encoder_weights_ ? *encoder_weights_ : *request_encoder_weights;
        encoder_weight_metrics.source_bytes=weights.bytes();
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
        config.compiled_gpu_blocks=compile_encoder_gpu;
        config.final_norm = false; // official checkpoint's pre-final-RMSNorm hidden state
        std::unique_ptr<ane::HybridFfn> request_encoder_runtime;
        ane::HybridFfn *encoder_runtime=nullptr;
        uint64_t encoder_calls_before=0;
        if(!r.encoder_ane_manifest.empty()) {
            const uint64_t physical=device_info().physical_memory;
            const size_t encoder_budget=std::min(uint64_t(1)<<30,
                uint64_t(runtime_ane_budget(physical, mx::get_active_memory(),
                    (runtime_ffn_ ? runtime_ffn_->metrics().runtime_weight_estimated_bytes : 0)+
                    (runtime_qkv_ ? runtime_qkv_->metrics().estimated_bytes : 0))));
            if(encoder_runtime_ && (!encoder_runtime_->usable_configuration() ||
                encoder_runtime_->metrics().runtime_weight_estimated_bytes>encoder_budget)) {
                encoder_runtime_.reset();encoder_runtime_identity_.clear();
            }
            if(retain_encoder_runtime) {
                encoder_reuse->reused=bool(encoder_runtime_);
                if(!encoder_runtime_) {
                    encoder_runtime_=std::make_unique<ane::HybridFfn>(r.encoder_ane_manifest,4096,12288,encoder_budget,cancelled,
                        false,nullptr,std::nullopt,encoder_channels);
                    encoder_runtime_identity_=encoder_identity;
                }
                encoder_runtime=encoder_runtime_.get();
            } else {
                request_encoder_runtime=std::make_unique<ane::HybridFfn>(r.encoder_ane_manifest,4096,12288,encoder_budget,cancelled,
                    false,nullptr,std::nullopt,encoder_channels);
                encoder_runtime=request_encoder_runtime.get();
            }
            encoder_runtime->begin_request();
            encoder_calls_before=encoder_runtime->metrics().runtime_calls;
        }
        ane::HybridFfn::SourceScope encoder_sources(encoder_runtime);
        TextEncoder encoder(weights, config,encoder_runtime);
        text = assembled.retain(encoder.encode_embeddings(assembled.embeddings, assembled.positions,
            assembled.embeddings.shape(1), event, cancelled, assembled.deepstack_deltas));
        mx::eval(text);
        encoder_sources.finish();
        if(encoder_runtime) {
            encoder_metrics=encoder_runtime->metrics();
            encoder_metrics->block_count=config.layers;
            encoder_reuse->calls_this_request=encoder_metrics->runtime_calls-encoder_calls_before;
            // Do not retain a failed/memory-declined executor or its scratch.
            if(encoder_runtime_ && !encoder_runtime_->available()) {
                encoder_runtime_.reset();encoder_runtime_identity_.clear();
            }
            encoder_metrics->session_released_after_encoding=!encoder_runtime_;
        }
        if(retain_encoder_weights) {
            auto observed=ane::observe_runtime_memory(mx::get_active_memory());
            auto decision=admit_encoder_weights(observed,weights.bytes(),upcoming_encoder_growth());
            if(decision.allowed()) {
                weights.materialize(); // seal lazy arrays before keeping their owners
                observed=ane::observe_runtime_memory(mx::get_active_memory());
                decision=admit_encoder_weights(observed,weights.bytes(),upcoming_encoder_growth());
            }
            require(encoder_source_generation(encoder_source_).identity==encoder_source->identity,
                "Qwen encoder source changed during retained encoding");
            if(decision.allowed()) {
                if(!encoder_weights_) {
                    encoder_weights_=std::move(request_encoder_weights);
                    encoder_gguf_=std::move(request_encoder_gguf);
                }
                encoder_weight_identity_=encoder_source->identity;
                encoder_weight_metrics.decline_reason.clear();
            } else {
                encoder_weight_metrics.decline_reason=ane::memory_denial_reason(decision.denial,observed);
                if(encoder_weights_) {
                    request_encoder_weights=std::move(encoder_weights_);
                    request_encoder_gguf=std::move(encoder_gguf_);
                }
                encoder_weight_identity_.clear();
            }
        }
        if(request_encoder_gguf)request_encoder_gguf->bank->check_unchanged();
        if(encoder_gguf_)encoder_gguf_->bank->check_unchanged();
        slots = assembled.image_slots;
        if (r.inputs.empty()) { conditioning_cache_.text = text; conditioning_cache_.prompt = r.prompt; }
    }
    mx::clear_cache();
    double text_seconds = seconds(text_start);
    dump("qwen21_text", text);
    std::vector<ReferenceLatents> references;
    auto image_start = Clock::now();
    if (edit_hit) {
        references = conditioning_cache_.edit->reference_latents;
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
    if (!edit_hit && !r.inputs.empty() && r.residency == "resident") {
        // Do not publish a cache entry if a reference changed during the
        // decode/encode pass. Hashing again also catches same-path overwrites
        // whose mtime or file size were preserved.
        for (size_t i = 0; i < r.inputs.size(); ++i) {
            checkpoint(cancelled);
            require(sha256_file(r.inputs[i].path) == image_sha256[i],
                    "Qwen21 reference changed while encoding conditioning");
        }
        conditioning_cache_.edit = ConditioningCache::Edit{r.prompt, r.qwen21_reference_size,
                                          std::move(image_sha256), text, slots, references};
    }
    images.clear(); mx::clear_cache();
    double image_seconds = seconds(image_start);
    // Original request-local path still releases its encoder before loading
    // and binding the separate low-rank adapter. No BF16/disk merge is added.
    if(!retain_encoder_weights)prepare_transformer(r,event,cancelled,fused_lora_ane || runtime_requested,fused_qkv,lora_fp16);
    // Transactional bank load replaces the complete Weights dictionary, so
    // snapshot the requested math recipe AFTER loading, including cold runs.
    if(transformer_.qwen_affine_shared_down()!=shared_down) {
        if(runtime_ffn_)runtime_ffn_->drain();
        clear_prefix_cache();mx::synchronize();transformer_.set_qwen_affine_shared_down(shared_down);
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
        const std::string identity=runtime_ffn_identity(manifest)+
            (bf16_operand_ranks?":bf16-operands-f32-ranks-v1":"")+(b_epilogue?":bf16-b-fused-epilogue-v1":"")+ffn_phase_identity;
        const bool native_channel_auto = ane::private_channel_count(12288) < 0;
        const std::string request_identity = identity + (native_channel_auto ?
            ":rows="+std::to_string((r.height/16)*(r.width/16))+":prefix="+std::to_string(text.shape(1))+
            ":adapter="+active_lora_identity_ : "");
        if (!runtime_ffn_ || !runtime_ffn_->usable_configuration() || runtime_manifest_ != request_identity ||
            (runtime_ffn_->available() && !r.loras.empty() && !runtime_ffn_->supports_lora_inputs())) {
            runtime_ffn_.reset();
            const size_t budget = runtime_ane_budget(device_info().physical_memory,
                                                     mx::get_active_memory(),
                encoder_runtime_ ? encoder_runtime_->metrics().runtime_weight_estimated_bytes : 0);
            std::optional<ane::HybridFfn::CalibrationWorkload> calibration;
            if(native_channel_auto) {
                emit(event,"calibrate_native_ane_channels",0,1);
                calibration.emplace();
                calibration->model_sha256=sha256_file(root_/"diffusion_models/qwen_image_2.1_bf16.safetensors");
                calibration->adapter_identity=r.loras.empty()?std::string{}:active_lora_identity_;
                calibration->encoding="dense-bf16-fused-gate-up";
                calibration->rows=(r.height/16)*(r.width/16);
                for(int ordinal:{0,7,15,23,31})for(const auto *suffix:{"gate_up.weight","out.weight"}) {
                    const auto &value=transformer_.at("transformer_blocks."+std::to_string(ordinal)+".img_mlp."+suffix);
                    calibration->source_generation += ":"+std::to_string(value.id());
                    calibration->source_owners.push_back(value.data_shared_ptr());
                }
                calibration->source_generation += ":prefix="+std::to_string(text.shape(1));
                const char *rank_dtype=std::getenv("TURBOCIDER_QWEN21_VIGGLE_LORA_FP16");
                calibration->gpu_configuration=std::string("qwen-lora-rank=")+(rank_dtype?rank_dtype:"<unset>")+
                    (transformer_.has_runtime_loras()?";shared-compiled-qwen-ffn-v1":";base-channel-eager-v1")+
                    (bf16_operand_ranks?";bf16-operands-f32-ranks-v1":"")+(b_epilogue?";bf16-b-fused-epilogue-v1":"");
                calibration->weights=[&](int ordinal) {
                    const auto prefix="transformer_blocks."+std::to_string(ordinal)+".img_mlp.";
                    auto gu=mx::split(transformer_.at(prefix+"gate_up.weight"),2,0);mx::eval(gu);
                    return std::vector<ane::FfnWeight>{{gu[0],std::nullopt,std::nullopt},{gu[1],std::nullopt,std::nullopt},
                        {transformer_.at(prefix+"out.weight"),std::nullopt,std::nullopt}};
                };
                using CalFunction=runtime_ffn::Function;
                auto lora_gpu=std::make_shared<std::vector<CalFunction>>();
                auto lora_down=std::make_shared<std::vector<CalFunction>>();
                using RangeKey=std::tuple<int,int,int>;
                auto lora_corrections=std::make_shared<std::map<RangeKey,CalFunction>>();
                auto lora_channels=std::make_shared<std::map<RangeKey,CalFunction>>();
                if(transformer_.has_runtime_loras())for(int ordinal=0;ordinal<32;++ordinal) {
                    const auto prefix="transformer_blocks."+std::to_string(ordinal)+".img_mlp.";
                    lora_gpu->push_back(runtime_ffn::full(transformer_,prefix));
                    lora_down->push_back(runtime_ffn::down_add(transformer_,prefix));
                }
                if(transformer_.has_runtime_loras())calibration->adapter=[this,lora_corrections,lora_down](int ordinal) {
                    const auto prefix="transformer_blocks."+std::to_string(ordinal)+".img_mlp.";
                    auto delta=[this,ordinal,prefix,lora_corrections](const Tensor &x,int first,int count) {
                        const RangeKey key{ordinal,first,count};
                        auto at=lora_corrections->find(key);
                        if(at==lora_corrections->end())at=lora_corrections->emplace(key,
                            runtime_ffn::corrections(transformer_,prefix,first,count)).first;
                        auto result=at->second({x});return std::make_pair(result[0],result[1]);
                    };
                    return ane::HybridFfn::Adapter{
                        [delta](const Tensor &x) {return delta(x,0,12288);},
                        [ordinal,lora_down](const Tensor &hidden,const Tensor &base) {
                            return lora_down->at(ordinal)({hidden,base})[0];
                        },
                        delta};
                };
                calibration->gpu=[&,lora_gpu](int ordinal,const Tensor &input) {
                    if(!lora_gpu->empty())return lora_gpu->at(ordinal)({input})[0];
                    static auto compiled=mx::compile([](const std::vector<Tensor>&a) {
                        auto gu=mx::split(mx::matmul(a[0],mx::transpose(a[1])),2,-1);
                        return std::vector<Tensor>{mx::matmul(silu(gu[0])*gu[1],mx::transpose(a[2]))};
                    });
                    const auto prefix="transformer_blocks."+std::to_string(ordinal)+".img_mlp.";
                    return compiled({input,transformer_.at(prefix+"gate_up.weight"),transformer_.at(prefix+"out.weight")})[0];
                };
                calibration->channel_gpu=[this,lora_channels](int ordinal,const Tensor &input,int first,int count) {
                    const auto prefix="transformer_blocks."+std::to_string(ordinal)+".img_mlp.";
                    if(transformer_.has_runtime_loras()) {
                        const RangeKey key{ordinal,first,count};
                        auto at=lora_channels->find(key);
                        if(at==lora_channels->end())at=lora_channels->emplace(key,
                            runtime_ffn::channels(transformer_,prefix,first,count,4096,12288,ane::configured_fp32_channel_join())).first;
                        auto result=at->second({input});return std::make_pair(result[0],result[1]);
                    }
                    auto gate=transformer_.project_slice(input,prefix+"gate_up",first,first+count,0,4096,false);
                    auto up=transformer_.project_slice(input,prefix+"gate_up",12288+first,12288+first+count,0,4096,false);
                    auto hidden=silu(gate)*up;
                    auto base=ane::configured_fp32_channel_join() ? transformer_.project_base_slice_fp32(hidden,prefix+"out",0,4096,first,first+count) :
                        transformer_.project_base_slice(hidden,prefix+"out",0,4096,first,first+count,false);
                    return std::make_pair(base,hidden);
                };
            }
            runtime_ffn_ = std::make_unique<ane::HybridFfn>(manifest, 4096, 12288, budget, cancelled,
                                                         !r.loras.empty(),calibration?&*calibration:nullptr);
            runtime_manifest_ = request_identity;
            if(native_channel_auto)emit(event,"calibrate_native_ane_channels",1,1);
        }
        runtime_ffn_->begin_request(active_lora_identity_);
    }
    if (runtime_requested)runtime_ffn_->set_gpu_layers(prefill_gpu_layers);
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
                                                     mx::get_active_memory(),
                encoder_runtime_ ? encoder_runtime_->metrics().runtime_weight_estimated_bytes : 0);
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
    if (!r.loras.empty())
        result.selection += fused_lora_ane
            ? "; experimental runtime LoRA with Viggle six-step schedule; adapter quality unqualified"
            : "; Viggle v0.2.1 r256 runtime LoRA; six-step student schedule";
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
    result.encoder_hybrid=std::move(encoder_metrics);
    encoder_weight_metrics.retained=bool(encoder_weights_);
    if(encoder_weights_ && !encoder_weight_metrics.source_bytes)encoder_weight_metrics.source_bytes=encoder_weights_->bytes();
    encoder_weight_metrics.retained_bytes=encoder_weights_ ? encoder_weights_->bytes() : 0;
    encoder_weight_metrics.loads_session_total=encoder_weight_loads_;
    result.encoder_weight_residency=encoder_weight_metrics;
    if(encoder_reuse) {
        encoder_reuse->retained=bool(encoder_runtime_);
        encoder_reuse->retained_estimated_bytes=encoder_runtime_ ? encoder_runtime_->metrics().runtime_weight_estimated_bytes : 0;
        result.encoder_runtime_reuse=encoder_reuse;
    }
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
        result.backend = runtime_ffn_->backend_label();
        result.precision = runtime_ffn_->precision_label();
        result.selection = runtime_ffn_->selection_label();
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
    if (!r.encoder_ane_manifest.empty())
        result.selection += hit ? "; encoder conditioning cache hit (no new ANE call)" :
            result.encoder_runtime_reuse && result.encoder_runtime_reuse->calls_this_request ?
            "; explicit shared runtime Qwen3-VL language FFN; vision/attention remain GPU; physical overlap unverified" :
            "; explicit encoder runtime attempted; no model ANE call; complete GPU language FFN";
    if(retain_encoder_runtime)result.selection+="; explicit bounded encoder executor retention";
    if(retain_encoder_weights)result.selection+="; explicit admitted encoder source retention; no weight copy/precision change";
    if(compile_encoder_gpu)result.selection+="; experimental compiled Qwen3-VL GPU blocks with dynamic original source arrays, FP32 norm and unchanged mRoPE/DeepStack order; original-dtype native GQA attention";
    if(share_lora_ranks)result.selection+="; experimental operation-local shared gate/up LoRA input ranks";
    result.student_ffn_reuse_enabled=student_ffn_reuse;
    result.student_ffn_requested_layers=student_ffn_reuse?student_reuse_layers:0;
    if(student_ffn_reuse)result.selection+="; experimental six-step student final FFN reuse, includes full LoRA FFN output; layers="+std::to_string(student_reuse_layers);
    if(joint_ab)result.selection+="; experimental joint BF16 LoRA A/B operands, F32 ranks and fused B epilogue; all eligible adapter projections, not FFN-only";
    else {
        if(bf16_operand_ranks)result.selection+="; experimental original BF16 LoRA A operands with FP32 ranks and B/delta arithmetic";
        if(b_epilogue)result.selection+="; experimental BF16 LoRA B operands with fused F32 scale/base epilogue, original F32 A ranks";
    }
    if(runtime_requested && ffn_phase!=RuntimeFfnPhase::All)
        result.selection+=std::string("; experimental runtime FFN phase=")+runtime_ffn_phase_name(ffn_phase)+"; other phase uses complete unsplit GPU blocks";
    if(runtime_requested && !prefill_gpu_layers.empty())
        result.selection+="; experimental runtime prefill complete-GPU FFN blocks="+prefill_gpu_layer_list(prefill_gpu_layers);
    if(runtime_requested || profile_steps) {
        result.qwen_ffn_phases.emplace();
        result.qwen_ffn_phases->policy=runtime_requested ? runtime_ffn_phase_name(ffn_phase) : "gpu";
    }
    if(split_down_ranks)result.selection+="; experimental FP32 split down-LoRA input ranks, ONE joined B/delta rounding";
    if (runtime_requested && !r.loras.empty())
        result.shared_lora_ranks = SharedLoraRankMetrics{share_lora_ranks};
    if (lora_fp16 && !r.loras.empty())
        result.selection += "; experimental FP16 low-rank LoRA matmuls";
    if (lora_1024_generation(r))
        result.selection += "; experimental 1024px six-step runtime LoRA generation, FP32 rank; quality unqualified";
    if (norm_rope && std::string_view(norm_rope) == "1")
        result.selection += "; experimental fused Metal Q/K norm-RoPE";
    result.timings.hybrid = (hybrid_requested || runtime_requested || qkv_requested) ? seconds(hybrid_start) : 0;
    result.checkpoint = transformer_source_.filename().string();
    if(gguf_transformer || gguf_encoder) {
        result.backend="mlx_cpp_metal_qwen21_gguf";
        result.precision=std::string(gguf_transformer ? "q4_k_m_dit" : "bf16_dit")+"+"+
            (gguf_encoder ? "q4_k_m_text" : "bf16_text")+"+bf16_vae";
        result.selection+="; explicit mixed K affine Q4/Q8 GPU, FP16 typed coefficients/I/O; Q6_K group requantization; dense embedding only";
        if(shared_down)result.selection+="; experimental typed shared-word MPP FFN down kernel";
    }
    result.text_tokens = result.valid_text_tokens = text.shape(1);
    for (const auto &ref : references) result.reference_tokens += ref.latents.shape(1);
    result.total_tokens = result.text_tokens + result.reference_tokens + r.height / 16 * (r.width / 16);
    result.timings.text = text_seconds; result.timings.image = image_seconds;
    emit(event, (hybrid_requested || runtime_requested || qkv_requested) ? "route_gpu_ane" : "route_gpu", 1, 1);
    const uint64_t coreml_calls_before = hybrid_requested ? hybrid_->metrics().runtime_calls : 0;
    if (!prepare_only) {
        auto schedule = r.loras.empty() ? sigmas(r.width, r.height, r.steps) :
                                      viggle_v021_sigmas(r.width, r.height);
        mx::eval(schedule);
        auto latents = mx::astype(mx::random::normal({1, r.height / 16 * (r.width / 16), 64},
            mx::float32, mx::random::key(r.seed)), gguf_transformer ? mx::float16 : mx::bfloat16);
        text=mx::astype(text,latents.dtype());mx::eval(text);
        if (!r.noise_path.empty()) {
            auto [values, metadata] = mx::load_safetensors(r.noise_path);
            require(values.count("tensor") || values.count("initial"), "Qwen21 noise file requires tensor or initial");
            const auto &noise = values.at(values.count("tensor") ? "tensor" : "initial");
            require(noise.shape() == latents.shape(), "Qwen21 noise tensor shape mismatch");
            latents = mx::astype(noise, latents.dtype());
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
            active_lora_identity_ + ":" + (lora_fp16 ? "fp16" : "fp32") +
            (bf16_operand_ranks ? ":bf16-operands-f32-ranks-v1:" : ":") +
            (b_epilogue ? "bf16-b-fused-epilogue-v1:" : "") +
            (student_ffn_reuse ? "student-final-ffn-reuse-v1-layers="+std::to_string(student_reuse_layers)+":" : "") +
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
            std::vector<std::vector<ane::FfnWeight>> runtime_packed_sources;
            using RuntimeFunction = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;
            std::vector<RuntimeFunction> runtime_packed_gpu;
            RuntimeFunction gpu_qkv;
            std::vector<RuntimeFunction> runtime_lora_gpu, runtime_lora_gate_up, runtime_lora_down_add;
            std::vector<RuntimeFunction> runtime_lora_gate_up_channels;
            std::vector<RuntimeFunction> runtime_lora_channel_gpu;
            std::vector<RuntimeFunction> runtime_lora_input_ranks;
            std::vector<RuntimeFunction> runtime_lora_down_gpu_ranks,runtime_lora_down_split_add;
            ane::HybridFfn::SourceScope runtime_sources_scope(runtime_requested ? runtime_ffn_.get() : nullptr);
            if (runtime_requested) {
                for (int block = 0; block < 32; ++block) {
                    const auto p = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
                    if(gguf_transformer) {
                        runtime_packed_sources.push_back(gguf_ffn_sources(transformer_,p));
                        runtime_packed_gpu.push_back(runtime_ffn::full(transformer_,p));
                    } else {
                        auto gu = mx::split(transformer_.at(p + "gate_up.weight"), 2, 0);
                        runtime_weights.push_back({gu[0], gu[1], transformer_.at(p + "out.weight")});
                        mx::eval(runtime_weights.back());
                    }
                    if (transformer_.has_runtime_loras()) {
                        // The ordinary GPU Transformer compiles this same
                        // projection/low-rank arithmetic inside its blocks.
                        // Preserve fusion across the new FFN boundary too.
                        // Closures are request-local: never reuse captures
                        // after a different adapter has rebound the weights.
                        runtime_lora_gpu.push_back(runtime_ffn::full(transformer_,p));
                        runtime_lora_gate_up.push_back(runtime_ffn::corrections(transformer_,p,0,12288));
                        if (runtime_ffn_->channel_split()) {
                            const int first=runtime_ffn_->gpu_channels(),count=runtime_ffn_->ane_channels();
                            const bool block_shared=share_lora_ranks && transformer_.lora_rank_count(p+"gate_up")>0;
                            runtime_lora_gate_up_channels.push_back(block_shared ?
                                runtime_ffn::corrections_shared_ranks(transformer_,p,first,count) : runtime_ffn::corrections(transformer_,p,first,count));
                            // Same checkpoint-only down and per-projection
                            // FP32 rank/BF16 rounding as the existing callback.
                            // Request-local captures cannot outlive/reuse a
                            // differently rebound adapter or channel share.
                            runtime_lora_channel_gpu.push_back(runtime_ffn::channels(transformer_,p,0,first,4096,12288,runtime_ffn_->fp32_channel_join(),block_shared));
                            runtime_lora_input_ranks.push_back(block_shared ? runtime_ffn::input_ranks(transformer_,p) : RuntimeFunction{});
                            const bool block_down=split_down_ranks && transformer_.lora_rank_count(p+"out")>0;
                            runtime_lora_down_gpu_ranks.push_back(block_down ? runtime_ffn::down_gpu_ranks(transformer_,p,first) : RuntimeFunction{});
                            runtime_lora_down_split_add.push_back(block_down ? runtime_ffn::down_add_split_ranks(transformer_,p,first) : RuntimeFunction{});
                        }
                        runtime_lora_down_add.push_back(runtime_ffn::down_add(transformer_,p));
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
                    if(gguf_transformer)runtime_ffn_->stage_weights(block,rows,runtime_packed_sources.at(block));
                    else runtime_ffn_->stage(block, rows, runtime_weights.at(block));
                });
                auto run_ffn = [&](int block, const Tensor &input) {
                    const auto p = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
                    std::vector<Tensor> shared_inputs{input};
                    const bool block_shared=share_lora_ranks && runtime_ffn_->channel_split() && bool(runtime_lora_input_ranks.at(block));
                    if(block_shared) {
                        const auto ranks=runtime_lora_input_ranks.at(block)({input});
                        shared_inputs.insert(shared_inputs.end(),ranks.begin(),ranks.end());
                        ++result.shared_lora_ranks->prepared_sets;
                    }
                    bool shared_correction = false, shared_gpu = false, used_fallback = false;
                    ane::HybridFfn::Adapter adapter{
                        [&](const Tensor &x) {
                            auto gu = runtime_lora_gate_up.at(block)({x});
                            return std::make_pair(gu[0], gu[1]);
                        },
                        [&](const Tensor &h, const Tensor &base) {
                            return runtime_lora_down_add.at(block)({h, base})[0];
                        },
                        [&](const Tensor &x, int first, int count) {
                            require(first==runtime_ffn_->gpu_channels() && count==runtime_ffn_->ane_channels(),
                                    "Qwen channel LoRA correction range changed within request");
                            auto gu=runtime_lora_gate_up_channels.at(block)(block_shared ? shared_inputs : std::vector<Tensor>{x});
                            shared_correction = block_shared;
                            return std::make_pair(gu[0],gu[1]);
                        }};
                    if(split_down_ranks && runtime_ffn_->channel_split() && bool(runtime_lora_down_gpu_ranks.at(block))) {
                        adapter.channel_down_ranks=ane::HybridFfn::Adapter::ChannelDownRanks{
                            uint64_t(transformer_.lora_rank_width(p+"out"))*3*sizeof(float),
                            [&](const Tensor &h){return runtime_lora_down_gpu_ranks.at(block)({h});},
                            [&](const Tensor &h,const std::vector<Tensor> &ranks,const Tensor &base) {
                                std::vector<Tensor> args{h,base};args.insert(args.end(),ranks.begin(),ranks.end());
                                return runtime_lora_down_split_add.at(block)(args)[0];
                            }};
                    }
                    auto output = runtime_ffn_->run(block, input, [&](const Tensor &x) {
                        used_fallback = true;
                        if(gguf_transformer)return runtime_packed_gpu.at(block)({x})[0];
                        if (transformer_.has_runtime_loras()) return runtime_lora_gpu.at(block)({x})[0];
                        return runtime_gpu({x, transformer_.at(p + "gate_up.weight"),
                                              transformer_.at(p + "out.weight")})[0];
                    }, cancelled, transformer_.has_runtime_loras() ? &adapter : nullptr,
                    [&](const Tensor &x,int first,int count) {
                        if (transformer_.has_runtime_loras()) {
                            require(first==0 && count==runtime_ffn_->gpu_channels(),
                                    "Qwen GPU channel LoRA range changed within request");
                            auto result=runtime_lora_channel_gpu.at(block)(block_shared ? shared_inputs : std::vector<Tensor>{x});
                            shared_gpu = block_shared;
                            return std::make_pair(result[0],result[1]);
                        }
                        auto g=transformer_.project_slice(x,p+"gate_up",first,first+count,0,4096,false);
                        auto u=transformer_.project_slice(x,p+"gate_up",12288+first,12288+first+count,0,4096,false);
                        auto hidden=silu(g)*u;
                        auto base=runtime_ffn_->fp32_channel_join() ? transformer_.project_base_slice_fp32(hidden,p+"out",0,4096,first,first+count) :
                            transformer_.project_base_slice(hidden,p+"out",0,4096,first,first+count,false);
                        return std::make_pair(base,hidden);
                    },[&](int next) {
                        if(gguf_transformer)return next<32 ? runtime_packed_sources.at(next) : std::vector<ane::FfnWeight>{};
                        std::vector<ane::FfnWeight> sources;
                        if(next<32)for(const auto &weight:runtime_weights.at(next))sources.push_back({weight,std::nullopt,std::nullopt});
                        return sources;
                    });
                    if (shared_correction && shared_gpu && !used_fallback) {
                        ++result.shared_lora_ranks->completed_hybrid_blocks;
                        result.shared_lora_ranks->completed_adapter_rank_arrays += shared_inputs.size()-1;
                    }
                    return output;
                };
                // Select by the Transformer's actual prefix-reuse state, not
                // by a denoise step index. Clearing a callback preserves the
                // original lazy/compiled GPU block: no split or timing fence.
                dit.set_prefill_mlp(runtime_ffn_phase_runs(ffn_phase,false) ? Transformer::DecodeMLP(run_ffn) : Transformer::DecodeMLP{});
                dit.set_decode_mlp(runtime_ffn_phase_runs(ffn_phase,true) ? Transformer::DecodeMLP(run_ffn) : Transformer::DecodeMLP{});
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
                const bool prefix_reused=dit.prefix_matches(text,r.height/16,r.width/16,references);
                const auto phase_before=runtime_requested ? runtime_ffn_->metrics() : HybridMetrics{};
                const auto step_started = result.qwen_ffn_phases || profile_steps ? Clock::now() : Clock::time_point{};
                const auto prediction_before = profile_steps && hybrid_requested
                    ? hybrid_->metrics().prediction_seconds : 0.;
                if(student_ffn_reuse && step==r.steps-2) {
                    const uint64_t extra=uint64_t(32)*uint64_t(r.height/16)*uint64_t(r.width/16)*4096*2;
                    const uint64_t reserve=uint64_t(8)<<30,physical=device_info().physical_memory;
                    const auto active=mx::get_active_memory();
                    require(physical>reserve && extra<=physical-reserve && active<=physical-reserve-extra,
                        "student FFN cache lacks admitted allocator headroom plus8GiB reserve");
                }
                if (reuse_final_ffn || hybrid_reuse_ffn || hybrid_reuse_last16 || student_ffn_reuse)
                    dit.set_ffn_cache_mode(half_reuse_ffn && step == r.steps - 3 ? Transformer::FFNCacheMode::Capture :
                                           half_reuse_ffn && step == r.steps - 2 ? Transformer::FFNCacheMode::ReuseEvenAndCapture :
                                           !half_reuse_ffn && step == r.steps - 2 ? Transformer::FFNCacheMode::Capture :
                                           step == r.steps - 1 ? ((hybrid_reuse_last16 || (student_ffn_reuse && student_reuse_layers==16)) ? Transformer::FFNCacheMode::ReuseLast16 :
                                                                  Transformer::FFNCacheMode::Reuse) :
                                           Transformer::FFNCacheMode::Off);
                auto noise = dit.forward(latents, text, schedule.data<float>()[step], r.height / 16, r.width / 16,
                                         true, nullptr, references);
                latents = latents + noise * Tensor(schedule.data<float>()[step+1] - schedule.data<float>()[step], latents.dtype());
                mx::eval(latents);
                require(mx::all(mx::isfinite(latents)).item<bool>(), "nonfinite Qwen21 latent");
                const double step_seconds=result.qwen_ffn_phases || profile_steps ? seconds(step_started) : 0.;
                if(result.qwen_ffn_phases) {
                    const auto after=runtime_requested ? runtime_ffn_->metrics() : HybridMetrics{};
                    require(after.runtime_calls>=phase_before.runtime_calls &&
                        after.runtime_weight_channel_blocks>=phase_before.runtime_weight_channel_blocks,
                        "Qwen FFN phase counters regressed during a completed step");
                    auto &phase=prefix_reused ? result.qwen_ffn_phases->decode : result.qwen_ffn_phases->prefill;
                    ++phase.steps;phase.rows=prefix_reused ? uint64_t(r.height/16)*(r.width/16) : uint64_t(result.total_tokens);
                    phase.step_seconds+=step_seconds;
                    phase.runtime_calls+=after.runtime_calls-phase_before.runtime_calls;
                    phase.completed_channel_blocks+=after.runtime_weight_channel_blocks-phase_before.runtime_weight_channel_blocks;
                }
                if(student_ffn_reuse) {
                    const auto bytes=dit.ffn_cache_logical_bytes();
                    require(bytes<=(uint64_t(256)<<20),"student FFN cache exceeded logical256MiB bound");
                    result.student_ffn_peak_logical_bytes=std::max(result.student_ffn_peak_logical_bytes,bytes);
                    result.student_ffn_captured_blocks+=dit.last_ffn_captured_blocks();
                    result.student_ffn_reused_blocks+=dit.last_ffn_reused_blocks();
                }
                if (profile_steps) {
                    const auto prediction = hybrid_requested
                        ? hybrid_->metrics().prediction_seconds - prediction_before : 0.;
                    std::cerr << "{\"qwen21_step\":" << step
                              << ",\"phase\":\"" << (prefix_reused ? "decode" : "prefill")
                              << "\",\"seconds\":" << step_seconds
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
                runtime_sources_scope.finish();
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
        if(student_ffn_reuse) {
            require(result.student_ffn_captured_blocks==32 && result.student_ffn_reused_blocks==student_reuse_layers,
                "student FFN reuse did not complete one captured and one reused decode step");
            dit.clear_step_cache(); // request-owned activation bank released before VAE
        }
        result.db_cache_steps = dit.db_cached_steps();
        result.timings.denoise = seconds(dit_start); result.actual_steps = r.steps;
        if(transformer_gguf_)transformer_gguf_->bank->check_unchanged();
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
        if (r.residency == "component_staged") {
            hybrid_mlp_.reset(); clear_prefix_cache();
            fused_qkv_weights_.clear(); transformer_.clear(); mx::clear_cache();
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
        if(gguf_transformer) {
            result.backend=runtime_ffn_->backend_label(true);
            result.precision="q4_k_m_dit+"+std::string(gguf_encoder ? "q4_k_m_text" : "bf16_text")+"+bf16_vae+runtime_w8a8_ffn_fp16_io";
            result.selection+="; mixed-K affine source GPU/ANE intermediate-channel FFN with complete packed GPU fallback";
        }
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
        transformer_.clear(); vae_.clear(); mx::clear_cache();
    }
    result.timings.wall = seconds(start);
    result.active_bytes = mx::get_active_memory(); result.peak_bytes = mx::get_peak_memory();
    return result;
} catch (...) {
    encoder_runtime_.reset();encoder_runtime_identity_.clear();
    encoder_weights_.reset();encoder_weight_identity_.clear();
    encoder_gguf_.reset();
    if (runtime_ffn_) runtime_ffn_->drain();
    if (runtime_qkv_) runtime_qkv_->drain();
    try { mx::synchronize(); } catch (...) {}
    clear_prefix_cache();
    fused_qkv_weights_.clear();
    hybrid_mlp_.reset(); hybrid_.reset(); hybrid_manifest_.clear(); hybrid_runtime_options_.clear();
    if (transformer_.has("transformer_blocks.0.attn.qkv_packed.weight"))
        transformer_.clear(); // A cancelled pack may have replaced only some layers.
    throw;
}
} // namespace tc::qwen21
