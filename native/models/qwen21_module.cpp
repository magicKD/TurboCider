#include "qwen21/pipeline.hpp"
#include "qwen21/diagnostic_options.hpp"
#include "qwen21/viggle_adapter.hpp"
#include <cmath>
#include <cstdlib>
#include <string_view>

namespace tc {
ModelModule qwen21_module() {
    return {"qwen-image-2.1",
        [] { return Recipe{"qwen-image-2.1", {{"text_encode", {}}, {"denoise", {"text_encode"}, 40},
                                            {"vae_decode", {"denoise"}}, {"export", {"vae_decode"}}}, true}; },
        [](const Request &r) {
            require(r.operation == "image.generate" || r.operation == "image.edit", "unsupported Qwen21 operation");
            require(r.frames == 1 && !r.audio, "Qwen21 produces one RGBA image without audio");
            require(r.width % 32 == 0 && r.height % 32 == 0 && int64_t(r.width) * r.height <= 8388608,
                    "Qwen21 dimensions must be multiples of 32 within 8 megapixels");
            require(r.model_variant == "auto" || r.model_variant == "qwen-image-2.1", "incorrect Qwen21 variant");
            require(r.encoder_ane_manifest.empty(), "Qwen21 encoder ANE is not implemented");
            require(r.qwen21_reference_size == 1024 ||
                        ((r.qwen21_reference_size == 256 || r.qwen21_reference_size == 512) &&
                         r.allow_approximation &&
                         r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3),
                    "Qwen21 256/512px reference resize requires explicit approximation and 1...3 edit images");
            require(!r.qwen21_gpu_w8a16 || r.qwen21_w8a8,
                    "Qwen21 W8A16 GPU suffix requires explicit W8A8 Core ML opt-in");
            require(!r.qwen21_w8a8 || r.steps >= 2,
                    "Qwen21 W8A8 needs at least one cached decode step");
            const char *rect_flag = std::getenv("TURBOCIDER_QWEN21_RECT_W8A8_DIAGNOSTIC");
            require(qwen21::binary_option_or_unset(rect_flag),
                    "Qwen21 rectangular W8A8 diagnostic accepts only 0 or 1");
            if (qwen21::option_enabled(rect_flag))
                require(r.allow_approximation && r.execution == "gpu_ane" && r.qwen21_w8a8 &&
                            !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty() &&
                            ((r.width == 768 && r.height == 512) ||
                             (r.width == 512 && r.height == 768)) &&
                            r.steps >= 20 && r.steps <= 40 && r.loras.empty() &&
                            (r.operation == "image.generate" ||
                             (r.operation == "image.edit" && !r.inputs.empty() &&
                              r.inputs.size() <= 3 && r.qwen21_reference_size == 1024)) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN")),
                        "Qwen21 rectangular W8A8 needs explicit base 768x512/512x768 GPU/ANE with 20...40 steps, 0...3 full-size references and no other cache");
            const char *lora_ane_flag = std::getenv("TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC");
            const char *gate_up_flag = std::getenv("TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC");
            const char *lora_ref512_flag = std::getenv("TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC");
            const bool runtime_ane = r.hybrid_mlp_mode == "runtime";
            const bool runtime_qkv = r.hybrid_mlp_mode == "runtime_qkv";
            require(r.hybrid_mlp_mode == "auto" || r.hybrid_mlp_mode == "base_fused" ||
                        r.hybrid_mlp_mode == "lora_suffix" || r.hybrid_mlp_mode == "lora_gate_up" ||
                        r.hybrid_mlp_mode == "lora_fused" || runtime_ane || runtime_qkv,
                    "unsupported Qwen21 hybrid_mlp_mode");
            if (runtime_qkv)
                require(r.execution == "gpu_ane" && r.allow_approximation && !r.ane_manifest.empty() &&
                            r.operation == "image.generate" && r.inputs.empty() &&
                            r.width == 1024 && r.height == 1024 &&
                            r.residency == "resident" && r.loras.empty() && !r.qwen21_w8a8 &&
                            !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty() &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC")) &&
                            qwen21::tiled_prefill_layer_count(
                                std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC") ?
                                std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC") : "0") == 0,
                        "Qwen21 runtime QKV needs explicit 1024px base text-to-image resident GPU/ANE MatMul, exact GPU FFN, no fused QKV or caches");
            if (runtime_ane)
                require(r.execution == "gpu_ane" && r.allow_approximation && !r.ane_manifest.empty() &&
                            !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
                            (r.loras.empty() || r.lora_strategy == "inference_time") &&
                            r.qwen21_gpu_full_ffn_blocks.empty() && r.residency == "resident" &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV")),
                        "Qwen21 runtime-weight FFN requires explicit resident GPU/ANE FP16, unmerged runtime LoRA and no frozen W8A8 or DBCache/prefix reuse");
            if (r.hybrid_mlp_mode != "auto" && !runtime_ane && !runtime_qkv)
                require(r.execution == "gpu_ane" && r.qwen21_w8a8 &&
                            r.allow_approximation && !r.ane_manifest.empty() &&
                            (r.hybrid_mlp_mode == "base_fused" ? r.loras.empty() :
                             r.hybrid_mlp_mode == "lora_fused" || !r.loras.empty()),
                        "explicit Qwen21 hybrid_mlp_mode requires a W8A8 manifest and matching base/LoRA request");
            const bool lora_base_ane = qwen21::lora_base_ane(r);
            const bool gate_up_ane = qwen21::gate_up_ane(r);
            const bool fused_lora_ane = qwen21::fused_lora_ane(r);
            require(qwen21::binary_option_or_unset(gate_up_flag),
                    "Qwen21 LoRA-compatible gate/up diagnostic accepts only 0 or 1");
            if (gate_up_ane || fused_lora_ane)
                require(r.execution == "gpu_ane" && r.qwen21_w8a8 &&
                            r.allow_approximation && !r.qwen21_gpu_w8a16 &&
                            r.qwen21_gpu_full_ffn_blocks.empty() &&
                            r.width == 512 && r.height == 512 &&
                            (r.loras.empty() ? (r.steps == 5 || r.steps == 20 || r.steps == 40 ||
                                                (fused_lora_ane && r.steps == 6)) :
                                (r.steps == 6 && lora_base_ane)) &&
                            (r.operation == "image.generate" ||
                             (r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3)) &&
                            !qwen21::option_enabled(rect_flag) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC")) &&
                            qwen21::tiled_prefill_layer_count(
                                std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC") ?
                                std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC") : "0") == 0,
                        "Qwen21 alternate W8A8 graph needs explicit 512px qualified base or six-step runtime LoRA with no cache or tiled prefill");
            require(qwen21::binary_option_or_unset(lora_ref512_flag),
                    "Qwen21 runtime LoRA/512px reference diagnostic accepts only 0 or 1");
            // One resident batch may switch base -> LoRA -> base on the same
            // runtime graph. The environment flag spans that whole batch;
            // allow its base requests without relaxing GPU/frozen eligibility.
            const bool runtime_ref512_base = runtime_ane && r.execution == "gpu_ane" &&
                r.loras.empty();
            if (qwen21::option_enabled(lora_ref512_flag))
                require(r.allow_approximation && r.width == 512 && r.height == 512 &&
                            r.steps == 6 && r.operation == "image.edit" &&
                            r.qwen21_reference_size == 512 && r.inputs.size() >= 1 &&
                            r.inputs.size() <= 3 && (r.loras.size() == 1 || runtime_ref512_base) &&
                            (r.execution == "gpu" ||
                             (r.execution == "gpu_ane" && (lora_base_ane || runtime_ane))),
                        "Qwen21 LoRA 512px reference resize requires explicit six-step editing with 1...3 references: Viggle on GPU/ANE or shared runtime base");
            require(qwen21::binary_option_or_unset(lora_ane_flag),
                    "Qwen21 runtime LoRA/base ANE diagnostic accepts only 0 or 1");
            if (lora_base_ane && (!fused_lora_ane || !r.loras.empty()))
                require(r.allow_approximation && r.execution == "gpu_ane" && r.qwen21_w8a8 &&
                            !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty() &&
                            r.width == 512 && r.height == 512 && r.steps == 6 &&
                            (r.qwen21_reference_size == 1024 ||
                             (r.qwen21_reference_size == 512 &&
                              qwen21::option_enabled(lora_ref512_flag))) && r.loras.size() == 1 &&
                            (r.operation == "image.generate" ||
                             (r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3)),
                        "Qwen21 runtime LoRA/base ANE requires explicit six-step 512px Viggle W8A8 hybrid with 0...3 full-size or diagnostic resized-512 references");
            const char *db_cache = std::getenv("TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC");
            const char *db_threshold = std::getenv("TURBOCIDER_QWEN21_DBCACHE_THRESHOLD");
            const char *db_max = std::getenv("TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE");
            require(qwen21::binary_option_or_unset(db_cache) &&
                        qwen21::db_cache_threshold(db_threshold) > 0.f &&
                        qwen21::db_cache_max_consecutive(db_max) > 0 &&
                        (!db_max || qwen21::option_enabled(db_cache)),
                    "Qwen21 DBCache requires explicit opt-in for a finite threshold in (0, 0.5] and a max consecutive skip count of 1...8");
            if (qwen21::option_enabled(db_cache))
                require(r.allow_approximation && r.steps >= 20 && r.steps <= 40 &&
                            ((r.width == 512 && r.height == 512) ||
                             (((r.width == 768 && r.height == 512) ||
                               (r.width == 512 && r.height == 768)) &&
                              (r.execution == "gpu" || qwen21::option_enabled(rect_flag)) &&
                              r.qwen21_reference_size == 1024)) &&
                            (r.execution == "gpu" || r.execution == "gpu_ane") &&
                            (r.operation == "image.generate" ||
                             (r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3)) &&
                            r.loras.empty() &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC")) &&
                            !qwen21::option_enabled(std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC")),
                        "Qwen21 DBCache is diagnostic-only for 20...40-step 512px or validated 768x512/512x768 base GPU or GPU/ANE with 0...3 references and no step-FFN reuse");
            const char *tiled_prefill = std::getenv("TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC");
            const int tiled_layers = qwen21::tiled_prefill_layer_count(tiled_prefill ? tiled_prefill : "0");
            const char *tiled_prefix_kv = std::getenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV");
            const char *tiled_prefix_reuse = std::getenv(
                "TURBOCIDER_QWEN21_TILED_PREFILL_PREFIX_KV_DIAGNOSTIC");
            require(qwen21::binary_option_or_unset(tiled_prefix_reuse),
                    "Qwen21 tiled prefill prefix-KV diagnostic accepts only 0 or 1");
            require(tiled_layers >= 0,
                    "TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC accepts only 0, 1, 8, 16, 20 or 24");
            if (tiled_layers > 0)
                require(r.execution == "gpu_ane" && r.qwen21_w8a8 && r.allow_approximation &&
                            r.width == 512 && r.height == 512 && r.operation == "image.edit" &&
                            r.qwen21_reference_size == 512 && r.inputs.size() >= 1 &&
                            r.inputs.size() <= 3 && r.qwen21_gpu_full_ffn_blocks.empty() &&
                            r.loras.empty() &&
                            ((!tiled_prefix_kv || std::string_view(tiled_prefix_kv) != "1") ||
                             (tiled_prefix_reuse && std::string_view(tiled_prefix_reuse) == "1" &&
                              tiled_layers == 16)),
                        "Qwen21 tiled first-step W8A8 needs 1...3 resized-512 edit references, full layer coverage; cross-request prefix-KV needs the explicit last-16 reuse diagnostic");
            if (tiled_prefix_reuse && std::string_view(tiled_prefix_reuse) == "1")
                require(tiled_prefix_kv && std::string_view(tiled_prefix_kv) == "1" &&
                            tiled_layers == 16 &&
                            r.residency == "resident" &&
                            r.steps == 5 && r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
                            r.execution == "gpu_ane" && r.operation == "image.edit" &&
                            r.width == 512 && r.height == 512 && r.qwen21_reference_size == 512 &&
                            !r.inputs.empty() && r.inputs.size() <= 3 &&
                            r.qwen21_gpu_full_ffn_blocks.empty() && r.loras.empty() &&
                            r.allow_approximation,
                        "Qwen21 tiled prefix-KV reuse needs an explicit resident five-step 512px-reference W8A8 edit and last-16 tiled prefill");
            const char *prefix_target_only = std::getenv(
                "TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC");
            require(qwen21::binary_option_or_unset(prefix_target_only),
                    "Qwen21 tiled prefix target-only diagnostic accepts only 0 or 1");
            if (prefix_target_only && std::string_view(prefix_target_only) == "1")
                require(tiled_prefix_reuse && std::string_view(tiled_prefix_reuse) == "1" &&
                            r.residency == "resident" && r.allow_approximation &&
                            r.execution == "gpu_ane" && r.qwen21_w8a8 &&
                            !r.qwen21_gpu_w8a16 && r.loras.empty(),
                        "Qwen21 tiled prefix target-only needs explicit resident W8A8 tiled prefix reuse");
            if (!r.loras.empty()) {
                const bool experimental_adapter = fused_lora_ane || runtime_ane;
                require(r.loras.size() == 1 && r.loras[0].role == "transformer" &&
                            (experimental_adapter
                                ? (std::isfinite(r.loras[0].strength) &&
                                   r.loras[0].strength >= -8.f && r.loras[0].strength <= 8.f)
                                : (r.loras[0].strength == 1.f &&
                                   qwen21::viggle_v021_adapter(
                                       std::filesystem::path(r.loras[0].path).filename().string()))) &&
                            r.steps == 6 && (r.execution == "gpu" ||
                                             (r.execution == "gpu_ane" && (lora_base_ane || runtime_ane))) &&
                            r.allow_approximation &&
                            r.width == 512 && r.height == 512 &&
                            (r.qwen21_reference_size == 1024 ||
                             (r.qwen21_reference_size == 512 &&
                              qwen21::option_enabled(lora_ref512_flag))) &&
                            (r.operation != "image.edit" || r.inputs.size() <= 3),
                        "Qwen21 requires one six-step runtime transformer LoRA with explicit 512px approximation; lora_fused/runtime permit an unqualified adapter/strength with the Viggle schedule");
            }
            const char *norm_rope = std::getenv("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE");
            const char *fused_qkv = std::getenv("TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC");
            const char *last_target = std::getenv("TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC");
            require(qwen21::binary_option_or_unset(last_target),
                    "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC accepts only 0 or 1");
            if (last_target && std::string_view(last_target) == "1") {
                const bool edit = r.operation == "image.edit" &&
                    (r.qwen21_reference_size == 512 || r.qwen21_reference_size == 1024) &&
                    !r.inputs.empty() && r.inputs.size() <= 3;
                const bool plain_gpu = r.execution == "gpu" &&
                    (edit || (r.operation == "image.generate" && r.inputs.empty()));
                const bool tiled_hybrid = r.execution == "gpu_ane" && edit && r.qwen21_w8a8 &&
                    !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty() &&
                    tiled_layers == 16;
                const bool full_reference_hybrid = r.execution == "gpu_ane" && edit &&
                    r.qwen21_reference_size == 1024 && r.qwen21_w8a8 &&
                    !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty() &&
                    std::getenv("TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC") &&
                    std::string_view(std::getenv("TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC")) == "1";
                require(r.allow_approximation && r.width == 512 && r.height == 512 &&
                            r.loras.empty() && (plain_gpu || tiled_hybrid || full_reference_hybrid),
                        "Qwen21 last-block target-only diagnostic needs explicit 512px GPU, last-16 tiled W8A8, or full-reference W8A8 editing, without LoRA");
            }
            const char *local_references = std::getenv("TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION");
            require(qwen21::binary_option_or_unset(fused_qkv),
                    "TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC accepts only 0 or 1");
            if (fused_qkv && std::string_view(fused_qkv) == "1") {
                const char *paired_rope = std::getenv("TURBOCIDER_QWEN21_METAL_QK_ROPE");
                const bool diagnostic_hybrid = r.execution == "gpu_ane" && r.qwen21_w8a8 &&
                    !r.qwen21_gpu_w8a16 && r.qwen21_gpu_full_ffn_blocks.empty() &&
                    r.operation == "image.edit" && r.qwen21_reference_size == 512 &&
                    !r.inputs.empty() && r.inputs.size() <= 3;
                const bool supported_input =
                    (r.operation == "image.generate" && r.inputs.empty()) ||
                    (r.operation == "image.edit" && r.qwen21_reference_size == 512 &&
                     !r.inputs.empty() && r.inputs.size() <= 3);
                require((r.execution == "gpu" || diagnostic_hybrid) && r.allow_approximation &&
                            r.width == 512 && r.height == 512 && r.loras.empty() &&
                            supported_input &&
                            (!norm_rope || std::string_view(norm_rope) != "1") &&
                            (!paired_rope || std::string_view(paired_rope) != "1") &&
                            (!local_references || std::string_view(local_references) == "0"),
                        "Qwen21 fused QKV diagnostic requires explicit 512px BF16 GPU generation or 1...3 resized-512 edit references (optionally W8A8/BF16 hybrid), without LoRA or other Q/K/attention experiment");
            }
            const char *profile_segments = std::getenv("TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS");
            require(qwen21::binary_option_or_unset(profile_segments),
                    "TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS accepts only 0 or 1");
            if (profile_segments && std::string_view(profile_segments) == "1") {
                const char *resident_prefix = std::getenv("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV");
                require(r.execution == "gpu" && r.width == 512 && r.height == 512 &&
                            r.operation == "image.edit" && r.qwen21_reference_size == 1024 &&
                            !r.inputs.empty() && r.inputs.size() <= 3 && r.loras.empty() &&
                            (!resident_prefix || std::string_view(resident_prefix) != "1") &&
                            (!local_references || std::string_view(local_references) == "0"),
                        "Qwen21 segment profiler needs 512px BF16 GPU editing with 1...3 full-size references and exact prefill");
            }
            require(!local_references || std::string_view(local_references) == "0" ||
                        std::string_view(local_references) == "1" ||
                        std::string_view(local_references) == "2" ||
                        std::string_view(local_references) == "3",
                    "TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION accepts only 0, 1, 2 or 3");
            if (local_references && std::string_view(local_references) != "0") {
                const bool local_lora = !r.loras.empty() && r.steps == 6 &&
                    std::string_view(local_references) == "3" &&
                    (r.execution == "gpu" || lora_base_ane);
                require((r.execution == "gpu" ||
                         (std::string_view(local_references) == "3" && r.execution == "gpu_ane" &&
                          r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
                          r.qwen21_gpu_full_ffn_blocks.empty())) && r.allow_approximation &&
                            r.width == 512 && r.height == 512 &&
                            r.operation == "image.edit" && r.qwen21_reference_size == 1024 &&
                            r.inputs.size() >= 2 && r.inputs.size() <= 3 &&
                            (r.loras.empty() || local_lora),
                        "Qwen21 reference-local attention needs 512px 2...3-full-reference GPU/hybrid editing; LoRA only supports six-step last-16 locality with the explicit base-ANE LoRA diagnostic for hybrid");
            }
            require(qwen21::binary_option_or_unset(norm_rope),
                    "TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE accepts only 0 or 1");
            if (norm_rope && std::string_view(norm_rope) == "1") {
                const char *paired = std::getenv("TURBOCIDER_QWEN21_METAL_QK_ROPE");
                // Reuse the same token-wise GPU kernel for long base sequences.
                // Do not widen the separate LoRA/editing qualification here.
                const bool base_1024 = r.width == 1024 && r.height == 1024 &&
                    r.operation == "image.generate" && r.inputs.empty() && r.loras.empty() &&
                    r.residency == "resident";
                require((r.execution == "gpu" ||
                         (r.execution == "gpu_ane" &&
                          (r.qwen21_w8a8 || runtime_ane || (runtime_qkv && base_1024)) &&
                          !r.qwen21_gpu_w8a16)) &&
                            r.allow_approximation &&
                            ((r.width == 512 && r.height == 512) || base_1024) &&
                            (!paired || std::string_view(paired) != "1"),
                        "Qwen21 fused Q/K norm-RoPE needs explicit 512px or resident 1024px base generation, GPU/W8A8/runtime FFN/QKV hybrid with BF16 GPU, approximation opt-in and no paired RoPE flag");
            }
            const char *reuse_flag = std::getenv("TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN");
            const char *hybrid_reuse = std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC");
            const char *hybrid_reuse_last16 =
                std::getenv("TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC");
            const bool hybrid_edit_base = r.execution == "gpu_ane" && r.qwen21_w8a8 &&
                r.allow_approximation && r.steps == 5 && r.width == 512 && r.height == 512 &&
                r.operation == "image.edit" && r.qwen21_reference_size == 512 &&
                !r.inputs.empty() && r.inputs.size() <= 3 &&
                r.qwen21_gpu_full_ffn_blocks.empty() && r.loras.empty() && tiled_layers == 16;
            require(qwen21::binary_option_or_unset(hybrid_reuse_last16),
                    "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC accepts only 0 or 1");
            if (qwen21::option_enabled(hybrid_reuse_last16))
                require(hybrid_edit_base && !gate_up_ane && !r.qwen21_gpu_w8a16 &&
                            !qwen21::option_enabled(hybrid_reuse) &&
                            !qwen21::option_enabled(reuse_flag),
                        "Qwen21 hybrid last-16 final FFN reuse needs 512px five-step W8A8 editing with last-16 tiled prefill, no LoRA or other final-step reuse");
            require(qwen21::binary_option_or_unset(hybrid_reuse),
                    "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC accepts only 0 or 1");
            if (qwen21::option_enabled(hybrid_reuse))
                require(hybrid_edit_base && !gate_up_ane &&
                            !qwen21::option_enabled(reuse_flag),
                        "Qwen21 hybrid final-step FFN reuse needs five-step W8A8 512px editing, 1...3 resized-512 references, last-16 tiled prefill, and no GPU reuse/LoRA");
            require(qwen21::binary_option_or_unset(reuse_flag),
                    "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN accepts only 0 or 1");
            if (reuse_flag && std::string_view(reuse_flag) == "1")
                require(r.execution == "gpu" && r.allow_approximation &&
                            r.width == 512 && r.height == 512 &&
                            (r.operation != "image.edit" ||
                             ((r.qwen21_reference_size == 256 || r.qwen21_reference_size == 512) &&
                              r.inputs.size() >= 1 && r.inputs.size() <= 3)),
                        "Qwen21 final-step FFN reuse needs explicit GPU and approximation opt-in at 512px; edits require 1...3 references resized to 256px or 512px");
            const char *half_reuse = std::getenv("TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN");
            require(qwen21::binary_option_or_unset(half_reuse),
                    "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN accepts only 0 or 1");
            if (half_reuse && std::string_view(half_reuse) == "1")
                require(reuse_flag && std::string_view(reuse_flag) == "1",
                        "Qwen21 penultimate even-layer FFN reuse requires final-step FFN reuse");
            const char *hybrid_half = std::getenv(
                "TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC");
            require(qwen21::binary_option_or_unset(hybrid_half),
                    "Qwen21 hybrid penultimate FFN reuse accepts only 0 or 1");
            if (hybrid_half && std::string_view(hybrid_half) == "1")
                require(!gate_up_ane && hybrid_reuse_last16 &&
                            std::string_view(hybrid_reuse_last16) == "1" &&
                            (!half_reuse || std::string_view(half_reuse) != "1"),
                        "Qwen21 hybrid penultimate even FFN reuse requires last-16 final reuse and excludes GPU penultimate reuse");
            if (!r.loras.empty())
                require(!reuse_flag || std::string_view(reuse_flag) != "1",
                        "Viggle six-step LoRA has not been validated with final-step FFN reuse");
            const char *lora_fp16 = std::getenv("TURBOCIDER_QWEN21_VIGGLE_LORA_FP16");
            require(qwen21::binary_option_or_unset(lora_fp16),
                    "TURBOCIDER_QWEN21_VIGGLE_LORA_FP16 accepts only 0 or 1");
            // With no adapter attached this flag has no effect, allowing a
            // resident session to switch back to the base GPU model.
            require(r.qwen21_gpu_full_ffn_blocks.empty() ||
                        (r.qwen21_w8a8 && r.qwen21_gpu_full_ffn_blocks == std::vector<int>{3, 5, 7}),
                    "Qwen21 full GPU fallback requires explicit W8A8 and validated layers 3,5,7");
            if (r.execution == "gpu_ane") {
                // Local benchmark escape hatch, not a quality-qualified public
                // route. The runtime still verifies the exact 4096-row,
                // checkpoint-matched, 32-block compiled W8A8 manifest.
                const char *diagnostic = std::getenv("TURBOCIDER_QWEN21_1024_W8A8_DIAGNOSTIC");
                const bool diagnostic_t2i = diagnostic && std::string_view(diagnostic) == "1" &&
                    r.width == 1024 && r.height == 1024 && r.qwen21_w8a8 &&
                    r.operation == "image.generate" && r.inputs.empty() &&
                    r.qwen21_reference_size == 1024 && r.qwen21_gpu_full_ffn_blocks.empty();
                const bool diagnostic_rectangle = qwen21::option_enabled(rect_flag);
                const bool diagnostic_lora_base_ane = lora_base_ane;
                // The Core ML FFN only sees the 512px output's 1024 decode
                // tokens. Full-size references are cached as GPU prefix KV,
                // so this tests their *actual* fidelity and prefill cost
                // without changing the calibrated 1024-row ANE graph.
                const char *full_ref = std::getenv("TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC");
                const bool diagnostic_full_ref = full_ref && std::string_view(full_ref) == "1" &&
                    r.width == 512 && r.height == 512 && r.qwen21_w8a8 &&
                    r.operation == "image.edit" && !r.inputs.empty() && r.inputs.size() <= 3 &&
                    r.qwen21_reference_size == 1024 && r.qwen21_gpu_full_ffn_blocks.empty();
                const bool supported_512 = r.width == 512 && r.height == 512 &&
                    ((r.operation == "image.generate" && r.inputs.empty() &&
                      r.qwen21_reference_size == 1024 && r.qwen21_gpu_full_ffn_blocks.empty()) ||
                     (r.qwen21_w8a8 && r.operation == "image.edit" &&
                      (r.qwen21_reference_size == 256 ||
                       (r.qwen21_reference_size == 512 && r.qwen21_gpu_full_ffn_blocks.empty())) &&
                      r.inputs.size() >= 1 && r.inputs.size() <= 3 &&
                      (r.qwen21_gpu_full_ffn_blocks.empty() ||
                       r.qwen21_gpu_full_ffn_blocks == std::vector<int>{3, 5, 7})));
                require(r.allow_approximation && !r.ane_manifest.empty() &&
                            (runtime_ane || runtime_qkv || diagnostic_t2i || diagnostic_rectangle || diagnostic_lora_base_ane ||
                             diagnostic_full_ref || supported_512),
                        "Qwen21 gpu_ane requires a validated 512px route; 1024px text-to-image, full-reference editing and 768x512/512x768 tiling are diagnostic-only");
            } else {
                require(r.ane_manifest.empty() && r.encoder_ane_manifest.empty() &&
                            !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 &&
                            r.qwen21_gpu_full_ffn_blocks.empty(),
                        "Qwen21 ANE manifests require execution=gpu_ane");
            }
            require(r.residency == "resident" || r.residency == "component_staged", "Qwen21 supports resident or component_staged residency");
            require(!r.streaming_offload && r.quantized_cache.empty(), "Qwen21 streaming/quantized cache are not implemented");
            if (r.operation == "image.generate") require(r.inputs.empty(), "Qwen21 image.generate takes no images");
            else {
                require(!r.inputs.empty() && r.inputs.size() <= 10, "Qwen21 image.edit requires 1...10 references");
                for (const auto &input : r.inputs)
                    require(input.kind == "image" && input.role == "reference", "Qwen21 editing requires image/reference inputs");
            }
        },
        [](const std::filesystem::path &root) { return std::make_unique<qwen21::Session>(root); },
        [] {
            ModelDescriptor d;
            d.id = "qwen-image-2.1"; d.name = "Qwen Image 2.1 (experimental)"; d.executable = true;
            d.operations = d.executor_operations = {"image.generate", "image.edit"};
            d.inputs = {"text", "image"}; d.roles = {"reference"}; d.max_images = 10; d.output = "image";
            d.steps = 40; d.frames = 1; d.width = d.height = 1024; d.default_audio = false;
            d.supports_lora = true; d.runtime_lora = true;
            d.lora_mode = "inference-time-viggle-v0.2.1-r128/r256; alternate six-step adapters experimental in explicit lora_fused";
            d.lora_strategies = {"inference_time"}; d.default_lora_strategy = "inference_time";
            d.default_residency = "component_staged"; d.backend = "mlx_cpp_metal";
            d.supports_gpu_ane = true;
            d.runtime_dependency = "bundled-native-mlx-cpp";
            d.candidate_limitations = {
                "experimental: actual edit material fidelity is under investigation; not quality-qualified",
                "BF16 Comfy checkpoint plus official processor/tokenizer.json required",
                "reference images are resized to approximately 1024 squared pixels with 32-aligned dimensions",
                "RGBA is preserved; App masks are visual references, not hard pixel-preserving inpainting",
                "native PE-T2I is optional and slow; PE-I2I requires explicit prompt_enhance_edit_experimental with FP32 vision, supported 8-bit files, and is not quality-qualified; BF16 visual parity remains unaccepted",
                "experimental gpu_ane: explicit FP16 512x512 text-to-image or W8A8 512x512 edit with 1...3 references scaled to 256; full 32-layer coverage is faster but changes some edited details, while GPU-only blocks 3,5,7 remain an opt-in alternative; device placement and broad quality are not qualified"
            };
            return d;
        }};
}
} // namespace tc
