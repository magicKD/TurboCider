#include "pipeline.hpp"
#include "../../backends/ane_memory.hpp"
#include "../../components/vae/tiny.hpp"
#include "../../media/image.hpp"
#include "bf16_streaming.hpp"
#include "checkpoint_recipe.hpp"
#include "conditioning.hpp"
#include "diagnostic_options.hpp"
#include "encoder_convrot.hpp"
#include "memory_policy.hpp"
#include "scheduler.hpp"
#include "vae.hpp"
#include "vae_decode_residency.hpp"
#if TURBOCIDER_ENABLE_PRIVATE_ANE
#include "convrot_stream_ane.hpp"
#endif
#include <mlx/random.h>
#include <cstdlib>
#include <thread>
#include <bit>

namespace tc::qwen21 {
namespace {
uint32_t option(const char *name, uint32_t fallback, uint32_t maximum) {
    const char *raw = std::getenv(name);
    if (!raw) return fallback;
    const auto text = std::string_view(raw);
    require(!text.empty() && text.size() <= 2 && text.find_first_not_of("0123456789") == std::string_view::npos,
            "Qwen BF16 stream option requires a bounded integer");
    const auto n = std::stoul(std::string(text));
    require(n <= maximum, "Qwen BF16 stream option exceeds bound");
    return uint32_t(n);
}

struct ScopedCacheLimit {
    size_t previous;
    explicit ScopedCacheLimit(size_t bytes) : previous(mx::set_cache_limit(bytes)) {}
    ~ScopedCacheLimit() { mx::set_cache_limit(previous); }
};

void check_convrot_stream_headroom(std::atomic<bool> &cancelled) {
    // Pressure notifications can lag a phase's completed releases. Wait only
    // for that transient flag, retaining cancellation and every memory guard.
    for(int retry=0;retry<300;++retry) {
        checkpoint(cancelled);
        const auto observed=ane::observe_streaming_memory(mx::get_active_memory());
        const auto admission=ane::admit_streaming_memory(observed,
            {uint64_t(4)<<30,observed.memory.physical_bytes},0);
        if(admission.allowed())return;
        if(admission.denial!=ane::MemoryDenial::SystemPressure)break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ane::require_streaming_memory_headroom(mx::get_active_memory(), 0, "Qwen21 ConvRot streaming");
}
}

RunResult Session::run_weight_streamed(const Request &requested, const Event &event,
                                       std::atomic<bool> &cancelled, bool warmup, bool prepare_only,
                                       bool vae_decode_only) {
    Request r = requested;
    const bool convrot = r.residency == "streamed";
    const bool runtime_ane = convrot_stream_ane_requested(r);
    const bool turbo = turbo_checkpoint(transformer_source_);
    const bool convrot_encoder = convrot_encoder_checkpoint(encoder_source_);
    const bool convrot_dit = convrot_checkpoint(transformer_source_);
    const bool safetensors = transformer_source_.extension() == ".safetensors" &&
                             encoder_source_.extension() == ".safetensors";
    if (convrot) {
        require(safetensors && (convrot_encoder || convrot_dit),
                "Qwen21 ConvRot streaming needs original safetensors and at least one ConvRot component");
        if (!runtime_ane) r.execution = "gpu";
        r.model_variant = turbo ? "qwen-image-2.1-turbo" : "qwen-image-2.1";
        require(!r.prompt.empty(), "Qwen21 streaming requires a prompt");
        if (!warmup && !prepare_only)
            require(!r.output.empty() && std::filesystem::path(r.output).extension() == ".png",
                    "Qwen21 streaming requires a PNG output path");
    } else {
        require(!prepare_only && r.model == "qwen-image-2.1" && r.operation == "image.generate" && r.inputs.empty() &&
                r.width == 512 && r.height == 512 && r.steps >= 2 && r.steps <= 40 && r.execution == "gpu" &&
                r.residency == "component_staged" && r.loras.empty() && !r.prompt_enhance && r.hybrid_mlp_mode == "auto" &&
                r.ane_manifest.empty() && r.encoder_ane_manifest.empty() && r.qwen21_dit_cache == "off" &&
                !r.qwen21_w8a8 && !r.qwen21_gpu_w8a16 && !r.streaming.active() && !r.streaming_offload &&
                !r.memory_constrained.enabled && !r.memory_budget_bytes && safetensors && !convrot_encoder && !convrot_dit,
                "Qwen original-BF16 streaming requires component-staged 512px base GPU generation or warmup");
    }
    for (const char *flag : {"TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS", "TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME",
        "TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU", "TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU", "TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC",
        "TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV", "TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC",
        "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN", "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN",
        "TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC", "TURBOCIDER_QWEN21_METAL_QK_ROPE",
        "TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE", "TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS", "TURBOCIDER_QWEN21_PROFILE_GPU_OPS",
        "TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS", "TURBOCIDER_QWEN21_GGUF_SHARED_DOWN"})
        require(!option_enabled(std::getenv(flag)), "Qwen weight streaming conflicts with another math/cache/profiling recipe");

    const bool w8a8_available = convrot && convrot_w8a8_available();
    const bool encoder_w8a8 = convrot_encoder && configured_convrot_w8a8(true, w8a8_available);
    const bool dit_w8a8 = convrot_dit && configured_convrot_w8a8(false, w8a8_available);
    require(w8a8_available || (!encoder_w8a8 && !dit_w8a8),
            "ConvRot W8A8 requires M5 or later with macOS 26.4 or later");
    require(!runtime_ane || (turbo && dit_w8a8), "ConvRot streaming Runtime ANE requires the original Turbo ConvRot DiT and GPU W8A8");
    const bool timestep_cache=stream_timestep_cache_request(requested,turbo,convrot_encoder,convrot_dit,
        dit_w8a8,runtime_ane,std::getenv("TURBOCIDER_QWEN21_STREAM_TIMESTEP_CACHE"));
    const auto bf16_mode = convrot ? StreamWeightMode::Bf16ConvRotCompanion : StreamWeightMode::Bf16;
    const auto encoder_mode = !convrot_encoder ? bf16_mode :
        encoder_w8a8 ? StreamWeightMode::ConvRotW8A8 : StreamWeightMode::ConvRotFloat;
    const auto dit_mode = !convrot_dit ? bf16_mode :
        dit_w8a8 ? StreamWeightMode::ConvRotW8A8 : StreamWeightMode::ConvRotFloat;
    const auto encoder_prefix = convrot ? 0u : option("TURBOCIDER_QWEN21_BF16_STREAM_ENCODER_PREFIX", 24, 34);
    const auto dit_prefix = convrot ? 0u : option("TURBOCIDER_QWEN21_BF16_STREAM_DIT_PREFIX", 24, 30);
    const uint64_t budget = convrot ? (r.qwen21_stream_weight_budget_bytes ?
        r.qwen21_stream_weight_budget_bytes : convrot_stream_weight_budget_bytes) :
        uint64_t(option("TURBOCIDER_QWEN21_BF16_STREAM_WEIGHT_GIB", 11, 20)) << 30;
    require(budget > 0, "Qwen streaming weight budget must be positive");

    const auto start = Clock::now();
    const auto elapsed = [](auto begin) { return std::chrono::duration<double>(Clock::now() - begin).count(); };
    checkpoint(cancelled);
    mx::synchronize();
    unload();
    if (convrot) check_convrot_stream_headroom(cancelled);
    mx::reset_peak_memory();
    ScopedCacheLimit cache_limit(convrot ? std::min(r.allocator_cache_bytes, uint64_t(256) << 20) : r.allocator_cache_bytes);
    const auto stats_start = convrot_w8a8_stats();

    RunResult result;
    result.request = r;
    result.plan = make_plan(r);
    result.prepared = prepare_only;
    result.warmup = warmup;
    result.backend = convrot ? "mlx_cpp_metal_qwen21_convrot_streaming" : "mlx_cpp_metal_qwen21_bf16_streaming_experimental";
    result.checkpoint = transformer_source_.filename().string();
    result.selection = convrot ? "ConvRot encoder and DiT weight streaming with two refill slots and phase-local weights" :
        "Private original-BF16 component-staged encoder and DiT with two refill slots";
    const bool tiny_decoder = r.tiny_vae && r.tiny_vae->decoder == "tiny";
    result.precision = std::string(convrot_dit ? (dit_w8a8 ? "w8a8_convrot_dit" : "int8_convrot_dit") : "bf16_dit") + "+" +
        (convrot_encoder ? (encoder_w8a8 ? "w8a8_convrot_text" : "int8_convrot_text") : "bf16_text") + "+" +
        (tiny_decoder ? "tiny_vae" : "bf16_vae");
    auto emit = [&](const char *phase, int completed, int total) { if (event) event(phase, completed, total); };
    auto make_encoder_plan = [&] {
        return bf16_stream_plan(encoder_source_, true, 36, encoder_prefix, 1, cancelled, encoder_mode);
    };
    auto make_dit_plan = [&] {
        auto plan=bf16_stream_plan(transformer_source_, false, 32, dit_prefix, uint32_t(r.steps), cancelled, dit_mode);
        if(convrot && r.qwen21_stream_weight_budget_bytes) {
            const auto observation=ane::observe_streaming_memory(mx::get_active_memory());
            const auto cache_headroom=convrot_stream_cache_headroom(observation);
            plan=bf16_stream_cache_plan(std::move(plan),budget,[cache_headroom](uint64_t capacity) {
                return capacity<=cache_headroom;
            });
        }
        return plan;
    };
    if (prepare_only) {
        bf16_stream_scope([&] {
            emit("load_qwen21_text", 0, 1);
            (void)make_encoder_plan();
            emit("load_qwen21_text", 1, 1);
            emit("load_qwen21_transformer", 0, 1);
            (void)make_dit_plan();
            emit("load_qwen21_transformer", 1, 1);
        });
        result.selection += "; source layouts verified";
        result.timings.wall = elapsed(start);
        result.active_bytes = mx::get_active_memory();
        result.peak_bytes = mx::get_peak_memory();
        return result;
    }
    if (convrot) {
        result.qwen_convrot_streaming.emplace();
        result.qwen_convrot_streaming->requested_weight_budget_bytes=r.qwen21_stream_weight_budget_bytes;
    }
    else result.qwen_bf16_streaming.emplace();
    result.qwen_ffn_phases.emplace();
    result.qwen_ffn_phases->policy = runtime_ane ? "convrot_stream_runtime_ane_decode" : "gpu";
    auto dump = [&](const char *name, const Tensor &tensor) {
        if (r.dump.empty()) return;
        std::filesystem::create_directories(r.dump);
        mx::save_safetensors((std::filesystem::path(r.dump) / (std::string(name) + ".safetensors")).string(), {{"tensor", tensor}});
    };

    components::TinyPreview preview(r, event, cancelled);
    Tensor text(0.f);
    const auto text_start = Clock::now();
    bf16_stream_scope([&] {
        emit("load_qwen21_text", 0, 1);
        Bf16StreamBank bank(make_encoder_plan(), budget, cancelled);
        emit("load_qwen21_text", 1, 1);
        Tokenizer tokenizer(root_ / "processor");
        const auto tokens = tokenizer.raw_bounded(reference_prompt_template(r.prompt, 0), Tokenizer::qwen21_limit);
        auto assembled = convrot_encoder ?
            assemble_prompt_rows(tokens, bank.embedding_rows(tokens), bank.embedding_vocabulary(), {}) :
            assemble_prompt(tokens, bank.resident(), {});
        TextConfig config;
        config.final_norm = false;
        TextEncoder encoder(bank.resident(), config);
        encoder.set_layer_weights([&](int layer) -> const Weights & { return bank.acquire(layer); },
                                  [&](int layer) { bank.retire(layer); });
        bank.begin_pass(0);
        text = assembled.retain(encoder.encode_embeddings(assembled.embeddings, assembled.positions,
                                                         assembled.embeddings.shape(1), event, cancelled));
        mx::eval(text);
        require(mx::all(mx::isfinite(text)).item<bool>(), "nonfinite streamed Qwen encoder output");
        bank.finish_pass();
        bank.finish();
        if (convrot) result.qwen_convrot_streaming->encoder = bank.metrics();
        else result.qwen_bf16_streaming->encoder = bank.metrics();
    });
    const auto stats_after_encoder = convrot_w8a8_stats();
    mx::clear_cache();
    result.timings.text = elapsed(text_start);
    dump("qwen21_text", text);
    const int rows = r.height / 16;
    const int columns = r.width / 16;
    const int image_tokens = rows * columns;
    result.text_tokens = result.valid_text_tokens = text.shape(1);
    result.total_tokens = text.shape(1) + image_tokens;
    auto schedule = turbo ? turbo_sigmas(r.width, r.height, r.steps) : sigmas(r.width, r.height, r.steps);
    mx::eval(schedule);
    auto latents = mx::astype(mx::random::normal({1, image_tokens, 64}, mx::float32, mx::random::key(r.seed)), mx::bfloat16);
    text = mx::astype(text, mx::bfloat16);
    mx::eval(text);
    if (!r.noise_path.empty()) {
        auto [values, metadata] = mx::load_safetensors(r.noise_path);
        require(values.count("tensor") || values.count("initial"), "Qwen noise file missing tensor");
        const auto &noise = values.at(values.count("tensor") ? "tensor" : "initial");
        require(noise.shape() == latents.shape(), "Qwen noise shape mismatch");
        latents = mx::astype(noise, mx::bfloat16);
    }
    dump("qwen21_initial", latents);
    auto spatial = [&](const Tensor &sample) {
        return mx::transpose(mx::reshape(sample, {1, rows, columns, 64}), {0, 3, 1, 2});
    };

    emit("load_qwen21_transformer", 0, 1);
    const auto denoise_start = Clock::now();
    bf16_stream_scope([&] {
        auto dit_plan=make_dit_plan();
        const auto time_source=timestep_cache ? dit_plan.lease->descriptor().files.at(0).content_digest : std::string{};
        Bf16StreamBank bank(std::move(dit_plan), budget, cancelled);
        emit("load_qwen21_transformer", 1, 1);
#if TURBOCIDER_ENABLE_PRIVATE_ANE
        std::unique_ptr<ConvrotStreamAne> hybrid;
        if (runtime_ane) {
            const auto setup = Clock::now();
            hybrid = std::make_unique<ConvrotStreamAne>(r.ane_manifest, cancelled);
            result.timings.hybrid = elapsed(setup);
        }
#endif
        Transformer dit(bank.resident());
#if TURBOCIDER_ENABLE_PRIVATE_ANE
        if (hybrid) hybrid->attach(dit);
#endif
        dit.set_layer_weights([&](int layer) -> const Weights & {
            const auto &weights = bank.acquire(layer);
#if TURBOCIDER_ENABLE_PRIVATE_ANE
            if (hybrid) hybrid->acquire(layer, weights);
#endif
            return weights;
        }, [&](int layer) {
#if TURBOCIDER_ENABLE_PRIVATE_ANE
            if (hybrid) hybrid->retire(layer);
#endif
            bank.retire(layer);
        });
        emit("denoise", 0, r.steps);
        const float *sigma = schedule.data<float>();
        std::vector<Transformer::PreparedTimestepCondition> time_conditions;
        if(timestep_cache) {
            require(latents.dtype()==mx::bfloat16 && text.dtype()==mx::bfloat16 &&
                schedule.dtype()==mx::float32 && schedule.shape()==mx::Shape{9},
                "Qwen timestep cache requires actual BF16 inputs and the full authored F32 schedule");
            for(size_t i=0;i<turbo_sigma_nodes.size();++i)
                require(std::bit_cast<uint32_t>(sigma[i])==std::bit_cast<uint32_t>(turbo_sigma_nodes[i]),
                    "Qwen timestep cache actual schedule differs from authored F32 sigma bits");
            auto &metrics=result.qwen_convrot_streaming->timestep_cache.emplace();
            metrics.source_sha256=time_source;metrics.schedule_identity=std::string(turbo_sampler_identity);
            metrics.active_before_prepare=mx::get_active_memory();metrics.cache_before_prepare=mx::get_cache_memory();
            checkpoint(cancelled);const auto prepare_start=Clock::now();
            time_conditions=dit.prepare_timestep_conditions(std::vector<float>(sigma,sigma+r.steps),latents.dtype(),time_source);
            metrics.prepare_seconds=elapsed(prepare_start);metrics.frames=uint32_t(time_conditions.size());
            for(const auto &frame:time_conditions) {
                metrics.condition_logical_bytes+=frame.logical_bytes();
                metrics.condition_capacity_bytes+=frame.capacity_bytes();
            }
            require(metrics.frames==8 && metrics.condition_logical_bytes==655360,
                "Qwen timestep cache full original M2 condition coverage changed");
            // Preparation synchronizes and publishes only independent leaves.
            // No reader/refill pass has begun. Retire both actual source owners
            // through the bank; cumulative reads and initial admission stay intact.
            checkpoint(cancelled);
            metrics.active_before_release=mx::get_active_memory();metrics.cache_before_release=mx::get_cache_memory();
            const auto release_start=Clock::now();
            metrics.released_weight_capacity_bytes=bank.release_fixed_fields(Transformer::timestep_condition_weight_names());
            metrics.active_after_release=mx::get_active_memory();metrics.cache_after_release=mx::get_cache_memory();
            mx::clear_cache();
            metrics.active_after_clear=mx::get_active_memory();metrics.cache_after_clear=mx::get_cache_memory();
            metrics.release_seconds=elapsed(release_start);
            require(metrics.released_weight_capacity_bytes==203423744 &&
                metrics.active_before_release>=metrics.active_after_release &&
                metrics.active_before_release-metrics.active_after_release>=metrics.released_weight_capacity_bytes,
                "Qwen timestep source owners did not release their actual independent capacity");
        }
        for (int step = 0; step < r.steps; ++step) {
          bf16_stream_scope([&] {
            checkpoint(cancelled);
            if (convrot) {
                const auto observed=ane::observe_streaming_memory(mx::get_active_memory());
                const auto admission=ane::admit_streaming_memory(observed,
                    {uint64_t(4)<<30,observed.memory.physical_bytes},uint64_t(1)<<29);
                if(!admission.allowed()) {
                    // No new weight allocation or source read is needed to
                    // resume: the original two slots have stayed live.
                    bank.evict_cached_layers();
                }
                check_convrot_stream_headroom(cancelled);
            }
            bank.begin_pass(uint32_t(step));
            const auto step_start = Clock::now();
            const bool reuse = dit.prefix_matches(text, rows, columns, {});
            auto velocity = timestep_cache ?
                dit.forward(latents, text, sigma[step], rows, columns, true,nullptr,{},&time_conditions.at(size_t(step))) :
                dit.forward(latents, text, sigma[step], rows, columns, true);
            std::optional<Tensor> preview_estimate;
            if (preview.due(step + 1) && step + 1 < r.steps)
                preview_estimate = components::flow_preview_estimate(latents, velocity, sigma[step]);
            latents = latents + velocity * Tensor(sigma[step + 1] - sigma[step], latents.dtype());
            mx::eval(latents);
            require(mx::all(mx::isfinite(latents)).item<bool>(), "nonfinite streamed Qwen latent");
            bank.finish_pass();
            auto &phase = reuse ? result.qwen_ffn_phases->decode : result.qwen_ffn_phases->prefill;
            ++phase.steps;
            phase.rows = reuse ? uint64_t(image_tokens) : uint64_t(result.total_tokens);
            phase.step_seconds += elapsed(step_start);
            ++result.actual_steps;
            emit("denoise", step + 1, r.steps);
            if (preview.due(step + 1)) preview.publish(spatial(preview_estimate ? *preview_estimate : latents), step + 1);
          });
        }
#if TURBOCIDER_ENABLE_PRIVATE_ANE
        if (hybrid) {
            hybrid->finish();
            result.hybrid = hybrid->metrics();
            result.selection += hybrid->available() ? "; private Runtime ANE FFN diagnostic" :
                "; GPU fallback: " + hybrid->reason();
        }
#endif
        bank.finish();
        if (convrot) result.qwen_convrot_streaming->denoiser = bank.metrics();
        else result.qwen_bf16_streaming->denoiser = bank.metrics();
    });
    if (convrot) {
        const auto stats_end = convrot_w8a8_stats();
        result.qwen_convrot_w8a8 = QwenConvrotW8A8Metrics{
            .encoder_enabled = encoder_w8a8,
            .dit_enabled = dit_w8a8,
            .encoder_projections = stats_after_encoder.projections - stats_start.projections,
            .encoder_activation_rows = stats_after_encoder.activation_rows - stats_start.activation_rows,
            .dit_projections = stats_end.projections - stats_after_encoder.projections,
            .dit_activation_rows = stats_end.activation_rows - stats_after_encoder.activation_rows,
        };
    }
    mx::clear_cache();
    result.timings.denoise = elapsed(denoise_start);
    dump("qwen21_latents", latents);

    checkpoint(cancelled);
    if (convrot) check_convrot_stream_headroom(cancelled);
    const auto decode_start = Clock::now();
    auto pixels = [&] {
        Tensor decoded(0.f);
        if (tiny_decoder) {
            components::TinyAutoencoder tiny(r.tiny_vae->path, r.model, false, true);
            result.selection += "; TAEQI2.1 final decode";
            decoded = mx::clip(tiny.decode(spatial(latents), event, cancelled), Tensor(0.f), Tensor(1.f)) * 2.f - 1.f;
            mx::eval(decoded);
        } else {
            Weights vae;
            emit("load_qwen21_vae", 0, 1);
            vae.load_file(root_ / "vae/qwen_image_2.1_vae_bf16.safetensors");
            if (vae_decode_only)
                result.qwen_stream_vae_decode_residency = retain_streamed_vae_decode_weights(vae, cancelled);
            vae.materialize();
            emit("load_qwen21_vae", 1, 1);
            VAE decoder(vae);
            decoded = decoder.decode(spatial(latents), event, cancelled);
            mx::eval(decoded);
        }
        return mx::transpose(decoded, {0, 2, 3, 1});
    }();
    mx::eval(pixels);
    require(mx::all(mx::isfinite(pixels)).item<bool>(), "nonfinite streamed Qwen pixels");
    dump("qwen21_pixels", pixels);
    result.timings.decode = elapsed(decode_start);
    if (!warmup) {
        checkpoint(cancelled);
        emit("export", 0, 1);
        save_rgba_png(pixels, r.output);
        emit("export", 1, 1);
    }
    mx::clear_cache();
    result.timings.wall = elapsed(start);
    result.active_bytes = mx::get_active_memory();
    result.peak_bytes = mx::get_peak_memory();
    return result;
}
} // namespace tc::qwen21
