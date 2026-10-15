#include "native_session.hpp"
#include "native_route.hpp"
#include "native_memory_plan.hpp"
#include "mlx_native_denoiser.hpp"
#include "viggle_adapter.hpp"
#include "../../backends/metal_graph/mlx_bridge.hpp"
#include "../../runtime/memory_execution.hpp"
#include "../../runtime/streaming/canonical_encoding.hpp"
#include "../../core/tokenizer.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <optional>

namespace tc::qwen21 {
namespace {
using namespace native_graph;
using namespace metal_graph;
MatrixView imported_matrix(Backend& backend, const Tensor& value) {
    auto view = import_mlx_matrix(backend, value);
    return {view.storage, view.rows, view.columns, view.row_stride_bytes};
}
constexpr uint64_t native_allocator_domain = 0x51574e4d4554414cULL;
constexpr size_t final_latent_bytes = 1024 * 64 * sizeof(uint16_t);

struct AdapterPreparation {
    std::string revision;
    uint32_t max_rank = 0;
    bool ffn = false;
    uint64_t resolved_bindings = 0;
    DenoiserScheduleVariant schedule = DenoiserScheduleVariant::Base20;
};
void encode_factor(streaming::CanonicalEncoder& out, const Tensor& factor, mx::Dtype dtype) {
    require((dtype == mx::bfloat16 || dtype == mx::float32) && dtype == factor.dtype() &&
            factor.ndim() == 2 && factor.shape(0) > 0 && factor.shape(1) > 0 &&
            factor.shape(0) <= 32768 && factor.shape(1) <= 32768 &&
            factor.strides()[1] == 1 && factor.strides()[0] == factor.shape(1) &&
            factor.is_available() && factor.buffer().ptr() && factor.offset() >= 0,
            "native LoRA factors must be produced compact BF16/FP32 matrices before phase admission");
    const auto offset = size_t(factor.offset());
    require(offset <= factor.buffer_size() && factor.nbytes() <= factor.buffer_size() - offset,
            "native LoRA factor escapes its retained source backing");
    out.string_field("dtype", dtype == mx::bfloat16 ? "bf16" : "fp32");
    out.unsigned_field("rows", uint64_t(factor.shape(0)));
    out.unsigned_field("columns", uint64_t(factor.shape(1)));
}
AdapterPreparation prepare_adapter(const Request& request, const Weights& weights,
    const std::shared_ptr<const streaming::SourceLease>& source,
    native_lora::KernelVariant kernel) {
    const auto snapshot = weights.runtime_lora_snapshot();
    if (request.loras.empty()) {
        require(!source && snapshot.bindings.empty() && !weights.has_runtime_loras(),
                "native base request cannot contain an unrequested adapter source or resolved terms");
        return {};
    }
    require(request.execution == "gpu" && source && source->has_verified_content() &&
            source->file_count() == request.loras.size() && source->generation() != 0,
            "native runtime LoRA needs the exact verified held source set and GPU execution");
    require(!snapshot.bindings.empty() && snapshot.native_metadata_compatible && !snapshot.fp16_rank_math,
            "native runtime LoRA metadata/rank arithmetic is unsupported: " + snapshot.native_rejection_reason);
    source->revalidate_after_drain();
    streaming::CanonicalEncoder identity("tc-qwen-native-runtime-lora-v1");
    identity.string_field("verified_artifact", source->artifact_digest());
    identity.unsigned_field("source_generation", source->generation());
    identity.string_field("precision_recipe", native_lora::recipe_for_variant(kernel));
    identity.begin_list("ordered_requested_adapters", request.loras.size());
    for (size_t i = 0; i < request.loras.size(); ++i) {
        const auto& adapter = request.loras[i];
        const auto logical = "native-adapter-" + std::to_string(i);
        const auto& file = source->file(logical);
        require(adapter.role == "transformer" && std::isfinite(adapter.strength) &&
                std::abs(adapter.strength) <= 8.f && !adapter.path.empty() &&
                file.path == std::filesystem::absolute(adapter.path).lexically_normal() &&
                !file.content_digest.empty(),
                "native adapter request order, role, path or verified content differs from the held source");
        identity.unsigned_field("request_ordinal", i);
        identity.string_field("logical_id", logical);
        identity.string_field("role", adapter.role);
        identity.string_field("verified_content_sha256", file.content_digest);
        identity.unsigned_field("strength_f32_bits", std::bit_cast<uint32_t>(adapter.strength));
    }
    AdapterPreparation result;
    result.resolved_bindings = snapshot.bindings.size();
    if (request.steps == 6) {
        require(request.loras.size() == 1 && request.loras.front().strength == 1.f,
                "native six-step schedule requires exactly one pinned student at strength one");
        const auto* pinned = viggle_v021_adapter(
            std::filesystem::path(request.loras.front().path).filename().string());
        require(pinned && pinned->rank == "r128" &&
                source->file("native-adapter-0").content_digest == pinned->sha256,
                "native six-step schedule requires the verified local Viggle v0.2.1 r128 content pin");
        result.schedule = DenoiserScheduleVariant::ViggleLoRA6;
    } else {
        require(request.steps == 20, "native ordinary LoRA currently requires the explicit 20-step base schedule");
        for (const auto& adapter : request.loras)
            require(!viggle_v021_adapter(std::filesystem::path(adapter.path).filename().string()),
                    "native pinned Viggle student cannot use the ordinary base schedule");
    }
    identity.unsigned_field("schedule_variant", uint64_t(result.schedule));
    std::vector<bool> consumed(request.loras.size(), false);
    identity.begin_list("resolved_targets", snapshot.bindings.size());
    std::string previous;
    for (const auto& binding : snapshot.bindings) {
        require(!binding.prefix.empty() && binding.prefix > previous && !binding.ordered_terms.empty(),
                "native resolved adapter targets must be nonempty, unique and ordered");
        previous = binding.prefix;
        identity.string_field("target", binding.prefix);
        identity.begin_list("ordered_terms", binding.ordered_terms.size());
        size_t previous_ordinal = 0;
        for (size_t i = 0; i < binding.ordered_terms.size(); ++i) {
            const auto& term = binding.ordered_terms[i];
            const auto ordinal = term.source_adapter_ordinal;
            require(ordinal < request.loras.size() && (!i || ordinal >= previous_ordinal) &&
                    term.lease_backed && term.source_lease.get() == source.get() &&
                    term.source_role == "transformer" &&
                    term.source_logical_id == "native-adapter-" + std::to_string(ordinal) &&
                    std::bit_cast<uint32_t>(term.source_strength) ==
                        std::bit_cast<uint32_t>(request.loras[ordinal].strength),
                    "native adapter term does not retain the exact ordered request source lease");
            previous_ordinal = ordinal;
            identity.unsigned_field("term_index", i);
            identity.unsigned_field("source_ordinal", ordinal);
            identity.string_field("source_logical_id", term.source_logical_id);
            encode_factor(identity, term.down, term.down_dtype);
            encode_factor(identity, term.up, term.up_dtype);
            require(term.down.shape(0) <= 128 && term.up.shape(1) == term.down.shape(0) &&
                    term.output_start >= 0 && term.output_end > term.output_start &&
                    term.up.shape(0) == term.output_end - term.output_start && std::isfinite(term.scale),
                    "native adapter rank, output range or effective scale is unsupported");
            identity.unsigned_field("effective_scale_f32_bits", std::bit_cast<uint32_t>(term.scale));
            identity.unsigned_field("first_output", uint64_t(term.output_start));
            identity.unsigned_field("output_count", uint64_t(term.output_end - term.output_start));
            result.max_rank = std::max(result.max_rank, uint32_t(term.down.shape(0)));
            result.ffn = result.ffn || binding.prefix.find(".img_mlp.") != std::string::npos;
            consumed[ordinal] = true;
        }
    }
    require(std::all_of(consumed.begin(), consumed.end(), [](bool value) { return value; }),
            "native adapter source has no consumed runtime projection");
    // The importer additionally proves every output range is fully consumed
    // by a supported projection. No pathname/stat tuple grants this identity.
    result.revision = identity.sha256();
    return result;
}
}

struct NativeSession::Impl {
    // Reverse destruction order is part of the contract: keep the phase and
    // held file generations alive until every imported buffer owner is gone.
    std::shared_ptr<const streaming::SourceLease> source, conditioning, adapter_source;
    std::optional<Tensor> text_owner, latent_owner;
    std::unique_ptr<NativePhaseMemory> phase;
    std::unique_ptr<Backend> backend;
    std::optional<NativeDenoiser> model;
    std::unique_ptr<DenoiserRun> run;
    NativeDenoiserLoRAMetrics adapter_metrics;
    bool result_copied = false;
    bool native_released = false;
    bool phase_finalized = false;
    bool cleanup_quarantined = false;
    std::string cleanup_error;

    void quarantine(std::string_view reason) noexcept {
        cleanup_quarantined = true;
        try { cleanup_error.assign(reason); } catch (...) {}
    }

    void revalidate() const {
        source->revalidate_after_drain();
        conditioning->revalidate_after_drain();
        if (adapter_source) adapter_source->revalidate_after_drain();
    }
    void release_native() {
        run.reset();
        model.reset();
        backend.reset();
        native_released = true;
    }
};

NativeSession::NativeSession(const Request& request,
    std::shared_ptr<const streaming::SourceLease> source,
    std::shared_ptr<const streaming::SourceLease> conditioning,
    const Weights& weights, const Tensor& text, const Tensor& latents,
    const Tensor& schedule, MemoryExecutionContext* parent_context,
    std::shared_ptr<const streaming::SourceLease> adapter_source)
    : impl_(std::make_unique<Impl>()) {
    // Explicit Base/W1 and the narrow GPU Viggle candidate share admission;
    // no adapter/kernel choice grants automatic or App performance status.
    validate_native_denoiser_request(request);
    require(native_denoiser_requested(request), "native Qwen delegate requires explicit executor selection");
    const bool hybrid = request.execution == "gpu_ane";
    if (hybrid) {
#if !defined(TURBOCIDER_ENABLE_PRIVATE_ANE) || !TURBOCIDER_ENABLE_PRIVATE_ANE
        throw std::invalid_argument("native Qwen GPU/ANE requires a private-enabled build");
#else
        const std::filesystem::path cache(request.qwen21_native_ane_cache);
        require(!cache.empty() && cache.is_absolute(),
                "native Qwen GPU/ANE requires an explicit absolute program cache");
#endif
    }
    require(!parent_context, "native Qwen phase cannot nest a parent memory ledger");
    require(bool(source) && bool(conditioning), "native Qwen requires held source and conditioning descriptors");
    require(text.dtype() == mx::bfloat16 && text.ndim() == 3 && text.shape(0) == 1 &&
            text.shape(1) > 0 && text.shape(1) <= Tokenizer::qwen21_limit && text.shape(2) == 4096,
            "native Qwen requires original produced BF16 text [1,tokens,4096]");
    require(latents.dtype() == mx::bfloat16 && latents.shape() == mx::Shape{1,1024,64},
            "native Qwen requires BF16 512px latent geometry");
    require(request.steps > 0 && schedule.dtype() == mx::float32 && schedule.ndim() == 1 &&
            schedule.size() == size_t(request.steps) + 1 &&
            schedule.strides()[0] == 1,
            "native Qwen requires the original request-length FP32 Euler schedule");
    impl_->source = std::move(source);
    impl_->conditioning = std::move(conditioning);
    impl_->adapter_source = std::move(adapter_source);
    impl_->text_owner = text;
    impl_->latent_owner = latents;
    // One explicit MLX→native readiness boundary. No MLX operators are built
    // by step(), and native commands never infer readiness from a raw handle.
    mx::eval({text, latents, schedule});
    mx::synchronize();
    // Pipeline has already materialized every base/factor backing. Inspect
    // produced metadata only; no factor eval/cast/copy can occur after baseline.
    const auto kernel = request.qwen21_native_lora_kernel == "sgmma" ?
        native_lora::KernelVariant::SgMmaR128 : native_lora::KernelVariant::Scalar;
    const auto adapter = prepare_adapter(request, weights, impl_->adapter_source, kernel);
    impl_->revalidate();
    mx::clear_cache();
    const auto baseline = observe_process_memory();
    const DenoiserGeometry geometry;
    NativePhaseShape phase_shape{geometry, 1024, uint32_t(text.shape(1)), {}, hybrid};
    phase_shape.weight_scale_cache_entries = request.qwen21_native_scale_cache ? geometry.layers : 0;
    phase_shape.lora_max_rank = adapter.max_rank;
    phase_shape.lora_ffn = adapter.ffn;
    std::string phase_source(impl_->source->digest());
    if (!adapter.revision.empty()) {
        streaming::CanonicalEncoder identity("tc-qwen-native-adapted-phase-v1");
        identity.string_field("base_source_revision", phase_source);
        identity.string_field("adapter_revision", adapter.revision);
        phase_source = identity.sha256();
    }
    NativePhaseLimits limits;
    // The legacy request.memory_budget_bytes is a denoiser residency hint,
    // checked by ResidencyPolicy. It is not a process-wide phase budget.
    // This scope uses observed physical capacity minus the explicit reserve.
    impl_->phase = std::make_unique<NativePhaseMemory>(phase_shape,
        std::move(phase_source), baseline, limits,
        native_allocator_domain, impl_->source->generation(), parent_context);
    BackendOptions options;
    options.allocation = impl_->phase->hooks();
    impl_->backend = std::make_unique<Backend>(std::move(options));
    NativeDenoiserExecutionOptions execution;
    execution.hybrid_ffn = hybrid;
    execution.ane_program_cache = request.qwen21_native_ane_cache;
    execution.allocation = impl_->phase->hooks();
    execution.register_weight_staging = request.qwen21_native_w8_staging == "register";
    execution.cache_weight_scales = request.qwen21_native_scale_cache;
    execution.adapter_revision = adapter.revision;
    execution.lora_kernel = kernel;
    execution.schedule_variant = adapter.schedule;
    auto imported = adapter.revision.empty() ?
        import_base_denoiser_weights(*impl_->backend, weights, geometry) :
        import_runtime_lora_denoiser_weights(*impl_->backend, weights, geometry);
    if (!adapter.revision.empty()) {
        auto& metrics = impl_->adapter_metrics;
        metrics.enabled = true;
        metrics.variant = adapter.schedule == DenoiserScheduleVariant::ViggleLoRA6 ?
            "viggle-r128-6step" : "runtime-lora-base20";
        metrics.adapter_count = impl_->adapter_source->file_count();
        metrics.source_generation = impl_->adapter_source->generation();
        metrics.verified_artifact_digest = impl_->adapter_source->artifact_digest();
        metrics.adapter_sha256 = impl_->adapter_source->file("native-adapter-0").content_digest;
        metrics.source_strength = request.loras.front().strength;
        metrics.resolved_bindings = adapter.resolved_bindings;
        metrics.adapter_revision = adapter.revision;
        metrics.max_rank = adapter.max_rank;
        auto count = [&](const std::vector<OrderedLoRATerm>& terms) {
            metrics.logical_projections += !terms.empty();
            metrics.ordered_terms += terms.size();
        };
        for (const auto* projection : {&imported.image_input, &imported.time_input,
            &imported.time_output, &imported.text_input, &imported.text_output,
            &imported.modulation, &imported.output_scale, &imported.output}) count(projection->lora);
        for (const auto& block : imported.blocks) {
            for (const auto* projection : {&block.query, &block.key, &block.value,
                 &block.attention_output}) count(projection->lora);
            for (const auto* terms : {&block.gate_lora, &block.up_lora, &block.down_lora}) count(*terms);
        }
    }
    impl_->model.emplace(NativeDenoiser::prepare(*impl_->backend, std::move(imported),
        std::string(impl_->source->digest()), geometry, std::move(execution)));
    if (impl_->adapter_metrics.enabled) {
        const auto actual_recipe = impl_->model->lora_precision_recipe();
        require(actual_recipe == native_lora::recipe_for_variant(kernel),
                "native adapter revision and prepared kernel arithmetic recipe differ");
        impl_->adapter_metrics.precision_recipe = actual_recipe;
    }
    DenoiserInput input;
    input.latents = imported_matrix(*impl_->backend, latents);
    input.text = imported_matrix(*impl_->backend, text);
    input.height = request.height / 16;
    input.width = request.width / 16;
    // This is a request-local label attached to immutable, produced owners.
    // It is not a model content hash or a cross-request conditioning cache key.
    input.identity = {std::string(impl_->source->digest()),
        std::string(impl_->conditioning->digest()) + ":request-text:" +
            std::to_string(impl_->source->generation()), adapter.revision, impl_->source->generation()};
    input.sigmas.assign(schedule.data<float>(), schedule.data<float>() + schedule.size());
    impl_->run = impl_->model->begin(std::move(input));
    impl_->phase->context().checkpoint("native_qwen_begin");
}

NativeSession::~NativeSession() noexcept {
    if (!impl_) return;
    // Destruction is not a physical-completion proof. If a caller bypasses the
    // enclosing Session quarantine contract, retain the entire execution owner
    // for process lifetime rather than freeing source/phase state under Metal.
    if (!abort(std::chrono::milliseconds(0), "native Qwen owner destroyed"))
        (void)impl_.release();
}

Completion NativeSession::step() {
    require(impl_ && impl_->run && !impl_->native_released && !impl_->cleanup_quarantined,
            "native Qwen delegate is not running");
    impl_->revalidate();
    impl_->phase->context().checkpoint("native_qwen_before_step");
    auto result = impl_->run->step();
    require(result.ok && result.physically_complete, "native Qwen step failed: " + result.error);
    impl_->revalidate();
    impl_->phase->context().checkpoint("native_qwen_after_step");
    return result;
}

Completion NativeSession::physical_status() const {
    if (!impl_ || !impl_->run) return {true, true, false, 0, 0, 0., {}};
    return impl_->run->physical_status();
}

Completion NativeSession::drain(std::chrono::milliseconds timeout) {
    require(timeout.count() >= 0, "native Qwen drain timeout must be nonnegative");
    if (!impl_ || !impl_->run) return {true, true, false, 0, 0, 0., {}};
    const auto completion = impl_->run->drain(timeout);
    // Source validation cannot grant physical completion or revive a retired
    // result. Only inspect after the captured domains really finished.
    if (completion.physically_complete) impl_->revalidate();
    return completion;
}

NativeDenoiserLoRAMetrics NativeSession::lora_metrics() const {
    require(impl_ && impl_->run && impl_->result_copied && !impl_->native_released &&
            !impl_->cleanup_quarantined && impl_->adapter_metrics.enabled &&
            impl_->run->state() == DenoiserState::Complete,
            "native LoRA receipt requires the completed retained run and published latents");
    const auto completion = impl_->run->physical_status();
    require(completion.ok && completion.physically_complete,
            "native LoRA receipt requires physical completion");
    impl_->revalidate();
    auto metrics = impl_->adapter_metrics;
    metrics.completed = true;
    metrics.completed_steps = impl_->run->completed_steps();
    metrics.scratch_bytes = impl_->run->memory().lora_scratch_bytes;
    return metrics;
}

NativeHybridDenoiserMetrics NativeSession::hybrid_metrics() const {
    require(impl_ && impl_->run && !impl_->native_released,
            "native Qwen metrics require the retained run before finish");
    const auto native=impl_->run->hybrid_metrics();
    NativeHybridDenoiserMetrics result;
    result.enabled=native.enabled;
    result.successful_ffns=native.successful_ffns;
    result.completed_hybrid_steps=native.completed_hybrid_steps;
    result.surface_bytes=native.surface_bytes;
    result.partial_bytes=native.partial_bytes;
    result.producer_seconds=native.producer_seconds;
    result.submit_ane_seconds=native.submit_ane_seconds;
    result.branch_join_seconds=native.branch_join_seconds;
    result.consumer_seconds=native.consumer_seconds;
    result.total_seconds=native.total_seconds;
    result.producer_gpu_span_seconds=native.producer_gpu_span_seconds;
    result.gpu_complement_span_seconds=native.gpu_complement_span_seconds;
    result.device_timed_ffns=native.device_timed_ffns;
    result.weight_staging=native.weight_staging;
    result.weight_scale_cache_enabled=native.weight_scale_cache_enabled;
    result.weight_scale_cache_entries=native.weight_scale_cache_entries;
    result.weight_scale_cache_payload_bytes=native.weight_scale_cache_payload_bytes;
    result.weight_scale_cache_hits=native.weight_scale_cache_hits;
    result.weight_scale_cache_misses=native.weight_scale_cache_misses;
    return result;
}

Tensor NativeSession::finish_latents() {
    require(impl_ && impl_->run && !impl_->result_copied && !impl_->native_released &&
            !impl_->cleanup_quarantined,
            "native Qwen result is unavailable or already published");
    const auto completion = impl_->run->physical_status();
    require(completion.ok && completion.physically_complete,
            "native Qwen output requires physical completion");
    auto source = impl_->run->final_latents();
    impl_->revalidate();
    require(source.rows == 1024 && source.columns == 64 &&
            source.storage.scalar_type() == ScalarType::BF16 && source.storage.contents(),
            "native Qwen final output is not shared BF16 latent storage");
    auto output = allocate_mlx_matrix(1024, 64);
    {
        auto destination = imported_matrix(*impl_->backend, output);
        require(destination.storage.contents() && destination.row_stride_bytes == 128 &&
                destination.storage.size_bytes() == final_latent_bytes,
                "native Qwen output boundary extent mismatch");
        auto* target = static_cast<std::byte*>(destination.storage.contents());
        auto* input = static_cast<const std::byte*>(source.storage.contents());
        for (uint32_t row = 0; row < 1024; ++row)
            std::memcpy(target + row * 128, input + row * source.row_stride_bytes, 128);
    }
    // The small MLX publication backing belongs to the phase's explicit
    // framework allowance; it is not manufactured as a native StorageLease.
    impl_->phase->context().checkpoint("native_qwen_output_boundary");
    impl_->result_copied = true;
    return output;
}

MemoryExecutionReport NativeSession::finish() {
    require(impl_ && impl_->run && impl_->result_copied && !impl_->native_released &&
            !impl_->cleanup_quarantined,
            "native Qwen cannot finish before publishing complete latents");
    const auto completion = impl_->run->physical_status();
    require(completion.ok && completion.physically_complete, "native Qwen finish needs physical drain");
    impl_->revalidate();
    impl_->release_native();
    impl_->phase->finish_after_release(true);
    // Set this only after the ledger has reached Succeeded. If phase finish
    // throws after owner release, abort must still collect its disposition.
    impl_->phase_finalized = true;
    return impl_->phase->context().take_report();
}

bool NativeSession::abort(std::chrono::milliseconds timeout, const std::string& reason) noexcept {
    if (!impl_) return true;
    if (impl_->cleanup_quarantined) return false;
    if (impl_->phase_finalized) return impl_->native_released;
    try {
        const auto completion = drain(timeout);
        if (!completion.physically_complete) {
            if (impl_->phase) {
                const auto disposition = impl_->phase->fail_after_drain(false, reason);
                impl_->quarantine(memory_failure_disposition_name(disposition));
            } else impl_->quarantine("physical completion is unproven");
            close_memory_binding();
            return false;
        }
        if (!impl_->native_released) impl_->release_native();
        if (impl_->phase) {
            const auto disposition = impl_->phase->fail_after_drain(true, reason);
            const bool clean = disposition == MemoryFailureDisposition::None ||
                               disposition == MemoryFailureDisposition::Clean;
            if (!clean || !impl_->phase->context().finished()) {
                impl_->quarantine(memory_failure_disposition_name(disposition));
                close_memory_binding();
                return false;
            }
        }
        impl_->phase_finalized = true;
        return true;
    } catch (const std::exception& error) {
        // Preserve the cleanup exception and terminate the phase as well.
        // Only release_native() records proven completion plus owner release;
        // an exception while draining must not manufacture that proof.
        try {
            if (impl_->phase)
                (void)impl_->phase->fail_after_drain(impl_->native_released, reason);
        } catch (...) {}
        impl_->quarantine(error.what());
        close_memory_binding();
        return false;
    } catch (...) {
        try {
            if (impl_->phase)
                (void)impl_->phase->fail_after_drain(impl_->native_released, reason);
        } catch (...) {}
        impl_->quarantine("unknown native Qwen cleanup failure");
        close_memory_binding();
        return false;
    }
}

bool NativeSession::cleanup_quarantined() const noexcept {
    return impl_ && impl_->cleanup_quarantined;
}

std::string_view NativeSession::cleanup_failure() const noexcept {
    if (!impl_ || !impl_->cleanup_quarantined) return {};
    return impl_->cleanup_error.empty() ? std::string_view("native cleanup is unsafe")
                                       : std::string_view(impl_->cleanup_error);
}

void NativeSession::close_memory_binding() noexcept {
    if (impl_ && impl_->phase) impl_->phase->close();
}

} // namespace tc::qwen21
