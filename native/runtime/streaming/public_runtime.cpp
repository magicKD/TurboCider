#include "public_runtime.hpp"

#include "canonical_encoding.hpp"
#include "preset_resolver.hpp"
#include "public_request_validation.hpp"
#include "../session.hpp"
#include "../../core/common.hpp"

#include <bit>

namespace tc::streaming {
namespace {

template <class T>
void encode_optional_unsigned(
        CanonicalEncoder &out, std::string_view name,
        const std::optional<T> &value) {
    const std::string prefix(name);
    out.boolean_field(prefix + ".present", value.has_value());
    if (value)
        out.unsigned_field(prefix + ".value",
                           static_cast<uint64_t>(*value));
}

void encode_optional_bool(
        CanonicalEncoder &out, std::string_view name,
        const std::optional<bool> &value) {
    const std::string prefix(name);
    out.boolean_field(prefix + ".present", value.has_value());
    if (value) out.boolean_field(prefix + ".value", *value);
}

void encode_selector(
        CanonicalEncoder &out, std::string_view name,
        const std::optional<StreamingSelector> &value) {
    const std::string prefix(name);
    out.boolean_field(prefix + ".present", value.has_value());
    if (!value) return;
    const auto &selector = *value;
    encode_optional_bool(out, prefix + ".enabled", selector.enabled);
    encode_optional_unsigned(
        out, prefix + ".schema_version", selector.schema_version);
    out.optional_string_field(
        prefix + ".selection", selector.selection);
    out.optional_string_field(
        prefix + ".retention", selector.retention);
    encode_optional_unsigned(
        out, prefix + ".target_request_memory_bytes",
        selector.target_request_memory_bytes);
    out.optional_string_field(prefix + ".preset_id", selector.preset_id);
    encode_optional_unsigned(
        out, prefix + ".preset_revision", selector.preset_revision);
    out.optional_string_field(
        prefix + ".catalog_revision", selector.catalog_revision);
    out.optional_string_field(
        prefix + ".expected_resolution_digest",
        selector.expected_resolution_digest);
}

std::string preflight_request_digest(const Request &request) {
    CanonicalEncoder out("tc-public-streaming-preflight-v1");
    out.string_field("model", request.model);
    out.string_field("operation", request.operation);
    out.string_field("execution", request.execution);
    out.string_field("prompt", request.prompt);
    out.string_field("output", request.output);
    out.string_field("dump", request.dump);
    out.string_field("noise_path", request.noise_path);
    out.string_field("profile", request.profile);
    out.string_field("profile_identity", request.profile_identity);
    out.string_field("model_variant", request.model_variant);
    out.string_field("residency", request.residency);
    out.string_field("lora_strategy", request.lora_strategy);
    out.string_field("ltx_backend", request.ltx_backend);
    out.unsigned_field("width", static_cast<uint64_t>(request.width));
    out.unsigned_field("height", static_cast<uint64_t>(request.height));
    out.unsigned_field("steps", static_cast<uint64_t>(request.steps));
    out.unsigned_field("frames", static_cast<uint64_t>(request.frames));
    out.unsigned_field("fps", static_cast<uint64_t>(request.fps));
    out.unsigned_field("seed", request.seed);
    out.unsigned_field(
        "allocator_cache_bytes", request.allocator_cache_bytes);
    out.unsigned_field(
        "warmup_iterations",
        static_cast<uint64_t>(request.warmup_iterations));
    out.boolean_field("audio", request.audio);
    out.boolean_field("dynamic_text", request.dynamic_text);
    out.boolean_field("vsa", request.vsa);
    out.unsigned_field(
        "vsa_sparsity_bits",
        std::bit_cast<uint64_t>(request.vsa_sparsity));
    out.unsigned_field(
        "vsa_tile_size", static_cast<uint64_t>(request.vsa_tile_size));
    out.string_field("vsa_prefix_mode", request.vsa_prefix_mode);
    out.unsigned_field(
        "vsa_dense_first_n_steps",
        static_cast<uint64_t>(request.vsa_dense_first_n_steps));
    out.begin_list(
        "vsa_dense_layers", request.vsa_dense_layers.size());
    for (const int layer : request.vsa_dense_layers)
        out.unsigned_field(
            "vsa_dense_layer", static_cast<uint64_t>(layer));
    out.string_field("vsa_impl", request.vsa_impl);
    out.boolean_field("ltx_fast_av", request.ltx_fast_av);
    out.boolean_field(
        "ltx_video_attention_batch", request.ltx_video_attention_batch);
    out.boolean_field("ltx_sol_stage1", request.ltx_sol_stage1);
    out.boolean_field("ltx_sol_stage2", request.ltx_sol_stage2);
    out.unsigned_field(
        "ltx_sol_tau_bits",
        std::bit_cast<uint64_t>(request.ltx_sol_tau));
    out.unsigned_field(
        "ltx_sparse_mode",
        static_cast<uint64_t>(request.ltx_sparse_mode));
    out.unsigned_field(
        "ltx_sparse_radius",
        static_cast<uint64_t>(request.ltx_sparse_radius));
    out.unsigned_field(
        "ltx_sparse_anchor_stride",
        static_cast<uint64_t>(request.ltx_sparse_anchor_stride));
    out.unsigned_field(
        "ltx_sparse_tokens_per_frame",
        static_cast<uint64_t>(request.ltx_sparse_tokens_per_frame));
    out.unsigned_field(
        "ltx_sparse_keep_blocks",
        static_cast<uint64_t>(request.ltx_sparse_keep_blocks));
    out.unsigned_field(
        "ltx_sol_dense_edge_blocks",
        static_cast<uint64_t>(request.ltx_sol_dense_edge_blocks));
    out.unsigned_field(
        "ltx_sol_dense_edge_steps",
        static_cast<uint64_t>(request.ltx_sol_dense_edge_steps));
    out.unsigned_field(
        "ltx_stage2_text_rows",
        static_cast<uint64_t>(request.ltx_stage2_text_rows));
    out.begin_list("inputs", request.inputs.size());
    for (const auto &input : request.inputs) {
        out.string_field("input.kind", input.kind);
        out.string_field("input.role", input.role);
        out.string_field("input.path", input.path);
        out.string_field("input.text", input.text);
        out.string_field("input.audio_path", input.audio_path);
        out.boolean_field("input.include_audio", input.include_audio);
        out.unsigned_field(
            "input.strength_bits",
            std::bit_cast<uint32_t>(input.strength));
    }
    out.begin_list("loras", request.loras.size());
    for (const auto &lora : request.loras) {
        out.string_field("lora.path", lora.path);
        out.string_field("lora.role", lora.role);
        out.unsigned_field(
            "lora.strength_bits",
            std::bit_cast<uint32_t>(lora.strength));
    }
    out.string_field("ane_manifest", request.ane_manifest);
    out.string_field("encoder_ane_manifest", request.encoder_ane_manifest);
    out.string_field("quantized_cache", request.quantized_cache);
    out.boolean_field("allow_approximation", request.allow_approximation);
    out.boolean_field("compile_gpu", request.compile_gpu);
    out.unsigned_field("lora_count", request.loras.size());
    out.boolean_field("streaming_config_specified",
                      request.streaming.specified());
    out.boolean_field("residency_specified",
                      request.residency_specified);
    out.boolean_field("memory_budget_specified",
                      request.memory_budget_specified);
    out.boolean_field("streaming_offload_specified",
                      request.streaming_offload_specified);
    out.unsigned_field("memory_budget_bytes",
                       request.memory_budget_bytes);
    out.boolean_field("streaming_offload", request.streaming_offload);
    out.unsigned_field(
        "memory_constrained.specified_fields",
        request.memory_constrained.specified_fields);
    out.boolean_field("memory_constrained_enabled",
                      request.memory_constrained.enabled);
    out.unsigned_field(
        "memory_constrained.limit_bytes",
        request.memory_constrained.limit_bytes);
    out.unsigned_field(
        "memory_constrained.buffer_percent",
        request.memory_constrained.buffer_percent);
    out.unsigned_field(
        "memory_constrained.min_free_bytes",
        request.memory_constrained.min_free_bytes);
    out.unsigned_field(
        "memory_constrained.max_refill_slots",
        request.memory_constrained.max_refill_slots);
    out.boolean_field(
        "memory_constrained.allow_quality_preserving_tiling",
        request.memory_constrained.allow_quality_preserving_tiling);
    encode_selector(out, "selector", request.streaming_selector);
    encode_selector(
        out, "selector_requested",
        request.streaming_selector_requested);
    return out.sha256();
}

} // namespace

PublicStreamingCoordinator::PublicStreamingCoordinator(
        ModelSession &session, std::string engine_model_id,
        std::string execution_container,
        const StreamingCatalogProvider &catalog_provider)
    : session_(session), engine_model_id_(std::move(engine_model_id)),
      execution_container_(std::move(execution_container)),
      catalog_provider_(catalog_provider) {
    require(!engine_model_id_.empty(), "streaming_engine_unavailable");
    require(!execution_container_.empty(),
            "streaming_execution_container_mismatch");
}

PublicStreamingPreflight PublicStreamingCoordinator::preflight(
        const Request &request) const {
    require(request.model == engine_model_id_,
            "streaming_engine_model_mismatch");
    validate_public_streaming_request(request);
    auto catalog = catalog_provider_.snapshot();
    require(catalog != nullptr, "streaming_catalog_unavailable");
    // This check deliberately precedes make_plan/model validation at the API
    // layer while still allowing pure host tests to inject a read-only
    // provider directly into this C++ coordinator.
    require(!catalog->records.empty(),
            "catalog_has_no_public_records");
    return PublicStreamingPreflight(
        std::move(catalog), preflight_request_digest(request));
}

std::shared_ptr<const ResolvedRequestExecution>
PublicStreamingCoordinator::resolve_normalized(
        Request request, const StreamingDeviceIdentity &device,
        PublicStreamingPreflight preflight) const {
    require(preflight.catalog_ != nullptr,
            "streaming_preflight_mismatch");
    require(preflight.request_digest_ ==
                preflight_request_digest(request),
            "streaming_preflight_mismatch");
    PublicResolveInput input{request, device, execution_container_};
    auto probe = session_.probe_public_streaming(input);
    require(probe != nullptr, "streaming_public_probe_unavailable");
    require(probe->model_id() == engine_model_id_ &&
                probe->workload_identity().model == engine_model_id_ &&
                probe->workload_identity().execution_container ==
                    execution_container_,
            "streaming_probe_identity_mismatch");

    auto selected = PublicPresetResolver::select(
        *request.streaming_selector, *probe, device,
        *preflight.catalog_);
    require(probe->component_policy_revision() ==
                selected.record.plan.component_policy_revision,
            "streaming_probe_identity_mismatch");
    auto snapshot = session_.compile_public_streaming(
        probe, selected.record);
    require(snapshot != nullptr,
            "streaming_public_snapshot_unavailable");
    auto selection = PublicPresetResolver::authorize(
        selected, *probe, *snapshot, device);

    request.streaming_selector.reset();
    request.streaming_selector_requested.reset();
    request.streaming = selection.record.plan.canonical_config;
    request.streaming_requested = request.streaming;
    const auto request_digest = streaming_workload_identity_digest(
        probe->workload_identity());
    return std::make_shared<const ResolvedRequestExecution>(
        ResolvedRequestExecution{
            std::move(request), std::move(selection), std::move(probe),
            std::move(snapshot), request_digest});
}

void PublicStreamingCoordinator::revalidate(
        const ResolvedRequestExecution &execution,
        const StreamingDeviceIdentity &current_device) const {
    require(execution.probe != nullptr &&
                execution.model_snapshot != nullptr,
            "streaming_authority_mismatch");
    const auto *probe_lease = execution.probe->source_lease();
    const auto *snapshot_lease = execution.model_snapshot->source_lease();
    require(probe_lease != nullptr && snapshot_lease != nullptr,
            "streaming_source_lease_required");
    require(probe_lease == snapshot_lease &&
                probe_lease->generation() != 0 &&
                probe_lease->digest() ==
                    execution.probe->source_identity()
                        .source_snapshot_digest &&
                snapshot_lease->digest() ==
                    execution.model_snapshot->source_identity()
                        .source_snapshot_digest,
            "streaming_source_lease_mismatch");
    snapshot_lease->revalidate_paths();
    snapshot_lease->revalidate_open_files();
    execution.model_snapshot->revalidate_source();
    auto catalog = catalog_provider_.snapshot();
    require(catalog != nullptr, "streaming_catalog_unavailable");
    const auto replay = PublicPresetResolver::select(
        execution.selection.exact_selector, *execution.probe,
        current_device, *catalog);
    require(replay.record.canonical_record_digest ==
                execution.selection.record.canonical_record_digest,
            "streaming_resolution_stale");
    require(execution.selection.authority &&
                execution.selection.authority->matches(
                    execution.selection.record,
                    *execution.model_snapshot, current_device),
            "streaming_authority_mismatch");
    require(execution.selection.exact_selector.expected_resolution_digest &&
                *execution.selection.exact_selector
                     .expected_resolution_digest ==
                    execution.selection.resolution_digest,
            "streaming_resolution_stale");
}

} // namespace tc::streaming
