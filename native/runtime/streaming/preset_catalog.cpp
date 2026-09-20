#include "preset_catalog.hpp"
#include "../../core/contracts.hpp"

#include "canonical_encoding.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace tc::streaming {
namespace {

void catalog_check(bool value, const std::string &message) {
    if (!value)
        throw std::invalid_argument("streaming_catalog_invalid: " + message);
}

bool valid_identifier(const std::string &value, size_t maximum = 256) {
    return !value.empty() && value.size() <= maximum &&
        value.find('\0') == std::string::npos;
}

bool valid_digest(const std::string &value) {
    return value.size() == 64 &&
        std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
        });
}

bool released(const StreamingPresetRecord &record, bool experimental) {
    return record.release.channel == "public-stable" ||
        (experimental && record.release.channel == "public-experimental");
}

bool device_memory_matches(const StreamingPresetRecord &record,
                           uint64_t physical) {
    if (!physical || physical < record.device.minimum_physical_memory_bytes)
        return false;
    return !record.device.maximum_physical_memory_bytes ||
        physical <= record.device.maximum_physical_memory_bytes;
}

bool fits(const StreamingPresetRecord &record, uint64_t target) {
    if (!record.calibration.complete ||
        record.calibration.scope != "execution_process_tree_v1" ||
        record.calibration.calibrated_request_bytes > target)
        return false;
    const uint64_t margin = streaming_target_margin_bytes(target);
    return margin <= target - record.calibration.calibrated_request_bytes;
}

void encode_config(CanonicalEncoder &out, const StreamingConfig &config) {
    out.boolean_field("config.enabled.present", config.enabled.has_value());
    if (config.enabled)
        out.boolean_field("config.enabled.value", *config.enabled);
    out.boolean_field(
        "config.schema_version.present", config.schema_version.has_value());
    if (config.schema_version)
        out.unsigned_field("config.schema_version.value", *config.schema_version);
    out.optional_string_field("config.selection", config.selection);
    out.optional_string_field("config.retention", config.retention);
    out.begin_list("config.stages", config.stages.size());
    for (const auto &[id, stage] : config.stages) {
        out.string_field("stage.id", id);
        out.optional_string_field("stage.residency", stage.residency);
        auto optional_number = [&](std::string_view name,
                                   const std::optional<uint32_t> &value) {
            const std::string prefix(name);
            out.boolean_field(prefix + ".present", value.has_value());
            if (value) out.unsigned_field(prefix + ".value", *value);
        };
        optional_number("stage.block_group_size", stage.block_group_size);
        optional_number("stage.slot_count", stage.slot_count);
        optional_number(
            "stage.resident_prefix_blocks", stage.resident_prefix_blocks);
        optional_number("stage.prefetch_distance", stage.prefetch_distance);
        optional_number("stage.io_workers", stage.io_workers);
    }
}

void encode_source(CanonicalEncoder &out, const PresetSourceIdentity &source) {
    if (source.identity_version == 2)
        out.unsigned_field("source.identity_version", 2);
    out.string_field("source.model_variant", source.model_variant);
    out.string_field("source.weight_format", source.weight_format);
    out.string_field(
        "source.artifact_manifest_digest", source.artifact_manifest_digest);
    if (source.identity_version == 1)
        out.string_field(
            "source.source_snapshot_digest", source.source_snapshot_digest);
}

void encode_runtime(CanonicalEncoder &out, const PresetRuntimeIdentity &runtime) {
    out.string_field("runtime.turbocider_build_id", runtime.turbocider_build_id);
    out.string_field("runtime.runtime_revision", runtime.runtime_revision);
    out.string_field("runtime.adapter_revision", runtime.adapter_revision);
    out.string_field("runtime.reader_revision", runtime.reader_revision);
    out.string_field("runtime.kernel_revision", runtime.kernel_revision);
    out.string_field(
        "runtime.allocator_policy_revision",
        runtime.allocator_policy_revision);
}

void encode_workload(CanonicalEncoder &out, const PresetWorkload &workload) {
    out.string_field("workload.model", workload.model);
    out.string_field("workload.operation", workload.operation);
    out.string_field("workload.execution", workload.execution);
    out.string_field("workload.device_class", workload.device_class);
    out.string_field(
        "workload.execution_container", workload.execution_container);
    out.unsigned_field("workload.width", workload.width);
    out.unsigned_field("workload.height", workload.height);
    out.unsigned_field("workload.frames", workload.frames);
    out.unsigned_field("workload.fps", workload.fps);
    out.unsigned_field("workload.steps", workload.steps);
    out.unsigned_field("workload.batch", workload.batch);
    out.boolean_field("workload.audio", workload.audio);
    out.boolean_field("workload.dynamic_text", workload.dynamic_text);
    out.boolean_field("workload.approximation", workload.approximation);
    out.string_field(
        "workload.conditioning_revision", workload.conditioning_revision);
    out.string_field(
        "workload.vae_policy_revision", workload.vae_policy_revision);
    out.string_field("workload.feature_digest", workload.feature_digest);
    out.begin_list("workload.token_shapes", workload.token_shapes.size());
    for (const auto &token : workload.token_shapes) {
        out.string_field("token.encoder", token.encoder);
        out.string_field(
            "token.tokenizer_revision", token.tokenizer_revision);
        out.string_field("token.template_revision", token.template_revision);
        out.unsigned_field("token.valid_rows", token.valid_rows);
        out.unsigned_field("token.padded_rows", token.padded_rows);
        out.unsigned_field("token.compute_rows", token.compute_rows);
    }
}

void validate_identity(const PresetSourceIdentity &source) {
    catalog_check(source.identity_version == 1 || source.identity_version == 2,
                  "unsupported source identity version");
    catalog_check(valid_identifier(source.model_variant),
                  "missing/invalid model variant");
    catalog_check(valid_identifier(source.weight_format),
                  "missing/invalid weight format");
    catalog_check(valid_digest(source.artifact_manifest_digest),
                  "invalid artifact manifest digest");
    if (source.identity_version == 1)
        catalog_check(valid_digest(source.source_snapshot_digest),
                      "invalid source snapshot digest");
    else
        catalog_check(source.source_snapshot_digest.empty(),
                      "portable source identity must not contain a snapshot");
}

void validate_identity(const PresetRuntimeIdentity &runtime) {
    catalog_check(valid_identifier(runtime.turbocider_build_id),
                  "invalid TurboCider build identity");
    catalog_check(valid_identifier(runtime.runtime_revision),
                  "invalid runtime revision");
    catalog_check(valid_identifier(runtime.adapter_revision),
                  "invalid adapter revision");
    catalog_check(valid_identifier(runtime.reader_revision),
                  "invalid reader revision");
    catalog_check(valid_identifier(runtime.kernel_revision),
                  "invalid kernel revision");
    catalog_check(valid_identifier(runtime.allocator_policy_revision),
                  "invalid allocator policy revision");
}

void validate_workload(const PresetWorkload &workload) {
    catalog_check(valid_identifier(workload.model), "invalid model identity");
    catalog_check(valid_identifier(workload.operation), "invalid operation");
    catalog_check(valid_identifier(workload.execution), "invalid execution");
    catalog_check(valid_identifier(workload.device_class),
                  "invalid device class");
    catalog_check(valid_identifier(workload.execution_container),
                  "invalid execution container");
    catalog_check(workload.width && workload.height && workload.frames &&
                      workload.steps && workload.batch,
                  "invalid workload dimensions");
    catalog_check(valid_identifier(workload.conditioning_revision),
                  "invalid conditioning revision");
    catalog_check(valid_identifier(workload.vae_policy_revision),
                  "invalid VAE policy revision");
    catalog_check(valid_digest(workload.feature_digest),
                  "invalid feature digest");
    catalog_check(workload.token_shapes.size() <= 8,
                  "too many token shapes");
    for (const auto &token : workload.token_shapes) {
        catalog_check(valid_identifier(token.encoder),
                      "invalid token encoder");
        catalog_check(valid_identifier(token.tokenizer_revision),
                      "invalid tokenizer revision");
        catalog_check(valid_identifier(token.template_revision),
                      "invalid template revision");
        catalog_check(token.valid_rows && token.padded_rows &&
                          token.compute_rows &&
                          token.valid_rows <= token.padded_rows &&
                          token.padded_rows <= token.compute_rows,
                      "invalid token shape");
    }
}

} // namespace

uint64_t streaming_target_margin_bytes(uint64_t target) {
    if (!supported_streaming_target(target))
        throw std::invalid_argument("streaming_target_unsupported");
    const uint64_t ten_percent = target / 10 + (target % 10 != 0);
    return std::max<uint64_t>(512ull << 20, ten_percent);
}

bool supported_streaming_target(uint64_t target) noexcept {
    return std::find(public_streaming_targets.begin(),
                     public_streaming_targets.end(), target) !=
        public_streaming_targets.end();
}

std::string canonical_streaming_preset_record(
        const StreamingPresetRecord &record) {
    validate_identity(record.source);
    CanonicalEncoder out(record.source.identity_version == 2
        ? "tc-streaming-preset-record-v2" : "tc-streaming-preset-record-v1");
    out.string_field("id", record.id);
    out.unsigned_field("revision", record.revision);
    out.string_field("catalog_revision", record.catalog_revision);
    encode_source(out, record.source);
    encode_workload(out, record.workload);
    encode_runtime(out, record.runtime);
    out.unsigned_field(
        "device.minimum_physical_memory_bytes",
        record.device.minimum_physical_memory_bytes);
    out.unsigned_field(
        "device.maximum_physical_memory_bytes",
        record.device.maximum_physical_memory_bytes);
    encode_config(out, record.plan.canonical_config);
    out.string_field("plan.layout_digest", record.plan.layout_digest);
    out.string_field(
        "plan.component_policy_revision",
        record.plan.component_policy_revision);
    out.string_field("plan.pass_transition", record.plan.pass_transition);
    out.string_field("plan.multi_pool_policy", record.plan.multi_pool_policy);
    out.boolean_field("calibration.complete", record.calibration.complete);
    out.unsigned_field(
        "calibration.calibrated_request_bytes",
        record.calibration.calibrated_request_bytes);
    out.string_field("calibration.scope", record.calibration.scope);
    out.string_field(
        "calibration.estimator_revision",
        record.calibration.estimator_revision);
    out.string_field(
        "calibration.calibration_id", record.calibration.calibration_id);
    out.string_field(
        "calibration.execution_container",
        record.calibration.execution_container);
    out.string_field(
        "calibration.evidence_digest", record.calibration.evidence_digest);
    out.unsigned_field(
        "calibration.confirmation_sample_count",
        record.calibration.confirmation_sample_count);
    out.unsigned_field(
        "calibration.maximum_sample_gap_ns",
        record.calibration.maximum_sample_gap_ns);
    out.unsigned_field("performance.rank", record.performance.rank);
    out.unsigned_field(
        "performance.logical_read_bytes",
        record.performance.logical_read_bytes);
    out.string_field("performance.profile_id", record.performance.profile_id);
    out.string_field(
        "performance.comparison_kind", record.performance.comparison_kind);
    out.string_field(
        "performance.confidence_status",
        record.performance.confidence_status);
    out.string_field(
        "performance.evidence_digest", record.performance.evidence_digest);
    out.string_field("release.channel", record.release.channel);
    out.boolean_field("release.revoked", record.release.revoked);
    out.string_field(
        "release.reviewed_commit", record.release.reviewed_commit);
    out.string_field("release.review_digest", record.release.review_digest);
    return out.bytes();
}

std::string streaming_preset_record_digest(
        const StreamingPresetRecord &record) {
    CanonicalEncoder out(record.source.identity_version == 2
        ? "tc-streaming-preset-record-digest-v2" : "tc-streaming-preset-record-digest-v1");
    out.string_field(
        "canonical_record", canonical_streaming_preset_record(record));
    return out.sha256();
}

StreamingPresetRecord finalize_streaming_preset_record(
        StreamingPresetRecord record) {
    record.canonical_record_digest = streaming_preset_record_digest(record);
    return record;
}

void validate_streaming_preset_record(
        const StreamingPresetRecord &record,
        std::string_view catalog_revision) {
    catalog_check(valid_identifier(record.id, 128), "invalid preset id");
    catalog_check(record.revision > 0, "invalid preset revision");
    catalog_check(record.catalog_revision == catalog_revision,
                  "record catalog revision mismatch");
    validate_identity(record.source);
    validate_workload(record.workload);
    validate_identity(record.runtime);
    catalog_check(record.device.minimum_physical_memory_bytes > 0,
                  "missing minimum physical memory");
    catalog_check(
        !record.device.maximum_physical_memory_bytes ||
            record.device.maximum_physical_memory_bytes >=
                record.device.minimum_physical_memory_bytes,
        "invalid physical memory range");
    validate_streaming_config(record.plan.canonical_config);
    catalog_check(record.plan.canonical_config.active(),
                  "canonical plan is disabled");
    catalog_check(valid_digest(record.plan.layout_digest),
                  "invalid layout digest");
    catalog_check(valid_identifier(record.plan.component_policy_revision),
                  "invalid component policy revision");
    catalog_check(valid_identifier(record.plan.pass_transition),
                  "invalid pass transition");
    catalog_check(valid_identifier(record.plan.multi_pool_policy),
                  "invalid multi-pool policy");
    catalog_check(record.calibration.complete,
                  "public record calibration is incomplete");
    catalog_check(record.calibration.calibrated_request_bytes > 0,
                  "missing calibrated request bytes");
    catalog_check(record.calibration.scope == "execution_process_tree_v1",
                  "unsupported calibration scope");
    catalog_check(valid_identifier(record.calibration.estimator_revision),
                  "invalid estimator revision");
    catalog_check(valid_identifier(record.calibration.calibration_id),
                  "invalid calibration id");
    catalog_check(record.calibration.execution_container ==
                      record.workload.execution_container,
                  "calibration container mismatch");
    catalog_check(valid_digest(record.calibration.evidence_digest),
                  "invalid calibration evidence digest");
    catalog_check(record.calibration.confirmation_sample_count > 0,
                  "missing calibration confirmation samples");
    catalog_check(valid_identifier(record.performance.profile_id),
                  "invalid performance profile");
    catalog_check(valid_identifier(record.performance.comparison_kind),
                  "invalid performance comparison kind");
    catalog_check(valid_identifier(record.performance.confidence_status),
                  "invalid performance confidence status");
    catalog_check(valid_digest(record.performance.evidence_digest),
                  "invalid performance evidence digest");
    catalog_check(record.release.channel == "public-stable" ||
                      record.release.channel == "public-experimental" ||
                      record.release.channel == "revoked",
                  "invalid release channel");
    catalog_check(valid_identifier(record.release.reviewed_commit),
                  "missing reviewed commit");
    catalog_check(valid_digest(record.release.review_digest),
                  "invalid review digest");
    catalog_check(valid_digest(record.canonical_record_digest),
                  "invalid canonical record digest");
    catalog_check(record.canonical_record_digest ==
                      streaming_preset_record_digest(record),
                  "canonical record digest mismatch");
}

static bool basic_workload_matches(const PresetWorkload &a,
                                   const PresetWorkload &b) {
    return std::tie(a.model, a.operation, a.execution, a.device_class,
                    a.execution_container, a.width, a.height, a.frames, a.fps,
                    a.steps, a.batch, a.audio, a.dynamic_text, a.approximation) ==
           std::tie(b.model, b.operation, b.execution, b.device_class,
                    b.execution_container, b.width, b.height, b.frames, b.fps,
                    b.steps, b.batch, b.audio, b.dynamic_text, b.approximation);
}

PresetWorkload basic_streaming_workload(const Request &request,
        std::string device_class, std::string execution_container) {
    PresetWorkload result;
    result.model = request.model;
    result.operation = request.operation;
    result.execution = request.execution;
    result.device_class = std::move(device_class);
    result.execution_container = std::move(execution_container);
    result.width = static_cast<uint32_t>(request.width);
    result.height = static_cast<uint32_t>(request.height);
    result.frames = static_cast<uint32_t>(request.frames);
    // Image adapters have no frame rate; the legacy request defaults to 24.
    result.fps = request.operation.starts_with("image.") ? 0 :
        static_cast<uint32_t>(request.fps);
    result.steps = static_cast<uint32_t>(request.steps);
    result.batch = 1;
    result.audio = request.audio;
    result.dynamic_text = request.dynamic_text;
    result.approximation = request.allow_approximation;
    return result;
}

static PresetResolution resolve_preset_impl(
        const PresetResolveQuery &query,
        const StreamingPresetCatalog &catalog, bool discovery) {
    PresetResolution result;
    if (!supported_streaming_target(query.target_request_memory_bytes)) {
        result.rejection_code = "unsupported_memory_target";
        return result;
    }
    if (query.catalog_revision && *query.catalog_revision != catalog.revision) {
        result.rejection_code = "streaming_resolution_stale";
        return result;
    }
    if (catalog.records.empty()) {
        result.rejection_code = "catalog_has_no_public_records";
        return result;
    }

    bool model_seen = false, workload_seen = false, source_seen = false;
    bool runtime_seen = false, device_seen = false, release_seen = false;
    bool preset_seen = false, calibration_seen = false;
    std::vector<const StreamingPresetRecord *> candidates;
    for (const auto &record : catalog.records) {
        validate_streaming_preset_record(record, catalog.revision);
        if (record.workload.model != query.workload.model)
            continue;
        model_seen = true;
        if (discovery ? !basic_workload_matches(record.workload, query.workload)
                      : !(record.workload == query.workload))
            continue;
        workload_seen = true;
        if (!discovery && query.require_exact_identity && !(record.source == query.source))
            continue;
        source_seen = true;
        if (!discovery && query.require_exact_identity && !(record.runtime == query.runtime))
            continue;
        runtime_seen = true;
        if (!device_memory_matches(record, query.physical_memory_bytes))
            continue;
        device_seen = true;
        if (record.release.revoked ||
            !released(record, query.allow_experimental))
            continue;
        release_seen = true;
        if (query.preset_id) {
            if (record.id != *query.preset_id || !query.preset_revision ||
                record.revision != *query.preset_revision)
                continue;
        }
        preset_seen = true;
        if (!record.calibration.complete) {
            result.rejected_preset_ids.push_back(record.id);
            continue;
        }
        calibration_seen = true;
        if (!fits(record, query.target_request_memory_bytes)) {
            result.rejected_preset_ids.push_back(record.id);
            continue;
        }
        candidates.push_back(&record);
    }

    if (candidates.empty()) {
        result.rejection_code = !model_seen ? "unsupported_model_or_format" :
            !workload_seen ? "unvalidated_workload" :
            !source_seen ? "artifact_verification_required" :
            !runtime_seen ? "streaming_resolution_stale" :
            !device_seen ? "unvalidated_device" :
            !release_seen ? "preset_not_public" :
            query.preset_id && !preset_seen ? "streaming_resolution_stale" :
            !calibration_seen ? "memory_calibration_incomplete" :
            "no_preset_fits_target";
        return result;
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto *left, const auto *right) {
        return std::tie(left->performance.rank,
                        left->calibration.calibrated_request_bytes,
                        left->performance.logical_read_bytes,
                        left->id, left->revision) <
               std::tie(right->performance.rank,
                        right->calibration.calibrated_request_bytes,
                        right->performance.logical_read_bytes,
                        right->id, right->revision);
    });
    result.selected = *candidates.front();
    return result;
}

PresetResolution resolve_streaming_preset(
        const PresetResolveQuery &query, const StreamingPresetCatalog &catalog) {
    return resolve_preset_impl(query, catalog, false);
}

PresetCandidateResolution find_streaming_preset_candidate(
        const PresetResolveQuery &query, const StreamingPresetCatalog &catalog) {
    auto result = resolve_preset_impl(query, catalog, true);
    return {std::move(result.selected), std::move(result.rejection_code)};
}

const StreamingPresetCatalog &production_streaming_preset_catalog() {
    // Deliberately empty until a model/workload/device record has passed the
    // full public acceptance protocol. Tests inject their own fixture catalog.
    static const StreamingPresetCatalog catalog{
        "tc-streaming-catalog-empty-v1", {}};
    return catalog;
}

} // namespace tc::streaming
