#include "bridge.hpp"

#ifdef TURBOCIDER_ENABLE_TEST_HOOKS

#include "../../runtime/streaming/preset_catalog.hpp"

#include <cmath>
#include <limits>
#include <set>

namespace tc {
namespace {

NSDictionary *object_value(id value, const std::string &path) {
    require([value isKindOfClass:NSDictionary.class], path + " must be an object");
    return value;
}

NSArray *array_value(id value, const std::string &path) {
    require([value isKindOfClass:NSArray.class], path + " must be an array");
    return value;
}

void exact_keys(NSDictionary *value, NSArray<NSString *> *expected,
                const std::string &path) {
    require(value.count == expected.count,
            path + " has missing or unknown fields");
    NSSet *allowed = [NSSet setWithArray:expected];
    for (id raw_key in value) {
        require([raw_key isKindOfClass:NSString.class],
                path + " contains a non-string key");
        require([allowed containsObject:raw_key],
                path + " contains unknown field: " +
                    std::string([raw_key UTF8String]));
    }
}

std::string required_string(NSDictionary *value, NSString *key,
                            const std::string &path) {
    require(value[key] != nil, path + " is required");
    return string_value(value, key);
}

uint64_t required_unsigned(NSDictionary *value, NSString *key,
                           const std::string &path,
                           uint64_t maximum = (1ull << 53) - 1) {
    id raw = value[key];
    require(raw != nil && [raw isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)raw) != CFBooleanGetTypeID(),
            path + " must be an unsigned integer");
    const double number = [raw doubleValue];
    require(std::isfinite(number) && number >= 0 &&
                number <= static_cast<double>(maximum) &&
                number == std::floor(number),
            path + " is outside the supported integer range");
    const uint64_t result = [raw unsignedLongLongValue];
    require(static_cast<double>(result) == number,
            path + " cannot be represented exactly");
    return result;
}

uint32_t required_u32(NSDictionary *value, NSString *key,
                      const std::string &path) {
    return static_cast<uint32_t>(required_unsigned(
        value, key, path, std::numeric_limits<uint32_t>::max()));
}

bool required_bool(NSDictionary *value, NSString *key,
                   const std::string &path) {
    id raw = value[key];
    require(raw != nil &&
                CFGetTypeID((__bridge CFTypeRef)raw) == CFBooleanGetTypeID(),
            path + " must be boolean");
    return [raw boolValue];
}

streaming::PresetSourceIdentity parse_source(NSDictionary *value,
                                             const std::string &path) {
    const uint32_t version = value[@"identity_version"]
        ? required_u32(value, @"identity_version", path + ".identity_version") : 1;
    require(version == 1 || version == 2, path + " unsupported source identity version");
    NSMutableArray *keys = [@[@"model_variant", @"weight_format",
                              @"artifact_manifest_digest"] mutableCopy];
    if (version == 1) [keys addObject:@"source_snapshot_digest"];
    if (value[@"identity_version"]) [keys addObject:@"identity_version"];
    exact_keys(value, keys, path);
    return {
        required_string(value, @"model_variant", path + ".model_variant"),
        required_string(value, @"weight_format", path + ".weight_format"),
        required_string(value, @"artifact_manifest_digest",
                        path + ".artifact_manifest_digest"),
        version == 1 ? required_string(value, @"source_snapshot_digest",
                        path + ".source_snapshot_digest") : std::string{},
        version,
    };
}

streaming::PresetRuntimeIdentity parse_runtime(NSDictionary *value,
                                               const std::string &path) {
    exact_keys(value, @[
        @"turbocider_build_id", @"runtime_revision", @"adapter_revision",
        @"reader_revision", @"kernel_revision",
        @"allocator_policy_revision"
    ], path);
    return {
        required_string(value, @"turbocider_build_id",
                        path + ".turbocider_build_id"),
        required_string(value, @"runtime_revision",
                        path + ".runtime_revision"),
        required_string(value, @"adapter_revision",
                        path + ".adapter_revision"),
        required_string(value, @"reader_revision",
                        path + ".reader_revision"),
        required_string(value, @"kernel_revision",
                        path + ".kernel_revision"),
        required_string(value, @"allocator_policy_revision",
                        path + ".allocator_policy_revision"),
    };
}

streaming::PresetWorkload parse_workload(NSDictionary *value,
                                         const std::string &path) {
    exact_keys(value, @[
        @"model", @"operation", @"execution", @"device_class",
        @"execution_container", @"width", @"height", @"frames", @"fps",
        @"steps", @"batch", @"audio", @"dynamic_text", @"approximation",
        @"conditioning_revision", @"vae_policy_revision", @"feature_digest",
        @"token_shapes"
    ], path);
    streaming::PresetWorkload result;
    result.model = required_string(value, @"model", path + ".model");
    result.operation = required_string(
        value, @"operation", path + ".operation");
    result.execution = required_string(
        value, @"execution", path + ".execution");
    result.device_class = required_string(
        value, @"device_class", path + ".device_class");
    result.execution_container = required_string(
        value, @"execution_container", path + ".execution_container");
    result.width = required_u32(value, @"width", path + ".width");
    result.height = required_u32(value, @"height", path + ".height");
    result.frames = required_u32(value, @"frames", path + ".frames");
    result.fps = required_u32(value, @"fps", path + ".fps");
    result.steps = required_u32(value, @"steps", path + ".steps");
    result.batch = required_u32(value, @"batch", path + ".batch");
    result.audio = required_bool(value, @"audio", path + ".audio");
    result.dynamic_text = required_bool(
        value, @"dynamic_text", path + ".dynamic_text");
    result.approximation = required_bool(
        value, @"approximation", path + ".approximation");
    result.conditioning_revision = required_string(
        value, @"conditioning_revision", path + ".conditioning_revision");
    result.vae_policy_revision = required_string(
        value, @"vae_policy_revision", path + ".vae_policy_revision");
    result.feature_digest = required_string(
        value, @"feature_digest", path + ".feature_digest");
    NSArray *tokens = array_value(value[@"token_shapes"],
                                  path + ".token_shapes");
    require(tokens.count <= 8, path + ".token_shapes exceeds 8");
    for (NSUInteger index = 0; index < tokens.count; ++index) {
        const std::string token_path = path + ".token_shapes[" +
            std::to_string(index) + "]";
        NSDictionary *token = object_value(tokens[index], token_path);
        exact_keys(token, @[
            @"encoder", @"tokenizer_revision", @"template_revision",
            @"valid_rows", @"padded_rows", @"compute_rows"
        ], token_path);
        result.token_shapes.push_back({
            required_string(token, @"encoder", token_path + ".encoder"),
            required_string(token, @"tokenizer_revision",
                            token_path + ".tokenizer_revision"),
            required_string(token, @"template_revision",
                            token_path + ".template_revision"),
            required_u32(token, @"valid_rows", token_path + ".valid_rows"),
            required_u32(token, @"padded_rows", token_path + ".padded_rows"),
            required_u32(token, @"compute_rows", token_path + ".compute_rows"),
        });
    }
    return result;
}

StreamingConfig parse_config(NSDictionary *value, const std::string &path) {
    exact_keys(value, @[
        @"enabled", @"schema_version", @"selection", @"retention", @"stages"
    ], path);
    NSDictionary *stages = object_value(value[@"stages"], path + ".stages");
    for (NSString *stage_id in stages) {
        const std::string stage_path = path + ".stages." +
            std::string(stage_id.UTF8String);
        exact_keys(object_value(stages[stage_id], stage_path), @[
            @"residency", @"block_group_size", @"slot_count",
            @"resident_prefix_blocks", @"prefetch_distance", @"io_workers"
        ], stage_path);
    }
    StreamingConfig result;
    parse_streaming_config(value, result, "test_catalog");
    validate_streaming_config(result);
    return result;
}

streaming::StreamingPresetRecord parse_record(NSDictionary *value,
                                              const std::string &path) {
    exact_keys(value, @[
        @"id", @"revision", @"catalog_revision", @"source", @"workload",
        @"runtime", @"device", @"plan", @"calibration", @"performance",
        @"release", @"canonical_record_digest"
    ], path);
    streaming::StreamingPresetRecord result;
    result.id = required_string(value, @"id", path + ".id");
    result.revision = required_u32(value, @"revision", path + ".revision");
    result.catalog_revision = required_string(
        value, @"catalog_revision", path + ".catalog_revision");
    result.source = parse_source(
        object_value(value[@"source"], path + ".source"), path + ".source");
    result.workload = parse_workload(
        object_value(value[@"workload"], path + ".workload"),
        path + ".workload");
    result.runtime = parse_runtime(
        object_value(value[@"runtime"], path + ".runtime"), path + ".runtime");

    NSDictionary *device = object_value(value[@"device"], path + ".device");
    exact_keys(device, @[
        @"minimum_physical_memory_bytes", @"maximum_physical_memory_bytes"
    ], path + ".device");
    result.device.minimum_physical_memory_bytes = required_unsigned(
        device, @"minimum_physical_memory_bytes",
        path + ".device.minimum_physical_memory_bytes");
    result.device.maximum_physical_memory_bytes = required_unsigned(
        device, @"maximum_physical_memory_bytes",
        path + ".device.maximum_physical_memory_bytes");

    NSDictionary *plan = object_value(value[@"plan"], path + ".plan");
    exact_keys(plan, @[
        @"canonical_config", @"layout_digest", @"component_policy_revision",
        @"pass_transition", @"multi_pool_policy"
    ], path + ".plan");
    result.plan.canonical_config = parse_config(
        object_value(plan[@"canonical_config"], path + ".plan.canonical_config"),
        path + ".plan.canonical_config");
    result.plan.layout_digest = required_string(
        plan, @"layout_digest", path + ".plan.layout_digest");
    result.plan.component_policy_revision = required_string(
        plan, @"component_policy_revision",
        path + ".plan.component_policy_revision");
    result.plan.pass_transition = required_string(
        plan, @"pass_transition", path + ".plan.pass_transition");
    result.plan.multi_pool_policy = required_string(
        plan, @"multi_pool_policy", path + ".plan.multi_pool_policy");

    NSDictionary *calibration = object_value(
        value[@"calibration"], path + ".calibration");
    exact_keys(calibration, @[
        @"complete", @"calibrated_request_bytes", @"scope",
        @"estimator_revision", @"calibration_id", @"execution_container",
        @"evidence_digest", @"confirmation_sample_count",
        @"maximum_sample_gap_ns"
    ], path + ".calibration");
    result.calibration.complete = required_bool(
        calibration, @"complete", path + ".calibration.complete");
    result.calibration.calibrated_request_bytes = required_unsigned(
        calibration, @"calibrated_request_bytes",
        path + ".calibration.calibrated_request_bytes");
    result.calibration.scope = required_string(
        calibration, @"scope", path + ".calibration.scope");
    result.calibration.estimator_revision = required_string(
        calibration, @"estimator_revision",
        path + ".calibration.estimator_revision");
    result.calibration.calibration_id = required_string(
        calibration, @"calibration_id", path + ".calibration.calibration_id");
    result.calibration.execution_container = required_string(
        calibration, @"execution_container",
        path + ".calibration.execution_container");
    result.calibration.evidence_digest = required_string(
        calibration, @"evidence_digest", path + ".calibration.evidence_digest");
    result.calibration.confirmation_sample_count = required_unsigned(
        calibration, @"confirmation_sample_count",
        path + ".calibration.confirmation_sample_count");
    result.calibration.maximum_sample_gap_ns = required_unsigned(
        calibration, @"maximum_sample_gap_ns",
        path + ".calibration.maximum_sample_gap_ns");

    NSDictionary *performance = object_value(
        value[@"performance"], path + ".performance");
    exact_keys(performance, @[
        @"rank", @"logical_read_bytes", @"profile_id", @"comparison_kind",
        @"confidence_status", @"evidence_digest"
    ], path + ".performance");
    result.performance.rank = required_u32(
        performance, @"rank", path + ".performance.rank");
    result.performance.logical_read_bytes = required_unsigned(
        performance, @"logical_read_bytes",
        path + ".performance.logical_read_bytes");
    result.performance.profile_id = required_string(
        performance, @"profile_id", path + ".performance.profile_id");
    result.performance.comparison_kind = required_string(
        performance, @"comparison_kind",
        path + ".performance.comparison_kind");
    result.performance.confidence_status = required_string(
        performance, @"confidence_status",
        path + ".performance.confidence_status");
    result.performance.evidence_digest = required_string(
        performance, @"evidence_digest",
        path + ".performance.evidence_digest");

    NSDictionary *release = object_value(value[@"release"], path + ".release");
    exact_keys(release, release[@"policy_revision"] ? @[
        @"channel", @"revoked", @"reviewed_commit", @"review_digest", @"policy_revision"
    ] : @[@"channel", @"revoked", @"reviewed_commit", @"review_digest"], path + ".release");
    if (release[@"policy_revision"])
        result.release.policy_revision = required_string(release, @"policy_revision", path + ".release.policy_revision");
    result.release.channel = required_string(
        release, @"channel", path + ".release.channel");
    result.release.revoked = required_bool(
        release, @"revoked", path + ".release.revoked");
    result.release.reviewed_commit = required_string(
        release, @"reviewed_commit", path + ".release.reviewed_commit");
    result.release.review_digest = required_string(
        release, @"review_digest", path + ".release.review_digest");
    result.canonical_record_digest = required_string(
        value, @"canonical_record_digest", path + ".canonical_record_digest");
    return result;
}

} // namespace

std::shared_ptr<const streaming::StreamingPresetCatalog>
parse_test_streaming_catalog_json(const char *json_value) {
    NSDictionary *root = parse_json(json_value);
    exact_keys(root, @[@"schema", @"revision", @"records"], "test_catalog");
    require(required_string(root, @"schema", "test_catalog.schema") ==
                "turbocider-streaming-test-catalog-v1",
            "test_catalog.schema is unsupported");
    streaming::StreamingPresetCatalog catalog;
    catalog.revision = required_string(
        root, @"revision", "test_catalog.revision");
    require(!catalog.revision.empty() && catalog.revision.size() <= 256,
            "test_catalog.revision is invalid");
    NSArray *records = array_value(root[@"records"], "test_catalog.records");
    require(records.count > 0 && records.count <= 64,
            "test_catalog.records must contain 1...64 records");
    std::set<std::string> ids;
    std::set<std::string> digests;
    for (NSUInteger index = 0; index < records.count; ++index) {
        const std::string path = "test_catalog.records[" +
            std::to_string(index) + "]";
        auto record = parse_record(
            object_value(records[index], path), path);
        streaming::validate_streaming_preset_record(
            record, catalog.revision);
        require(ids.insert(record.id).second,
                "test_catalog contains duplicate preset id");
        require(digests.insert(record.canonical_record_digest).second,
                "test_catalog contains duplicate record digest");
        catalog.records.push_back(std::move(record));
    }
    return std::make_shared<const streaming::StreamingPresetCatalog>(
        std::move(catalog));
}

NSDictionary *test_streaming_catalog_record_dictionary(
        const streaming::StreamingPresetRecord &record) {
    NSMutableArray *tokens = [NSMutableArray array];
    for (const auto &token : record.workload.token_shapes) {
        [tokens addObject:@{
            @"encoder": @(token.encoder.c_str()),
            @"tokenizer_revision": @(token.tokenizer_revision.c_str()),
            @"template_revision": @(token.template_revision.c_str()),
            @"valid_rows": @(token.valid_rows),
            @"padded_rows": @(token.padded_rows),
            @"compute_rows": @(token.compute_rows),
        }];
    }
    NSMutableDictionary *workload = [@{
        @"model": @(record.workload.model.c_str()),
        @"operation": @(record.workload.operation.c_str()),
        @"execution": @(record.workload.execution.c_str()),
        @"device_class": @(record.workload.device_class.c_str()),
        @"execution_container":
            @(record.workload.execution_container.c_str()),
        @"width": @(record.workload.width),
        @"height": @(record.workload.height),
        @"frames": @(record.workload.frames),
        @"fps": @(record.workload.fps),
        @"steps": @(record.workload.steps),
        @"batch": @(record.workload.batch),
        @"audio": @(record.workload.audio),
        @"dynamic_text": @(record.workload.dynamic_text),
        @"approximation": @(record.workload.approximation),
        @"conditioning_revision":
            @(record.workload.conditioning_revision.c_str()),
        @"vae_policy_revision":
            @(record.workload.vae_policy_revision.c_str()),
        @"feature_digest": @(record.workload.feature_digest.c_str()),
        @"token_shapes": tokens,
    } mutableCopy];

    NSMutableDictionary *source = [@{
        @"model_variant": @(record.source.model_variant.c_str()),
        @"weight_format": @(record.source.weight_format.c_str()),
        @"artifact_manifest_digest":
            @(record.source.artifact_manifest_digest.c_str()),
    } mutableCopy];
    if (record.source.identity_version == 1)
        source[@"source_snapshot_digest"] = @(record.source.source_snapshot_digest.c_str());
    else
        source[@"identity_version"] = @(record.source.identity_version);
    NSDictionary *runtime = @{
        @"turbocider_build_id":
            @(record.runtime.turbocider_build_id.c_str()),
        @"runtime_revision": @(record.runtime.runtime_revision.c_str()),
        @"adapter_revision": @(record.runtime.adapter_revision.c_str()),
        @"reader_revision": @(record.runtime.reader_revision.c_str()),
        @"kernel_revision": @(record.runtime.kernel_revision.c_str()),
        @"allocator_policy_revision":
            @(record.runtime.allocator_policy_revision.c_str()),
    };
    NSDictionary *device = @{
        @"minimum_physical_memory_bytes":
            @(record.device.minimum_physical_memory_bytes),
        @"maximum_physical_memory_bytes":
            @(record.device.maximum_physical_memory_bytes),
    };
    NSDictionary *plan = @{
        @"canonical_config":
            streaming_config_dictionary(record.plan.canonical_config),
        @"layout_digest": @(record.plan.layout_digest.c_str()),
        @"component_policy_revision":
            @(record.plan.component_policy_revision.c_str()),
        @"pass_transition": @(record.plan.pass_transition.c_str()),
        @"multi_pool_policy": @(record.plan.multi_pool_policy.c_str()),
    };
    NSDictionary *calibration = @{
        @"complete": @(record.calibration.complete),
        @"calibrated_request_bytes":
            @(record.calibration.calibrated_request_bytes),
        @"scope": @(record.calibration.scope.c_str()),
        @"estimator_revision":
            @(record.calibration.estimator_revision.c_str()),
        @"calibration_id": @(record.calibration.calibration_id.c_str()),
        @"execution_container":
            @(record.calibration.execution_container.c_str()),
        @"evidence_digest": @(record.calibration.evidence_digest.c_str()),
        @"confirmation_sample_count":
            @(record.calibration.confirmation_sample_count),
        @"maximum_sample_gap_ns":
            @(record.calibration.maximum_sample_gap_ns),
    };
    NSDictionary *performance = @{
        @"rank": @(record.performance.rank),
        @"logical_read_bytes": @(record.performance.logical_read_bytes),
        @"profile_id": @(record.performance.profile_id.c_str()),
        @"comparison_kind": @(record.performance.comparison_kind.c_str()),
        @"confidence_status":
            @(record.performance.confidence_status.c_str()),
        @"evidence_digest": @(record.performance.evidence_digest.c_str()),
    };
    NSMutableDictionary *release = [@{
        @"channel": @(record.release.channel.c_str()),
        @"revoked": @(record.release.revoked),
        @"reviewed_commit": @(record.release.reviewed_commit.c_str()),
        @"review_digest": @(record.release.review_digest.c_str()),
    } mutableCopy];
    if (!record.release.policy_revision.empty())
        release[@"policy_revision"] = @(record.release.policy_revision.c_str());
    return @{
        @"id": @(record.id.c_str()),
        @"revision": @(record.revision),
        @"catalog_revision": @(record.catalog_revision.c_str()),
        @"source": source,
        @"workload": workload,
        @"runtime": runtime,
        @"device": device,
        @"plan": plan,
        @"calibration": calibration,
        @"performance": performance,
        @"release": release,
        @"canonical_record_digest":
            @(record.canonical_record_digest.c_str()),
    };
}

} // namespace tc

#endif
