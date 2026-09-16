#include "memory_manifest.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

using namespace tc;

namespace {

constexpr const char *checkpoint =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr const char *evidence =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

AllocationSiteSpec site(std::string id, MemoryClass memory_class,
                        UpperProvenance provenance, bool aliasable = false) {
    AllocationSiteSpec result;
    result.site_id = std::move(id);
    result.component = "synthetic";
    result.stage = "test";
    result.memory_class = memory_class;
    result.lifetime = AllocationLifetime::Stage;
    result.provenance = provenance;
    result.aliasable = aliasable;
    return result;
}

MemoryManifest manifest() {
    MemoryManifest result;
    result.candidate_id = "synthetic_streamed_v1";
    result.checkpoint_digest = checkpoint;
    result.backend_revision = "synthetic-backend-v1";
    result.runtime_revision = "memory-runtime-v3";
    result.sites = {
        site("synthetic.activation", MemoryClass::Activation,
             UpperProvenance::ExactShapeFormula),
        site("synthetic.weights", MemoryClass::Weights,
             UpperProvenance::ExactMetadata, true),
    };
    result.instances = {
        {"synthetic.activation", 0, 20, 2, 3, 0, "activation.complete"},
        {"synthetic.weights", 0, 40, 1, 3, 7, "weights.complete"},
        {"synthetic.weights", 1, 30, 3, 4, 7, "weights.complete"},
    };
    return result;
}

MemoryCandidateKey key() {
    return {
        "synthetic_streamed_v1", "synthetic", checkpoint,
        "c_metal", "bf16", "test", "video.generate",
        "64x64x1.s1", "gpu_euler", 2, "none",
        "memory-runtime-v3", "AppleGPU-Test",
    };
}

template <typename Function>
void expect_rejected(Function function, const char *needle) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &error) {
        rejected = std::string(error.what()).find(needle) != std::string::npos;
    }
    assert(rejected);
}

void test_sha256_vectors() {
    assert(memory_sha256_hex("") ==
           "e3b0c44298fc1c149afbf4c8996fb924"
           "27ae41e4649b934ca495991b7852b855");
    assert(memory_sha256_hex("abc") ==
           "ba7816bf8f01cfea414140de5dae2223"
           "b00361a396177a9cb410ff61f20015ad");
}

void test_manifest_digest_is_order_independent() {
    auto first = manifest();
    auto reordered = first;
    std::reverse(reordered.sites.begin(), reordered.sites.end());
    std::reverse(reordered.instances.begin(), reordered.instances.end());
    assert(first.digest() == reordered.digest());
    reordered.instances.front().upper_bytes++;
    assert(first.digest() != reordered.digest());
}

void test_manifest_rejects_incomplete_or_unsafe_sites() {
    auto missing = manifest();
    missing.instances.erase(missing.instances.begin());
    expect_rejected([&] { missing.validate(); }, "missing required");

    auto heuristic = manifest();
    heuristic.sites.front().provenance =
        UpperProvenance::HeuristicNotExecutable;
    expect_rejected([&] { heuristic.validate(); }, "is heuristic");

    auto invalid_alias = manifest();
    invalid_alias.instances.front().alias_group = 9;
    expect_rejected([&] { invalid_alias.validate(); }, "non-aliasable");

    auto duplicate = manifest();
    duplicate.instances.push_back(duplicate.instances.back());
    expect_rejected([&] { duplicate.validate(); }, "duplicate allocation");

    auto overlap = manifest();
    overlap.instances[2].live_begin = 2;
    expect_rejected([&] { overlap.validate(); }, "alias lifetimes overlap");
}

void test_candidate_key_binds_schedule_fields() {
    const auto original = key();
    auto changed = original;
    changed.refill_slots = 3;
    assert(!(original == changed));
    assert(original.digest() != changed.digest());
    changed = original;
    changed.tiling_mode = "spatial_2x2";
    assert(original.digest() != changed.digest());
}

void test_registry_is_release_and_manifest_gated() {
    const auto test_manifest = manifest();
    const auto manifest_digest = test_manifest.digest();
    MemoryCapabilityRegistry registry;
    MemoryCapabilityRecord experimental;
    experimental.key = key();
    experimental.level = MemoryCapabilityLevel::EnvelopeValidated;
    experimental.state = MemoryCertificationState::ExperimentalGuarded;
    experimental.manifest_digest = manifest_digest;
    experimental.framework_upper_bytes = 5;
    experimental.framework_provenance = "synthetic-envelope-v1";
    experimental.maximum_validated_upper_bytes = 75;
    registry.add(experimental);
    assert(!registry.lookup(key(), manifest_digest).matched);
    assert(registry.lookup(key(), manifest_digest, true).matched);
    assert(!registry.lookup(key(), std::string(64, 'c'), true).matched);

    MemoryCapabilityRegistry missing_framework;
    auto invalid_framework = experimental;
    invalid_framework.framework_upper_bytes = 0;
    expect_rejected(
        [&] { missing_framework.add(invalid_framework); },
        "framework upper is zero");

    MemoryCapabilityRegistry implicit_epochs;
    auto invalid_epochs = experimental;
    invalid_epochs.require_explicit_epoch = false;
    expect_rejected(
        [&] { implicit_epochs.add(invalid_epochs); },
        "must require explicit epochs");

    MemoryCapabilityRegistry missing_schedule;
    auto invalid_schedule = experimental;
    invalid_schedule.require_explicit_schedule = true;
    expect_rejected(
        [&] { missing_schedule.add(invalid_schedule); },
        "explicit schedule is absent");

    MemoryCapabilityRegistry unbound_schedule;
    invalid_schedule.require_explicit_schedule = false;
    invalid_schedule.schedule.push_back({});
    expect_rejected(
        [&] { unbound_schedule.add(invalid_schedule); },
        "schedule bindings are not required");

    auto certified_key = key();
    certified_key.device_family = "AppleGPU-Release";
    MemoryCapabilityRecord certified;
    certified.key = certified_key;
    certified.level = MemoryCapabilityLevel::L3Certified;
    certified.state = MemoryCertificationState::Certified;
    certified.manifest_digest = manifest_digest;
    certified.evidence_digest = evidence;
    certified.framework_upper_bytes = 5;
    certified.framework_provenance = "synthetic-envelope-v1";
    certified.maximum_validated_upper_bytes = 75;
    certified.release_enabled = true;
    registry.add(certified);
    const auto match = registry.lookup(certified_key, manifest_digest);
    assert(match.matched && match.record);
    assert(match.record->state == MemoryCertificationState::Certified);

    expect_rejected([&] { registry.add(certified); }, "duplicate capability");
}

void test_manifest_builder_validates_and_freezes() {
    MemoryManifestBuilder builder(
        "synthetic_builder_v1", checkpoint, "synthetic-backend-v1",
        "memory-runtime-v3");
    builder.add_site(site("builder.weights", MemoryClass::Weights,
                          UpperProvenance::ExactMetadata, true));
    builder.add_site(site("builder.activation", MemoryClass::Activation,
                          UpperProvenance::ExactShapeFormula));
    builder.add_instance({"builder.weights", 0, 12, 0, 2, 4,
                          "weights.complete"});
    builder.add_instance({"builder.activation", 0, 8, 1, 2, 0,
                          "activation.complete"});
    auto value = builder.build();
    assert(value.candidate_id == "synthetic_builder_v1");
    assert(value.digest().size() == 64);
    expect_rejected(
        [&] { builder.add_site(site("builder.late", MemoryClass::Output,
                                    UpperProvenance::ExactShapeFormula)); },
        "already finalized");
    expect_rejected([&] { (void)builder.build(); }, "already finalized");
}

void test_manifest_builder_rejects_duplicate_site_before_build() {
    MemoryManifestBuilder builder(
        "synthetic_duplicate_v1", checkpoint, "synthetic-backend-v1",
        "memory-runtime-v3");
    builder.add_site(site("builder.duplicate", MemoryClass::Weights,
                          UpperProvenance::ExactMetadata));
    expect_rejected(
        [&] { builder.add_site(site("builder.duplicate", MemoryClass::Weights,
                                    UpperProvenance::ExactMetadata)); },
        "duplicate allocation site");
}

} // namespace

int main() {
    test_sha256_vectors();
    test_manifest_digest_is_order_independent();
    test_manifest_rejects_incomplete_or_unsafe_sites();
    test_candidate_key_binds_schedule_fields();
    test_registry_is_release_and_manifest_gated();
    test_manifest_builder_validates_and_freezes();
    test_manifest_builder_rejects_duplicate_site_before_build();
    std::cout << "memory manifest tests passed\n";
    return 0;
}
