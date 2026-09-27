#include "memory_probe.hpp"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>

using namespace tc;

namespace {

struct Fixture {
    std::filesystem::path root;
    Request request;
    ExecutionPlan plan;
    MemoryDeviceIdentity device;
    std::unique_ptr<MemoryCheckpointHashCache> hash_cache;
    std::string checkpoint_digest;
};

Fixture make_fixture() {
    Fixture fixture;
    static uint64_t ordinal = 0;
    fixture.root = std::filesystem::temp_directory_path() /
        ("turbocider-memory-probe-" + std::to_string(getpid()) + "-" +
         std::to_string(++ordinal));
    assert(std::filesystem::create_directories(fixture.root));
    fixture.hash_cache = std::make_unique<MemoryCheckpointHashCache>();
    std::filesystem::create_directories(
        fixture.root / "checkpoint");
    std::filesystem::create_directories(
        fixture.root / "memory-capabilities" / "synthetic_streamed_v1");
    {
        std::ofstream checkpoint(fixture.root / "checkpoint" / "shard-0.bin",
                                 std::ios::binary);
        checkpoint << "synthetic-checkpoint-v1";
    }

    const auto checkpoint_path = fixture.root / "checkpoint" / "shard-0.bin";
    const auto file_sha = fixture.hash_cache->sha256(
        checkpoint_path, MemoryModelRootTrust::OwnedImmutable);
    fixture.checkpoint_digest = memory_checkpoint_identity_digest({
        {"checkpoint/shard-0.bin", 23, file_sha}});

    fixture.request.model = "synthetic";
    fixture.request.operation = "video.generate";
    fixture.request.width = 64;
    fixture.request.height = 64;
    fixture.request.frames = 1;
    fixture.request.steps = 1;
    fixture.request.audio = false;
    fixture.request.memory_constrained.enabled = true;
    fixture.request.memory_constrained.specified_fields =
        MemoryFieldEnabled | MemoryFieldLimit;
    fixture.request.memory_constrained.limit_bytes = 1024;
    fixture.request.memory_constrained.refill_slots = 2;

    EffectiveMemoryPolicy policy;
    policy.enabled = true;
    policy.adapter_candidate = "synthetic_streamed_v1";
    policy.candidate_backend = "c_metal";
    policy.candidate_dtype = "bf16";
    policy.candidate_model_variant = "synthetic";
    policy.candidate_sampler_mode = "gpu_euler";
    policy.candidate_tiling_mode = "none";
    policy.refill_slots = 2;
    policy.effective_budget_bytes = 1024;
    fixture.plan.request = fixture.request;
    fixture.plan.memory_policy = policy;
    fixture.device = {"AppleGPU-Test", "memory-runtime-v3"};
    return fixture;
}

std::filesystem::path sidecar_path(const Fixture &fixture) {
    return memory_capability_probe_path(
        fixture.root, "synthetic_streamed_v1", fixture.request, 2);
}

void write_sidecar(const Fixture &fixture, bool wrong_hash = false,
                   bool wrong_key = false) {
    const auto path = sidecar_path(fixture);
    const auto checkpoint_digest = wrong_hash ? std::string(64, 'b')
                                              : fixture.checkpoint_digest;
    const auto adapter = wrong_key ? "other_adapter" : "synthetic_streamed_v1";
    const std::string json =
        std::string("{") +
        "\"schema\":\"turbocider.memory_probe.v1\","
        "\"key\":{"
        "\"adapter\":\"" + adapter + "\","
        "\"model_id\":\"synthetic\","
        "\"checkpoint_digest\":\"" + checkpoint_digest + "\","
        "\"backend\":\"c_metal\","
        "\"dtype\":\"bf16\","
        "\"model_variant\":\"synthetic\","
        "\"operation\":\"video.generate\","
        "\"shape_bucket\":\"w64_h64_f1_s1_audio0_input0_lora0\","
        "\"sampler_mode\":\"gpu_euler\","
        "\"refill_slots\":2,"
        "\"tiling_mode\":\"none\","
        "\"runtime_revision\":\"memory-runtime-v3\","
        "\"device_family\":\"AppleGPU-Test\"},"
        "\"checkpoint_files\":[{"
        "\"logical_name\":\"checkpoint/shard-0.bin\","
        "\"size_bytes\":23,"
        "\"sha256\":\"" +
        fixture.hash_cache->sha256(
            fixture.root / "checkpoint" / "shard-0.bin",
            MemoryModelRootTrust::OwnedImmutable) +
        "\"}],"
        "\"manifest\":{"
        "\"schema\":\"turbocider.memory_manifest.v1\","
        "\"candidate_id\":\"synthetic_streamed_v1\","
        "\"checkpoint_digest\":\"" + fixture.checkpoint_digest + "\","
        "\"backend_revision\":\"synthetic-backend-v1\","
        "\"runtime_revision\":\"memory-runtime-v3\","
        "\"sites\":[{"
        "\"site_id\":\"synthetic.activation\","
        "\"component\":\"synthetic\","
        "\"stage\":\"denoise\","
        "\"memory_class\":\"activation\","
        "\"lifetime\":\"stage\","
        "\"provenance\":\"exact_shape_formula\","
        "\"required\":true}],"
        "\"instances\":[{"
        "\"site_id\":\"synthetic.activation\","
        "\"instance_id\":0,"
        "\"upper_bytes\":16,"
        "\"live_begin\":1,"
        "\"live_end\":2,"
        "\"last_use_event\":\"synthetic.activation.complete\"}]"
        "}}";
    std::ofstream output(path, std::ios::binary);
    output << json;
    assert(output.good());
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

void test_missing_sidecar_is_nullopt() {
    auto fixture = make_fixture();
    auto probe = load_memory_capability_probe(
        fixture.root, sidecar_path(fixture), fixture.plan, fixture.device,
        MemoryModelRootTrust::OwnedImmutable, *fixture.hash_cache);
    assert(!probe.has_value());
}

void test_valid_sidecar_is_metadata_only_and_deterministic() {
    auto fixture = make_fixture();
    write_sidecar(fixture);
    const auto path = sidecar_path(fixture);
    auto first = load_memory_capability_probe(
        fixture.root, path, fixture.plan, fixture.device,
        MemoryModelRootTrust::OwnedImmutable, *fixture.hash_cache);
    auto second = load_memory_capability_probe(
        fixture.root, path, fixture.plan, fixture.device,
        MemoryModelRootTrust::OwnedImmutable, *fixture.hash_cache);
    assert(first.has_value() && second.has_value());
    assert(first->key.digest() == second->key.digest());
    assert(first->manifest.digest() == second->manifest.digest());
    assert(first->manifest.instances.size() == 1);
}

void test_checkpoint_and_identity_mismatch_fail_closed() {
    auto fixture = make_fixture();
    write_sidecar(fixture, true, false);
    expect_rejected([&] {
        (void)load_memory_capability_probe(
            fixture.root, sidecar_path(fixture), fixture.plan, fixture.device,
            MemoryModelRootTrust::OwnedImmutable, *fixture.hash_cache);
    }, "canonical checkpoint digest mismatch");

    write_sidecar(fixture, false, true);
    expect_rejected([&] {
        (void)load_memory_capability_probe(
            fixture.root, sidecar_path(fixture), fixture.plan, fixture.device,
            MemoryModelRootTrust::OwnedImmutable, *fixture.hash_cache);
    }, "capability sidecar identity mismatch");
}

void test_sidecar_path_cannot_escape_root() {
    auto fixture = make_fixture();
    const auto outside = fixture.root.parent_path() / "outside-probe.json";
    expect_rejected([&] {
        (void)load_memory_capability_probe(
            fixture.root, outside, fixture.plan, fixture.device,
            MemoryModelRootTrust::OwnedImmutable, *fixture.hash_cache);
    }, "sidecar escapes model root");
}

void test_hash_cache_invalidates_same_size_file_change() {
    auto fixture = make_fixture();
    const auto checkpoint = fixture.root / "checkpoint" / "shard-0.bin";
    const auto before = fixture.hash_cache->sha256(
        checkpoint, MemoryModelRootTrust::OwnedImmutable);
    {
        std::ofstream changed(checkpoint, std::ios::binary | std::ios::trunc);
        changed << "synthetic-checkpoint-v2";
        assert(changed.good());
    }
    assert(std::filesystem::file_size(checkpoint) == 23);
    const auto after = fixture.hash_cache->sha256(
        checkpoint, MemoryModelRootTrust::OwnedImmutable);
    assert(before != after);
}

} // namespace

int main() {
    test_missing_sidecar_is_nullopt();
    test_valid_sidecar_is_metadata_only_and_deterministic();
    test_checkpoint_and_identity_mismatch_fail_closed();
    test_sidecar_path_cannot_escape_root();
    test_hash_cache_invalidates_same_size_file_change();
    std::cout << "memory probe tests passed\n";
    return 0;
}
