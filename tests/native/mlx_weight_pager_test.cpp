#include "../../native/runtime/streaming/mlx_weight_pager.hpp"

#include <atomic>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace tc {

// Keep this focused test independent from the full backend translation unit.
// These are the only Weights operations used by MlxWeightPager and the test.
void Weights::bind_arrays(const std::vector<std::string> &keys,
                          const std::vector<Tensor> &arrays, size_t offset) {
    if (offset > arrays.size() || keys.size() > arrays.size() - offset)
        throw std::invalid_argument("weight binding array range is invalid");
    values_.clear();
    runtime_loras_.clear();
    values_.reserve(keys.size());
    for (size_t index = 0; index < keys.size(); ++index)
        values_.emplace(keys[index], arrays[offset + index]);
}

const Tensor &Weights::at(const std::string &key) const {
    const auto found = values_.find(key);
    if (found == values_.end())
        throw std::out_of_range("missing test weight: " + key);
    return found->second;
}

bool Weights::has(const std::string &key) const {
    return values_.contains(key);
}

} // namespace tc

namespace {

using tc::streaming::BlockSpec;
using tc::streaming::Descriptor;
using tc::streaming::FieldSpec;
using tc::streaming::Group;
using tc::streaming::Materialization;
using tc::streaming::MlxWeightPager;
using tc::streaming::PoolLayout;
using tc::streaming::SlotLayout;
using tc::streaming::SourceArtifact;
using tc::streaming::SourceFileIdentity;
using tc::streaming::SourceIdentityKind;
using tc::streaming::SourceRange;
using tc::streaming::StageDescriptor;
using tc::streaming::StageLayout;

constexpr const char *kArtifact0 = "fixture-00001.safetensors";
constexpr const char *kArtifact1 = "fixture-00002.safetensors";

struct Fixture {
    Descriptor descriptor;
    StageLayout layout;
    std::vector<unsigned char> first;
    std::vector<unsigned char> second;
};

void write_bytes(const std::filesystem::path &path,
                 const std::vector<unsigned char> &bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("cannot create fixture artifact");
    stream.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!stream)
        throw std::runtime_error("cannot write fixture artifact");
}

std::vector<unsigned char> sequence(unsigned char begin, size_t count) {
    std::vector<unsigned char> result(count);
    for (size_t index = 0; index < count; ++index)
        result[index] = static_cast<unsigned char>(begin + index);
    return result;
}

FieldSpec field(std::string name, uint32_t artifact, uint64_t offset,
                std::vector<uint64_t> shape, uint64_t bytes) {
    SourceRange source;
    source.artifact = artifact;
    source.offset = offset;
    source.bytes = bytes;
    source.tensor = name;
    source.dtype = "BF16";
    source.shape = shape;

    Materialization materialization;
    materialization.format = "BF16";
    materialization.storage_mode = "mlx-metal-shared";
    materialization.conversion = "copy-bf16-v1";
    materialization.shape = std::move(shape);
    materialization.reads.push_back(std::move(source));

    FieldSpec result;
    result.name = name;
    result.storage_id = name + ":storage";
    result.bytes = bytes;
    // Match the production Flux descriptor: slot spans are 256-byte aligned,
    // while source reads and MLX arrays retain the exact tensor byte count.
    result.alignment = 256;
    result.materialization = std::move(materialization);
    return result;
}

PoolLayout pool(uint32_t id, std::string layout_class,
                std::vector<uint64_t> capacities) {
    PoolLayout result;
    result.id = id;
    result.layout_class = std::move(layout_class);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        SlotLayout layout;
        layout.field_capacity = capacities;
        for (const uint64_t bytes : capacities)
            layout.capacity_bytes += bytes;
        result.capacity_bytes += layout.capacity_bytes;
        result.slots.push_back(std::move(layout));
    }
    return result;
}

Group group(uint32_t id, uint32_t pool_id, uint32_t slot,
            uint32_t block, std::vector<uint64_t> field_bytes) {
    Group result;
    result.id = id;
    result.pool = pool_id;
    result.slot = slot;
    result.blocks.push_back(block);
    result.field_bytes = std::move(field_bytes);
    for (const uint64_t bytes : result.field_bytes)
        result.bytes += bytes;
    return result;
}

Fixture make_fixture(const std::filesystem::path &root) {
    Fixture fixture;
    fixture.first = sequence(0x10, 28);
    fixture.second = sequence(0x80, 24);
    write_bytes(root / kArtifact0, fixture.first);
    write_bytes(root / kArtifact1, fixture.second);

    fixture.descriptor.model = "pager-fixture";
    fixture.descriptor.checkpoint_identity = "snapshot:test";
    fixture.descriptor.backend_revision = "mlx-weight-pager-test-v1";
    fixture.descriptor.artifacts = {
        SourceArtifact{kArtifact0, "snapshot:first", fixture.first.size(),
                       SourceIdentityKind::snapshot},
        SourceArtifact{kArtifact1, "snapshot:second", fixture.second.size(),
                       SourceIdentityKind::snapshot},
    };

    StageDescriptor stage;
    stage.id = "denoiser";
    stage.resident_fields.push_back(
        field("fixed.weight", 0, 0, {2}, 4));
    stage.blocks = {
        BlockSpec{0, "dual", {
            field("dual.0.matrix", 0, 4, {2, 2}, 8),
            field("dual.0.norm", 0, 12, {2}, 4)}},
        BlockSpec{1, "dual", {
            field("dual.1.matrix", 0, 16, {2, 2}, 8),
            field("dual.1.norm", 0, 24, {2}, 4)}},
        BlockSpec{2, "single", {
            field("single.0.matrix", 1, 0, {3, 2}, 12)}},
        BlockSpec{3, "single", {
            field("single.1.matrix", 1, 12, {3, 2}, 12)}},
    };
    fixture.descriptor.stages.push_back(std::move(stage));

    fixture.layout.id = "denoiser";
    fixture.layout.slot_count = 2;
    fixture.layout.group_size = 1;
    fixture.layout.distance = 1;
    fixture.layout.workers = 2;
    fixture.layout.pass_count = 1;
    fixture.layout.multi_pool_policy =
        tc::streaming::MultiPoolPolicy::retain_all;
    fixture.layout.resident_bytes = 4;
    fixture.layout.resident_source_read_bytes = 4;
    fixture.layout.pools = {
        pool(0, "dual", {256, 256}),
        pool(1, "single", {256}),
    };
    fixture.layout.groups = {
        group(0, 0, 0, 0, {256, 256}),
        group(1, 0, 1, 1, {256, 256}),
        group(2, 1, 0, 2, {256}),
        group(3, 1, 1, 3, {256}),
    };
    for (auto &group_value : fixture.layout.groups)
        group_value.bytes = 12;
    fixture.layout.peak_pool_bytes =
        fixture.layout.pools[0].capacity_bytes +
        fixture.layout.pools[1].capacity_bytes;
    fixture.layout.suffix_content_bytes_per_pass = 48;
    fixture.layout.source_read_bytes_per_pass = 48;
    return fixture;
}

tc_stream_slot_ticket_v1 ticket(const Group &group_value) {
    tc_stream_slot_ticket_v1 result{};
    result.struct_size = sizeof(result);
    result.version = TC_STREAM_SLOT_ABI_V1;
    result.pool = group_value.pool;
    result.slot = group_value.slot;
    result.request_generation = 1;
    result.content_generation = group_value.id + 1;
    result.item = {0, 0, 0, group_value.id};
    return result;
}

void expect_bytes(const tc::Tensor &value,
                  const std::vector<unsigned char> &source,
                  size_t offset, size_t count) {
    assert(value.nbytes() == count);
    assert(std::memcmp(value.data<char>(), source.data() + offset, count) == 0);
}

void rejects(const std::function<void()> &operation,
             const std::string &expected) {
    try {
        operation();
    } catch (const std::exception &error) {
        if (std::string(error.what()).find(expected) == std::string::npos)
            throw std::runtime_error(
                "unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("operation unexpectedly succeeded: " + expected);
}

void test_valid(const std::filesystem::path &root) {
    auto fixture = make_fixture(root);
    auto &stage = fixture.descriptor.stages.front();
    MlxWeightPager pager(root, fixture.descriptor, stage, fixture.layout);
    tc::Weights resident;
    std::atomic<bool> cancel{false};
    pager.load_resident(resident, &cancel);
    assert(resident.has("fixed.weight"));
    expect_bytes(resident.at("fixed.weight"), fixture.first, 0, 4);

    pager.create_pool(fixture.layout.pools[0]);
    pager.create_pool(fixture.layout.pools[1]);
    assert(pager.metrics().slot_arrays_allocated == 7);

    auto first_ticket = ticket(fixture.layout.groups[0]);
    assert(pager.fill(fixture.layout.groups[0], first_ticket, &cancel) == 12);
    auto first = pager.bind(fixture.layout.groups[0], first_ticket);
    expect_bytes(first.at("dual.0.matrix"), fixture.first, 4, 8);
    expect_bytes(first.at("dual.0.norm"), fixture.first, 12, 4);

    cancel.store(true, std::memory_order_release);
    auto second_ticket = ticket(fixture.layout.groups[1]);
    rejects([&] {
        (void)pager.fill(fixture.layout.groups[1], second_ticket, &cancel);
    }, "streaming_cancelled");
    rejects([&] {
        (void)pager.bind(fixture.layout.groups[1], second_ticket);
    }, "slot content identity");
    cancel.store(false, std::memory_order_release);
    assert(pager.fill(fixture.layout.groups[1], second_ticket, &cancel) == 12);
    auto second = pager.bind(fixture.layout.groups[1], second_ticket);
    expect_bytes(second.at("dual.1.matrix"), fixture.first, 16, 8);
    expect_bytes(second.at("dual.1.norm"), fixture.first, 24, 4);

    for (size_t index = 2; index < fixture.layout.groups.size(); ++index) {
        const auto &group_value = fixture.layout.groups[index];
        auto value_ticket = ticket(group_value);
        assert(pager.fill(group_value, value_ticket, &cancel) == 12);
        auto weights = pager.bind(group_value, value_ticket);
        const std::string key = index == 2 ?
            "single.0.matrix" : "single.1.matrix";
        expect_bytes(weights.at(key), fixture.second, (index - 2) * 12, 12);
    }

    auto wrong = first_ticket;
    wrong.pool = 1;
    rejects([&] { (void)pager.bind(fixture.layout.groups[0], wrong); },
            "bind identity");

    pager.destroy_pool(0);
    rejects([&] {
        (void)pager.bind(fixture.layout.groups[0], first_ticket);
    }, "unavailable");
    pager.create_pool(fixture.layout.pools[0]);
    assert(pager.fill(fixture.layout.groups[0], first_ticket, &cancel) == 12);
    auto recreated = pager.bind(fixture.layout.groups[0], first_ticket);
    expect_bytes(recreated.at("dual.0.matrix"), fixture.first, 4, 8);

    const auto &metrics = pager.metrics();
    assert(metrics.resident_bytes_loaded == 4);
    assert(metrics.streamed_bytes_loaded == 60);
    assert(metrics.slot_fills == 5);
    assert(metrics.slot_arrays_allocated == 11);
    pager.check_open_files();

    const auto original = root / kArtifact0;
    std::filesystem::rename(original, root / "fixture-00001.original");
    write_bytes(original, fixture.first);
    rejects([&] { pager.check_open_files(); }, "checkpoint_changed");
}

void test_short_read(const std::filesystem::path &root) {
    auto fixture = make_fixture(root);
    auto &stage = fixture.descriptor.stages.front();
    MlxWeightPager pager(root, fixture.descriptor, stage, fixture.layout);
    pager.create_pool(fixture.layout.pools[1]);
    const auto path = root / kArtifact1;
    assert(::truncate(path.c_str(), 6) == 0);
    std::atomic<bool> cancel{false};
    auto value_ticket = ticket(fixture.layout.groups[2]);
    rejects([&] {
        (void)pager.fill(fixture.layout.groups[2], value_ticket, &cancel);
    }, "truncated");
    rejects([&] {
        (void)pager.bind(fixture.layout.groups[2], value_ticket);
    }, "slot content identity");
}

void test_lease_backed(const std::filesystem::path &root) {
    auto fixture = make_fixture(root);
    std::vector<SourceFileIdentity> files;
    for (const auto *name : {kArtifact0, kArtifact1}) {
        SourceFileIdentity file;
        file.logical_id = name;
        file.path = root / name;
        files.push_back(std::move(file));
    }
    auto lease = tc::streaming::SourceLease::capture(std::move(files));
    auto &stage = fixture.descriptor.stages.front();
    MlxWeightPager pager(lease, fixture.descriptor, stage, fixture.layout);
    assert(lease->file_count() == 2);
    tc::Weights resident;
    std::atomic<bool> cancel{false};
    pager.load_resident(resident, &cancel);
    expect_bytes(resident.at("fixed.weight"), fixture.first, 0, 4);
    pager.create_pool(fixture.layout.pools[1]);
    auto value_ticket = ticket(fixture.layout.groups[2]);
    assert(pager.fill(fixture.layout.groups[2], value_ticket, &cancel) == 12);
    auto weights = pager.bind(fixture.layout.groups[2], value_ticket);
    expect_bytes(weights.at("single.0.matrix"), fixture.second, 0, 12);
    pager.check_open_files();

    const auto original = root / kArtifact1;
    std::filesystem::rename(original, root / "fixture-00002.original");
    write_bytes(original, fixture.second);
    // APFS updates the retained inode ctime during rename, so the conservative
    // lease may report either the opened generation or the named path stale.
    // Both are valid fail-closed outcomes; never require a path-only error.
    rejects([&] { pager.check_open_files(); }, "source");
}

void test_invalid_setup(const std::filesystem::path &root) {
    auto fixture = make_fixture(root);
    auto invalid_artifact = fixture.descriptor;
    invalid_artifact.stages[0].blocks[0].fields[0]
        .materialization->reads[0].artifact = 7;
    rejects([&] {
        MlxWeightPager pager(root, invalid_artifact,
                             invalid_artifact.stages.front(), fixture.layout);
    }, "source artifact");

    auto invalid_range = fixture.descriptor;
    invalid_range.stages[0].blocks[0].fields[0]
        .materialization->reads[0].offset = fixture.first.size() - 2;
    rejects([&] {
        MlxWeightPager pager(root, invalid_range,
                             invalid_range.stages.front(), fixture.layout);
    }, "source range exceeds");

    auto invalid_shape = fixture.descriptor;
    invalid_shape.stages[0].blocks[0].fields[0]
        .materialization->shape = {3};
    invalid_shape.stages[0].blocks[0].fields[0]
        .materialization->reads[0].shape = {3};
    rejects([&] {
        MlxWeightPager pager(root, invalid_shape,
                             invalid_shape.stages.front(), fixture.layout);
    }, "bytes differ");

    auto invalid_layout = fixture.layout;
    invalid_layout.groups[0].field_bytes[0] = 6;
    rejects([&] {
        MlxWeightPager pager(root, fixture.descriptor,
                             fixture.descriptor.stages.front(), invalid_layout);
    }, "field geometry");
}

} // namespace

int main(int argc, char **argv) {
    try {
        assert(argc == 2);
        const std::filesystem::path root(argv[1]);
        const auto valid = root / "valid";
        const auto short_read = root / "short-read";
        const auto lease_backed = root / "lease-backed";
        const auto invalid = root / "invalid";
        std::filesystem::create_directories(valid);
        std::filesystem::create_directories(short_read);
        std::filesystem::create_directories(lease_backed);
        std::filesystem::create_directories(invalid);
        test_valid(valid);
        test_short_read(short_read);
        test_lease_backed(lease_backed);
        test_invalid_setup(invalid);
        std::cout << "PASS MLX weight pager: resident load, retained dual/single "
                     "pool fill/bind, cancellation, short read, stale path, "
                     "lease-backed source validation, and pool recreate\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
