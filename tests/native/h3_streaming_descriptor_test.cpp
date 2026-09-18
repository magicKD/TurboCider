#include "h3_streaming_descriptor.hpp"
#include "h3_streaming_policy.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" h3_gpu_tensor *h3_gpu_tensor_load_bf16(
    h3_gpu *, const char *, uint64_t, size_t) {
    return nullptr;
}
extern "C" h3_gpu_tensor *h3_gpu_tensor_load_f32(
    h3_gpu *, const char *, uint64_t, size_t) {
    return nullptr;
}
extern "C" const char *h3_gpu_error(const h3_gpu *) { return "unused"; }

namespace {

tc::StreamingConfig config(uint32_t prefix = 1, uint32_t slots = 2,
                           uint32_t group = 1) {
    tc::StreamingConfig value;
    value.enabled = true;
    value.schema_version = 1;
    value.selection = "manual";
    value.retention = "request";
    value.stages["denoiser"] = {
        "streamed", group, slots, prefix, slots > 1 ? 1u : 0u, 1};
    return value;
}

tc::h3::StreamingWorkload workload(uint32_t active = 25) {
    tc::h3::StreamingWorkload value;
    value.width = 768;
    value.height = 768;
    value.frames = 5;
    value.fps = 24;
    value.text_rows = 16;
    value.steps = 4;
    value.active_blocks = active;
    value.audio = false;
    return value;
}

template <class Function>
void rejects(Function function, const char *part) {
    try {
        function();
    } catch (const std::invalid_argument &error) {
        if (std::string(error.what()).find(part) != std::string::npos)
            return;
        std::cerr << error.what() << " expected " << part << '\n';
        std::abort();
    }
    std::cerr << "expected rejection containing " << part << '\n';
    std::abort();
}

uint64_t matrix_bytes() {
    return (uint64_t{H3_DIT_INNER} * 3u * H3_DIT_HIDDEN +
            uint64_t{H3_DIT_HIDDEN} * H3_DIT_INNER +
            uint64_t{H3_DIT_FFN} * 2u * H3_DIT_HIDDEN +
            uint64_t{H3_DIT_HIDDEN} * H3_DIT_FFN) *
           sizeof(uint16_t);
}

std::pair<std::shared_ptr<const tc::streaming::SourceLease>,
          std::vector<std::string>>
lease_for(const std::string &directory) {
    std::vector<tc::streaming::SourceFileIdentity> files;
    std::vector<std::string> logical_ids;
    for (unsigned index = 1; index <= 4; ++index) {
        char name[64] = {};
        std::snprintf(name, sizeof(name), "model-%05u.safetensors", index);
        tc::streaming::SourceFileIdentity file;
        file.logical_id = std::string("transformer/") + name;
        file.path = std::filesystem::path(directory) / name;
        logical_ids.push_back(file.logical_id);
        files.push_back(std::move(file));
    }
    return {tc::streaming::SourceLease::capture(std::move(files)),
            std::move(logical_ids)};
}

struct FakeExecution {
    static constexpr uint32_t groups = 24;
    static constexpr uint32_t passes = 4;
    std::array<std::atomic<uint32_t>, 2> slots{};
    std::array<std::atomic<uint32_t>, groups * passes> fills{};
    std::array<std::atomic<uint32_t>, groups * passes> encodes{};
    std::atomic<uint32_t> prefix{0}, allocations{0}, destroys{0};
    std::atomic<uint64_t> sequence{0};
    int fail_pass = -1, fail_group = -1;
};

int fake_allocate(void *user, uint32_t slot, uint64_t capacity,
                  char *, size_t) {
    auto &fake = *static_cast<FakeExecution *>(user);
    assert(slot < fake.slots.size() && capacity == matrix_bytes());
    fake.slots[slot].store(UINT32_MAX, std::memory_order_relaxed);
    fake.allocations.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

void fake_destroy(void *user) {
    static_cast<FakeExecution *>(user)->destroys.fetch_add(
        1, std::memory_order_relaxed);
}

int fake_fill(void *user, const tc_stream_slot_ticket_v1 *ticket,
              const tc_stream_group_v1 *group,
              tc_stream_cancel_query_v1 cancelled, const void *cancel_user,
              uint64_t *bytes, char *error, size_t error_size) {
    auto &fake = *static_cast<FakeExecution *>(user);
    if (cancelled(cancel_user)) return 0;
    assert(ticket && group && group->block_count == 1 &&
           ticket->item.pass < FakeExecution::passes &&
           ticket->item.group < FakeExecution::groups);
    if (static_cast<int>(ticket->item.pass) == fake.fail_pass &&
        static_cast<int>(ticket->item.group) == fake.fail_group) {
        std::snprintf(error, error_size, "injected H3 carry fill failure");
        return 0;
    }
    const size_t index = ticket->item.pass * FakeExecution::groups +
                         ticket->item.group;
    assert(fake.fills[index].fetch_add(1, std::memory_order_relaxed) == 0);
    fake.slots[ticket->slot].store(group->blocks[0],
                                   std::memory_order_release);
    *bytes = group->content_bytes;
    return 1;
}

int fake_prefix(void *user, uint32_t pass, char *, size_t) {
    auto &fake = *static_cast<FakeExecution *>(user);
    assert(fake.prefix.fetch_add(1, std::memory_order_relaxed) == pass);
    return 1;
}

int fake_prepare(void *user, const tc_stream_slot_ticket_v1 *ticket,
                 const tc_stream_group_v1 *group, char *, size_t) {
    auto &fake = *static_cast<FakeExecution *>(user);
    return fake.slots[ticket->slot].load(std::memory_order_acquire) ==
           group->blocks[0];
}

int fake_encode(void *user, const tc_stream_slot_ticket_v1 *ticket,
                const tc_stream_group_v1 *group,
                const tc_stream_completion_sink_v1 *sink,
                tc_stream_reader_set_v1 *readers, char *error,
                size_t error_size) {
    auto &fake = *static_cast<FakeExecution *>(user);
    if (!fake_prepare(user, ticket, group, error, error_size)) return 0;
    const size_t index = ticket->item.pass * FakeExecution::groups +
                         ticket->item.group;
    assert(fake.encodes[index].fetch_add(1, std::memory_order_relaxed) == 0);
    readers->count = 1;
    readers->fences[0] = {
        1, fake.sequence.fetch_add(1, std::memory_order_relaxed) + 1};
    return sink->post(sink->user, ticket, readers->fences[0], 0);
}

int fake_drain(void *, char *, size_t) { return 1; }

tc_stream_adapter_v1 fake_adapter(FakeExecution &fake) {
    return {sizeof(tc_stream_adapter_v1), TC_STREAM_SLOT_ABI_V1, &fake,
            fake_allocate, fake_destroy, fake_fill, fake_prefix,
            fake_prepare, fake_encode, fake_drain};
}

void execute_fake_plan(const tc_stream_stage_plan_v3 &plan) {
    FakeExecution fake;
    auto adapter = fake_adapter(fake);
    char error[1024] = {};
    tc_stream_executor *executor = nullptr;
    assert(tc_stream_executor_create_v3(
        &plan, &adapter, &executor, error, sizeof(error)));
    for (uint32_t pass = 0; pass < FakeExecution::passes; ++pass) {
        assert(tc_stream_executor_run_pass(
            executor, pass, pass, error, sizeof(error)));
        const uint32_t expected = (pass + 1) * FakeExecution::groups +
            (pass + 1 < FakeExecution::passes ? 1u : 0u);
        uint32_t fills = 0;
        for (const auto &count : fake.fills)
            fills += count.load(std::memory_order_relaxed);
        assert(fills == expected);
    }
    assert(tc_stream_executor_finish(executor, error, sizeof(error)));
    tc_stream_counters_v1 counters{};
    assert(tc_stream_executor_counters(
        executor, &counters, error, sizeof(error)));
    assert(counters.pool_creates == 1 && counters.slot_bundles == 2 &&
           counters.fills == FakeExecution::groups * FakeExecution::passes &&
           counters.groups_submitted == counters.fills &&
           counters.content_bytes_loaded == counters.fills * matrix_bytes());
    for (size_t index = 0; index < fake.fills.size(); ++index) {
        assert(fake.fills[index].load(std::memory_order_relaxed) == 1);
        assert(fake.encodes[index].load(std::memory_order_relaxed) == 1);
    }
    assert(fake.prefix.load(std::memory_order_relaxed) == FakeExecution::passes);
    assert(tc_stream_executor_destroy(&executor, error, sizeof(error)) &&
           !executor);
    assert(fake.allocations.load(std::memory_order_relaxed) == 2 &&
           fake.destroys.load(std::memory_order_relaxed) == 1);
}

void exercise_fake_failures(const tc_stream_stage_plan_v3 &plan) {
    char error[1024] = {};
    {
        FakeExecution fake;
        auto adapter = fake_adapter(fake);
        tc_stream_executor *executor = nullptr;
        assert(tc_stream_executor_create_v3(
            &plan, &adapter, &executor, error, sizeof(error)));
        assert(tc_stream_executor_run_pass(
            executor, 0, 0, error, sizeof(error)));
        tc_stream_executor_cancel(executor);
        assert(!tc_stream_executor_run_pass(
            executor, 1, 1, error, sizeof(error)));
        assert(std::string(error).find("cancelled") != std::string::npos);
        assert(tc_stream_executor_destroy(
            &executor, error, sizeof(error)) && !executor);
        assert(fake.destroys.load(std::memory_order_relaxed) == 1);
    }
    {
        FakeExecution fake;
        fake.fail_pass = 1;
        fake.fail_group = 0;
        auto adapter = fake_adapter(fake);
        tc_stream_executor *executor = nullptr;
        error[0] = '\0';
        assert(tc_stream_executor_create_v3(
            &plan, &adapter, &executor, error, sizeof(error)));
        assert(!tc_stream_executor_run_pass(
            executor, 0, 0, error, sizeof(error)));
        assert(std::string(error).find("injected H3 carry fill failure") !=
               std::string::npos);
        assert(tc_stream_executor_destroy(
            &executor, error, sizeof(error)) && !executor);
        assert(fake.destroys.load(std::memory_order_relaxed) == 1);
    }
}

} // namespace

int main(int argc, char **argv) {
    assert(argc == 5);
    try {
        const std::string valid = argv[1];
        tc::h3::StreamingMetadata metadata(valid);
        assert(metadata.shard_count() == 4);
        assert(metadata.block_count() == H3_DIT_BLOCKS);
        assert(metadata.snapshot_identity() != 0);

        const auto work = workload();
        const auto descriptor = metadata.describe(work);
        assert(descriptor.artifacts.size() == 4);
        assert(descriptor.stages.size() == 1);
        assert(descriptor.stages[0].blocks.size() == 25);
        assert(descriptor.stages[0].passes.size() == 4);
        assert(descriptor.stages[0].blocks.front().id == 0);
        assert(descriptor.stages[0].blocks[1].id == 2);
        assert(descriptor.stages[0].blocks.back().id == 49);
        for (const auto &block : descriptor.stages[0].blocks) {
            assert(block.fields.size() == 4);
            for (const auto &field : block.fields) {
                assert(field.materialization.has_value());
                assert(field.materialization->reads.size() == 1);
                assert(field.materialization->reads[0].artifact < 4);
                assert(field.materialization->reads[0].bytes == field.bytes);
            }
        }

        const tc::h3::StreamingPlanView plan(valid, config(), work, 77);
        const auto &stage = plan.layout().stages.at(0);
        const uint64_t block_bytes = matrix_bytes();
        assert(plan.layout().materializations_complete);
        assert(stage.prefix == 1 && stage.slot_count == 2 &&
               stage.group_size == 1);
        assert(stage.groups.size() == 24);
        assert(stage.groups.front().blocks.size() == 1 &&
               stage.groups.front().blocks.front() == 2);
        assert(stage.groups.back().blocks.front() == 49);
        assert(stage.prefix_bytes == block_bytes);
        assert(stage.peak_pool_bytes == 2 * block_bytes);
        assert(stage.prefix_source_read_bytes == block_bytes);
        assert(stage.source_read_bytes_per_pass == 24 * block_bytes);
        assert(stage.suffix_content_bytes_per_pass == 24 * block_bytes);
        const auto &c_plan = plan.c_plan();
        assert(c_plan.struct_size == sizeof(c_plan));
        assert(c_plan.version == TC_STREAM_SLOT_ABI_V3);
        assert(c_plan.request_generation == 77);
        assert(c_plan.pass_transition ==
               TC_STREAM_PASS_CARRY_FIRST_GROUP_V3);
        assert(c_plan.slot_count == 2 && c_plan.group_count == 24);
        assert(c_plan.slot_capacity_bytes[0] == block_bytes &&
               c_plan.slot_capacity_bytes[1] == block_bytes);
        assert(c_plan.groups[0].group == 0 && c_plan.groups[0].slot == 0);
        assert(c_plan.groups[23].group == 23 && c_plan.groups[23].slot == 1);
        execute_fake_plan(c_plan);
        exercise_fake_failures(c_plan);

        auto [lease, logical_ids] = lease_for(valid);
        tc::h3::StreamingMetadata lease_metadata(lease, logical_ids);
        assert(lease_metadata.source_lease().get() == lease.get());
        assert(lease_metadata.snapshot_identity() ==
               metadata.snapshot_identity());
        const tc::h3::StreamingPlanView lease_plan(
            lease, logical_ids, config(), work, 77);
        assert(lease_plan.metadata().source_lease().get() == lease.get());
        assert(lease_plan.descriptor().checkpoint_identity ==
               plan.descriptor().checkpoint_identity);
        assert(lease_plan.layout().canonical == plan.layout().canonical);
        assert(lease_plan.layout().digest == plan.layout().digest);
        assert(lease_plan.c_plan().request_generation == 77);

        char source_error[512] = {};
        {
            auto caller_fd = lease->duplicate_fd(logical_ids.front());
            h3_st_header leased_header{};
            assert(h3_st_read_header_fd(
                lease->file(logical_ids.front()).path.c_str(),
                caller_fd.get(), &leased_header, source_error,
                sizeof(source_error)));
            assert(leased_header.descriptor >= 0 &&
                   leased_header.tensor_count > 0);
            caller_fd = tc::streaming::OwnedSourceFd{};
            struct stat retained_status{};
            assert(fstat(leased_header.descriptor, &retained_status) == 0 &&
                   retained_status.st_size > 8);
            h3_st_free_header(&leased_header);
        }
        {
            std::unique_ptr<h3_weight_store,
                            decltype(&h3_weight_store_free)> store(
                h3_weight_store_open(valid.c_str(), source_error,
                                     sizeof(source_error)),
                h3_weight_store_free);
            assert(store);
            std::vector<tc::streaming::OwnedSourceFd> descriptors;
            std::vector<h3_weight_source_v1> sources;
            descriptors.reserve(logical_ids.size());
            sources.reserve(logical_ids.size());
            for (const auto &id : logical_ids) {
                const auto &file = lease->file(id);
                descriptors.push_back(lease->duplicate_fd(id));
                sources.push_back({file.path.c_str(),
                                   descriptors.back().get()});
            }
            assert(!h3_weight_store_bind_sources(
                store.get(), sources.data(), sources.size() - 1,
                source_error, sizeof(source_error)));
            for (size_t index = 0;
                 index < h3_weight_store_shards(store.get()); ++index)
                assert(h3_weight_store_header(store.get(), index)->descriptor < 0);
            source_error[0] = '\0';
            assert(h3_weight_store_bind_sources(
                store.get(), sources.data(), sources.size(),
                source_error, sizeof(source_error)));
            descriptors.clear();
            for (size_t index = 0;
                 index < h3_weight_store_shards(store.get()); ++index) {
                const auto *header = h3_weight_store_header(store.get(), index);
                assert(header && header->descriptor >= 0);
                struct stat retained_status{};
                assert(fstat(header->descriptor, &retained_status) == 0 &&
                       static_cast<uint64_t>(retained_status.st_size) ==
                           header->file_size);
            }
        }
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(), work, 0); },
                "generation");

        const tc::h3::StreamingPlanView no_prefix(
            valid, config(0), work, 78);
        assert(no_prefix.layout().stages[0].groups.size() == 25);
        assert(no_prefix.layout().stages[0].groups.front().blocks.front() == 0);
        assert(no_prefix.layout().digest != plan.layout().digest);
        assert(no_prefix.c_plan().pass_transition ==
               TC_STREAM_PASS_CARRY_FIRST_GROUP_V3);

        const tc::h3::StreamingPlanView repeat(valid, config(), work, 77);
        assert(repeat.layout().canonical == plan.layout().canonical);
        assert(repeat.layout().digest == plan.layout().digest);
        auto changed = work;
        changed.steps = 5;
        const tc::h3::StreamingPlanView changed_steps(
            valid, config(), changed, 79);
        assert(changed_steps.layout().digest != plan.layout().digest);
        changed = work;
        changed.active_blocks = 45;
        const tc::h3::StreamingPlanView changed_blocks(
            valid, config(), changed, 80);
        assert(changed_blocks.layout().digest != plan.layout().digest);

        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(1, 1), work, 1); },
                "slot count");
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(1, 3), work, 1); },
                "slot count");
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(1, 2, 2), work, 1); },
                "group size");
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(24), work, 1); },
                "slot_count_exceeds_groups");
        auto resident = config();
        resident.stages["denoiser"] = {"resident", {}, {}, {}, {}, {}};
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, resident, work, 1); },
                "K=2/G=1");
        changed = work;
        changed.active_blocks = 24;
        rejects([&] { metadata.describe(changed); }, "active block count");
        changed = work;
        changed.width = 736;
        rejects([&] { metadata.describe(changed); }, "normalized");
        changed = work;
        changed.frames = 6;
        rejects([&] { metadata.describe(changed); }, "frame count");
        changed = work;
        changed.token_reduction = true;
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(), changed, 1); },
                "dynamic/fused shortcut");
        changed = work;
        changed.first_block_cache = true;
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(), changed, 1); },
                "dynamic/fused shortcut");

        rejects([&] { tc::h3::StreamingMetadata value(argv[2]);
                      value.describe(work); },
                "required H3 matrix is absent");
        rejects([&] { tc::h3::StreamingMetadata value(argv[3]);
                      value.describe(work); },
                "wrong dtype or shape");
        rejects([&] { tc::h3::StreamingMetadata value(argv[4]);
                      value.describe(work); },
                "duplicate H3 matrix");

        metadata.check_unchanged();
        lease_metadata.check_unchanged();
        const std::string first_shard = valid + "/model-00001.safetensors";
        auto held_fd = lease->duplicate_fd(logical_ids.front());
        std::array<unsigned char, 8> original_prefix{};
        assert(pread(held_fd.get(), original_prefix.data(),
                     original_prefix.size(), 0) ==
               static_cast<ssize_t>(original_prefix.size()));
        const std::string moved_shard = first_shard + ".moved";
        assert(rename(first_shard.c_str(), moved_shard.c_str()) == 0);
        {
            std::ofstream replacement(first_shard, std::ios::binary);
            replacement << "replacement";
        }
        std::array<unsigned char, 8> retained_prefix{};
        assert(pread(held_fd.get(), retained_prefix.data(),
                     retained_prefix.size(), 0) ==
                   static_cast<ssize_t>(retained_prefix.size()) &&
               retained_prefix == original_prefix);
        rejects([&] { lease->revalidate_paths(); }, "source");
        rejects([&] { lease->revalidate_open_files(); }, "source");
        rejects([&] { lease_metadata.check_unchanged(); }, "source");
        rejects([&] { metadata.check_unchanged(); }, "checkpoint_changed");
        rejects([&] { metadata.describe(work); }, "checkpoint_changed");

        const int descriptor_fd = open(
            moved_shard.c_str(), O_WRONLY | O_CLOEXEC);
        assert(descriptor_fd >= 0);
        struct stat status{};
        assert(fstat(descriptor_fd, &status) == 0 && status.st_size > 8);
        assert(ftruncate(descriptor_fd, status.st_size - 1) == 0);
        close(descriptor_fd);
        rejects([&] { lease->revalidate_open_files(); }, "source");

        std::cout << "PASS H3 descriptor: 4 shards, uniform active IDs, "
                     "BF16 source ranges, path/lease layout parity, atomic "
                     "descriptor binding and stale snapshot rejection; layout="
                  << plan.layout().digest << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
