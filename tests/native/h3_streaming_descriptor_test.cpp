#include "h3_streaming_descriptor.hpp"
#include "h3_streaming_policy.h"

#include <cassert>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

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

        const tc::h3::StreamingPlanView plan(valid, config(), work);
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

        const tc::h3::StreamingPlanView no_prefix(valid, config(0), work);
        assert(no_prefix.layout().stages[0].groups.size() == 25);
        assert(no_prefix.layout().stages[0].groups.front().blocks.front() == 0);
        assert(no_prefix.layout().digest != plan.layout().digest);

        const tc::h3::StreamingPlanView repeat(valid, config(), work);
        assert(repeat.layout().canonical == plan.layout().canonical);
        assert(repeat.layout().digest == plan.layout().digest);
        auto changed = work;
        changed.steps = 5;
        const tc::h3::StreamingPlanView changed_steps(
            valid, config(), changed);
        assert(changed_steps.layout().digest != plan.layout().digest);
        changed = work;
        changed.active_blocks = 45;
        const tc::h3::StreamingPlanView changed_blocks(
            valid, config(), changed);
        assert(changed_blocks.layout().digest != plan.layout().digest);

        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(1, 1), work); },
                "slot count");
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(1, 3), work); },
                "slot count");
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(1, 2, 2), work); },
                "group size");
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(24), work); },
                "slot_count_exceeds_groups");
        auto resident = config();
        resident.stages["denoiser"] = {"resident", {}, {}, {}, {}, {}};
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, resident, work); },
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
                          valid, config(), changed); },
                "dynamic/fused shortcut");
        changed = work;
        changed.first_block_cache = true;
        rejects([&] { tc::h3::StreamingPlanView value(
                          valid, config(), changed); },
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
        const std::string first_shard = valid + "/model-00001.safetensors";
        const int descriptor_fd = open(first_shard.c_str(), O_WRONLY | O_CLOEXEC);
        assert(descriptor_fd >= 0);
        struct stat status{};
        assert(fstat(descriptor_fd, &status) == 0 && status.st_size > 8);
        assert(ftruncate(descriptor_fd, status.st_size - 1) == 0);
        close(descriptor_fd);
        rejects([&] { metadata.check_unchanged(); }, "checkpoint_changed");
        rejects([&] { metadata.describe(work); }, "checkpoint_changed");

        std::cout << "PASS H3 descriptor: 4 shards, uniform active IDs, "
                     "BF16 source ranges, K2/G1 plan and stale snapshot "
                     "rejection; layout="
                  << plan.layout().digest << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
