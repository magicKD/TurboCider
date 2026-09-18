#include "models/z_image/streaming_descriptor.hpp"

#include <cassert>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

tc::StreamingConfig config(uint32_t prefix = 3, uint32_t slots = 2,
                           uint32_t group = 1, uint32_t distance = 0,
                           uint32_t workers = 1) {
    tc::StreamingConfig value;
    value.enabled = true;
    value.schema_version = 1;
    value.selection = "manual";
    value.retention = "request";
    value.stages["denoiser"] = {
        "streamed", group, slots, prefix, distance, workers};
    return value;
}

tc::z_image::StreamingWorkload workload() {
    return {256, 256, 32, 3};
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

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 2) {
            const tc::z_image::StreamingMetadata metadata(argv[1]);
            const tc::z_image::StreamingPlanView plan(
                argv[1], config(), workload());
            std::cout << "PASS real Z-Image metadata shadow: blocks="
                      << metadata.block_count() << " fields="
                      << metadata.tensors_per_block() << " block_bytes="
                      << metadata.block_bytes() << " fixed_bytes="
                      << metadata.fixed_bytes() << " layout="
                      << plan.layout().digest << '\n';
            return 0;
        }
        assert(argc == 6);
        const std::string valid = argv[1];
        tc::z_image::StreamingMetadata metadata(valid);
        assert(metadata.block_count() == 30);
        assert(metadata.tensors_per_block() == 13);
        assert(metadata.block_bytes() != 0 && metadata.fixed_bytes() != 0);
        assert(!metadata.snapshot_identity().empty());

        const auto work = workload();
        const auto descriptor_value = metadata.describe(work);
        assert(descriptor_value.artifacts.size() == 1);
        assert(descriptor_value.stages.size() == 1);
        const auto &described = descriptor_value.stages.front();
        assert(described.blocks.size() == 30);
        assert(described.passes.size() == work.steps);
        for (const auto &block : described.blocks) {
            assert(block.fields.size() == 13);
            for (const auto &field : block.fields) {
                assert(field.materialization.has_value());
                assert(field.materialization->reads.size() == 1);
                assert(field.materialization->reads[0].artifact == 0);
                assert(field.materialization->reads[0].bytes == field.bytes);
            }
        }

        const tc::z_image::StreamingPlanView plan(valid, config(), work);
        const auto &stage = plan.layout().stages.at(0);
        assert(plan.layout().materializations_complete);
        assert(stage.prefix == 3 && stage.slot_count == 2 &&
               stage.group_size == 1 && stage.distance == 0 &&
               stage.workers == 1);
        assert(stage.groups.size() == 27 && stage.pools.size() == 1);
        assert(stage.prefix_bytes == 3 * metadata.block_bytes());
        assert(stage.peak_pool_bytes == 2 * metadata.block_bytes());
        assert(stage.prefix_source_read_bytes ==
               3 * metadata.block_bytes());
        assert(stage.source_read_bytes_per_pass ==
               27 * metadata.block_bytes());
        assert(stage.suffix_content_bytes_per_pass ==
               27 * metadata.block_bytes());
        for (uint32_t index = 0; index < stage.groups.size(); ++index) {
            assert(stage.groups[index].blocks.size() == 1);
            assert(stage.groups[index].blocks.front() == index + 3);
            assert(stage.groups[index].slot == index % 2);
        }

        const tc::z_image::StreamingPlanView repeat(valid, config(), work);
        assert(repeat.layout().canonical == plan.layout().canonical);
        assert(repeat.layout().digest == plan.layout().digest);
        tc::streaming::SourceFileIdentity leased_file;
        leased_file.logical_id = "transformer";
        leased_file.path = valid;
        const auto lease = tc::streaming::SourceLease::capture(
            {std::move(leased_file)});
        const tc::z_image::StreamingMetadata leased_metadata(
            lease, "transformer");
        const tc::z_image::StreamingPlanView leased_plan(
            lease, config(), work);
        assert(leased_metadata.lease_ptr() == lease);
        assert(leased_plan.lease_ptr() == lease);
        assert(leased_plan.layout().canonical == plan.layout().canonical);
        assert(leased_plan.layout().digest == plan.layout().digest);
        auto changed = work;
        changed.steps = 4;
        const tc::z_image::StreamingPlanView changed_steps(
            valid, config(), changed);
        assert(changed_steps.layout().digest != plan.layout().digest);

        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 1), work); },
                "slot count");
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 3), work); },
                "slot count");
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 2, 2), work); },
                "group size");
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 2, 1, 1), work); },
                "D=0");
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 2, 1, 0, 2), work); },
                "Q=1");
        auto resident = config();
        resident.stages["denoiser"] = {"resident", {}, {}, {}, {}, {}};
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, resident, work); },
                "K=2/G=1");
        changed = work;
        changed.width = 250;
        rejects([&] { metadata.describe(changed); }, "normalized");
        changed = work;
        changed.caption_rows = 31;
        rejects([&] { metadata.describe(changed); }, "normalized");

        rejects([&] { tc::z_image::StreamingMetadata value(argv[2]); },
                "30 dense blocks");
        rejects([&] { tc::z_image::StreamingMetadata value(argv[3]); },
                "BF16");
        rejects([&] { tc::z_image::StreamingMetadata value(argv[4]); },
                "matching tensor layouts");
        rejects([&] { tc::z_image::StreamingMetadata value(argv[5]); },
                "overlapping or noncontiguous");

        metadata.check_unchanged();
        const int checkpoint_fd = ::open(valid.c_str(), O_WRONLY | O_CLOEXEC);
        assert(checkpoint_fd >= 0);
        struct stat status{};
        assert(::fstat(checkpoint_fd, &status) == 0 && status.st_size > 8);
        assert(::ftruncate(checkpoint_fd, status.st_size - 1) == 0);
        ::close(checkpoint_fd);
        rejects([&] { metadata.check_unchanged(); }, "checkpoint_changed");
        rejects([&] { metadata.describe(work); }, "checkpoint_changed");
        rejects([&] { leased_metadata.check_unchanged(); },
                "source fd changed");

        std::cout << "PASS Z-Image descriptor: header-only 30x13 BF16 "
                     "projection, shared SourceLease, K2/G1/D0/Q1 layout, "
                     "malformed metadata and stale snapshot rejection; layout="
                  << plan.layout().digest << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
