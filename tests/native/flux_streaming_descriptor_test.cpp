#include "models/flux2/streaming_descriptor.hpp"

#include <cassert>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

tc::StreamingConfig config(uint32_t prefix = 0, uint32_t slots = 2,
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

tc::flux2::StreamingWorkload workload() {
    return {256, 256, 32, 0, 4};
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
        if (argc == 3 && std::string(argv[1]) == "--4b") {
            const std::string root = argv[2];
            tc::flux2::StreamingMetadata metadata(root, "flux2-klein-4b");
            assert(metadata.hidden_size() == 3072 && metadata.head_count() == 24);
            assert(metadata.dual_block_count() == 5 && metadata.single_block_count() == 20);
            assert(metadata.dual_block_bytes() == 2ull * (26ull * 3072 * 3072 + 512));
            assert(metadata.single_block_bytes() == 2ull * (13ull * 3072 * 3072 + 256));
            const auto descriptor = metadata.describe(workload());
            assert(descriptor.artifacts.size() == 1 && descriptor.stages[0].blocks.size() == 25);
            const auto &fixed = descriptor.stages[0].resident_fields;
            assert(fixed.size() == 9);
            bool found_context = false;
            for (const auto &field : fixed) {
                if (field.materialization->reads[0].tensor == "context_embedder.weight") {
                    assert(field.bytes == 3072ull * 7680 * 2);
                    found_context = true;
                }
            }
            assert(found_context);
            for (uint32_t d : {0u, 1u}) for (uint32_t q : {1u, 2u}) {
                tc::flux2::StreamingPlanView plan(root, "flux2-klein-4b", config(0, 2, 1, d, q), workload());
                const auto &stage = plan.layout().stages[0];
                assert(stage.groups.size() == 25 && stage.pools.size() == 2);
                assert(stage.pools[0].layout_class == "flux2-klein-4b-dual-bf16-v1");
                assert(stage.pools[1].layout_class == "flux2-klein-4b-single-bf16-v1");
                assert(stage.resident_source_read_bytes == metadata.fixed_bytes());
                for (uint32_t i = 0; i < 25; ++i) {
                    assert(stage.groups[i].blocks == std::vector<uint32_t>{i});
                    assert(descriptor.stages[0].blocks[i].fields.size() == (i < 5 ? 16 : 4));
                }
            }
            std::vector<tc::streaming::SourceFileIdentity> files;
            for (const auto *name : {"config.json", "diffusion_pytorch_model.safetensors"}) {
                tc::streaming::SourceFileIdentity file;
                file.logical_id = name; file.path = std::filesystem::path(root) / name;
                files.push_back(std::move(file));
            }
            auto lease = tc::streaming::SourceLease::capture(std::move(files));
            tc::flux2::StreamingPlanView leased(lease, "flux2-klein-4b", config(), workload());
            tc::flux2::StreamingPlanView direct(root, "flux2-klein-4b", config(), workload());
            assert(leased.layout().digest == direct.layout().digest);
            rejects([&] { tc::flux2::StreamingPlanView bad(root, "flux2-klein-4b", config(1), workload()); }, "requires P0");
            std::cout << "PASS FLUX 4B single-file descriptor: 5+20 blocks, 3072/7680 geometry, two pools, D/Q layouts, lease parity\n";
            return 0;
        }
        if (argc == 2) {
            const tc::flux2::StreamingMetadata metadata(
                argv[1], "flux2-klein-9b");
            const tc::flux2::StreamingPlanView plan(
                argv[1], "flux2-klein-9b", config(), workload());
            std::cout << "PASS real FLUX 9B metadata shadow: dual="
                      << metadata.dual_block_count() << "x"
                      << metadata.dual_fields_per_block() << " single="
                      << metadata.single_block_count() << "x"
                      << metadata.single_fields_per_block() << " fixed_bytes="
                      << metadata.fixed_bytes() << " layout="
                      << plan.layout().digest << '\n';
            return 0;
        }
        assert(argc == 6);
        const std::string valid = argv[1];
        tc::flux2::StreamingMetadata metadata(valid, "flux2-klein-9b");
        assert(metadata.dual_block_count() == 8);
        assert(metadata.single_block_count() == 24);
        assert(metadata.dual_block_bytes() != 0);
        assert(metadata.single_block_bytes() != 0);
        assert(metadata.fixed_bytes() != 0);
        assert(!metadata.snapshot_identity().empty());

        const auto work = workload();
        const auto descriptor = metadata.describe(work);
        assert(descriptor.artifacts.size() == 2);
        assert(descriptor.stages.size() == 1);
        const auto &described = descriptor.stages.front();
        assert(described.blocks.size() == 32);
        assert(described.resident_fields.size() == 9);
        assert(described.passes.size() == work.steps);
        for (const auto &field : described.resident_fields) {
            assert(field.materialization.has_value());
            assert(field.materialization->reads.size() == 1);
            assert(field.materialization->reads[0].bytes == field.bytes);
        }
        for (uint32_t block = 0; block < described.blocks.size(); ++block) {
            const size_t expected_fields = block < 8 ? 16 : 4;
            const auto &value = described.blocks[block];
            assert(value.id == block && value.fields.size() == expected_fields);
            assert(value.streamable && value.safe_boundary_after);
            for (const auto &field : value.fields) {
                assert(field.materialization.has_value());
                assert(field.materialization->reads.size() == 1);
                assert(field.materialization->reads[0].artifact < 2);
                assert(field.materialization->reads[0].bytes == field.bytes);
            }
        }

        const tc::flux2::StreamingPlanView plan(
            valid, "flux2-klein-9b", config(), work);
        const auto &stage = plan.layout().stages.at(0);
        assert(plan.layout().materializations_complete);
        assert(stage.prefix == 0 && stage.slot_count == 2 &&
               stage.group_size == 1 && stage.distance == 0 &&
               stage.workers == 1 && stage.groups.size() == 32 &&
               stage.pools.size() == 2);
        assert(stage.multi_pool_policy ==
               tc::streaming::MultiPoolPolicy::retain_all);
        assert(stage.resident_bytes == metadata.fixed_bytes());
        assert(stage.resident_source_read_bytes == metadata.fixed_bytes());
        assert(stage.pools[0].layout_class ==
               "flux2-klein-9b-dual-bf16-v1");
        assert(stage.pools[1].layout_class ==
               "flux2-klein-9b-single-bf16-v1");
        assert(stage.pools[0].capacity_bytes ==
               2 * metadata.dual_block_bytes());
        assert(stage.pools[1].capacity_bytes ==
               2 * metadata.single_block_bytes());
        assert(stage.peak_pool_bytes ==
               stage.pools[0].capacity_bytes +
                   stage.pools[1].capacity_bytes);
        for (uint32_t index = 0; index < stage.groups.size(); ++index) {
            assert(stage.groups[index].blocks.size() == 1);
            assert(stage.groups[index].blocks.front() == index);
            assert(stage.groups[index].pool == (index < 8 ? 0 : 1));
            assert(stage.groups[index].slot == index % 2);
        }

        const tc::flux2::StreamingPlanView repeat(
            valid, "flux2-klein-9b", config(), work);
        assert(repeat.layout().canonical == plan.layout().canonical);
        assert(repeat.layout().digest == plan.layout().digest);
        auto changed = work;
        changed.reference_tokens = 16;
        const tc::flux2::StreamingPlanView changed_workload(
            valid, "flux2-klein-9b", config(), changed);
        assert(changed_workload.layout().digest != plan.layout().digest);

        rejects([&] { tc::flux2::StreamingPlanView value(
                          valid, "flux2-klein-9b", config(0, 1), work); },
                "slot count");
        rejects([&] { tc::flux2::StreamingPlanView value(
                          valid, "flux2-klein-9b", config(0, 2, 2), work); },
                "unsupported group size");
        const tc::flux2::StreamingPlanView overlap(
            valid, "flux2-klein-9b", config(0, 2, 1, 1, 2), work);
        assert(overlap.layout().stages.front().distance == 1);
        assert(overlap.layout().stages.front().workers == 2);
        assert(overlap.layout().digest != plan.layout().digest);
        rejects([&] { tc::flux2::StreamingPlanView value(
                          valid, "flux2-klein-9b", config(1), work); },
                "P0");
        rejects([&] { tc::flux2::StreamingMetadata value(
                          valid, "flux2-klein-4b"); },
                "configuration does not match");

        changed = work;
        changed.width = 250;
        rejects([&] { metadata.describe(changed); }, "normalized");
        changed = work;
        changed.reference_tokens = 20000;
        rejects([&] { metadata.describe(changed); }, "workspace budget");

        rejects([&] { tc::flux2::StreamingMetadata value(
                          argv[2], "flux2-klein-9b"); },
                "index tensor set");
        rejects([&] { tc::flux2::StreamingMetadata value(
                          argv[3], "flux2-klein-9b"); },
                "BF16");
        rejects([&] { tc::flux2::StreamingMetadata value(
                          argv[4], "flux2-klein-9b"); },
                "tensor shape differs");
        rejects([&] { tc::flux2::StreamingMetadata value(
                          argv[5], "flux2-klein-9b"); },
                "overlapping or noncontiguous");

        metadata.check_unchanged();
        const std::string shard = valid +
            "/diffusion_pytorch_model-00002-of-00002.safetensors";
        const int descriptor_fd = ::open(shard.c_str(), O_WRONLY | O_CLOEXEC);
        assert(descriptor_fd >= 0);
        struct stat status{};
        assert(::fstat(descriptor_fd, &status) == 0 && status.st_size > 8);
        assert(::ftruncate(descriptor_fd, status.st_size - 1) == 0);
        ::close(descriptor_fd);
        rejects([&] { metadata.check_unchanged(); }, "checkpoint_changed");
        rejects([&] { metadata.describe(work); }, "checkpoint_changed");

        std::cout << "PASS FLUX 9B descriptor: header-only two-shard "
                     "8x16 dual + 24x4 single projection, ordered two-class "
                     "K2/G1 layout, malformed metadata and stale snapshot "
                     "rejection; layout="
                  << plan.layout().digest << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
