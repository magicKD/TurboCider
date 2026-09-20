#include "models/z_image/streaming_descriptor.hpp"

#include <cassert>
#include <fcntl.h>
#include <fstream>
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
        rejects([&] { (void) metadata.describe_verified(work); },
                "artifact_verification_required");
        {
            using tc::streaming::SourceFileIdentity;
            using tc::streaming::SourceLease;
            auto file = [](const std::string &path) {
                SourceFileIdentity value;
                value.logical_id = "transformer";
                value.path = path;
                return value;
            };
            const auto original = SourceLease::capture_verified({file(valid)});
            const tc::z_image::StreamingMetadata original_metadata(original);
            const auto portable = original_metadata.describe_verified(work);
            const auto original_layout = tc::streaming::compile_layout(config(), portable);
            const tc::z_image::StreamingPlanView verified_plan(original, config(), work);
            assert(verified_plan.layout().digest == original_layout.digest);
            assert(verified_plan.lease_ptr() == original);
            assert(portable.artifacts.front().identity_kind ==
                   tc::streaming::SourceIdentityKind::content_sha256);
            const std::string copy = valid + ".copy.safetensors";
            std::filesystem::copy_file(valid, copy);
            const auto copied = SourceLease::capture_verified({file(copy)});
            const tc::z_image::StreamingMetadata copied_metadata(copied);
            assert(original->digest() != copied->digest());
            assert(original->generation() != copied->generation());
            assert(original->artifact_digest() == copied->artifact_digest());
            assert(original_layout.digest == tc::streaming::compile_layout(
                config(), copied_metadata.describe_verified(work)).digest);
            const tc::z_image::StreamingPlanView copied_plan(copied, config(), work);
            assert(copied_plan.layout().digest == verified_plan.layout().digest);
            assert(copied_plan.lease().generation() != verified_plan.lease().generation());
            assert(original_layout.digest != tc::streaming::compile_layout(
                config(), original_metadata.describe(work)).digest);
            auto changed_work = work;
            changed_work.steps += 1;
            assert(original_layout.digest != tc::streaming::compile_layout(
                config(), copied_metadata.describe_verified(changed_work)).digest);
            assert(original_layout.digest != tc::streaming::compile_layout(
                config(3, 2, 1, 1, 2), portable).digest);
            const auto ready = SourceLease::capture_preverified({file(copy)});
            assert(ready->verification_bytes_read() == 0);
            assert(ready->artifact_digest() == original->artifact_digest());
            const tc::z_image::StreamingPlanView ready_plan(ready, config(), work);
            assert(ready_plan.layout().digest == verified_plan.layout().digest);
            // Serialized caller hashes do not grant a content proof.
            auto claimed = file(copy);
            claimed.content_digest = copied->file("transformer").content_digest;
            const tc::z_image::StreamingMetadata untrusted(SourceLease::capture({claimed}));
            rejects([&] { (void) untrusted.describe_verified(work); },
                    "artifact_verification_required");
            const int fd = ::open(copy.c_str(), O_WRONLY);
            assert(fd >= 0);
            const char changed = 1;
            assert(::pwrite(fd, &changed, 1,
                            copied->file("transformer").bytes - 1) == 1);
            assert(::close(fd) == 0);
            rejects([&] { (void) copied_metadata.describe_verified(work); },
                    "checkpoint_changed");
            const auto updated = SourceLease::capture_verified({file(copy)});
            const tc::z_image::StreamingMetadata updated_metadata(updated);
            assert(original_layout.digest != tc::streaming::compile_layout(
                config(), updated_metadata.describe_verified(work)).digest);
            std::filesystem::remove(copy);
            rejects([&] { (void) updated_metadata.describe_verified(work); },
                    "checkpoint_changed");
            // Non-transformer inputs also belong to the checkpoint identity.
            const std::string auxiliary_path = valid + ".tokenizer";
            { std::ofstream out(auxiliary_path); out << "tokenizer-v1"; }
            auto auxiliary = file(auxiliary_path);
            auxiliary.logical_id = "tokenizer";
            const auto complete = SourceLease::capture_verified({file(valid), auxiliary});
            const tc::z_image::StreamingMetadata complete_metadata(complete);
            const auto complete_layout = tc::streaming::compile_layout(
                config(), complete_metadata.describe_verified(work));
            { std::ofstream out(auxiliary_path); out << "tokenizer-v2"; }
            rejects([&] { (void) complete_metadata.describe_verified(work); },
                    "checkpoint_changed");
            const auto new_complete = SourceLease::capture_verified({auxiliary, file(valid)});
            const tc::z_image::StreamingMetadata new_complete_metadata(new_complete);
            assert(complete_layout.digest != tc::streaming::compile_layout(
                config(), new_complete_metadata.describe_verified(work)).digest);
            std::filesystem::remove(auxiliary_path);
        }
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

        const tc::z_image::StreamingPlanView single_slot(
            valid, config(0, 1), work);
        const auto &single_stage = single_slot.layout().stages.at(0);
        assert(single_stage.prefix == 0 && single_stage.slot_count == 1 &&
               single_stage.groups.size() == 30 &&
               single_stage.peak_pool_bytes == metadata.block_bytes());
        for (const auto &group : single_stage.groups)
            assert(group.slot == 0 && group.pool == 0 &&
                   group.blocks.size() == 1);
        assert(single_slot.layout().digest != plan.layout().digest);

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
                          valid, config(3, 3), work); },
                "slot count");
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 2, 2), work); },
                "group size");
        for (uint32_t distance : {0u, 1u}) {
            for (uint32_t workers : {1u, 2u}) {
                const tc::z_image::StreamingPlanView parallel(
                    valid, config(3, 2, 1, distance, workers), work);
                const auto &actual = parallel.layout().stages.front();
                assert(actual.distance == distance && actual.workers == workers);
                assert(actual.slot_count == 2 && actual.pools.size() == 1);
                assert(actual.groups.size() == stage.groups.size());
            }
        }
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 1, 1, 1, 1), work); },
                "prefetch_distance");
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 1, 1, 0, 2), work); },
                "io_workers");
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 2, 1, 2, 1), work); },
                "prefetch_distance");
        rejects([&] { tc::z_image::StreamingPlanView value(
                          valid, config(3, 2, 1, 1, 3), work); },
                "io_workers");
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
                     "projection, shared SourceLease, K1/K2 G1 layouts and K2 D0/D1 Q1/Q2, "
                     "portable verified layouts, malformed metadata and stale snapshot rejection; layout="
                  << plan.layout().digest << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
