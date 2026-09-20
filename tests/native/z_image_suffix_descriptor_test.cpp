#include "models/z_image/streaming_descriptor.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace tc;

StreamingConfig config(unsigned prefix) {
    StreamingConfig c;
    c.enabled = true; c.schema_version = 1; c.selection = "manual"; c.retention = "request";
    c.stages["denoiser"] = {"streamed", 1, prefix == 29 ? 1u : 2u, prefix, 0, 1};
    return c;
}
template<class F> void rejects(F f) {
    try { f(); } catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("expected suffix metadata rejection");
}
int main(int argc, char **argv) {
    assert(argc == 3 || argc == 5);
    z_image::StreamingMetadata metadata(argv[1]);
    const z_image::StreamingWorkload work{512, 512, 64, 9};
    const auto exact_before = streaming::compile_layout(config(7), metadata.describe(work)).digest;
    for (unsigned a : {1u, 2560u, 5120u, 10239u}) {
        const auto plan = metadata.describe_gpu_suffix(work, a);
        const auto repeated = metadata.describe_gpu_suffix(work, a);
        assert(plan.recipe_digest == repeated.recipe_digest);
        assert(plan.packing.size() == 32);
        constexpr uint64_t full = 3840ull * 10240 * 2;
        const uint64_t suffix = 3840ull * (10240 - a) * 2;
        assert(plan.setup_read_bytes == 32 * full);
        assert(plan.setup_write_bytes == 32 * suffix);
        assert(plan.descriptor.artifacts.size() == 2);
        assert(plan.descriptor.artifacts[1].identity_kind == streaming::SourceIdentityKind::snapshot);
        assert(plan.descriptor.artifacts[1].identity == "recipe:" + plan.recipe_digest);
        for (size_t i = 0; i < plan.packing.size(); ++i) {
            const auto &r = plan.packing[i];
            assert(r.destination_offset == i * suffix && r.destination_bytes == suffix);
            assert(r.source.bytes == full && r.source.artifact == 0);
            assert(r.first_gpu_channel == a);
            const auto prefix = i < 2 ? "noise_refiner." + std::to_string(i)
                                      : "layers." + std::to_string(i - 2);
            assert(r.source.tensor == prefix + ".feed_forward.w2.weight");
        }
        const auto exact = metadata.describe(work);
        for (size_t block = 0; block < 30; ++block) {
            const auto &old_fields = exact.stages[0].blocks[block].fields;
            const auto &new_fields = plan.descriptor.stages[0].blocks[block].fields;
            assert(old_fields.size() == new_fields.size());
            for (size_t index = 0; index < old_fields.size(); ++index) {
                const auto &before = old_fields[index];
                const auto &after = new_fields[index];
                const auto &read = after.materialization->reads[0];
                if (!before.name.starts_with("feed_forward.w")) {
                    assert(before == after);
                } else {
                    assert(after.bytes == suffix && read.bytes == suffix);
                    assert(after.materialization->shape == read.shape);
                    if (before.name == "feed_forward.w2.weight") {
                        assert(read.artifact == 1 && read.offset == (block + 2) * suffix);
                        assert((read.shape == std::vector<uint64_t>{3840, 10240 - a}));
                        assert(plan.packing[block + 2].source == before.materialization->reads[0]);
                    } else {
                        assert(read.artifact == 0);
                        assert(read.offset == before.materialization->reads[0].offset + uint64_t(a) * 3840 * 2);
                        assert((read.shape == std::vector<uint64_t>{10240 - a, 3840}));
                    }
                }
            }
        }
        auto other_work = work;
        other_work.width = 256;
        assert(metadata.describe_gpu_suffix(other_work, a).recipe_digest == plan.recipe_digest);
        unsigned noise = 0, context = 0, derived = 0;
        for (const auto &f : plan.descriptor.stages[0].resident_fields) {
            const auto &m = *f.materialization;
            if (f.name.starts_with("noise_refiner.") && f.name.find(".feed_forward.w") != std::string::npos) {
                ++noise; assert(f.bytes == suffix);
                if (f.name.find(".w2.") != std::string::npos) {
                    ++derived; assert(m.reads[0].artifact == 1);
                } else assert(m.reads[0].artifact == 0);
            }
            if (f.name.starts_with("context_refiner.") && f.name.find(".feed_forward.w") != std::string::npos) {
                ++context; assert(f.bytes == full && m.reads[0].artifact == 0);
                assert(m.shape == m.reads[0].shape);
            }
        }
        // The sparse fixture has only the FFN fixed fields and one embedder.
        if (std::string(argv[2]) == "fixture") {
            assert(noise == 6 && context == 6 && derived == 2);
            for (unsigned prefix : {0u, 7u, 29u}) {
                const auto layout = streaming::compile_layout(config(prefix), plan.descriptor);
                const auto &stage = layout.stages[0];
                const uint64_t block = 3 * suffix + 10 * 256;
                assert(layout.materializations_complete);
                assert(stage.resident_bytes == 256 + 6 * suffix + 6 * full);
                assert(stage.prefix_bytes == prefix * block);
                assert(stage.peak_pool_bytes == (prefix == 29 ? 1 : 2) * block);
                assert(stage.suffix_content_bytes_per_pass == (30 - prefix) * block);
                assert(stage.source_read_bytes_per_pass == (30 - prefix) * block);
                assert(stage.prefix_source_read_bytes == prefix * block);
                assert(stage.resident_source_read_bytes == stage.resident_bytes);
            }
        } else {
            const auto layout = streaming::compile_layout(config(7), plan.descriptor);
            std::cout << "real a=" << a << " fixed=" << layout.stages[0].resident_bytes
                      << " prefix=" << layout.stages[0].prefix_bytes
                      << " pool=" << layout.stages[0].peak_pool_bytes
                      << " pack_read=" << plan.setup_read_bytes
                      << " pack_write=" << plan.setup_write_bytes << '\n';
        }
        assert(plan.recipe_digest != metadata.describe_gpu_suffix(work, a == 1 ? 2 : 1).recipe_digest);
    }
    assert(exact_before == streaming::compile_layout(config(7), metadata.describe(work)).digest);
    rejects([&] { metadata.describe_gpu_suffix(work, 0); });
    rejects([&] { metadata.describe_gpu_suffix(work, 10240); });
    if (argc == 5) for (int i = 3; i < argc; ++i) {
        z_image::StreamingMetadata bad(argv[i]);
        rejects([&] { bad.describe_gpu_suffix(work, 2560); });
    }
    std::cout << "PASS suffix metadata: 32 mappings, fixed/context/prefix/slots, setup/refill accounting, identities and exact isolation\n";
}
