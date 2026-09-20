#include "models/z_image/weight_stream.hpp"
#include "models/z_image/streaming_descriptor.hpp"
#include <bit>
#include <cassert>
#include <dirent.h>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <set>
#include <unistd.h>
using namespace tc;

size_t fds() {
    DIR *dir = opendir("/dev/fd"); assert(dir);
    size_t count = 0;
    while (auto *entry = readdir(dir)) if (entry->d_name[0] != '.') ++count;
    closedir(dir); return count;
}
float value(unsigned branch, unsigned projection) {
    return std::bit_cast<float>(uint32_t(0x3f00 + branch + projection) << 16);
}
void verify(const Weights &weights) {
    for (const auto &name : weights.sorted_keys()) {
        const auto &tensor = weights.at(name);
        assert(tensor.dtype() == mx::bfloat16);
        if (name.find(".feed_forward.w") == std::string::npos || name.starts_with("context_refiner.")) {
            require(mx::sum(mx::abs(mx::astype(tensor, mx::float32))).item<float>() == 0, "unchanged field differs");
            if (name.starts_with("context_refiner."))
                assert(tensor.size() == 3840 * 10240);
            continue;
        }
        const auto dot = name.find('.');
        const unsigned block = std::stoul(name.substr(dot + 1, name.find('.', dot + 1) - dot - 1));
        const unsigned branch = name.starts_with("noise_refiner.") ? block : block + 2;
        const unsigned projection = name[name.find(".feed_forward.w") + 15] - '0';
        assert(projection >= 1 && projection <= 3);
        const mx::Shape expected_shape = projection == 2 ? mx::Shape{3840, 1} : mx::Shape{1, 3840};
        assert(tensor.shape() == expected_shape);
        std::vector<float> expected(3840, 0.f);
        if (projection == 2) for (auto i : {0, 1024, 3839}) expected[i] = value(branch, projection);
        else for (auto i : {0, 2000, 3839}) expected[i] = value(branch, projection);
        const auto oracle = mx::array(expected.data(), expected_shape, mx::float32);
        require(mx::array_equal(mx::astype(tensor, mx::float32), oracle).item<bool>(), "GPU suffix value differs: " + name);
    }
}
int main(int argc, char **argv) {
    assert(argc == 2);
    std::atomic<bool> cancelled{false}, worker_cancel{false};
    streaming::SourceFileIdentity identity;
    identity.logical_id = "transformer"; identity.path = argv[1];
    std::shared_ptr<const z_image::GpuSuffixSource> source;
    {
        auto lease = streaming::SourceLease::capture_verified({identity});
        z_image::StreamingMetadata metadata(lease);
        source = metadata.materialize_gpu_suffix({512, 512, 64, 9}, 10239, cancelled);
    }
    mx::set_default_device(mx::Device::gpu);
    mx::eval(mx::ones({1}));
    mx::synchronize();
    const auto before = fds();
    const uint64_t suffix_bytes = 3840 * 2;
    const uint64_t block_bytes = 3 * suffix_bytes + 10 * 256;
    const uint64_t fixed_bytes = 256 + 6 * suffix_bytes + 6ull * 3840 * 10240 * 2;
    auto event = [](const std::string &, int, int) {};
    // Invalid budget and cancellation after fixed weights load close all reader
    // fds, retain no output weights, and leave the immutable source reusable.
    for (bool cancel_load : {false, true}) {
        Weights fixed;
        bool caught = false;
        try {
            ZImageWeightStream bad(source, 2, 2, cancel_load ? 0 : fixed_bytes + 4 * block_bytes - 1, 0, fixed,
                [&](const std::string &phase, int index, int) {
                    if (cancel_load && phase == "load_z_image_stream" && index == 0) cancelled = true;
                }, cancelled);
        } catch (const Cancelled &) { assert(cancel_load); caught = true; }
        catch (const std::invalid_argument &) { assert(!cancel_load); caught = true; }
        assert(caught && fixed.sorted_keys().empty());
        cancelled = false;
        assert(fds() == before);
    }
    for (unsigned slots : {1u, 2u}) {
        Weights fixed;
        {
            ZImageWeightStream reader(source, 2, slots, fixed_bytes + (2 + slots) * block_bytes, 0, fixed, event, cancelled);
            assert(reader.metrics().pinned_blocks == 2 && reader.metrics().refill_slots == slots);
            assert(reader.metrics().block_bytes == block_bytes);
            assert(reader.metrics().activation_reserve_bytes == fixed_bytes);
            assert(reader.metrics().estimated_working_set_bytes == fixed_bytes + (2 + slots) * block_bytes);
            assert(reader.metrics().request_bytes_loaded == fixed_bytes - 128 + 2 * block_bytes);
            assert(reader.metrics().request_pack_read_bytes == 0 && reader.metrics().request_pack_write_bytes == 0);
            assert(reader.metrics().mlp_prefix_channels == 10239);
            verify(fixed);
            for (unsigned i = 0; i < 2; ++i) verify(reader.prefix_weights(i));
            reader.create_exact_pool(slots, block_bytes);
            std::set<const void *> buffers;
            for (unsigned pass = 0; pass < 2; ++pass) {
                reader.reset_metrics();
                for (unsigned block = 2; block < 30; block += slots) {
                    std::vector<std::future<uint64_t>> pending;
                    for (unsigned slot = 0; slot < slots; ++slot)
                        pending.push_back(std::async(std::launch::async, [&, slot, block] {
                            return reader.fill_exact(slot, block + slot, &worker_cancel);
                        }));
                    for (unsigned slot = 0; slot < slots; ++slot) {
                        assert(pending[slot].get() == block_bytes);
                        auto weights = reader.bind_exact(slot, block + slot);
                        verify(weights); // GPU consumers complete before either slot is reused.
                        for (const auto &key : weights.sorted_keys()) buffers.insert(weights.at(key).data<char>());
                    }
                }
                assert(reader.metrics().request_bytes_loaded == 28 * block_bytes);
                assert(reader.metrics().request_slot_fills == 28 && reader.metrics().request_slot_allocations == 0);
            }
            assert(buffers.size() == slots * 13);
            worker_cancel = true;
            bool stopped = false;
            try { reader.fill_exact(0, 2, &worker_cancel); } catch (const std::exception &) { stopped = true; }
            assert(stopped && reader.metrics().request_slot_fills == 28);
            worker_cancel = false;
            reader.fill_exact(0, 2, &worker_cancel);
            verify(reader.bind_exact(0, 2));
            reader.check_unchanged();
            mx::synchronize();
            reader.destroy_exact_pool();
        }
        fixed.clear(); mx::clear_cache();
        assert(fds() == before);
    }
    // The reader retains the source after the caller releases its reference.
    {
        Weights fixed;
        ZImageWeightStream reader(source, 0, 1, 0, 0, fixed, event, cancelled);
        source.reset();
        reader.create_exact_pool(1, block_bytes);
        reader.fill_exact(0, 0, &worker_cancel);
        verify(reader.bind_exact(0, 0));
        const auto moved = std::string(argv[1]) + ".moved";
        assert(!rename(argv[1], moved.c_str()));
        bool rejected = false;
        try { reader.fill_exact(0, 1, &worker_cancel); } catch (const std::exception &) { rejected = true; }
        assert(rejected);
        mx::synchronize(); reader.destroy_exact_pool();
    }
    std::cout << "PASS verified GPU suffix reader: fixed/context/prefix values, K1/K2 concurrent fills, 2 passes, buffer reuse, I/O counts, cancel/retry and source ownership\n";
}
