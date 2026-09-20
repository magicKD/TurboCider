#include "models/z_image/weight_stream.hpp"
#include <iostream>
#include <set>
#include <fstream>
#include <thread>

void convrot_test(const char *path, unsigned depth, bool suffix) {
    std::atomic<bool> cancel{false};
    auto event = [](const std::string &, int, int) {};
    tc::Weights reference, fixed;
    // First obtain the exact compact layout; budget 0 deliberately streams all.
    uint64_t block_bytes, reserved;
    {
        tc::ZImageWeightStream sizing(path, 0, 0, fixed, event, cancel, suffix ? 256 : 0, depth);
        block_bytes = sizing.metrics().block_bytes;
        reserved = sizing.metrics().activation_reserve_bytes;
    }
    reference.load_file(path);
    reference.cast_unquantized_float32(tc::mx::bfloat16);
    reference.pack_convrot_q8();
    reference.materialize();
    fixed.clear();
    const auto budget = reserved + block_bytes * (depth + 4);
    tc::ZImageWeightStream stream(path, budget, 0, fixed, event, cancel, suffix ? 256 : 0, depth);
    tc::require(stream.metrics().pinned_blocks == 3 && stream.metrics().refill_slots == depth + 1,
                "ConvRot prefetch budget did not include all slots");
    auto verify = [&](const tc::Weights &weights) {
        for (const auto &name : weights.sorted_keys()) {
            const auto &actual = weights.at(name);
            auto expected = reference.at(name);
            const bool compact = suffix && !name.starts_with("context_refiner.") &&
                                 name.find(".feed_forward.") != std::string::npos &&
                                 !name.ends_with(".comfy_quant");
            if (compact && (name.find(".w1.") != std::string::npos || name.find(".w3.") != std::string::npos))
                expected = tc::slice_axis(expected, 0, 256, 512);
            else if (compact) {
                const int first = name.ends_with(".weight") ? 64 : 8;
                expected = tc::slice_axis(expected, 1, first, first * 2);
            }
            tc::mx::eval(expected);
            tc::require(actual.shape() == expected.shape() && actual.dtype() == expected.dtype(),
                        "ConvRot stream layout mismatch: " + name);
            tc::require(tc::mx::array_equal(actual, expected).item<bool>(),
                        "ConvRot CPU packing differs from resident MLX packing: " + name);
        }
    };
    verify(fixed);
    std::set<const void *> buffers;
    for (int pass = 0; pass < 3; ++pass) {
        stream.reset_metrics();
        stream.begin_pass();
        for (int layer = 0; layer < 30; ++layer) {
            auto weights = stream.acquire(layer);
            verify(weights); // evaluated on the GPU before slot reuse
            if (layer >= 3)
                for (const auto &name : weights.sorted_keys()) buffers.insert(weights.at(name).data<char>());
        }
        tc::require(stream.metrics().request_slot_refills == 27 && stream.metrics().request_slot_allocations == 0,
                    "ConvRot refill/allocation count mismatch");
    }
    tc::require(buffers.size() == (depth + 1) * 31, "ConvRot buffers grew across passes");
    tc::Weights all_fixed;
    tc::ZImageWeightStream resident(path, reserved + 31 * block_bytes, 0, all_fixed, event, cancel,
                                    suffix ? 256 : 0, depth);
    tc::require(resident.metrics().fully_resident && resident.metrics().refill_slots == 0,
                "sufficient budget did not retain all INT8 blocks");
    resident.reset_metrics();
    resident.begin_pass();
    for (int layer = 0; layer < 30; ++layer) verify(resident.acquire(layer));
    tc::require(resident.metrics().request_bytes_loaded == 0, "fully resident INT8 reloaded weights");
    tc::Weights small_fixed;
    tc::ZImageWeightStream clamped(path, reserved + 2 * block_bytes, 0, small_fixed, event, cancel,
                                   suffix ? 256 : 0, depth);
    tc::require(clamped.metrics().refill_slots == 2 && clamped.metrics().pinned_blocks == 0,
                "tight budget failed to reduce lookahead to two reusable slots");
    clamped.begin_pass();
    for (int layer = 0; layer < 30; ++layer) verify(clamped.acquire(layer));
    stream.begin_pass();
    cancel = true;
    bool cancelled = false;
    try { stream.acquire(0); } catch (const tc::Cancelled &) { cancelled = true; }
    tc::require(cancelled, "cancelled ConvRot stream kept executing");
    std::cout << "PASS: ConvRot exact GPU values, bounded lookahead, suffix, residency and cancellation\n";
}

void suffix_test(const char *path, bool cancel_pack) {
    std::atomic<bool> cancel{false};
    tc::Weights fixed;
    auto event = [](const std::string &, int, int) {};
    if (cancel_pack) {
        bool caught = false;
        try {
            tc::ZImageWeightStream interrupted(path, 1368, 0, fixed,
                [&](const std::string &phase, int done, int) {
                    if (phase == "pack_z_image_suffix" && done == 1) cancel = true;
                }, cancel, 2);
        } catch (const tc::Cancelled &) { caught = true; }
        tc::require(caught && fixed.bytes() == 0, "suffix packing cancellation leaked weights");
        cancel = false;
    }
    tc::ZImageWeightStream stream(path, 1368, 0, fixed, event, cancel, 2);
    auto m = stream.metrics();
    tc::require(m.block_bytes == 176 && m.pinned_blocks == 3 && m.streamed_blocks == 27,
                "suffix working set planner used full MLP size");
    tc::require(m.mlp_prefix_channels == 2 && m.suffix_pack_bytes == 32 * 32 &&
                    m.request_pack_read_bytes == 32 * 48 && m.request_pack_write_bytes == 32 * 32,
                "suffix preparation byte accounting mismatch");
    auto verify = [](const tc::Weights &weights, int base, bool trim) {
        for (const auto &name : weights.sorted_keys()) {
            const auto &tensor = weights.at(name);
            const bool mlp = name.find("feed_forward") != std::string::npos;
            const bool down = name.find(".w2.") != std::string::npos;
            const bool compact = mlp && trim && !name.starts_with("context_refiner.");
            const int original_cols = mlp ? (down ? 6 : 4) : 2;
            const int rows = mlp ? (down || compact ? 4 : 6) : 2;
            const int cols = compact ? 4 : original_cols;
            tc::require(tensor.shape() == tc::mx::Shape{rows, cols}, "wrong compact tensor shape");
            auto values = tc::mx::astype(tensor, tc::mx::float32);
            tc::mx::eval(values);
            int layer_base = base;
            if (!name.starts_with("layers."))
                layer_base = name.find(".1.") != std::string::npos ? 2 : 1;
            for (int row = 0; row < rows; ++row)
                for (int col = 0; col < cols; ++col) {
                    const auto expected = float(layer_base +
                        (row + (compact && !down ? 2 : 0)) * original_cols +
                        col + (compact && down ? 2 : 0));
                    tc::require(values.data<float>()[row * cols + col] == expected,
                                "incorrect suffix values or stale buffer: " + name);
                }
        }
    };
    verify(fixed, 1, true);
    std::set<const void *> buffers;
    for (int pass = 0; pass < 3; ++pass) {
        stream.reset_metrics();
        stream.begin_pass();
        for (int layer = 0; layer < 30; ++layer) {
            auto weights = stream.acquire(layer);
            verify(weights, layer + 1, true);
            if (layer >= 3)
                for (const auto &name : weights.sorted_keys()) buffers.insert(weights.at(name).data<char>());
        }
        auto m = stream.metrics();
        tc::require(m.request_bytes_loaded == 27 * 176 && m.request_slot_allocations == 0 &&
                        m.request_pack_read_bytes == 0 && m.request_pack_write_bytes == 0 &&
                        m.request_pack_seconds == 0 && m.request_slot_refills == 27,
                    "suffix was repacked or full weights were read in a warm pass");
    }
    tc::require(buffers.size() == 26, "compact buffers grew across passes");
    // The derived down projection must never mask a changed source checkpoint.
    std::ofstream(path, std::ios::app).put('x');
    bool changed = false;
    try { stream.begin_pass(); }
    catch (const std::exception &e) { changed = std::string(e.what()).find("changed") != std::string::npos; }
    tc::require(changed, "modified source reused compact suffix");
    std::cout << "PASS: compact suffix values, source identity, read bytes, two slots and cancellation\n";
}

int main(int argc, char **argv) {
    try {
        tc::require(argc == 2 || argc == 3, "expected checkpoint [--suffix|--cancel-pack]");
        tc::configure_streams();
        if (argc == 3) {
            const std::string mode = argv[2];
            if (mode.starts_with("--convrot-")) {
                convrot_test(argv[1], unsigned(mode.at(10) - '0'), mode.ends_with("-suffix"));
                return 0;
            }
            suffix_test(argv[1], std::string(argv[2]) == "--cancel-pack");
            return 0;
        }
        std::atomic<bool> cancel{false};
        auto event = [](const std::string &, int, int) {};
        for (uint32_t slots : {1u, 2u}) {
            tc::Weights exact_fixed;
            const uint64_t expected_bytes = 8 + (3 + slots) * 13 * 8;
            tc::ZImageWeightStream exact(
                argv[1], 3, slots, expected_bytes, 0, exact_fixed,
                event, cancel);
            const auto initial = exact.metrics();
            tc::require(initial.pinned_blocks == 3 &&
                            initial.refill_slots == slots &&
                            initial.estimated_working_set_bytes == expected_bytes,
                        "exact working set is incorrect");
            exact.create_exact_pool(slots, initial.block_bytes);
            for (uint32_t slot = 0; slot < slots; ++slot) {
                tc::require(exact.fill_exact(slot, 3 + slot, nullptr) == 13 * 8,
                            "exact fill byte count is incorrect");
                auto weights = exact.bind_exact(slot, 3 + slot);
                tc::require(weights.sorted_keys().size() == 13,
                            "exact fill lost block tensors");
                auto sum = tc::Tensor(0.f);
                for (const auto &name : weights.sorted_keys())
                    sum = sum + tc::mx::sum(tc::mx::astype(
                        weights.at(name), tc::mx::float32));
                tc::require(sum.item<float>() == float(4 + slot) * 4.f * 13.f,
                            "exact fill produced incorrect values");
            }
            if (slots == 2) {
                constexpr uint32_t rounds = 1000;
                std::atomic<unsigned> ready{0};
                std::array<std::thread, 2> readers;
                std::array<std::exception_ptr, 2> failures{};
                for (uint32_t slot = 0; slot < 2; ++slot) {
                    readers[slot] = std::thread([&, slot] {
                        ready.fetch_add(1);
                        while (ready.load() != 2) std::this_thread::yield();
                        try {
                            for (uint32_t i = 0; i < rounds; ++i)
                                exact.fill_exact(slot, 3 + (i + slot) % 27, nullptr);
                        } catch (...) { failures[slot] = std::current_exception(); }
                    });
                }
                for (auto &reader : readers) reader.join();
                for (auto &failure : failures) if (failure) std::rethrow_exception(failure);
                tc::require(exact.metrics().request_slot_fills == 2 + 2 * rounds &&
                            exact.metrics().request_slot_refills == 2 + 2 * rounds,
                            "parallel exact refill accounting lost updates");
                tc::require(exact.metrics().request_bytes_loaded ==
                            initial.request_bytes_loaded + (2 + 2 * rounds) * 13 * 8,
                            "parallel exact byte accounting mismatch");
                for (uint32_t slot = 0; slot < 2; ++slot) {
                    const uint32_t block = 3 + (rounds - 1 + slot) % 27;
                    auto weights = exact.bind_exact(slot, block);
                    auto sum = tc::Tensor(0.f);
                    for (const auto &name : weights.sorted_keys())
                        sum = sum + tc::mx::sum(tc::mx::astype(weights.at(name), tc::mx::float32));
                    tc::require(sum.item<float>() == float(block + 1) * 4.f * 13.f,
                                "parallel exact readers corrupted independent slots");
                }
            }
            std::atomic<bool> worker_cancel{true};
            bool rejected = false;
            try { exact.fill_exact(0, 5, &worker_cancel); }
            catch (const std::exception &e) {
                rejected = std::string(e.what()) == "streaming_cancelled";
            }
            tc::require(rejected, "exact fill ignored worker cancellation");
            exact.destroy_exact_pool();
            tc::require(exact.metrics().request_slot_allocations == slots,
                        "exact pool allocated extra backing");
        }

        tc::Weights fixed;
        // 5 block-sized allocations: 3 retained layers and two refill slots.
        tc::ZImageWeightStream stream(argv[1], 8 + 5 * 13 * 8, 0, fixed, event, cancel);
        tc::require(stream.metrics().pinned_blocks == 3, "wrong pinned count");
        tc::require(stream.metrics().refill_slots == 2, "not double buffered");
        std::set<const void *> buffers;
        for (int pass = 0; pass < 3; ++pass) {
            stream.reset_metrics();
            stream.begin_pass();
            for (int layer = 0; layer < 30; ++layer) {
                auto weights = stream.acquire(layer);
                float expected = 0;
                auto sum = tc::Tensor(0.f);
                for (const auto &name : weights.sorted_keys()) {
                    const auto &tensor = weights.at(name);
                    sum = sum + tc::mx::sum(tc::mx::astype(tensor, tc::mx::float32));
                    expected += float(layer + 1) * 4;
                    if (layer >= 3) buffers.insert(tensor.data<char>());
                }
                tc::require(sum.item<float>() == expected, "streamed tensor overwritten or stale");
            }
            const auto &m = stream.metrics();
            tc::require(m.request_slot_allocations == 0 && m.request_slot_refills == 27,
                        "unexpected allocation or refill count");
            tc::require(m.request_bytes_loaded == 27 * 13 * 8, "wrong read accounting");
        }
        tc::require(buffers.size() == 26, "buffers grew across layers or passes");
        stream.begin_pass();
        cancel = true;
        bool cancelled = false;
        try { stream.acquire(0); } catch (const tc::Cancelled &) { cancelled = true; }
        tc::require(cancelled, "cancelled stream kept executing");
        std::cout << "PASS: exact GPU values, two reusable slots, repeated passes and cancellation\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
