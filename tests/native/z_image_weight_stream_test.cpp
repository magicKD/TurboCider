#include "models/z_image/weight_stream.hpp"
#include <iostream>
#include <set>
#include <fstream>

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
            suffix_test(argv[1], std::string(argv[2]) == "--cancel-pack");
            return 0;
        }
        std::atomic<bool> cancel{false};
        auto event = [](const std::string &, int, int) {};
        {
            tc::Weights exact_fixed;
            // One refill slot is the low-memory exact layout.  It deliberately
            // serializes read/compute, but must allocate and account for only
            // one block-sized backing bundle.
            tc::ZImageWeightStream exact(
                argv[1], 3, 1, 8 + 4 * 13 * 8, 0, exact_fixed,
                event, cancel);
            const auto initial = exact.metrics();
            tc::require(initial.pinned_blocks == 3 &&
                            initial.refill_slots == 1 &&
                            initial.estimated_working_set_bytes ==
                                8 + 4 * 13 * 8,
                        "single-slot exact working set is incorrect");
            exact.create_exact_pool(1, initial.block_bytes);
            tc::require(exact.fill_exact(0, 3, nullptr) == 13 * 8,
                        "single-slot exact fill byte count is incorrect");
            auto weights = exact.bind_exact(0, 3);
            auto sum = tc::Tensor(0.f);
            for (const auto &name : weights.sorted_keys())
                sum = sum + tc::mx::sum(tc::mx::astype(
                    weights.at(name), tc::mx::float32));
            tc::require(sum.item<float>() == 4.f * 4.f,
                        "single-slot exact fill produced incorrect values");
            weights.clear();
            exact.destroy_exact_pool();
            tc::require(exact.metrics().request_slot_allocations == 1,
                        "single-slot exact pool allocated extra backing");
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
