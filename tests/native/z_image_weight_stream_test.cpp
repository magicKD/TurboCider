#include "models/z_image/weight_stream.hpp"
#include <iostream>
#include <set>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 2, "expected checkpoint");
        tc::configure_streams();
        std::atomic<bool> cancel{false};
        tc::Weights fixed;
        auto event = [](const std::string &, int, int) {};
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
