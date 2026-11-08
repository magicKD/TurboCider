#include "models/z_image/ane_row_tiling.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace tc;
    configure_streams();
    constexpr int channels = 7;
    for (int capacity : {1056, 1536, 4128}) {
        auto backing = mx::zeros({1, capacity, channels}, mx::float16);
        mx::eval(backing);
        for (int rows : {32, 1056, 1536, 1568, 1792, 2048}) {
            auto input = mx::astype(mx::reshape(
                mx::sin(mx::arange(rows * channels, mx::float32)), {1, rows, channels}), mx::float16);
            auto expected = mx::astype(input * Tensor(2.f) + Tensor(1.f), mx::float16);
            mx::eval(input, expected);
            int calls = 0;
            auto predict = [&](const Tensor &tile) {
                assert(tile.shape() == backing.shape());
                assert(tile.flags().row_contiguous && tile.dtype() == mx::float16);
                auto output = mx::contiguous(mx::astype(tile * Tensor(2.f) + Tensor(1.f), mx::float16));
                mx::eval(output);
                // Deliberately reuse and overwrite one output buffer, like
                // Core ML. Retaining lazy slices would corrupt earlier tiles.
                std::memcpy(backing.data<mx::float16_t>(), output.data<mx::float16_t>(),
                            size_t(capacity) * channels * sizeof(mx::float16_t));
                ++calls;
                return backing;
            };
            auto actual = z_image::predict_ane_rows(input, capacity, predict);
            mx::eval(actual);
            assert(calls == (rows + capacity - 1) / capacity);
            assert(mx::array_equal(expected, actual).item<bool>());
        }
    }
    std::cout << "PASS: complete long-caption FFN coverage, tail padding, and reused-output isolation\n";
}
