#include "../../native/backends/ane_w8a8_math.hpp"
#include <iostream>

int main() {
    using namespace tc::ane;
    for (int size : {128, 512}) {
        std::vector<float> a(size), b(size);
        for (int i = 0; i < size; ++i) { a[i] = float(i % 7 - 3) / 8; b[i] = float(i % 11 - 5) / 8; }
        double before = 0; for (int i = 0; i < size; ++i) before += double(a[i]) * b[i];
        auto rotated = a; rotate_block(rotated, 20260930);
        // Independent dense Hadamard matrix, not the butterfly under test.
        const double norm = 1 / std::sqrt(double(size));
        for (int out = 0; out < size; ++out) {
            double expected = 0; for (int in = 0; in < size; ++in) expected += ((std::popcount(unsigned(out & in)) & 1) ? -1 : 1) * double(a[in]) * rotation_sign(20260930, in);
            expected *= norm;
            if (std::abs(rotated[out] - expected) > 2e-6) return 1;
        }
        rotate_block(a, 20260930); rotate_block(b, 20260930);
        double after = 0; for (int i = 0; i < size; ++i) after += double(a[i]) * b[i];
        if (std::abs(after - before) > 2e-5) return 1;
    }
    if (normalized_scale(0) != 0x5800 || normalized_scale(1e-30f) != 1) return 1;
    const auto scale = tc::gguf::float_to_fp16_rne(128.f);
    for (const auto &[value, expected] : {std::pair{.5f, 0}, {1.5f, 2}, {2.5f, 2}, {-.5f, 0}, {-1.5f, -2}, {-2.5f, -2}, {1000.f, 127}, {-1000.f, -127}})
        if (quantize_rotated(value, scale) != expected) return 1;
    std::cout << "PASS independent H128/H512 orthogonality/dense oracle, deterministic signs, scale tiny/zero, signed RNE ties/clipping\n";
}
