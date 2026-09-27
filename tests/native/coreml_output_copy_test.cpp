#include "backends/coreml_output_copy.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

static void check(size_t rows, size_t channels, size_t rs, size_t cs) {
    constexpr uint16_t guard = 0x5a3c;
    std::vector<uint16_t> source((rows - 1) * rs + (channels - 1) * cs + 1);
    std::mt19937 random(42);
    for (auto &value : source) value = static_cast<uint16_t>(random());
    const auto original = source;
    for (bool optimized : {false, true}) {
        std::vector<uint16_t> output(rows * channels + 32, guard);
        tc::copy_coreml_fp16(output.data() + 16, source.data(), rows, channels, rs, cs, optimized);
        for (size_t i = 0; i < 16; ++i)
            assert(output[i] == guard && output[output.size() - 1 - i] == guard);
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < channels; ++c)
                assert(output[16 + r * channels + c] == original[r * rs + c * cs]);
        assert(source == original);
    }
}

int main(int argc, char **) {
    for (size_t rows : {1, 7, 31, 32, 33, 65, 1056})
        for (size_t channels : {1, 5, 31, 32, 33, 70, 3840}) {
            check(rows, channels, channels, 1);
            check(rows, channels, channels + 17, 1);
            check(rows, channels, 1, rows);
            check(rows, channels, 1, rows + 17);
            check(rows, channels, channels * 3 + 7, 3);
        }
    // Exhaust every FP16 bit pattern, rather than comparing only finite floats.
    std::vector<uint16_t> all(65536), copied(all.size());
    for (size_t i = 0; i < all.size(); ++i) all[i] = static_cast<uint16_t>(i);
    tc::copy_coreml_fp16(copied.data(), all.data(), 256, 256, 1, 256, true);
    for (size_t r = 0; r < 256; ++r)
        for (size_t c = 0; c < 256; ++c) assert(copied[r * 256 + c] == all[c * 256 + r]);
    std::cout << "PASS: bit-exact FP16 copies, padded/transposed/general strides, tile edges and canaries\n";
    if (argc > 1) {
        const size_t rows = 1056, channels = 3840;
        std::vector<uint16_t> source(rows * channels, 0x1234), output(source.size());
        auto volatile copy = &tc::copy_coreml_fp16;
        for (bool transposed : {false, true}) {
            for (bool optimized : {false, true}) {
                const auto begin = std::chrono::steady_clock::now();
                for (int i = 0; i < 288; ++i)
                    copy(output.data(), source.data(), rows, channels,
                         transposed ? 1 : channels, transposed ? rows : 1, optimized);
                std::cout << "transpose=" << transposed << " optimized=" << optimized << " seconds="
                          << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() << '\n';
            }
        }
    }
}
