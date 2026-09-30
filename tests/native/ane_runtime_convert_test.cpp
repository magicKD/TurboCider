#include "../../native/backends/ane_runtime_convert.hpp"
#include <array>
#include <cassert>
#include <limits>
#include <vector>

int main() {
    using namespace tc::ane;
    // Preserve padded/scattered inputs, odd row counts, scalar tails, output
    // bounds and both overflow gates. Retained after removing the parallel
    // candidate: these test the matrix wrapper, not a scheduling policy.
    for (size_t rows : {size_t(0), size_t(3), size_t(257)})
    for (size_t col_stride : {size_t(1), size_t(2)})
    for (auto dtype : {DType::BF16, DType::FP16}) {
        constexpr size_t cols = 4097;
        const size_t pitch = cols * col_stride + 7;
        std::vector<uint16_t> input(rows * pitch, 0x3c04);
        std::vector<uint16_t> converted(rows * cols + 1, 0xabcd), serial(converted);
        for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < cols; ++c)
            input[r * pitch + c * col_stride] = uint16_t((r * 29 + c * 13) % 0x7c00);
        const auto saved = input;
        for (float scale : {1.f, 4096.f}) {
            const auto expected = restore_fp16_matrix(input.data(), serial.data(), rows, cols,
                pitch, col_stride, dtype, scale, true);
            const auto actual = restore_fp16_matrix(input.data(), converted.data(), rows, cols,
                pitch, col_stride, dtype, scale);
            assert(actual.source_finite == expected.source_finite);
            assert(actual.restored_finite == expected.restored_finite);
            assert(converted == serial && input == saved && converted.back() == 0xabcd);
            const auto unused = restore_fp16_matrix(input.data(), nullptr, rows, cols,
                pitch, col_stride, dtype, scale);
            assert(unused.source_finite == expected.source_finite);
            assert(unused.restored_finite == expected.restored_finite);
        }
        if (rows) for (size_t task = 0; task < 4; ++task) {
            const size_t offset = (rows * task / 4) * pitch + col_stride * 3;
            for (uint16_t invalid : {uint16_t(0x7c00), uint16_t(0x7e00)}) {
                const auto before = input[offset];
                input[offset] = invalid;
                const auto actual = restore_fp16_matrix(input.data(), converted.data(), rows, cols,
                    pitch, col_stride, dtype, 1.f);
                assert(!actual.source_finite && !actual.restored_finite);
                input[offset] = before;
            }
        }
    }
    // Every BF16/FP16 bit pattern; include overflow, subnormals and NaN/Inf.
    std::vector<uint16_t> source(65536), vectorized(65536), scalar(65536);
    for (size_t i = 0; i < source.size(); ++i) source[i] = uint16_t(i);
    for (auto dtype : {DType::BF16, DType::FP16}) for (float scale : {1.f, .25f, .0625f}) {
        assert(!convert_fp16_row(source.data(), vectorized.data(), source.size(), dtype, false, scale));
        assert(!convert_fp16_row(source.data(), scalar.data(), source.size(), dtype, true, scale));
        for (size_t i = 0; i < source.size(); ++i) {
            if ((scalar[i] & 0x7c00) != 0x7c00) assert(vectorized[i] == scalar[i]);
            else assert((vectorized[i] & 0x7c00) == 0x7c00);
        }
    }
    std::vector<float> fp32(109);
    for (size_t i = 0; i < fp32.size(); ++i) fp32[i] = (float(i) - 47.f) / 113.f;
    for (size_t length = 0; length <= fp32.size(); ++length) {
        vectorized.assign(fp32.size() + 1, 0xabcd);
        scalar.assign(fp32.size() + 1, 0xabcd);
        assert(convert_fp16_row(fp32.data(), vectorized.data(), length, DType::FP32));
        assert(convert_fp16_row(fp32.data(), scalar.data(), length, DType::FP32, true));
        assert(vectorized == scalar); // also checks tail writes stay in range
    }
    for (float bad : {65536.f, -65536.f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
        fp32[3] = bad;
        assert(!convert_fp16_row(fp32.data(), vectorized.data(), fp32.size(), DType::FP32));
    }
    assert(round_bf16(1.f) == 0x3f80);
    assert(round_bf16(std::bit_cast<float>(0x3f808000u)) == 0x3f80); // tie to even
    assert(round_bf16(std::bit_cast<float>(0x3f818000u)) == 0x3f82);
    assert(std::isfinite(std::bit_cast<float>(uint32_t(round_bf16(262144.f)) << 16)));
    for (auto dtype : {DType::BF16, DType::FP16}) for (float scale : {1.f, 16.f, 4096.f}) {
        vectorized.resize(source.size()); scalar.resize(source.size());
        const auto simd = restore_fp16_row(source.data(), vectorized.data(), source.size(), dtype, scale);
        const auto ref = restore_fp16_row(source.data(), scalar.data(), source.size(), dtype, scale, true);
        assert(simd.source_finite == ref.source_finite);
        assert(simd.restored_finite == ref.restored_finite);
        for (size_t i = 0; i < source.size(); ++i) {
            // Sign/payload of FP16 NaNs may canonicalize differently in SIMD.
            const auto exponent = dtype == DType::BF16 ? 0x7f80u : 0x7c00u;
            if ((scalar[i] & exponent) != exponent) assert(vectorized[i] == scalar[i]);
            else assert((vectorized[i] & exponent) == exponent);
        }
    }
    for (size_t length = 0; length < 33; ++length) {
        vectorized.assign(34, 0xabcd); scalar.assign(34, 0xabcd);
        restore_fp16_row(source.data(), vectorized.data(), length, DType::BF16, 16.f);
        restore_fp16_row(source.data(), scalar.data(), length, DType::BF16, 16.f, true);
        assert(vectorized == scalar);
    }
    // Validate-only must not weaken either overflow gate. Repeat EVERY FP16
    // bit pattern across one SIMD group plus a scalar tail, including finite
    // values that overflow only after restoration into FP16.
    std::array<uint16_t, 9> repeated{}, restored{};
    for (auto dtype : {DType::BF16, DType::FP16}) for (float scale : {
            0.f, 1.f, 16.f, -16.f, 4096.f, -4096.f, 4097.f, 8192.f,
            std::numeric_limits<float>::max(), std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN()}) {
        for (uint32_t bits = 0; bits < 65536; ++bits) {
            repeated.fill(uint16_t(bits));
            const auto expected = restore_fp16_row(repeated.data(), restored.data(), repeated.size(), dtype, scale, true);
            for (bool scalar_only : {false, true}) {
                const auto inspected = restore_fp16_row(repeated.data(), nullptr, repeated.size(), dtype, scale, scalar_only);
                assert(inspected.source_finite == expected.source_finite);
                assert(inspected.restored_finite == expected.restored_finite);
            }
            for (auto value : repeated) assert(value == bits); // read-only input
        }
    }
    const auto empty = restore_fp16_row(nullptr, nullptr, 0, DType::BF16, 1.f);
    assert(empty.source_finite && empty.restored_finite);
    // All supported positive power-of-two headrooms, every FP16 bit pattern,
    // both SIMD and scalar tail. Finite outputs (including signed zero) must
    // match the scalar restoration BIT FOR BIT, not just a tolerance.
    // Retained independently of the deferred integer-conversion experiment.
    std::array<uint16_t, 9> fast{};
    for (int exponent = 0; exponent <= 12; ++exponent) {
        const float scale = std::ldexp(1.f, exponent);
        for (uint32_t bits = 0; bits < 65536; ++bits) {
            repeated.fill(uint16_t(bits));
            const auto expected = restore_fp16_row(repeated.data(), restored.data(), repeated.size(), DType::BF16, scale, true);
            const auto actual = restore_fp16_row(repeated.data(), fast.data(), repeated.size(), DType::BF16, scale);
            assert(actual.source_finite == expected.source_finite);
            assert(actual.restored_finite == expected.restored_finite);
            if (expected.source_finite) assert(fast == restored);
            else for (size_t i = 0; i < fast.size(); ++i) {
                assert((fast[i] & 0x7f80) == 0x7f80);
                // A signaling/quiet NaN must never become infinity.
                assert(bool(fast[i] & 0x7f) == bool(restored[i] & 0x7f));
                if (!(fast[i] & 0x7f)) assert(fast[i] == restored[i]);
            }
        }
        // A special lane must not mask invalid data or change its neighbors.
        for (size_t lane = 0; lane < repeated.size(); ++lane) {
            for (uint16_t special : {uint16_t(0), uint16_t(0x8000), uint16_t(1),
                                    uint16_t(0x83ff), uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7e00)}) {
                repeated.fill(0x3c04); // BF16 tie, rounds toward even
                repeated[lane] = special;
                const auto expected = restore_fp16_row(repeated.data(), restored.data(), repeated.size(), DType::BF16, scale, true);
                const auto actual = restore_fp16_row(repeated.data(), fast.data(), repeated.size(), DType::BF16, scale);
                assert(actual.source_finite == expected.source_finite);
                assert(actual.restored_finite == expected.restored_finite);
                for (size_t i = 0; i < fast.size(); ++i)
                    if ((repeated[i] & 0x7c00) != 0x7c00) assert(fast[i] == restored[i]);
            }
        }
    }
}
