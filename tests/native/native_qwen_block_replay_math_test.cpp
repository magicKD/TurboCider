// CPU-only BF16 replay scanner checks. The fixture uses known BF16 bit patterns
// and byte offsets; it does not create a backend, load a model, or dispatch work.
#include "../../tools/native/native_qwen_block_replay_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace math = tc::qwen21::block_replay_math;
namespace {
size_t checks = 0;

void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void rejects(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, message);
}

void close(double actual, double expected, const char* message) {
    check(std::isfinite(actual) && std::abs(actual - expected) <=
          std::max(1e-14, std::abs(expected) * 1e-13), message);
}

template<size_t Size>
void put(std::array<std::byte, Size>& storage, size_t offset, uint16_t bits) {
    std::memcpy(storage.data() + offset, &bits, sizeof(bits));
}

template<size_t Size>
math::View contiguous(const std::array<uint16_t, Size>& values) {
    return {reinterpret_cast<const std::byte*>(values.data()), sizeof(values),
            1, 1, Size, sizeof(values), sizeof(values)};
}

void test_known_difference() {
    // BF16 constants for 1, 2, 3, 4, with only the first candidate value changed.
    const std::array<uint16_t, 4> reference{0x3f80, 0x4000, 0x4040, 0x4080};
    const std::array<uint16_t, 4> candidate{0x4000, 0x4000, 0x4040, 0x4080};
    const auto ref = contiguous(reference);
    math::validate(ref);
    const auto exact = math::compare(ref, ref);
    check(exact.values == 4 && exact.relative_rms == 0 && exact.max_abs == 0,
          "exact BF16 match has nonzero error or wrong value count");
    const auto difference = math::compare(contiguous(candidate), ref);
    close(difference.relative_rms, std::sqrt(1.0 / 30.0),
          "known BF16 relative RMS must use reference energy");
    close(difference.max_abs, 1.0, "known BF16 maximum absolute difference");

}

void test_padded_offsets() {
    // Poison prefix, row padding, head padding and suffix with BF16 NaN bits.
    // Both views end exactly after the final logical value, before tail padding.
    alignas(uint16_t) std::array<std::byte, 44> reference;
    alignas(uint16_t) std::array<std::byte, 32> candidate;
    reference.fill(std::byte{0xff});
    candidate.fill(std::byte{0xff});
    // Each head contains two rows of two values. The second head is negative.
    put(reference, 4, 0x3f80); put(reference, 6, 0x4000);
    put(reference, 12, 0x4040); put(reference, 14, 0x4080);
    put(reference, 28, 0xbf80); put(reference, 30, 0xc000);
    put(reference, 36, 0xc040); put(reference, 38, 0xc080);
    put(candidate, 2, 0x3f80); put(candidate, 4, 0x4000);
    put(candidate, 8, 0x4040); put(candidate, 10, 0x4080);
    put(candidate, 18, 0xbf80); put(candidate, 20, 0xc000);
    put(candidate, 24, 0xc040); put(candidate, 26, 0xc080);
    const math::View ref{reference.data() + 4, 36, 2, 2, 2, 24, 8};
    const math::View selected{candidate.data() + 2, 26, 2, 2, 2, 16, 6};
    math::validate(ref);
    math::validate(selected);
    const std::array<size_t, 4> row_offsets{4, 12, 28, 36};
    size_t row_index = 0;
    bool rows_match = true;
    math::for_each_row(ref, [&](const std::byte* row, size_t row_bytes) {
        rows_match = rows_match && row_index < row_offsets.size() &&
                     row == reference.data() + row_offsets[row_index] && row_bytes == 4;
        ++row_index;
    });
    check(rows_match && row_index == 4, "logical row order or row payload includes padding");
    const auto exact = math::compare(selected, ref);
    check(exact.values == 8 && exact.relative_rms == 0 && exact.max_abs == 0,
          "padded heads or retained data offsets changed exact match");
    put(candidate, 2, 0x4000);
    const auto difference = math::compare(selected, ref);
    close(difference.relative_rms, std::sqrt(1.0 / 60.0),
          "padded BF16 view relative RMS");
    close(difference.max_abs, 1.0, "padded BF16 view maximum absolute difference");

    // Byte-backed views may start at an odd address: BF16 reads use memcpy.
    alignas(uint16_t) std::array<std::byte, 10> unaligned;
    unaligned.fill(std::byte{0xff});
    put(unaligned, 1, 0x3f80); put(unaligned, 3, 0x4000);
    put(unaligned, 5, 0x4040); put(unaligned, 7, 0x4080);
    const math::View odd{unaligned.data() + 1, 9, 1, 1, 4, 8, 8};
    math::validate(odd); // An odd trailing padding byte is also allowed.
    const std::array<uint16_t, 4> packed{0x3f80, 0x4000, 0x4040, 0x4080};
    const auto unaligned_match = math::compare(odd, contiguous(packed));
    check(unaligned_match.values == 4 && unaligned_match.relative_rms == 0 &&
          unaligned_match.max_abs == 0, "unaligned BF16 memcpy read failed");
}

void test_interleaved_heads() {
    // Physical token-major rows differ from the scanner's head-major traversal.
    const std::array<uint16_t, 12> packed{
        0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0,
        0xbf80, 0xc000, 0xc040, 0xc080, 0xc0a0, 0xc0c0};
    const std::array<size_t, 6> row_offsets{2, 18, 34, 8, 24, 40};
    alignas(uint16_t) std::array<std::byte, 46> storage;
    storage.fill(std::byte{0xff}); // NaNs in prefix, gaps and suffix must be ignored.
    for (size_t row = 0; row < row_offsets.size(); ++row) {
        put(storage, row_offsets[row], packed[row * 2]);
        put(storage, row_offsets[row] + 2, packed[row * 2 + 1]);
    }
    const math::View interleaved{storage.data() + 2, 42, 2, 3, 2, 6, 16};
    const math::View reference{reinterpret_cast<const std::byte*>(packed.data()),
                               sizeof(packed), 2, 3, 2, 12, 4};
    math::validate(interleaved);
    const auto exact = math::compare(interleaved, reference);
    check(exact.values == 12 && exact.relative_rms == 0 && exact.max_abs == 0,
          "interleaved padded heads differ from packed logical values");
    std::array<std::byte, sizeof(packed)> collected{};
    size_t row_index = 0;
    bool rows_match = true;
    math::for_each_row(interleaved, [&](const std::byte* row, size_t row_bytes) {
        if (row_index >= row_offsets.size() || row_bytes != 4)
            throw std::runtime_error("invalid interleaved logical row payload");
        rows_match = rows_match && row == storage.data() + row_offsets[row_index];
        std::memcpy(collected.data() + row_index * 4, row, row_bytes);
        ++row_index;
    });
    check(rows_match && row_index == 6 &&
          std::memcmp(collected.data(), packed.data(), sizeof(packed)) == 0,
          "interleaved row traversal did not collect packed logical bytes");
    auto invalid = interleaved;
    invalid.outer_stride_bytes = 2;
    rejects([&] { math::validate(invalid); }, "head stride shorter than BF16 row accepted");
    invalid = interleaved; invalid.row_stride_bytes = 8;
    rejects([&] { math::validate(invalid); }, "overlapping interleaved head/token strides accepted");
    invalid = interleaved; --invalid.bytes;
    rejects([&] { math::validate(invalid); }, "one-byte-short interleaved extent accepted");
}

void test_zero_and_wide_values() {
    const std::array<uint16_t, 4> zeros{0, 0, 0, 0};
    const std::array<uint16_t, 4> signed_zeros{0x8000, 0, 0x8000, 0};
    const auto exact = math::compare(contiguous(signed_zeros), contiguous(zeros));
    check(exact.values == 4 && exact.relative_rms == 0 && exact.max_abs == 0,
          "zero reference or signed zero comparison failed");
    const std::array<uint16_t, 4> negative_one{0xbf80, 0, 0, 0};
    const auto difference = math::compare(contiguous(negative_one), contiguous(zeros));
    close(difference.relative_rms, 1e10, "zero reference denominator floor");
    close(difference.max_abs, 1.0, "negative BF16 decoding");

    // The BF16 maximum is 255 * 2^120; the comparison value is 128 * 2^120.
    // Their squares overflow FP32, so finite results require double accumulation.
    const std::array<uint16_t, 1> maximum{0x7f7f};
    const std::array<uint16_t, 1> half{0x7f00};
    const auto wide = math::compare(contiguous(half), contiguous(maximum));
    close(wide.relative_rms, 127.0 / 255.0, "finite BF16 energy overflowed");
    close(wide.max_abs, std::ldexp(127.0, 120), "large BF16 decoding");
}

void test_nonfinite_and_shape() {
    const std::array<uint16_t, 4> finite{0x3f80, 0x4000, 0x4040, 0x4080};
    const auto valid = contiguous(finite);
    for (uint16_t bits : {uint16_t(0x7fc1), uint16_t(0x7f80), uint16_t(0xff80)}) {
        auto invalid = finite;
        invalid[3] = bits;
        rejects([&] { (void)math::compare(contiguous(invalid), valid); },
                "nonfinite candidate BF16 accepted");
        rejects([&] { (void)math::compare(valid, contiguous(invalid)); },
                "nonfinite reference BF16 accepted");
    }
    auto reshaped = valid;
    reshaped.rows = 2; reshaped.columns = 2; reshaped.row_stride_bytes = 4;
    math::validate(reshaped);
    rejects([&] { (void)math::compare(reshaped, valid); },
            "different logical rows and columns accepted");
    auto heads = valid;
    heads.outer = 2; heads.columns = 2;
    heads.row_stride_bytes = 4; heads.outer_stride_bytes = 4;
    math::validate(heads);
    rejects([&] { (void)math::compare(heads, reshaped); },
            "different logical head count accepted");
}

void test_invalid_views() {
    alignas(uint16_t) const std::array<std::byte, 36> storage{};
    const math::View valid{storage.data(), storage.size(), 2, 2, 2, 24, 8};
    const auto reject_change = [&](auto mutate, const char* message) {
        auto invalid = valid;
        mutate(invalid);
        rejects([&] { math::validate(invalid); }, message);
    };
    reject_change([](auto& v) { v.data = nullptr; }, "null BF16 data accepted");
    reject_change([](auto& v) { v.bytes = 0; }, "empty BF16 byte extent accepted");
    reject_change([](auto& v) { v.outer = 0; }, "empty head dimension accepted");
    reject_change([](auto& v) { v.rows = 0; }, "empty row dimension accepted");
    reject_change([](auto& v) { v.columns = 0; }, "empty column dimension accepted");
    reject_change([](auto& v) { v.row_stride_bytes = 0; }, "zero row stride accepted");
    reject_change([](auto& v) { v.row_stride_bytes = 2; }, "short row stride accepted");
    reject_change([](auto& v) { v.row_stride_bytes = 5; }, "odd BF16 row stride accepted");
    reject_change([](auto& v) { v.outer_stride_bytes = 0; }, "zero head stride accepted");
    reject_change([](auto& v) { v.outer_stride_bytes = 10; }, "short head stride accepted");
    reject_change([](auto& v) { v.outer_stride_bytes = 13; }, "odd BF16 head stride accepted");
    reject_change([](auto& v) { --v.bytes; }, "one-byte-short final extent accepted");

    const size_t maximum = std::numeric_limits<size_t>::max();
    auto arithmetic = valid;
    arithmetic.bytes = maximum;
    arithmetic.columns = maximum / 2 + 1;
    arithmetic.row_stride_bytes = arithmetic.outer_stride_bytes = maximum - 1;
    rejects([&] { math::validate(arithmetic); }, "column byte count overflow accepted");
    arithmetic = {storage.data(), maximum, 1, maximum / 4 + 2, 1, maximum - 1, 4};
    rejects([&] { math::validate(arithmetic); }, "row stride multiplication overflow accepted");
    arithmetic = {storage.data(), maximum, 1, 2, 2, maximum - 1, maximum - 1};
    rejects([&] { math::validate(arithmetic); }, "row extent addition overflow accepted");
    arithmetic = {storage.data(), maximum, maximum / 4 + 2, 1, 1, 4, 2};
    rejects([&] { math::validate(arithmetic); }, "head stride multiplication overflow accepted");
    arithmetic = {storage.data(), maximum, 2, 1, 2, maximum - 1, 4};
    rejects([&] { math::validate(arithmetic); }, "head extent addition overflow accepted");
    arithmetic = {storage.data(), maximum, maximum / 4 + 1, 2, 2, 8, 4};
    rejects([&] { math::validate(arithmetic); }, "logical BF16 value count overflow accepted");
    arithmetic = {reinterpret_cast<const std::byte*>(std::numeric_limits<uintptr_t>::max() - 1),
                  4, 1, 1, 2, 4, 4};
    rejects([&] { math::validate(arithmetic); }, "BF16 pointer extent overflow accepted");

}
} // namespace

int main() {
    try {
        test_known_difference();
        test_padded_offsets();
        test_interleaved_heads();
        test_zero_and_wide_values();
        test_nonfinite_and_shape();
        test_invalid_views();
        std::cout << "native_qwen_block_replay_math_cpu=PASS checks=" << checks
                  << " device_created=0 model_loads=0\n";
    } catch (const std::exception& error) {
        std::cerr << "native_qwen_block_replay_math_cpu=FAIL " << error.what() << '\n';
        return 1;
    }
}
