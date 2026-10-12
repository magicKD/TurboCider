#include "../../native/backends/private/ane_a8_single_pass.hpp"
#include <cfenv>
#include <iostream>
#include <limits>

namespace {
using namespace tc::ane;
namespace gguf = tc::gguf;
void require(bool ok, const char *reason) { if (!ok) throw std::runtime_error(reason); }
template<class F> void rejected(F fn) {
    try { fn(); } catch (const CapabilityError &) { return; }
    throw std::runtime_error("invalid single-pass flag accepted");
}
// Deliberately independent dense decoder and old two-pass traversal: each
// H128 is recomputed for codes after reducing all blocks for the row scale.
float decode(const uint8_t *at, DType dtype) {
    const uint32_t low = uint32_t(at[0]) | uint32_t(at[1]) << 8;
    if (dtype == DType::FP16) return gguf::fp16_to_float(uint16_t(low));
    if (dtype == DType::BF16) return std::bit_cast<float>(low << 16);
    return std::bit_cast<float>(low | uint32_t(at[2]) << 16 | uint32_t(at[3]) << 24);
}
std::array<float, 128> old_block(const uint8_t *source, size_t pitch, DType dtype, const W8StageSpec &spec,
                                int row, int block, std::span<const float> scale, uint32_t &flags) {
    const size_t item = dtype == DType::FP32 ? 4 : 2;
    std::array<float, 128> v;
    for (int lane = 0; lane < 128; ++lane) {
        const int col = spec.column_begin + block * 128 + lane;
        float value = decode(source + size_t(spec.row_begin + row) * pitch + size_t(col) * item, dtype);
        if (!std::isfinite(value)) { flags |= 1; value = 0; }
        if (!scale.empty()) {
            if (!std::isfinite(scale[col]) || scale[col] < .0625f || scale[col] > 16) { flags |= 16; value = 0; }
            else { value /= scale[col]; if (!std::isfinite(value)) { flags |= 16; value = 0; } }
        }
        v[lane] = value * float(rotation_sign(spec.rotation_seed, lane));
    }
    for (int span = 1; span < 128; span *= 2) for (int base = 0; base < 128; base += 2 * span)
        for (int lane = 0; lane < span; ++lane) {
            const float a = v[base + lane], b = v[base + lane + span];
            v[base + lane] = a + b; v[base + lane + span] = a - b;
        }
    const float norm = 1.f / std::sqrt(128.f);
    for (float &value : v) { value *= norm; if (!std::isfinite(value)) flags |= 2; }
    return v;
}
a8_single_pass::Row old_two_pass(const uint8_t *source, size_t pitch, DType dtype, const W8StageSpec &spec,
                                int row, std::span<const float> scale) {
    a8_single_pass::Row result; result.codes.resize(spec.columns);
    float peak = 0;
    for (int block = 0; block < spec.columns / 128; ++block) {
        const auto rotated = old_block(source, pitch, dtype, spec, row, block, scale, result.flags);
        for (float value : rotated) peak = std::max(peak, std::abs(value));
    }
    const float normalized = peak == 0 ? 128.f : std::max((peak / 127.f) * 128.f, 0x1p-24f);
    // The independent GGUF CPU converter throws for overflow. The old
    // Metal scale pass emits infinity, flags4 and substitutes exact128.
    result.scale = std::isfinite(normalized) && normalized < 65520.f ? gguf::float_to_fp16_rne(normalized) : 0x7c00;
    if (!result.scale || (result.scale & 0x7c00) == 0x7c00) { result.flags |= 4; result.scale = 0x5800; }
    const float row_scale = gguf::fp16_to_float(result.scale);
    for (int block = 0; block < spec.columns / 128; ++block) {
        const auto rotated = old_block(source, pitch, dtype, spec, row, block, scale, result.flags);
        for (int lane = 0; lane < 128; ++lane) {
            float q = std::clamp((rotated[lane] / row_scale) * 128.f, -127.f, 127.f);
            if (!std::isfinite(q)) { result.flags |= 8; q = 0; }
            result.codes[block * 128 + lane] = int8_t(std::nearbyint(q));
        }
    }
    return result;
}
struct Matrix {
    size_t pitch, offset = 3;
    DType dtype;
    std::vector<uint8_t> storage;
    Matrix(int rows, int cols, DType format) : pitch(size_t(cols) * (format == DType::FP32 ? 4 : 2) + 7), dtype(format),
        storage(offset + size_t(rows) * pitch + 9, 0xa5) {}
    void put(int row, int col, float value) {
        uint8_t *at = storage.data() + offset + size_t(row) * pitch + size_t(col) * (dtype == DType::FP32 ? 4 : 2);
        if (dtype == DType::FP32) { const uint32_t bits = std::bit_cast<uint32_t>(value); std::memcpy(at, &bits, 4); }
        else { const uint16_t bits = dtype == DType::FP16 ? gguf::float_to_fp16_rne(value) : gguf::float_to_bf16_rne(value); std::memcpy(at, &bits, 2); }
    }
    const uint8_t *data() const { return storage.data() + offset; }
};
int comparisons = 0;
void compare(const Matrix &input, const W8StageSpec &spec, std::span<const float> scale, uint32_t required_flags = 0) {
    const auto untouched = input.storage;
    const size_t code_pitch = size_t(spec.rows) + 11, code_offset = 13;
    std::vector<uint8_t> output(code_offset + size_t(spec.columns) * code_pitch + 17, 0xa5);
    for (int row = 0; row < spec.rows; ++row) {
        const auto baseline = old_two_pass(input.data(), input.pitch, input.dtype, spec, row, scale);
        const auto candidate = a8_single_pass::row_model(input.data(), input.pitch, input.dtype, spec, row, scale);
        require(candidate.scale == baseline.scale, "A8 single-pass FP16 scale bits changed");
        require(candidate.codes == baseline.codes, "A8 single-pass signed RNE codes changed");
        require(candidate.flags == baseline.flags && (candidate.flags & required_flags) == required_flags, "A8 single-pass failure flags changed");
        for (int col = 0; col < spec.columns; ++col) output[code_offset + size_t(col) * code_pitch + row] = uint8_t(candidate.codes[col]);
        ++comparisons;
    }
    for (size_t i = 0; i < output.size(); ++i) {
        const bool data = i >= code_offset && i < code_offset + size_t(spec.columns) * code_pitch && (i - code_offset) % code_pitch < size_t(spec.rows);
        if (!data) require(output[i] == 0xa5, "A8 transpose pitch/offset padding overwritten");
    }
    require(untouched == input.storage, "A8 candidate overwrote input/padding");
}
void test_math() {
    for (int cols : {3840, 4096}) for (DType dtype : {DType::FP16, DType::BF16, DType::FP32}) {
        W8StageSpec spec{1, 2, 0, cols, 128, 20260930, true};
        for (int pattern = 0; pattern < 7; ++pattern) {
            Matrix matrix(4, cols, dtype); uint32_t random = 0x812fe74a;
            for (int row = 0; row < 4; ++row) for (int col = 0; col < cols; ++col) {
                random = random * 1664525 + 1013904223;
                float value = 0;
                if (pattern == 1) value = float(int(random >> 16) - 32768) / 4096;
                if (pattern == 2) value = ((col + row) & 1) ? 0x1p-24f : -0x1p-24f;
                if (pattern == 3) value = ((col + row) % 5 == 0) ? .3f : -1.7f;
                if (pattern >= 4 && col == cols - 128 + (pattern == 4 ? 31 : pattern == 5 ? 32 : 64)) value = float((row & 1) ? -123 : 123);
                matrix.put(row, col, value);
            }
            compare(matrix, spec, {});
            std::vector<float> ones(cols, 1);
            auto scaled = spec; scaled.inverse_column_scale = true;
            compare(matrix, scaled, ones);
            for (int col = 0; col < cols; ++col) ones[col] = col % 11 == 0 ? .3f : col % 11 == 1 ? 1.7f : std::ldexp(1.f, col % 9 - 4);
            compare(matrix, scaled, ones);
        }
        Matrix tiny_impulse(4, cols, dtype);
        for (int row = 0; row < 4; ++row) for (int col = 0; col < cols; ++col) tiny_impulse.put(row, col, 0);
        tiny_impulse.put(1, cols - 1, 0x1p-24f); tiny_impulse.put(2, 32, -0x1p-24f);
        compare(tiny_impulse, spec, {});
        require(a8_single_pass::row_model(tiny_impulse.data(), tiny_impulse.pitch, dtype, spec, 0).scale == 1,
                "tiny impulse did not retain minimum FP16 subnormal scale");
        // Mathematical helper also checks physical S1 coordinates even
        // though production eligibility intentionally requires column0.
        Matrix offset_input(4, cols + 128, dtype);
        for (int row = 0; row < 4; ++row) for (int col = 0; col < cols + 128; ++col) offset_input.put(row, col, float(col % 23 - 11) / 8);
        auto offset_spec = spec; offset_spec.column_begin = 128; offset_spec.inverse_column_scale = true;
        std::vector<float> scales(cols + 128);
        for (size_t col = 0; col < scales.size(); ++col) scales[col] = col < 128 ? .0625f : col % 2 ? .3f : 1.7f;
        compare(offset_input, offset_spec, scales);
    }
    // Last-block lane63 (cross-SIMD boundary), invalid input sanitization,
    // normalized FP16 overflow and scaled/rotation overflow are distinct.
    for (int cols : {3840, 4096}) {
        W8StageSpec spec{1, 1, 0, cols, 128, 20260930, true};
        for (const auto &[value, flags] : {std::pair{123.f, 0u}, {std::numeric_limits<float>::quiet_NaN(), 1u},
            {std::numeric_limits<float>::infinity(), 1u}, {0x1p+20f, 4u}}) {
            Matrix matrix(3, cols, DType::FP32);
            for (int row = 0; row < 3; ++row) for (int col = 0; col < cols; ++col) matrix.put(row, col, 0);
            matrix.put(1, cols - 128 + 63, value); compare(matrix, spec, {}, flags);
        }
        for (float bad : {0.f, -1.f, .03125f, 32.f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            Matrix matrix(3, cols, DType::FP32);
            for (int row = 0; row < 3; ++row) for (int col = 0; col < cols; ++col) matrix.put(row, col, .25f);
            std::vector<float> scale(cols, 1); scale[cols - 1] = bad;
            auto scaled = spec; scaled.inverse_column_scale = true; compare(matrix, scaled, scale, 16);
        }
        Matrix overflow(3, cols, DType::FP32);
        for (int row = 0; row < 3; ++row) for (int col = 0; col < cols; ++col) overflow.put(row, col, 0);
        overflow.put(1, 31, std::numeric_limits<float>::max()); overflow.put(1, 32, std::numeric_limits<float>::max());
        compare(overflow, spec, {}, 2);
        std::vector<float> scale(cols, 1); scale[31] = scale[32] = .0625f;
        auto scaled = spec; scaled.inverse_column_scale = true; compare(overflow, scaled, scale, 16);
    }
    // A signed tie must use the rounded FP16 scale and RNE, not C++'s
    // truncation or a compiler fast-math reciprocal approximation.
    const std::array<float, 6> ties{-2.5f, -1.5f, -.5f, .5f, 1.5f, 2.5f};
    const std::array<int8_t, 6> expected{-2, -2, 0, 0, 2, 2};
    for (size_t i = 0; i < ties.size(); ++i) {
        uint32_t flags = 0;
        require(a8_single_pass::retained_code(ties[i], 0x5800, flags) == expected[i] && !flags &&
                quantize_rotated(ties[i], 0x5800) == expected[i], "signed RNE tie changed");
    }
}
void test_policy() {
    require(!a8_single_pass::requested_flag(nullptr) && !a8_single_pass::requested_flag("0") && a8_single_pass::requested_flag("1"), "A8 explicit flag policy");
    for (const char *flag : {"", "true", "01", "2", "-1"}) rejected([&] { a8_single_pass::requested_flag(flag); });
    for (int cols : {3840, 4096}) {
        DeviceWeightView source; source.rows = 1056; source.cols = cols;
        W8StageSpec spec{0, 1056, 0, cols, 128, 20260930, true};
        for (DType dtype : {DType::FP16, DType::BF16, DType::FP32}) { source.dense_dtype = dtype; require(a8_single_pass::eligible(source, spec), "eligible dense A8 rejected"); }
        auto changed = spec; changed.transpose = false; require(!a8_single_pass::eligible(source, changed), "weight path eligible");
        changed = spec; changed.rotation_block = 512; require(!a8_single_pass::eligible(source, changed), "down H512 eligible");
        changed = spec; changed.columns -= 128; require(!a8_single_pass::eligible(source, changed), "partial hidden eligible");
        changed = spec; changed.column_begin = 128; require(!a8_single_pass::eligible(source, changed), "column slice eligible");
        changed = spec; changed.row_begin = 1; require(!a8_single_pass::eligible(source, changed), "out-of-bounds rows eligible");
        changed = spec; changed.inverse_column_scale = true; require(!a8_single_pass::eligible(source, changed), "missing inverse S1 eligible");
        changed.column_scale.emplace(); require(a8_single_pass::eligible(source, changed), "valid inverse S1 policy rejected");
        changed.inverse_column_scale = false; require(!a8_single_pass::eligible(source, changed), "noninverse A8 S1 eligible");
        source.encoding = DeviceWeightEncoding::AffineQ4; require(!a8_single_pass::eligible(source, spec), "packed path eligible");
        source.encoding = DeviceWeightEncoding::Dense; source.scales.emplace(); require(!a8_single_pass::eligible(source, spec), "affine metadata on dense eligible");
        source.scales.reset(); source.dense_dtype = static_cast<DType>(99); require(!a8_single_pass::eligible(source, spec), "unknown dtype eligible");
        source.dense_dtype = DType::FP16; source.cols = 3584; spec.columns = 3584; require(!a8_single_pass::eligible(source, spec), "unsupported hidden eligible");
    }
}
}
int main() {
    try {
        require(std::fesetround(FE_TONEAREST) == 0, "cannot set CPU RNE mode");
        test_policy(); test_math();
        std::cout << "A8 single-pass CPU: " << comparisons << " independent two-pass comparisons PASS; policy PASS; GPU untested\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
