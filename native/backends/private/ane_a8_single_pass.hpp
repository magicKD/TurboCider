#pragma once
#include "../ane_w8a8_math.hpp"
#include <cstring>
#include <string_view>
#include <vector>

namespace tc::ane::a8_single_pass {
// Production selection is shared with the CPU policy checks. The source
// extent, aliases and S1 lifetime are still validated by Device::stage_w8.
inline bool requested_flag(const char *value) {
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw CapabilityError("private ANE A8 single-pass requires 0 or 1");
}
inline bool eligible(const DeviceWeightView &source, const W8StageSpec &spec) {
    return source.encoding == DeviceWeightEncoding::Dense && !source.scales && !source.offsets &&
        (source.dense_dtype == DType::FP16 || source.dense_dtype == DType::BF16 || source.dense_dtype == DType::FP32) &&
        spec.transpose && spec.basis == W8Basis::SylvesterDH && spec.activation_group_size == 0 && spec.rotation_block == 128 && (source.cols == 3840 || source.cols == 4096) &&
        spec.column_begin == 0 && spec.columns == source.cols && spec.rows > 0 && spec.rows <= 32768 &&
        spec.row_begin >= 0 && spec.row_begin <= source.rows && spec.rows <= source.rows - spec.row_begin &&
        bool(spec.column_scale) == spec.inverse_column_scale;
}

// CPU model of the candidate's bounded per-lane retention. Independent
// two-pass tests replay the old kernel recipe rather than calling this model.
// No device, allocator or model access is performed here.
struct Row { uint16_t scale = 0; uint32_t flags = 0; std::vector<int8_t> codes; };
inline uint16_t scale_bits(float value) {
    // Match transfer_source's integer FP16 conversion, including invalid
    // outputs subsequently handled by the stage status flag (not a throw).
    const uint32_t bits = std::bit_cast<uint32_t>(value), sign = (bits >> 16) & 0x8000;
    const uint32_t e = (bits >> 23) & 255, m = bits & 0x7fffff;
    if (e == 255) return uint16_t(sign | 0x7c00 | (m ? 0x0200 : 0));
    if (e > 142) return uint16_t(sign | 0x7c00);
    if (e >= 113) return uint16_t(sign | ((((bits & 0x7fffffff) + 0xfff + ((bits >> 13) & 1)) >> 13) - 0x1c000));
    if (e < 102) return uint16_t(sign);
    const uint32_t shift = 126 - e, significand = m | 0x800000;
    const uint32_t result = significand >> shift, rest = significand & ((1u << shift) - 1), tie = 1u << (shift - 1);
    return uint16_t(sign | (result + uint32_t(rest > tie || (rest == tie && (result & 1)))));
}
inline int8_t retained_code(float value, uint16_t scale, uint32_t &flags) {
    float q = std::clamp((value / gguf::fp16_to_float(scale)) * 128.f, -127.f, 127.f);
    if (!std::isfinite(q)) { flags |= 8; q = 0; }
    const float lower = std::floor(q), fraction = q - lower;
    const int integer = int(lower);
    return int8_t(integer + (fraction > .5f || (fraction == .5f && (integer & 1))));
}
inline float dense_value(const uint8_t *at, DType dtype) {
    uint16_t bits16;
    if (dtype == DType::FP32) { float value; std::memcpy(&value, at, 4); return value; }
    std::memcpy(&bits16, at, 2);
    return dtype == DType::BF16 ? std::bit_cast<float>(uint32_t(bits16) << 16) : gguf::fp16_to_float(bits16);
}
inline Row row_model(const uint8_t *source, size_t pitch, DType dtype, const W8StageSpec &spec,
                     int row, std::span<const float> scale = {}) {
    if (!source || row < 0 || row >= spec.rows || !spec.transpose || spec.rotation_block != 128 ||
        (spec.columns != 3840 && spec.columns != 4096) || spec.column_begin < 0 || spec.row_begin < 0 ||
        (dtype != DType::FP16 && dtype != DType::BF16 && dtype != DType::FP32) ||
        (!scale.empty() && (scale.size() < size_t(spec.column_begin + spec.columns) || !spec.inverse_column_scale)) ||
        (scale.empty() && spec.inverse_column_scale)) throw CapabilityError("A8 single-pass CPU model geometry");
    const size_t item = dtype == DType::FP32 ? 4 : 2;
    if (pitch < size_t(spec.column_begin + spec.columns) * item) throw CapabilityError("A8 single-pass CPU model pitch");
    Row result; result.codes.resize(spec.columns);
    std::array<std::array<float, 32>, 128> retained{};
    float peak = 0;
    const int blocks = spec.columns / 128;
    const float norm = 1.f / std::sqrt(128.f);
    for (int block = 0; block < blocks; ++block) {
        std::array<float, 128> values;
        for (int lane = 0; lane < 128; ++lane) {
            const size_t column = size_t(spec.column_begin + block * 128 + lane);
            float value = dense_value(source + size_t(spec.row_begin + row) * pitch + column * item, dtype);
            if (!std::isfinite(value)) { result.flags |= 1; value = 0; }
            if (!scale.empty()) {
                const float s = scale[column];
                if (!std::isfinite(s) || s < 1.f / 16 || s > 16) { result.flags |= 16; value = 0; }
                else { value = value / s; if (!std::isfinite(value)) { result.flags |= 16; value = 0; } }
            }
            values[lane] = value * float(rotation_sign(spec.rotation_seed, lane));
        }
        // Ordered butterfly operations match the seven shuffle/TG spans.
        for (int span = 1; span < 128; span *= 2) {
            const auto before = values;
            for (int lane = 0; lane < 128; ++lane)
                values[lane] = (lane & span) ? before[lane ^ span] - before[lane] : before[lane] + before[lane ^ span];
        }
        for (int lane = 0; lane < 128; ++lane) {
            const float value = values[lane] * norm;
            if (!std::isfinite(value)) result.flags |= 2;
            retained[lane][block] = value;
            peak = std::fmax(peak, std::abs(value));
        }
    }
    const float normalized = peak == 0 ? 128.f : std::fmax((peak / 127.f) * 128.f, 0x1p-24f);
    result.scale = scale_bits(normalized);
    if (!result.scale || (result.scale & 0x7c00) == 0x7c00) { result.flags |= 4; result.scale = 0x5800; }
    for (int block = 0; block < blocks; ++block) for (int lane = 0; lane < 128; ++lane)
        result.codes[block * 128 + lane] = retained_code(retained[lane][block], result.scale, result.flags);
    return result;
}
} // namespace tc::ane::a8_single_pass
