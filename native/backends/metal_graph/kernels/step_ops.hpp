#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace tc::metal_graph::step_ops {

inline constexpr std::string_view precision_recipe = "euler-bf16-dt-bf16-product-bf16-add-v1";
enum class Element : uint8_t { BF16, FP32 };
constexpr size_t element_bytes(Element type) {
    return type == Element::BF16 ? 2 : type == Element::FP32 ? 4 : 0;
}
struct Layout {
    const void *allocation = nullptr;
    size_t offset = 0, available = 0, stride = 0;
    uint32_t rows = 0, columns = 0;
    Element type = Element::BF16;
};
struct FlagLayout { const void *allocation = nullptr; size_t offset = 0, available = 0; };

inline size_t validate_layout(Layout layout) {
    const auto item = element_bytes(layout.type);
    if (!layout.allocation || !item || !layout.rows || !layout.columns ||
        layout.rows > INT32_MAX || layout.columns > INT32_MAX ||
        layout.offset % item || layout.stride % item || layout.stride / item > UINT32_MAX ||
        size_t(layout.columns) > std::numeric_limits<size_t>::max() / item)
        throw std::invalid_argument("native step matrix owner/type/geometry/alignment invalid");
    const size_t last = size_t(layout.columns) * item;
    if (layout.stride < last || size_t(layout.rows - 1) >
        (std::numeric_limits<size_t>::max() - last) / layout.stride)
        throw std::invalid_argument("native step matrix stride/extent overflow");
    const size_t bytes = size_t(layout.rows - 1) * layout.stride + last;
    if (bytes > layout.available || layout.offset > std::numeric_limits<size_t>::max() - bytes)
        throw std::invalid_argument("native step matrix exceeds retained view");
    return bytes;
}
inline bool ranges_overlap(size_t a, size_t a_bytes, size_t b, size_t b_bytes) {
    return a <= b ? b - a < a_bytes : a - b < b_bytes;
}
inline void distinct(Layout a, Layout b) {
    const auto a_bytes = validate_layout(a), b_bytes = validate_layout(b);
    if (a.allocation == b.allocation && ranges_overlap(a.offset, a_bytes, b.offset, b_bytes))
        throw std::invalid_argument("native step writable view aliases a read operand");
}
inline void same_shape_type(Layout a, Layout b) {
    (void)validate_layout(a); (void)validate_layout(b);
    if (a.rows != b.rows || a.columns != b.columns || a.type != b.type)
        throw std::invalid_argument("native step logical shape/type mismatch");
}
inline void validate_copy(Layout source, Layout destination) {
    same_shape_type(source, destination); distinct(destination, source);
}

// Round-to-nearest-even BF16. NaNs remain NaNs instead of integer overflow
// turning some payloads into zero. This also produces the actual dt POD bits.
inline uint16_t bf16_bits(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
        return uint16_t((bits >> 16) | 0x0040u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
inline float bf16_value(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
inline bool bf16_finite(uint16_t value) { return (value & 0x7f80u) != 0x7f80u; }
inline uint16_t checked_dt(float dt) {
    const auto value = bf16_bits(dt);
    if (!std::isfinite(dt) || !bf16_finite(value))
        throw std::invalid_argument("native Euler dt must be finite after BF16 rounding");
    return value;
}
inline void validate_euler(Layout latents, Layout noise, float dt, Layout output) {
    same_shape_type(latents, noise); same_shape_type(latents, output);
    if (latents.type != Element::BF16)
        throw std::invalid_argument("native Euler requires BF16 latents/noise/output");
    (void)checked_dt(dt); distinct(output, latents); distinct(output, noise);
}
inline uint16_t reference_euler(uint16_t latent, uint16_t noise, float dt) {
    const auto product = bf16_bits(bf16_value(noise) * bf16_value(checked_dt(dt)));
    return bf16_bits(bf16_value(latent) + bf16_value(product));
}
inline void validate_flag(FlagLayout flag) {
    if (!flag.allocation || flag.offset % sizeof(uint32_t) || flag.available < sizeof(uint32_t) ||
        flag.offset > std::numeric_limits<size_t>::max() - sizeof(uint32_t))
        throw std::invalid_argument("native step finite flag requires an aligned retained UInt32");
}
inline void validate_finite(Layout input, FlagLayout flag) {
    const auto bytes = validate_layout(input); validate_flag(flag);
    if (input.allocation == flag.allocation && ranges_overlap(input.offset, bytes, flag.offset, sizeof(uint32_t)))
        throw std::invalid_argument("native step finite flag aliases checked matrix");
}

struct CopyParameters { uint32_t rows, columns, input_stride, output_stride; };
struct EulerParameters { uint32_t rows, columns, latent_stride, noise_stride, output_stride, dt_bits; };
struct FiniteParameters { uint32_t rows, columns, input_stride; };
static_assert(sizeof(CopyParameters) == 16 && sizeof(EulerParameters) == 24 && sizeof(FiniteParameters) == 12);

inline constexpr std::string_view source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct TCStepCopy { uint rows, columns, input_stride, output_stride; };
struct TCStepEuler { uint rows, columns, latent_stride, noise_stride, output_stride, dt_bits; };
struct TCStepFinite { uint rows, columns, input_stride; };

// Integer loads/stores preserve every BF16/FP32 bit, including NaN payloads,
// signed zeros and row padding (padding is never copied or read).
kernel void tc_step_copy_bf16(device const ushort *input [[buffer(0)]], device ushort *output [[buffer(1)]],
    constant TCStepCopy& p [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.columns || pos.y >= p.rows) return;
    output[ulong(pos.y) * p.output_stride + pos.x] = input[ulong(pos.y) * p.input_stride + pos.x];
}
kernel void tc_step_copy_f32(device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]],
    constant TCStepCopy& p [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.columns || pos.y >= p.rows) return;
    output[ulong(pos.y) * p.output_stride + pos.x] = input[ulong(pos.y) * p.input_stride + pos.x];
}
kernel void tc_step_euler_bf16(device const bfloat *latents [[buffer(0)]], device const bfloat *noise [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant TCStepEuler& p [[buffer(3)]], uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.columns || pos.y >= p.rows) return;
    const float dt = as_type<float>(p.dt_bits << 16);
    const bfloat product = bfloat(float(noise[ulong(pos.y) * p.noise_stride + pos.x]) * dt);
    output[ulong(pos.y) * p.output_stride + pos.x] =
        bfloat(float(latents[ulong(pos.y) * p.latent_stride + pos.x]) + float(product));
}
kernel void tc_step_reset_invalid(device atomic_uint *invalid [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0) atomic_store_explicit(invalid, 0u, memory_order_relaxed);
}
kernel void tc_step_finite_bf16(device const ushort *input [[buffer(0)]], device atomic_uint *invalid [[buffer(1)]],
    constant TCStepFinite& p [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.columns || pos.y >= p.rows) return;
    const ushort value = input[ulong(pos.y) * p.input_stride + pos.x];
    if ((value & 0x7f80u) == 0x7f80u) atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
}
kernel void tc_step_finite_f32(device const uint *input [[buffer(0)]], device atomic_uint *invalid [[buffer(1)]],
    constant TCStepFinite& p [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.columns || pos.y >= p.rows) return;
    const uint value = input[ulong(pos.y) * p.input_stride + pos.x];
    if ((value & 0x7f800000u) == 0x7f800000u) atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
}
)metal";

} // namespace tc::metal_graph::step_ops
