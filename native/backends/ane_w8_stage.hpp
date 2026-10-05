#pragma once
#include "ane_runtime.hpp"
#include <tuple>

namespace tc::ane::w8_stage {
// Shared by the compact scale cache and two-bank prefetch matching. A reused
// Metal address with a different allocator generation is always a miss.
inline bool same_generation(const std::weak_ptr<void> &a, const std::weak_ptr<void> &b) {
    return !a.owner_before(b) && !b.owner_before(a);
}
inline bool same_matrix(const DeviceMatrixView &a, const DeviceMatrixView &b) {
    return same_generation(a.allocation_identity, b.allocation_identity) &&
        std::tie(a.buffer, a.buffer_bytes, a.offset_bytes, a.rows, a.cols, a.row_stride_bytes, a.dtype) ==
        std::tie(b.buffer, b.buffer_bytes, b.offset_bytes, b.rows, b.cols, b.row_stride_bytes, b.dtype);
}
inline bool same_optional_matrix(const std::optional<DeviceMatrixView> &a, const std::optional<DeviceMatrixView> &b) {
    return bool(a) == bool(b) && (!a || same_matrix(*a, *b));
}
inline bool same_region(const DeviceWeightRegion &a, const DeviceWeightRegion &b) {
    const auto &x = a.source, &y = b.source;
    const auto &p = a.selection, &q = b.selection;
    return same_generation(x.allocation_identity, y.allocation_identity) &&
        same_optional_matrix(x.scales, y.scales) && same_optional_matrix(x.offsets, y.offsets) &&
        same_optional_matrix(p.column_scale, q.column_scale) &&
        std::tie(x.buffer, x.buffer_bytes, x.offset_bytes, x.row_stride_bytes, x.rows, x.cols, x.encoding, x.dense_dtype, x.group_size, x.immutable_generation) ==
        std::tie(y.buffer, y.buffer_bytes, y.offset_bytes, y.row_stride_bytes, y.rows, y.cols, y.encoding, y.dense_dtype, y.group_size, y.immutable_generation) &&
        std::tie(p.row_begin, p.rows, p.column_begin, p.columns, p.rotation_block, p.rotation_seed, p.transpose, p.inverse_column_scale) ==
        std::tie(q.row_begin, q.rows, q.column_begin, q.columns, q.rotation_block, q.rotation_seed, q.transpose, q.inverse_column_scale);
}
inline bool live_key(const DeviceWeightRegion &key) {
    return !key.source.allocation_identity.expired() &&
        (!key.source.scales || !key.source.scales->allocation_identity.expired()) &&
        (!key.source.offsets || !key.source.offsets->allocation_identity.expired()) &&
        (!key.selection.column_scale || !key.selection.column_scale->allocation_identity.expired());
}
inline DeviceWeightRegion weak_key(DeviceWeightView source, W8StageSpec spec) {
    source.owner.reset();
    if (source.scales) source.scales->owner.reset();
    if (source.offsets) source.offsets->owner.reset();
    if (spec.column_scale) spec.column_scale->owner.reset();
    return {std::move(source), std::move(spec)};
}
inline void validate_column_scale(const DeviceWeightView &source, const W8StageSpec &spec) {
    const auto check = [](bool valid, const char *reason) { if (!valid) throw CapabilityError(reason); };
    if (!spec.column_scale) {
        check(!spec.inverse_column_scale, "W8 inverse S1 requires column scale");
        return;
    }
    const auto &scale = *spec.column_scale;
    check(spec.rotation_block == 128 && spec.inverse_column_scale == spec.transpose,
          "W8 S1 requires W H128 multiply or transpose A8 H128 divide");
    check(source.cols > 0 && source.cols <= 32768 && scale.owner && scale.buffer &&
          !scale.allocation_identity.expired() && scale.rows == 1 && scale.cols == source.cols && scale.dtype == DType::FP32,
          "W8 S1 requires immutable live FP32 full-column vector");
    const size_t row = size_t(scale.cols) * sizeof(float);
    const size_t pitch = scale.row_stride_bytes ? scale.row_stride_bytes : row;
    check(pitch >= row && pitch <= UINT32_MAX && pitch % sizeof(float) == 0 &&
          scale.offset_bytes % sizeof(float) == 0 && scale.offset_bytes <= scale.buffer_bytes &&
          row <= scale.buffer_bytes - scale.offset_bytes, "W8 S1 extent/pitch mismatch");
    check(scale.buffer != source.buffer && (!source.scales || scale.buffer != source.scales->buffer) &&
          (!source.offsets || scale.buffer != source.offsets->buffer), "W8 S1 aliases source/metadata");
}
inline void validate_swiglu_column_scales(std::span<const DeviceWeightRegion> weights, int hidden) {
    if (weights.size() != 3) throw CapabilityError("W8 SwiGLU requires gate/up/down");
    const auto &g = weights[0].selection, &u = weights[1].selection, &d = weights[2].selection;
    if (!same_optional_matrix(g.column_scale, u.column_scale) || g.inverse_column_scale || u.inverse_column_scale ||
        d.column_scale || d.inverse_column_scale)
        throw CapabilityError("W8 gate/up must share S1 generation; down cannot use S1");
    for (const auto &weight : weights) validate_column_scale(weight.source, weight.selection);
    if (g.column_scale && (weights[0].source.cols != hidden || weights[1].source.cols != hidden ||
                          g.column_begin != 0 || u.column_begin != 0 || g.columns != hidden || u.columns != hidden))
        throw CapabilityError("W8 gate/up S1 must cover complete activation hidden columns");
}
} // namespace tc::ane::w8_stage
