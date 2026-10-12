#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tc::qwen21::block_replay_math {

// A borrowed CPU view of BF16 [outer, rows, columns]. The pointer already
// includes the view's physical offset. Padding is outside the logical values.
// Either outer groups or row groups must occupy separate physical spans;
// logical traversal always remains outer, row, column.
struct View {
    const std::byte* data = nullptr;
    size_t bytes = 0;
    size_t outer = 0;
    size_t rows = 0;
    size_t columns = 0;
    size_t outer_stride_bytes = 0;
    size_t row_stride_bytes = 0;
};

struct Difference {
    double relative_rms = 0;
    double max_abs = 0;
    size_t values = 0;
};

namespace detail {
inline size_t checked_add(size_t left, size_t right) {
    if (right > std::numeric_limits<size_t>::max() - left)
        throw std::invalid_argument("BF16 replay view extent overflow");
    return left + right;
}

inline size_t checked_multiply(size_t left, size_t right) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        throw std::invalid_argument("BF16 replay view extent overflow");
    return left * right;
}

struct CheckedView {
    size_t row_bytes;
    size_t values;
};

inline CheckedView check(View view) {
    if (!view.data || !view.bytes || !view.outer || !view.rows || !view.columns)
        throw std::invalid_argument("BF16 replay view must be nonempty");

    const size_t row_bytes = checked_multiply(view.columns, size_t{2});
    if (view.row_stride_bytes % 2 || view.row_stride_bytes < row_bytes)
        throw std::invalid_argument("BF16 replay view has an invalid row stride");
    if (view.outer_stride_bytes % 2 || view.outer_stride_bytes < row_bytes)
        throw std::invalid_argument("BF16 replay view has an invalid outer stride");
    const size_t row_extent = checked_add(
        checked_multiply(view.rows - 1, view.row_stride_bytes), row_bytes);
    const size_t outer_offset = checked_multiply(view.outer - 1, view.outer_stride_bytes);
    const size_t outer_extent = checked_add(outer_offset, row_bytes);
    const bool outer_separated = view.outer_stride_bytes >= row_extent;
    const bool row_separated = view.row_stride_bytes >= outer_extent;
    if (!outer_separated && !row_separated)
        throw std::invalid_argument("BF16 replay view has overlapping group spans");
    const size_t extent = checked_add(
        outer_offset, row_extent);
    if (extent > view.bytes)
        throw std::invalid_argument("BF16 replay view exceeds its CPU buffer");

    const auto address = reinterpret_cast<uintptr_t>(view.data);
    if (extent - 1 > std::numeric_limits<uintptr_t>::max() - address)
        throw std::invalid_argument("BF16 replay view address overflow");
    const size_t values = checked_multiply(
        checked_multiply(view.outer, view.rows), view.columns);
    return {row_bytes, values};
}

inline float read_bf16(const std::byte* data) {
    uint16_t bits;
    std::memcpy(&bits, data, sizeof(bits));
    return std::bit_cast<float>(uint32_t(bits) << 16);
}
} // namespace detail

inline void validate(View view) {
    (void)detail::check(view);
}

// Hash callers can update an incremental CPU digest with each logical row.
// No padding, copies, allocations, or tensor evaluation are performed here.
template <class F>
inline void for_each_row(View view, F&& function) {
    const auto checked = detail::check(view);
    for (size_t outer = 0; outer < view.outer; ++outer) {
        const auto* head = view.data + outer * view.outer_stride_bytes;
        for (size_t row = 0; row < view.rows; ++row)
            function(head + row * view.row_stride_bytes, checked.row_bytes);
    }
}

inline Difference compare(View candidate, View reference) {
    const auto checked = detail::check(candidate);
    (void)detail::check(reference);
    if (candidate.outer != reference.outer || candidate.rows != reference.rows ||
        candidate.columns != reference.columns)
        throw std::invalid_argument("BF16 replay comparison shape mismatch");

    double squared_error = 0;
    double squared_reference = 0;
    double max_abs = 0;
    for (size_t outer = 0; outer < candidate.outer; ++outer) {
        const auto* candidate_head = candidate.data + outer * candidate.outer_stride_bytes;
        const auto* reference_head = reference.data + outer * reference.outer_stride_bytes;
        for (size_t row = 0; row < candidate.rows; ++row) {
            const auto* candidate_row = candidate_head + row * candidate.row_stride_bytes;
            const auto* reference_row = reference_head + row * reference.row_stride_bytes;
            for (size_t column = 0; column < candidate.columns; ++column) {
                const float candidate_value = detail::read_bf16(candidate_row + column * 2);
                const float reference_value = detail::read_bf16(reference_row + column * 2);
                if (!std::isfinite(candidate_value) || !std::isfinite(reference_value))
                    throw std::invalid_argument("BF16 replay comparison contains a nonfinite value");
                const double value = reference_value;
                const double error = double(candidate_value) - value;
                const double absolute_error = std::abs(error);
                squared_error += error * error;
                squared_reference += value * value;
                if (absolute_error > max_abs) max_abs = absolute_error;
            }
        }
    }
    const double denominator = squared_reference > 1e-20 ? squared_reference : 1e-20;
    const double relative_rms = std::sqrt(squared_error / denominator);
    if (!std::isfinite(squared_error) || !std::isfinite(squared_reference) ||
        !std::isfinite(relative_rms) || !std::isfinite(max_abs))
        throw std::invalid_argument("BF16 replay comparison result is nonfinite");
    return {relative_rms, max_abs, checked.values};
}

} // namespace tc::qwen21::block_replay_math
