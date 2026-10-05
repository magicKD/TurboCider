#pragma once

#include <optional>
#include <span>
#include "ane_memory.hpp"

namespace tc::ane {
struct CalibrationSurfaceShape { uint64_t rows, columns, item_bytes; };
struct GpuCalibrationMemoryPlan {
    uint64_t surface_bytes, internal_allowance_bytes, estimated_bytes;
};

// Actual surface layout: 64-byte row pitch, page-rounded allocation. Includes
// TWO mutable W banks and TWO bounded activation/correction slots, plus each
// immutable restore snapshot. Caller-owned GPU head/output tensors and any
// resident ANE calibration arena must be charged separately as resident
// optional bytes. The internal allowance is opportunistic, NOT proof that
// driver/Metal memory has a hard bound.
inline std::optional<GpuCalibrationMemoryPlan> plan_gpu_calibration_memory(
        uint64_t rows, uint64_t hidden, uint64_t width, bool corrections,
        std::span<const CalibrationSurfaceShape> snapshots, uint64_t page_bytes) {
    if (!rows || rows > 4224 || !hidden || hidden > 32768 || hidden % 128 ||
        !width || width > 32768 || width % 512 || !page_bytes ||
        (page_bytes & (page_bytes - 1))) return std::nullopt;
    constexpr auto max = UINT64_MAX;
    uint64_t total = 0;
    auto add = [&](uint64_t r, uint64_t c, uint64_t item, uint64_t count) {
        if (!r || r > 32768 || !c || c > 32768 || (item != 1 && item != 2) ||
            c > (max - 63) / item) return false;
        const auto pitch = (c * item + 63) / 64 * 64;
        if (r > (max - page_bytes + 1) / pitch) return false;
        const auto allocation = (r * pitch + page_bytes - 1) / page_bytes * page_bytes;
        if (count > (max - total) / allocation) return false;
        total += allocation * count;
        return true;
    };
    if (!add(width, hidden, 1, 4) || !add(width, 1, 2, 4) ||
        !add(hidden, width, 1, 2) || !add(hidden, 1, 2, 2) ||
        !add(hidden, rows, 1, 2) || !add(1, rows, 2, 2) ||
        (corrections && !add(width, rows, 2, 4))) return std::nullopt;
    for (const auto &snapshot : snapshots)
        if (!add(snapshot.rows, snapshot.columns, snapshot.item_bytes, 1)) return std::nullopt;
    constexpr uint64_t allowance = (uint64_t(128) << 20) + (uint64_t(4) << 20);
    if (total > max - allowance) return std::nullopt;
    return GpuCalibrationMemoryPlan{total, allowance, total + allowance};
}
} // namespace tc::ane
