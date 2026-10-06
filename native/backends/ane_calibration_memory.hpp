#pragma once

#include <optional>
#include <span>
#include <array>
#include <functional>
#include <vector>
#include <cmath>
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

struct NativeChannelCalibrationMemoryPlan {
    uint64_t surface_bytes, gpu_scratch_upper_bytes, gpu_restore_bytes, input_bytes;
    uint64_t internal_allowance_bytes, estimated_bytes;
};
// Complete independent one/four arena plus caller-owned GPU payloads. The
// streamed LoRA arm holds ONE correction pair/full hidden/join at a time;
// four immutable ANE W/DG/DU/Y bindings and frozen restore copies remain.
// Remove only three retained GPU head result planes; keep the conservative
// projection scratch term and allowances. This
// is an opportunistic admission estimate, not a hard process/driver RAM cap.
inline std::optional<NativeChannelCalibrationMemoryPlan> plan_native_channel_calibration_memory(
        uint64_t actual_rows, uint64_t bucket, uint64_t hidden, uint64_t full_width,
        uint64_t channels, bool lora, bool streamed, uint64_t page_bytes, bool fp32_output = false) {
    if (!actual_rows || actual_rows > bucket || !full_width || full_width > 16384 ||
            !channels || channels >= full_width || (streamed && !lora)) return std::nullopt;
    std::vector<CalibrationSurfaceShape> snapshots;
    for (int layer = 0; layer < 4; ++layer) {
        for (int p = 0; p < 2; ++p) {
            snapshots.push_back({channels, hidden, 1}); snapshots.push_back({channels, 1, 2});
        }
        snapshots.push_back({hidden, channels, 1}); snapshots.push_back({hidden, 1, 2});
        snapshots.push_back({hidden+1+(lora?channels:0), bucket, 2});
        snapshots.push_back({hidden, bucket, 2}); snapshots.push_back({hidden, 1, 2});
        snapshots.push_back({1, bucket, 2});
        if (lora) for (int p = 0; p < 3; ++p) snapshots.push_back({channels, bucket, 2});
    }
    snapshots.push_back({hidden, bucket, 1}); snapshots.push_back({1, bucket, 2});
    const auto arena = plan_gpu_calibration_memory(bucket, hidden, channels, lora, snapshots, page_bytes);
    if (!arena) return std::nullopt;
    const uint64_t live = streamed ? 1 : 4;
    const uint64_t scratch = actual_rows*((streamed?5:8)*hidden+6*(full_width-channels))*2+(128ull<<20)+
        (lora ? bucket*live*(3*channels+full_width)*2 : 0)+(fp32_output?actual_rows*live*hidden*2:0);
    const uint64_t targets = bucket*live*(hidden*(fp32_output?4:2)+(lora?channels*2:0));
    // Charge both input wrappers conservatively even if padding aliases input.
    const uint64_t inputs = (actual_rows+bucket)*hidden*2;
    return NativeChannelCalibrationMemoryPlan{arena->surface_bytes, scratch, targets, inputs,
        arena->internal_allowance_bytes, arena->estimated_bytes+scratch+targets+inputs};
}

struct ChannelSamplingPlan { std::array<int, 2> channels; bool memory_limited; };
// Prefer the original ~0.4/~0.8 points. If their COMPLETE arenas cannot fit,
// select two independently admitted aligned points; never extrapolate a fit
// back to an unmeasured share or invent a failed point's timing.
inline std::optional<ChannelSamplingPlan> plan_channel_sampling(
        int full_width, const std::function<bool(int)> &admitted) {
    if (full_width <= 1024 || full_width > 16384 || full_width % 512 || !admitted) return std::nullopt;
    const std::array<int, 2> preferred{int(std::round(.4*(full_width/512)))*512,
                                     int(std::round(.8*(full_width/512)))*512};
    if (admitted(preferred[0]) && admitted(preferred[1])) return ChannelSamplingPlan{preferred, false};
    int high = 0;
    for (int candidate = preferred[1]; candidate >= 1024; candidate -= 512)
        if (admitted(candidate)) { high = candidate; break; }
    if (!high) return std::nullopt;
    const int half = std::max(512, int(std::round(double(high)/1024))*512);
    for (int candidate = std::min(half, high-512); candidate >= 512; candidate -= 512)
        if (admitted(candidate)) return ChannelSamplingPlan{{candidate, high}, true};
    for (int candidate = half+512; candidate < high; candidate += 512)
        if (admitted(candidate)) return ChannelSamplingPlan{{candidate, high}, true};
    return std::nullopt;
}
} // namespace tc::ane
