#pragma once

#include "ane_program.hpp"
#include "../ane_calibration_memory.hpp"
#include <array>

namespace tc::ane::private_api {

struct CalibrationBindings {
    std::vector<std::pair<std::string, Surface>> inputs, outputs;
};
struct CalibrationBatchResult {
    bool ok = false;
    double seconds = 0;
    uint64_t ane_calls = 0;
    std::string error;
};

// A standalone calibration batch, NOT an inference result. Producers must
// have joined before construction; inputs stay immutable through measure().
// Each evaluation has independently owned output surfaces. Binding and the
// already-met ANE wait are outside the clock; actual client submissions and
// completion joins remain inside. This is a host batch span, not a physical
// ANE kernel duration, and does not by itself prove device overlap.
//
// GPU callbacks must operate on separate mutable scratch and must not wait
// for this batch's ANE output. finish_gpu must drain every producer even when
// submit_gpu throws after partial submission. It is called on that path too.
class CalibrationBatch {
  public:
    CalibrationBatch(Device &, Program &, std::vector<CalibrationBindings>, uint64_t &timeline);
    CalibrationBatch(const CalibrationBatch &) = delete;
    CalibrationBatch &operator=(const CalibrationBatch &) = delete;
    CalibrationBatchResult measure(bool ane,
                                   const std::function<void()> &submit_gpu = {},
                                   const std::function<void()> &finish_gpu = {});
  private:
    Device device_;
    std::vector<PreparedRequest> prepared_;
    uint64_t ready_ = 0;
    bool consumed_ = false;
};

struct W8GpuCalibrationLayer {
    std::array<DeviceWeightRegion, 3> weights;
    DeviceMatrixView activation;
    std::optional<std::pair<DeviceMatrixView, DeviceMatrixView>> corrections;
    // Completed ANE snapshots only, never live/borrowed output. Constructor
    // copies these surfaces and all scales into independent private storage.
    std::vector<Download> restoration;
};
struct W8GpuCalibrationStats {
    uint64_t layers = 0, weight_projections = 0, activation_packs = 0;
    uint64_t correction_uploads = 0, restore_downloads = 0, joins = 0;
    bool prefetch = false, independent_gpu_transfer = false, completed = false;
};

// Complete GPU-side traffic for a fixed candidate: physical-source W8 stage,
// A8 rotate/pack, optional correction upload, family GPU head, independent
// scaled restore and family join. Mutable W/A scratch never aliases prepared
// ANE bindings. Five source versions permit 1-vs-4 with the last future bank.
// No inference output is published by this measurement adapter.
class W8GpuCalibrationWork {
  public:
    using Head = std::function<void(int layer, int gpu_channels)>;
    using Join = std::function<void(int layer)>;
    using Fence = std::function<void()>;
    W8GpuCalibrationWork(Device &, GraphShape, std::vector<W8GpuCalibrationLayer>,
                         MemoryLimits, uint64_t resident_optional_bytes, uint64_t mlx_active_bytes);
    ~W8GpuCalibrationWork();
    W8GpuCalibrationWork(const W8GpuCalibrationWork &) = delete;
    W8GpuCalibrationWork &operator=(const W8GpuCalibrationWork &) = delete;
    // Outside clock: select 1/4, compile/prepare restore bindings, reset stats.
    void prepare(int count, bool prefetch);
    // Inside clock. On submit failure the caller MUST still call finish().
    void submit(const Head &);
    void finish(const Fence &heads, const Join &, const Fence &joins);
    W8GpuCalibrationStats stats() const;
    uint64_t estimated_bytes() const;
    uint64_t allocated_surface_bytes() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tc::ane::private_api
