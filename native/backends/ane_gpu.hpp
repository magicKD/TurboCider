#pragma once

// Shared Metal/IOSurface W8/A8/FP32 restoration. No private ANE client API.
#include "ane_runtime.hpp"
#include <chrono>
#include <functional>
#include <utility>

namespace tc::ane::gpu {
bool healthy(); // secondary staging/transfer safety domain; never disables the family's GPU fallback
enum class Element { FP16, I8 };
class Device;
class Transfer;
class QuantStage;
inline constexpr size_t scale_cache_budget_bytes = 4u << 20;
uint64_t configured_weight_code_cache_bytes();

class Surface {
  public:
    Surface(Device &, uint32_t rows, uint32_t columns, Element);
    size_t bytes() const;
    size_t pitch() const;
    uint32_t rows() const;
    uint32_t columns() const;
    Element element() const;
    void *data() const; // caller must not access while the ticket is live
    void *native_iosurface() const; // borrowed; retain this Surface through Core ML/GPU completion
    bool is_view() const { return row_begin_ || row_count_; }
    Surface slice_rows(uint32_t begin, uint32_t count) const; // consumer view only, not an ANE binding
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    uint32_t row_begin_ = 0, row_count_ = 0;
    friend class Transfer;
    friend class Device;
};

struct Upload { DeviceMatrixView source; Surface destination; int begin_row = 0; float scale = 1.f; };
enum class RowScalePolicy : uint32_t { Positive, SignedFinite };
struct Download {
    Surface source;
    std::optional<DeviceMatrixView> destination; // null: finite/range validation only
    int begin_row = 0;
    DType dtype = DType::FP16;
    float scale = 1.f;
    std::optional<Surface> row_scales = std::nullopt, token_scales = std::nullopt;
    std::optional<Surface> second_token_scales = std::nullopt;
    // Direct ConvRot retains finite signed/zero checkpoint row scales. Token
    // quantizer scales stay strictly positive; Sylvester's default unchanged.
    RowScalePolicy row_scale_policy = RowScalePolicy::Positive;
};

class Device {
  public:
    Device(); // legacy Private staging flags, no private client API
    Device(bool scale_cache, bool specialize);
    Device(bool scale_cache, bool specialize, uint64_t weight_code_cache_bytes);
    void *shared_event() const;
    void release_after_failure(uint64_t value) const;
    std::string name() const;
    // Commit a separate leading command buffer. Do not defer the signal
    // until the command buffer containing the matching wait is committed.
    void signal(uint64_t value);
    // Calibration only: all producers are already joined and all requests
    // prepared but not submitted. Meet their dependency on the CPU BEFORE
    // starting the timer; no Metal/ANE handoff is part of an alone sample.
    // Normal inference continues to signal from its GPU producer.
    void release_prepared(uint64_t value);
    bool wait(uint64_t value, std::chrono::milliseconds timeout);
    uint64_t value() const;
    Transfer prepare_transfer(std::vector<Upload>, std::vector<Download>,
                              uint64_t ready, uint64_t done);
    // Calibration GPU arm: one command buffer, no shared-event signal/wait
    // and no live ANE output dependency. Producers/source snapshots must be
    // ready and remain immutable until completion; normal inference uses
    // prepare_transfer(), never this independent restore path.
    Transfer prepare_gpu_transfer(std::vector<Upload>, std::vector<Download>);
    Transfer prepare_upload(std::vector<Upload>); // public CPU/NE input fence, no output dependency
    // A separate staging queue/event: next-layer readiness MUST NOT advance
    // the current layer's activation/ANE-done timeline.
    QuantStage stage_w8(DeviceWeightView, W8StageSpec, Surface codes, Surface scales);
    WeightCacheStats scale_cache_stats() const;
    WeightCodeCacheReport weight_code_cache_stats() const;
    uint64_t weight_code_cache_budget_bytes() const;
    void clear_weight_code_cache(); // producer tickets retain their own capacity leases
    StagePipelineStats stage_pipeline_stats() const;
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    Transfer prepare_transfer_impl(std::vector<Upload>, std::vector<Download>,
                                  std::optional<std::pair<uint64_t, uint64_t>>, bool upload_only = false);
    friend class Surface;
    friend class Transfer;
    friend class QuantStage;
};

struct Completion {
    bool ok = false, timed_out = false;
    std::string error;
};
class QuantStage {
  public:
    Completion finish(std::chrono::milliseconds timeout = std::chrono::seconds(30));
    uint32_t validation_flags() const;
    void *ready_event() const; // opaque MTLSharedEvent retained by this ticket
    uint64_t ready_value() const { return 1; }
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Device;
};
// One input CB commits its ready signal before a separate output CB waits for
// ANE done. A sticky CPU failure flag suppresses output reads after failure/
// timeout; GPU validation has its own flag, never clearing the CPU flag.
class Transfer {
  public:
    std::function<void()> failure_callback() const;
    void submit();
    Completion finish(std::chrono::milliseconds timeout = std::chrono::seconds(30));
    uint32_t validation_flags() const;
    bool independent_gpu() const;
  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Device;
};
} // namespace tc::ane::gpu
