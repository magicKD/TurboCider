#pragma once

#include "../../backends/mlx.hpp"
#include "../../core/stream_slot_c.h"
#include "layout.hpp"
#include "source_lease.hpp"

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>

namespace tc::streaming {

struct MlxWeightPagerMetrics {
    uint64_t resident_bytes_loaded = 0;
    uint64_t streamed_bytes_loaded = 0;
    uint64_t slot_arrays_allocated = 0;
    uint64_t slot_fills = 0;
    double resident_load_seconds = 0;
    double streamed_load_seconds = 0;
    double maximum_fill_seconds = 0;
    int64_t maximum_fill_group = -1;
};

// Direct BF16 safetensors -> MLX shared-buffer pager.  The owner thread
// creates every MLX array during setup; refill workers only call pread into a
// vacant slot and never touch MLX stream state.  The first implementation is
// deliberately strict: one source range per field, no conversion/derivation,
// and one block per group.  Unsupported descriptors fail before execution.
class MlxWeightPager {
  public:
    struct State;

    MlxWeightPager(const std::filesystem::path &artifact_root,
                   const Descriptor &, const StageDescriptor &,
                   const StageLayout &);
    MlxWeightPager(std::shared_ptr<const SourceLease>,
                   const Descriptor &, const StageDescriptor &,
                   const StageLayout &);
    ~MlxWeightPager();

    MlxWeightPager(const MlxWeightPager &) = delete;
    MlxWeightPager &operator=(const MlxWeightPager &) = delete;

    void load_resident(Weights &destination,
                       const std::atomic<bool> *cancel = nullptr);
    void create_pool(const PoolLayout &);
    void destroy_pool(uint32_t pool) noexcept;
    uint64_t fill(const Group &, const tc_stream_slot_ticket_v1 &,
                  const std::atomic<bool> *cancel);
    Weights bind(const Group &, const tc_stream_slot_ticket_v1 &) const;
    void check_open_files() const;

    const MlxWeightPagerMetrics &metrics() const noexcept { return metrics_; }

  private:
    std::unique_ptr<State> state_;
    MlxWeightPagerMetrics metrics_;
    mutable std::mutex metrics_mutex_;
};

} // namespace tc::streaming
