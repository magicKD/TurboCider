#pragma once

#include "context.hpp"
#include "resolved_request.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace tc::streaming {

enum class PublicRunPhase : uint8_t {
    created,
    gpu_revalidated,
    running,
    draining,
    receipt_sealed,
    source_revalidated,
    completed,
    quarantined,
};

// Owns all mutable streaming state for one public generate call.  The
// resolved request, authority and source lease remain immutable; executors,
// receipt recorders and boundary events never escape this object.
class PublicStreamingRunContext final {
  public:
    PublicStreamingRunContext(
        std::shared_ptr<const ResolvedRequestExecution> execution,
        std::atomic<bool> &cancel);
    ~PublicStreamingRunContext();

    PublicStreamingRunContext(const PublicStreamingRunContext &) = delete;
    PublicStreamingRunContext &operator=(const PublicStreamingRunContext &) = delete;

    const ResolvedRequestExecution &execution() const noexcept;
    const SourceLease &lease() const;
    uint64_t source_generation() const;
    uint64_t request_generation() const noexcept;

    StageExecutor &attach_stage(
        uint32_t stage_index, std::shared_ptr<ModelSlotAdapter> adapter,
        std::string implementation);
    void finish_stage(uint32_t stage_index);
    void record_boundary(ActualBoundaryReceipt boundary);

    void mark_gpu_revalidated();
    void mark_running();
    void drain_all();
    std::shared_ptr<const ActualExecutionReceipt> seal_receipt(
        std::string implementation, std::string component_policy_revision);
    void revalidate_source_after_drain();
    void complete();

    void quarantine(std::string reason) noexcept;
    bool quarantined() const noexcept;
    PublicRunPhase phase() const noexcept;
    const std::string &quarantine_reason() const noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace tc::streaming
