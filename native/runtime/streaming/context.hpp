#pragma once
#include "actual_receipt.hpp"
#include "io_executor.hpp"
#include "slot_pool.hpp"
#include <memory>
#include <optional>

namespace tc::streaming {

struct ReaderSet {
    std::array<tc_stream_reader_fence_v1, TC_STREAM_MAX_READER_QUEUES> fences{};
    uint32_t count = 0;
    // The adapter has already reached a synchronous device completion point.
    // The executor still seals the declared reader identities, then retires
    // them directly without a completion-mailbox round trip.
    bool already_complete = false;
};
struct ExecutionCounters {
    uint64_t pool_creates = 0, slot_bundles = 0, fills = 0;
    uint64_t bytes_loaded = 0, groups_submitted = 0;
    double wait_seconds = 0;
};

class ModelSlotAdapter {
public:
    virtual ~ModelSlotAdapter() = default;
    // serial is the compatibility default.  An adapter must explicitly opt in
    // before the executor may keep multiple layout-class pools live.
    virtual bool supports_multi_pool_policy(MultiPoolPolicy policy) const noexcept {
        return policy == MultiPoolPolicy::serial;
    }
    virtual void create_pool(const PoolLayout &) = 0;
    // Called at an ordered class barrier after the preceding pool has drained.
    // Serial adapters normally need no extra action because create_pool made
    // the only live pool active. Retained adapters use this to select backing.
    virtual void select_pool(const PoolLayout &) {}
    virtual FillJob make_fill_job(const Group &, const tc_stream_slot_ticket_v1 &) = 0;
    virtual void encode_prefix(uint32_t pass) = 0;
    virtual void prepare_group(const Group &, const tc_stream_slot_ticket_v1 &) = 0;
    // Some synchronous backends need the next vacant slot's fill to start
    // after the current content is claimed but before encode blocks for its
    // reader completion. This keeps startup at one fill while restoring the
    // specialized K2 steady-state overlap. The default preserves the original
    // executor ordering.
    virtual bool overlap_next_fill_after_claim() const noexcept { return false; }
    // Register completion handlers BEFORE commit; every last-reader completion
    // posts to mailbox. Return one last reader per queue, not a scheduled event.
    virtual ReaderSet encode_group(const Group &, const tc_stream_slot_ticket_v1 &,
                                   CompletionMailbox &) = 0;
    // Must also drain prefix and every callback. false means quarantine: no free.
    virtual bool drain() noexcept = 0;
    virtual void destroy_pool() noexcept = 0;
    // Retained adapters own more than one live pool. The default preserves the
    // serial ABI; opt-in adapters override this pool-addressed destruction.
    virtual void destroy_pool(uint32_t) noexcept { destroy_pool(); }
};

// A failed drain retains adapter AND callback mailbox. Caller must keep this
// object alive in its quarantined session. Destruction with an unsafe pool
// terminates rather than silently free GPU/worker-visible storage.
class StageExecutor {
public:
    StageExecutor(uint32_t stage, uint64_t request_generation,
                  std::shared_ptr<ModelSlotAdapter> adapter);
    ~StageExecutor();
    StageExecutor(const StageExecutor &) = delete;
    StageExecutor &operator=(const StageExecutor &) = delete;
    void begin(const StageLayout &);
    void run_pass(uint32_t pass, uint32_t step, std::atomic<bool> &cancel,
                  std::chrono::milliseconds stall_timeout = std::chrono::seconds(60));
    // Opt-in only. Must be called by the owner after begin() and before the
    // first pass. The default/private executor path leaves this disabled.
    void enable_receipt(ExecutionReceiptOptions);
    ExecutionCounters finish();
    ExecutionCounters counters() const;
    std::shared_ptr<const ActualStageReceipt> receipt() const;
    ExecutionCounters run(const StageLayout &, std::atomic<bool> &cancel,
                          std::chrono::milliseconds stall_timeout = std::chrono::seconds(60));
    bool quarantined() const noexcept { return quarantined_; }
    bool retry_drain() noexcept;
private:
    uint32_t stage_;
    uint64_t request_;
    std::shared_ptr<ModelSlotAdapter> adapter_;
    std::unique_ptr<CompletionMailbox> mailbox_;
    std::unique_ptr<ActualReceiptRecorder> receipt_recorder_;
    std::shared_ptr<const ActualStageReceipt> receipt_;
    struct State;
    std::unique_ptr<State> state_;
    void create_pool(uint32_t pool_index);
    void activate_pool(uint32_t pool_index);
    void drain_active_pool(const std::optional<tc_stream_slot_ticket_v1> &carry);
    void destroy_pools() noexcept;
    bool consume();
    void check_cancel(const std::atomic<bool> &) const;
    bool pools_live_ = false, quarantined_ = false, used_ = false;
};
} // namespace tc::streaming
