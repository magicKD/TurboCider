#pragma once
#include "io_executor.hpp"
#include "slot_pool.hpp"
#include <memory>

namespace tc::streaming {

struct ReaderSet {
    std::array<tc_stream_reader_fence_v1, TC_STREAM_MAX_READER_QUEUES> fences{};
    uint32_t count = 0;
};
struct ExecutionCounters {
    uint64_t pool_creates = 0, slot_bundles = 0, fills = 0;
    uint64_t bytes_loaded = 0, groups_submitted = 0;
};

class ModelSlotAdapter {
public:
    virtual ~ModelSlotAdapter() = default;
    virtual void create_pool(const PoolLayout &) = 0;
    virtual FillJob make_fill_job(const Group &, const tc_stream_slot_ticket_v1 &) = 0;
    virtual void encode_prefix(uint32_t pass) = 0;
    virtual void prepare_group(const Group &, const tc_stream_slot_ticket_v1 &) = 0;
    // Register completion handlers BEFORE commit; every last-reader completion
    // posts to mailbox. Return one last reader per queue, not a scheduled event.
    virtual ReaderSet encode_group(const Group &, const tc_stream_slot_ticket_v1 &,
                                   CompletionMailbox &) = 0;
    // Must also drain prefix and every callback. false means quarantine: no free.
    virtual bool drain() noexcept = 0;
    virtual void destroy_pool() noexcept = 0;
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
    ExecutionCounters finish();
    ExecutionCounters counters() const;
    ExecutionCounters run(const StageLayout &, std::atomic<bool> &cancel,
                          std::chrono::milliseconds stall_timeout = std::chrono::seconds(60));
    bool quarantined() const noexcept { return quarantined_; }
    bool retry_drain() noexcept;
private:
    uint32_t stage_;
    uint64_t request_;
    std::shared_ptr<ModelSlotAdapter> adapter_;
    std::unique_ptr<CompletionMailbox> mailbox_;
    struct State;
    std::unique_ptr<State> state_;
    bool consume();
    void check_cancel(const std::atomic<bool> &) const;
    bool pool_live_ = false, quarantined_ = false, used_ = false;
};
} // namespace tc::streaming
