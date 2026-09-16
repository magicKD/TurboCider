#pragma once

#include "memory_accounting.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tc {

enum class MemoryPressureState : uint8_t {
    Normal,
    Tight,
    Critical,
};

struct MemoryPressureSample {
    uint64_t committed_bytes = 0;
    uint64_t reserved_bytes = 0;
    uint64_t budget_bytes = 0;
    uint64_t process_footprint_bytes = 0;
    uint64_t system_available_bytes = 0;
    uint64_t system_reserve_bytes = 0;
    bool allocation_failed = false;
    bool observed_over_budget = false;
};

struct MemoryPressureDecision {
    MemoryPressureState state = MemoryPressureState::Normal;
    bool stop_optional_prefetch = false;
    bool drain_completions = false;
    bool abort_request = false;
};

/* Identity handed to an asynchronous completion source after a storage lease
 * has entered the ledger's pending-release state.  It is deliberately a
 * value type: Metal/IO callbacks may copy it into their closure without
 * retaining the request scheduler. */
struct MemoryCompletionToken {
    uint64_t value = 0;
    uint64_t pending_release = 0;
    uint64_t allocator_domain = 0;
    uint64_t generation = 0;
    uint32_t stage_id = 0;
    uint32_t slot_id = 0;

    explicit operator bool() const {
        return value != 0 && pending_release != 0 &&
               allocator_domain != 0 && generation != 0;
    }
};

struct MemoryCompletionDrainResult {
    uint64_t consumed = 0;
    uint64_t failed = 0;
    uint64_t stale = 0;
    bool mailbox_overflow = false;
    std::string failure;

    bool ok() const {
        return failed == 0 && stale == 0 && !mailbox_overflow &&
               failure.empty();
    }
};

class MemoryStageScheduler;
class MemoryExecutionContext;
class MemoryTraceBuffer;

class MemoryAllocationTxn {
  public:
    MemoryAllocationTxn() = default;
    MemoryAllocationTxn(MemoryAllocationTxn &&) noexcept;
    MemoryAllocationTxn &operator=(MemoryAllocationTxn &&) noexcept;
    MemoryAllocationTxn(const MemoryAllocationTxn &) = delete;
    MemoryAllocationTxn &operator=(const MemoryAllocationTxn &) = delete;
    ~MemoryAllocationTxn() noexcept = default;

    explicit operator bool() const { return bool(reservation_); }
    uint64_t upper_bytes() const { return upper_bytes_; }
    StorageLease commit(StorageId storage, uint64_t actual_bytes);
    void cancel() noexcept;

  private:
    friend class MemoryStageLease;
    MemoryAllocationTxn(MemoryReservation, uint64_t);

    MemoryReservation reservation_;
    uint64_t upper_bytes_ = 0;
};

/* A reservation carrying the manifest allocation-site identity.  Keeping the
 * identity with the transaction makes it impossible for a bridge to reserve
 * one site and later report another site only in a log message. */
class MemorySiteToken {
  public:
    MemorySiteToken() = default;
    MemorySiteToken(MemorySiteToken &&) noexcept = default;
    MemorySiteToken &operator=(MemorySiteToken &&) noexcept = default;
    MemorySiteToken(const MemorySiteToken &) = delete;
    MemorySiteToken &operator=(const MemorySiteToken &) = delete;
    ~MemorySiteToken() noexcept = default;

    explicit operator bool() const {
        return !site_id.empty() && static_cast<bool>(transaction_) &&
               !committed;
    }
    const std::string &site() const { return site_id; }
    MemoryClass memory_class() const { return memory_class_; }
    uint64_t upper_bytes() const { return upper_bytes_; }
    bool committed_state() const { return committed; }

  private:
    friend class MemoryStageLease;
    MemorySiteToken(std::string site, MemoryClass memory_class,
                    MemoryAllocationTxn transaction)
        : site_id(std::move(site)), memory_class_(memory_class),
          upper_bytes_(transaction.upper_bytes()),
          transaction_(std::move(transaction)) {}

    std::string site_id;
    MemoryClass memory_class_ = MemoryClass::UnknownExternal;
    uint64_t upper_bytes_ = 0;
    MemoryAllocationTxn transaction_;
    bool committed = false;
};

class MemoryStageLease {
  public:
    MemoryStageLease() = default;
    MemoryStageLease(MemoryStageLease &&) noexcept;
    MemoryStageLease &operator=(MemoryStageLease &&) noexcept;
    MemoryStageLease(const MemoryStageLease &) = delete;
    MemoryStageLease &operator=(const MemoryStageLease &) = delete;
    ~MemoryStageLease() noexcept;

    explicit operator bool() const { return scheduler_ != nullptr; }
    const std::string &name() const { return name_; }

    /* Production adapters reserve first, perform the real allocation, commit
     * its actual backing size, then adopt the lease into the stage. */
    std::optional<MemoryAllocationTxn> reserve(
        MemoryClass memory_class, uint64_t upper_bytes,
        std::string tag = {});
    std::optional<MemorySiteToken> reserve_site(
        std::string site_id, MemoryClass memory_class,
        uint64_t upper_bytes);
    /* Preferred constrained adapter API.  The execution context derives the
     * current epoch and compiled instance constraint; the stage only carries
     * the resulting transaction and cannot substitute another site later. */
    std::optional<MemorySiteToken> reserve_site(
        MemoryExecutionContext &context, std::string site_id,
        MemoryClass memory_class, uint64_t upper_bytes);
    /* Constrained callers must provide the compiled site/instance constraint
     * explicitly.  This overload preserves the site identity through the
     * transaction and never falls back to generic reserve(). */
    std::optional<MemorySiteToken> reserve_site(
        MemoryClass memory_class, uint64_t upper_bytes,
        const MemorySiteReservationConstraint &constraint);
    StorageLease commit_site(MemorySiteToken &&token,
                             StorageId storage, uint64_t actual_bytes);
    void adopt(StorageLease lease);

    /* Compatibility helper for logical tests and adapters whose storage is
     * already externally owned.  New allocator bridges should use reserve()
     * so the real allocation provably occurs after admission. */
    bool allocate(MemoryClass memory_class, uint64_t upper_bytes,
                  const StorageId &storage, std::string tag = {});
    std::vector<PendingReleaseId> retire_all();
    size_t active_allocation_count() const { return active_.size(); }

  private:
    friend class MemoryStageScheduler;
    MemoryStageLease(MemoryStageScheduler *, std::string);
    void reset() noexcept;

    MemoryStageScheduler *scheduler_ = nullptr;
    std::string name_;
    std::vector<StorageLease> active_;
};

class MemoryStageScheduler {
  public:
    explicit MemoryStageScheduler(MemoryLedger &ledger,
                                  MemoryTraceBuffer *trace = nullptr)
        : ledger_(ledger), trace_(trace) {}

    MemoryStageLease begin_stage(std::string name);
    /* Register a pending release before publishing its token to a callback.
     * The callback must call post_completion(); the request owner later calls
     * drain_completion_mailbox() at a safe point. */
    MemoryCompletionToken expect_completion(
        PendingReleaseId pending, uint64_t allocator_domain,
        uint64_t generation, uint32_t stage_id = 0,
        uint32_t slot_id = 0);
    MemoryCompletionToken retire_async(
        StorageLease lease, uint64_t allocator_domain,
        uint64_t generation, uint32_t stage_id = 0,
        uint32_t slot_id = 0);
    bool post_completion(MemoryCompletionToken token, int status = 0,
                         bool retain_as_cache = false) noexcept;
    MemoryCompletionDrainResult drain_completion_mailbox() noexcept;
    bool complete(PendingReleaseId pending, bool retain_as_cache = false) noexcept;
    MemoryPressureDecision observe(const MemoryPressureSample &sample);
    MemoryPressureState pressure_state() const;
    uint64_t transition_count() const;
    size_t pending_count() const;
    size_t outstanding_completion_count() const;
    size_t completion_mailbox_count() const;
    bool completion_mailbox_overflowed() const;

  private:
    friend class MemoryStageLease;
    void remember_pending(PendingReleaseId pending);

    struct CompletionRecord {
        MemoryCompletionToken token;
        bool queued = false;
    };
    struct CompletionMessage {
        MemoryCompletionToken token;
        int status = 0;
        bool retain_as_cache = false;
    };

    static constexpr size_t kMaximumCompletionRecords = 4096;
    static constexpr size_t kCompletionMailboxCapacity = 1024;

    MemoryLedger &ledger_;
    MemoryTraceBuffer *trace_ = nullptr;
    mutable std::mutex mutex_;
    MemoryPressureState pressure_state_ = MemoryPressureState::Normal;
    uint64_t transition_count_ = 0;
    unsigned normal_recovery_samples_ = 0;
    std::vector<PendingReleaseId> pending_;
    uint64_t next_completion_token_ = 1;
    std::map<uint64_t, CompletionRecord> completions_;
    std::array<CompletionMessage, kCompletionMailboxCapacity>
        completion_mailbox_{};
    size_t completion_mailbox_head_ = 0;
    size_t completion_mailbox_size_ = 0;
    bool completion_mailbox_overflow_ = false;
    std::string completion_failure_;
};

const char *memory_pressure_state_name(MemoryPressureState);

} // namespace tc
