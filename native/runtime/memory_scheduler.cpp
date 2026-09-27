#include "memory_scheduler.hpp"
#include "memory_trace.hpp"

#include "../core/common.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

namespace tc {

MemoryAllocationTxn::MemoryAllocationTxn(MemoryReservation reservation,
                                         uint64_t upper_bytes)
    : reservation_(std::move(reservation)), upper_bytes_(upper_bytes) {}

MemoryAllocationTxn::MemoryAllocationTxn(MemoryAllocationTxn &&other) noexcept
    : reservation_(std::move(other.reservation_)),
      upper_bytes_(std::exchange(other.upper_bytes_, 0)) {}

MemoryAllocationTxn &MemoryAllocationTxn::operator=(
        MemoryAllocationTxn &&other) noexcept {
    if (this == &other) return *this;
    reservation_ = std::move(other.reservation_);
    upper_bytes_ = std::exchange(other.upper_bytes_, 0);
    return *this;
}

StorageLease MemoryAllocationTxn::commit(StorageId storage,
                                         uint64_t actual_bytes) {
    require(static_cast<bool>(reservation_),
            "memory_lifetime_violation: inactive allocation transaction");
    require(actual_bytes > 0 && actual_bytes <= upper_bytes_,
            "memory_lifetime_violation: allocation exceeds transaction upper");
    if (storage.capacity) {
        require(storage.capacity == actual_bytes,
                "memory_lifetime_violation: storage capacity differs from actual allocation");
    } else {
        storage.capacity = actual_bytes;
    }
    auto lease = reservation_.commit(storage);
    upper_bytes_ = 0;
    return lease;
}

void MemoryAllocationTxn::cancel() noexcept {
    reservation_.cancel();
    upper_bytes_ = 0;
}

MemoryStageLease::MemoryStageLease(MemoryStageScheduler *scheduler,
                                   std::string name)
    : scheduler_(scheduler), name_(std::move(name)) {
    require(scheduler_ != nullptr && !name_.empty(),
            "memory_scheduler_invalid: stage name is empty");
}

MemoryStageLease::MemoryStageLease(MemoryStageLease &&other) noexcept
    : scheduler_(std::exchange(other.scheduler_, nullptr)),
      name_(std::move(other.name_)), active_(std::move(other.active_)) {}

MemoryStageLease &MemoryStageLease::operator=(MemoryStageLease &&other) noexcept {
    if (this == &other) return *this;
    reset();
    scheduler_ = std::exchange(other.scheduler_, nullptr);
    name_ = std::move(other.name_);
    active_ = std::move(other.active_);
    return *this;
}

MemoryStageLease::~MemoryStageLease() noexcept { reset(); }

void MemoryStageLease::reset() noexcept {
    /* Active StorageLease destructors release synchronously.  Pending tokens
     * remain owned by the ledger until their GPU completion is acknowledged;
     * the scheduler deliberately does not guess that completion occurred. */
    active_.clear();
    scheduler_ = nullptr;
    name_.clear();
}

bool MemoryStageLease::allocate(MemoryClass memory_class, uint64_t upper_bytes,
                                const StorageId &storage, std::string tag) {
    auto transaction = reserve(memory_class, upper_bytes, std::move(tag));
    if (!transaction) return false;
    auto lease = transaction->commit(storage, storage.capacity);
    adopt(std::move(lease));
    return true;
}

std::optional<MemoryAllocationTxn> MemoryStageLease::reserve(
        MemoryClass memory_class, uint64_t upper_bytes, std::string tag) {
    require(scheduler_ != nullptr,
            "memory_scheduler_invalid: inactive stage reservation");
    auto reservation = scheduler_->ledger_.try_reserve(
        memory_class, upper_bytes, std::move(tag));
    if (!reservation) return std::nullopt;
    return MemoryAllocationTxn(std::move(*reservation), upper_bytes);
}

std::optional<MemorySiteToken> MemoryStageLease::reserve_site(
        std::string site_id, MemoryClass memory_class,
        uint64_t upper_bytes) {
    require(!site_id.empty(),
            "memory_manifest_invalid: runtime allocation site is empty");
    auto transaction = reserve(memory_class, upper_bytes, site_id);
    if (!transaction) return std::nullopt;
    return MemorySiteToken(std::move(site_id), memory_class,
                           std::move(*transaction));
}

std::optional<MemorySiteToken> MemoryStageLease::reserve_site(
        MemoryClass memory_class, uint64_t upper_bytes,
        const MemorySiteReservationConstraint &constraint) {
    require(scheduler_ != nullptr,
            "memory_scheduler_invalid: inactive stage reservation");
    require(!constraint.site_id.empty(),
            "memory_manifest_invalid: runtime allocation site is empty");
    auto reservation = scheduler_->ledger_.try_reserve_site(
        memory_class, upper_bytes, constraint);
    if (!reservation) return std::nullopt;
    return MemorySiteToken(
        constraint.site_id, memory_class,
        MemoryAllocationTxn(std::move(*reservation), upper_bytes));
}

StorageLease MemoryStageLease::commit_site(
        MemorySiteToken &&token, StorageId storage, uint64_t actual_bytes) {
    require(static_cast<bool>(token),
            "memory_lifetime_violation: inactive allocation site token");
    const auto site = token.site_id;
    auto lease = token.transaction_.commit(storage, actual_bytes);
    if (scheduler_->trace_) {
        const auto snapshot = scheduler_->ledger_.snapshot();
        scheduler_->trace_->record(
            MemoryTraceEventKind::Commit, name_, site, 0,
            token.upper_bytes_, actual_bytes, snapshot.committed_bytes,
            snapshot.reserved_bytes, snapshot.pending_release_bytes);
    }
    token.committed = true;
    token.site_id.clear();
    token.upper_bytes_ = 0;
    return lease;
}

void MemoryStageLease::adopt(StorageLease lease) {
    require(scheduler_ != nullptr,
            "memory_scheduler_invalid: inactive stage adoption");
    require(static_cast<bool>(lease),
            "memory_lifetime_violation: cannot adopt inactive storage lease");
    active_.push_back(std::move(lease));
}

std::vector<PendingReleaseId> MemoryStageLease::retire_all() {
    require(scheduler_ != nullptr,
            "memory_scheduler_invalid: inactive stage retirement");
    std::vector<PendingReleaseId> result;
    for (auto &lease : active_) {
        if (!lease) continue;
        const auto capacity = lease.capacity();
        auto pending = lease.retire();
        result.push_back(pending);
        scheduler_->remember_pending(pending);
        if (scheduler_->trace_) {
            const auto snapshot = scheduler_->ledger_.snapshot();
            scheduler_->trace_->record(
                MemoryTraceEventKind::Retire, name_, {}, 0,
                capacity, capacity, snapshot.committed_bytes,
                snapshot.reserved_bytes, snapshot.pending_release_bytes);
        }
    }
    active_.clear();
    return result;
}

MemoryStageLease MemoryStageScheduler::begin_stage(std::string name) {
    require(!name.empty(), "memory_scheduler_invalid: stage name is empty");
    return MemoryStageLease(this, std::move(name));
}

void MemoryStageScheduler::remember_pending(PendingReleaseId pending) {
    if (!pending) return;
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(pending);
}

MemoryCompletionToken MemoryStageScheduler::expect_completion(
        PendingReleaseId pending, uint64_t allocator_domain,
        uint64_t generation, uint32_t stage_id, uint32_t slot_id) {
    require(pending && allocator_domain && generation,
            "memory_lifetime_violation: invalid completion identity");
    std::lock_guard<std::mutex> lock(mutex_);
    require(std::any_of(
                pending_.begin(), pending_.end(),
                [&](const auto &item) { return item.value == pending.value; }),
            "memory_lifetime_violation: completion does not own a pending release");
    require(completions_.size() < kMaximumCompletionRecords,
            "memory_lifetime_violation: completion registry capacity exceeded");
    for (const auto &[id, record] : completions_) {
        (void)id;
        require(record.token.pending_release != pending.value,
                "memory_lifetime_violation: pending release already has a completion token");
    }
    const uint64_t value = next_completion_token_++;
    require(value != 0,
            "memory_lifetime_violation: completion token overflow");
    MemoryCompletionToken token{
        value, pending.value, allocator_domain, generation,
        stage_id, slot_id};
    completions_.emplace(value, CompletionRecord{token, false});
    return token;
}

MemoryCompletionToken MemoryStageScheduler::retire_async(
        StorageLease lease, uint64_t allocator_domain,
        uint64_t generation, uint32_t stage_id, uint32_t slot_id) {
    require(static_cast<bool>(lease),
            "memory_lifetime_violation: cannot retire an inactive async lease");
    require(allocator_domain && generation,
            "memory_lifetime_violation: invalid async release identity");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        require(completions_.size() < kMaximumCompletionRecords,
                "memory_lifetime_violation: completion registry capacity exceeded");
    }
    const auto capacity = lease.capacity();
    const auto pending = lease.retire();
    remember_pending(pending);
    auto token = expect_completion(
        pending, allocator_domain, generation, stage_id, slot_id);
    if (trace_) {
        const auto snapshot = ledger_.snapshot();
        trace_->record(
            MemoryTraceEventKind::Retire, "async", {}, 0,
            capacity, capacity, snapshot.committed_bytes,
            snapshot.reserved_bytes, snapshot.pending_release_bytes,
            stage_id, slot_id);
    }
    return token;
}

bool MemoryStageScheduler::post_completion(
        MemoryCompletionToken token, int status,
        bool retain_as_cache) noexcept {
    if (!token) return false;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = completions_.find(token.value);
        if (found == completions_.end() || found->second.queued ||
            found->second.token.pending_release != token.pending_release ||
            found->second.token.allocator_domain != token.allocator_domain ||
            found->second.token.generation != token.generation ||
            found->second.token.stage_id != token.stage_id ||
            found->second.token.slot_id != token.slot_id)
            return false;
        if (completion_mailbox_size_ == kCompletionMailboxCapacity) {
            completion_mailbox_overflow_ = true;
            if (completion_failure_.empty())
                completion_failure_ =
                    "memory_lifetime_violation: completion mailbox overflow";
            return false;
        }
        const size_t tail =
            (completion_mailbox_head_ + completion_mailbox_size_) %
            kCompletionMailboxCapacity;
        completion_mailbox_[tail] =
            CompletionMessage{token, status, retain_as_cache};
        completion_mailbox_size_++;
        found->second.queued = true;
        return true;
    } catch (...) {
        return false;
    }
}

MemoryCompletionDrainResult
MemoryStageScheduler::drain_completion_mailbox() noexcept {
    MemoryCompletionDrainResult result;
    for (;;) {
        CompletionMessage message;
        bool have_message = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!completion_mailbox_size_) {
                result.mailbox_overflow = completion_mailbox_overflow_;
                if (result.failure.empty())
                    result.failure = completion_failure_;
                break;
            }
            message = completion_mailbox_[completion_mailbox_head_];
            completion_mailbox_head_ =
                (completion_mailbox_head_ + 1) %
                kCompletionMailboxCapacity;
            completion_mailbox_size_--;
            have_message = true;
        }
        if (!have_message) break;

        bool token_matches = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = completions_.find(message.token.value);
            token_matches = found != completions_.end() &&
                found->second.queued &&
                found->second.token.pending_release ==
                    message.token.pending_release &&
                found->second.token.allocator_domain ==
                    message.token.allocator_domain &&
                found->second.token.generation ==
                    message.token.generation &&
                found->second.token.stage_id == message.token.stage_id &&
                found->second.token.slot_id == message.token.slot_id;
            if (token_matches) completions_.erase(found);
        }
        if (!token_matches) {
            result.stale++;
            if (result.failure.empty())
                result.failure =
                    "memory_lifetime_violation: stale completion token";
            continue;
        }

        const PendingReleaseId pending{message.token.pending_release};
        const bool completed = ledger_.complete_pending(
            pending, message.retain_as_cache);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (completed) {
                pending_.erase(std::remove_if(
                    pending_.begin(), pending_.end(),
                    [&](const auto &item) {
                        return item.value == pending.value;
                    }), pending_.end());
            }
        }
        if (!completed) {
            result.stale++;
            if (result.failure.empty())
                result.failure =
                    "memory_lifetime_violation: completion release is no longer pending";
            continue;
        }
        if (trace_) {
            const auto snapshot = ledger_.snapshot();
            trace_->record(
                MemoryTraceEventKind::Completion, "mailbox", {}, 0,
                0, 0, snapshot.committed_bytes, snapshot.reserved_bytes,
                snapshot.pending_release_bytes, message.token.stage_id,
                message.token.slot_id, message.status);
        }
        result.consumed++;
        if (message.status != 0) {
            result.failed++;
            if (result.failure.empty()) {
                std::ostringstream failure;
                failure << "memory_lifetime_violation: asynchronous completion failed with status "
                        << message.status;
                result.failure = failure.str();
            }
        }
    }
    return result;
}

bool MemoryStageScheduler::complete(PendingReleaseId pending,
                                    bool retain_as_cache) noexcept {
    if (!pending) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto completion = std::find_if(
            completions_.begin(), completions_.end(),
            [&](const auto &item) {
                return item.second.token.pending_release == pending.value;
            });
        if (completion != completions_.end()) return false;
    }
    const bool completed = ledger_.complete_pending(pending, retain_as_cache);
    if (completed) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.erase(std::remove_if(
            pending_.begin(), pending_.end(),
            [&](const auto &item) { return item.value == pending.value; }),
            pending_.end());
        if (trace_) {
            const auto snapshot = ledger_.snapshot();
            trace_->record(
                MemoryTraceEventKind::Completion, "synchronous", {}, 0,
                0, 0, snapshot.committed_bytes, snapshot.reserved_bytes,
                snapshot.pending_release_bytes);
        }
    }
    return completed;
}

MemoryPressureDecision MemoryStageScheduler::observe(
        const MemoryPressureSample &sample) {
    require(sample.budget_bytes > 0,
            "memory_scheduler_invalid: pressure budget is zero");
    require(sample.committed_bytes <=
                std::numeric_limits<uint64_t>::max() - sample.reserved_bytes,
            "memory_scheduler_invalid: accounted byte overflow");
    const uint64_t accounted = sample.committed_bytes + sample.reserved_bytes;
    const bool critical = sample.allocation_failed || sample.observed_over_budget ||
        accounted > sample.budget_bytes ||
        (sample.system_available_bytes &&
         sample.system_available_bytes < sample.system_reserve_bytes);
    const auto above_percent = [](uint64_t value, uint64_t total,
                                  unsigned percent) {
        const uint64_t threshold =
            (total / 100u) * percent + ((total % 100u) * percent) / 100u;
        return value > threshold;
    };
    const bool low_system_headroom = sample.system_available_bytes &&
        sample.system_available_bytes >= sample.system_reserve_bytes &&
        sample.system_available_bytes - sample.system_reserve_bytes <
            sample.system_reserve_bytes;
    const bool tight = above_percent(accounted, sample.budget_bytes, 85) ||
                       low_system_headroom;
    const bool recovered = !above_percent(accounted, sample.budget_bytes, 80) &&
                           !low_system_headroom;
    std::lock_guard<std::mutex> lock(mutex_);
    auto next = pressure_state_;
    if (pressure_state_ == MemoryPressureState::Critical) {
        next = MemoryPressureState::Critical;
    } else if (critical) {
        next = MemoryPressureState::Critical;
        normal_recovery_samples_ = 0;
    } else if (tight) {
        next = MemoryPressureState::Tight;
        normal_recovery_samples_ = 0;
    } else if (pressure_state_ == MemoryPressureState::Tight) {
        normal_recovery_samples_ = recovered ? normal_recovery_samples_ + 1 : 0;
        if (normal_recovery_samples_ >= 3) {
            next = MemoryPressureState::Normal;
            normal_recovery_samples_ = 0;
        }
    } else {
        next = MemoryPressureState::Normal;
        normal_recovery_samples_ = 0;
    }
    if (next != pressure_state_) {
        pressure_state_ = next;
        ++transition_count_;
    }
    return {
        pressure_state_,
        pressure_state_ != MemoryPressureState::Normal,
        pressure_state_ != MemoryPressureState::Normal,
        pressure_state_ == MemoryPressureState::Critical,
    };
}

MemoryPressureState MemoryStageScheduler::pressure_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pressure_state_;
}

uint64_t MemoryStageScheduler::transition_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return transition_count_;
}

size_t MemoryStageScheduler::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
}

size_t MemoryStageScheduler::outstanding_completion_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completions_.size();
}

size_t MemoryStageScheduler::completion_mailbox_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completion_mailbox_size_;
}

bool MemoryStageScheduler::completion_mailbox_overflowed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completion_mailbox_overflow_;
}

const char *memory_pressure_state_name(MemoryPressureState state) {
    switch (state) {
    case MemoryPressureState::Normal: return "normal";
    case MemoryPressureState::Tight: return "tight";
    case MemoryPressureState::Critical: return "critical";
    }
    return "critical";
}

} // namespace tc
