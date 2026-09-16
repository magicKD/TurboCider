#include "memory_scheduler.hpp"
#include "memory_trace.hpp"

#include <cassert>
#include <iostream>
#include <vector>

using namespace tc;

namespace {

StorageId storage(uint64_t handle, uint64_t capacity,
                  uint64_t generation = 1) {
    return {1, handle, capacity, generation};
}

void test_stage_scope_releases_active_storage() {
    MemoryLedger ledger(100, 10);
    MemoryStageScheduler scheduler(ledger);
    {
        auto stage = scheduler.begin_stage("text");
        assert(stage.allocate(MemoryClass::Conditioning, 30,
                              storage(1, 25), "text.embedding"));
        assert(stage.active_allocation_count() == 1);
        assert(ledger.snapshot().storage_bytes == 25);
    }
    const auto snapshot = ledger.snapshot();
    assert(snapshot.storage_bytes == 0);
    assert(snapshot.reserved_bytes == 0);
}

void test_failed_allocate_does_not_leak_reservation() {
    MemoryLedger ledger(50, 10);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("denoise");
    assert(!stage.allocate(MemoryClass::Activation, 50,
                           storage(2, 50), "too-large"));
    assert(ledger.snapshot().reservation_count == 0);
    bool rejected = false;
    try {
        (void)stage.allocate(MemoryClass::Activation, 20,
                             storage(3, 21), "actual-over-upper");
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
    assert(ledger.snapshot().reservation_count == 0);
}

void test_allocation_transaction_reserves_before_real_allocation() {
    MemoryLedger ledger(100, 10);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("transaction");
    bool allocation_started = false;
    auto transaction = stage.reserve(
        MemoryClass::Weights, 50, "transaction.weights");
    assert(transaction && transaction->upper_bytes() == 50);
    auto snapshot = ledger.snapshot();
    assert(snapshot.reserved_bytes == 50);
    assert(snapshot.storage_bytes == 0);

    allocation_started = true;
    StorageId allocated = storage(30, 0);
    auto lease = transaction->commit(allocated, 40);
    assert(allocation_started);
    snapshot = ledger.snapshot();
    assert(snapshot.reserved_bytes == 0);
    assert(snapshot.storage_bytes == 40);
    stage.adopt(std::move(lease));
    assert(stage.active_allocation_count() == 1);
}

void test_allocation_transaction_failure_cancels_reservation() {
    MemoryLedger ledger(100);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("transaction-failure");
    {
        auto transaction = stage.reserve(
            MemoryClass::Activation, 20, "transaction.activation");
        assert(transaction);
        bool rejected = false;
        try {
            auto lease = transaction->commit(storage(31, 0), 21);
            (void)lease;
        } catch (const std::exception &) {
            rejected = true;
        }
        assert(rejected);
        assert(ledger.snapshot().reserved_bytes == 20);
    }
    assert(ledger.snapshot().reserved_bytes == 0);
    assert(ledger.snapshot().reservation_count == 0);
}

void test_site_token_preserves_identity_through_commit() {
    MemoryLedger ledger(100);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("site-token");
    auto token = stage.reserve_site(
        "h3.dit.refill_slot.0", MemoryClass::RefillSlot, 30);
    assert(token && token->site() == "h3.dit.refill_slot.0");
    assert(token->memory_class() == MemoryClass::RefillSlot);
    assert(token->upper_bytes() == 30);
    auto lease = stage.commit_site(std::move(*token), storage(32, 28), 28);
    assert(lease);
    assert(token->committed_state());
    assert(!static_cast<bool>(*token));
    stage.adopt(std::move(lease));
    assert(ledger.snapshot().storage_bytes == 28);
}

void test_constrained_site_token_uses_compiled_constraint() {
    MemoryLedger ledger(100);
    ledger.set_site_constraints_required(true);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("constrained-site-token");
    MemorySiteReservationConstraint constraint;
    constraint.site_id = "h3.dit.refill_slot.0";
    constraint.epoch = 1;
    constraint.maximum_live_instances = 1;
    constraint.aggregate_upper_bytes = 30;
    constraint.instances = {{0, 30, 1, 2, 0}};
    auto token = stage.reserve_site(
        MemoryClass::RefillSlot, 20, constraint);
    assert(token && token->site() == constraint.site_id);
    auto lease = stage.commit_site(
        std::move(*token), storage(33, 20), 20);
    stage.adopt(std::move(lease));
    assert(ledger.snapshot().site_active_count == 1);
}

void test_scheduler_emits_owner_thread_lifecycle_trace() {
    MemoryLedger ledger(100);
    MemoryTraceBuffer trace(16);
    MemoryStageScheduler scheduler(ledger, &trace);
    auto stage = scheduler.begin_stage("trace-stage");
    auto token = stage.reserve_site(
        "trace.activation", MemoryClass::Activation, 20);
    assert(token);
    auto lease = stage.commit_site(
        std::move(*token), storage(34, 20, 12), 20);
    stage.adopt(std::move(lease));
    const auto pending = stage.retire_all();
    assert(pending.size() == 1);
    assert(scheduler.complete(pending.front()));
    const auto events = trace.drain();
    bool committed = false;
    bool retired = false;
    bool completed = false;
    for (const auto &event : events) {
        committed = committed || event.kind == MemoryTraceEventKind::Commit;
        retired = retired || event.kind == MemoryTraceEventKind::Retire;
        completed = completed ||
            event.kind == MemoryTraceEventKind::Completion;
    }
    assert(committed && retired && completed);
}

void test_pending_storage_waits_for_completion() {
    MemoryLedger ledger(100);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("block");
    assert(stage.allocate(MemoryClass::RefillSlot, 40,
                          storage(4, 40), "slot0"));
    const auto pending = stage.retire_all();
    assert(pending.size() == 1);
    assert(stage.active_allocation_count() == 0);
    assert(scheduler.pending_count() == 1);
    auto snapshot = ledger.snapshot();
    assert(snapshot.pending_release_bytes == 40);
    assert(snapshot.pending_release_count == 1);
    assert(scheduler.complete(pending.front()));
    assert(scheduler.pending_count() == 0);
    snapshot = ledger.snapshot();
    assert(snapshot.pending_release_bytes == 0);
    assert(snapshot.storage_count == 0);
    assert(!scheduler.complete(pending.front()));
}

void test_completion_mailbox_requires_matching_generation_and_drains_on_owner() {
    MemoryLedger ledger(100);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("mailbox");
    assert(stage.allocate(MemoryClass::RefillSlot, 40,
                          storage(5, 40, 7), "slot0"));
    const auto pending = stage.retire_all();
    assert(pending.size() == 1);
    const auto token = scheduler.expect_completion(pending.front(), 9, 7, 3, 1);
    assert(token);
    auto stale = token;
    stale.generation++;
    assert(!scheduler.post_completion(stale));
    assert(scheduler.outstanding_completion_count() == 1);
    assert(scheduler.post_completion(token));
    assert(!scheduler.post_completion(token));
    assert(scheduler.completion_mailbox_count() == 1);
    const auto drained = scheduler.drain_completion_mailbox();
    assert(drained.ok());
    assert(drained.consumed == 1);
    assert(scheduler.pending_count() == 0);
    assert(scheduler.outstanding_completion_count() == 0);
    assert(scheduler.completion_mailbox_count() == 0);
    assert(ledger.snapshot().pending_release_count == 0);
}

void test_completion_failure_releases_lease_but_marks_drain_failed() {
    MemoryLedger ledger(100);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("mailbox-failure");
    assert(stage.allocate(MemoryClass::Activation, 20,
                          storage(6, 20, 8), "activation"));
    const auto pending = stage.retire_all();
    const auto token = scheduler.expect_completion(pending.front(), 10, 8);
    assert(scheduler.post_completion(token, 17));
    const auto drained = scheduler.drain_completion_mailbox();
    assert(!drained.ok());
    assert(drained.consumed == 1 && drained.failed == 1);
    assert(ledger.snapshot().pending_release_count == 0);
}

void test_async_retire_creates_pending_and_completion_atomically_for_caller() {
    MemoryLedger ledger(100);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("async-retire");
    auto transaction = stage.reserve(
        MemoryClass::Weights, 30, "async.weight");
    assert(transaction);
    auto lease = transaction->commit(storage(7, 30, 9), 30);
    const auto token = scheduler.retire_async(std::move(lease), 11, 9, 4, 2);
    assert(token && scheduler.pending_count() == 1);
    assert(scheduler.outstanding_completion_count() == 1);
    assert(ledger.snapshot().pending_release_bytes == 30);
    assert(scheduler.post_completion(token));
    const auto drained = scheduler.drain_completion_mailbox();
    assert(drained.ok() && drained.consumed == 1);
    assert(ledger.snapshot().storage_count == 0);
}

void test_completion_identity_rejects_domain_stage_and_slot_mismatch() {
    MemoryLedger ledger(100);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("completion-identity");
    auto transaction = stage.reserve(
        MemoryClass::Activation, 20, "completion.identity");
    assert(transaction);
    auto lease = transaction->commit(storage(8, 20, 10), 20);
    const auto token = scheduler.retire_async(
        std::move(lease), 12, 10, 5, 3);
    auto wrong = token;
    wrong.allocator_domain++;
    assert(!scheduler.post_completion(wrong));
    wrong = token;
    wrong.stage_id++;
    assert(!scheduler.post_completion(wrong));
    wrong = token;
    wrong.slot_id++;
    assert(!scheduler.post_completion(wrong));
    assert(scheduler.post_completion(token));
    const auto drained = scheduler.drain_completion_mailbox();
    assert(drained.ok() && drained.consumed == 1);
}

void test_completion_mailbox_overflow_is_sticky_but_releasable() {
    constexpr size_t capacity = 1024;
    MemoryLedger ledger(4096);
    MemoryStageScheduler scheduler(ledger);
    auto stage = scheduler.begin_stage("mailbox-overflow");
    std::vector<MemoryCompletionToken> tokens;
    tokens.reserve(capacity + 1);
    for (size_t index = 0; index <= capacity; index++) {
        auto transaction = stage.reserve(
            MemoryClass::Activation, 1, "mailbox.overflow");
        assert(transaction);
        auto lease = transaction->commit(
            storage(1000 + index, 1, 11), 1);
        tokens.push_back(scheduler.retire_async(
            std::move(lease), 13, 11, 6,
            static_cast<uint32_t>(index + 1)));
    }
    for (size_t index = 0; index < capacity; index++)
        assert(scheduler.post_completion(tokens[index]));
    assert(!scheduler.post_completion(tokens.back()));
    assert(scheduler.completion_mailbox_overflowed());
    auto drained = scheduler.drain_completion_mailbox();
    assert(!drained.ok());
    assert(drained.mailbox_overflow && drained.consumed == capacity);
    assert(scheduler.pending_count() == 1);
    assert(scheduler.outstanding_completion_count() == 1);

    /* The overflow is a permanent request failure, but the owner must still
     * be able to consume the final completion and empty the ledger before
     * quarantining the worker. */
    assert(scheduler.post_completion(tokens.back()));
    drained = scheduler.drain_completion_mailbox();
    assert(!drained.ok() && drained.consumed == 1);
    assert(scheduler.pending_count() == 0);
    assert(scheduler.outstanding_completion_count() == 0);
    assert(ledger.snapshot().storage_count == 0);
}

MemoryPressureSample sample(uint64_t accounted, uint64_t budget = 100) {
    MemoryPressureSample result;
    result.committed_bytes = accounted;
    result.budget_bytes = budget;
    result.system_available_bytes = 1000;
    result.system_reserve_bytes = 100;
    return result;
}

void test_pressure_hysteresis_and_critical_stickiness() {
    MemoryLedger ledger(100);
    MemoryStageScheduler scheduler(ledger);
    auto decision = scheduler.observe(sample(80));
    assert(decision.state == MemoryPressureState::Normal);
    decision = scheduler.observe(sample(86));
    assert(decision.state == MemoryPressureState::Tight);
    assert(decision.stop_optional_prefetch);
    assert(decision.drain_completions);
    assert(!decision.abort_request);
    for (int index = 0; index < 2; ++index) {
        decision = scheduler.observe(sample(79));
        assert(decision.state == MemoryPressureState::Tight);
    }
    decision = scheduler.observe(sample(79));
    assert(decision.state == MemoryPressureState::Normal);
    assert(scheduler.transition_count() == 2);

    auto critical = sample(101);
    decision = scheduler.observe(critical);
    assert(decision.state == MemoryPressureState::Critical);
    assert(decision.abort_request);
    decision = scheduler.observe(sample(10));
    assert(decision.state == MemoryPressureState::Critical);
    assert(scheduler.transition_count() == 3);
}

void test_system_reserve_is_an_independent_gate() {
    MemoryLedger ledger(100);
    MemoryStageScheduler scheduler(ledger);
    auto value = sample(20);
    value.system_available_bytes = 99;
    value.system_reserve_bytes = 100;
    const auto decision = scheduler.observe(value);
    assert(decision.state == MemoryPressureState::Critical);
    assert(decision.abort_request);
}

} // namespace

int main() {
    test_stage_scope_releases_active_storage();
    test_failed_allocate_does_not_leak_reservation();
    test_allocation_transaction_reserves_before_real_allocation();
    test_allocation_transaction_failure_cancels_reservation();
    test_site_token_preserves_identity_through_commit();
    test_constrained_site_token_uses_compiled_constraint();
    test_scheduler_emits_owner_thread_lifecycle_trace();
    test_pending_storage_waits_for_completion();
    test_completion_mailbox_requires_matching_generation_and_drains_on_owner();
    test_completion_failure_releases_lease_but_marks_drain_failed();
    test_async_retire_creates_pending_and_completion_atomically_for_caller();
    test_completion_identity_rejects_domain_stage_and_slot_mismatch();
    test_completion_mailbox_overflow_is_sticky_but_releasable();
    test_pressure_hysteresis_and_critical_stickiness();
    test_system_reserve_is_an_independent_gate();
    std::cout << "memory scheduler tests passed\n";
    return 0;
}
