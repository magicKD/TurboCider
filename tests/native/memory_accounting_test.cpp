#include "memory_accounting.hpp"

#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

using namespace tc;

static StorageId storage(uint64_t handle, uint64_t capacity,
                         uint64_t generation = 1) {
    return {1, handle, capacity, generation};
}

static void test_reservation_and_alias_deduplication() {
    MemoryLedger ledger(100, 10);
    auto first = ledger.try_reserve(MemoryClass::Weights, 60, "first");
    assert(first && first->reserved_bytes() == 60);
    auto snapshot = ledger.snapshot();
    assert(snapshot.committed_bytes == 70);
    assert(snapshot.reserved_bytes == 60);

    auto first_lease = first->commit(storage(7, 40));
    snapshot = ledger.snapshot();
    assert(snapshot.committed_bytes == 50);
    assert(snapshot.active_bytes == 50);
    assert(snapshot.process_baseline_bytes == 10);
    assert(snapshot.storage_bytes == 40);
    assert(snapshot.known_bytes == 50);
    assert(snapshot.unknown_bytes == 0);
    assert(snapshot.storage_count == 1);

    auto alias = ledger.try_reserve(MemoryClass::Weights, 40, "alias");
    assert(alias);
    auto alias_lease = alias->commit(storage(7, 40));
    snapshot = ledger.snapshot();
    assert(snapshot.committed_bytes == 50);
    assert(snapshot.active_bytes == 50);
    assert(snapshot.storage_count == 1);

    first_lease.release();
    assert(ledger.snapshot().committed_bytes == 50);
    alias_lease.release();
    snapshot = ledger.snapshot();
    assert(snapshot.committed_bytes == 10);
    assert(snapshot.storage_count == 0);
}

static void test_unknown_storage_is_reported_separately() {
    MemoryLedger ledger(100, 10);
    auto reservation = ledger.try_reserve(
        MemoryClass::UnknownExternal, 25, "opaque_graph");
    assert(reservation);
    auto snapshot = ledger.snapshot();
    assert(snapshot.committed_bytes == 35);
    assert(snapshot.known_bytes == 10);
    assert(snapshot.unknown_bytes == 25);
    assert(snapshot.peak_unknown_bytes == 25);

    auto lease = reservation->commit(storage(11, 20));
    snapshot = ledger.snapshot();
    assert(snapshot.storage_bytes == 20);
    assert(snapshot.known_bytes == 30);
    assert(snapshot.unknown_bytes == 20);
    assert(snapshot.peak_unknown_bytes == 25);
    lease.release();
    assert(ledger.snapshot().unknown_bytes == 0);
}

static void test_oversized_commit_rolls_back_with_reservation_alive() {
    MemoryLedger ledger(100);
    auto reservation = ledger.try_reserve(MemoryClass::Activation, 20);
    assert(reservation);
    bool rejected = false;
    try {
        auto lease = reservation->commit(storage(8, 21));
        (void)lease;
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
    assert(ledger.snapshot().reserved_bytes == 20);
    reservation->cancel();
    assert(ledger.snapshot().committed_bytes == 0);
}

static void test_pending_release_is_not_returned_early() {
    MemoryLedger ledger(100, 10);
    auto reservation = ledger.try_reserve(MemoryClass::RefillSlot, 30);
    assert(reservation);
    auto lease = reservation->commit(storage(9, 30));
    auto pending = lease.retire();
    auto snapshot = ledger.snapshot();
    assert(snapshot.committed_bytes == 40);
    assert(snapshot.active_bytes == 10);
    assert(snapshot.pending_release_bytes == 30);
    assert(snapshot.pending_release_count == 1);

    assert(ledger.complete_pending(pending));
    snapshot = ledger.snapshot();
    assert(snapshot.committed_bytes == 10);
    assert(snapshot.pending_release_bytes == 0);
    assert(!ledger.complete_pending(pending));
}

static void test_cache_remains_charged_and_can_be_reactivated() {
    MemoryLedger ledger(100);
    const auto id = storage(10, 25);
    auto reservation = ledger.try_reserve(MemoryClass::AllocatorCache, 25);
    assert(reservation);
    auto lease = reservation->commit(id);
    lease.cache();
    auto snapshot = ledger.snapshot();
    assert(snapshot.committed_bytes == 25);
    assert(snapshot.cached_bytes == 25);

    auto reused = ledger.reactivate_cached(id);
    assert(reused);
    snapshot = ledger.snapshot();
    assert(snapshot.active_bytes == 25);
    assert(snapshot.cached_bytes == 0);
    reused->release();
    assert(ledger.snapshot().committed_bytes == 0);

    auto second = ledger.try_reserve(MemoryClass::AllocatorCache, 25);
    assert(second);
    auto cached = second->commit(id);
    cached.cache();
    assert(ledger.drop_cached(id));
    assert(ledger.snapshot().committed_bytes == 0);
    assert(!ledger.drop_cached(id));
}

static void test_reservation_admission_is_atomic() {
    MemoryLedger ledger(100);
    std::atomic<int> attempted{0};
    std::atomic<int> admitted{0};
    auto worker = [&] {
        auto reservation = ledger.try_reserve(MemoryClass::Weights, 80);
        attempted.fetch_add(1);
        while (attempted.load() != 2)
            std::this_thread::yield();
        if (reservation) admitted.fetch_add(1);
    };
    std::thread first(worker);
    std::thread second(worker);
    first.join();
    second.join();
    assert(admitted.load() == 1);
    assert(ledger.snapshot().committed_bytes == 0);
    assert(ledger.snapshot().peak_committed_bytes == 80);
}

static void test_process_admission_tracks_boundary_samples() {
    const auto observation = observe_process_memory();
    if (!observation.available) return;
    const uint64_t margin = 64ull << 20;
    assert(observation.process_footprint_bytes <= UINT64_MAX - margin);
    MemoryAdmission admission(observation.process_footprint_bytes + margin,
                              1ull << 20, 0);
    auto reservation = admission.try_reserve(
        MemoryClass::Activation, 1ull << 20, "test_activation");
    assert(reservation);
    auto lease = reservation->commit(storage(100, 1ull << 20));
    admission.checkpoint("unit_test");
    const auto metrics = admission.metrics();
    assert(metrics.observation_count >= 2);
    assert(metrics.observed_within_budget);
    assert(metrics.planned_increment_bytes == (1ull << 20));
    assert(metrics.allocation_ceiling_bytes ==
           metrics.initial_process_footprint_bytes + (1ull << 20));
    assert(metrics.ledger_budget_bytes ==
           metrics.initial_process_footprint_bytes + (1ull << 20));
    assert(metrics.ledger_process_baseline_bytes ==
           metrics.initial_process_footprint_bytes);
    assert(metrics.ledger_storage_bytes == (1ull << 20));
    assert(metrics.ledger_known_bytes ==
           metrics.initial_process_footprint_bytes + (1ull << 20));
    assert(metrics.ledger_unknown_bytes == 0);
    assert(metrics.ledger_reserved_bytes == 0);
    assert(metrics.ledger_peak_committed_bytes >=
           metrics.initial_process_footprint_bytes + (1ull << 20));
    lease.release();
}

static void test_swap_activity_probe_has_stable_apple_counters() {
    const auto first = observe_swap_activity();
    assert(first.available);
    assert(first.source == "host_vm_info64");
    const auto second = observe_swap_activity();
    assert(second.available);
    assert(second.source == first.source);
    assert(second.swapins >= first.swapins);
    assert(second.swapouts >= first.swapouts);
}

static void test_injected_observer_tracks_unattributed_and_overage() {
    std::vector<ProcessMemoryObservation> observations = {
        {true, 100, 2000, 1000, "fake"},
        {true, 150, 2000, 1000, "fake"},
        {true, 140, 2000, 1000, "fake"},
        {true, 205, 2000, 1000, "fake"},
    };
    size_t next = 0;
    MemoryAdmission admission(200, 80, 0, [&] {
        assert(next < observations.size());
        return observations[next++];
    });
    assert(admission.metrics().allocation_ceiling_bytes == 180);

    auto known = admission.try_reserve(MemoryClass::Activation, 20, "known");
    assert(known);
    auto known_lease = known->commit(storage(200, 20));
    admission.checkpoint("known");
    auto metrics = admission.metrics();
    assert(metrics.unattributed_process_footprint_bytes == 30);
    assert(metrics.peak_unattributed_process_footprint_bytes == 30);

    auto unknown = admission.try_reserve(
        MemoryClass::UnknownExternal, 10, "unknown");
    assert(unknown);
    auto unknown_lease = unknown->commit(storage(201, 10));
    admission.checkpoint("unknown");
    metrics = admission.metrics();
    assert(metrics.ledger_unknown_bytes == 10);
    assert(metrics.ledger_peak_unknown_bytes == 10);
    assert(metrics.unattributed_process_footprint_bytes == 10);
    assert(metrics.peak_unattributed_process_footprint_bytes == 30);

    bool rejected = false;
    try {
        admission.checkpoint("over_budget");
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
    metrics = admission.metrics();
    assert(!metrics.observed_within_budget);
    assert(metrics.peak_observed_over_budget_bytes == 5);
    assert(metrics.last_phase == "over_budget");
    unknown_lease.release();
    known_lease.release();
}

static void test_framework_upper_is_removed_from_concrete_ceiling() {
    std::vector<ProcessMemoryObservation> observations = {
        {true, 100, 2000, 1000, "fake"},
        {true, 180, 2000, 1000, "fake"},
        {true, 181, 2000, 1000, "fake"},
    };
    size_t next = 0;
    MemoryAdmission admission(200, 80, 20, 0, [&] {
        assert(next < observations.size());
        return observations[next++];
    });
    assert(admission.metrics().framework_upper_bytes == 20);
    assert(admission.metrics().planned_process_upper_bytes == 180);
    assert(admission.metrics().allocation_ceiling_bytes == 160);
    auto concrete = admission.try_reserve(
        MemoryClass::Activation, 60, "concrete");
    assert(concrete);
    auto lease = concrete->commit(storage(300, 60));
    assert(!admission.try_reserve(MemoryClass::Activation, 1, "overflow"));
    admission.checkpoint("framework_within_upper");
    bool rejected = false;
    try {
        admission.checkpoint("framework_over_upper");
    } catch (const std::exception &error) {
        rejected = std::string(error.what()).find("framework upper") !=
                   std::string::npos;
    }
    assert(rejected);
    lease.release();
}

static void test_site_constraints_follow_reserve_pending_and_epoch() {
    MemoryLedger ledger(100);
    MemorySiteReservationConstraint constraint;
    constraint.site_id = "synthetic.site";
    constraint.epoch = 1;
    constraint.maximum_live_instances = 2;
    constraint.aggregate_upper_bytes = 15;
    constraint.instances = {
        {0, 10, 1, 3, 0},
        {1, 10, 1, 3, 0},
    };
    auto first = ledger.try_reserve_site(
        MemoryClass::Activation, 10, constraint);
    assert(first);
    auto snapshot = ledger.snapshot();
    assert(snapshot.site_allocation_count == 1);
    assert(snapshot.site_reserved_count == 1);
    bool aggregate_rejected = false;
    try {
        (void)ledger.try_reserve_site(
            MemoryClass::Activation, 10, constraint);
    } catch (const std::exception &error) {
        aggregate_rejected = std::string(error.what()).find(
            "aggregate upper") != std::string::npos;
    }
    assert(aggregate_rejected);
    auto lease = first->commit(storage(400, 10));
    snapshot = ledger.snapshot();
    assert(snapshot.site_active_count == 1);
    const auto pending = lease.retire();
    snapshot = ledger.snapshot();
    assert(snapshot.site_pending_count == 1);
    ledger.validate_site_epoch(2);
    bool lifetime_rejected = false;
    try {
        ledger.validate_site_epoch(3);
    } catch (const std::exception &error) {
        lifetime_rejected = std::string(error.what()).find(
            "outlived manifest interval") != std::string::npos;
    }
    assert(lifetime_rejected);
    assert(ledger.complete_pending(pending));
    snapshot = ledger.snapshot();
    assert(snapshot.site_allocation_count == 0);
    assert(snapshot.storage_count == 0);
    ledger.validate_site_epoch(3);
}

int main() {
    test_reservation_and_alias_deduplication();
    test_unknown_storage_is_reported_separately();
    test_oversized_commit_rolls_back_with_reservation_alive();
    test_pending_release_is_not_returned_early();
    test_cache_remains_charged_and_can_be_reactivated();
    test_reservation_admission_is_atomic();
    test_process_admission_tracks_boundary_samples();
    test_swap_activity_probe_has_stable_apple_counters();
    test_injected_observer_tracks_unattributed_and_overage();
    test_framework_upper_is_removed_from_concrete_ceiling();
    test_site_constraints_follow_reserve_pending_and_epoch();
    std::cout << "memory accounting tests passed\n";
    return 0;
}
