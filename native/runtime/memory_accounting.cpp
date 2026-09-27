#include "memory_accounting.hpp"

#include "../core/common.hpp"

#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace tc {
namespace detail {

struct StorageKey {
    uint64_t allocator_domain = 0;
    uint64_t handle = 0;
    uint64_t session_generation = 0;

    bool operator<(const StorageKey &other) const {
        return std::tie(allocator_domain, handle, session_generation) <
               std::tie(other.allocator_domain, other.handle,
                        other.session_generation);
    }
};

struct SiteKey {
    std::string site_id;
    uint32_t instance_id = 0;

    bool operator<(const SiteKey &other) const {
        return std::tie(site_id, instance_id) <
               std::tie(other.site_id, other.instance_id);
    }
};

enum class SiteAllocationStatus : uint8_t {
    Reserved,
    Active,
    Pending,
    Cached,
};

struct SiteAllocationRecord {
    MemoryClass memory_class = MemoryClass::UnknownExternal;
    uint64_t upper_bytes = 0;
    uint64_t live_begin = 0;
    uint64_t live_end = 0;
    uint64_t alias_group = 0;
    SiteAllocationStatus status = SiteAllocationStatus::Reserved;
    std::optional<StorageKey> storage;
};

struct ReservationRecord {
    MemoryClass memory_class = MemoryClass::UnknownExternal;
    uint64_t bytes = 0;
    std::string tag;
    std::optional<SiteKey> site;
};

struct StorageRecord {
    MemoryClass memory_class = MemoryClass::UnknownExternal;
    uint64_t capacity = 0;
    uint64_t active_references = 0;
    uint64_t pending_references = 0;
    uint64_t cached_references = 0;
};

struct PendingRecord {
    StorageKey storage;
    std::optional<SiteKey> site;
};

struct MemoryLedgerState {
    mutable std::mutex mutex;
    uint64_t budget_bytes = 0;
    uint64_t process_baseline_bytes = 0;
    uint64_t reserved_bytes = 0;
    uint64_t storage_bytes = 0;
    uint64_t peak_committed_bytes = 0;
    uint64_t peak_unknown_bytes = 0;
    uint64_t next_reservation_id = 1;
    uint64_t next_pending_id = 1;
    std::map<uint64_t, ReservationRecord> reservations;
    std::map<StorageKey, StorageRecord> storages;
    std::map<uint64_t, PendingRecord> pending;
    std::map<SiteKey, SiteAllocationRecord> site_allocations;
    bool site_constraints_required = false;
};

static StorageKey storage_key(const StorageId &storage) {
    return {storage.allocator_domain, storage.handle,
            storage.session_generation};
}

static uint64_t checked_committed(const MemoryLedgerState &state) {
    require(state.process_baseline_bytes <=
                std::numeric_limits<uint64_t>::max() - state.reserved_bytes,
            "memory_lifetime_violation: committed byte overflow");
    const uint64_t partial =
        state.process_baseline_bytes + state.reserved_bytes;
    require(partial <=
                std::numeric_limits<uint64_t>::max() - state.storage_bytes,
            "memory_lifetime_violation: committed byte overflow");
    return partial + state.storage_bytes;
}

static uint64_t unknown_bytes(const MemoryLedgerState &state) {
    uint64_t result = 0;
    for (const auto &[id, reservation] : state.reservations) {
        (void)id;
        if (reservation.memory_class == MemoryClass::UnknownExternal) {
            require(result <= std::numeric_limits<uint64_t>::max() -
                        reservation.bytes,
                    "memory_lifetime_violation: unknown byte overflow");
            result += reservation.bytes;
        }
    }
    for (const auto &[key, storage] : state.storages) {
        (void)key;
        if (storage.memory_class == MemoryClass::UnknownExternal) {
            require(result <= std::numeric_limits<uint64_t>::max() -
                        storage.capacity,
                    "memory_lifetime_violation: unknown byte overflow");
            result += storage.capacity;
        }
    }
    return result;
}

static void update_peak(MemoryLedgerState &state) {
    const uint64_t committed = checked_committed(state);
    if (committed > state.peak_committed_bytes)
        state.peak_committed_bytes = committed;
    const uint64_t unknown = unknown_bytes(state);
    if (unknown > state.peak_unknown_bytes)
        state.peak_unknown_bytes = unknown;
}

static void erase_storage_if_unreferenced(MemoryLedgerState &state,
                                          const StorageKey &key) {
    auto found = state.storages.find(key);
    if (found == state.storages.end()) return;
    const auto &record = found->second;
    if (record.active_references || record.pending_references ||
        record.cached_references)
        return;
    require(state.storage_bytes >= record.capacity,
            "memory_lifetime_violation: storage accounting underflow");
    state.storage_bytes -= record.capacity;
    state.storages.erase(found);
}

static void release_active_noexcept(
        const std::shared_ptr<MemoryLedgerState> &state,
        const StorageKey &key, const SiteKey *site_key) noexcept {
    if (!state) return;
    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        auto found = state->storages.find(key);
        if (found == state->storages.end() ||
            !found->second.active_references)
            return;
        if (site_key) {
            auto site = state->site_allocations.find(*site_key);
            require(site != state->site_allocations.end() &&
                        site->second.status == SiteAllocationStatus::Active &&
                        site->second.storage &&
                        !(*site->second.storage < key) &&
                        !(key < *site->second.storage),
                    "memory_lifetime_violation: invalid active site release");
            state->site_allocations.erase(site);
        }
        found->second.active_references--;
        erase_storage_if_unreferenced(*state, key);
    } catch (...) {
        // Destructors must not throw. A failed accounting assertion retains
        // the shared state and is surfaced by explicit operations/tests.
    }
}

} // namespace detail

const char *memory_class_name(MemoryClass memory_class) {
    switch (memory_class) {
    case MemoryClass::ProcessBaseline: return "process_baseline";
    case MemoryClass::Weights: return "weights";
    case MemoryClass::Activation: return "activation";
    case MemoryClass::Conditioning: return "conditioning";
    case MemoryClass::RefillSlot: return "refill_slot";
    case MemoryClass::ConversionScratch: return "conversion_scratch";
    case MemoryClass::Output: return "output";
    case MemoryClass::AllocatorCache: return "allocator_cache";
    case MemoryClass::CompileTemporary: return "compile_temporary";
    case MemoryClass::ChildProcessEnvelope: return "child_process_envelope";
    case MemoryClass::UnknownExternal: return "unknown_external";
    }
    return "unknown";
}

MemoryReservation::MemoryReservation(
        std::shared_ptr<detail::MemoryLedgerState> state,
        uint64_t reservation_id, uint64_t reserved_bytes)
    : state_(std::move(state)), reservation_id_(reservation_id),
      reserved_bytes_(reserved_bytes) {}

MemoryReservation::MemoryReservation(MemoryReservation &&other) noexcept
    : state_(std::move(other.state_)),
      reservation_id_(std::exchange(other.reservation_id_, 0)),
      reserved_bytes_(std::exchange(other.reserved_bytes_, 0)) {}

MemoryReservation &MemoryReservation::operator=(
        MemoryReservation &&other) noexcept {
    if (this == &other) return *this;
    cancel();
    state_ = std::move(other.state_);
    reservation_id_ = std::exchange(other.reservation_id_, 0);
    reserved_bytes_ = std::exchange(other.reserved_bytes_, 0);
    return *this;
}

MemoryReservation::~MemoryReservation() noexcept { cancel(); }

void MemoryReservation::cancel() noexcept {
    if (!state_ || !reservation_id_) return;
    try {
        std::lock_guard<std::mutex> lock(state_->mutex);
        auto found = state_->reservations.find(reservation_id_);
        if (found != state_->reservations.end()) {
            if (state_->reserved_bytes >= found->second.bytes)
                state_->reserved_bytes -= found->second.bytes;
            if (found->second.site) {
                auto site = state_->site_allocations.find(
                    *found->second.site);
                if (site != state_->site_allocations.end() &&
                    site->second.status ==
                        detail::SiteAllocationStatus::Reserved)
                    state_->site_allocations.erase(site);
            }
            state_->reservations.erase(found);
        }
    } catch (...) {
    }
    reservation_id_ = 0;
    reserved_bytes_ = 0;
    state_.reset();
}

StorageLease MemoryReservation::commit(const StorageId &storage) {
    require(state_ && reservation_id_,
            "memory_lifetime_violation: inactive reservation commit");
    require(storage.allocator_domain && storage.handle && storage.capacity,
            "memory_lifetime_violation: invalid storage identity");
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto reservation = state_->reservations.find(reservation_id_);
    require(reservation != state_->reservations.end(),
            "memory_lifetime_violation: unknown reservation");
    require(storage.capacity <= reservation->second.bytes,
            "memory_lifetime_violation: allocation exceeds reservation");

    const auto key = detail::storage_key(storage);
    std::unique_ptr<detail::SiteKey> lease_site;
    if (reservation->second.site) {
        auto site = state_->site_allocations.find(*reservation->second.site);
        require(site != state_->site_allocations.end() &&
                    site->second.status ==
                        detail::SiteAllocationStatus::Reserved,
                "memory_lifetime_violation: site reservation is inactive");
        if (site->second.alias_group) {
            for (const auto &[other_key, other] : state_->site_allocations) {
                if (other_key.site_id == reservation->second.site->site_id &&
                    other_key.instance_id ==
                        reservation->second.site->instance_id)
                    continue;
                if (other.alias_group != site->second.alias_group ||
                    !other.storage)
                    continue;
                require(!(*other.storage < key) &&
                            !(key < *other.storage),
                        "memory_lifetime_violation: alias group committed to "
                        "different storage");
            }
        }
        lease_site = std::make_unique<detail::SiteKey>(
            *reservation->second.site);
    }
    auto existing = state_->storages.find(key);
    if (existing == state_->storages.end()) {
        require(state_->storage_bytes <=
                    std::numeric_limits<uint64_t>::max() - storage.capacity,
                "memory_lifetime_violation: storage byte overflow");
        detail::StorageRecord record;
        record.memory_class = reservation->second.memory_class;
        record.capacity = storage.capacity;
        record.active_references = 1;
        state_->storages.emplace(key, record);
        state_->storage_bytes += storage.capacity;
    } else {
        require(existing->second.capacity == storage.capacity,
                "memory_lifetime_violation: aliased storage capacity changed");
        require(existing->second.memory_class ==
                    reservation->second.memory_class,
                "memory_lifetime_violation: aliased storage class changed");
        existing->second.active_references++;
    }

    if (lease_site) {
        auto site = state_->site_allocations.find(*lease_site);
        site->second.status = detail::SiteAllocationStatus::Active;
        site->second.storage = key;
    }

    require(state_->reserved_bytes >= reservation->second.bytes,
            "memory_lifetime_violation: reservation accounting underflow");
    state_->reserved_bytes -= reservation->second.bytes;
    state_->reservations.erase(reservation);
    detail::update_peak(*state_);

    StorageLease lease(
        state_, key, storage.capacity, std::move(lease_site));
    reservation_id_ = 0;
    reserved_bytes_ = 0;
    state_.reset();
    return lease;
}

StorageLease::StorageLease() = default;

StorageLease::StorageLease(std::shared_ptr<detail::MemoryLedgerState> state,
                           const detail::StorageKey &key, uint64_t capacity,
                           std::unique_ptr<detail::SiteKey> site_key)
    : state_(std::move(state)),
      key_(std::make_unique<detail::StorageKey>(key)),
      site_key_(std::move(site_key)), capacity_(capacity),
      active_(true) {}

StorageLease::StorageLease(StorageLease &&other) noexcept
    : state_(std::move(other.state_)), key_(std::move(other.key_)),
      site_key_(std::move(other.site_key_)),
      capacity_(std::exchange(other.capacity_, 0)),
      active_(std::exchange(other.active_, false)) {}

StorageLease &StorageLease::operator=(StorageLease &&other) noexcept {
    if (this == &other) return *this;
    release();
    state_ = std::move(other.state_);
    key_ = std::move(other.key_);
    site_key_ = std::move(other.site_key_);
    capacity_ = std::exchange(other.capacity_, 0);
    active_ = std::exchange(other.active_, false);
    return *this;
}

StorageLease::~StorageLease() noexcept { release(); }

void StorageLease::release() noexcept {
    if (active_ && key_)
        detail::release_active_noexcept(
            state_, *key_, site_key_ ? site_key_.get() : nullptr);
    active_ = false;
    capacity_ = 0;
    key_.reset();
    site_key_.reset();
    state_.reset();
}

PendingReleaseId StorageLease::retire() {
    require(active_ && state_ && key_,
            "memory_lifetime_violation: inactive storage retirement");
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto storage = state_->storages.find(*key_);
    require(storage != state_->storages.end() &&
                storage->second.active_references,
            "memory_lifetime_violation: unknown active storage");
    const uint64_t pending_id = state_->next_pending_id++;
    require(pending_id != 0,
            "memory_lifetime_violation: pending id overflow");
    if (site_key_) {
        auto site = state_->site_allocations.find(*site_key_);
        require(site != state_->site_allocations.end() &&
                    site->second.status ==
                        detail::SiteAllocationStatus::Active,
                "memory_lifetime_violation: inactive site retirement");
    }
    state_->pending.emplace(
        pending_id,
        detail::PendingRecord{
            *key_, site_key_ ? std::optional<detail::SiteKey>(*site_key_) :
                              std::nullopt});
    storage->second.active_references--;
    storage->second.pending_references++;
    if (site_key_)
        state_->site_allocations.at(*site_key_).status =
            detail::SiteAllocationStatus::Pending;
    active_ = false;
    capacity_ = 0;
    key_.reset();
    site_key_.reset();
    state_.reset();
    return {pending_id};
}

void StorageLease::cache() {
    require(active_ && state_ && key_,
            "memory_lifetime_violation: inactive storage cache transition");
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto storage = state_->storages.find(*key_);
    require(storage != state_->storages.end() &&
                storage->second.active_references,
            "memory_lifetime_violation: unknown active storage");
    if (site_key_) {
        auto site = state_->site_allocations.find(*site_key_);
        require(site != state_->site_allocations.end() &&
                    site->second.status ==
                        detail::SiteAllocationStatus::Active,
                "memory_lifetime_violation: inactive site cache transition");
        site->second.status = detail::SiteAllocationStatus::Cached;
    }
    storage->second.active_references--;
    storage->second.cached_references++;
    active_ = false;
    capacity_ = 0;
    key_.reset();
    site_key_.reset();
    state_.reset();
}

MemoryLedger::MemoryLedger(uint64_t budget_bytes,
                           uint64_t process_baseline_bytes)
    : state_(std::make_shared<detail::MemoryLedgerState>()) {
    require(budget_bytes > 0,
            "memory_policy_invalid: ledger budget must be positive");
    require(process_baseline_bytes <= budget_bytes,
            "memory_budget_too_small: process baseline exceeds budget");
    state_->budget_bytes = budget_bytes;
    state_->process_baseline_bytes = process_baseline_bytes;
    state_->peak_committed_bytes = process_baseline_bytes;
    state_->peak_unknown_bytes = 0;
}

void MemoryLedger::set_process_baseline(uint64_t baseline_bytes) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    require(state_->reservations.empty() && state_->storages.empty(),
            "memory_lifetime_violation: process baseline must be initialized "
            "before allocations");
    require(baseline_bytes <= state_->budget_bytes,
            "memory_budget_too_small: process baseline exceeds budget");
    state_->process_baseline_bytes = baseline_bytes;
    state_->peak_committed_bytes = baseline_bytes;
}

std::optional<MemoryReservation> MemoryLedger::try_reserve(
        MemoryClass memory_class, uint64_t upper_bytes, std::string tag) {
    require(upper_bytes > 0,
            "memory_policy_invalid: reservation must be positive");
    std::lock_guard<std::mutex> lock(state_->mutex);
    require(!state_->site_constraints_required,
            "memory_estimate_unknown: unmanifested allocation is forbidden "
            "in constrained execution");
    const uint64_t committed = detail::checked_committed(*state_);
    require(committed <= state_->budget_bytes,
            "memory_lifetime_violation: ledger is already over budget");
    if (upper_bytes > state_->budget_bytes - committed)
        return std::nullopt;
    const uint64_t reservation_id = state_->next_reservation_id++;
    require(reservation_id != 0,
            "memory_lifetime_violation: reservation id overflow");
    state_->reservations.emplace(
        reservation_id,
        detail::ReservationRecord{
            memory_class, upper_bytes, std::move(tag), std::nullopt});
    state_->reserved_bytes += upper_bytes;
    detail::update_peak(*state_);
    return MemoryReservation(state_, reservation_id, upper_bytes);
}

void MemoryLedger::set_site_constraints_required(bool required) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    require(state_->reservations.empty() && state_->storages.empty() &&
                state_->site_allocations.empty(),
            "memory_lifetime_violation: site constraint mode must be set "
            "before allocations");
    state_->site_constraints_required = required;
}

std::optional<MemoryReservation> MemoryLedger::try_reserve_site(
        MemoryClass memory_class, uint64_t upper_bytes,
        const MemorySiteReservationConstraint &constraint) {
    require(upper_bytes > 0 && !constraint.site_id.empty(),
            "memory_policy_invalid: invalid site reservation");
    require(constraint.maximum_live_instances > 0 &&
                constraint.aggregate_upper_bytes > 0 &&
                !constraint.instances.empty(),
            "memory_manifest_invalid: incomplete site constraint");
    std::lock_guard<std::mutex> lock(state_->mutex);

    uint32_t occupied_count = 0;
    uint64_t occupied_upper = 0;
    for (const auto &[key, allocation] : state_->site_allocations) {
        if (key.site_id != constraint.site_id) continue;
        require(occupied_count != std::numeric_limits<uint32_t>::max(),
                "memory_lifetime_violation: site instance count overflow");
        occupied_count++;
        require(occupied_upper <= std::numeric_limits<uint64_t>::max() -
                    allocation.upper_bytes,
                "memory_lifetime_violation: site upper overflow");
        occupied_upper += allocation.upper_bytes;
    }
    require(occupied_count < constraint.maximum_live_instances,
            "memory_lifetime_violation: runtime allocation exceeds manifest "
            "site instance count: " + constraint.site_id);
    require(upper_bytes <= constraint.aggregate_upper_bytes &&
                occupied_upper <=
                    constraint.aggregate_upper_bytes - upper_bytes,
            "memory_lifetime_violation: runtime allocation exceeds manifest "
            "site aggregate upper: " + constraint.site_id);

    const MemorySiteInstanceConstraint *selected = nullptr;
    for (const auto &instance : constraint.instances) {
        require(instance.upper_bytes > 0 &&
                    instance.live_begin < instance.live_end,
                "memory_manifest_invalid: invalid compiled site instance");
        if (constraint.epoch < instance.live_begin ||
            constraint.epoch >= instance.live_end ||
            upper_bytes > instance.upper_bytes ||
            state_->site_allocations.count(
                detail::SiteKey{constraint.site_id, instance.instance_id}))
            continue;
        if (!selected ||
            std::tie(instance.upper_bytes, instance.instance_id) <
                std::tie(selected->upper_bytes, selected->instance_id))
            selected = &instance;
    }
    require(selected != nullptr,
            "memory_lifetime_violation: no manifest allocation instance is "
            "available at epoch for " + constraint.site_id);
    if (selected->alias_group) {
        for (const auto &[key, allocation] : state_->site_allocations) {
            (void)key;
            require(allocation.alias_group != selected->alias_group,
                    "memory_lifetime_violation: alias group is still in use");
        }
    }

    const uint64_t committed = detail::checked_committed(*state_);
    require(committed <= state_->budget_bytes,
            "memory_lifetime_violation: ledger is already over budget");
    if (upper_bytes > state_->budget_bytes - committed)
        return std::nullopt;
    const uint64_t reservation_id = state_->next_reservation_id++;
    require(reservation_id != 0,
            "memory_lifetime_violation: reservation id overflow");
    const detail::SiteKey site_key{
        constraint.site_id, selected->instance_id};
    auto inserted = state_->site_allocations.emplace(
        site_key,
        detail::SiteAllocationRecord{
            memory_class, upper_bytes, selected->live_begin,
            selected->live_end, selected->alias_group,
            detail::SiteAllocationStatus::Reserved, std::nullopt});
    require(inserted.second,
            "memory_lifetime_violation: manifest allocation instance is "
            "already occupied");
    try {
        state_->reservations.emplace(
            reservation_id,
            detail::ReservationRecord{
                memory_class, upper_bytes, constraint.site_id, site_key});
    } catch (...) {
        state_->site_allocations.erase(inserted.first);
        throw;
    }
    state_->reserved_bytes += upper_bytes;
    detail::update_peak(*state_);
    return MemoryReservation(state_, reservation_id, upper_bytes);
}

void MemoryLedger::validate_site_epoch(uint64_t epoch) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (const auto &[key, allocation] : state_->site_allocations)
        require(epoch < allocation.live_end,
                "memory_lifetime_violation: allocation instance outlived "
                "manifest interval: " + key.site_id);
}

std::optional<StorageLease> MemoryLedger::reactivate_cached(
        const StorageId &storage_id) {
    if (!storage_id.allocator_domain || !storage_id.handle ||
        !storage_id.capacity)
        return std::nullopt;
    const auto key = detail::storage_key(storage_id);
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto storage = state_->storages.find(key);
    if (storage == state_->storages.end() ||
        storage->second.capacity != storage_id.capacity ||
        !storage->second.cached_references)
        return std::nullopt;
    storage->second.cached_references--;
    storage->second.active_references++;
    std::unique_ptr<detail::SiteKey> site_key;
    for (auto &[candidate, allocation] : state_->site_allocations) {
        if (allocation.status != detail::SiteAllocationStatus::Cached ||
            !allocation.storage || *allocation.storage < key ||
            key < *allocation.storage)
            continue;
        allocation.status = detail::SiteAllocationStatus::Active;
        site_key = std::make_unique<detail::SiteKey>(candidate);
        break;
    }
    return StorageLease(
        state_, key, storage_id.capacity, std::move(site_key));
}

bool MemoryLedger::drop_cached(const StorageId &storage_id) noexcept {
    try {
        const auto key = detail::storage_key(storage_id);
        std::lock_guard<std::mutex> lock(state_->mutex);
        auto storage = state_->storages.find(key);
        if (storage == state_->storages.end() ||
            storage->second.capacity != storage_id.capacity ||
            !storage->second.cached_references)
            return false;
        storage->second.cached_references--;
        for (auto site = state_->site_allocations.begin();
             site != state_->site_allocations.end(); ++site) {
            const auto &allocation = site->second;
            if (allocation.status == detail::SiteAllocationStatus::Cached &&
                allocation.storage && !(*allocation.storage < key) &&
                !(key < *allocation.storage)) {
                state_->site_allocations.erase(site);
                break;
            }
        }
        detail::erase_storage_if_unreferenced(*state_, key);
        return true;
    } catch (...) {
        return false;
    }
}

bool MemoryLedger::complete_pending(PendingReleaseId pending,
                                    bool retain_as_cache) noexcept {
    if (!pending) return false;
    try {
        std::lock_guard<std::mutex> lock(state_->mutex);
        auto entry = state_->pending.find(pending.value);
        if (entry == state_->pending.end()) return false;
        const auto key = entry->second.storage;
        auto storage = state_->storages.find(key);
        if (storage == state_->storages.end() ||
            !storage->second.pending_references)
            return false;
        if (entry->second.site) {
            auto site = state_->site_allocations.find(*entry->second.site);
            if (site == state_->site_allocations.end() ||
                site->second.status !=
                    detail::SiteAllocationStatus::Pending ||
                !site->second.storage || *site->second.storage < key ||
                key < *site->second.storage)
                return false;
            if (retain_as_cache)
                site->second.status = detail::SiteAllocationStatus::Cached;
            else
                state_->site_allocations.erase(site);
        }
        storage->second.pending_references--;
        if (retain_as_cache) storage->second.cached_references++;
        state_->pending.erase(entry);
        detail::erase_storage_if_unreferenced(*state_, key);
        return true;
    } catch (...) {
        return false;
    }
}

MemorySnapshot MemoryLedger::snapshot() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    MemorySnapshot result;
    result.budget_bytes = state_->budget_bytes;
    result.process_baseline_bytes = state_->process_baseline_bytes;
    result.storage_bytes = state_->storage_bytes;
    result.committed_bytes = detail::checked_committed(*state_);
    result.unknown_bytes = detail::unknown_bytes(*state_);
    require(state_->process_baseline_bytes <=
                std::numeric_limits<uint64_t>::max() - state_->storage_bytes,
            "memory_lifetime_violation: known byte overflow");
    /* Reservations are capacity promises, not materialized process memory.
     * Known bytes therefore include the admission baseline and unique
     * committed backing, including backing whose semantic class is still
     * UnknownExternal. */
    result.known_bytes =
        state_->process_baseline_bytes + state_->storage_bytes;
    result.active_bytes = state_->process_baseline_bytes;
    result.reserved_bytes = state_->reserved_bytes;
    result.peak_committed_bytes = state_->peak_committed_bytes;
    result.peak_unknown_bytes = state_->peak_unknown_bytes;
    result.storage_count = state_->storages.size();
    result.reservation_count = state_->reservations.size();
    result.pending_release_count = state_->pending.size();
    result.site_allocation_count = state_->site_allocations.size();
    for (const auto &[key, allocation] : state_->site_allocations) {
        (void)key;
        switch (allocation.status) {
        case detail::SiteAllocationStatus::Reserved:
            result.site_reserved_count++;
            break;
        case detail::SiteAllocationStatus::Active:
            result.site_active_count++;
            break;
        case detail::SiteAllocationStatus::Pending:
            result.site_pending_count++;
            break;
        case detail::SiteAllocationStatus::Cached:
            result.site_cached_count++;
            break;
        }
    }
    for (const auto &[key, storage] : state_->storages) {
        (void)key;
        if (storage.active_references)
            result.active_bytes += storage.capacity;
        else if (storage.pending_references)
            result.pending_release_bytes += storage.capacity;
        else if (storage.cached_references)
            result.cached_bytes += storage.capacity;
    }
    return result;
}

ProcessMemoryObservation observe_process_memory() {
    ProcessMemoryObservation result;
#if defined(__APPLE__)
    task_vm_info_data_t task_info_value{};
    mach_msg_type_number_t task_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&task_info_value),
                  &task_count) != KERN_SUCCESS)
        return result;

    uint64_t physical = 0;
    size_t physical_size = sizeof(physical);
    if (sysctlbyname("hw.memsize", &physical, &physical_size, nullptr, 0) != 0)
        return result;

    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t statistics_count = HOST_VM_INFO64_COUNT;
    const mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    const kern_return_t page_status = host_page_size(host, &page_size);
    const kern_return_t statistics_status = host_statistics64(
        host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&statistics),
        &statistics_count);
    mach_port_deallocate(mach_task_self(), host);
    if (page_status != KERN_SUCCESS || statistics_status != KERN_SUCCESS)
        return result;

    uint64_t available_pages = statistics.free_count;
    const uint64_t additions[] = {
        statistics.inactive_count,
        statistics.speculative_count,
        statistics.purgeable_count,
    };
    for (const uint64_t pages : additions) {
        if (available_pages > std::numeric_limits<uint64_t>::max() - pages)
            return result;
        available_pages += pages;
    }
    if (page_size && available_pages >
            std::numeric_limits<uint64_t>::max() / page_size)
        return result;

    result.available = true;
    result.process_footprint_bytes = task_info_value.phys_footprint;
    result.physical_memory_bytes = physical;
    result.system_available_bytes = available_pages * page_size;
    result.source = "task_vm_info.phys_footprint+host_vm_info64";
#endif
    return result;
}

SwapActivityObservation observe_swap_activity() {
    SwapActivityObservation result;
#if defined(__APPLE__)
    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t statistics_count = HOST_VM_INFO64_COUNT;
    const mach_port_t host = mach_host_self();
    const kern_return_t status = host_statistics64(
        host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&statistics),
        &statistics_count);
    mach_port_deallocate(mach_task_self(), host);
    if (status != KERN_SUCCESS)
        return result;
    result.available = true;
    result.swapins = static_cast<uint64_t>(statistics.swapins);
    result.swapouts = static_cast<uint64_t>(statistics.swapouts);
    result.compressed_pages =
        static_cast<uint64_t>(statistics.compressor_page_count);
    result.source = "host_vm_info64";
#endif
    return result;
}

MemoryAdmission::MemoryAdmission(uint64_t budget_bytes,
                                 uint64_t planned_increment_bytes,
                                 uint64_t system_reserve_bytes)
    : MemoryAdmission(budget_bytes, planned_increment_bytes, 0,
                      system_reserve_bytes, {}) {}

MemoryAdmission::MemoryAdmission(uint64_t budget_bytes,
                                 uint64_t planned_increment_bytes,
                                 uint64_t system_reserve_bytes,
                                 ProcessMemoryObserver observer)
    : MemoryAdmission(budget_bytes, planned_increment_bytes, 0,
                      system_reserve_bytes, std::move(observer)) {}

MemoryAdmission::MemoryAdmission(uint64_t budget_bytes,
                                 uint64_t planned_increment_bytes,
                                 uint64_t framework_upper_bytes,
                                 uint64_t system_reserve_bytes,
                                 ProcessMemoryObserver observer)
    : ledger_(nullptr), observer_(std::move(observer)) {
    if (!observer_)
        observer_ = [] { return observe_process_memory(); };
    require(planned_increment_bytes > 0,
            "memory_estimate_unknown: planned increment is zero");
    require(framework_upper_bytes <= planned_increment_bytes,
            "memory_budget_too_small: framework upper exceeds planned increment");
    const auto observation = observer_();
    require(observation.available,
            "memory_observation_unreliable: process footprint is unavailable");
    require(observation.process_footprint_bytes <= budget_bytes,
            "memory_budget_too_small: existing process footprint exceeds "
            "the effective budget");
    require(planned_increment_bytes <=
                budget_bytes - observation.process_footprint_bytes,
            "memory_budget_too_small: process baseline plus planned increment "
            "exceeds the effective budget");
    require(system_reserve_bytes <= observation.system_available_bytes &&
                planned_increment_bytes <=
                    observation.system_available_bytes - system_reserve_bytes,
            "memory_admission_conflict: system available memory cannot cover "
            "the planned increment and requested reserve");
    const uint64_t allocation_ceiling =
        observation.process_footprint_bytes + planned_increment_bytes;
    const uint64_t concrete_increment =
        planned_increment_bytes - framework_upper_bytes;
    require(observation.process_footprint_bytes <=
                std::numeric_limits<uint64_t>::max() - concrete_increment,
            "memory_budget_too_small: concrete allocation ceiling overflow");
    const uint64_t concrete_ceiling =
        observation.process_footprint_bytes + concrete_increment;
    ledger_ = std::make_unique<MemoryLedger>(
        concrete_ceiling, observation.process_footprint_bytes);
    metrics_.budget_bytes = budget_bytes;
    metrics_.planned_increment_bytes = planned_increment_bytes;
    metrics_.framework_upper_bytes = framework_upper_bytes;
    metrics_.planned_process_upper_bytes = allocation_ceiling;
    metrics_.allocation_ceiling_bytes = concrete_ceiling;
    framework_upper_enforced_ = framework_upper_bytes > 0;
    metrics_.initial_process_footprint_bytes =
        observation.process_footprint_bytes;
    metrics_.peak_process_footprint_bytes = observation.process_footprint_bytes;
    metrics_.final_process_footprint_bytes = observation.process_footprint_bytes;
    metrics_.system_available_bytes_at_admission = observation.system_available_bytes;
    metrics_.final_system_available_bytes = observation.system_available_bytes;
    metrics_.minimum_system_available_bytes = observation.system_available_bytes;
    metrics_.observation_count = 1;
    metrics_.last_phase = "admission";
    metrics_.observation_source = observation.source;
    const auto snapshot = ledger_->snapshot();
    metrics_.ledger_process_baseline_bytes = snapshot.process_baseline_bytes;
    metrics_.ledger_storage_bytes = snapshot.storage_bytes;
    metrics_.ledger_known_bytes = snapshot.known_bytes;
    metrics_.ledger_budget_bytes = snapshot.budget_bytes;
    metrics_.ledger_unknown_bytes = snapshot.unknown_bytes;
    metrics_.ledger_peak_unknown_bytes = snapshot.peak_unknown_bytes;
    metrics_.ledger_committed_bytes = snapshot.committed_bytes;
    metrics_.ledger_active_bytes = snapshot.active_bytes;
    metrics_.ledger_reserved_bytes = snapshot.reserved_bytes;
    metrics_.ledger_pending_release_bytes = snapshot.pending_release_bytes;
    metrics_.ledger_cached_bytes = snapshot.cached_bytes;
    metrics_.ledger_peak_committed_bytes = snapshot.peak_committed_bytes;
    metrics_.ledger_storage_count = snapshot.storage_count;
    metrics_.ledger_site_allocation_count = snapshot.site_allocation_count;
    metrics_.ledger_site_reserved_count = snapshot.site_reserved_count;
    metrics_.ledger_site_active_count = snapshot.site_active_count;
    metrics_.ledger_site_pending_count = snapshot.site_pending_count;
    metrics_.ledger_site_cached_count = snapshot.site_cached_count;
}

std::optional<MemoryReservation> MemoryAdmission::try_reserve(
        MemoryClass memory_class, uint64_t upper_bytes, std::string tag) {
    require(ledger_ != nullptr,
            "memory_lifetime_violation: admission ledger unavailable");
    return ledger_->try_reserve(memory_class, upper_bytes, std::move(tag));
}

std::optional<MemoryReservation> MemoryAdmission::try_reserve_site(
        MemoryClass memory_class, uint64_t upper_bytes,
        const MemorySiteReservationConstraint &constraint) {
    require(ledger_ != nullptr,
            "memory_lifetime_violation: admission ledger unavailable");
    return ledger_->try_reserve_site(memory_class, upper_bytes, constraint);
}

void MemoryAdmission::validate_site_epoch(uint64_t epoch) const {
    require(ledger_ != nullptr,
            "memory_lifetime_violation: admission ledger unavailable");
    ledger_->validate_site_epoch(epoch);
}

void MemoryAdmission::checkpoint(const std::string &phase) {
    const auto observation = observer_ ? observer_() : observe_process_memory();
    require(observation.available,
            "memory_observation_unreliable: process footprint sample failed");
    const auto snapshot = ledger_->snapshot();
    const uint64_t unattributed =
        observation.process_footprint_bytes > snapshot.known_bytes ?
            observation.process_footprint_bytes - snapshot.known_bytes : 0;
    {
        std::lock_guard<std::mutex> lock(observation_mutex_);
        metrics_.observation_count++;
        metrics_.last_phase = phase;
        metrics_.final_process_footprint_bytes =
            observation.process_footprint_bytes;
        metrics_.final_system_available_bytes =
            observation.system_available_bytes;
        if (observation.system_available_bytes <
            metrics_.minimum_system_available_bytes)
            metrics_.minimum_system_available_bytes =
                observation.system_available_bytes;
        if (observation.process_footprint_bytes >
            metrics_.peak_process_footprint_bytes)
            metrics_.peak_process_footprint_bytes =
                observation.process_footprint_bytes;
        metrics_.unattributed_process_footprint_bytes = unattributed;
        if (unattributed > metrics_.peak_unattributed_process_footprint_bytes)
            metrics_.peak_unattributed_process_footprint_bytes = unattributed;
        if (observation.process_footprint_bytes > metrics_.budget_bytes) {
            const uint64_t over = observation.process_footprint_bytes -
                                  metrics_.budget_bytes;
            if (over > metrics_.peak_observed_over_budget_bytes)
                metrics_.peak_observed_over_budget_bytes = over;
        }
        if (observation.process_footprint_bytes > metrics_.budget_bytes)
            metrics_.observed_within_budget = false;
        if (framework_upper_enforced_ &&
            unattributed > metrics_.framework_upper_bytes)
            metrics_.observed_within_budget = false;
    }
    require(observation.process_footprint_bytes <= metrics_.budget_bytes,
            "memory_pressure_abort: observed process footprint exceeds the "
            "effective budget at " + phase);
    if (framework_upper_enforced_)
        require(unattributed <= metrics_.framework_upper_bytes,
                "memory_pressure_abort: unattributed process footprint exceeds "
                "framework upper at " + phase);
}

MemoryAdmissionMetrics MemoryAdmission::metrics() const {
    const auto snapshot = ledger_->snapshot();
    std::lock_guard<std::mutex> lock(observation_mutex_);
    auto result = metrics_;
    result.ledger_process_baseline_bytes = snapshot.process_baseline_bytes;
    result.ledger_storage_bytes = snapshot.storage_bytes;
    result.ledger_known_bytes = snapshot.known_bytes;
    result.ledger_budget_bytes = snapshot.budget_bytes;
    result.ledger_unknown_bytes = snapshot.unknown_bytes;
    result.ledger_peak_unknown_bytes = snapshot.peak_unknown_bytes;
    result.ledger_committed_bytes = snapshot.committed_bytes;
    result.ledger_active_bytes = snapshot.active_bytes;
    result.ledger_reserved_bytes = snapshot.reserved_bytes;
    result.ledger_pending_release_bytes = snapshot.pending_release_bytes;
    result.ledger_cached_bytes = snapshot.cached_bytes;
    result.ledger_peak_committed_bytes = snapshot.peak_committed_bytes;
    result.ledger_storage_count = snapshot.storage_count;
    result.ledger_site_allocation_count = snapshot.site_allocation_count;
    result.ledger_site_reserved_count = snapshot.site_reserved_count;
    result.ledger_site_active_count = snapshot.site_active_count;
    result.ledger_site_pending_count = snapshot.site_pending_count;
    result.ledger_site_cached_count = snapshot.site_cached_count;
    result.unattributed_process_footprint_bytes =
        result.final_process_footprint_bytes > snapshot.known_bytes ?
            result.final_process_footprint_bytes - snapshot.known_bytes : 0;
    if (result.unattributed_process_footprint_bytes >
        result.peak_unattributed_process_footprint_bytes)
        result.peak_unattributed_process_footprint_bytes =
            result.unattributed_process_footprint_bytes;
    return result;
}

} // namespace tc
