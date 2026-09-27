#include "audit.hpp"

#ifdef TURBOCIDER_ENABLE_AUDIT_COUNTERS

#include <array>
#include <atomic>

namespace tc::streaming {
namespace {
constexpr size_t counter_count = 7;
std::array<std::atomic<uint64_t>, counter_count> counters{};

size_t index(AuditCounter counter) noexcept {
    return static_cast<size_t>(counter);
}
}  // namespace

void audit_increment(AuditCounter counter, uint64_t amount) noexcept {
    counters[index(counter)].fetch_add(amount, std::memory_order_relaxed);
}

void audit_reset() noexcept {
    for (auto &counter : counters)
        counter.store(0, std::memory_order_relaxed);
}

AuditSnapshot audit_snapshot() noexcept {
    return {
        counters[index(AuditCounter::FrameworkHooks)].load(
            std::memory_order_relaxed),
        counters[index(AuditCounter::MemoryProbes)].load(
            std::memory_order_relaxed),
        counters[index(AuditCounter::WorkerThreads)].load(
            std::memory_order_relaxed),
        counters[index(AuditCounter::PoolAllocations)].load(
            std::memory_order_relaxed),
        counters[index(AuditCounter::CacheClearOrUnloadCalls)].load(
            std::memory_order_relaxed),
        counters[index(AuditCounter::SteadyFrameworkAllocations)].load(
            std::memory_order_relaxed),
        counters[index(AuditCounter::SteadyFrameworkThreadCreates)].load(
            std::memory_order_relaxed),
    };
}

}  // namespace tc::streaming

#endif
