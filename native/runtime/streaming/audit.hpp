#pragma once

#include <cstdint>

namespace tc::streaming {

enum class AuditCounter : uint32_t {
    FrameworkHooks = 0,
    MemoryProbes = 1,
    WorkerThreads = 2,
    PoolAllocations = 3,
    CacheClearOrUnloadCalls = 4,
    SteadyFrameworkAllocations = 5,
    SteadyFrameworkThreadCreates = 6,
};

struct AuditSnapshot {
    uint64_t framework_hooks = 0;
    uint64_t memory_probes = 0;
    uint64_t worker_threads = 0;
    uint64_t pool_allocations = 0;
    uint64_t cache_clear_or_unload_calls = 0;
    uint64_t steady_framework_allocations = 0;
    uint64_t steady_framework_thread_creates = 0;
};

#ifdef TURBOCIDER_ENABLE_AUDIT_COUNTERS
void audit_increment(AuditCounter, uint64_t amount = 1) noexcept;
void audit_reset() noexcept;
AuditSnapshot audit_snapshot() noexcept;
#else
inline void audit_increment(AuditCounter, uint64_t = 1) noexcept {}
inline void audit_reset() noexcept {}
inline AuditSnapshot audit_snapshot() noexcept { return {}; }
#endif

}  // namespace tc::streaming
