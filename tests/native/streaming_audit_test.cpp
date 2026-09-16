#include "../../native/runtime/streaming/audit.hpp"

#include <cassert>

int main() {
    using namespace tc::streaming;
    audit_reset();
    auto initial = audit_snapshot();
    assert(initial.framework_hooks == 0);
    assert(initial.memory_probes == 0);
    assert(initial.worker_threads == 0);
    assert(initial.pool_allocations == 0);
    assert(initial.cache_clear_or_unload_calls == 0);

    audit_increment(AuditCounter::FrameworkHooks, 2);
    audit_increment(AuditCounter::MemoryProbes, 3);
    audit_increment(AuditCounter::WorkerThreads, 4);
    audit_increment(AuditCounter::PoolAllocations, 5);
    audit_increment(AuditCounter::CacheClearOrUnloadCalls, 6);
    const auto observed = audit_snapshot();
#ifdef TURBOCIDER_ENABLE_AUDIT_COUNTERS
    assert(observed.framework_hooks == 2);
    assert(observed.memory_probes == 3);
    assert(observed.worker_threads == 4);
    assert(observed.pool_allocations == 5);
    assert(observed.cache_clear_or_unload_calls == 6);
#else
    assert(observed.framework_hooks == 0);
    assert(observed.memory_probes == 0);
    assert(observed.worker_threads == 0);
    assert(observed.pool_allocations == 0);
    assert(observed.cache_clear_or_unload_calls == 0);
#endif
    audit_reset();
    assert(audit_snapshot().framework_hooks == 0);
}
