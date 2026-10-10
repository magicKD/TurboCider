#pragma once
#include <cstdint>

namespace tc::ane {
// Converted W8 codes + normalized FP16 row scales, NOT source weights or
// activation cache. Capacity includes pending/escaped producer leases.
struct WeightCodeCacheReport {
    bool enabled = false;
    bool native_surface_storage = false;
    uint64_t budget_bytes = 0;
    uint64_t hits = 0, misses = 0, fills = 0, failed_fills = 0;
    uint64_t copy_hits = 0, surface_bind_hits = 0;
    uint64_t entries = 0, ready_entries = 0, retained_bytes = 0;
    uint64_t live_capacity_bytes = 0, peak_capacity_bytes = 0;
    uint64_t evictions = 0, declines = 0, ineligible = 0;
};
}
