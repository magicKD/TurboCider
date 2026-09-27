#pragma once

#include <cstdint>

namespace tc {

enum MemoryConstrainedField : uint32_t {
    MemoryFieldEnabled = 1u << 0,
    MemoryFieldLimit = 1u << 1,
    MemoryFieldBufferPercent = 1u << 2,
    MemoryFieldMinFree = 1u << 3,
    MemoryFieldMaxRefillSlots = 1u << 4,
    MemoryFieldAllowTiling = 1u << 5,
};

struct MemoryConstrainedConfig {
    uint32_t specified_fields = 0;
    bool enabled = false;
    uint64_t limit_bytes = 0;
    unsigned buffer_percent = 15;
    uint64_t min_free_bytes = 1ull << 30;
    unsigned max_refill_slots = 3;
    bool allow_quality_preserving_tiling = true;

    // Runtime-only fields. Parsers never accept these values and profile
    // overlay never copies them. They make the effective request idempotent
    // when a ModelSession calls make_plan() again.
    bool normalized = false;
    uint64_t effective_budget_bytes = 0;
    uint64_t denoiser_budget_bytes = 0;
    unsigned refill_slots = 0;

    bool specified() const { return specified_fields != 0; }
    bool has(MemoryConstrainedField field) const {
        return (specified_fields & static_cast<uint32_t>(field)) != 0;
    }
};

} // namespace tc
