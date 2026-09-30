#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace tc::ane {

// An opportunistic guard, not a promise that Core ML/driver memory is bounded.
// free_bytes is raw Mach free_count (which includes speculative pages).
// Inactive pages are distinct; only half, capped at 8 GiB, are credited to
// avoid rejecting a healthy resident model for file-cache reuse.
struct MemoryObservation {
    bool available = false;
    uint64_t physical_bytes = 0;
    uint64_t process_bytes = 0;
    uint64_t free_bytes = 0;
    uint64_t mlx_active_bytes = 0;
    uint64_t inactive_bytes = 0;
};

struct MemoryLimits {
    uint64_t system_reserve_bytes = uint64_t(4) << 30;
    uint64_t optional_bytes = uint64_t(2) << 30;
};

enum class MemoryDenial {
    None, Unavailable, Invalid, SystemReserve, ProcessReserve,
    MlxReserve, OptionalLimit, GrowthLimit,
};

inline std::string memory_denial_reason(MemoryDenial denial, const MemoryObservation &o) {
    const char *name = "unknown";
    switch (denial) {
    case MemoryDenial::None: name = "none"; break;
    case MemoryDenial::Unavailable: name = "observation_unavailable"; break;
    case MemoryDenial::Invalid: name = "observation_invalid"; break;
    case MemoryDenial::SystemReserve: name = "system_reserve"; break;
    case MemoryDenial::ProcessReserve: name = "process_reserve"; break;
    case MemoryDenial::MlxReserve: name = "mlx_reserve"; break;
    case MemoryDenial::OptionalLimit: name = "optional_limit"; break;
    case MemoryDenial::GrowthLimit: name = "growth_limit"; break;
    }
    return std::string(name) + "; free_mib=" + std::to_string(o.free_bytes >> 20) +
           "; inactive_mib=" + std::to_string(o.inactive_bytes >> 20) +
           "; process_mib=" + std::to_string(o.process_bytes >> 20);
}

struct MemoryDecision {
    MemoryDenial denial;
    uint64_t headroom_bytes;
    bool allowed() const { return denial == MemoryDenial::None; }
};

inline std::optional<uint64_t> free_page_bytes(uint64_t pages, uint64_t size) {
    if (!size || pages > std::numeric_limits<uint64_t>::max() / size) return std::nullopt;
    return pages * size;
}

// Resident optional bytes already contribute to process/free observations;
// charge them only to the logical optional-tier ceiling. growth is the NEW
// payload needed before old allocations can be released, not net growth.
inline MemoryDecision admit_memory(const MemoryObservation &o, MemoryLimits limits,
                                   uint64_t resident, uint64_t growth) {
    if (!o.available) return {MemoryDenial::Unavailable, 0};
    if (!o.physical_bytes || o.process_bytes > o.physical_bytes ||
        o.mlx_active_bytes > o.physical_bytes || o.free_bytes > o.physical_bytes ||
        o.inactive_bytes > o.physical_bytes)
        return {MemoryDenial::Invalid, 0};
    const auto reserve = limits.system_reserve_bytes;
    const auto credited_inactive = std::min(o.inactive_bytes / 2, uint64_t(8) << 30);
    const auto available = o.free_bytes + std::min(credited_inactive,
                                                 o.physical_bytes - o.free_bytes);
    if (reserve > o.physical_bytes || available < reserve)
        return {MemoryDenial::SystemReserve, 0};
    const auto ceiling = o.physical_bytes - reserve;
    if (o.process_bytes > ceiling) return {MemoryDenial::ProcessReserve, 0};
    if (o.mlx_active_bytes > ceiling) return {MemoryDenial::MlxReserve, 0};
    if (resident > limits.optional_bytes) return {MemoryDenial::OptionalLimit, 0};
    const auto headroom = std::min({available - reserve, ceiling - o.process_bytes,
                                   ceiling - o.mlx_active_bytes, limits.optional_bytes - resident});
    return {growth <= headroom ? MemoryDenial::None : MemoryDenial::GrowthLimit, headroom};
}

// Only the two host vectors owned by HybridFfn. MLX corrections and retained
// output, allocator slack, Core ML internals and competing processes are not
// represented by this calculation.
struct HostScratchPlan {
    uint64_t output_bytes, hidden_bytes, retained_bytes, new_payload_bytes;
};

inline std::optional<HostScratchPlan> plan_host_scratch(uint64_t rows, uint64_t hidden,
        uint64_t width, bool adapter, uint64_t output_capacity, uint64_t hidden_capacity) {
    constexpr auto max = std::numeric_limits<uint64_t>::max();
    if (!rows || !hidden || !width) return std::nullopt;
    auto bytes = [&](uint64_t cols) -> std::optional<uint64_t> {
        if (cols > max / 2 || rows > max / (2 * cols)) return std::nullopt;
        return rows * cols * 2;
    };
    const auto output = bytes(hidden);
    const auto restored = adapter ? bytes(width) : std::optional<uint64_t>(0);
    if (!output || !restored) return std::nullopt;
    const auto retained_out = std::max(output_capacity, *output);
    const auto retained_hidden = std::max(hidden_capacity, *restored);
    if (retained_out > max - retained_hidden) return std::nullopt;
    const auto new_out = *output > output_capacity ? *output : 0;
    const auto new_hidden = *restored > hidden_capacity ? *restored : 0;
    if (new_out > max - new_hidden) return std::nullopt;
    return HostScratchPlan{*output, *restored, retained_out + retained_hidden,
                           new_out + new_hidden};
}

MemoryObservation observe_runtime_memory(uint64_t mlx_active_bytes);

} // namespace tc::ane
