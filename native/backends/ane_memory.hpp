#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace tc::ane {

inline constexpr uint64_t packed_conversion_max_workers = 8;
inline constexpr uint64_t packed_conversion_scratch_per_worker = 2048;
struct ConversionPlan { uint64_t groups, workers, scratch_upper_bytes; };
inline std::optional<ConversionPlan> plan_packed_conversion(uint64_t rows, uint64_t cols,
                                                           uint64_t hardware_threads) {
    if (!rows || rows > 1048576 || !cols || cols > 32768) return std::nullopt;
    const auto groups = (rows + 15) / 16;
    const auto workers = rows * cols < 65536 ? 1 :
        std::min({groups, packed_conversion_max_workers, std::max(uint64_t(1), hardware_threads)});
    return ConversionPlan{groups, workers, workers * packed_conversion_scratch_per_worker};
}

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

enum class StreamingMemoryPressure { Normal, Warning, Critical };

// Darwin's public sysctl uses discrete dispatch NOTE_MEMORYSTATUS values,
// not XNU's internal pressure enum. Unknown values must remain unavailable.
inline std::optional<StreamingMemoryPressure> streaming_memory_pressure_from_flags(uint32_t flags) {
    switch (flags) {
    case 1: return StreamingMemoryPressure::Normal;
    case 2: return StreamingMemoryPressure::Warning;
    case 4: return StreamingMemoryPressure::Critical;
    default: return std::nullopt;
    }
}
inline const char *streaming_memory_pressure_name(std::optional<StreamingMemoryPressure> pressure) {
    if (!pressure) return "unavailable";
    switch (*pressure) {
    case StreamingMemoryPressure::Normal: return "normal";
    case StreamingMemoryPressure::Warning: return "warning";
    case StreamingMemoryPressure::Critical: return "critical";
    }
    return "unavailable";
}

struct StreamingMemoryObservation {
    MemoryObservation memory;
    std::optional<uint64_t> file_backed_non_speculative_bytes;
    std::optional<StreamingMemoryPressure> pressure;
};

struct MemoryLimits {
    uint64_t system_reserve_bytes = uint64_t(4) << 30;
    uint64_t optional_bytes = uint64_t(2) << 30;
};

enum class MemoryDenial {
    None, Unavailable, Invalid, SystemPressure, SystemReserve, ProcessReserve,
    MlxReserve, OptionalLimit, GrowthLimit,
};

inline std::string memory_denial_reason(MemoryDenial denial, const MemoryObservation &o) {
    const char *name = "unknown";
    switch (denial) {
    case MemoryDenial::None: name = "none"; break;
    case MemoryDenial::Unavailable: name = "observation_unavailable"; break;
    case MemoryDenial::Invalid: name = "observation_invalid"; break;
    case MemoryDenial::SystemPressure: name = "system_memory_pressure"; break;
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
namespace detail {
inline MemoryDecision admit_memory_with_credit(const MemoryObservation &o, MemoryLimits limits,
                                               uint64_t resident, uint64_t growth,
                                               uint64_t reclaimable_credit) {
    if (!o.available) return {MemoryDenial::Unavailable, 0};
    if (!o.physical_bytes || o.process_bytes > o.physical_bytes ||
        o.mlx_active_bytes > o.physical_bytes || o.free_bytes > o.physical_bytes ||
        o.inactive_bytes > o.physical_bytes)
        return {MemoryDenial::Invalid, 0};
    const auto reserve = limits.system_reserve_bytes;
    const auto available = o.free_bytes + std::min(reclaimable_credit,
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
} // namespace detail

inline MemoryDecision admit_memory(const MemoryObservation &o, MemoryLimits limits,
                                   uint64_t resident, uint64_t growth) {
    return detail::admit_memory_with_credit(o, limits, resident, growth,
                                           std::min(o.inactive_bytes / 2, uint64_t(8) << 30));
}

inline MemoryDecision admit_streaming_memory(const StreamingMemoryObservation &o,
                                             MemoryLimits limits, uint64_t growth) {
    if (!o.memory.available || !o.pressure || !o.file_backed_non_speculative_bytes)
        return {MemoryDenial::Unavailable, 0};
    const auto file_backed = *o.file_backed_non_speculative_bytes;
    if (file_backed > o.memory.physical_bytes) return {MemoryDenial::Invalid, 0};
    // Mach reports pageable file-backed pages separately from anonymous
    // pages. Under normal or warning pressure, count that pool once and keep the full
    // system reserve inside the estimate to allow for active or dirty pages.
    // A second fractional reserve can reject small streamed banks on a
    // healthy desktop. Never credit anonymous inactive pages.
    const auto credit = std::min(file_backed, uint64_t(8) << 30);
    const auto decision = detail::admit_memory_with_credit(o.memory, limits, 0, growth, credit);
    if (decision.denial == MemoryDenial::Invalid || decision.denial == MemoryDenial::ProcessReserve ||
        decision.denial == MemoryDenial::MlxReserve) return decision;
    // Warning is advisory: macOS can reclaim file pages without requiring a
    // normal notification first. Optional prefix caching is disabled separately.
    // Critical pressure still pauses all new growth, including zero-growth phases.
    switch (*o.pressure) {
    case StreamingMemoryPressure::Critical: return {MemoryDenial::SystemPressure, 0};
    case StreamingMemoryPressure::Normal:
    case StreamingMemoryPressure::Warning: return decision;
    }
    return {MemoryDenial::Unavailable, 0};
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
StreamingMemoryObservation observe_streaming_memory(uint64_t mlx_active_bytes);
void require_streaming_memory_headroom(uint64_t mlx_active_bytes, uint64_t growth_bytes,
                                       const char *context);

} // namespace tc::ane
