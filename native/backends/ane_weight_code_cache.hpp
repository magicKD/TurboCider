#pragma once
#include "../core/ane_weight_code_cache_report.hpp"
#include <atomic>
#include <cstddef>
#include <exception>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace tc::ane::gpu {
inline constexpr uint64_t weight_code_cache_max_bytes = uint64_t(2) << 30;
inline constexpr size_t weight_code_cache_max_entries = 128;

inline uint64_t parse_weight_code_cache_bytes(const char *raw) {
    if (!raw) return 0;
    if (!*raw) throw std::invalid_argument("weight code cache bytes require 0..2147483648");
    uint64_t value = 0;
    for (const char *p = raw; *p; ++p) {
        if (*p < '0' || *p > '9' || value > (weight_code_cache_max_bytes-uint64_t(*p-'0'))/10)
            throw std::invalid_argument("weight code cache bytes require 0..2147483648");
        value = value*10+uint64_t(*p-'0');
    }
    return value;
}

struct WeightCodeCachePlan {
    uint64_t codes_bytes, scales_bytes, allocation_upper;
};
inline std::optional<WeightCodeCachePlan> weight_code_cache_plan(uint64_t rows, uint64_t columns,
                                                                uint64_t page) {
    if (!rows || rows > 32768 || !columns || columns > 32768 || !page || (page & (page-1)))
        return std::nullopt;
    const uint64_t codes = rows*columns, scales = rows*2;
    if (page > weight_code_cache_max_bytes || codes > weight_code_cache_max_bytes ||
        scales > weight_code_cache_max_bytes) return std::nullopt;
    const uint64_t upper = (codes+page-1)/page*page+(scales+page-1)/page*page;
    return WeightCodeCachePlan{codes,scales,upper};
}

// Only converted-buffer capacity. Graph callers charge the entire configured
// upper to their existing optional/system admission before creating payloads.
// Evicting a cache entry cannot release an in-flight ticket's capacity claim.
class WeightCodeCacheLedger {
  public:
    explicit WeightCodeCacheLedger(uint64_t limit) : limit_(limit) {
        if (limit > weight_code_cache_max_bytes) throw std::invalid_argument("weight code cache upper exceeded");
    }
    bool reserve(uint64_t bytes) {
        if (!bytes || bytes > limit_) return false;
        auto used = live_.load();
        do { if (used > limit_-bytes) return false; }
        while (!live_.compare_exchange_weak(used,used+bytes));
        auto peak = peak_.load();
        while (peak < used+bytes && !peak_.compare_exchange_weak(peak,used+bytes)) {}
        return true;
    }
    void release(uint64_t bytes) noexcept {
        if (live_.fetch_sub(bytes) < bytes) std::terminate();
    }
    uint64_t live() const { return live_.load(); }
    uint64_t peak() const { return peak_.load(); }
  private:
    uint64_t limit_;
    std::atomic<uint64_t> live_{0}, peak_{0};
};
}
