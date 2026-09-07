#pragma once
#include "../core/contracts.hpp"
#include <string>

namespace tc {
// Compatibility is necessary but not a performance recommendation. Extend this
// table only with operation/shape/device-specific paired measurements.
struct AccelerationCase {
    const char *id;
    const char *model;
    const char *operation;
    int width, height, steps, min_tokens, max_tokens, bucket;
    const char *gpu;
    uint64_t memory;
};
inline constexpr AccelerationCase measured_hybrid_cases[] = {
    {"m4pro48-flux4b-t2i-512-v1", "flux2-klein-4b", "image.generate",
     512, 512, 4, 1025, 1088, 1088, "Apple M4 Pro", 48ull << 30},
    {"m4pro48-flux4b-t2i-1024-v1", "flux2-klein-4b", "image.generate",
     1024, 1024, 4, 4097, 4160, 4160, "Apple M4 Pro", 48ull << 30}
};
inline const AccelerationCase *hybrid_case(const Request &r, int tokens,
                                           const std::string &gpu, uint64_t memory) {
    for (const auto &entry : measured_hybrid_cases)
        if (r.residency == "resident" && r.model == entry.model && r.operation == entry.operation && r.inputs.empty() &&
            r.width == entry.width && r.height == entry.height && r.steps == entry.steps &&
            tokens >= entry.min_tokens && tokens <= entry.max_tokens &&
            gpu == entry.gpu && memory == entry.memory)
            return &entry;
    return nullptr;
}
} // namespace tc
