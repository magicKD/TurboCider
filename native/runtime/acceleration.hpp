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
    int mlp_width, ane_mlp_end;
};
inline constexpr AccelerationCase measured_hybrid_cases[] = {
    {"m4pro48-flux4b-t2i-512-v1", "flux2-klein-4b", "image.generate",
     512, 512, 4, 1025, 1088, 1088, "Apple M4 Pro", 48ull << 30, 9216, 9216},
    {"m4pro48-flux4b-t2i-1024-v1", "flux2-klein-4b", "image.generate",
     1024, 1024, 4, 4097, 4160, 4160, "Apple M4 Pro", 48ull << 30, 9216, 9216},
    {"m4max64-flux4b-t2i-512-a6144-v1", "flux2-klein-4b", "image.generate",
     512, 512, 4, 1025, 1088, 1088, "Apple M4 Max", 64ull << 30, 9216, 6144},
    {"m4max64-zimage-t2i-1024-a4096-v1", "z-image-turbo", "image.generate",
     1024, 1024, 9, 4128, 4128, 4128, "Apple M4 Max", 64ull << 30, 10240, 4096},
    // Paired M5 Pro measurements: docs/design/m5-ane-adaptation.md.
    {"m5pro24-flux4b-t2i-512-a6144-v1", "flux2-klein-4b", "image.generate",
     512, 512, 4, 1025, 1088, 1088, "Apple M5 Pro", 24ull << 30, 9216, 6144}
};
inline bool has_hybrid_measurement(const std::string &model, const std::string &gpu,
                                    uint64_t memory) {
    for (const auto &entry : measured_hybrid_cases)
        if (model == entry.model && gpu == entry.gpu && memory == entry.memory)
            return true;
    return false;
}
inline bool hybrid_partition_matches(const AccelerationCase &entry, int width,
                                      int start, int end) {
    return width == entry.mlp_width && start == 0 && end == entry.ane_mlp_end;
}
inline const AccelerationCase *hybrid_case(const Request &r, int tokens,
                                           const std::string &gpu, uint64_t memory) {
    for (const auto &entry : measured_hybrid_cases)
        if (r.residency == "resident" && r.model == entry.model &&
            r.operation == entry.operation && r.inputs.empty() && r.loras.empty() &&
            r.width == entry.width && r.height == entry.height && r.steps == entry.steps &&
            tokens >= entry.min_tokens && tokens <= entry.max_tokens &&
            gpu == entry.gpu && memory == entry.memory)
            return &entry;
    return nullptr;
}
} // namespace tc
