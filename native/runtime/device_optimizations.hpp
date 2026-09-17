#pragma once
#include <cstdint>
#include <string_view>

namespace tc {
// Built-in device policy, independent of user-selected execution/profile JSON.
// Unknown hardware keeps the pre-M5 implementation. Add cases only with evidence.
struct DeviceOptimizations {
    const char *id = "legacy";
    const char *gpu = "";
    uint64_t memory = 0;
    bool z_image_suffix_streaming = false;
    bool z_image_hybrid_segments = false;
    bool z_image_memory_lifecycle = false;
    bool z_image_smallest_partition = false;
    bool external_automatic_partitions = false;
    bool coreml_output_copy = false;
};
inline constexpr DeviceOptimizations legacy_device_optimizations{};
inline constexpr DeviceOptimizations measured_device_optimizations[] = {
    {"m5pro24-v1", "Apple M5 Pro", 24ull << 30,
     true, true, true, true, true, true},
};
inline const DeviceOptimizations &device_optimizations(std::string_view gpu, uint64_t memory) {
    for (const auto &profile : measured_device_optimizations)
        if (gpu == profile.gpu && memory == profile.memory)
            return profile;
    return legacy_device_optimizations;
}
} // namespace tc
