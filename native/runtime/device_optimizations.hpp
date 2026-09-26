#pragma once
#include <cstdint>
#include <stdexcept>
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
    bool z_image_int8_streaming = false;

    // This gate is shared by the native request path and the typed reader.
    // User profiles and experimental lookahead overrides cannot grant support.
    bool supports_z_image_streaming(bool convrot) const {
        return !convrot || z_image_int8_streaming;
    }
    unsigned z_image_stream_prefetch(bool convrot, bool hybrid, uint64_t budget,
                                     const char *override = nullptr) const {
        if (!supports_z_image_streaming(convrot))
            throw std::invalid_argument("Z-Image INT8 streaming is only enabled for Apple M5 Pro 24 GiB; use resident weights on this device");
        if (override) {
            if (!z_image_suffix_streaming)
                throw std::invalid_argument("Z-Image prefetch overrides are only enabled for Apple M5 Pro 24 GiB");
            const std::string_view value(override);
            if (value.size() != 1 || value[0] < '1' || value[0] > '8')
                throw std::invalid_argument("TURBOCIDER_Z_STREAM_PREFETCH must be 1 through 8");
            return unsigned(value[0] - '0');
        }
        return convrot && hybrid && z_image_int8_streaming && budget < (8ull << 30) ? 2 : 1;
    }
};
inline constexpr DeviceOptimizations legacy_device_optimizations{};
inline constexpr DeviceOptimizations measured_device_optimizations[] = {
    {"m5pro24-v1", "Apple M5 Pro", 24ull << 30,
     true, true, true, true, true, true, true},
};
inline const DeviceOptimizations &device_optimizations(std::string_view gpu, uint64_t memory) {
    for (const auto &profile : measured_device_optimizations)
        if (gpu == profile.gpu && memory == profile.memory)
            return profile;
    return legacy_device_optimizations;
}
} // namespace tc
