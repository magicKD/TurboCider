#pragma once
#include "common.hpp"
#include "../../runtime/device_optimizations.hpp"
namespace tc {
struct DeviceInfo {
    std::string gpu;
    uint64_t physical_memory = 0;
    const DeviceOptimizations &optimizations() const {
        return device_optimizations(gpu, physical_memory);
    }
};
DeviceInfo device_info();
std::filesystem::path qwen21_dit_cache_root();
bool z_image_mpp_swiglu_default();
bool z_image_virtual_norm_default();
bool z_image_small_shape_metal_default();
struct FluxConfiguration {
    int hidden = 0, heads = 0, dual_layers = 0, single_layers = 0;
};
FluxConfiguration flux_configuration(const std::filesystem::path &, const std::string &);
std::string sha256_file(const std::filesystem::path &);
// File and hash failures return an empty digest so callers can report context.
std::string sha256_file_or_empty(const std::filesystem::path &);
// Cancellable CPU-only overload. Caller retains and validates source identity.
std::string sha256_file(int source_fd, const std::atomic<bool> &cancelled);
struct CheckpointHashCache {
    std::filesystem::path path;
    std::filesystem::file_time_type mtime{};
    uintmax_t bytes = 0;
    std::string sha256;
};
std::string cached_checkpoint_sha256(const std::filesystem::path &,
                                     CheckpointHashCache &, const char *model);
} // namespace tc
