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
bool z_image_mpp_swiglu_default();
bool z_image_virtual_norm_default();
bool z_image_small_shape_metal_default();
struct FluxConfiguration {
    int hidden = 0, heads = 0, dual_layers = 0, single_layers = 0;
};
FluxConfiguration flux_configuration(const std::filesystem::path &, const std::string &);
std::string sha256_file(const std::filesystem::path &);
} // namespace tc
