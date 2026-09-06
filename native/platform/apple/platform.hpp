#pragma once
#include "common.hpp"
namespace tc {
struct DeviceInfo {
    std::string gpu;
    uint64_t physical_memory = 0;
};
DeviceInfo device_info();
struct FluxConfiguration {
    int hidden = 0, heads = 0, dual_layers = 0, single_layers = 0;
};
FluxConfiguration flux_configuration(const std::filesystem::path &, const std::string &);
std::string sha256_file(const std::filesystem::path &);
} // namespace tc
