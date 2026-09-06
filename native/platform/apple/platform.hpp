#pragma once
#include "common.hpp"
namespace tc {
struct DeviceInfo {
    std::string gpu;
    uint64_t physical_memory = 0;
};
DeviceInfo device_info();
void validate_flux_configuration(const std::filesystem::path &);
} // namespace tc
