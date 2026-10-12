#pragma once
#include <string_view>
namespace tc {
inline bool gguf_raw_gpu_profile(std::string_view profile) {
    return profile=="z-raw-gpu-affine-f16-v1" || profile=="z-raw-gpu-fixed-f16-v1" || profile=="z-raw-gpu-fixed-refresident-f16-v1" || profile=="z-raw-gpu-dependency-refresident-f16-v1";
}
inline bool gguf_fixed_gpu_profile(std::string_view profile) {
    return profile=="z-raw-gpu-fixed-f16-v1" || profile=="z-raw-gpu-fixed-refresident-f16-v1" || profile=="z-raw-gpu-dependency-refresident-f16-v1";
}
inline bool gguf_resident_refiner_profile(std::string_view profile) {
    return profile=="z-raw-gpu-fixed-refresident-f16-v1" || profile=="z-raw-gpu-dependency-refresident-f16-v1";
}
inline bool gguf_dependency_gpu_profile(std::string_view profile) {
    return profile=="z-raw-gpu-dependency-refresident-f16-v1";
}
} // namespace tc
