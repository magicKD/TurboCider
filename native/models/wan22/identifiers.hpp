#pragma once

#include <string_view>
#include <cstdint>
#include <cstddef>

namespace tc::wan22 {

inline constexpr std::string_view fastwan_id = "wan2.2-ti2v-5b-fastwan";
inline constexpr std::string_view quanhaol_id = "wan2.2-ti2v-5b-quanhaol";

inline bool is_model(std::string_view id) { return id == fastwan_id || id == quanhaol_id; }

// Shape-changing GPU optimizations stay on the device and software
// combination exercised by real-weight and full-trajectory tests.
inline bool qualified_m4_max(std::string_view gpu, uint64_t memory, int os_major, int os_minor,
                             int os_patch, std::string_view mlx_version) {
    return gpu == "Apple M4 Max" && memory >= (64ull << 30) && os_major == 26 && os_minor == 6 &&
           os_patch == 2 && mlx_version == "0.32.0";
}

inline int text_padding_alignment(std::string_view gpu, uint64_t memory, int os_major, int os_minor,
                                  int os_patch, std::string_view mlx_version, int valid, std::size_t sequence) {
    if (qualified_m4_max(gpu, memory, os_major, os_minor, os_patch, mlx_version) &&
        valid > 0 && valid <= 256 && sequence == 512)
        return 256;
    return 0;
}

inline bool fused_rope_eligible(std::string_view gpu, uint64_t memory, int os_major, int os_minor,
                                int os_patch, std::string_view mlx_version, int width, int height, int frames) {
    return qualified_m4_max(gpu, memory, os_major, os_minor, os_patch, mlx_version) && frames == 97 &&
           width >= 512 && width <= 768 && height >= 512 && height <= 768 && width % 32 == 0 && height % 32 == 0;
}

} // namespace tc::wan22
