#pragma once

#include <string_view>
#include <cstdint>
#include <cstddef>

namespace tc::wan22 {

inline constexpr std::string_view fastwan_id = "wan2.2-ti2v-5b-fastwan";
inline constexpr std::string_view quanhaol_id = "wan2.2-ti2v-5b-quanhaol";

inline bool is_model(std::string_view id) { return id == fastwan_id || id == quanhaol_id; }

// This changes GEMM/SDPA shapes. Keep it restricted to the exact device and
// software combination exercised by real-weight and full-trajectory tests.
inline int text_padding_alignment(std::string_view gpu, uint64_t memory, int os_major, int os_minor,
                                  int os_patch, std::string_view mlx_version, int valid, std::size_t sequence) {
    if (gpu == "Apple M4 Max" && memory >= (64ull << 30) && os_major == 26 && os_minor == 6 && os_patch == 2 &&
        mlx_version == "0.32.0" && valid > 0 && valid <= 256 && sequence == 512)
        return 256;
    return 0;
}

} // namespace tc::wan22
