#pragma once

#include <string_view>

namespace tc::wan22 {

inline constexpr std::string_view fastwan_id = "wan2.2-ti2v-5b-fastwan";
inline constexpr std::string_view quanhaol_id = "wan2.2-ti2v-5b-quanhaol";

inline bool is_model(std::string_view id) { return id == fastwan_id || id == quanhaol_id; }

} // namespace tc::wan22
