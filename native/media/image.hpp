#pragma once
#include "../backends/mlx.hpp"
namespace tc {
Tensor load_image_tensor(const std::filesystem::path &, int, int, bool reference);
void save_png(const Tensor &, const std::filesystem::path &);
} // namespace tc
