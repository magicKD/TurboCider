#pragma once
#include "../backends/mlx.hpp"
namespace tc {
Tensor load_image_tensor(const std::filesystem::path &, int, int, bool reference);
// Original oriented dimensions, straight-alpha RGBA in [0,1]. Transparent
// hidden RGB may be zeroed by the decoder; visible RGB and alpha are retained.
Tensor load_rgba_image_tensor(const std::filesystem::path &);
// Experimental PE 8-bit file decoder: uint8 NHWC RGB, drop alpha without compositing,
// preserve hidden RGB, and do not apply EXIF orientation or color management.
// PNG byte parity is tested; JPEG decoder parity is not yet exact.
// Unsupported raw pixel layouts fail explicitly rather than rasterizing and
// silently discarding hidden color. Distinct from the general RGBA importer.
Tensor load_pe_image_tensor(const std::filesystem::path &);
void save_png(const Tensor &, const std::filesystem::path &);
void save_rgba_png(const Tensor &, const std::filesystem::path &);
} // namespace tc
