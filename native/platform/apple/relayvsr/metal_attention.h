#pragma once
#include <ATen/core/Tensor.h>
#include <span>

namespace rv {
struct FlashKV;
bool flash_tensorops_available();
at::Tensor tensorops_window_attention(const at::Tensor& query, std::span<const FlashKV> planes,
                                      int height, int width, bool condition);
// Rotated Q/K and normalized V in [B,heads,tokens,64].
at::Tensor metal_local_attention(const at::Tensor& query, const at::Tensor& key,
                                 const at::Tensor& value, int height, int width,
                                 bool condition_only);
}
