#pragma once

#include <ATen/core/Tensor.h>

namespace rv {

struct ConvRotRows {
  at::Tensor codes, scales;
  int64_t rows;
};

// Uses the active LibTorch MPS stream. Callers keep the tensors in that stream
// and never read a command buffer's output before the normal MPS synchronization.
bool tensorops_hardware_available();
bool tensorops_int8_available();
bool convrot_int8_fits(int64_t rows, int64_t columns);
ConvRotRows prepare_convrot_int8(const at::Tensor& input);
at::Tensor project_int8(const ConvRotRows& input, const at::Tensor& weight,
                        const at::Tensor& scales, const at::Tensor& bias, bool gelu);

}  // namespace rv
