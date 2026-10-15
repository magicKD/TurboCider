#pragma once
// Compatibility include; the implementation is shared with Public Core ML.
#include "../ane_w8_kernels.hpp"
#include "../ane_a8_single_pass_kernels.hpp"
namespace tc::ane::private_api { using gpu::w8_source; using gpu::a8_single_pass_source; }
