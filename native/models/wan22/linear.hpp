#pragma once

#include "../../backends/mlx.hpp"

namespace tc::wan22 {

// Wan2.2's original dense checkpoint uses affine linears. Keep the bias in
// the GEMM epilogue so BF16 is rounded once, as with a fused affine linear.
inline Tensor linear(const Weights &weights, const Tensor &input, const std::string &prefix) {
    auto weight = weights.at(prefix + ".weight");
    if (weight.ndim() != 2)
        weight = mx::reshape(weight, {weight.shape(0), int(weight.size() / weight.shape(0))});
    require(weight.shape(1) == input.shape(-1), "invalid Wan2.2 affine geometry: " + prefix);
    if (!weights.has(prefix + ".bias")) return mx::matmul(input, mx::transpose(weight));
    return mx::addmm(mx::astype(weights.at(prefix + ".bias"), input.dtype()), input, mx::transpose(weight));
}

} // namespace tc::wan22
