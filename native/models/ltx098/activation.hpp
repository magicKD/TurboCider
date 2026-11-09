#pragma once

#include "../../backends/mlx.hpp"

namespace tc::ltx098 {

// The reference uses a single tanh-GELU operator on BF16 input. Compute the
// polynomial in FP32 and round only its output back to the activation dtype;
// the compiled graph keeps this as one elementwise GPU operation.
inline Tensor gelu_tanh(const Tensor &value) {
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        const auto x = mx::astype(args[0], mx::float32);
        const auto result = .5f * x * (1.f + mx::tanh(.7978845608028654f *
                                   (x + .044715f * x * x * x)));
        return std::vector<Tensor>{mx::astype(result, args[0].dtype())};
    }, true);
    return graph({value})[0];
}

// PyTorch's BF16 addcmul computes the product and sum before the single BF16
// output rounding. Keep the same boundary rather than rounding the product.
inline Tensor addcmul(const Tensor &base, const Tensor &left, const Tensor &right) {
    static auto graph = mx::compile([](const std::vector<Tensor> &args) {
        const auto result = mx::astype(args[0], mx::float32) +
            mx::astype(args[1], mx::float32) * mx::astype(args[2], mx::float32);
        return std::vector<Tensor>{mx::astype(result, args[0].dtype())};
    }, true);
    return graph({base, left, right})[0];
}

} // namespace tc::ltx098
