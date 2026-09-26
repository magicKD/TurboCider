#pragma once
#include "../../backends/mlx.hpp"

namespace tc::qwen21 {
// Pure tensor merge; does not extend the lifetime of borrowed Core ML data.
inline Tensor merge_mlp_partitions(const Tensor &gpu, const Tensor &ane, float scale) {
    static auto merge = mx::compile([](const std::vector<Tensor> &a) {
        return std::vector<Tensor>{a[0] + mx::astype(a[1], a[0].dtype()) * a[2]};
    });
    return merge({gpu, ane, Tensor(scale, gpu.dtype())})[0];
}
} // namespace tc::qwen21
