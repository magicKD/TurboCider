#pragma once
#include "../../backends/mlx.hpp"
namespace tc::z_image {
using HybridGpuGraph = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;
// Accept full weights or already packed suffixes; never slice a suffix twice.
HybridGpuGraph make_hybrid_gpu_graph(int hidden, int mlp_width, int gpu_mlp_start);
// Lazy join. The request owner must keep ANE backing/GPU weights alive and
// materialize all consumers before output or weight slot reuse. Not a drain.
Tensor join_hybrid_ffn(const Tensor &gpu, const Tensor &ane, const Tensor &scale);
} // namespace tc::z_image
