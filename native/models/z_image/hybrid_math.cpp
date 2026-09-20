#include "hybrid_math.hpp"

namespace tc::z_image {
std::function<std::vector<Tensor>(const std::vector<Tensor> &)>
make_hybrid_gpu_graph(int hidden, int mlp_width, int gpu_mlp_start) {
    require(hidden == 3840 && mlp_width == 10240 && gpu_mlp_start > 0 &&
                gpu_mlp_start < mlp_width,
            "unsupported Z-Image hybrid FFN geometry");
    return mx::compile(
        [hidden, mlp_width, gpu_mlp_start](const std::vector<Tensor> &args) {
            require(args.size() == 4, "Z-Image suffix requires input and three weights");
            require(args[0].ndim() == 3 && args[0].shape(0) == 1 && args[0].shape(1) > 0 &&
                        args[0].shape(2) == hidden, "Z-Image suffix input geometry mismatch");
            // Resident weights include the ANE prefix; streamed weights can
            // already be compact suffixes. Never slice the latter twice.
            const auto width = args[1].shape(0);
            require((width == mlp_width || width == mlp_width - gpu_mlp_start) &&
                        args[1].shape() == mx::Shape{width, hidden} &&
                        args[2].shape() == mx::Shape{width, hidden} &&
                        args[3].shape() == mx::Shape{hidden, width},
                    "Z-Image GPU suffix weight geometry mismatch");
            const int start = width == mlp_width ? gpu_mlp_start : 0;
            auto w1 = slice_axis(args[1], 0, start, width);
            auto w3 = slice_axis(args[2], 0, start, width);
            auto w2 = slice_axis(args[3], 1, start, width);
            auto gate = mx::matmul(args[0], mx::transpose(w1));
            auto up = mx::matmul(args[0], mx::transpose(w3));
            auto value = mx::matmul(silu(gate) * up, mx::transpose(w2));
            require(value.shape(-1) == hidden, "Z-Image hybrid FFN output mismatch");
            return std::vector<Tensor>{value};
    });
}

Tensor join_hybrid_ffn(const Tensor &gpu, const Tensor &ane, const Tensor &scale) {
    require(gpu.shape() == ane.shape() && gpu.ndim() == 3 && gpu.shape(0) == 1 &&
            gpu.shape(1) > 0 && gpu.shape(2) == 3840, "Z-Image FFN join shape mismatch");
    require(ane.dtype() == mx::float16 &&
            (gpu.dtype() == mx::bfloat16 || gpu.dtype() == mx::float16 || gpu.dtype() == mx::float32) &&
            scale.ndim() == 0 && scale.dtype() == gpu.dtype(), "Z-Image FFN join dtype/scale mismatch");
    // Deliberately cast before scaling: BF16 rounding here is part of the
    // model computation. Caller supplies scale from the verified partition.
    return gpu + mx::astype(ane, gpu.dtype()) * scale;
}
} // namespace tc::z_image
