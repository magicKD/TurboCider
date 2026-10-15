#pragma once
#include "../../backends/mlx.hpp"
#include <array>

namespace tc::qwen21 {
// Owned packed weights for one GPU FFN suffix. Inputs are compact BF16
// gate/up [2*M,H] and down [H,M] slices. Evaluation detaches all six buffers
// from the source checkpoint before its layer is released.
struct StagedQuantizedSuffix {
    std::array<Tensor, 6> tensors; // gate/up q, scales, biases; down q, scales, biases
    int bits;
    static constexpr int group_size = 64;
    uint64_t bytes() const {
        uint64_t total = 0;
        for (const auto &tensor : tensors) total += tensor.nbytes();
        return total;
    }
    std::vector<Tensor> arguments(const Tensor &input) const {
        return {input, tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5]};
    }
};

inline StagedQuantizedSuffix quantize_staged_suffix(const Tensor &gate_up,
                                                   const Tensor &down, int bits) {
    require(bits == 4 || bits == 8, "Qwen21 staged suffix quantization requires 4 or 8 bits");
    require(gate_up.ndim() == 2 && down.ndim() == 2 &&
                gate_up.shape(0) > 0 && gate_up.shape(1) > 0 &&
                down.shape(0) > 0 && down.shape(1) > 0 &&
                gate_up.shape(0) == 2 * down.shape(1) && gate_up.shape(1) == down.shape(0) &&
                gate_up.shape(1) % StagedQuantizedSuffix::group_size == 0 &&
                down.shape(1) % StagedQuantizedSuffix::group_size == 0,
            "Qwen21 staged suffix geometry must match affine group 64");
    require(gate_up.dtype() == mx::bfloat16 && down.dtype() == mx::bfloat16,
            "Qwen21 staged suffix quantization requires BF16 checkpoint slices");
    auto upper = mx::quantize(mx::astype(gate_up, mx::float16), 64, bits, "affine");
    auto lower = mx::quantize(mx::astype(down, mx::float16), 64, bits, "affine");
    require(upper.size() == 3 && lower.size() == 3,
            "Qwen21 staged affine suffix requires packed weights, scales and biases");
    for (const auto *packed : {&upper, &lower})
        require((*packed)[0].dtype() == mx::uint32 &&
                    (*packed)[1].dtype() == mx::float16 && (*packed)[2].dtype() == mx::float16,
                "Qwen21 staged suffix packing changed weight/scale/bias precision");
    std::vector<Tensor> complete{upper[0], upper[1], upper[2], lower[0], lower[1], lower[2]};
    mx::eval(complete);
    return {{{std::move(upper[0]), std::move(upper[1]), std::move(upper[2]),
              std::move(lower[0]), std::move(lower[1]), std::move(lower[2])}}, bits};
}

// Arrays are explicit arguments, so compiled FFNs cannot capture a source
// checkpoint or a previous request's bank. The caller narrows BF16 input to
// FP16 and casts this result back to BF16 before the Core ML partition merge.
inline Tensor staged_quantized_suffix_forward(const std::vector<Tensor> &args, int bits) {
    require(args.size() == 7 && (bits == 4 || bits == 8),
            "Qwen21 staged quantized suffix arguments are invalid");
    require(args[0].dtype() == mx::float16 && args[1].dtype() == mx::uint32 &&
                args[2].dtype() == mx::float16 && args[3].dtype() == mx::float16 &&
                args[4].dtype() == mx::uint32 && args[5].dtype() == mx::float16 &&
                args[6].dtype() == mx::float16,
            "Qwen21 staged quantized suffix requires explicit FP16 activations/scales/biases");
    auto projected = mx::quantized_matmul(args[0], args[1], args[2], args[3], true, 64, bits, "affine");
    require(projected.dtype() == mx::float16, "Qwen21 staged suffix projection changed FP16 compute output");
    auto parts = mx::split(projected, 2, -1);
    auto output = mx::quantized_matmul(silu(parts[0]) * parts[1], args[4], args[5], args[6],
                                       true, 64, bits, "affine");
    require(output.dtype() == mx::float16, "Qwen21 staged suffix down projection changed FP16 compute output");
    return output;
}
} // namespace tc::qwen21
