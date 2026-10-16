#pragma once
#include "../../backends/mlx.hpp"
#include "cider_w8a8.hpp"
#include <array>

namespace tc::qwen21 {
// Owned packed weights for one GPU FFN suffix. Inputs are compact BF16
// gate/up [2*M,H] and down [H,M] slices. Evaluation detaches all six buffers
// from the source checkpoint before its layer is released.
struct StagedQuantizedSuffix {
    std::array<Tensor, 6> tensors; // gate/up q, scales, biases; down q, scales, biases
    int bits;
    bool cider_w8a8 = false;
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
                                                   const Tensor &down, int bits,
                                                   bool cider_w8a8 = false,
                                                   const std::string &cache_key = {}, CiderCache *cache = nullptr) {
    require(bits == 4 || bits == 8, "Qwen21 staged suffix quantization requires 4 or 8 bits");
    require(!cider_w8a8 || (bits == 8 && cache), "Cider suffix requires 8-bit weights and activations");
    require(gate_up.ndim() == 2 && down.ndim() == 2 &&
                gate_up.shape(0) > 0 && gate_up.shape(1) > 0 &&
                down.shape(0) > 0 && down.shape(1) > 0 &&
                gate_up.shape(0) == 2 * down.shape(1) && gate_up.shape(1) == down.shape(0) &&
                gate_up.shape(1) % StagedQuantizedSuffix::group_size == 0 &&
                down.shape(1) % StagedQuantizedSuffix::group_size == 0,
            "Qwen21 staged suffix geometry must match affine group 64");
    require(gate_up.dtype() == mx::bfloat16 && down.dtype() == mx::bfloat16,
            "Qwen21 staged suffix quantization requires BF16 checkpoint slices");
    auto upper = cider_w8a8 ? cache->pack(gate_up, cache_key.empty() ? "" : cache_key + "/gate_up")
        : mx::quantize(mx::astype(gate_up, mx::float16), 64, bits, "affine");
    auto lower = cider_w8a8 ? cache->pack(down, cache_key.empty() ? "" : cache_key + "/down")
        : mx::quantize(mx::astype(down, mx::float16), 64, bits, "affine");
    require(upper.size() == 3 && lower.size() == 3,
            "Qwen21 staged suffix requires weights, scales and biases");
    for (const auto *packed : {&upper, &lower})
        require((*packed)[0].dtype() == (cider_w8a8 ? mx::int8 : mx::uint32) &&
                    (*packed)[1].dtype() == (cider_w8a8 ? mx::float32 : mx::float16) &&
                    (*packed)[2].dtype() == mx::float16,
                "Qwen21 staged suffix packing changed weight/scale/bias precision");
    std::vector<Tensor> complete{upper[0], upper[1], upper[2], lower[0], lower[1], lower[2]};
    mx::eval(complete);
    for (const auto &array : complete)
        require(array.is_available() && !array.has_primitive() && array.inputs().empty() &&
                    array.siblings().empty() && array.flags().row_contiguous &&
                    array.data_size() * array.itemsize() == array.nbytes(),
                "Qwen21 suffix cache retained source storage or an unfinished graph");
    return {{{std::move(upper[0]), std::move(upper[1]), std::move(upper[2]),
              std::move(lower[0]), std::move(lower[1]), std::move(lower[2])}}, bits, cider_w8a8};
}

// Arrays are explicit arguments, so compiled FFNs cannot capture a source
// checkpoint or a previous request's bank. The caller narrows BF16 input to
// FP16 and casts this result back to BF16 before the Core ML partition merge.
inline Tensor staged_quantized_suffix_forward(const std::vector<Tensor> &args, int bits,
                                              bool cider_w8a8 = false) {
    require(args.size() == 7 && (bits == 4 || bits == 8),
            "Qwen21 staged quantized suffix arguments are invalid");
    if (cider_w8a8) {
        require(bits == 8 && args[0].dtype() == mx::float16,
                "Cider staged suffix requires W8A8 with FP16 model boundaries");
        auto projected = cider_project(args[0], args, 1);
        auto parts = mx::split(projected, 2, -1);
        return cider_project(silu(parts[0]) * parts[1], args, 4);
    }
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
