#pragma once
#include "../../backends/mlx.hpp"
#include "cider_w8a8.hpp"
#include <array>

namespace tc::qwen21 {
// Decode-only attention projections. Each request owns four packed matrices;
// evaluation detaches the bank from the layer's streamed BF16 source buffers.
// Affine W4/W8 keeps BF16 activation arithmetic; explicit Cider W8 uses A8.
struct StagedAttentionWeights {
    std::vector<Tensor> arrays; // q/k/v/out, each packed weight, scale, bias
    int bits = 0;
    bool cider_w8a8 = false;
    uint64_t bytes() const {
        uint64_t total = 0;
        for (const auto &a : arrays) total += a.nbytes();
        return total;
    }
};

inline StagedAttentionWeights quantize_staged_attention(
        const std::array<Tensor, 4> &weights, int bits, bool cider_w8a8 = false,
        const std::string &cache_key = {}, CiderCache *cache = nullptr) {
    require(bits == 4 || bits == 8, "Qwen21 attention quantization needs 4 or 8 bits");
    require(!cider_w8a8 || (bits == 8 && cache), "Cider attention requires 8-bit weights and activations");
    StagedAttentionWeights result{{}, bits, cider_w8a8};
    constexpr std::array<const char *, 4> names{"to_q", "to_k", "to_v", "to_out.0"};
    for (size_t index = 0; index < weights.size(); ++index) {
        const auto &weight = weights[index];
        require(weight.ndim() == 2 && weight.shape(0) == weight.shape(1) &&
                    weight.shape() == weights[0].shape() &&
                    weight.shape(0) > 0 && weight.shape(1) % 64 == 0 &&
                    weight.dtype() == mx::bfloat16,
                "Qwen21 attention quantization needs square BF16 weights with group 64");
        auto packed = cider_w8a8 ? cache->pack(weight, cache_key.empty() ? "" : cache_key + "/" + names[index])
                                 : mx::quantize(weight, 64, bits, "affine");
        require(packed.size() == 3 &&
                    packed[0].dtype() == (cider_w8a8 ? mx::int8 : mx::uint32) &&
                    packed[1].dtype() == (cider_w8a8 ? mx::float32 : mx::bfloat16) &&
                    packed[2].dtype() == (cider_w8a8 ? mx::float16 : mx::bfloat16),
                "Qwen21 attention quantization changed the selected packed representation");
        result.arrays.insert(result.arrays.end(), packed.begin(), packed.end());
    }
    mx::eval(result.arrays);
    for (const auto &array : result.arrays)
        require(array.is_available() && !array.has_primitive() && array.inputs().empty() &&
                    array.siblings().empty() && array.flags().row_contiguous &&
                    array.data_size() * array.itemsize() == array.nbytes(),
                "Qwen21 attention cache retained source storage or an unfinished graph");
    return result;
}

inline Tensor staged_attention_project(const Tensor &input,
        const std::vector<Tensor> &args, size_t offset, int bits, bool cider_w8a8 = false) {
    if (cider_w8a8) {
        require(bits == 8 && input.dtype() == mx::bfloat16,
                "Cider staged attention requires W8A8 with BF16 model boundaries");
        return cider_project(input, args, offset);
    }
    require((bits == 4 || bits == 8) && offset <= args.size() && args.size() - offset >= 3 &&
                input.dtype() == mx::bfloat16 && args[offset + 1].dtype() == mx::bfloat16 &&
                args[offset + 2].dtype() == mx::bfloat16,
            "Qwen21 attention QMM arguments must keep BF16 activation/scales/biases");
    auto output = mx::quantized_matmul(input, args[offset], args[offset + 1],
                                      args[offset + 2], true, 64, bits, "affine");
    require(output.dtype() == mx::bfloat16, "Qwen21 attention QMM output must remain BF16");
    return output;
}
} // namespace tc::qwen21
