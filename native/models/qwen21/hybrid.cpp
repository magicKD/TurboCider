#include "hybrid.hpp"
#include "hybrid_merge.hpp"

namespace tc::qwen21 {
HybridMLP::HybridMLP(const Weights &weights, HybridSession &ane, bool gpu_w8a16)
    : ane_(ane), gpu_w8a16_(gpu_w8a16) {
    require(ane.hidden == 4096 && ane.mlp_width == 12288 && ane.ane_mlp_start == 0 &&
            ane.ane_mlp_end > 0 && ane.ane_mlp_end < 12288 && ane.block_count == 32 &&
            ane.checkpoint_sha_verified, "invalid or unverified Qwen21 MLP partition");
    const int width = ane.ane_mlp_end;
    for (int i = 0; i < 32; ++i) {
        auto prefix = "transformer_blocks." + std::to_string(i) + ".img_mlp.";
        const auto &fused = weights.at(prefix + "gate_up.weight");
        auto gate_up = mx::contiguous(mx::concatenate({slice_axis(fused, 0, width, 12288),
                                                       slice_axis(fused, 0, 12288 + width, 24576)}, 0));
        auto down = mx::contiguous(slice_axis(weights.at(prefix + "out.weight"), 1, width, 12288));
        if (gpu_w8a16_) {
            constexpr int group_size = 64;
            auto upper = mx::quantize(mx::astype(gate_up, mx::float16), group_size, 8, "affine");
            auto lower = mx::quantize(mx::astype(down, mx::float16), group_size, 8, "affine");
            require(upper.size() == 3 && lower.size() == 3,
                    "Qwen21 GPU W8A16 requires affine W8 scales and biases");
            mx::eval(upper);
            mx::eval(lower);
            fused_q_.push_back(std::move(upper[0]));
            fused_scales_.push_back(std::move(upper[1]));
            fused_biases_.push_back(std::move(upper[2]));
            down_q_.push_back(std::move(lower[0]));
            down_scales_.push_back(std::move(lower[1]));
            down_biases_.push_back(std::move(lower[2]));
        } else {
            mx::eval(gate_up, down);
            fused_.push_back(std::move(gate_up));
            down_.push_back(std::move(down));
        }
    }
    if (gpu_w8a16_) {
        suffix_ = mx::compile([](const std::vector<Tensor> &args) {
            constexpr int group_size = 64;
            auto projection = mx::quantized_matmul(args[0], args[1], args[2], args[3], true,
                                                   group_size, 8, "affine");
            auto parts = mx::split(projection, 2, -1);
            return std::vector<Tensor>{mx::quantized_matmul(silu(parts[0]) * parts[1],
                args[4], args[5], args[6], true, group_size, 8, "affine")};
        });
    } else {
        suffix_ = mx::compile([](const std::vector<Tensor> &args) {
            auto parts = mx::split(mx::matmul(args[0], mx::transpose(args[1])), 2, -1);
            return std::vector<Tensor>{mx::matmul(silu(parts[0]) * parts[1], mx::transpose(args[2]))};
        });
    }
}
Tensor HybridMLP::operator()(int block, const Tensor &input) {
    require(block >= 0 && block < 32 && input.shape() == mx::Shape{1, ane_.rows, 4096},
            "Qwen21 hybrid decode shape/block mismatch");
    auto packed = mx::contiguous(mx::astype(input, mx::float16));
    mx::eval(input, packed);
    auto gpu = gpu_w8a16_
        ? suffix_({input, fused_q_[block], fused_scales_[block], fused_biases_[block],
                   down_q_[block], down_scales_[block], down_biases_[block]})[0]
        : suffix_({input, fused_[block], down_[block]})[0];
    if (gpu_w8a16_) gpu = mx::astype(gpu, input.dtype());
    mx::async_eval(gpu);
    auto ane = ane_.predict(block, packed);
    auto result = merge_mlp_partitions(gpu, ane, ane_.output_scale);
    require(result.dtype() == input.dtype(), "Qwen21 hybrid FFN changed activation precision");
    // Transformer materializes the residual update before invoking the next
    // callback. Let that single barrier also consume this shared Core ML
    // output; a second barrier here only splits the same dependency chain.
    return result;
}
}
