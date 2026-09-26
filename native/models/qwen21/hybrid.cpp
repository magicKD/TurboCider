#include "hybrid.hpp"
#include "hybrid_merge.hpp"

namespace tc::qwen21 {
HybridMLP::HybridMLP(const Weights &weights, HybridSession &ane) : ane_(ane) {
    require(ane.hidden == 4096 && ane.mlp_width == 12288 && ane.ane_mlp_start == 0 &&
            ane.ane_mlp_end > 0 && ane.ane_mlp_end < 12288 && ane.block_count == 32 &&
            ane.checkpoint_sha_verified, "invalid or unverified Qwen21 MLP partition");
    const int width = ane.ane_mlp_end;
    for (int i = 0; i < 32; ++i) {
        auto prefix = "transformer_blocks." + std::to_string(i) + ".img_mlp.";
        const auto &fused = weights.at(prefix + "gate_up.weight");
        fused_.push_back(mx::contiguous(mx::concatenate({slice_axis(fused, 0, width, 12288),
                                                       slice_axis(fused, 0, 12288 + width, 24576)}, 0)));
        down_.push_back(mx::contiguous(slice_axis(weights.at(prefix + "out.weight"), 1, width, 12288)));
        mx::eval(fused_.back(), down_.back());
    }
    suffix_ = mx::compile([](const std::vector<Tensor> &args) {
        auto parts = mx::split(mx::matmul(args[0], mx::transpose(args[1])), 2, -1);
        return std::vector<Tensor>{mx::matmul(silu(parts[0]) * parts[1], mx::transpose(args[2]))};
    });
}
Tensor HybridMLP::operator()(int block, const Tensor &input) {
    require(block >= 0 && block < 32 && input.shape() == mx::Shape{1, ane_.rows, 4096},
            "Qwen21 hybrid decode shape/block mismatch");
    auto packed = mx::contiguous(mx::astype(input, mx::float16));
    mx::eval(input, packed);
    auto gpu = suffix_({input, fused_[block], down_[block]})[0];
    mx::async_eval(gpu);
    auto ane = ane_.predict(block, packed);
    auto result = merge_mlp_partitions(gpu, ane, ane_.output_scale);
    // Transformer materializes the residual update before invoking the next
    // callback. Let that single barrier also consume this shared Core ML
    // output; a second barrier here only splits the same dependency chain.
    return result;
}
}
