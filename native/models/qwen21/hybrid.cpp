#include "hybrid.hpp"
#include "hybrid_merge.hpp"
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace tc::qwen21 {
HybridMLP::HybridMLP(const Weights &weights, HybridSession &ane, bool gpu_w8a16,
                     const std::vector<int> &gpu_full_blocks)
    : weights_(weights), ane_(ane), gpu_w8a16_(gpu_w8a16),
      runtime_lora_suffix_(weights.has_runtime_loras()) {
    require(!runtime_lora_suffix_ || !gpu_w8a16_,
            "Qwen21 runtime LoRA GPU suffix requires BF16 weights");
    require(!runtime_lora_suffix_ || gpu_full_blocks.empty(),
            "Qwen21 runtime LoRA cannot use a base-only full GPU FFN block");
    require(ane.hidden == 4096 && ane.mlp_width == 12288 && ane.ane_mlp_start == 0 &&
            ane.ane_mlp_end > 0 && ane.ane_mlp_end < 12288 && ane.block_count == 32 &&
            ane.checkpoint_sha_verified, "invalid or unverified Qwen21 MLP partition");
    require(ane.mlp_output_kind.empty() ||
                ((ane.mlp_output_kind == "gate_up" || ane.mlp_output_kind == "fused_lora") &&
                 ane.output_channels == (ane.mlp_output_kind == "gate_up"
                    ? 2 * ane.ane_mlp_end : 4096 + ane.ane_mlp_end) &&
                 !gpu_w8a16_ && gpu_full_blocks.empty()),
            "Qwen21 LoRA graph needs a BF16 GPU complement");
    require(gpu_full_blocks.size() <= 3, "Qwen21 W8A8 requires at least 29/32 hybrid FFN layers");
    for (int block : gpu_full_blocks) {
        require(block >= 0 && block < 32 && !gpu_full_blocks_[block],
                "Qwen21 full-GPU FFN fallback needs unique layer indices in [0,31]");
        gpu_full_blocks_[block] = true;
    }
    const int width = ane.ane_mlp_end;
    for (int i = 0; i < 32; ++i) {
        auto prefix = "transformer_blocks." + std::to_string(i) + ".img_mlp.";
        const auto &fused = weights.at(prefix + "gate_up.weight");
        if (gpu_full_blocks_[i])
            full_weights_.emplace(i, std::make_pair(fused, weights.at(prefix + "out.weight")));
        if (runtime_lora_suffix_) continue; // use base slices plus low-rank updates at inference
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
    if (!runtime_lora_suffix_ && gpu_w8a16_) {
        suffix_ = mx::compile([](const std::vector<Tensor> &args) {
            constexpr int group_size = 64;
            auto projection = mx::quantized_matmul(args[0], args[1], args[2], args[3], true,
                                                   group_size, 8, "affine");
            auto parts = mx::split(projection, 2, -1);
            return std::vector<Tensor>{mx::quantized_matmul(silu(parts[0]) * parts[1],
                args[4], args[5], args[6], true, group_size, 8, "affine")};
        });
    } else if (!runtime_lora_suffix_) {
        suffix_ = mx::compile([](const std::vector<Tensor> &args) {
            auto parts = mx::split(mx::matmul(args[0], mx::transpose(args[1])), 2, -1);
            return std::vector<Tensor>{mx::matmul(silu(parts[0]) * parts[1], mx::transpose(args[2]))};
        });
    }
    if (!full_weights_.empty()) {
        full_ffn_ = mx::compile([](const std::vector<Tensor> &args) {
            auto parts = mx::split(mx::matmul(args[0], mx::transpose(args[1])), 2, -1);
            return std::vector<Tensor>{mx::matmul(silu(parts[0]) * parts[1], mx::transpose(args[2]))};
        });
    }
}
Tensor HybridMLP::operator()(int block, const Tensor &input) {
    return run(block, input, nullptr);
}
Tensor HybridMLP::run(int block, const Tensor &input, BridgeTiming *timing) {
    require(block >= 0 && block < 32 && input.shape() == mx::Shape{1, ane_.rows, 4096},
            "Qwen21 hybrid decode shape/block mismatch");
    const char *profile_flag = std::getenv("TURBOCIDER_QWEN21_PROFILE_RUNTIME_LORA_FFN");
    const bool profile_fused = !timing && ane_.mlp_output_kind == "fused_lora" &&
        profile_flag && std::string_view(profile_flag) == "1";
    BridgeTiming profile_timing;
    if (profile_fused) timing = &profile_timing;
    if (gpu_full_blocks_[block]) {
        const auto &weights = full_weights_.at(block);
        auto output = full_ffn_({input, weights.first, weights.second})[0];
        require(output.dtype() == input.dtype(), "Qwen21 full-GPU fallback changed activation precision");
        return output;
    }
    auto packed = mx::contiguous(mx::astype(input, mx::float16));
    auto started = timing ? Clock::now() : Clock::time_point{};
    mx::eval(input, packed);
    if (timing) timing->input_ready += std::chrono::duration<double>(Clock::now() - started).count();
    started = timing ? Clock::now() : Clock::time_point{};
    Tensor gpu = input;
    if (runtime_lora_suffix_) {
        const auto stem = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
        const int width = ane_.ane_mlp_end;
        auto gate = weights_.project_slice(input, stem + "gate_up", width, 12288, 0, 4096);
        auto up = weights_.project_slice(input, stem + "gate_up",
                                         12288 + width, 24576, 0, 4096);
        gpu = weights_.project_slice(silu(gate) * up, stem + "out", 0, 4096, width, 12288);
    } else {
        gpu = gpu_w8a16_
            ? suffix_({input, fused_q_[block], fused_scales_[block], fused_biases_[block],
                       down_q_[block], down_scales_[block], down_biases_[block]})[0]
            : suffix_({input, fused_[block], down_[block]})[0];
    }
    if (gpu_w8a16_) gpu = mx::astype(gpu, input.dtype());
    // Dynamic LoRA gate/up must be ready before Core ML may execute SiLU.
    // Submit the independent GPU suffix *after* that barrier so its kernels
    // can overlap the Core ML call instead of being accidentally drained by
    // mx::eval(deltas) on MLX's default GPU stream.
    if (ane_.mlp_output_kind != "fused_lora") mx::async_eval(gpu);
    if (timing) timing->gpu_submit += std::chrono::duration<double>(Clock::now() - started).count();
    started = timing ? Clock::now() : Clock::time_point{};
    Tensor ane = packed;
    if (ane_.mlp_output_kind == "fused_lora") {
        const auto stem = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
        const int width = ane_.ane_mlp_end;
        auto deltas = runtime_lora_suffix_
            ? mx::contiguous(mx::astype(mx::concatenate({
                weights_.lora_delta_slice(input, stem + "gate_up", 0, width, 0, 4096),
                weights_.lora_delta_slice(input, stem + "gate_up", 12288, 12288 + width, 0, 4096)
              }, -1), mx::float16))
            : mx::contiguous(mx::zeros({1, ane_.rows, 2 * width}, mx::float16));
        const auto delta_wait_started = timing ? Clock::now() : Clock::time_point{};
        mx::eval(deltas);
        if (timing) timing->lora_delta_wait +=
            std::chrono::duration<double>(Clock::now() - delta_wait_started).count();
        mx::async_eval(gpu);
        ane = ane_.predict_with_lora(block, packed, deltas);
    } else {
        ane = ane_.predict(block, packed);
    }
    if (timing) timing->prediction_api += std::chrono::duration<double>(Clock::now() - started).count();
    Tensor result = gpu;
    if (ane_.mlp_output_kind == "gate_up") {
        const auto stem = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
        const int width = ane_.ane_mlp_end;
        auto gate = mx::astype(slice_axis(ane, -1, 0, width), input.dtype());
        auto up = mx::astype(slice_axis(ane, -1, width, 2 * width), input.dtype());
        if (runtime_lora_suffix_) {
            // The base graph ends before the nonlinearity. Both adapter
            // contributions must be added here; adding them to a completed
            // base FFN output cannot reproduce the student network.
            gate = gate + weights_.lora_delta_slice(input, stem + "gate_up", 0, width, 0, 4096);
            up = up + weights_.lora_delta_slice(input, stem + "gate_up", 12288, 12288 + width, 0, 4096);
        }
        auto prefix = weights_.project_slice(silu(gate) * up, stem + "out",
                                              0, 4096, 0, width);
        result = gpu + prefix;
    } else if (ane_.mlp_output_kind == "fused_lora") {
        const auto stem = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
        auto prefix = mx::astype(slice_axis(ane, -1, 0, 4096), input.dtype());
        auto hidden = mx::astype(slice_axis(ane, -1, 4096, ane_.output_channels), input.dtype());
        result = gpu + prefix;
        if (runtime_lora_suffix_)
            result = result + weights_.lora_delta_slice(hidden, stem + "out",
                                                         0, 4096, 0, ane_.ane_mlp_end);
    } else {
        result = merge_mlp_partitions(gpu, ane, ane_.output_scale);
    }
    require(result.dtype() == input.dtype(), "Qwen21 hybrid FFN changed activation precision");
    if (profile_fused)
        std::cerr << "{\"qwen21_runtime_lora_ffn_block\":" << block
                  << ",\"input_ready_seconds\":" << timing->input_ready
                  << ",\"gpu_graph_submit_seconds\":" << timing->gpu_submit
                  << ",\"lora_delta_wait_seconds\":" << timing->lora_delta_wait
                  << ",\"predict_and_submit_seconds\":" << timing->prediction_api
                  << "}" << std::endl;
    // Transformer materializes the residual update before invoking the next
    // callback. Let that single barrier also consume this shared Core ML
    // output; a second barrier here only splits the same dependency chain.
    return result;
}
Tensor HybridMLP::tiled_sequence(int block, const Tensor &input) {
    require(input.ndim() == 3 && input.shape(0) == 1 && input.shape(1) > ane_.rows &&
                input.shape(2) == 4096, "Qwen21 tiled FFN requires a long sequence");
    std::vector<Tensor> pieces;
    const char *profile_flag = std::getenv("TURBOCIDER_QWEN21_PROFILE_TILED_FFN");
    const bool profile = profile_flag && std::string_view(profile_flag) == "1";
    BridgeTiming timing;
    double merge_wait = 0;
    for (int64_t start = 0; start < input.shape(1); start += ane_.rows) {
        const int64_t length = std::min<int64_t>(ane_.rows, input.shape(1) - start);
        auto tile = slice_axis(input, 1, start, start + length);
        if (length < ane_.rows)
            tile = mx::pad(tile, {{0, 0}, {0, ane_.rows - length}, {0, 0}});
        auto result = run(block, tile, profile ? &timing : nullptr);
        // The compiled merge allocates a distinct MLX output. Materialize it
        // before Core ML reuses its borrowed output backing on the next tile;
        // an additional copy of the merged output would only add GPU traffic.
        const auto merge_started = profile ? Clock::now() : Clock::time_point{};
        mx::eval(result);
        if (profile) merge_wait += std::chrono::duration<double>(Clock::now() - merge_started).count();
        pieces.push_back(length == ane_.rows ? result : slice_axis(result, 1, 0, length));
    }
    if (profile)
        std::cerr << "{\"qwen21_tiled_ffn_block\":" << block
                  << ",\"tiles\":" << (input.shape(1) + ane_.rows - 1) / ane_.rows
                  << ",\"input_ready_seconds\":" << timing.input_ready
                  << ",\"gpu_submit_seconds\":" << timing.gpu_submit
                  << ",\"prediction_api_seconds\":" << timing.prediction_api
                  << ",\"merge_wait_seconds\":" << merge_wait << "}" << std::endl;
    return mx::concatenate(pieces, 1);
}
Tensor HybridMLP::tiled_target_with_prefix_tail(int block, const Tensor &target,
                                                const Tensor &prefix_tail) {
    require(target.shape() == mx::Shape{1, ane_.rows, 4096} &&
                prefix_tail.ndim() == 3 && prefix_tail.shape(0) == 1 &&
                prefix_tail.shape(1) > 0 && prefix_tail.shape(1) < ane_.rows &&
                prefix_tail.shape(2) == 4096 && prefix_tail.dtype() == target.dtype(),
            "Qwen21 repeated prefill FFN tile geometry/dtype mismatch");
    const int64_t tail = prefix_tail.shape(1);
    auto first = mx::concatenate({prefix_tail, slice_axis(target, 1, 0, ane_.rows - tail)}, 1);
    auto first_output = run(block, first, nullptr);
    mx::eval(first_output); // Core ML will reuse its shared output on the next tile
    auto last = mx::pad(slice_axis(target, 1, ane_.rows - tail, ane_.rows),
                        {{0, 0}, {0, ane_.rows - tail}, {0, 0}});
    auto last_output = run(block, last, nullptr);
    mx::eval(last_output);
    return mx::concatenate({slice_axis(first_output, 1, tail, ane_.rows),
                            slice_axis(last_output, 1, 0, tail)}, 1);
}
}
