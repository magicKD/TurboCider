#pragma once
#include "../../backends/coreml.hpp"
#include <array>
#include <map>
#include <optional>

namespace tc::qwen21 {
// Experimental checkpoint-bound MLP channel split, explicitly opted into.
// CPU_AND_NE policy does not by itself establish actual device placement.
class HybridMLP {
  public:
    HybridMLP(const Weights &, HybridSession &, bool gpu_w8a16 = false,
              const std::vector<int> &gpu_full_blocks = {}, bool layer_staged = false);
    // The result borrows the session-wide Core ML output backing. Caller must
    // materialize its consumer before any subsequent prediction on ane_.
    // Transformer::forward enforces this at the residual-update boundary.
    Tensor operator()(int block, const Tensor &);
    // Reuse a fixed-row graph over successive FFN sequence tiles, for either
    // prefill or diagnostic long decode. Pad and trim the final partial tile.
    Tensor tiled_sequence(int block, const Tensor &);
    // A resident prefix-KV hit still reproduces the first-step W8A8 tile
    // boundary: prepend the cached unfinished prefix FFN input to the new
    // target rows before invoking the same 1024-row model.
    Tensor tiled_target_with_prefix_tail(int block, const Tensor &target,
                                         const Tensor &prefix_tail);
  private:
    struct BridgeTiming {
        double input_ready = 0, gpu_submit = 0, prediction_api = 0,
            lora_delta_wait = 0;
    };
    Tensor run(int block, const Tensor &input, BridgeTiming *timing);
    const Weights &weights_;
    HybridSession &ane_;
    bool gpu_w8a16_ = false;
    bool runtime_lora_suffix_ = false;
    bool layer_staged_ = false;
    // Populate only while visiting each decode layer, after materializing
    // compact slices. Original checkpoint tensors are released that layer.
    // 6144-channel W8A8 leaves exactly 4.5 GiB of BF16 suffixes across 32 layers.
    std::array<std::optional<std::pair<Tensor, Tensor>>, 32> staged_suffix_;
    std::array<bool, 32> gpu_full_blocks_{};
    std::map<int, std::pair<Tensor, Tensor>> full_weights_;
    std::vector<Tensor> fused_, down_;
    std::vector<Tensor> fused_q_, fused_scales_, fused_biases_;
    std::vector<Tensor> down_q_, down_scales_, down_biases_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)> suffix_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)> full_ffn_;
};
}
