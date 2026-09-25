#pragma once
#include "../../backends/coreml.hpp"

namespace tc::qwen21 {
// Experimental checkpoint-bound MLP channel split, explicitly opted into.
// CPU_AND_NE policy does not by itself establish actual device placement.
class HybridMLP {
  public:
    HybridMLP(const Weights &, HybridSession &, bool gpu_w8a16 = false);
    // The result borrows the session-wide Core ML output backing. Caller must
    // materialize its consumer before any subsequent prediction on ane_.
    // Transformer::forward enforces this at the residual-update boundary.
    Tensor operator()(int block, const Tensor &);
  private:
    HybridSession &ane_;
    bool gpu_w8a16_ = false;
    std::vector<Tensor> fused_, down_;
    std::vector<Tensor> fused_q_, fused_scales_, fused_biases_;
    std::vector<Tensor> down_q_, down_scales_, down_biases_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)> suffix_;
};
}
