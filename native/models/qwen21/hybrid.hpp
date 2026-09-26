#pragma once
#include "../../backends/coreml.hpp"

namespace tc::qwen21 {
// Experimental checkpoint-bound MLP channel split, explicitly opted into.
// CPU_AND_NE policy does not by itself establish actual device placement.
class HybridMLP {
  public:
    HybridMLP(const Weights &, HybridSession &);
    // The result borrows the session-wide Core ML output backing. Caller must
    // materialize its consumer before any subsequent prediction on ane_.
    // Transformer::forward enforces this at the residual-update boundary.
    Tensor operator()(int block, const Tensor &);
  private:
    HybridSession &ane_;
    std::vector<Tensor> fused_, down_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)> suffix_;
};
}
