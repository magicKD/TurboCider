#pragma once
#include "../../backends/mlx.hpp"

namespace tc::qwen21 {
struct VisionConfig {
    int patch = 16, temporal = 2, channels = 3;
    int hidden = 1152, heads = 16, layers = 27, merge = 2, position_side = 48;
    std::vector<int> deepstack_layers{8, 16, 24};
    // PE Qwen3.5 uses Transformers activation precision and exact merger
    // GELU, with no DeepStack. Default preserves the Qwen21/mflux path.
    bool pe_qwen35 = false;
    // Experimental PE diagnostic: keep the complete visual tower in FP32
    // while retaining the checkpoint's original weights. This option does
    // not enable production PE routing or imply BF16/quality acceptance.
    bool pe_fp32 = false;
    // Experimental reduction diagnostic; never selected by production routing.
    bool pe_compensated_projection = false;
};
struct VisionFeatures {
    Tensor merged;
    std::vector<Tensor> deepstack;
};
class VisionEncoder {
  public:
    VisionEncoder(const Weights &, VisionConfig = {});
    // Single still-image patch rows in processor order (2x2 merge groups).
    VisionFeatures encode(const Tensor &patches, int grid_height, int grid_width,
                          const Event &, std::atomic<bool> &,
                          std::unordered_map<std::string, Tensor> *trace = nullptr) const;
  private:
    const Weights &weights_;
    VisionConfig config_;
    Tensor norm(const Tensor &, const std::string &) const;
    Tensor project(const Tensor &, const std::string &) const;
    Tensor merge(const Tensor &, const std::string &, bool postshuffle) const;
};
} // namespace tc::qwen21
