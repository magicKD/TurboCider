#pragma once
#include "../../backends/mlx.hpp"
#include "../../core/tokenizer.hpp"
#include <array>

namespace tc::qwen21 {
struct TextConfig {
    int layers = 36;
    int heads = 32;
    int kv_heads = 8;
    int head_dim = 128;
    float theta = 5000000.f;
    float epsilon = 1e-6f;
    std::array<int, 3> mrope_sections{24, 20, 20};
    bool final_norm = true; // mflux T2I; edit callers can explicitly select the raw final hidden state
};

// Accept HF model.language_model.* and Comfy model.* language weights. The visual tower is
// separate; encode_embeddings also accepts image-token embeddings and 3-axis
// positions so it can be reused by the multimodal conditioning pipeline.
class TextEncoder {
  public:
    TextEncoder(const Weights &, TextConfig = {});
    Tensor encode(const Tokens &, const Event &, std::atomic<bool> &) const;
    Tensor encode_embeddings(const Tensor &, const Tensor &positions,
                             int valid_tokens, const Event &, std::atomic<bool> &,
                             const std::vector<Tensor> &deepstack_deltas = {}) const;
    static std::string prompt_template(const std::string &);
    static std::string system_prefix();

  private:
    const Weights &weights_;
    TextConfig config_;
    std::string language_prefix_;
};
} // namespace tc::qwen21
