#pragma once
#include "pe_delta.hpp"
#include <array>

namespace tc::qwen21::pe {
struct LanguageConfig {
    int hidden = 4096, layers = 32, heads = 16, kv_heads = 4, head_dim = 256;
    int rotary_dim = 64;
    int linear_key_heads = 16, linear_value_heads = 32;
    int linear_key_dim = 128, linear_value_dim = 128, conv_kernel = 4;
    float epsilon = 1e-6f, theta = 10000000.f;
    std::array<int, 3> mrope_sections{11, 11, 10};
    std::vector<bool> full_attention; // empty means every fourth layer
    bool chunked_prefill = true; // recurrent diagnostic fallback remains available
};

// Stateful single-sequence Qwen3.5 language backbone. Call reset between
// independent prompts. Caller owns the weights and serializes access.
// Optional multi-token chunked prefill; single-token calls stay recurrent.
class LanguageModel {
  public:
    LanguageModel(const Weights &, LanguageConfig = {});
    void reset();
    int cached_tokens() const { return tokens_; }
    Tensor forward_ids(const Tensor &, const Event &, std::atomic<bool> &);
    // positions is [3,sequence], allowing future vision mRoPE integration.
    Tensor forward_embeddings(const Tensor &, const Tensor &positions,
                              const Event &, std::atomic<bool> &);
  private:
    struct State { std::optional<Tensor> k, v, convolution, recurrent; };
    const Weights &weights_;
    LanguageConfig config_;
    std::string prefix_;
    int tokens_ = 0;
    std::vector<State> states_;
    Tensor linear_attention(const Tensor &, int, std::atomic<bool> &);
    Tensor full_attention(const Tensor &, int, const Tensor &, const Tensor &);
};
}
