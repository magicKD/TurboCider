#pragma once

#include "../../backends/mlx.hpp"
#include "../../core/tokenizer.hpp"

namespace tc {
class HybridSession;
}

namespace tc::components {

// The same Qwen3 backbone serves different diffusion conditioning contracts.
// Hidden-state taps and accumulation precision belong to the consumer, not
// to checkpoint discovery. Do not share prompt outputs across these policies.
struct Qwen3Conditioning {
    int heads = 32;
    int kv_heads = 8;
    int head_dim = 128;
    float rope_theta = 1000000.f;
    float norm_epsilon = 1e-6f;
    std::vector<int> output_layers;
    bool float32_residual = false;
    std::string progress_phase;

    static Qwen3Conditioning flux_klein();
    static Qwen3Conditioning z_image();
};

// The Core ML branch is a fixed-shape token-wise MLP.  Keep its padding
// policy explicit so short prompts can fail closed to the exact GPU path
// instead of paying for a much wider synthetic tail.
struct Qwen3PrefillPlan {
    int actual_tokens = 0;
    int selected_bucket = 0;
    int compute_tokens = 0;
    int padding_tokens = 0;
    int minimum_profitable_rows = 0;
    bool use_hybrid = false;
    bool fixed_shape = false;
    std::string reason;
};

Qwen3PrefillPlan qwen3_prefill_plan(const std::filesystem::path &manifest,
                                    int actual_tokens);
Qwen3PrefillPlan qwen3_prefill_plan_rows(int actual_tokens, int bucket,
                                         int minimum_profitable_rows = 0);

Tensor qwen3_conditioning(const Tokens &, const Weights &, const Qwen3Conditioning &,
                          const Event &, std::atomic<bool> &,
                          HybridSession *hybrid = nullptr);

// Resolve the immutable safetensors file/index used for Core ML provenance
// when a text encoder is represented by a directory.
std::filesystem::path qwen3_checkpoint_path(const std::filesystem::path &);
bool qwen3_hybrid_profitable(int tokens);

} // namespace tc::components
