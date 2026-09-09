#pragma once

#include "../../backends/mlx.hpp"
#include "../../core/tokenizer.hpp"

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

Tensor qwen3_conditioning(const Tokens &, const Weights &, const Qwen3Conditioning &,
                          const Event &, std::atomic<bool> &);

} // namespace tc::components
