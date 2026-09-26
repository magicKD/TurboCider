#pragma once
#include "../../backends/mlx.hpp"

namespace tc::qwen21::pe {
struct DeltaResult {
    Tensor output; // [batch, time, heads, value_dim], query dtype
    Tensor state;  // [batch, heads, key_dim, value_dim], FP32
};

// Qwen3.5 gated-delta recurrent primitive. Q/K heads must already be repeated
// to the value-head count. g is log decay; beta is the sigmoid update gate.
// This correctness/decode implementation is not a parallel prefill kernel.
DeltaResult recurrent_delta(const Tensor &query, const Tensor &key,
    const Tensor &value, const Tensor &g, const Tensor &beta,
    const std::optional<Tensor> &initial_state, bool normalize_qk,
    std::atomic<bool> &cancelled);

// Parallel intra-chunk unit-triangular solve and a scan over chunk states.
// FP32 accumulation/state, original Q dtype for output; supports padded tails.
DeltaResult chunked_delta(const Tensor &query, const Tensor &key,
    const Tensor &value, const Tensor &g, const Tensor &beta,
    const std::optional<Tensor> &initial_state, bool normalize_qk,
    std::atomic<bool> &cancelled, int chunk_size = 64);
}
