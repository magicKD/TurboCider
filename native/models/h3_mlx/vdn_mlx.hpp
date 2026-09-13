#pragma once

#include "checkpoint.hpp"
#include "geometry.hpp"
#include "vdn.hpp"

namespace tc::h3_mlx {

struct VDNForwardOutput {
    // The gated/windowed softmax output, still in [sequence, heads, head_dim].
    Tensor softmax;
    // The linear branch readout, in [sequence, heads * head_dim].  It is zero
    // for text/audio and for the two anchor frames.
    Tensor linear_readout;
};

// FP32 inverse of I + A for the small per-frame/head VDN systems. The
// explicit backend switch is public so a deterministic diagnostic can compare
// the custom Metal kernel against MLX's CPU linalg implementation without
// mutating process environment between lazy graph constructions.
Tensor vdn_spd_inverse(const Tensor &a, bool cpu_reference);

// Exact VDN span-mask attention exposed for the native GPU qualification
// probe. `reference=true` selects the former take+MLX-SDPA graph; false uses
// the fused register-resident online-softmax Metal kernel.
Tensor vdn_window_softmax_for_test(const Tensor &q, const Tensor &k,
                                   const Tensor &v,
                                   const PackedLayout &layout,
                                   int chunk, int radius,
                                   bool reference);
Tensor vdn_window_softmax_mma_for_test(const Tensor &q, const Tensor &k,
                                       const Tensor &v,
                                       const PackedLayout &layout,
                                       int chunk, int radius);

VDNForwardOutput vdn_forward(const Checkpoint &, const Tensor &x,
                             const Tensor &q_raw, const Tensor &k_raw,
                             const Tensor &v_raw, const Tensor &q_softmax,
                             const Tensor &k_softmax, const Tensor &v_softmax,
                             const PackedLayout &, int block_index);

} // namespace tc::h3_mlx
