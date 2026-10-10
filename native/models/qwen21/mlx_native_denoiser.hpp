#pragma once

#include "native_denoiser.hpp"
#include "../../backends/mlx.hpp"

namespace tc::qwen21::native_graph {
// Boundary only. Requires produced immutable BF16 source arrays, with their
// original MLX Data owners retained. No eval, quantization, dense copy or model
// loading here. The Session/source lease must additionally prove base identity
// and external queue readiness; this helper cannot inspect merged adapters.
DenoiserWeights import_base_denoiser_weights(metal_graph::Backend&, const Weights&,
                                            DenoiserGeometry = {});
// Explicit runtime-adapter import. Every bound term must be consumed by a
// supported model projection, come from a verified held SourceLease and have
// produced BF16/FP32 compact factors. Requires caller queue synchronization;
// never evaluates/casts/merges factors or silently imports pathname adapters.
// Adapter identity/schedule admission remains the enclosing Session's duty.
DenoiserWeights import_runtime_lora_denoiser_weights(metal_graph::Backend&,
                                                    const Weights&,
                                                    DenoiserGeometry = {});
} // namespace tc::qwen21::native_graph
