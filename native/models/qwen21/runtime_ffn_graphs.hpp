#pragma once

#include "../../backends/mlx.hpp"

namespace tc::qwen21::runtime_ffn {
using Function = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;

// Request/calibration-local factories. Capture the SAME resident Weights;
// never retain these closures after an adapter rebind or use a global cache.
// Calibration and inference share the graph bodies and rounding boundaries,
// not just a mathematically similar eager implementation.
inline Function full(const Weights &weights, const std::string &prefix) {
    return mx::compile([&weights, prefix](const std::vector<Tensor> &a) {
        auto halves = mx::split(weights.project(a[0], prefix+"gate_up"), 2, -1);
        return std::vector<Tensor>{weights.project(silu(halves[0])*halves[1], prefix+"out")};
    });
}

inline Function corrections(const Weights &weights, const std::string &prefix,
                            int first, int count, int hidden = 4096, int width = 12288) {
    require(first >= 0 && count > 0 && first <= width-count && hidden > 0,
            "invalid Qwen FFN correction graph range");
    return mx::compile([&weights, prefix, first, count, hidden, width](const std::vector<Tensor> &a) {
        // Logical halves retain separate adapter intersections and the existing
        // FP32 accumulation/final BF16 delta rounding, including stacked LoRA.
        auto gate = weights.lora_delta_slice(a[0], prefix+"gate_up", first, first+count, 0, hidden);
        auto up = weights.lora_delta_slice(a[0], prefix+"gate_up", width+first, width+first+count, 0, hidden);
        return std::vector<Tensor>{mx::contiguous(gate), mx::contiguous(up)};
    });
}

inline Function channels(const Weights &weights, const std::string &prefix,
                         int first, int count, int hidden = 4096, int width = 12288,
                         bool fp32_partial = false) {
    require(first >= 0 && count > 0 && first <= width-count && hidden > 0,
            "invalid Qwen FFN channel graph range");
    return mx::compile([&weights, prefix, first, count, hidden, width, fp32_partial](const std::vector<Tensor> &a) {
        auto gate = weights.project_slice(a[0], prefix+"gate_up", first, first+count, 0, hidden, false);
        auto up = weights.project_slice(a[0], prefix+"gate_up", width+first, width+first+count, 0, hidden, false);
        auto intermediate = silu(gate)*up;
        auto base = fp32_partial ? weights.project_base_slice_fp32(intermediate,prefix+"out",0,hidden,first,first+count) :
            weights.project_base_slice(intermediate, prefix+"out", 0, hidden, first, first+count, false);
        // Down-LoRA is NOT rounded once per shard. Apply it to joined hidden.
        return std::vector<Tensor>{base, intermediate};
    });
}

inline Function down_add(const Weights &weights, const std::string &prefix,
                         int hidden = 4096, int width = 12288) {
    require(hidden > 0 && width > 0, "invalid Qwen FFN down graph geometry");
    return mx::compile([&weights, prefix, hidden, width](const std::vector<Tensor> &a) {
        auto delta = weights.lora_delta_slice(a[0], prefix+"out", 0, hidden, 0, width);
        return std::vector<Tensor>{mx::astype(mx::astype(a[1], mx::float32)+mx::astype(delta, mx::float32), a[1].dtype())};
    });
}
} // namespace tc::qwen21::runtime_ffn
