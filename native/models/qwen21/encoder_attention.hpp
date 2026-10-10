#pragma once
#include "../../backends/mlx.hpp"
#include <cmath>
#include <cstdlib>

namespace tc::qwen21::encoder_attention {
inline bool default_enabled() {
    // Process-fixed diagnostic reference, not a request option. Freezing it
    // prevents a Session from silently reusing conditioning across recipes.
    static const bool enabled = [] {
        const char *value = std::getenv("TURBOCIDER_QWEN21_ENCODER_REFERENCE");
        require(!value || std::string_view(value) == "0" || std::string_view(value) == "1",
                "Qwen21 encoder reference diagnostic must be 0 or 1");
        return !value || std::string_view(value) == "0";
    }();
    return enabled;
}
struct LanguagePlan {
    bool grouped_kv = false;
    bool causal_without_mask = false;
};
inline LanguagePlan language_plan(int rows, int valid, bool efficient) {
    require(rows > 0 && valid > 0 && valid <= rows, "invalid encoder attention rows");
    // Preserve the small-row expansion policy used by the common Qwen3
    // encoder. The edit prefill contains hundreds/thousands of image tokens.
    return {efficient && rows >= 128, efficient && valid == rows};
}

inline Tensor vision(const Tensor &q, const Tensor &k, const Tensor &v, bool efficient) {
    // Qwen3-VL's 1152/16=72 head width is not a fused SDPA width in pinned
    // MLX. Zero-pad only the feature axis, keep the ORIGINAL 1/sqrt(72)
    // scale and slice output before merging heads. No token/pixel reduction.
    require(q.ndim() == 4 && q.shape() == k.shape() && q.shape() == v.shape(),
            "invalid Qwen21 visual attention geometry");
    // The unfused BF16 path has intermediate rounding boundaries. Keep it
    // unchanged; Qwen21's visual processor/positions/RoPE use FP32 here.
    if (!efficient || q.shape(-1) != 72 || q.shape(2) <= 8 ||
        q.dtype() != mx::float32 || k.dtype() != mx::float32 || v.dtype() != mx::float32)
        return attend(q, k, v);
    auto pad = [](const Tensor &x) {
        return mx::pad(x, std::vector<int>{3}, mx::Shape{0}, mx::Shape{8}, Tensor(0, x.dtype()));
    };
    auto output = mx::fast::scaled_dot_product_attention(pad(q), pad(k), pad(v),
                                                        1.f / std::sqrt(72.f));
    output = slice_axis(output, -1, 0, 72);
    return mx::reshape(mx::transpose(output, {0, 2, 1, 3}),
                       {q.shape(0), q.shape(2), q.shape(1) * 72});
}
} // namespace tc::qwen21::encoder_attention
