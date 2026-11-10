#pragma once

#include "../../backends/mlx.hpp"

#include <limits>

namespace tc::wan22 {

// Apply interleaved RoPE to Q/K and pack Q/K/V into head-major storage in
// one pass. FP32 multiplication boundaries match the existing eager path.
inline std::vector<Tensor> fused_rope_heads(const Tensor &q, const Tensor &k, const Tensor &v,
                                            const Tensor &cosine, const Tensor &sine, int heads) {
    require(q.ndim() == 3 && q.shape(0) == 1 && q.shape(1) > 0 && q.shape(2) > 0 &&
                q.shape() == k.shape() && q.shape() == v.shape() && q.dtype() == mx::bfloat16 &&
                k.dtype() == q.dtype() && v.dtype() == q.dtype() && heads > 0 && q.shape(2) % heads == 0,
            "invalid Wan2.2 fused RoPE Q/K/V geometry");
    const int rows = q.shape(1), dim = q.shape(2) / heads;
    require(dim % 2 == 0 && cosine.shape() == mx::Shape{rows, dim} && sine.shape() == cosine.shape() &&
                cosine.dtype() == mx::float32 && sine.dtype() == mx::float32,
            "invalid Wan2.2 fused RoPE frequency geometry");
    const size_t pairs = size_t(rows) * heads * (dim / 2);
    require(pairs <= size_t(std::numeric_limits<int>::max()), "Wan2.2 fused RoPE grid exceeds supported bounds");
    static auto kernel = mx::fast::metal_kernel(
        "tc_wan22_rope_qkv_heads_v1", {"q", "k", "v", "cosine", "sine"}, {"qo", "ko", "vo"}, R"metal(
#pragma clang fp contract(off)
        uint pair = thread_position_in_grid.x;
        if (pair >= PAIRS) return;
        uint d = (pair % (D / 2)) * 2;
        uint temp = pair / (D / 2);
        uint token = temp % N;
        uint head = temp / N;
        uint src = (token * H + head) * D + d;
        uint dst = (head * N + token) * D + d;
        float c0 = cosine[token * D + d], c1 = cosine[token * D + d + 1];
        float s0 = sine[token * D + d], s1 = sine[token * D + d + 1];
        float qa = float(q[src]), qb = float(q[src + 1]);
        float ka = float(k[src]), kb = float(k[src + 1]);
        qo[dst] = T(qa * c0 - qb * s0);
        qo[dst + 1] = T(qb * c1 + qa * s1);
        ko[dst] = T(ka * c0 - kb * s0);
        ko[dst + 1] = T(kb * c1 + ka * s1);
        vo[dst] = v[src];
        vo[dst + 1] = v[src + 1];
        )metal");
    const mx::Shape output_shape{1, heads, rows, dim};
    return kernel({q, k, v, cosine, sine}, {output_shape, output_shape, output_shape},
                  {q.dtype(), q.dtype(), q.dtype()}, {int(pairs), 1, 1}, {256, 1, 1},
                  {{"T", q.dtype()}, {"N", rows}, {"H", heads}, {"D", dim}, {"PAIRS", int(pairs)}}, {}, false, {});
}

} // namespace tc::wan22
