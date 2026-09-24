#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::z_metal {
namespace mx = mlx::core;

// One SIMD group per 128-dimensional head. Read packed token-major QKV
// directly and emit the head-major layout consumed by SDPA. The sequence
// length is a template constant, so all output addressing is shape-specialized.
// BF16/FP16 rounding between RMSNorm and RoPE is part of the model contract.
inline std::vector<mx::array> prepare_qkv(const mx::array &qkv,
                                         const mx::array &qw,
                                         const mx::array &kw,
                                         const mx::array &freqs) {
    if (qkv.ndim() != 3 || qkv.shape(0) != 1 || qkv.shape(2) != 11520 ||
        qw.shape() != mx::Shape{128} || kw.shape() != mx::Shape{128} ||
        freqs.shape() != mx::Shape{qkv.shape(1), 64, 2} ||
        (qkv.dtype() != mx::bfloat16 && qkv.dtype() != mx::float16 &&
         qkv.dtype() != mx::float32))
        throw std::invalid_argument("Z-Image Metal QKV geometry/dtype mismatch");
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_packed_qkv_norm_rope", {"qkv", "qw", "kw", "freqs"},
        {"q", "k", "v"}, R"metal(
        uint lane = thread_index_in_simdgroup;
        uint task = thread_position_in_grid.x / 32;
        if (task >= N * 30) return;
        uint token = task / 30;
        uint head = task % 30;
        uint src = token * 11520 + head * 128;
        uint dst = (head * N + token) * 128;
        float qx[4], kx[4];
        float qs = 0.0f, ks = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            uint d = lane * 4 + i;
            qx[i] = float(qkv[src + d]);
            kx[i] = float(qkv[src + 3840 + d]);
            qs += qx[i] * qx[i];
            ks += kx[i] * kx[i];
            v[dst + d] = qkv[src + 7680 + d];
        }
        float qi = metal::precise::rsqrt(simd_sum(qs) / 128.0f + 1e-5f);
        float ki = metal::precise::rsqrt(simd_sum(ks) / 128.0f + 1e-5f);
        for (uint i = 0; i < 4; ++i) {
            uint d = lane * 4 + i;
            qx[i] = float(T((qx[i] * qi) * float(qw[d])));
            kx[i] = float(T((kx[i] * ki) * float(kw[d])));
        }
        for (uint i = 0; i < 4; i += 2) {
            uint d = lane * 4 + i;
            float c = float(freqs[token * 128 + d]);
            float s = float(freqs[token * 128 + d + 1]);
            q[dst + d] = T(qx[i] * c - qx[i+1] * s);
            q[dst + d + 1] = T(qx[i] * s + qx[i+1] * c);
            k[dst + d] = T(kx[i] * c - kx[i+1] * s);
            k[dst + d + 1] = T(kx[i] * s + kx[i+1] * c);
        }
        )metal");
    const int n = qkv.shape(1);
    const mx::Shape shape{1, 30, n, 128};
    return kernel({qkv, qw, kw, freqs}, {shape, shape, shape},
                  {qkv.dtype(), qkv.dtype(), qkv.dtype()},
                  {n * 30 * 32, 1, 1}, {128, 1, 1},
                  {{"T", qkv.dtype()}, {"N", n}}, {}, false, {});
}
} // namespace tc::z_metal
