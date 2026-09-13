#pragma once

// The panelled Cholesky and triangular-inverse kernels below follow the
// Apache-2.0 OpenVDN reference algorithm. This port is self-contained and
// uses MLX's runtime custom-kernel interface.

namespace tc::h3_mlx {

inline constexpr const char *kVDNMetalHeader = R"METAL(
#include <metal_stdlib>
using namespace metal;

#define VDN_NB 32
#define VDN_D_MAX 128
#define VDN_THREADS 128
)METAL";

// L = chol(I + A), one threadgroup per flattened (frame, head) matrix.
inline constexpr const char *kVDNCholeskyBody = R"METAL(
  threadgroup float panel[VDN_D_MAX * VDN_NB];
  threadgroup float lrow[VDN_NB * (VDN_D_MAX - VDN_NB)];
  const uint batch = threadgroup_position_in_grid.x;
  const int tid = (int)thread_position_in_threadgroup.x;
  const device float *Ab = A + (uint)batch * d * d;
  device float *Lb = L + (uint)batch * d * d;

  for (int i = tid; i < d * d; i += VDN_THREADS) Lb[i] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  for (int c0 = 0; c0 < d; c0 += VDN_NB) {
    const int nb = min(VDN_NB, d - c0);
    for (int r = c0 + tid; r < d; r += VDN_THREADS) {
      for (int c = 0; c < nb; ++c) {
        const int col = c0 + c;
        float v = (r >= col) ? Ab[(uint)r * d + col] : 0.0f;
        if (r == col) v += 1.0f;
        panel[(uint)r * VDN_NB + c] = v;
      }
    }
    for (int i = tid; i < nb * c0; i += VDN_THREADS)
      lrow[i] = Lb[(uint)(c0 + i / c0) * d + (i % c0)];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = c0 + tid; r < d; r += VDN_THREADS) {
      float acc[VDN_NB];
      for (int c = 0; c < VDN_NB; ++c)
        acc[c] = (c < nb) ? panel[(uint)r * VDN_NB + c] : 0.0f;
      for (int k = 0; k < c0; ++k) {
        const float lrk = Lb[(uint)r * d + k];
        for (int c = 0; c < VDN_NB; ++c)
          acc[c] -= lrk * lrow[(uint)c * c0 + k];
      }
      for (int c = 0; c < nb; ++c) panel[(uint)r * VDN_NB + c] = acc[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = 0; c < nb; ++c) {
      if (tid == 0) {
        float sum = panel[(uint)(c0 + c) * VDN_NB + c];
        for (int k = 0; k < c; ++k) {
          const float v = panel[(uint)(c0 + c) * VDN_NB + k];
          sum -= v * v;
        }
        // I + A is SPD. Clamp only a malformed/rounding-induced failure so
        // one bad matrix cannot poison the whole denoise graph with NaNs.
        panel[(uint)(c0 + c) * VDN_NB + c] = sqrt(max(sum, 1e-12f));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float diag = panel[(uint)(c0 + c) * VDN_NB + c];
      for (int r = c0 + c + 1 + tid; r < d; r += VDN_THREADS) {
        float sum = panel[(uint)r * VDN_NB + c];
        for (int k = 0; k < c; ++k)
          sum -= panel[(uint)r * VDN_NB + k] *
                 panel[(uint)(c0 + c) * VDN_NB + k];
        panel[(uint)r * VDN_NB + c] = sum / diag;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = c0 + tid; r < d; r += VDN_THREADS)
      for (int c = 0; c < nb; ++c)
        if (r >= c0 + c) Lb[(uint)r * d + (c0 + c)] =
            panel[(uint)r * VDN_NB + c];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  }
)METAL";

// X = L^-1, one threadgroup per (matrix, block-column).
inline constexpr const char *kVDNTrinvBody = R"METAL(
  threadgroup float xcol[VDN_D_MAX * VDN_NB];
  threadgroup float ltile[VDN_NB * VDN_NB];
  threadgroup float acc[VDN_NB * VDN_NB];
  const int tid = (int)thread_position_in_threadgroup.x;
  const int j = (int)threadgroup_position_in_grid.x;
  const uint batch = threadgroup_position_in_grid.y;
  const int c0 = j * VDN_NB;
  if (c0 >= d) return;
  const int nb = min(VDN_NB, d - c0);
  const device float *Lb = L + (uint)batch * d * d;
  device float *Xb = X + (uint)batch * d * d;
  for (int r = tid; r < c0; r += VDN_THREADS)
    for (int c = 0; c < nb; ++c) Xb[(uint)r * d + c0 + c] = 0.0f;
  for (int i = tid; i < VDN_D_MAX * VDN_NB; i += VDN_THREADS)
    xcol[i] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tid; i < nb * nb; i += VDN_THREADS)
    ltile[(uint)(i / nb) * VDN_NB + i % nb] =
        Lb[(uint)(c0 + i / nb) * d + c0 + (i % nb)];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < nb; ++i) {
    if (tid < nb) {
      float s = (i == tid) ? 1.0f : 0.0f;
      for (int k = 0; k < i; ++k)
        s -= ltile[(uint)i * VDN_NB + k] *
             xcol[(uint)(c0 + k) * VDN_NB + tid];
      xcol[(uint)(c0 + i) * VDN_NB + tid] =
          s / ltile[(uint)i * VDN_NB + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int r0 = c0 + nb; r0 < d; r0 += VDN_NB) {
    const int mb = min(VDN_NB, d - r0);
    for (int i = tid; i < mb * nb; i += VDN_THREADS) acc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = c0; k0 < r0; k0 += VDN_NB) {
      const int kb = min(VDN_NB, d - k0);
      for (int i = tid; i < mb * kb; i += VDN_THREADS)
        ltile[(uint)(i / kb) * VDN_NB + i % kb] =
            Lb[(uint)(r0 + i / kb) * d + k0 + i % kb];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = tid; i < mb * nb; i += VDN_THREADS) {
        const int r = i / nb, c = i % nb;
        float s = 0.0f;
        for (int k = 0; k < kb; ++k)
          s += ltile[(uint)r * VDN_NB + k] *
               xcol[(uint)(k0 + k) * VDN_NB + c];
        acc[i] += s;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = tid; i < mb * mb; i += VDN_THREADS)
      ltile[(uint)(i / mb) * VDN_NB + i % mb] =
          Lb[(uint)(r0 + i / mb) * d + r0 + i % mb];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < mb; ++i) {
      if (tid < nb) {
        float s = -acc[(uint)i * nb + tid];
        for (int k = 0; k < i; ++k)
          s -= ltile[(uint)i * VDN_NB + k] *
               xcol[(uint)(r0 + k) * VDN_NB + tid];
        xcol[(uint)(r0 + i) * VDN_NB + tid] =
            s / ltile[(uint)i * VDN_NB + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (int r = c0 + tid; r < d; r += VDN_THREADS)
    for (int c = 0; c < nb; ++c)
      Xb[(uint)r * d + c0 + c] = xcol[(uint)r * VDN_NB + c];
)METAL";

// inv = Linv^T Linv and transition = diag(alpha) inv.
inline constexpr const char *kVDNInvTransitionBody = R"METAL(
  const uint batch = threadgroup_position_in_grid.z;
  const int i0 = (int)threadgroup_position_in_grid.x * VDN_NB;
  const int j0 = (int)threadgroup_position_in_grid.y * VDN_NB;
  if (i0 >= d || j0 >= d) return;
  const device float *Lb = Linv + (uint)batch * d * d;
  const device float *ab = alpha + (uint)batch * d;
  device float *ib = inv + (uint)batch * d * d;
  device float *tb = trans + (uint)batch * d * d;
  const int tx = (int)thread_position_in_threadgroup.x;
  const int ty = (int)thread_position_in_threadgroup.y;
  float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  const int ii[2] = {i0 + ty, i0 + ty + 16};
  const int jj[2] = {j0 + tx, j0 + tx + 16};
  const int kstart = max(i0, j0);
  for (int k = kstart; k < d; ++k) {
    const float la = (k >= ii[0]) ? Lb[(uint)k * d + ii[0]] : 0.0f;
    const float lb = (k >= ii[1] && ii[1] < d) ?
        Lb[(uint)k * d + ii[1]] : 0.0f;
    const float ra = (k >= jj[0]) ? Lb[(uint)k * d + jj[0]] : 0.0f;
    const float rb = (k >= jj[1] && jj[1] < d) ?
        Lb[(uint)k * d + jj[1]] : 0.0f;
    sum[0][0] += la * ra; sum[0][1] += la * rb;
    sum[1][0] += lb * ra; sum[1][1] += lb * rb;
  }
  for (int a = 0; a < 2; ++a) {
    if (ii[a] >= d) continue;
    const float scale = ab[ii[a]];
    for (int b = 0; b < 2; ++b) {
      if (jj[b] >= d) continue;
      const uint off = (uint)ii[a] * d + jj[b];
      ib[off] = sum[a][b];
      tb[off] = scale * sum[a][b];
    }
  }
)METAL";

} // namespace tc::h3_mlx
