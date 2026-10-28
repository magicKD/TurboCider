// Small teaching kernels, independently implemented for this wiki.
// Correctness/tail controls first; these are not tuned production kernels.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Params { uint m, n, k, count; float scale, bias; uint flags, steps; };

kernel void vector_scalar(device const float* x [[buffer(0)]], device const float* r [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) out[i] = x[i] + r[i];
}
kernel void vector4(device const float* x [[buffer(0)]], device const float* r [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    const uint i = tid * 4;
    if (i + 3 < p.count) {
        const float4 a = *reinterpret_cast<device const float4*>(x + i);
        const float4 b = *reinterpret_cast<device const float4*>(r + i);
        *reinterpret_cast<device float4*>(out + i) = a + b;
    } else for (uint col = i; col < p.count; ++col) out[col] = x[col] + r[col];
}
kernel void scale_bias(device const float* x [[buffer(0)]], device const float* unused [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) out[i] = x[i] * p.scale + p.bias;
}
kernel void silu_only(device const float* x [[buffer(0)]], device const float* unused [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) out[i] = x[i] / (1.f + exp(-x[i]));
}
kernel void fused_epilogue(device const float* x [[buffer(0)]], device const float* r [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        float value = x[i] * p.scale + p.bias;
        out[i] = value / (1.f + exp(-value)) + r[i];
    }
}
kernel void gemm_scalar(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.m * p.n) return;
    const uint row = i / p.n, col = i % p.n;
    float total = 0;
    for (uint k = 0; k < p.k; ++k) total += a[row * p.k + k] * b[k * p.n + col];
    out[i] = total;
}
kernel void gemm_sg8(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float left[64], right[64], result[64];
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint base = 0; base < p.k; base += 8) {
        for (uint index = lane; index < 64; index += 32) {
            uint row = tile.y * 8 + index / 8, col = base + index % 8;
            left[index] = row < p.m && col < p.k ? a[row * p.k + col] : 0.f;
            row = base + index / 8; col = tile.x * 8 + index % 8;
            right[index] = row < p.k && col < p.n ? b[row * p.n + col] : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 l, r;
        simdgroup_load(l, left, 8); simdgroup_load(r, right, 8);
        simdgroup_multiply_accumulate(acc, l, r, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, result, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = lane; index < 64; index += 32) {
        const uint row = tile.y * 8 + index / 8, col = tile.x * 8 + index % 8;
        if (row < p.m && col < p.n) out[row * p.n + col] = result[index];
    }
}
kernel void rms_only(device const float* x [[buffer(0)]], device const float* unused [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.m) return;
    float energy = 0;
    for (uint c = 0; c < p.k; ++c) energy += x[row * p.k + c] * x[row * p.k + c];
    const float inverse = rsqrt(energy / float(p.k) + 1e-6f);
    for (uint c = 0; c < p.k; ++c) out[row * p.k + c] = x[row * p.k + c] * inverse;
}
kernel void rope_only(device const float* x [[buffer(0)]], device const float* unused [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.m) return;
    for (uint c = 0; c < p.k; c += 2) {
        const float angle = float(row) * pow(10000.f, -float(c) / float(p.k));
        const float a = x[row * p.k + c], b = x[row * p.k + c + 1];
        out[row * p.k + c] = a * cos(angle) - b * sin(angle);
        out[row * p.k + c + 1] = a * sin(angle) + b * cos(angle);
    }
}
kernel void norm_rope(device const float* x [[buffer(0)]], device const float* unused [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.m) return;
    float energy = 0;
    for (uint c = 0; c < p.k; ++c) energy += x[row * p.k + c] * x[row * p.k + c];
    const float inverse = rsqrt(energy / float(p.k) + 1e-6f);
    for (uint c = 0; c < p.k; c += 2) {
        const float angle = float(row) * pow(10000.f, -float(c) / float(p.k));
        const float a = x[row * p.k + c] * inverse, b = x[row * p.k + c + 1] * inverse;
        out[row * p.k + c] = a * cos(angle) - b * sin(angle);
        out[row * p.k + c + 1] = a * sin(angle) + b * cos(angle);
    }
}

// One-query-per-thread oracles: reveal stable online-softmax arithmetic without
// pretending this is a tuned FlashAttention threadgroup implementation.
kernel void attention_dense(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device float* out [[buffer(3)]], constant Params& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.m) return;
    float scores[128], peak = -INFINITY, denominator = 0;
    for (uint col = 0; col < p.n; ++col) {
        float dot = 0;
        for (uint d = 0; d < p.k; ++d) dot += q[row * p.k + d] * k[col * p.k + d];
        scores[col] = ((p.flags & 1) && col > row) ? -INFINITY : dot * rsqrt(float(p.k));
        peak = max(peak, scores[col]);
    }
    for (uint col = 0; col < p.n; ++col) { scores[col] = exp(scores[col] - peak); denominator += scores[col]; }
    for (uint d = 0; d < p.k; ++d) {
        float sum = 0;
        for (uint col = 0; col < p.n; ++col) sum += scores[col] * v[col * p.k + d];
        out[row * p.k + d] = sum / denominator;
    }
}
kernel void attention_online(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device float* out [[buffer(3)]], constant Params& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.m) return;
    float accumulator[32], peak = -INFINITY, denominator = 0;
    for (uint d = 0; d < p.k; ++d) accumulator[d] = 0;
    for (uint first = 0; first < p.n; first += 8) {
        float scores[8], tile_peak = -INFINITY;
        uint count = min(8u, p.n - first);
        for (uint c = 0; c < count; ++c) {
            float dot = 0;
            for (uint d = 0; d < p.k; ++d) dot += q[row * p.k + d] * k[(first + c) * p.k + d];
            scores[c] = ((p.flags & 1) && first + c > row) ? -INFINITY : dot * rsqrt(float(p.k));
            tile_peak = max(tile_peak, scores[c]);
        }
        if (tile_peak == -INFINITY) continue;
        const float next_peak = max(peak, tile_peak);
        const float correction = denominator > 0 ? exp(peak - next_peak) : 0.f;
        denominator *= correction;
        for (uint d = 0; d < p.k; ++d) accumulator[d] *= correction;
        for (uint c = 0; c < count; ++c) {
            const float probability = exp(scores[c] - next_peak);
            denominator += probability;
            for (uint d = 0; d < p.k; ++d) accumulator[d] += probability * v[(first + c) * p.k + d];
        }
        peak = next_peak;
    }
    for (uint d = 0; d < p.k; ++d) out[row * p.k + d] = accumulator[d] / denominator;
}
kernel void independent_branch(device const float* x [[buffer(0)]], device const float* unused [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    float value = x[i];
    for (uint step = 0; step < p.steps; ++step) value = value * p.scale + p.bias;
    out[i] = value;
}
