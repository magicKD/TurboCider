#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tc::metal_graph::qwen_ops {

inline constexpr std::string_view precision_recipe =
    "qwen-ln-centered-fp32-bf16-mod-bf16-rms128-preweight-bf16-pairrope-fp32-v2";
inline constexpr std::string_view conditioning_recipe =
    "qwen-condition-silu-bf16-fast-exp-denom-recip-gelu-bf16-scalars-rms-fp32-expand-v3";
struct Geometry {
    uint32_t hidden = 4096, heads = 32, head_dim = 128;
};
inline void validate_geometry(Geometry g) {
    if (g.head_dim != 128 || g.heads == 0 || g.heads > 32 ||
        g.hidden != g.heads * g.head_dim)
        throw std::invalid_argument("native Qwen ops require 1..32 heads of dimension 128");
}
inline void validate_epsilon(float epsilon) {
    // Changing epsilon is a separate model/precision recipe, not a tuning knob.
    if (!std::isfinite(epsilon) || epsilon != 1e-6f)
        throw std::invalid_argument("native Qwen ops require model epsilon 1e-6");
}
inline void validate_modulation_rows(uint32_t rows, uint32_t prefix_rows) {
    if (!rows || prefix_rows > rows)
        throw std::invalid_argument("native Qwen modulation prefix must be within output rows");
}
// Pure layout contract used by host bindings and CPU-only validation tests.
inline size_t matrix_extent(uint32_t rows, uint32_t columns, size_t stride,
                            size_t element_bytes, size_t offset, size_t available) {
    if (!rows || !columns || !element_bytes || offset % element_bytes ||
        stride % element_bytes || stride / element_bytes > UINT32_MAX ||
        size_t(columns) > std::numeric_limits<size_t>::max() / element_bytes)
        throw std::invalid_argument("native Qwen matrix extent/type alignment invalid");
    const size_t tail = size_t(columns) * element_bytes;
    if (stride < tail || size_t(rows - 1) >
        (std::numeric_limits<size_t>::max() - tail) / stride)
        throw std::invalid_argument("native Qwen matrix stride/extent overflow");
    const size_t bytes = size_t(rows - 1) * stride + tail;
    if (bytes > available)
        throw std::invalid_argument("native Qwen matrix exceeds retained buffer view");
    return bytes;
}
inline size_t head_extent(uint32_t heads, uint32_t tokens, uint32_t dimension,
                         size_t token_stride, size_t head_stride, size_t element,
                         size_t offset, size_t available) {
    const size_t one_head = matrix_extent(tokens, dimension, token_stride,
                                         element, offset, available);
    if (!heads || head_stride % element || head_stride / element > UINT32_MAX ||
        head_stride < one_head || size_t(heads - 1) >
        (std::numeric_limits<size_t>::max() - one_head) / head_stride)
        throw std::invalid_argument("native Qwen head stride/extent invalid");
    const size_t bytes = size_t(heads - 1) * head_stride + one_head;
    if (bytes > available)
        throw std::invalid_argument("native Qwen heads exceed retained buffer view");
    return bytes;
}
inline bool ranges_overlap(size_t a, size_t a_bytes, size_t b, size_t b_bytes) {
    return a <= b ? b - a < a_bytes : a - b < b_bytes;
}

struct NormParameters {
    uint32_t rows, input_stride, scale_stride, shift_stride, output_stride;
    uint32_t scale_rows, shift_rows, flags;
    float epsilon;
};
struct PointParameters {
    uint32_t rows, input_stride, branch_stride, mod_stride, output_stride;
    uint32_t mod_rows, shift_stride, shift_rows, flags;
};
struct QkParameters {
    uint32_t rows, q_stride, k_stride, cosine_stride, sine_stride;
    uint32_t q_token_stride, q_head_stride, k_token_stride, k_head_stride;
    float epsilon;
};
struct PackParameters {
    uint32_t rows, row_stride, token_stride, head_stride;
};
struct CopyParameters {
    uint32_t rows, input_token_stride, input_head_stride, output_token_stride, output_head_stride;
};
struct UnaryParameters { uint32_t rows, input_stride, output_stride; };
struct ConditionParameters {
    uint32_t rows, input_stride, output_stride;
    float epsilon;
};
struct ExpandParameters {
    uint32_t rows, prefix_rows, source_stride, output0_stride, output1_stride;
    uint32_t output2_stride, output3_stride;
};
static_assert(sizeof(NormParameters) == 36 && sizeof(PointParameters) == 36);
static_assert(sizeof(QkParameters) == 40 && sizeof(PackParameters) == 16);
static_assert(sizeof(CopyParameters) == 20);
static_assert(sizeof(UnaryParameters) == 12 && sizeof(ConditionParameters) == 16);
static_assert(sizeof(ExpandParameters) == 28);

inline constexpr std::string_view source_body = R"metal(
#include <metal_stdlib>
using namespace metal;

struct TCQNorm {
    uint rows, input_stride, scale_stride, shift_stride, output_stride;
    uint scale_rows, shift_rows, flags;
    float epsilon;
};
struct TCQPoint {
    uint rows, input_stride, branch_stride, mod_stride, output_stride;
    uint mod_rows, shift_stride, shift_rows, flags;
};
struct TCQQk {
    uint rows, q_stride, k_stride, cosine_stride, sine_stride;
    uint q_token_stride, q_head_stride, k_token_stride, k_head_stride;
    float epsilon;
};
struct TCQPack { uint rows, row_stride, token_stride, head_stride; };
struct TCQCopy { uint rows, input_token_stride, input_head_stride, output_token_stride, output_head_stride; };
struct TCQUnary { uint rows, input_stride, output_stride; };
struct TCQCondition { uint rows, input_stride, output_stride; float epsilon; };
struct TCQExpand {
    uint rows, prefix_rows, source_stride, output0_stride, output1_stride;
    uint output2_stride, output3_stride;
};

inline bfloat tc_q_mod(bfloat x, bfloat scale) {
    bfloat factor = bfloat(1.0f + float(scale));
    return bfloat(float(x) * float(factor));
}

// One 256-thread group owns a row. FP32 mean/centered variance and no affine norm;
// BF16 norm output is rounded before any optional scale/shift arithmetic.
kernel void tc_qwen_layer_norm(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* scale [[buffer(1)]],
    device const bfloat* shift [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant TCQNorm& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float means[8];
    threadgroup float variances[8];
    // Match normal MLX LayerNorm's centered variance. E[x*x]-mean*mean
    // catastrophically cancels for a high-mean row with small BF16 variation.
    constexpr uint reads = (TC_Q_H + 255) / 256;
    float centered[reads];
    float sum = 0.0f;
    for (uint i = 0; i < reads; ++i) {
        uint c = tid + i * 256;
        centered[i] = c < TC_Q_H ? float(input[ulong(row) * p.input_stride + c]) : 0.0f;
        sum += centered[i];
    }
    float ss = simd_sum(sum);
    if (lane == 0) means[simd] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = simd_sum(lane < 8 ? means[lane] : 0.0f) / float(TC_Q_H);
    float square = 0.0f;
    for (uint i = 0; i < reads; ++i) {
        uint c = tid + i * 256;
        centered[i] = c < TC_Q_H ? centered[i] - mean : 0.0f;
        square += centered[i] * centered[i];
    }
    float sq = simd_sum(square);
    if (lane == 0) variances[simd] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance = simd_sum(lane < 8 ? variances[lane] : 0.0f) / float(TC_Q_H);
    float inv = metal::precise::rsqrt(variance + p.epsilon);
    for (uint i = 0; i < reads; ++i) {
        uint c = tid + i * 256;
        if (c >= TC_Q_H) continue;
        bfloat value = bfloat(centered[i] * inv);
        if (p.flags & 1u)
            value = tc_q_mod(value, scale[ulong(p.scale_rows == 1 ? 0 : row) * p.scale_stride + c]);
        if (p.flags & 2u)
            value = bfloat(float(value) + float(shift[ulong(p.shift_rows == 1 ? 0 : row) * p.shift_stride + c]));
        output[ulong(row) * p.output_stride + c] = value;
    }
}

kernel void tc_qwen_modulation(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* scale [[buffer(1)]],
    device const bfloat* shift [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant TCQPoint& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x, row = gid.y;
    if (col >= TC_Q_H || row >= p.rows) return;
    bfloat value = tc_q_mod(input[ulong(row) * p.input_stride + col],
        scale[ulong(p.mod_rows == 1 ? 0 : row) * p.mod_stride + col]);
    if (p.flags & 1u)
        value = bfloat(float(value) + float(shift[ulong(p.shift_rows == 1 ? 0 : row) * p.shift_stride + col]));
    output[ulong(row) * p.output_stride + col] = value;
}

kernel void tc_qwen_residual(
    device const bfloat* residual [[buffer(0)]],
    device const bfloat* branch [[buffer(1)]],
    device const bfloat* gate [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant TCQPoint& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x, row = gid.y;
    if (col >= TC_Q_H || row >= p.rows) return;
    bfloat g = gate[ulong(p.mod_rows == 1 ? 0 : row) * p.mod_stride + col];
    if (!(p.flags & 1u)) g = bfloat(metal::precise::tanh(float(g)));
    bfloat product = bfloat(float(g) * float(branch[ulong(row) * p.branch_stride + col]));
    output[ulong(row) * p.output_stride + col] =
        bfloat(float(residual[ulong(row) * p.input_stride + col]) + float(product));
}

// Adapt the D=128 grouping of qwen21/metal/qk_norm_rope.hpp, but preserve
// normal MLX RMSNorm's additional BF16 boundary BEFORE multiplying weight.
// This v2 recipe intentionally differs from the old experimental fused path.
// One SIMD group owns one head; the weighted BF16 result enters FP32 RoPE.
kernel void tc_qwen_qk_norm_rope(
    device const bfloat* q [[buffer(0)]], device const bfloat* k [[buffer(1)]],
    device const bfloat* qw [[buffer(2)]], device const bfloat* kw [[buffer(3)]],
    device const float* cosine [[buffer(4)]], device const float* sine [[buffer(5)]],
    device bfloat* qout [[buffer(6)]], device bfloat* kout [[buffer(7)]],
    constant TCQQk& p [[buffer(8)]],
    uint task [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = task / TC_Q_HEADS, head = task % TC_Q_HEADS;
    if (row >= p.rows) return;
    ulong qs = ulong(row) * p.q_stride + head * 128;
    ulong ks = ulong(row) * p.k_stride + head * 128;
    float qx[4], kx[4], qsum = 0.0f, ksum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint d = lane * 4 + i;
        qx[i] = float(q[qs+d]); kx[i] = float(k[ks+d]);
        qsum += qx[i] * qx[i]; ksum += kx[i] * kx[i];
    }
    float qi = metal::precise::rsqrt(simd_sum(qsum) / 128.0f + p.epsilon);
    float ki = metal::precise::rsqrt(simd_sum(ksum) / 128.0f + p.epsilon);
    for (uint i = 0; i < 4; ++i) {
        uint d = lane * 4 + i;
        bfloat qnormalized = bfloat(qx[i] * qi);
        bfloat knormalized = bfloat(kx[i] * ki);
        qx[i] = float(bfloat(float(qnormalized) * float(qw[d])));
        kx[i] = float(bfloat(float(knormalized) * float(kw[d])));
    }
    ulong qdst = ulong(head) * p.q_head_stride + ulong(row) * p.q_token_stride;
    ulong kdst = ulong(head) * p.k_head_stride + ulong(row) * p.k_token_stride;
    for (uint i = 0; i < 4; i += 2) {
        uint d = lane * 4 + i;
        float c = cosine[ulong(row) * p.cosine_stride + d/2];
        float s = sine[ulong(row) * p.sine_stride + d/2];
        qout[qdst+d] = bfloat(qx[i]*c - qx[i+1]*s);
        qout[qdst+d+1] = bfloat(qx[i+1]*c + qx[i]*s);
        kout[kdst+d] = bfloat(kx[i]*c - kx[i+1]*s);
        kout[kdst+d+1] = bfloat(kx[i+1]*c + kx[i]*s);
    }
}

kernel void tc_qwen_pack_heads(device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]], constant TCQPack& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x, row = gid.y;
    if (col >= TC_Q_H || row >= p.rows) return;
    uint head = col / 128, dim = col % 128;
    output[ulong(head)*p.head_stride + ulong(row)*p.token_stride + dim] =
        input[ulong(row)*p.row_stride + col];
}
kernel void tc_qwen_unpack_heads(device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]], constant TCQPack& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x, row = gid.y;
    if (col >= TC_Q_H || row >= p.rows) return;
    uint head = col / 128, dim = col % 128;
    output[ulong(row)*p.row_stride + col] =
        input[ulong(head)*p.head_stride + ulong(row)*p.token_stride + dim];
}
kernel void tc_qwen_copy_heads(device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]], constant TCQCopy& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x, row = gid.y;
    if (col >= TC_Q_H || row >= p.rows) return;
    uint head = col / 128, dim = col % 128;
    output[ulong(head)*p.output_head_stride + ulong(row)*p.output_token_stride + dim] =
        input[ulong(head)*p.input_head_stride + ulong(row)*p.input_token_stride + dim];
}

kernel void tc_qwen_silu(device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]], constant TCQUnary& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x, row = gid.y;
    if (col >= TC_Q_H || row >= p.rows) return;
    bfloat x = input[ulong(row)*p.input_stride + col];
    // MLX Sigmoid<bfloat> uses its BF16 math overload for exp; the integer
    // literals in 1/(1+exp(abs(x))) preserve BF16 arithmetic at every stage.
    // Compiled MLX uses the fast exp variant. A near-midpoint input (-6.84375)
    // distinguishes it from precise exp after BF16 rounding. Keep this choice
    // local to SiLU; other operators retain their own precision contracts.
    bfloat exponential = bfloat(metal::fast::exp(metal::abs(float(x))));
    bfloat denominator = bfloat(1.0f + float(exponential));
    bfloat y = bfloat(1.0f / float(denominator));
    bfloat sigmoid = x < 0 ? y : bfloat(1.0f-float(y));
    output[ulong(row)*p.output_stride + col] = bfloat(float(x)*float(sigmoid));
}

kernel void tc_qwen_gelu(device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]], constant TCQUnary& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x, row = gid.y;
    if (col >= TC_Q_H || row >= p.rows) return;
    bfloat x = input[ulong(row)*p.input_stride + col];
    // transformer.cpp constructs every scalar in activation dtype. Keep the
    // Power primitive's one BF16 output rather than chained BF16 multiplies.
    bfloat half_x = bfloat(float(bfloat(0.5f))*float(x));
    bfloat cubed = bfloat(metal::pow(float(x),float(bfloat(3.0f))));
    bfloat cubic_term = bfloat(float(bfloat(0.044715f))*float(cubed));
    bfloat inner = bfloat(float(x)+float(cubic_term));
    bfloat argument = bfloat(float(bfloat(0.7978845608028654f))*float(inner));
    bfloat activation = bfloat(metal::precise::tanh(float(argument)));
    bfloat factor = bfloat(float(bfloat(1.0f))+float(activation));
    output[ulong(row)*p.output_stride + col] = bfloat(float(half_x)*float(factor));
}

kernel void tc_qwen_condition_norm(device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]], device bfloat* output [[buffer(2)]],
    constant TCQCondition& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float squares[8];
    constexpr uint reads = (TC_Q_H + 255) / 256;
    float x[reads], square = 0.0f;
    for (uint i=0; i<reads; ++i) {
        uint col = tid+i*256;
        x[i] = col < TC_Q_H ? float(input[ulong(row)*p.input_stride+col]) : 0.0f;
        square += x[i]*x[i];
    }
    float sq = simd_sum(square);
    if (lane == 0) squares[simd] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = simd_sum(lane < 8 ? squares[lane] : 0.0f)/float(TC_Q_H);
    float inv = metal::precise::rsqrt(mean+p.epsilon);
    for (uint i=0; i<reads; ++i) {
        uint col = tid+i*256;
        if (col >= TC_Q_H) continue;
        // Unlike Q/K fast RMSNorm, no BF16 boundary before the shifted weight.
        float normalized = x[i]*inv;
        float scale = float(weight[col])+1.0f;
        output[ulong(row)*p.output_stride+col] = bfloat(normalized*scale);
    }
}

kernel void tc_qwen_expand_modulation(device const bfloat* source [[buffer(0)]],
    device bfloat* output0 [[buffer(1)]], device bfloat* output1 [[buffer(2)]],
    device bfloat* output2 [[buffer(3)]], device bfloat* output3 [[buffer(4)]],
    constant TCQExpand& p [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x, row = gid.y;
    if (col >= TC_Q_H || row >= p.rows) return;
    ulong offset = ulong(row < p.prefix_rows ? 1 : 0)*p.source_stride + col;
    output0[ulong(row)*p.output0_stride+col] = source[offset];
    output1[ulong(row)*p.output1_stride+col] = source[offset+TC_Q_H];
    output2[ulong(row)*p.output2_stride+col] = source[offset+2*TC_Q_H];
    output3[ulong(row)*p.output3_stride+col] = source[offset+3*TC_Q_H];
}
)metal";

inline std::string source(Geometry geometry) {
    validate_geometry(geometry);
    return "#define TC_Q_H " + std::to_string(geometry.hidden) +
        "\n#define TC_Q_HEADS " + std::to_string(geometry.heads) + "\n" +
        std::string(source_body);
}
} // namespace tc::metal_graph::qwen_ops
