#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

inline float ltx_bf16_to_f32(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline ushort ltx_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return ushort(bits >> 16);
}

struct ltx_linear_args {
    uint rows;
    uint input_dim;
    uint output_dim;
    uint has_bias;
};

struct ltx_heads_args {
    uint rows;
    uint heads;
    uint head_dim;
    uint has_gate;
};

struct ltx_sol_args {
    uint rows;
    uint heads;
    uint head_dim;
    uint blocks;
    uint sink_start;
    uint sink_end;
    uint sink_query_start;
    uint sink_query_end;
    uint head_major_input;
    uint head_major_output;
    float scale_log2;
    float tau;
};

struct ltx_broadcast_args {
    uint rows;
    uint columns;
    uint parameter_rows;
};

struct ltx_split_args {
    uint rows;
    uint columns;
    uint conditioned_prefix_rows;
};

struct ltx_slice_columns_args {
    uint rows;
    uint input_columns;
    uint start_column;
    uint output_columns;
};

struct ltx_row_partition_args {
    uint input_rows;
    uint columns;
    uint start_row;
    uint output_rows;
};

struct ltx_concat_rows_args {
    uint prefix_rows;
    uint suffix_rows;
    uint columns;
};

struct ltx_diffusion_args {
    uint elements;
    float sigma;
    float sigma_next;
    float eta;
    float s_noise;
};

struct ltx_latent_stats_args {
    uint rows;
    uint channels;
    uint normalize;
};

kernel void ltx_add_f32(device float *output [[buffer(0)]],
                        device const float *left [[buffer(1)]],
                        device const float *right [[buffer(2)]],
                        constant uint &elements [[buffer(3)]],
                        uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = left[index] + right[index];
}

kernel void ltx_add_bf16(device ushort *output [[buffer(0)]],
                         device const ushort *left [[buffer(1)]],
                         device const ushort *right [[buffer(2)]],
                         constant uint &elements [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        output[index] = ltx_f32_to_bf16(
            ltx_bf16_to_f32(left[index]) + ltx_bf16_to_f32(right[index]));
    }
}

kernel void ltx_scale_f32(device float *output [[buffer(0)]],
                          device const float *input [[buffer(1)]],
                          constant float &scale [[buffer(2)]],
                          constant uint &elements [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = input[index] * scale;
}

kernel void ltx_velocity_to_denoised_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *sample [[buffer(1)]],
        device const ushort *velocity [[buffer(2)]],
        constant ltx_diffusion_args &args [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.elements) return;
    float value = fma(-args.sigma, ltx_bf16_to_f32(velocity[index]),
                      ltx_bf16_to_f32(sample[index]));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_velocity_to_denoised_bf16_split(
        device ushort *output [[buffer(0)]],
        device const ushort *sample [[buffer(1)]],
        device const ushort *velocity [[buffer(2)]],
        constant ltx_split_args &split [[buffer(3)]],
        constant float &generated_sigma [[buffer(4)]],
        constant float &conditioned_sigma [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    uint elements = split.rows * split.columns;
    if (index >= elements) return;
    uint row = index / split.columns;
    float sigma = row < split.conditioned_prefix_rows ?
        conditioned_sigma : generated_sigma;
    float value = fma(-sigma, ltx_bf16_to_f32(velocity[index]),
                      ltx_bf16_to_f32(sample[index]));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_euler_step_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *sample [[buffer(1)]],
        device const ushort *denoised [[buffer(2)]],
        constant ltx_diffusion_args &args [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.elements) return;
    float x = ltx_bf16_to_f32(sample[index]);
    float x0 = ltx_bf16_to_f32(denoised[index]);
    float velocity = (x - x0) / args.sigma;
    output[index] = ltx_f32_to_bf16(
        fma(velocity, args.sigma_next - args.sigma, x));
}

kernel void ltx_euler_ancestral_step_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *sample [[buffer(1)]],
        device const ushort *denoised [[buffer(2)]],
        device const float *noise [[buffer(3)]],
        constant ltx_diffusion_args &args [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.elements) return;
    float x0 = ltx_bf16_to_f32(denoised[index]);
    if (args.sigma_next == 0.0f) {
        output[index] = denoised[index];
        return;
    }
    float downstep_ratio =
        1.0f + (args.sigma_next / args.sigma - 1.0f) * args.eta;
    float sigma_down = args.sigma_next * downstep_ratio;
    float ratio = sigma_down / args.sigma;
    float deterministic = fma(
        ratio, ltx_bf16_to_f32(sample[index]), (1.0f - ratio) * x0);
    if (args.eta > 0.0f) {
        float alpha_next = 1.0f - args.sigma_next;
        float alpha_down = 1.0f - sigma_down;
        float variance = args.sigma_next * args.sigma_next -
            sigma_down * sigma_down * alpha_next * alpha_next /
                (alpha_down * alpha_down);
        float renoise = sqrt(max(variance, 0.0f));
        deterministic = fma(noise[index], args.s_noise * renoise,
                            alpha_next / alpha_down * deterministic);
    }
    output[index] = ltx_f32_to_bf16(deterministic);
}

kernel void ltx_renoise_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *clean [[buffer(1)]],
        device const ushort *noise [[buffer(2)]],
        constant ltx_diffusion_args &args [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.elements) return;
    float value = fma(ltx_bf16_to_f32(noise[index]), args.sigma,
                      ltx_bf16_to_f32(clean[index]) *
                          (1.0f - args.sigma));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_condition_prefix_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *clean_prefix [[buffer(1)]],
        constant ltx_diffusion_args &args [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.elements) return;
    float value = fma(ltx_bf16_to_f32(output[index]), args.sigma,
                      ltx_bf16_to_f32(clean_prefix[index]) *
                          (1.0f - args.sigma));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_latent_stats_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        device const ushort *mean [[buffer(2)]],
        device const ushort *standard_deviation [[buffer(3)]],
        constant ltx_latent_stats_args &args [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.channels;
    if (index >= elements) return;
    uint channel = index % args.channels;
    float value = ltx_bf16_to_f32(input[index]);
    float mean_value = ltx_bf16_to_f32(mean[channel]);
    float std_value = ltx_bf16_to_f32(standard_deviation[channel]);
    value = args.normalize ?
        (value - mean_value) / std_value :
        fma(value, std_value, mean_value);
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_residual_gate_f32(device float *output [[buffer(0)]],
                                  device const float *residual [[buffer(1)]],
                                  device const float *branch [[buffer(2)]],
                                  device const float *gate [[buffer(3)]],
                                  constant uint &elements [[buffer(4)]],
                                  uint index [[thread_position_in_grid]]) {
    if (index < elements)
        output[index] = fma(branch[index], gate[index], residual[index]);
}

kernel void ltx_residual_gate_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *residual [[buffer(1)]],
        device const ushort *branch [[buffer(2)]],
        device const ushort *gate [[buffer(3)]],
        constant ltx_broadcast_args &args [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.columns;
    if (index >= elements) return;
    uint row = index / args.columns;
    uint column = index % args.columns;
    uint gate_row = args.parameter_rows == 1u ? 0u : row;
    float value = fma(
        ltx_bf16_to_f32(branch[index]),
        ltx_bf16_to_f32(gate[gate_row * args.columns + column]),
        ltx_bf16_to_f32(residual[index]));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_residual_gate_bf16_split(
        device ushort *output [[buffer(0)]],
        device const ushort *residual [[buffer(1)]],
        device const ushort *branch [[buffer(2)]],
        device const ushort *generated_gate [[buffer(3)]],
        device const ushort *conditioned_gate [[buffer(4)]],
        constant ltx_split_args &args [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.columns;
    if (index >= elements) return;
    uint row = index / args.columns;
    uint column = index % args.columns;
    device const ushort *gate = row < args.conditioned_prefix_rows ?
        conditioned_gate : generated_gate;
    float value = fma(
        ltx_bf16_to_f32(branch[index]), ltx_bf16_to_f32(gate[column]),
        ltx_bf16_to_f32(residual[index]));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_affine_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        device const ushort *scale [[buffer(2)]],
        device const ushort *shift [[buffer(3)]],
        constant ltx_broadcast_args &args [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.columns;
    if (index >= elements) return;
    uint row = index / args.columns;
    uint column = index % args.columns;
    uint parameter_row = args.parameter_rows == 1u ? 0u : row;
    uint parameter = parameter_row * args.columns + column;
    float value = fma(
        ltx_bf16_to_f32(input[index]),
        1.0f + ltx_bf16_to_f32(scale[parameter]),
        ltx_bf16_to_f32(shift[parameter]));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_affine_bf16_split(
        device ushort *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        device const ushort *generated_scale [[buffer(2)]],
        device const ushort *generated_shift [[buffer(3)]],
        device const ushort *conditioned_scale [[buffer(4)]],
        device const ushort *conditioned_shift [[buffer(5)]],
        constant ltx_split_args &args [[buffer(6)]],
        uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.columns;
    if (index >= elements) return;
    uint row = index / args.columns;
    uint column = index % args.columns;
    device const ushort *scale = row < args.conditioned_prefix_rows ?
        conditioned_scale : generated_scale;
    device const ushort *shift = row < args.conditioned_prefix_rows ?
        conditioned_shift : generated_shift;
    float value = fma(
        ltx_bf16_to_f32(input[index]),
        1.0f + ltx_bf16_to_f32(scale[column]),
        ltx_bf16_to_f32(shift[column]));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_slice_columns_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        constant ltx_slice_columns_args &args [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.output_columns;
    if (index >= elements) return;
    uint row = index / args.output_columns;
    uint column = index % args.output_columns;
    output[index] = input[
        row * args.input_columns + args.start_column + column];
}

kernel void ltx_cast_f32_f16(device half *output [[buffer(0)]],
                             device const float *input [[buffer(1)]],
                             constant uint &elements [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = half(input[index]);
}

kernel void ltx_cast_f16_f32(device float *output [[buffer(0)]],
                             device const half *input [[buffer(1)]],
                             constant uint &elements [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = float(input[index]);
}

kernel void ltx_cast_f32_bf16(device ushort *output [[buffer(0)]],
                              device const float *input [[buffer(1)]],
                              constant uint &elements [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = ltx_f32_to_bf16(input[index]);
}

kernel void ltx_cast_bf16_f32(device float *output [[buffer(0)]],
                              device const ushort *input [[buffer(1)]],
                              constant uint &elements [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = ltx_bf16_to_f32(input[index]);
}

kernel void ltx_cast_bf16_f16(device half *output [[buffer(0)]],
                              device const ushort *input [[buffer(1)]],
                              constant uint &elements [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = half(ltx_bf16_to_f32(input[index]));
}

kernel void ltx_cast_f16_bf16(device ushort *output [[buffer(0)]],
                              device const half *input [[buffer(1)]],
                              constant uint &elements [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index < elements) output[index] = ltx_f32_to_bf16(float(input[index]));
}

kernel void ltx_slice_rows_bf16_f16(
        device half *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        constant ltx_row_partition_args &args [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    uint elements = args.output_rows * args.columns;
    if (index >= elements) return;
    uint source = args.start_row * args.columns + index;
    output[index] = half(ltx_bf16_to_f32(input[source]));
}

kernel void ltx_concat_rows_bf16_f16(
        device ushort *output [[buffer(0)]],
        device const ushort *prefix [[buffer(1)]],
        device const half *suffix [[buffer(2)]],
        constant ltx_concat_rows_args &args [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    uint prefix_elements = args.prefix_rows * args.columns;
    uint elements = (args.prefix_rows + args.suffix_rows) * args.columns;
    if (index >= elements) return;
    output[index] = index < prefix_elements ? prefix[index] :
        ltx_f32_to_bf16(float(suffix[index - prefix_elements]));
}

kernel void ltx_join_bf16_f16(device ushort *output [[buffer(0)]],
                              device const ushort *gpu_partial [[buffer(1)]],
                              device const half *ane_partial [[buffer(2)]],
                              constant uint &elements [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
    if (index < elements) {
        float value = ltx_bf16_to_f32(gpu_partial[index]) +
                      float(ane_partial[index]);
        output[index] = ltx_f32_to_bf16(value);
    }
}

kernel void ltx_join_residual_gate_bf16_f16(
        device ushort *output [[buffer(0)]],
        device const ushort *residual [[buffer(1)]],
        device const ushort *gpu_partial [[buffer(2)]],
        device const half *ane_partial [[buffer(3)]],
        device const ushort *gate [[buffer(4)]],
        constant ltx_broadcast_args &args [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.columns;
    if (index >= elements) return;
    uint row = index / args.columns;
    uint column = index % args.columns;
    uint gate_row = args.parameter_rows == 1u ? 0u : row;
    ushort joined = ltx_f32_to_bf16(
        ltx_bf16_to_f32(gpu_partial[index]) + float(ane_partial[index]));
    float value = fma(
        ltx_bf16_to_f32(joined),
        ltx_bf16_to_f32(gate[gate_row * args.columns + column]),
        ltx_bf16_to_f32(residual[index]));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_gelu_tanh_f32(device float *output [[buffer(0)]],
                              device const float *input [[buffer(1)]],
                              constant uint &elements [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    float x = input[index];
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    // The exact tanh is saturated well before |inner|=10.  Clamping avoids
    // exp overflow in the device implementation for large activations.
    output[index] = 0.5f * x *
        (1.0f + tanh(clamp(inner, -10.0f, 10.0f)));
}

kernel void ltx_gelu_tanh_bf16(device ushort *output [[buffer(0)]],
                               device const ushort *input [[buffer(1)]],
                               constant uint &elements [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    float x = ltx_bf16_to_f32(input[index]);
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    float value = 0.5f * x *
        (1.0f + tanh(clamp(inner, -10.0f, 10.0f)));
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_silu_bf16(device ushort *output [[buffer(0)]],
                          device const ushort *input [[buffer(1)]],
                          constant uint &elements [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    float x = clamp(ltx_bf16_to_f32(input[index]), -80.0f, 80.0f);
    output[index] = ltx_f32_to_bf16(x / (1.0f + exp(-x)));
}

kernel void ltx_sigmoid2_bf16(device ushort *output [[buffer(0)]],
                              device const ushort *input [[buffer(1)]],
                              constant uint &elements [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    float x = clamp(ltx_bf16_to_f32(input[index]), -20.0f, 20.0f);
    output[index] = ltx_f32_to_bf16(2.0f / (1.0f + exp(-x)));
}

kernel void ltx_rms_norm_f32(device float *output [[buffer(0)]],
                             device const float *input [[buffer(1)]],
                             constant uint &columns [[buffer(2)]],
                             constant float &epsilon [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    uint base = row * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 256u) {
        float value = input[base + column];
        sum = fma(value, value, sum);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_rms = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += 256u)
        output[base + column] = input[base + column] * inverse_rms;
}

kernel void ltx_rms_norm_weighted_f32(
        device float *output [[buffer(0)]],
        device const float *input [[buffer(1)]],
        device const float *weight [[buffer(2)]],
        constant uint &columns [[buffer(3)]],
        constant float &epsilon [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    uint base = row * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 256u) {
        float value = input[base + column];
        sum = fma(value, value, sum);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_rms = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += 256u)
        output[base + column] =
            input[base + column] * inverse_rms * weight[column];
}

kernel void ltx_rms_norm_bf16(device ushort *output [[buffer(0)]],
                              device const ushort *input [[buffer(1)]],
                              constant uint &columns [[buffer(2)]],
                              constant float &epsilon [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    uint base = row * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 256u) {
        float value = ltx_bf16_to_f32(input[base + column]);
        sum = fma(value, value, sum);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_rms = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += 256u) {
        float value = ltx_bf16_to_f32(input[base + column]) * inverse_rms;
        output[base + column] = ltx_f32_to_bf16(value);
    }
}

kernel void ltx_rms_norm_weighted_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        device const ushort *weight [[buffer(2)]],
        constant uint &columns [[buffer(3)]],
        constant float &epsilon [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    uint base = row * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 256u) {
        float value = ltx_bf16_to_f32(input[base + column]);
        sum = fma(value, value, sum);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_rms = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += 256u) {
        float value = ltx_bf16_to_f32(input[base + column]) * inverse_rms *
            ltx_bf16_to_f32(weight[column]);
        output[base + column] = ltx_f32_to_bf16(value);
    }
}

kernel void ltx_adaln_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        device const ushort *scale [[buffer(2)]],
        device const ushort *shift [[buffer(3)]],
        constant ltx_broadcast_args &args [[buffer(4)]],
        constant float &epsilon [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    uint base = row * args.columns;
    float sum = 0.0f;
    for (uint column = lane; column < args.columns; column += 256u) {
        float value = ltx_bf16_to_f32(input[base + column]);
        sum = fma(value, value, sum);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_rms = rsqrt(sums[0] / float(args.columns) + epsilon);
    uint parameter_row = args.parameter_rows == 1u ? 0u : row;
    uint parameter_base = parameter_row * args.columns;
    for (uint column = lane; column < args.columns; column += 256u) {
        float normalized =
            ltx_bf16_to_f32(input[base + column]) * inverse_rms;
        float value = fma(
            normalized,
            1.0f + ltx_bf16_to_f32(scale[parameter_base + column]),
            ltx_bf16_to_f32(shift[parameter_base + column]));
        output[base + column] = ltx_f32_to_bf16(value);
    }
}

kernel void ltx_adaln_bf16_split(
        device ushort *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        device const ushort *generated_scale [[buffer(2)]],
        device const ushort *generated_shift [[buffer(3)]],
        device const ushort *conditioned_scale [[buffer(4)]],
        device const ushort *conditioned_shift [[buffer(5)]],
        constant ltx_split_args &args [[buffer(6)]],
        constant float &epsilon [[buffer(7)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    uint base = row * args.columns;
    float sum = 0.0f;
    for (uint column = lane; column < args.columns; column += 256u) {
        float value = ltx_bf16_to_f32(input[base + column]);
        sum = fma(value, value, sum);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_rms = rsqrt(sums[0] / float(args.columns) + epsilon);
    device const ushort *scale = row < args.conditioned_prefix_rows ?
        conditioned_scale : generated_scale;
    device const ushort *shift = row < args.conditioned_prefix_rows ?
        conditioned_shift : generated_shift;
    for (uint column = lane; column < args.columns; column += 256u) {
        float normalized =
            ltx_bf16_to_f32(input[base + column]) * inverse_rms;
        float value = fma(
            normalized, 1.0f + ltx_bf16_to_f32(scale[column]),
            ltx_bf16_to_f32(shift[column]));
        output[base + column] = ltx_f32_to_bf16(value);
    }
}

kernel void ltx_adaln_bf16_f16(
        device ushort *output_bf16 [[buffer(0)]],
        device half *output_f16 [[buffer(1)]],
        device const ushort *input [[buffer(2)]],
        device const ushort *scale [[buffer(3)]],
        device const ushort *shift [[buffer(4)]],
        constant ltx_broadcast_args &args [[buffer(5)]],
        constant float &epsilon [[buffer(6)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    uint base = row * args.columns;
    float sum = 0.0f;
    for (uint column = lane; column < args.columns; column += 256u) {
        float value = ltx_bf16_to_f32(input[base + column]);
        sum = fma(value, value, sum);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_rms = rsqrt(sums[0] / float(args.columns) + epsilon);
    uint parameter_row = args.parameter_rows == 1u ? 0u : row;
    uint parameter_base = parameter_row * args.columns;
    for (uint column = lane; column < args.columns; column += 256u) {
        float normalized =
            ltx_bf16_to_f32(input[base + column]) * inverse_rms;
        float value = fma(
            normalized,
            1.0f + ltx_bf16_to_f32(scale[parameter_base + column]),
            ltx_bf16_to_f32(shift[parameter_base + column]));
        ushort rounded = ltx_f32_to_bf16(value);
        output_bf16[base + column] = rounded;
        output_f16[base + column] = half(ltx_bf16_to_f32(rounded));
    }
}

kernel void ltx_output_adaln_bf16(
        device ushort *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        device const ushort *embedded_timestep [[buffer(2)]],
        device const ushort *shift_table [[buffer(3)]],
        device const ushort *scale_table [[buffer(4)]],
        constant uint &columns [[buffer(5)]],
        constant float &epsilon [[buffer(6)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    uint base = row * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 256u)
        sum += ltx_bf16_to_f32(input[base + column]);
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sums[0] / float(columns);

    float square_sum = 0.0f;
    for (uint column = lane; column < columns; column += 256u) {
        float centered = ltx_bf16_to_f32(input[base + column]) - mean;
        square_sum = fma(centered, centered, square_sum);
    }
    sums[lane] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_std = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += 256u) {
        float normalized =
            (ltx_bf16_to_f32(input[base + column]) - mean) * inverse_std;
        float embedded = ltx_bf16_to_f32(embedded_timestep[column]);
        float shift = ltx_bf16_to_f32(shift_table[column]) + embedded;
        float scale = ltx_bf16_to_f32(scale_table[column]) + embedded;
        output[base + column] = ltx_f32_to_bf16(
            fma(normalized, 1.0f + scale, shift));
    }
}

kernel void ltx_output_adaln_bf16_split(
        device ushort *output [[buffer(0)]],
        device const ushort *input [[buffer(1)]],
        device const ushort *generated_embedded_timestep [[buffer(2)]],
        device const ushort *conditioned_embedded_timestep [[buffer(3)]],
        device const ushort *shift_table [[buffer(4)]],
        device const ushort *scale_table [[buffer(5)]],
        constant ltx_split_args &args [[buffer(6)]],
        constant float &epsilon [[buffer(7)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    uint base = row * args.columns;
    float sum = 0.0f;
    for (uint column = lane; column < args.columns; column += 256u)
        sum += ltx_bf16_to_f32(input[base + column]);
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sums[0] / float(args.columns);
    float square_sum = 0.0f;
    for (uint column = lane; column < args.columns; column += 256u) {
        float centered = ltx_bf16_to_f32(input[base + column]) - mean;
        square_sum = fma(centered, centered, square_sum);
    }
    sums[lane] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_std = rsqrt(sums[0] / float(args.columns) + epsilon);
    device const ushort *embedded_timestep =
        row < args.conditioned_prefix_rows ?
        conditioned_embedded_timestep : generated_embedded_timestep;
    for (uint column = lane; column < args.columns; column += 256u) {
        float normalized =
            (ltx_bf16_to_f32(input[base + column]) - mean) * inverse_std;
        float embedded = ltx_bf16_to_f32(embedded_timestep[column]);
        float shift = ltx_bf16_to_f32(shift_table[column]) + embedded;
        float scale = ltx_bf16_to_f32(scale_table[column]) + embedded;
        output[base + column] = ltx_f32_to_bf16(
            fma(normalized, 1.0f + scale, shift));
    }
}

kernel void ltx_linear_f32(device const float *input [[buffer(0)]],
                           device const float *weight [[buffer(1)]],
                           device const float *bias [[buffer(2)]],
                           device float *output [[buffer(3)]],
                           constant ltx_linear_args &args [[buffer(4)]],
                           uint2 lane [[thread_position_in_threadgroup]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16u + lane.y;
    uint column = group.x * 16u + lane.x;
    float sum = args.has_bias && column < args.output_dim ?
        bias[column] : 0.0f;
    uint tile_count = (args.input_dim + 15u) / 16u;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_column = tile * 16u + lane.x;
        input_tile[lane.y][lane.x] =
            row < args.rows && input_column < args.input_dim ?
            input[row * args.input_dim + input_column] : 0.0f;
        uint weight_column = tile * 16u + lane.y;
        weight_tile[lane.y][lane.x] =
            column < args.output_dim && weight_column < args.input_dim ?
            weight[column * args.input_dim + weight_column] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16u; k++)
            sum = fma(input_tile[lane.y][k], weight_tile[k][lane.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[row * args.output_dim + column] = sum;
}

kernel void ltx_linear_bf16(device const ushort *input [[buffer(0)]],
                            device const ushort *weight [[buffer(1)]],
                            device const ushort *bias [[buffer(2)]],
                            device ushort *output [[buffer(3)]],
                            constant ltx_linear_args &args [[buffer(4)]],
                            uint2 lane [[thread_position_in_threadgroup]],
                            uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16u + lane.y;
    uint column = group.x * 16u + lane.x;
    float sum = args.has_bias && column < args.output_dim ?
        ltx_bf16_to_f32(bias[column]) : 0.0f;
    uint tile_count = (args.input_dim + 15u) / 16u;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_column = tile * 16u + lane.x;
        input_tile[lane.y][lane.x] =
            row < args.rows && input_column < args.input_dim ?
            ltx_bf16_to_f32(input[row * args.input_dim + input_column]) :
            0.0f;
        uint weight_column = tile * 16u + lane.y;
        weight_tile[lane.y][lane.x] =
            column < args.output_dim && weight_column < args.input_dim ?
            ltx_bf16_to_f32(
                weight[column * args.input_dim + weight_column]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16u; k++)
            sum = fma(input_tile[lane.y][k], weight_tile[k][lane.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[row * args.output_dim + column] = ltx_f32_to_bf16(sum);
}

kernel void ltx_convrot_bf16(device ushort *output [[buffer(0)]],
                             device const ushort *input [[buffer(1)]],
                             constant uint &columns [[buffer(2)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float current[256];
    threadgroup float next[256];
    uint base = group.y * columns + group.x * 256u;
    current[lane] = ltx_bf16_to_f32(input[base + lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // H_256 = H_4 kron H_4 kron H_4 kron H_4.  H_4 is symmetric,
    // therefore this is the same transform used for x @ H in Comfy ConvRot.
    for (uint stride = 1u; stride < 256u; stride *= 4u) {
        uint digit = (lane / stride) & 3u;
        uint butterfly_base = lane - digit * stride;
        float a = current[butterfly_base];
        float b = current[butterfly_base + stride];
        float c = current[butterfly_base + 2u * stride];
        float d = current[butterfly_base + 3u * stride];
        switch (digit) {
            case 0u: next[lane] = a + b + c - d; break;
            case 1u: next[lane] = a + b - c + d; break;
            case 2u: next[lane] = a - b + c + d; break;
            default: next[lane] = -a + b + c + d; break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        current[lane] = next[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[base + lane] = ltx_f32_to_bf16(current[lane] * 0.0625f);
}

kernel void ltx_pack_heads_bf16(
                            device ushort *output [[buffer(0)]],
                            device const ushort *input [[buffer(1)]],
                            constant ltx_heads_args &args [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.heads * args.head_dim;
    if (index >= elements) return;
    uint dimension = index % args.head_dim;
    uint row = (index / args.head_dim) % args.rows;
    uint head = index / (args.head_dim * args.rows);
    uint source = row * args.heads * args.head_dim +
        head * args.head_dim + dimension;
    output[index] = input[source];
}

kernel void ltx_pack_rope_split_bf16(
                            device ushort *output [[buffer(0)]],
                            device const ushort *input [[buffer(1)]],
                            device const ushort *cosine [[buffer(2)]],
                            device const ushort *sine [[buffer(3)]],
                            constant ltx_heads_args &args [[buffer(4)]],
                            uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.heads * args.head_dim;
    if (index >= elements) return;
    uint dimension = index % args.head_dim;
    uint row = (index / args.head_dim) % args.rows;
    uint head = index / (args.head_dim * args.rows);
    uint half_dim = args.head_dim / 2u;
    uint pair = dimension % half_dim;
    uint source_base = row * args.heads * args.head_dim +
        head * args.head_dim;
    float first = ltx_bf16_to_f32(input[source_base + pair]);
    float second = ltx_bf16_to_f32(
        input[source_base + pair + half_dim]);
    uint frequency = (head * args.rows + row) * half_dim + pair;
    float c = ltx_bf16_to_f32(cosine[frequency]);
    float s = ltx_bf16_to_f32(sine[frequency]);
    float value = dimension < half_dim ?
        fma(-second, s, first * c) :
        fma(first, s, second * c);
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_unpack_heads_gate_bf16(
                            device ushort *output [[buffer(0)]],
                            device const ushort *input [[buffer(1)]],
                            device const ushort *gate [[buffer(2)]],
                            constant ltx_heads_args &args [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
    uint elements = args.rows * args.heads * args.head_dim;
    if (index >= elements) return;
    uint dimension = index % args.head_dim;
    uint head = (index / args.head_dim) % args.heads;
    uint row = index / (args.head_dim * args.heads);
    uint source = (head * args.rows + row) * args.head_dim + dimension;
    float value = ltx_bf16_to_f32(input[source]);
    if (args.has_gate)
        value *= ltx_bf16_to_f32(gate[row * args.heads + head]);
    output[index] = ltx_f32_to_bf16(value);
}

kernel void ltx_linear_int8_weight_bf16(
                            device const ushort *input [[buffer(0)]],
                            device const char *weight [[buffer(1)]],
                            device const float *weight_scale [[buffer(2)]],
                            device const ushort *bias [[buffer(3)]],
                            device ushort *output [[buffer(4)]],
                            constant ltx_linear_args &args [[buffer(5)]],
                            uint2 lane [[thread_position_in_threadgroup]],
                            uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16u + lane.y;
    uint column = group.x * 16u + lane.x;
    float sum = 0.0f;
    uint tile_count = (args.input_dim + 15u) / 16u;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_column = tile * 16u + lane.x;
        input_tile[lane.y][lane.x] =
            row < args.rows && input_column < args.input_dim ?
            ltx_bf16_to_f32(input[row * args.input_dim + input_column]) :
            0.0f;
        uint weight_column = tile * 16u + lane.y;
        weight_tile[lane.y][lane.x] =
            column < args.output_dim && weight_column < args.input_dim ?
            float(weight[column * args.input_dim + weight_column]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16u; k++)
            sum = fma(input_tile[lane.y][k], weight_tile[k][lane.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim) {
        sum *= weight_scale[column];
        if (args.has_bias) sum += ltx_bf16_to_f32(bias[column]);
        output[row * args.output_dim + column] = ltx_f32_to_bf16(sum);
    }
}

inline uint ltx_sol_index(uint row, uint head, uint dimension,
                          constant ltx_sol_args &args, bool head_major) {
    return head_major ?
        (head * args.rows + row) * args.head_dim + dimension :
        (row * args.heads + head) * args.head_dim + dimension;
}

kernel void ltx_sol_reduce_summaries_bf16(
                         device const ushort *query [[buffer(0)]],
                         device const ushort *key [[buffer(1)]],
                         device const ushort *value [[buffer(2)]],
                         device float *query_centroids [[buffer(3)]],
                         device ushort *key_centroids [[buffer(4)]],
                         device ushort *value_sums [[buffer(5)]],
                         constant ltx_sol_args &args [[buffer(6)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint2 group [[threadgroup_position_in_grid]]) {
    uint block = group.x;
    uint head = group.y;
    if (block >= args.blocks || head >= args.heads ||
        tid >= args.head_dim) return;
    uint begin = block * 64u;
    uint end = min(begin + 64u, args.rows);
    float query_sum = 0.0f;
    float key_sum = 0.0f;
    float value_sum = 0.0f;
    for (uint row = begin; row < end; row++) {
        uint index = ltx_sol_index(
            row, head, tid, args, args.head_major_input != 0u);
        query_sum += ltx_bf16_to_f32(query[index]);
        key_sum += ltx_bf16_to_f32(key[index]);
        value_sum += ltx_bf16_to_f32(value[index]);
    }
    uint summary = (head * args.blocks + block) * args.head_dim + tid;
    float inverse = 1.0f / float(end - begin);
    query_centroids[summary] = query_sum * inverse;
    key_centroids[summary] = ltx_f32_to_bf16(key_sum * inverse);
    value_sums[summary] = ltx_f32_to_bf16(value_sum);
}

kernel void ltx_sol_thresholds_diag_bf16(
                         device const float *query_centroids [[buffer(0)]],
                         device const ushort *key_centroids [[buffer(1)]],
                         device float *thresholds [[buffer(2)]],
                         constant ltx_sol_args &args [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint2 group [[threadgroup_position_in_grid]]) {
    uint query_block = group.x;
    uint head = group.y;
    threadgroup float mean_parts[128];
    threadgroup float variance_parts[128];
    float mean_part = 0.0f;
    float variance_part = 0.0f;
    if (query_block < args.blocks && head < args.heads &&
        tid < args.head_dim) {
        uint base = head * args.blocks * args.head_dim + tid;
        float key_mean = 0.0f;
        for (uint block = 0; block < args.blocks; block++)
            key_mean += ltx_bf16_to_f32(
                key_centroids[base + block * args.head_dim]);
        key_mean /= float(args.blocks);
        float key_variance = 0.0f;
        for (uint block = 0; block < args.blocks; block++) {
            float delta = ltx_bf16_to_f32(
                key_centroids[base + block * args.head_dim]) - key_mean;
            key_variance = fma(delta, delta, key_variance);
        }
        key_variance /= float(args.blocks);
        float q = query_centroids[
            (head * args.blocks + query_block) * args.head_dim + tid];
        mean_part = q * key_mean;
        variance_part = q * q * key_variance;
    }
    mean_parts[tid] = mean_part;
    variance_parts[tid] = variance_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride; stride >>= 1u) {
        if (tid < stride) {
            mean_parts[tid] += mean_parts[tid + stride];
            variance_parts[tid] += variance_parts[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u && query_block < args.blocks && head < args.heads) {
        float mean = mean_parts[0] * args.scale_log2;
        float variance = max(variance_parts[0], 0.0f) *
            args.scale_log2 * args.scale_log2;
        thresholds[head * args.blocks + query_block] =
            mean + args.tau * sqrt(variance + 1.0e-6f);
    }
}

kernel void ltx_sol_key_stats_bf16(
                         device const ushort *key_centroids [[buffer(0)]],
                         device float *statistics [[buffer(1)]],
                         constant ltx_sol_args &args [[buffer(2)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint head [[threadgroup_position_in_grid]]) {
    if (head >= args.heads || tid >= args.head_dim) return;
    uint base = head * args.blocks * args.head_dim + tid;
    float mean = 0.0f;
    for (uint block = 0; block < args.blocks; block++)
        mean += ltx_bf16_to_f32(
            key_centroids[base + block * args.head_dim]);
    mean /= float(args.blocks);
    float variance = 0.0f;
    for (uint block = 0; block < args.blocks; block++) {
        float delta = ltx_bf16_to_f32(
            key_centroids[base + block * args.head_dim]) - mean;
        variance = fma(delta, delta, variance);
    }
    variance /= float(args.blocks);
    uint stats_base = head * args.head_dim * 2u;
    statistics[stats_base + tid] = mean;
    statistics[stats_base + args.head_dim + tid] = variance;
}

kernel void ltx_sol_thresholds_stats_bf16(
                         device const float *query_centroids [[buffer(0)]],
                         device const float *statistics [[buffer(1)]],
                         device float *thresholds [[buffer(2)]],
                         constant ltx_sol_args &args [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint2 group [[threadgroup_position_in_grid]]) {
    uint query_block = group.x;
    uint head = group.y;
    threadgroup float mean_parts[128];
    threadgroup float variance_parts[128];
    float mean_part = 0.0f;
    float variance_part = 0.0f;
    if (query_block < args.blocks && head < args.heads &&
        tid < args.head_dim) {
        uint stats_base = head * args.head_dim * 2u;
        float q = query_centroids[
            (head * args.blocks + query_block) * args.head_dim + tid];
        mean_part = q * statistics[stats_base + tid];
        variance_part = q * q *
            statistics[stats_base + args.head_dim + tid];
    }
    mean_parts[tid] = mean_part;
    variance_parts[tid] = variance_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride; stride >>= 1u) {
        if (tid < stride) {
            mean_parts[tid] += mean_parts[tid + stride];
            variance_parts[tid] += variance_parts[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u && query_block < args.blocks && head < args.heads) {
        float mean = mean_parts[0] * args.scale_log2;
        float variance = max(variance_parts[0], 0.0f) *
            args.scale_log2 * args.scale_log2;
        thresholds[head * args.blocks + query_block] =
            mean + args.tau * sqrt(variance + 1.0e-6f);
    }
}

kernel void ltx_sol_route_mask_bf16(
                         device const float *query_centroids [[buffer(0)]],
                         device const ushort *key_centroids [[buffer(1)]],
                         device const float *thresholds [[buffer(2)]],
                         device float *routes [[buffer(3)]],
                         constant ltx_sol_args &args [[buffer(4)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint2 group [[threadgroup_position_in_grid]]) {
    uint query_block = group.x;
    uint head = group.y;
    if (query_block >= args.blocks || head >= args.heads) return;
    bool query_sink = query_block >= args.sink_query_start &&
                      query_block < args.sink_query_end;
    uint query_base =
        (head * args.blocks + query_block) * args.head_dim;
    for (uint key_block = tid; key_block < args.blocks; key_block += 128u) {
        bool neighbor = abs(int(query_block) - int(key_block)) <= 1;
        bool key_sink = key_block >= args.sink_start &&
                        key_block < args.sink_end;
        bool exact = query_sink || neighbor || key_sink;
        if (!exact) {
            uint key_base =
                (head * args.blocks + key_block) * args.head_dim;
            float score = 0.0f;
            for (uint dimension = 0; dimension < args.head_dim; dimension++)
                score = fma(query_centroids[query_base + dimension],
                    ltx_bf16_to_f32(key_centroids[key_base + dimension]),
                    score);
            exact = score * args.scale_log2 >
                thresholds[head * args.blocks + query_block];
        }
        routes[(head * args.blocks + query_block) * args.blocks +
               key_block] = exact ? 1.0f : 0.0f;
    }
}

kernel void ltx_sol_attention_bf16(
                         device const ushort *query [[buffer(0)]],
                         device const ushort *key [[buffer(1)]],
                         device const ushort *value [[buffer(2)]],
                         device const ushort *key_centroids [[buffer(3)]],
                         device const ushort *value_sums [[buffer(4)]],
                         device const float *routes [[buffer(5)]],
                         device ushort *output [[buffer(6)]],
                         constant ltx_sol_args &args [[buffer(7)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint group [[threadgroup_position_in_grid]]) {
    uint row = group % args.rows;
    uint head = group / args.rows;
    if (head >= args.heads) return;
    uint dimensions[4] = {lane, lane + 32u, lane + 64u, lane + 96u};
    float q[4];
    float numerator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint slot = 0; slot < 4u; slot++) {
        uint dimension = dimensions[slot];
        q[slot] = dimension < args.head_dim ? ltx_bf16_to_f32(query[
            ltx_sol_index(row, head, dimension, args,
                          args.head_major_input != 0u)]) : 0.0f;
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;
    uint query_block = row / 64u;
    uint route_base = (head * args.blocks + query_block) * args.blocks;
    for (uint block = 0; block < args.blocks; block++) {
        bool exact = routes[route_base + block] != 0.0f;
        uint begin = block * 64u;
        uint end = min(begin + 64u, args.rows);
        uint count = exact ? end - begin : 1u;
        for (uint item = 0; item < count; item++) {
            uint key_row = begin + item;
            uint summary_base =
                (head * args.blocks + block) * args.head_dim;
            float partial = 0.0f;
            for (uint slot = 0; slot < 4u; slot++) {
                uint dimension = dimensions[slot];
                float k = 0.0f;
                if (dimension < args.head_dim) {
                    k = exact ? ltx_bf16_to_f32(key[
                        ltx_sol_index(key_row, head, dimension, args,
                                      args.head_major_input != 0u)]) :
                        ltx_bf16_to_f32(
                            key_centroids[summary_base + dimension]);
                }
                partial = fma(q[slot], k, partial);
            }
            float score = simd_sum(partial) * args.scale_log2;
            float next_maximum = max(maximum, score);
            float correction = maximum == -INFINITY ? 0.0f :
                fast::exp2(maximum - next_maximum);
            float probability = fast::exp2(score - next_maximum);
            float multiplicity = exact ? 1.0f : float(end - begin);
            denominator = denominator * correction +
                          probability * multiplicity;
            for (uint slot = 0; slot < 4u; slot++) {
                uint dimension = dimensions[slot];
                if (dimension >= args.head_dim) continue;
                float v = exact ? ltx_bf16_to_f32(value[
                    ltx_sol_index(key_row, head, dimension, args,
                                  args.head_major_input != 0u)]) :
                    ltx_bf16_to_f32(value_sums[summary_base + dimension]);
                numerator[slot] = numerator[slot] * correction +
                                  probability * v;
            }
            maximum = next_maximum;
        }
    }
    for (uint slot = 0; slot < 4u; slot++) {
        uint dimension = dimensions[slot];
        if (dimension >= args.head_dim) continue;
        uint index = ltx_sol_index(
            row, head, dimension, args, args.head_major_output != 0u);
        output[index] = ltx_f32_to_bf16(numerator[slot] / denominator);
    }
}

/*
 * Tiled Sol-Attention core for the LTX video self-attention shape.
 *
 * The simdgroup-matrix tiling follows the public PyTorch MPS prefill kernel
 * and the local ComfyUI-SolAttn-MPS reference.  LTX has at most 63 64-token
 * blocks at the supported 480p Stage-2 shape, so one summary tile covers the
 * full sequence.  Routing remains in the existing exact/summary preparation
 * kernels; this kernel only replaces the scalar attention accumulation.
 */
#define LTX_SOL_UNROLL _Pragma("clang loop unroll(full)")

template <typename T>
struct ltx_sol_pointer_element {};
template <typename T>
struct ltx_sol_pointer_element<device T *> { using type = remove_cv_t<T>; };
template <typename T>
using ltx_sol_pointer_element_t =
    typename ltx_sol_pointer_element<remove_cv_t<T>>::type;

template <int value>
using ltx_sol_int = integral_constant<int, value>;

template <typename T, short BROWS, short BCOLS,
          short DST_ROW, short DST_COL, short REDUCTION_DIM,
          short TGP_SIZE,
          short N_READS = (BCOLS * BROWS) / TGP_SIZE,
          short TCOLS = BCOLS / N_READS,
          short TROWS = TGP_SIZE / TCOLS>
struct ltx_sol_block_loader {
    static constant constexpr short vector_size = N_READS;
    const int source_ld;
    const int tile_stride;
    const short row;
    const short column;
    threadgroup T *destination;
    const device T *source;

    inline ltx_sol_block_loader(
            const device T *source_, int source_ld_,
            threadgroup T *destination_,
            ushort simdgroup [[simdgroup_index_in_threadgroup]],
            ushort lane [[thread_index_in_simdgroup]])
        : source_ld(source_ld_),
          tile_stride(REDUCTION_DIM ? BCOLS : BROWS * source_ld_),
          row((simdgroup * 32 + lane) / TCOLS),
          column(vector_size * ((simdgroup * 32 + lane) % TCOLS)),
          destination(destination_ + row * DST_ROW + column * DST_COL),
          source(source_ + row * source_ld_ + column) {}

    inline void load_unsafe() const {
        LTX_SOL_UNROLL
        for (short i = 0; i < BROWS; i += TROWS) {
            LTX_SOL_UNROLL
            for (short j = 0; j < vector_size; j++)
                destination[i * DST_ROW + j * DST_COL] =
                    source[i * source_ld + j];
        }
    }

    inline void load_safe(short2 source_shape) const {
        source_shape -= short2(column, row);
        if (source_shape.x <= 0 || source_shape.y <= 0) {
            LTX_SOL_UNROLL
            for (short i = 0; i < BROWS; i += TROWS) {
                LTX_SOL_UNROLL
                for (short j = 0; j < vector_size; j++)
                    destination[i * DST_ROW + j * DST_COL] = T(0);
            }
            return;
        }
        LTX_SOL_UNROLL
        for (short i = 0; i < BROWS; i += TROWS) {
            LTX_SOL_UNROLL
            for (short j = 0; j < vector_size; j++) {
                bool valid = i < source_shape.y && j < source_shape.x;
                destination[i * DST_ROW + j * DST_COL] =
                    valid ? source[i * source_ld + j] : T(0);
            }
        }
    }
};

struct ltx_sol_max_op {
    template <typename T>
    inline static T apply(T x, T y) { return max(x, y); }
};
struct ltx_sol_sum_op {
    template <typename T>
    inline static T apply(T x, T y) { return x + y; }
};
struct ltx_sol_mul_op {
    template <typename T>
    inline static T apply(T x, T y) { return x * y; }
};
struct ltx_sol_exp_sub_op {
    template <typename T>
    inline static T apply(T x, T y) {
        return y == -INFINITY ? T(0) : fast::exp2(x - y);
    }
};

template <typename T>
struct ltx_sol_mma_fragment {
    static constant constexpr int rows = 8;
    static constant constexpr int columns = 8;
    static constant constexpr int elements = 2;
    static constant constexpr int element_rows = 1;
    static constant constexpr int element_columns = 2;
    using matrix_type = metal::simdgroup_matrix<T, rows, columns>;
    using fragment_type = metal::vec<T, elements>;

    template <typename U>
    using typed_matrix = metal::simdgroup_matrix<U, rows, columns>;
    template <typename U>
    using typed_fragment = metal::vec<U, elements>;

    inline static short2 coordinate(
            ushort lane [[thread_index_in_simdgroup]]) {
        short quad = lane / 4;
        short row = (quad & 4) + ((lane / 2) % 4);
        short column = (quad & 2) * 2 + (lane % 2) * 2;
        return short2(column, row);
    }

    template <typename Source, typename StrideX, typename StrideY>
    inline static void load(thread fragment_type &destination,
                            Source source, StrideX stride_x,
                            StrideY stride_y) {
        LTX_SOL_UNROLL
        for (short i = 0; i < element_rows; i++) {
            LTX_SOL_UNROLL
            for (short j = 0; j < element_columns; j++)
                destination[i * element_columns + j] = T(
                    source[i * stride_x.value + j * stride_y.value]);
        }
    }

    template <typename Destination, typename StrideY>
    inline static void store_safe(
            const thread fragment_type &source, Destination destination,
            int stride_x, StrideY stride_y, short limit_x, short limit_y,
            short offset_x, short offset_y) {
        using U = ltx_sol_pointer_element_t<Destination>;
        LTX_SOL_UNROLL
        for (short i = 0; i < element_rows; i++) {
            LTX_SOL_UNROLL
            for (short j = 0; j < element_columns; j++) {
                if (offset_x + i < limit_x && offset_y + j < limit_y)
                    destination[(offset_x + i) * stride_x +
                                (offset_y + j) * stride_y.value] =
                        U(source[i * element_columns + j]);
            }
        }
    }

    template <typename A, typename B, typename C>
    inline static void mma(thread fragment_type &destination,
                           thread typed_fragment<A> &left,
                           thread typed_fragment<B> &right,
                           thread typed_fragment<C> &accumulator) {
        matrix_type destination_matrix;
        typed_matrix<A> left_matrix;
        typed_matrix<B> right_matrix;
        typed_matrix<C> accumulator_matrix;
        reinterpret_cast<thread typed_fragment<A> &>(
            left_matrix.thread_elements()) = left;
        reinterpret_cast<thread typed_fragment<B> &>(
            right_matrix.thread_elements()) = right;
        reinterpret_cast<thread typed_fragment<C> &>(
            accumulator_matrix.thread_elements()) = accumulator;
        simdgroup_multiply_accumulate(destination_matrix, left_matrix,
                                      right_matrix, accumulator_matrix);
        destination = reinterpret_cast<thread fragment_type &>(
            destination_matrix.thread_elements());
    }

    template <typename Operation>
    inline static void row_reduce(const thread fragment_type &input,
                                  thread T *output) {
        T thread_value = Operation::apply(input.x, input.y);
        T quad_value = simd_shuffle_xor(thread_value, ushort(1));
        quad_value = Operation::apply(thread_value, quad_value);
        T simd_value = simd_shuffle_xor(quad_value, ushort(8));
        simd_value = Operation::apply(quad_value, simd_value);
        output[0] = Operation::apply(output[0], simd_value);
    }

    template <typename Operation>
    inline static void row_binary(thread fragment_type &input,
                                  thread T *row_values) {
        LTX_SOL_UNROLL
        for (short j = 0; j < element_columns; j++)
            input[j] = Operation::apply(input[j], row_values[0]);
    }
};

template <typename T, int TILE_ROWS, int TILE_COLUMNS,
          typename Fragment = ltx_sol_mma_fragment<T>>
struct ltx_sol_mma_tile {
    using fragment_type = typename Fragment::fragment_type;
    static constant constexpr int fragment_rows = Fragment::rows;
    static constant constexpr int fragment_columns = Fragment::columns;
    static constant constexpr int tile_columns = TILE_COLUMNS;
    static constant constexpr int rows_per_thread =
        TILE_ROWS * Fragment::element_rows;
    fragment_type fragments[TILE_ROWS * TILE_COLUMNS];

    inline void clear() {
        LTX_SOL_UNROLL
        for (short index = 0; index < TILE_ROWS * TILE_COLUMNS; index++)
            fragments[index] = fragment_type(0);
    }
    inline thread fragment_type &at(short row, short column) {
        return fragments[row * TILE_COLUMNS + column];
    }
    inline const thread fragment_type &at(short row, short column) const {
        return fragments[row * TILE_COLUMNS + column];
    }
    template <typename Operation>
    inline void row_reduce(thread T values[rows_per_thread]) const {
        LTX_SOL_UNROLL
        for (short i = 0; i < TILE_ROWS; i++) {
            LTX_SOL_UNROLL
            for (short j = 0; j < TILE_COLUMNS; j++)
                Fragment::template row_reduce<Operation>(
                    at(i, j), &values[i * Fragment::element_rows]);
        }
    }
    template <typename Operation>
    inline void row_binary(thread T values[rows_per_thread]) {
        LTX_SOL_UNROLL
        for (short i = 0; i < TILE_ROWS; i++) {
            LTX_SOL_UNROLL
            for (short j = 0; j < TILE_COLUMNS; j++)
                Fragment::template row_binary<Operation>(
                    at(i, j), &values[i * Fragment::element_rows]);
        }
    }
    template <typename U, int WIDTH_X, int WIDTH_Y,
              int STRIDE_X, int STRIDE_Y>
    inline void load(const threadgroup U *source) {
        LTX_SOL_UNROLL
        for (short i = 0; i < TILE_ROWS; i++) {
            LTX_SOL_UNROLL
            for (short j = 0; j < TILE_COLUMNS; j++)
                Fragment::load(
                    at(i, j),
                    &source[(i * fragment_rows) * WIDTH_X * STRIDE_X +
                            (j * fragment_columns) * WIDTH_Y * STRIDE_Y],
                    ltx_sol_int<STRIDE_X>{}, ltx_sol_int<STRIDE_Y>{});
        }
    }
    template <typename U, int WIDTH_X, int WIDTH_Y>
    inline void store_safe(device U *destination, int leading_dimension,
                           short2 destination_shape) const {
        LTX_SOL_UNROLL
        for (short i = 0; i < TILE_ROWS; i++) {
            LTX_SOL_UNROLL
            for (short j = 0; j < TILE_COLUMNS; j++)
                Fragment::store_safe(
                    at(i, j), destination, leading_dimension,
                    ltx_sol_int<1>{}, destination_shape.y,
                    destination_shape.x,
                    (i * fragment_rows) * WIDTH_X,
                    (j * fragment_columns) * WIDTH_Y);
        }
    }
};

template <typename D, typename A, typename B, typename C,
          int M, int N, int K, typename FD, typename FA,
          typename FB, typename FC>
inline void ltx_sol_tile_multiply(
        thread ltx_sol_mma_tile<D, M, N, FD> &destination,
        thread ltx_sol_mma_tile<A, M, K, FA> &left,
        thread ltx_sol_mma_tile<B, K, N, FB> &right,
        thread ltx_sol_mma_tile<C, M, N, FC> &accumulator) {
    LTX_SOL_UNROLL
    for (short m = 0; m < M; m++) {
        LTX_SOL_UNROLL
        for (short n = 0; n < N; n++) {
            short column = (m % 2) ? (N - 1 - n) : n;
            LTX_SOL_UNROLL
            for (short k = 0; k < K; k++)
                FD::mma(destination.at(m, column), left.at(m, k),
                        right.at(k, column), accumulator.at(m, column));
        }
    }
}

template <bool WEIGHT_DENOMINATOR, typename ScoreTile, typename OutputTile>
inline void ltx_sol_accumulate_tile(
        thread ScoreTile &scores, thread OutputTile &output,
        threadgroup bfloat *values, short values_offset,
        thread float *maximum, thread float *denominator,
        uint block_start, short block_column,
        uint blocks, uint rows) {
    constexpr short value_tiles = OutputTile::tile_columns;
    constexpr short key_tiles = ScoreTile::tile_columns;
    using Fragment = ltx_sol_mma_fragment<float>;
    ltx_sol_mma_tile<float, 1, 1, Fragment> value_tile;
    float next_maximum[ScoreTile::rows_per_thread];
    float correction[ScoreTile::rows_per_thread];
    LTX_SOL_UNROLL
    for (short row = 0; row < ScoreTile::rows_per_thread; row++)
        next_maximum[row] = maximum[row];
    scores.template row_reduce<ltx_sol_max_op>(next_maximum);
    scores.template row_binary<ltx_sol_exp_sub_op>(next_maximum);
    LTX_SOL_UNROLL
    for (short row = 0; row < ScoreTile::rows_per_thread; row++) {
        correction[row] = next_maximum[row] == -INFINITY ? 1.0f :
            fast::exp2(maximum[row] - next_maximum[row]);
        maximum[row] = next_maximum[row];
    }
    output.template row_binary<ltx_sol_mul_op>(correction);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    LTX_SOL_UNROLL
    for (short value_index = 0; value_index < value_tiles; value_index++) {
        LTX_SOL_UNROLL
        for (short key_index = 0; key_index < key_tiles; key_index++) {
            short key_offset = key_index * 8;
            short dimension_offset = value_index * 8;
            value_tile.template load<bfloat, 1, 1, 128, 1>(
                &values[values_offset + key_offset * 128 +
                        dimension_offset]);
            simdgroup_barrier(mem_flags::mem_none);
            Fragment::mma(output.at(0, value_index),
                          scores.at(0, key_index), value_tile.at(0, 0),
                          output.at(0, value_index));
        }
    }

    if constexpr (WEIGHT_DENOMINATOR) {
        LTX_SOL_UNROLL
        for (short key_index = 0; key_index < key_tiles; key_index++) {
            LTX_SOL_UNROLL
            for (short element = 0; element < Fragment::element_columns;
                 element++) {
                uint block = block_start + block_column +
                    key_index * 8 + element;
                uint length = block < blocks ?
                    min(64u, rows - block * 64u) : 0u;
                scores.at(0, key_index)[element] *= float(length);
            }
        }
    }
    float tile_sum[ScoreTile::rows_per_thread] = {0.0f};
    scores.template row_reduce<ltx_sol_sum_op>(tile_sum);
    LTX_SOL_UNROLL
    for (short row = 0; row < ScoreTile::rows_per_thread; row++)
        denominator[row] = denominator[row] * correction[row] +
                           tile_sum[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void ltx_sol_attention_tiled_bf16(
                         device const bfloat *query [[buffer(0)]],
                         device const bfloat *key [[buffer(1)]],
                         device const bfloat *value [[buffer(2)]],
                         device const bfloat *key_centroids [[buffer(3)]],
                         device const bfloat *value_sums [[buffer(4)]],
                         device const float *routes [[buffer(5)]],
                         device bfloat *output [[buffer(6)]],
                         constant ltx_sol_args &args [[buffer(7)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simdgroup [[simdgroup_index_in_threadgroup]],
                         uint2 group [[threadgroup_position_in_grid]]) {
    constexpr short BLOCK = 64;
    constexpr short DIMENSION = 128;
    constexpr short QUERY_LD = 128;
    constexpr short KEY_LD = 64;
    constexpr short VALUE_LD = 128;
    using Fragment = ltx_sol_mma_fragment<float>;
    using QueryLoader = ltx_sol_block_loader<
        bfloat, BLOCK, DIMENSION, QUERY_LD, 1, 1, 256>;
    using KeyLoader = ltx_sol_block_loader<
        bfloat, BLOCK, DIMENSION, 1, KEY_LD, 0, 256>;
    using ValueLoader = ltx_sol_block_loader<
        bfloat, BLOCK, DIMENSION, VALUE_LD, 1, 0, 256>;

    uint query_block = group.x;
    uint head = group.y;
    if (query_block >= args.blocks || head >= args.heads ||
        args.head_dim != DIMENSION || args.blocks > 64u) return;
    uint query_start = query_block * BLOCK;
    uint query_count = min(uint(BLOCK), args.rows - query_start);
    uint head_offset = head * args.rows * DIMENSION;
    query += head_offset + query_start * DIMENSION;
    key += head_offset;
    value += head_offset;
    key_centroids += head * args.blocks * DIMENSION;
    value_sums += head * args.blocks * DIMENSION;
    routes += (head * args.blocks + query_block) * args.blocks;
    output += head_offset + query_start * DIMENSION;

    threadgroup bfloat query_shared[BLOCK * QUERY_LD];
    threadgroup bfloat key_value_shared[BLOCK * VALUE_LD];
    threadgroup bfloat *keys = key_value_shared;
    threadgroup bfloat *values = key_value_shared;

    QueryLoader query_loader(
        query, DIMENSION, query_shared, simdgroup, lane);
    if (query_count < BLOCK)
        query_loader.load_safe(short2(DIMENSION, query_count));
    else
        query_loader.load_unsafe();

    uint route_mask_lo = 0u;
    uint route_mask_hi = 0u;
    if (lane == 0u) {
        for (uint block = 0; block < args.blocks; block++) {
            uint route_bit = uint(routes[block] != 0.0f);
            if (block < 32u)
                route_mask_lo |= route_bit << block;
            else
                route_mask_hi |= route_bit << (block - 32u);
        }
    }
    route_mask_lo |= simd_shuffle_xor(route_mask_lo, ushort(16));
    route_mask_hi |= simd_shuffle_xor(route_mask_hi, ushort(16));
    route_mask_lo |= simd_shuffle_xor(route_mask_lo, ushort(8));
    route_mask_hi |= simd_shuffle_xor(route_mask_hi, ushort(8));
    route_mask_lo |= simd_shuffle_xor(route_mask_lo, ushort(4));
    route_mask_hi |= simd_shuffle_xor(route_mask_hi, ushort(4));
    route_mask_lo |= simd_shuffle_xor(route_mask_lo, ushort(2));
    route_mask_hi |= simd_shuffle_xor(route_mask_hi, ushort(2));
    route_mask_lo |= simd_shuffle_xor(route_mask_lo, ushort(1));
    route_mask_hi |= simd_shuffle_xor(route_mask_hi, ushort(1));

    ltx_sol_mma_tile<float, 1, 1, Fragment> query_tile;
    ltx_sol_mma_tile<float, 1, 8, Fragment> key_tile;
    ltx_sol_mma_tile<float, 1, 8, Fragment> score_tile;
    ltx_sol_mma_tile<float, 1, 16, Fragment> output_tile;
    output_tile.clear();
    short2 coordinate = Fragment::coordinate(lane);
    short row = coordinate.y;
    short column = coordinate.x;
    short query_row = 8 * simdgroup;
    short query_offset = (query_row + row) * QUERY_LD + column;
    short key_offset = row * KEY_LD + column;
    short value_offset = row * VALUE_LD + column;
    float score_scale = args.scale_log2;
    float maximum[1] = {-INFINITY};
    float denominator[1] = {0.0f};

    threadgroup_barrier(mem_flags::mem_threadgroup);
    KeyLoader summary_key_loader(
        key_centroids, DIMENSION, keys, simdgroup, lane);
    summary_key_loader.load_safe(short2(DIMENSION, args.blocks));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    score_tile.clear();
    LTX_SOL_UNROLL
    for (short dimension_tile = 0; dimension_tile < 16;
         dimension_tile++) {
        query_tile.template load<bfloat, 1, 1, QUERY_LD, 1>(
            &query_shared[query_offset + dimension_tile * 8]);
        key_tile.template load<bfloat, 1, 1, KEY_LD, 1>(
            &keys[key_offset + dimension_tile * 8 * KEY_LD]);
        simdgroup_barrier(mem_flags::mem_none);
        ltx_sol_tile_multiply(score_tile, query_tile, key_tile, score_tile);
    }
    LTX_SOL_UNROLL
    for (short key_index = 0; key_index < 8; key_index++) {
        LTX_SOL_UNROLL
        for (short element = 0; element < Fragment::elements; element++) {
            uint block = column + key_index * 8 + element;
            score_tile.at(0, key_index)[element] *= score_scale;
            bool routed = block < 32u ?
                ((route_mask_lo >> block) & 1u) != 0u :
                ((route_mask_hi >> (block - 32u)) & 1u) != 0u;
            if (block >= args.blocks || routed)
                score_tile.at(0, key_index)[element] = -INFINITY;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    ValueLoader summary_value_loader(
        value_sums, DIMENSION, values, simdgroup, lane);
    summary_value_loader.load_safe(short2(DIMENSION, args.blocks));
    ltx_sol_accumulate_tile<true>(
        score_tile, output_tile, values, value_offset,
        maximum, denominator, 0u, column, args.blocks, args.rows);

    for (uint block = 0; block < args.blocks; block++) {
        bool routed = block < 32u ?
            ((route_mask_lo >> block) & 1u) != 0u :
            ((route_mask_hi >> (block - 32u)) & 1u) != 0u;
        if (!routed) continue;
        uint key_start = block * BLOCK;
        uint key_count = min(uint(BLOCK), args.rows - key_start);
        KeyLoader exact_key_loader(
            key + key_start * DIMENSION, DIMENSION,
            keys, simdgroup, lane);
        ValueLoader exact_value_loader(
            value + key_start * DIMENSION, DIMENSION,
            values, simdgroup, lane);
        if (key_count < BLOCK)
            exact_key_loader.load_safe(short2(DIMENSION, key_count));
        else
            exact_key_loader.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        score_tile.clear();
        LTX_SOL_UNROLL
        for (short dimension_tile = 0; dimension_tile < 16;
             dimension_tile++) {
            query_tile.template load<bfloat, 1, 1, QUERY_LD, 1>(
                &query_shared[query_offset + dimension_tile * 8]);
            key_tile.template load<bfloat, 1, 1, KEY_LD, 1>(
                &keys[key_offset + dimension_tile * 8 * KEY_LD]);
            simdgroup_barrier(mem_flags::mem_none);
            ltx_sol_tile_multiply(
                score_tile, query_tile, key_tile, score_tile);
        }
        LTX_SOL_UNROLL
        for (short key_index = 0; key_index < 8; key_index++) {
            LTX_SOL_UNROLL
            for (short element = 0; element < Fragment::elements;
                 element++) {
                uint key_column = column + key_index * 8 + element;
                score_tile.at(0, key_index)[element] *= score_scale;
                if (key_column >= key_count)
                    score_tile.at(0, key_index)[element] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (key_count < BLOCK)
            exact_value_loader.load_safe(short2(DIMENSION, key_count));
        else
            exact_value_loader.load_unsafe();
        ltx_sol_accumulate_tile<false>(
            score_tile, output_tile, values, value_offset,
            maximum, denominator, 0u, 0, args.blocks, args.rows);
    }

    if (maximum[0] == -INFINITY) denominator[0] = 1.0f;
    float inverse_denominator[1] = {1.0f / denominator[0]};
    output_tile.template row_binary<ltx_sol_mul_op>(inverse_denominator);
    device bfloat *output_pointer =
        output + uint(query_row + row) * DIMENSION + column;
    if (uint(query_row + row) < query_count)
        output_tile.template store_safe<bfloat, 1, 1>(
            output_pointer, DIMENSION,
            short2(DIMENSION - column,
                   query_count - (query_row + row)));
}

#undef LTX_SOL_UNROLL
