#include "ltx_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int near(float actual, float expected, float tolerance) {
    float scale = fmaxf(1.0f, fabsf(expected));
    return fabsf(actual - expected) <= tolerance * scale;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16u;
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void convrot_256_cpu(const uint16_t *input, uint16_t *output,
                            unsigned rows, unsigned columns) {
    for (unsigned row = 0; row < rows; row++)
        for (unsigned group = 0; group < columns / 256u; group++) {
            float current[256];
            float next[256];
            unsigned base = row * columns + group * 256u;
            for (unsigned lane = 0; lane < 256u; lane++)
                current[lane] = bf16_to_f32(input[base + lane]);
            for (unsigned stride = 1u; stride < 256u; stride *= 4u) {
                for (unsigned lane = 0; lane < 256u; lane++) {
                    unsigned digit = (lane / stride) & 3u;
                    unsigned butterfly_base = lane - digit * stride;
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
                }
                memcpy(current, next, sizeof(current));
            }
            for (unsigned lane = 0; lane < 256u; lane++)
                output[base + lane] = f32_to_bf16(current[lane] * 0.0625f);
        }
}

static void linear_int8_dequant_bf16_cpu(
        const uint16_t *input, const int8_t *weight,
        const float *scale, const uint16_t *bias,
        uint16_t *output, unsigned rows,
        unsigned input_dim, unsigned output_dim) {
    for (unsigned row = 0; row < rows; row++)
        for (unsigned column = 0; column < output_dim; column++) {
            float sum = 0.0f;
            for (unsigned k = 0; k < input_dim; k++) {
                float dequantized = bf16_to_f32(f32_to_bf16(
                    (float)weight[column * input_dim + k] * scale[column]));
                sum = fmaf(bf16_to_f32(input[row * input_dim + k]),
                           dequantized, sum);
            }
            if (bias) sum += bf16_to_f32(bias[column]);
            output[row * output_dim + column] = f32_to_bf16(sum);
        }
}

static void linear_bf16_cpu(
        const uint16_t *input, const uint16_t *weight,
        const uint16_t *bias, uint16_t *output,
        unsigned rows, unsigned input_dim, unsigned output_dim) {
    for (unsigned row = 0; row < rows; row++)
        for (unsigned column = 0; column < output_dim; column++) {
            float sum = bias ? bf16_to_f32(bias[column]) : 0.0f;
            for (unsigned k = 0; k < input_dim; k++)
                sum = fmaf(
                    bf16_to_f32(input[row * input_dim + k]),
                    bf16_to_f32(weight[column * input_dim + k]), sum);
            output[row * output_dim + column] = f32_to_bf16(sum);
        }
}

static void silu_bf16_cpu(uint16_t *values, unsigned count) {
    for (unsigned index = 0; index < count; index++) {
        float x = bf16_to_f32(values[index]);
        values[index] = f32_to_bf16(x / (1.0f + expf(-x)));
    }
}

static void gelu_tanh_bf16_cpu(uint16_t *values, unsigned count) {
    for (unsigned index = 0; index < count; index++) {
        float x = bf16_to_f32(values[index]);
        float inner = 0.7978845608028654f *
            (x + 0.044715f * x * x * x);
        values[index] = f32_to_bf16(
            0.5f * x * (1.0f + tanhf(inner)));
    }
}

static void rms_norm_weighted_bf16_cpu(uint16_t *values,
                                        const uint16_t *weight,
                                        unsigned rows, unsigned columns,
                                        float epsilon) {
    for (unsigned row = 0; row < rows; row++) {
        double sum = 0.0;
        for (unsigned column = 0; column < columns; column++) {
            double value = bf16_to_f32(values[row * columns + column]);
            sum += value * value;
        }
        float inverse_rms = 1.0f /
            sqrtf((float)(sum / (double)columns) + epsilon);
        for (unsigned column = 0; column < columns; column++) {
            unsigned index = row * columns + column;
            values[index] = f32_to_bf16(
                bf16_to_f32(values[index]) * inverse_rms *
                bf16_to_f32(weight[column]));
        }
    }
}

static int test_elementwise(ltx_gpu *gpu, char *error, size_t error_size) {
    enum { count = 64 };
    float left[count];
    float right[count];
    float gate[count];
    float output[count];
    for (unsigned index = 0; index < count; index++) {
        left[index] = (float)index * 0.25f - 4.0f;
        right[index] = (float)(count - index) * 0.125f;
        gate[index] = (float)((index % 7u) + 1u) * 0.1f;
    }

    ltx_gpu_buffer *a = ltx_gpu_buffer_new(gpu, sizeof(left), error,
                                            error_size);
    ltx_gpu_buffer *b = ltx_gpu_buffer_new(gpu, sizeof(right), error,
                                            error_size);
    ltx_gpu_buffer *g = ltx_gpu_buffer_new(gpu, sizeof(gate), error,
                                            error_size);
    ltx_gpu_buffer *c = ltx_gpu_buffer_new(gpu, sizeof(output), error,
                                            error_size);
    int ok = a && b && g && c &&
        ltx_gpu_buffer_write(a, left, sizeof(left), error, error_size) &&
        ltx_gpu_buffer_write(b, right, sizeof(right), error, error_size) &&
        ltx_gpu_buffer_write(g, gate, sizeof(gate), error, error_size) &&
        ltx_gpu_add_f32(gpu, c, a, b, count, error, error_size) &&
        ltx_gpu_buffer_read(c, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++)
        ok = near(output[index], left[index] + right[index], 1e-6f);

    ok = ok && ltx_gpu_scale_f32(gpu, c, a, -1.75f, count, error,
                                  error_size) &&
        ltx_gpu_buffer_read(c, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++)
        ok = near(output[index], left[index] * -1.75f, 1e-6f);

    ok = ok && ltx_gpu_residual_gate_f32(gpu, c, a, b, g, count,
                                          error, error_size) &&
        ltx_gpu_buffer_read(c, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++)
        ok = near(output[index], left[index] + right[index] * gate[index],
                  2e-6f);

    ltx_gpu_buffer_free(a);
    ltx_gpu_buffer_free(b);
    ltx_gpu_buffer_free(g);
    ltx_gpu_buffer_free(c);
    if (!ok && !error[0])
        snprintf(error, error_size, "elementwise GPU parity failed");
    return ok;
}

static int test_bf16_slice_and_add(ltx_gpu *gpu, char *error,
                                   size_t error_size) {
    enum {
        rows = 3,
        input_columns = 7,
        start_column = 2,
        output_columns = 3,
        input_count = rows * input_columns,
        output_count = rows * output_columns,
    };
    uint16_t left[input_count];
    uint16_t right[input_count];
    uint16_t output[output_count];
    for (unsigned index = 0u; index < input_count; index++) {
        left[index] = f32_to_bf16((float)index * 0.125f - 0.75f);
        right[index] = f32_to_bf16(
            0.5f - (float)(index % input_columns) * 0.0625f);
    }
    ltx_gpu_buffer *left_full = ltx_gpu_buffer_new_copy(
        gpu, left, sizeof(left), error, error_size);
    ltx_gpu_buffer *right_full = ltx_gpu_buffer_new_copy(
        gpu, right, sizeof(right), error, error_size);
    ltx_gpu_buffer *left_slice = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    ltx_gpu_buffer *right_slice = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    ltx_gpu_buffer *sum = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    int ok = left_full && right_full && left_slice && right_slice && sum &&
        ltx_gpu_slice_columns_bf16(
            gpu, left_slice, left_full, rows, input_columns,
            start_column, output_columns, error, error_size) &&
        ltx_gpu_slice_columns_bf16(
            gpu, right_slice, right_full, rows, input_columns,
            start_column, output_columns, error, error_size) &&
        ltx_gpu_add_bf16(
            gpu, sum, left_slice, right_slice, output_count,
            error, error_size) &&
        ltx_gpu_buffer_read(
            sum, output, sizeof(output), error, error_size);
    for (unsigned row = 0u; ok && row < rows; row++)
        for (unsigned column = 0u; column < output_columns; column++) {
            unsigned source = row * input_columns + start_column + column;
            unsigned destination = row * output_columns + column;
            uint16_t expected = f32_to_bf16(
                bf16_to_f32(left[source]) + bf16_to_f32(right[source]));
            if (output[destination] != expected) {
                snprintf(error, error_size,
                         "BF16 slice/add mismatch at %u,%u", row, column);
                ok = 0;
                break;
            }
        }
    ltx_gpu_buffer_free(left_full);
    ltx_gpu_buffer_free(right_full);
    ltx_gpu_buffer_free(left_slice);
    ltx_gpu_buffer_free(right_slice);
    ltx_gpu_buffer_free(sum);
    return ok;
}

static int test_bf16_f16_row_partition(ltx_gpu *gpu, char *error,
                                       size_t error_size) {
    enum {
        rows = 4,
        columns = 5,
        prefix_rows = 2,
        suffix_rows = rows - prefix_rows,
        count = rows * columns,
        suffix_count = suffix_rows * columns,
    };
    uint16_t input[count];
    uint16_t joined[count];
    float sliced[suffix_count];
    for (unsigned index = 0u; index < count; index++)
        input[index] = f32_to_bf16((float)((int)index - 7) * 0.5f);

    ltx_gpu_buffer *source = ltx_gpu_buffer_new_copy(
        gpu, input, sizeof(input), error, error_size);
    ltx_gpu_buffer *suffix_f16 = ltx_gpu_buffer_new(
        gpu, suffix_count * sizeof(uint16_t), error, error_size);
    ltx_gpu_buffer *suffix_f32 = ltx_gpu_buffer_new(
        gpu, sizeof(sliced), error, error_size);
    ltx_gpu_buffer *output = ltx_gpu_buffer_new(
        gpu, sizeof(joined), error, error_size);
    int ok = source && suffix_f16 && suffix_f32 && output &&
        ltx_gpu_slice_rows_bf16_f16(
            gpu, suffix_f16, source, rows, columns,
            prefix_rows, suffix_rows, error, error_size) &&
        ltx_gpu_cast_f16_f32(
            gpu, suffix_f32, suffix_f16, suffix_count,
            error, error_size) &&
        ltx_gpu_buffer_read(
            suffix_f32, sliced, sizeof(sliced), error, error_size);
    for (unsigned index = 0u; ok && index < suffix_count; index++) {
        float expected = bf16_to_f32(
            input[prefix_rows * columns + index]);
        if (sliced[index] != expected) {
            snprintf(error, error_size,
                     "BF16/F16 row-slice mismatch at %u", index);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_concat_rows_bf16_f16(
        gpu, output, source, suffix_f16,
        prefix_rows, suffix_rows, columns, error, error_size) &&
        ltx_gpu_buffer_read(
            output, joined, sizeof(joined), error, error_size);
    for (unsigned index = 0u; ok && index < count; index++) {
        if (joined[index] != input[index]) {
            snprintf(error, error_size,
                     "BF16/F16 row-concat mismatch at %u", index);
            ok = 0;
        }
    }

    ltx_gpu_buffer_free(source);
    ltx_gpu_buffer_free(suffix_f16);
    ltx_gpu_buffer_free(suffix_f32);
    ltx_gpu_buffer_free(output);
    return ok;
}

static int test_diffusion_steps(ltx_gpu *gpu, char *error,
                                size_t error_size) {
    enum { count = 17 };
    uint16_t sample[count];
    uint16_t velocity[count];
    uint16_t denoised[count];
    uint16_t noise_bf16[count];
    uint16_t output[count];
    float noise_f32[count];
    for (unsigned index = 0; index < count; index++) {
        sample[index] = f32_to_bf16(
            sinf((float)index * 0.31f) * 1.25f);
        velocity[index] = f32_to_bf16(
            cosf((float)index * 0.17f) * 0.75f);
        noise_f32[index] =
            sinf((float)index * 0.23f + 0.4f) * 0.9f;
        noise_bf16[index] = f32_to_bf16(noise_f32[index]);
    }

    ltx_gpu_buffer *x = ltx_gpu_buffer_new_copy(
        gpu, sample, sizeof(sample), error, error_size);
    ltx_gpu_buffer *v = ltx_gpu_buffer_new_copy(
        gpu, velocity, sizeof(velocity), error, error_size);
    ltx_gpu_buffer *n16 = ltx_gpu_buffer_new_copy(
        gpu, noise_bf16, sizeof(noise_bf16), error, error_size);
    ltx_gpu_buffer *n32 = ltx_gpu_buffer_new_copy(
        gpu, noise_f32, sizeof(noise_f32), error, error_size);
    ltx_gpu_buffer *x0 = ltx_gpu_buffer_new(
        gpu, sizeof(denoised), error, error_size);
    ltx_gpu_buffer *y = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    int ok = x && v && n16 && n32 && x0 && y;
    const float sigma = 0.9f;
    const float sigma_next = 0.6f;

    ok = ok && ltx_gpu_velocity_to_denoised_bf16(
        gpu, x0, x, v, count, sigma, error, error_size) &&
        ltx_gpu_buffer_read(x0, denoised, sizeof(denoised),
                            error, error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        float expected = fmaf(
            -sigma, bf16_to_f32(velocity[index]),
            bf16_to_f32(sample[index]));
        if (denoised[index] != f32_to_bf16(expected)) {
            snprintf(error, error_size,
                     "GPU velocity-to-denoised mismatch at %u", index);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_euler_step_bf16(
        gpu, y, x, x0, count, sigma, sigma_next,
        error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        float sample_value = bf16_to_f32(sample[index]);
        float denoised_value = bf16_to_f32(denoised[index]);
        float recovered_velocity =
            (sample_value - denoised_value) / sigma;
        float expected = fmaf(
            recovered_velocity, sigma_next - sigma, sample_value);
        if (output[index] != f32_to_bf16(expected)) {
            snprintf(error, error_size,
                     "GPU Euler mismatch at %u", index);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_euler_ancestral_step_bf16(
        gpu, y, x, x0, n32, count, sigma, sigma_next,
        1.0f, 1.0f, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    float downstep_ratio =
        1.0f + (sigma_next / sigma - 1.0f);
    float sigma_down = sigma_next * downstep_ratio;
    float ratio = sigma_down / sigma;
    float alpha_next = 1.0f - sigma_next;
    float alpha_down = 1.0f - sigma_down;
    float variance = sigma_next * sigma_next -
        sigma_down * sigma_down * alpha_next * alpha_next /
            (alpha_down * alpha_down);
    float renoise = sqrtf(fmaxf(variance, 0.0f));
    for (unsigned index = 0; ok && index < count; index++) {
        float deterministic = fmaf(
            ratio, bf16_to_f32(sample[index]),
            (1.0f - ratio) * bf16_to_f32(denoised[index]));
        float expected = fmaf(noise_f32[index], renoise,
                              alpha_next / alpha_down * deterministic);
        if (output[index] != f32_to_bf16(expected)) {
            snprintf(error, error_size,
                     "GPU ancestral Euler mismatch at %u", index);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_euler_ancestral_step_bf16(
        gpu, y, x, x0, NULL, count, 0.421875f, 0.0f,
        1.0f, 1.0f, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        if (output[index] != denoised[index]) {
            snprintf(error, error_size,
                     "GPU terminal ancestral mismatch at %u", index);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_renoise_bf16(
        gpu, y, x0, n16, count, 0.25f, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        float expected = fmaf(
            bf16_to_f32(noise_bf16[index]), 0.25f,
            bf16_to_f32(denoised[index]) * 0.75f);
        if (output[index] != f32_to_bf16(expected)) {
            snprintf(error, error_size,
                     "GPU latent renoise mismatch at %u", index);
            ok = 0;
        }
    }

    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(v);
    ltx_gpu_buffer_free(n16);
    ltx_gpu_buffer_free(n32);
    ltx_gpu_buffer_free(x0);
    ltx_gpu_buffer_free(y);
    if (!ok && !error[0])
        snprintf(error, error_size, "GPU diffusion-step parity failed");
    return ok;
}

static int test_split_conditioning(ltx_gpu *gpu, char *error,
                                   size_t error_size) {
    enum {
        rows = 4,
        columns = 17,
        count = rows * columns,
        conditioned_rows = 2,
        conditioned_count = conditioned_rows * columns,
    };
    uint16_t sample[count];
    uint16_t velocity[count];
    uint16_t clean[conditioned_count];
    uint16_t generated_scale[columns];
    uint16_t generated_shift[columns];
    uint16_t conditioned_scale[columns];
    uint16_t conditioned_shift[columns];
    uint16_t generated_gate[columns];
    uint16_t conditioned_gate[columns];
    uint16_t output[count];
    for (unsigned index = 0; index < count; index++) {
        sample[index] = f32_to_bf16(
            sinf((float)index * 0.13f) * 1.25f + 0.125f);
        velocity[index] = f32_to_bf16(
            cosf((float)index * 0.19f) * 0.625f - 0.0625f);
        if (index < conditioned_count)
            clean[index] = f32_to_bf16(
                sinf((float)index * 0.07f + 0.3f) * 0.75f);
    }
    for (unsigned column = 0; column < columns; column++) {
        generated_scale[column] = f32_to_bf16(
            (float)((int)(column % 7u) - 3) * 0.03125f);
        generated_shift[column] = f32_to_bf16(
            (float)((int)(column % 5u) - 2) * 0.0234375f);
        conditioned_scale[column] = f32_to_bf16(
            (float)((int)(column % 9u) - 4) * -0.015625f);
        conditioned_shift[column] = f32_to_bf16(
            (float)((int)(column % 6u) - 3) * 0.01953125f);
        generated_gate[column] = f32_to_bf16(
            0.2f + (float)(column % 4u) * 0.0625f);
        conditioned_gate[column] = f32_to_bf16(
            -0.1f + (float)(column % 5u) * 0.046875f);
    }

#define LTX_SPLIT_BUFFER(NAME, DATA) \
    ltx_gpu_buffer *NAME = ltx_gpu_buffer_new_copy( \
        gpu, (DATA), sizeof(DATA), error, error_size)
    LTX_SPLIT_BUFFER(x, sample);
    LTX_SPLIT_BUFFER(v, velocity);
    LTX_SPLIT_BUFFER(c, clean);
    LTX_SPLIT_BUFFER(gs, generated_scale);
    LTX_SPLIT_BUFFER(gh, generated_shift);
    LTX_SPLIT_BUFFER(cs, conditioned_scale);
    LTX_SPLIT_BUFFER(ch, conditioned_shift);
    LTX_SPLIT_BUFFER(gg, generated_gate);
    LTX_SPLIT_BUFFER(cg, conditioned_gate);
#undef LTX_SPLIT_BUFFER
    ltx_gpu_buffer *y = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    int ok = x && v && c && gs && gh && cs && ch && gg && cg && y;
    const float generated_sigma = 0.8f;
    const float conditioned_sigma = 0.2f;

    ok = ok && ltx_gpu_velocity_to_denoised_bf16_split(
        gpu, y, x, v, rows, columns, conditioned_rows,
        generated_sigma, conditioned_sigma, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        unsigned row = index / columns;
        float sigma = row < conditioned_rows ?
            conditioned_sigma : generated_sigma;
        float expected = fmaf(
            -sigma, bf16_to_f32(velocity[index]),
            bf16_to_f32(sample[index]));
        if (output[index] != f32_to_bf16(expected)) {
            snprintf(error, error_size,
                     "split velocity mismatch at %u", index);
            ok = 0;
        }
    }

    const float strengths[] = {0.0f, 0.5f, 1.0f};
    for (unsigned strength_index = 0;
         ok && strength_index < sizeof(strengths) / sizeof(strengths[0]);
         strength_index++) {
        float mask = 1.0f - strengths[strength_index];
        ok = ltx_gpu_buffer_write(
                y, sample, sizeof(sample), error, error_size) &&
            ltx_gpu_condition_prefix_bf16(
                gpu, y, c, conditioned_count, mask,
                error, error_size) &&
            ltx_gpu_buffer_read(
                y, output, sizeof(output), error, error_size);
        for (unsigned index = 0; ok && index < count; index++) {
            uint16_t expected = sample[index];
            if (index < conditioned_count)
                expected = f32_to_bf16(fmaf(
                    bf16_to_f32(sample[index]), mask,
                    bf16_to_f32(clean[index]) * (1.0f - mask)));
            if (output[index] != expected) {
                snprintf(error, error_size,
                         "conditioning blend mismatch at %u strength %.2f",
                         index, strengths[strength_index]);
                ok = 0;
            }
        }
    }

    ok = ok && ltx_gpu_affine_bf16_split(
        gpu, y, x, gs, gh, cs, ch, rows, columns,
        conditioned_rows, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        unsigned row = index / columns;
        unsigned column = index % columns;
        const uint16_t *scale = row < conditioned_rows ?
            conditioned_scale : generated_scale;
        const uint16_t *shift = row < conditioned_rows ?
            conditioned_shift : generated_shift;
        float expected = fmaf(
            bf16_to_f32(sample[index]),
            1.0f + bf16_to_f32(scale[column]),
            bf16_to_f32(shift[column]));
        if (output[index] != f32_to_bf16(expected)) {
            snprintf(error, error_size, "split affine mismatch at %u", index);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_residual_gate_bf16_split(
        gpu, y, x, v, gg, cg, rows, columns,
        conditioned_rows, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        unsigned row = index / columns;
        unsigned column = index % columns;
        const uint16_t *gate = row < conditioned_rows ?
            conditioned_gate : generated_gate;
        float expected = fmaf(
            bf16_to_f32(velocity[index]), bf16_to_f32(gate[column]),
            bf16_to_f32(sample[index]));
        if (output[index] != f32_to_bf16(expected)) {
            snprintf(error, error_size,
                     "split residual gate mismatch at %u", index);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_adaln_bf16_split(
        gpu, y, x, gs, gh, cs, ch, rows, columns,
        conditioned_rows, 1e-6f, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned row = 0; ok && row < rows; row++) {
        double square_sum = 0.0;
        for (unsigned column = 0; column < columns; column++) {
            float value = bf16_to_f32(sample[row * columns + column]);
            square_sum += (double)value * value;
        }
        float inverse_rms = 1.0f /
            sqrtf((float)(square_sum / (double)columns) + 1e-6f);
        const uint16_t *scale = row < conditioned_rows ?
            conditioned_scale : generated_scale;
        const uint16_t *shift = row < conditioned_rows ?
            conditioned_shift : generated_shift;
        for (unsigned column = 0; ok && column < columns; column++) {
            unsigned index = row * columns + column;
            float expected = fmaf(
                bf16_to_f32(sample[index]) * inverse_rms,
                1.0f + bf16_to_f32(scale[column]),
                bf16_to_f32(shift[column]));
            if (!near(bf16_to_f32(output[index]),
                      bf16_to_f32(f32_to_bf16(expected)), 8e-3f)) {
                snprintf(error, error_size,
                         "split AdaLN mismatch at %u", index);
                ok = 0;
            }
        }
    }

    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(v);
    ltx_gpu_buffer_free(c);
    ltx_gpu_buffer_free(gs);
    ltx_gpu_buffer_free(gh);
    ltx_gpu_buffer_free(cs);
    ltx_gpu_buffer_free(ch);
    ltx_gpu_buffer_free(gg);
    ltx_gpu_buffer_free(cg);
    ltx_gpu_buffer_free(y);
    return ok;
}

static int test_adaln_and_residual_bf16(ltx_gpu *gpu, char *error,
                                        size_t error_size) {
    enum { rows = 3, columns = 17, count = rows * columns };
    uint16_t input[count];
    uint16_t residual[count];
    uint16_t branch[count];
    uint16_t scale[count];
    uint16_t shift[count];
    uint16_t gate[count];
    uint16_t output[count];
    for (unsigned index = 0; index < count; index++) {
        input[index] = f32_to_bf16(
            sinf((float)index * 0.17f) * 1.75f +
            (float)(index % 5u) * 0.0625f);
        residual[index] = f32_to_bf16(
            cosf((float)index * 0.11f) * 0.75f);
        branch[index] = f32_to_bf16(
            sinf((float)index * 0.07f) * 0.5f);
        scale[index] = f32_to_bf16(
            (float)((int)(index % 9u) - 4) * 0.015625f);
        shift[index] = f32_to_bf16(
            (float)((int)(index % 7u) - 3) * 0.0234375f);
        gate[index] = f32_to_bf16(
            0.25f + (float)(index % 11u) * 0.03125f);
    }

#define LTX_NEW_BF16_BUFFER(NAME, DATA) \
    ltx_gpu_buffer *NAME = ltx_gpu_buffer_new_copy( \
        gpu, (DATA), sizeof(DATA), error, error_size)
    LTX_NEW_BF16_BUFFER(x, input);
    LTX_NEW_BF16_BUFFER(r, residual);
    LTX_NEW_BF16_BUFFER(b, branch);
    LTX_NEW_BF16_BUFFER(a, input);
    LTX_NEW_BF16_BUFFER(s, scale);
    LTX_NEW_BF16_BUFFER(h, shift);
    LTX_NEW_BF16_BUFFER(g, gate);
#undef LTX_NEW_BF16_BUFFER
    ltx_gpu_buffer *y = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    ltx_gpu_buffer *z = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    ltx_gpu_buffer *joined = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    ltx_gpu_buffer *ane_f16 = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    int ok = x && r && b && a && s && h && g && y && z && joined && ane_f16;
    const float epsilon = 1e-6f;

    for (unsigned mode = 0; ok && mode < 2u; mode++) {
        uint32_t parameter_rows = mode ? 1u : rows;
        ok = ltx_gpu_adaln_bf16(
                gpu, y, x, s, h, rows, columns, parameter_rows,
                epsilon, error, error_size) &&
            ltx_gpu_cast_bf16_f16(
                gpu, ane_f16, y, count, error, error_size) &&
            ltx_gpu_adaln_bf16_f16(
                gpu, z, joined, x, s, h, rows, columns, parameter_rows,
                epsilon, error, error_size) &&
            ltx_gpu_buffer_read(y, output, sizeof(output),
                                error, error_size);
        uint16_t fused_bf16[count];
        uint16_t staged_f16[count];
        uint16_t fused_f16[count];
        if (ok) ok = ltx_gpu_buffer_read(
            z, fused_bf16, sizeof(fused_bf16), error, error_size) &&
            ltx_gpu_buffer_read(
                ane_f16, staged_f16, sizeof(staged_f16), error, error_size) &&
            ltx_gpu_buffer_read(
                joined, fused_f16, sizeof(fused_f16), error, error_size);
        for (unsigned index = 0; ok && index < count; index++) {
            if (fused_bf16[index] != output[index] ||
                fused_f16[index] != staged_f16[index]) {
                snprintf(error, error_size,
                         "dual-output AdaLN mismatch at %u mode %u",
                         index, mode);
                ok = 0;
            }
        }
        for (unsigned row = 0; ok && row < rows; row++) {
            double sum = 0.0;
            for (unsigned column = 0; column < columns; column++) {
                float value = bf16_to_f32(input[row * columns + column]);
                sum += (double)value * value;
            }
            float inverse_rms = 1.0f /
                sqrtf((float)(sum / (double)columns) + epsilon);
            unsigned parameter_row = mode ? 0u : row;
            for (unsigned column = 0; ok && column < columns; column++) {
                unsigned index = row * columns + column;
                unsigned parameter = parameter_row * columns + column;
                float expected = bf16_to_f32(f32_to_bf16(fmaf(
                    bf16_to_f32(input[index]) * inverse_rms,
                    1.0f + bf16_to_f32(scale[parameter]),
                    bf16_to_f32(shift[parameter]))));
                if (!near(bf16_to_f32(output[index]), expected, 8e-3f)) {
                    snprintf(error, error_size,
                             "BF16 AdaLN mismatch at %u mode %u",
                             index, mode);
                    ok = 0;
                }
            }
        }
    }

    for (unsigned mode = 0; ok && mode < 2u; mode++) {
        uint32_t gate_rows = mode ? 1u : rows;
        ok = ltx_gpu_residual_gate_bf16(
                gpu, y, r, b, g, rows, columns, gate_rows,
                error, error_size) &&
            ltx_gpu_buffer_read(y, output, sizeof(output),
                                error, error_size);
        for (unsigned row = 0; ok && row < rows; row++) {
            unsigned gate_row = mode ? 0u : row;
            for (unsigned column = 0; ok && column < columns; column++) {
                unsigned index = row * columns + column;
                unsigned gate_index = gate_row * columns + column;
                float expected = bf16_to_f32(f32_to_bf16(fmaf(
                    bf16_to_f32(branch[index]),
                    bf16_to_f32(gate[gate_index]),
                    bf16_to_f32(residual[index]))));
                if (bf16_to_f32(output[index]) != expected) {
                    snprintf(error, error_size,
                             "BF16 residual gate mismatch at %u mode %u",
                             index, mode);
                    ok = 0;
                }
            }
        }
    }

    ok = ok && ltx_gpu_cast_bf16_f16(
        gpu, ane_f16, a, count, error, error_size);
    for (unsigned mode = 0; ok && mode < 2u; mode++) {
        uint32_t gate_rows = mode ? 1u : rows;
        ok = ltx_gpu_join_bf16_f16(
                gpu, joined, b, ane_f16, count, error, error_size) &&
            ltx_gpu_residual_gate_bf16(
                gpu, y, r, joined, g, rows, columns, gate_rows,
                error, error_size) &&
            ltx_gpu_join_residual_gate_bf16_f16(
                gpu, z, r, b, ane_f16, g, rows, columns, gate_rows,
                error, error_size) &&
            ltx_gpu_buffer_read(y, output, sizeof(output),
                                error, error_size);
        uint16_t fused[count];
        if (ok) ok = ltx_gpu_buffer_read(
            z, fused, sizeof(fused), error, error_size);
        for (unsigned index = 0; ok && index < count; index++) {
            if (fused[index] != output[index]) {
                snprintf(error, error_size,
                         "fused BF16/F16 join-residual mismatch at %u mode %u",
                         index, mode);
                ok = 0;
            }
        }
    }

    for (unsigned mode = 0; ok && mode < 2u; mode++) {
        uint32_t parameter_rows = mode ? 1u : rows;
        ok = ltx_gpu_affine_bf16(
                gpu, y, x, s, h, rows, columns, parameter_rows,
                error, error_size) &&
            ltx_gpu_buffer_read(y, output, sizeof(output),
                                error, error_size);
        for (unsigned row = 0; ok && row < rows; row++) {
            unsigned parameter_row = mode ? 0u : row;
            for (unsigned column = 0; ok && column < columns; column++) {
                unsigned index = row * columns + column;
                unsigned parameter = parameter_row * columns + column;
                float expected = bf16_to_f32(f32_to_bf16(fmaf(
                    bf16_to_f32(input[index]),
                    1.0f + bf16_to_f32(scale[parameter]),
                    bf16_to_f32(shift[parameter]))));
                if (bf16_to_f32(output[index]) != expected) {
                    snprintf(error, error_size,
                             "BF16 affine mismatch at %u mode %u",
                             index, mode);
                    ok = 0;
                }
            }
        }
    }

    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(r);
    ltx_gpu_buffer_free(b);
    ltx_gpu_buffer_free(a);
    ltx_gpu_buffer_free(s);
    ltx_gpu_buffer_free(h);
    ltx_gpu_buffer_free(g);
    ltx_gpu_buffer_free(y);
    ltx_gpu_buffer_free(z);
    ltx_gpu_buffer_free(joined);
    ltx_gpu_buffer_free(ane_f16);
    if (!ok && !error[0])
        snprintf(error, error_size, "BF16 AdaLN/residual parity failed");
    return ok;
}

static int test_cast_and_gelu(ltx_gpu *gpu, char *error,
                              size_t error_size) {
    static const float input[] = {
        -16.0f, -4.0f, -2.0f, -1.0f, -0.5f, -0.0f,
        0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 16.0f,
    };
    enum { count = sizeof(input) / sizeof(input[0]) };
    float output[count];
    ltx_gpu_buffer *f32_input = ltx_gpu_buffer_new(
        gpu, sizeof(input), error, error_size);
    ltx_gpu_buffer *f16 = ltx_gpu_buffer_new(
        gpu, count * sizeof(uint16_t), error, error_size);
    ltx_gpu_buffer *f32_output = ltx_gpu_buffer_new(
        gpu, sizeof(output), error, error_size);
    int ok = f32_input && f16 && f32_output &&
        ltx_gpu_buffer_write(f32_input, input, sizeof(input), error,
                             error_size) &&
        ltx_gpu_cast_f32_f16(gpu, f16, f32_input, count, error,
                             error_size) &&
        ltx_gpu_cast_f16_f32(gpu, f32_output, f16, count, error,
                             error_size) &&
        ltx_gpu_buffer_read(f32_output, output, sizeof(output), error,
                            error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        if (output[index] != input[index]) {
            snprintf(error, error_size,
                     "F16 round-trip mismatch at %u: %.9g != %.9g",
                     index, output[index], input[index]);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_cast_f32_bf16(gpu, f16, f32_input, count, error,
                                      error_size) &&
        ltx_gpu_cast_bf16_f32(gpu, f32_output, f16, count, error,
                              error_size) &&
        ltx_gpu_buffer_read(f32_output, output, sizeof(output), error,
                            error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        float expected = bf16_to_f32(f32_to_bf16(input[index]));
        if (output[index] != expected) {
            snprintf(error, error_size,
                     "BF16 round-trip mismatch at %u: %.9g != %.9g",
                     index, output[index], expected);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_gelu_tanh_f32(gpu, f32_output, f32_input, count,
                                      error, error_size) &&
        ltx_gpu_buffer_read(f32_output, output, sizeof(output), error,
                            error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        float x = input[index];
        float inner = 0.7978845608028654f *
            (x + 0.044715f * x * x * x);
        float expected = 0.5f * x * (1.0f + tanhf(inner));
        if (!near(output[index], expected, 3e-6f)) {
            snprintf(error, error_size,
                     "GELU mismatch at %u: %.9g != %.9g",
                     index, output[index], expected);
            ok = 0;
        }
    }


    ok = ok && ltx_gpu_cast_f32_bf16(gpu, f16, f32_input, count, error,
                                      error_size) &&
        ltx_gpu_gelu_tanh_bf16(gpu, f16, f16, count, error, error_size) &&
        ltx_gpu_cast_bf16_f32(gpu, f32_output, f16, count, error,
                              error_size) &&
        ltx_gpu_buffer_read(f32_output, output, sizeof(output), error,
                            error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        float x = bf16_to_f32(f32_to_bf16(input[index]));
        float inner = 0.7978845608028654f *
            (x + 0.044715f * x * x * x);
        float gelu = 0.5f * x * (1.0f + tanhf(inner));
        float expected = bf16_to_f32(f32_to_bf16(gelu));
        if (!near(output[index], expected, 8e-3f)) {
            snprintf(error, error_size,
                     "BF16 GELU mismatch at %u: %.9g != %.9g",
                     index, output[index], expected);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_cast_f32_bf16(gpu, f16, f32_input, count, error,
                                      error_size) &&
        ltx_gpu_silu_bf16(gpu, f16, f16, count, error, error_size) &&
        ltx_gpu_cast_bf16_f32(gpu, f32_output, f16, count, error,
                              error_size) &&
        ltx_gpu_buffer_read(f32_output, output, sizeof(output), error,
                            error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        float x = bf16_to_f32(f32_to_bf16(input[index]));
        float silu = x / (1.0f + expf(-x));
        float expected = bf16_to_f32(f32_to_bf16(silu));
        if (!near(output[index], expected, 8e-3f)) {
            snprintf(error, error_size,
                     "BF16 SiLU mismatch at %u: %.9g != %.9g",
                     index, output[index], expected);
            ok = 0;
        }
    }

    ltx_gpu_buffer_free(f32_input);
    ltx_gpu_buffer_free(f16);
    ltx_gpu_buffer_free(f32_output);
    if (!ok && !error[0])
        snprintf(error, error_size, "cast/GELU GPU parity failed");
    return ok;
}

static int test_rms_norm(ltx_gpu *gpu, char *error, size_t error_size) {
    enum { rows = 3, columns = 17, count = rows * columns };
    float input[count];
    float weight[columns];
    float output[count];
    float expected[count];
    for (unsigned column = 0; column < columns; column++)
        weight[column] = 0.75f + (float)column * 0.03125f;
    for (unsigned index = 0; index < count; index++)
        input[index] = sinf((float)index * 0.37f) * 3.0f +
            (float)(index % 5u) * 0.125f;

    ltx_gpu_buffer *x = ltx_gpu_buffer_new(gpu, sizeof(input), error,
                                            error_size);
    ltx_gpu_buffer *w = ltx_gpu_buffer_new(gpu, sizeof(weight), error,
                                            error_size);
    ltx_gpu_buffer *y = ltx_gpu_buffer_new(gpu, sizeof(output), error,
                                            error_size);
    ltx_gpu_buffer *bx = ltx_gpu_buffer_new(
        gpu, count * sizeof(uint16_t), error, error_size);
    ltx_gpu_buffer *bw = ltx_gpu_buffer_new(
        gpu, columns * sizeof(uint16_t), error, error_size);
    ltx_gpu_buffer *by = ltx_gpu_buffer_new(
        gpu, count * sizeof(uint16_t), error, error_size);
    int ok = x && w && y && bx && bw && by &&
        ltx_gpu_buffer_write(x, input, sizeof(input), error, error_size) &&
        ltx_gpu_buffer_write(w, weight, sizeof(weight), error, error_size);

    const float epsilon = 1e-6f;
    for (unsigned row = 0; row < rows; row++) {
        double sum = 0.0;
        for (unsigned column = 0; column < columns; column++) {
            double value = input[row * columns + column];
            sum += value * value;
        }
        float inverse_rms = 1.0f /
            sqrtf((float)(sum / (double)columns) + epsilon);
        for (unsigned column = 0; column < columns; column++)
            expected[row * columns + column] =
                input[row * columns + column] * inverse_rms;
    }

    ok = ok && ltx_gpu_rms_norm_f32(gpu, y, x, rows, columns, epsilon,
                                     error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned index = 0; ok && index < count; index++)
        ok = near(output[index], expected[index], 2e-5f);

    ok = ok && ltx_gpu_rms_norm_weighted_f32(
        gpu, y, x, w, rows, columns, epsilon, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned row = 0; ok && row < rows; row++)
        for (unsigned column = 0; ok && column < columns; column++) {
            unsigned index = row * columns + column;
            ok = near(output[index], expected[index] * weight[column],
                      2e-5f);
        }


    ok = ok && ltx_gpu_cast_f32_bf16(gpu, bx, x, count, error,
                                      error_size) &&
        ltx_gpu_cast_f32_bf16(gpu, bw, w, columns, error, error_size) &&
        ltx_gpu_rms_norm_bf16(gpu, by, bx, rows, columns, epsilon,
                              error, error_size) &&
        ltx_gpu_cast_bf16_f32(gpu, y, by, count, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned row = 0; ok && row < rows; row++) {
        double sum = 0.0;
        for (unsigned column = 0; column < columns; column++) {
            float value = bf16_to_f32(f32_to_bf16(
                input[row * columns + column]));
            sum += (double)value * value;
        }
        float inverse_rms = 1.0f /
            sqrtf((float)(sum / (double)columns) + epsilon);
        for (unsigned column = 0; ok && column < columns; column++) {
            unsigned index = row * columns + column;
            float value = bf16_to_f32(f32_to_bf16(input[index]));
            float bf16_expected = bf16_to_f32(
                f32_to_bf16(value * inverse_rms));
            ok = near(output[index], bf16_expected, 8e-3f);
        }
    }

    ok = ok && ltx_gpu_rms_norm_weighted_bf16(
        gpu, by, bx, bw, rows, columns, epsilon, error, error_size) &&
        ltx_gpu_cast_bf16_f32(gpu, y, by, count, error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned row = 0; ok && row < rows; row++) {
        double sum = 0.0;
        for (unsigned column = 0; column < columns; column++) {
            float value = bf16_to_f32(f32_to_bf16(
                input[row * columns + column]));
            sum += (double)value * value;
        }
        float inverse_rms = 1.0f /
            sqrtf((float)(sum / (double)columns) + epsilon);
        for (unsigned column = 0; ok && column < columns; column++) {
            unsigned index = row * columns + column;
            float value = bf16_to_f32(f32_to_bf16(input[index]));
            float bf16_weight = bf16_to_f32(f32_to_bf16(weight[column]));
            float bf16_expected = bf16_to_f32(
                f32_to_bf16(value * inverse_rms * bf16_weight));
            ok = near(output[index], bf16_expected, 8e-3f);
        }
    }

    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(w);
    ltx_gpu_buffer_free(y);
    ltx_gpu_buffer_free(bx);
    ltx_gpu_buffer_free(bw);
    ltx_gpu_buffer_free(by);
    if (!ok && !error[0])
        snprintf(error, error_size, "RMSNorm GPU parity failed");
    return ok;
}

static int test_linear(ltx_gpu *gpu, char *error, size_t error_size) {
    enum { rows = 19, input_dim = 128, output_dim = 128 };
    enum {
        input_count = rows * input_dim,
        weight_count = output_dim * input_dim,
        output_count = rows * output_dim,
    };
    float input[input_count];
    float weight[weight_count];
    float bias[output_dim];
    float output[output_count];
    uint16_t bf16_input[input_count];
    uint16_t bf16_weight[weight_count];
    uint16_t bf16_bias[output_dim];
    uint16_t bf16_output[output_count];
    for (unsigned index = 0; index < input_count; index++) {
        input[index] = sinf((float)index * 0.113f) * 0.5f;
        bf16_input[index] = f32_to_bf16(input[index]);
    }
    for (unsigned index = 0; index < weight_count; index++) {
        weight[index] = cosf((float)index * 0.071f) * 0.25f;
        bf16_weight[index] = f32_to_bf16(weight[index]);
    }
    for (unsigned index = 0; index < output_dim; index++) {
        bias[index] = (float)index * 0.03125f - 0.125f;
        bf16_bias[index] = f32_to_bf16(bias[index]);
    }

    ltx_gpu_buffer *x = ltx_gpu_buffer_new(gpu, sizeof(input), error,
                                            error_size);
    ltx_gpu_buffer *w = ltx_gpu_buffer_new(gpu, sizeof(weight), error,
                                            error_size);
    ltx_gpu_buffer *b = ltx_gpu_buffer_new(gpu, sizeof(bias), error,
                                            error_size);
    ltx_gpu_buffer *y = ltx_gpu_buffer_new(gpu, sizeof(output), error,
                                            error_size);
    ltx_gpu_buffer *bx = ltx_gpu_buffer_new(gpu, sizeof(bf16_input), error,
                                             error_size);
    ltx_gpu_buffer *bw = ltx_gpu_buffer_new(gpu, sizeof(bf16_weight), error,
                                             error_size);
    ltx_gpu_buffer *bb = ltx_gpu_buffer_new(gpu, sizeof(bf16_bias), error,
                                             error_size);
    ltx_gpu_buffer *by = ltx_gpu_buffer_new(gpu, sizeof(bf16_output), error,
                                             error_size);
    int ok = x && w && b && y && bx && bw && bb && by &&
        ltx_gpu_buffer_write(x, input, sizeof(input), error, error_size) &&
        ltx_gpu_buffer_write(w, weight, sizeof(weight), error, error_size) &&
        ltx_gpu_buffer_write(b, bias, sizeof(bias), error, error_size) &&
        ltx_gpu_linear_f32(gpu, y, x, w, b, rows, input_dim, output_dim,
                           error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned row = 0; ok && row < rows; row++)
        for (unsigned column = 0; ok && column < output_dim; column++) {
            float expected = bias[column];
            for (unsigned k = 0; k < input_dim; k++)
                expected = fmaf(input[row * input_dim + k],
                                weight[column * input_dim + k], expected);
            if (!near(output[row * output_dim + column], expected, 2e-5f)) {
                snprintf(error, error_size,
                         "F32 linear mismatch at (%u,%u): %.9g != %.9g",
                         row, column, output[row * output_dim + column],
                         expected);
                ok = 0;
            }
        }

    ok = ok &&
        ltx_gpu_buffer_write(bx, bf16_input, sizeof(bf16_input), error,
                             error_size) &&
        ltx_gpu_buffer_write(bw, bf16_weight, sizeof(bf16_weight), error,
                             error_size) &&
        ltx_gpu_buffer_write(bb, bf16_bias, sizeof(bf16_bias), error,
                             error_size) &&
        ltx_gpu_linear_bf16(gpu, by, bx, bw, bb, rows, input_dim,
                            output_dim, error, error_size) &&
        ltx_gpu_buffer_read(by, bf16_output, sizeof(bf16_output), error,
                            error_size);
    for (unsigned row = 0; ok && row < rows; row++)
        for (unsigned column = 0; ok && column < output_dim; column++) {
            float expected = bf16_to_f32(bf16_bias[column]);
            for (unsigned k = 0; k < input_dim; k++)
                expected = fmaf(
                    bf16_to_f32(bf16_input[row * input_dim + k]),
                    bf16_to_f32(bf16_weight[column * input_dim + k]),
                    expected);
            float actual = bf16_to_f32(
                bf16_output[row * output_dim + column]);
            float rounded = bf16_to_f32(f32_to_bf16(expected));
            if (!near(actual, rounded, 8e-3f)) {
                snprintf(error, error_size,
                         "BF16 linear mismatch at (%u,%u): %.9g != %.9g",
                         row, column, actual, rounded);
                ok = 0;
            }
        }

    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(w);
    ltx_gpu_buffer_free(b);
    ltx_gpu_buffer_free(y);
    ltx_gpu_buffer_free(bx);
    ltx_gpu_buffer_free(bw);
    ltx_gpu_buffer_free(bb);
    ltx_gpu_buffer_free(by);
    if (!ok && !error[0])
        snprintf(error, error_size, "linear GPU parity failed");
    return ok;
}

static int test_output_head(ltx_gpu *gpu, char *error,
                            size_t error_size) {
    enum { rows = 5, hidden_dim = 128, output_dim = 128 };
    enum {
        input_count = rows * hidden_dim,
        weight_count = output_dim * hidden_dim,
        output_count = rows * output_dim,
    };
    uint16_t input[input_count];
    uint16_t embedded[hidden_dim];
    uint16_t shift_table[hidden_dim];
    uint16_t scale_table[hidden_dim];
    uint16_t weight[weight_count];
    uint16_t bias[output_dim];
    uint16_t modulated[input_count];
    uint16_t actual_modulated[input_count];
    uint16_t expected[output_count];
    uint16_t actual[output_count];
    for (unsigned index = 0; index < input_count; index++)
        input[index] = f32_to_bf16(
            sinf((float)index * 0.071f) * 0.75f +
            (float)(index % 7u) * 0.015625f);
    for (unsigned index = 0; index < hidden_dim; index++) {
        embedded[index] = f32_to_bf16(
            cosf((float)index * 0.13f) * 0.03125f);
        shift_table[index] = f32_to_bf16(
            (float)((int)(index % 5u) - 2) * 0.0078125f);
        scale_table[index] = f32_to_bf16(
            (float)((int)(index % 7u) - 3) * 0.005859375f);
    }
    for (unsigned index = 0; index < weight_count; index++)
        weight[index] = f32_to_bf16(
            cosf((float)index * 0.023f) * 0.0625f);
    for (unsigned index = 0; index < output_dim; index++)
        bias[index] = f32_to_bf16(
            (float)((int)(index % 5u) - 2) * 0.015625f);

    for (unsigned row = 0; row < rows; row++) {
        double sum = 0.0;
        for (unsigned column = 0; column < hidden_dim; column++)
            sum += bf16_to_f32(input[row * hidden_dim + column]);
        float mean = (float)(sum / (double)hidden_dim);
        double square_sum = 0.0;
        for (unsigned column = 0; column < hidden_dim; column++) {
            double centered =
                bf16_to_f32(input[row * hidden_dim + column]) - mean;
            square_sum += centered * centered;
        }
        float inverse_std = 1.0f /
            sqrtf((float)(square_sum / (double)hidden_dim) + 1e-6f);
        for (unsigned column = 0; column < hidden_dim; column++) {
            float normalized =
                (bf16_to_f32(input[row * hidden_dim + column]) - mean) *
                inverse_std;
            float shift = bf16_to_f32(shift_table[column]) +
                bf16_to_f32(embedded[column]);
            float scale = bf16_to_f32(scale_table[column]) +
                bf16_to_f32(embedded[column]);
            modulated[row * hidden_dim + column] = f32_to_bf16(
                fmaf(normalized, 1.0f + scale, shift));
        }
    }
    linear_bf16_cpu(modulated, weight, bias, expected,
                    rows, hidden_dim, output_dim);

#define LTX_NEW_OUTPUT_BUFFER(NAME, DATA) \
    ltx_gpu_buffer *NAME = ltx_gpu_buffer_new_copy( \
        gpu, (DATA), sizeof(DATA), error, error_size)
    LTX_NEW_OUTPUT_BUFFER(x, input);
    LTX_NEW_OUTPUT_BUFFER(e, embedded);
    LTX_NEW_OUTPUT_BUFFER(sh, shift_table);
    LTX_NEW_OUTPUT_BUFFER(sc, scale_table);
    LTX_NEW_OUTPUT_BUFFER(w, weight);
    LTX_NEW_OUTPUT_BUFFER(b, bias);
#undef LTX_NEW_OUTPUT_BUFFER
    ltx_gpu_buffer *m = ltx_gpu_buffer_new(
        gpu, sizeof(actual_modulated), error, error_size);
    ltx_gpu_buffer *y = ltx_gpu_buffer_new(
        gpu, sizeof(actual), error, error_size);
    int ok = x && e && sh && sc && w && b && m && y &&
        ltx_gpu_output_adaln_bf16(
            gpu, m, x, e, sh, sc, rows, hidden_dim, 1e-6f,
            error, error_size) &&
        ltx_gpu_buffer_read(m, actual_modulated,
                            sizeof(actual_modulated), error, error_size);
    for (unsigned index = 0; ok && index < input_count; index++) {
        float expected_value = bf16_to_f32(modulated[index]);
        float actual_value = bf16_to_f32(actual_modulated[index]);
        if (!near(actual_value, expected_value, 1.5e-2f)) {
            snprintf(error, error_size,
                     "output AdaLN mismatch at %u: %.9g != %.9g",
                     index, actual_value, expected_value);
            ok = 0;
        }
    }
    ok = ok && ltx_gpu_linear_bf16(
            gpu, y, m, w, b, rows, hidden_dim, output_dim,
            error, error_size) &&
        ltx_gpu_buffer_read(y, actual, sizeof(actual), error, error_size);
    double difference2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    for (unsigned index = 0; ok && index < output_count; index++) {
        double reference = bf16_to_f32(expected[index]);
        double candidate = bf16_to_f32(actual[index]);
        double difference = candidate - reference;
        difference2 += difference * difference;
        reference2 += reference * reference;
        candidate2 += candidate * candidate;
        dot += reference * candidate;
    }
    if (ok) {
        double rel_l2 = reference2 > 0.0 ?
            sqrt(difference2 / reference2) : 0.0;
        double cosine = reference2 > 0.0 && candidate2 > 0.0 ?
            dot / sqrt(reference2 * candidate2) : 0.0;
        if (rel_l2 > 0.02 || cosine < 0.999) {
            snprintf(error, error_size,
                     "output-head parity failed: rel_l2=%.9g cosine=%.9g",
                     rel_l2, cosine);
            ok = 0;
        }
    }
    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(e);
    ltx_gpu_buffer_free(sh);
    ltx_gpu_buffer_free(sc);
    ltx_gpu_buffer_free(w);
    ltx_gpu_buffer_free(b);
    ltx_gpu_buffer_free(m);
    ltx_gpu_buffer_free(y);
    if (!ok && !error[0])
        snprintf(error, error_size, "BF16 output-head parity failed");
    return ok;
}

static int test_mps_adaln_single(ltx_gpu *gpu, char *error,
                                 size_t error_size) {
    enum {
        rows = 2,
        timestep_dim = 16,
        hidden_dim = 32,
        parameter_count = 3,
        parameter_dim = hidden_dim * parameter_count,
        input_count = rows * timestep_dim,
        linear1_weight_count = hidden_dim * timestep_dim,
        hidden_weight_count = hidden_dim * hidden_dim,
        parameter_weight_count = parameter_dim * hidden_dim,
        embedded_count = rows * hidden_dim,
        parameter_output_count = rows * parameter_dim,
    };
    uint16_t input[input_count];
    uint16_t linear1_weight[linear1_weight_count];
    uint16_t linear1_bias[hidden_dim];
    uint16_t linear2_weight[hidden_weight_count];
    uint16_t linear2_bias[hidden_dim];
    uint16_t parameter_weight[parameter_weight_count];
    uint16_t parameter_bias[parameter_dim];
    uint16_t hidden[embedded_count];
    uint16_t expected_embedded[embedded_count];
    uint16_t activated[embedded_count];
    uint16_t expected_parameters[parameter_output_count];
    uint16_t actual_embedded[embedded_count];
    uint16_t actual_parameters[parameter_output_count];

    for (unsigned index = 0; index < input_count; index++)
        input[index] = f32_to_bf16(
            sinf((float)index * 0.13f) * 0.25f);
    for (unsigned index = 0; index < linear1_weight_count; index++)
        linear1_weight[index] = f32_to_bf16(
            cosf((float)index * 0.017f) * 0.0625f);
    for (unsigned index = 0; index < hidden_weight_count; index++)
        linear2_weight[index] = f32_to_bf16(
            sinf((float)index * 0.011f) * 0.03125f);
    for (unsigned index = 0; index < parameter_weight_count; index++)
        parameter_weight[index] = f32_to_bf16(
            cosf((float)index * 0.007f) * 0.0234375f);
    for (unsigned index = 0; index < hidden_dim; index++) {
        linear1_bias[index] = f32_to_bf16(
            (float)((int)(index % 7u) - 3) * 0.0078125f);
        linear2_bias[index] = f32_to_bf16(
            (float)((int)(index % 5u) - 2) * 0.005859375f);
    }
    for (unsigned index = 0; index < parameter_dim; index++)
        parameter_bias[index] = f32_to_bf16(
            (float)((int)(index % 9u) - 4) * 0.00390625f);

    linear_bf16_cpu(input, linear1_weight, linear1_bias, hidden,
                    rows, timestep_dim, hidden_dim);
    silu_bf16_cpu(hidden, embedded_count);
    linear_bf16_cpu(hidden, linear2_weight, linear2_bias,
                    expected_embedded, rows, hidden_dim, hidden_dim);
    memcpy(activated, expected_embedded, sizeof(activated));
    silu_bf16_cpu(activated, embedded_count);
    linear_bf16_cpu(activated, parameter_weight, parameter_bias,
                    expected_parameters, rows, hidden_dim, parameter_dim);

#define LTX_NEW_BUFFER(NAME, DATA) \
    ltx_gpu_buffer *NAME = ltx_gpu_buffer_new( \
        gpu, sizeof(DATA), error, error_size)
    LTX_NEW_BUFFER(x, input);
    LTX_NEW_BUFFER(w1, linear1_weight);
    LTX_NEW_BUFFER(b1, linear1_bias);
    LTX_NEW_BUFFER(w2, linear2_weight);
    LTX_NEW_BUFFER(b2, linear2_bias);
    LTX_NEW_BUFFER(wp, parameter_weight);
    LTX_NEW_BUFFER(bp, parameter_bias);
    LTX_NEW_BUFFER(embedded, actual_embedded);
    LTX_NEW_BUFFER(parameters, actual_parameters);
#undef LTX_NEW_BUFFER
    int ok = x && w1 && b1 && w2 && b2 && wp && bp &&
        embedded && parameters;
#define LTX_WRITE(BUFFER, DATA) \
    (ok = ok && ltx_gpu_buffer_write( \
        (BUFFER), (DATA), sizeof(DATA), error, error_size))
    LTX_WRITE(x, input);
    LTX_WRITE(w1, linear1_weight);
    LTX_WRITE(b1, linear1_bias);
    LTX_WRITE(w2, linear2_weight);
    LTX_WRITE(b2, linear2_bias);
    LTX_WRITE(wp, parameter_weight);
    LTX_WRITE(bp, parameter_bias);
#undef LTX_WRITE
    ok = ok && ltx_gpu_adaln_single_mps_bf16(
        gpu, parameters, embedded, x, w1, b1, w2, b2, wp, bp,
        rows, timestep_dim, hidden_dim, parameter_count,
        error, error_size) &&
        ltx_gpu_buffer_read(embedded, actual_embedded,
                            sizeof(actual_embedded), error, error_size) &&
        ltx_gpu_buffer_read(parameters, actual_parameters,
                            sizeof(actual_parameters), error, error_size);
    for (unsigned index = 0; ok && index < embedded_count; index++) {
        float actual = bf16_to_f32(actual_embedded[index]);
        float expected = bf16_to_f32(expected_embedded[index]);
        if (!near(actual, expected, 0.025f)) {
            snprintf(error, error_size,
                     "AdaLN embedded mismatch at %u: %.9g != %.9g",
                     index, actual, expected);
            ok = 0;
        }
    }
    for (unsigned index = 0; ok && index < parameter_output_count; index++) {
        float actual = bf16_to_f32(actual_parameters[index]);
        float expected = bf16_to_f32(expected_parameters[index]);
        if (!near(actual, expected, 0.035f)) {
            snprintf(error, error_size,
                     "AdaLN parameter mismatch at %u: %.9g != %.9g",
                     index, actual, expected);
            ok = 0;
        }
    }

    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(w1);
    ltx_gpu_buffer_free(b1);
    ltx_gpu_buffer_free(w2);
    ltx_gpu_buffer_free(b2);
    ltx_gpu_buffer_free(wp);
    ltx_gpu_buffer_free(bp);
    ltx_gpu_buffer_free(embedded);
    ltx_gpu_buffer_free(parameters);
    if (!ok && !error[0])
        snprintf(error, error_size, "MPS AdaLN-single parity failed");
    return ok;
}

static int test_convrot_int8_linear(ltx_gpu *gpu, char *error,
                                    size_t error_size) {
    enum { rows = 3, input_dim = 512, output_dim = 17 };
    enum {
        input_count = rows * input_dim,
        weight_count = output_dim * input_dim,
        output_count = rows * output_dim,
    };
    uint16_t input[input_count];
    uint16_t rotated_expected[input_count];
    uint16_t rotated_actual[input_count];
    int8_t weight[weight_count];
    float scale[output_dim];
    uint16_t bias[output_dim];
    uint16_t output[output_count];
    for (unsigned index = 0; index < input_count; index++)
        input[index] = f32_to_bf16(
            sinf((float)index * 0.019f) * 1.75f +
            cosf((float)index * 0.007f) * 0.25f);
    for (unsigned index = 0; index < weight_count; index++)
        weight[index] = (int8_t)((int)(index * 37u % 251u) - 125);
    for (unsigned index = 0; index < output_dim; index++) {
        scale[index] = 0.00075f + (float)index * 0.00003125f;
        bias[index] = f32_to_bf16((float)index * 0.015625f - 0.0625f);
    }
    convrot_256_cpu(input, rotated_expected, rows, input_dim);

    ltx_gpu_buffer *x = ltx_gpu_buffer_new(gpu, sizeof(input), error,
                                            error_size);
    ltx_gpu_buffer *rx = ltx_gpu_buffer_new(gpu, sizeof(input), error,
                                             error_size);
    ltx_gpu_buffer *w = ltx_gpu_buffer_new(gpu, sizeof(weight), error,
                                            error_size);
    ltx_gpu_buffer *s = ltx_gpu_buffer_new(gpu, sizeof(scale), error,
                                            error_size);
    ltx_gpu_buffer *b = ltx_gpu_buffer_new(gpu, sizeof(bias), error,
                                            error_size);
    ltx_gpu_buffer *y = ltx_gpu_buffer_new(gpu, sizeof(output), error,
                                            error_size);
    int ok = x && rx && w && s && b && y &&
        ltx_gpu_buffer_write(x, input, sizeof(input), error, error_size) &&
        ltx_gpu_buffer_write(w, weight, sizeof(weight), error, error_size) &&
        ltx_gpu_buffer_write(s, scale, sizeof(scale), error, error_size) &&
        ltx_gpu_buffer_write(b, bias, sizeof(bias), error, error_size) &&
        ltx_gpu_convrot_bf16(gpu, rx, x, rows, input_dim, 256u,
                             error, error_size) &&
        ltx_gpu_buffer_read(rx, rotated_actual, sizeof(rotated_actual),
                            error, error_size);
    for (unsigned index = 0; ok && index < input_count; index++) {
        float actual = bf16_to_f32(rotated_actual[index]);
        float expected = bf16_to_f32(rotated_expected[index]);
        if (!near(actual, expected, 8e-3f)) {
            snprintf(error, error_size,
                     "ConvRot mismatch at %u: %.9g != %.9g",
                     index, actual, expected);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_linear_int8_weight_bf16(
        gpu, y, rx, w, s, b, rows, input_dim, output_dim,
        error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);
    for (unsigned row = 0; ok && row < rows; row++)
        for (unsigned column = 0; ok && column < output_dim; column++) {
            float sum = 0.0f;
            for (unsigned k = 0; k < input_dim; k++)
                sum = fmaf(bf16_to_f32(
                               rotated_expected[row * input_dim + k]),
                           (float)weight[column * input_dim + k], sum);
            sum = sum * scale[column] + bf16_to_f32(bias[column]);
            float expected = bf16_to_f32(f32_to_bf16(sum));
            float actual = bf16_to_f32(
                output[row * output_dim + column]);
            if (!near(actual, expected, 8e-3f)) {
                snprintf(error, error_size,
                         "ConvRot INT8 linear mismatch at (%u,%u): "
                         "%.9g != %.9g",
                         row, column, actual, expected);
                ok = 0;
            }
        }

    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(rx);
    ltx_gpu_buffer_free(w);
    ltx_gpu_buffer_free(s);
    ltx_gpu_buffer_free(b);
    ltx_gpu_buffer_free(y);
    if (!ok && !error[0])
        snprintf(error, error_size, "ConvRot INT8 linear parity failed");
    return ok;
}

static int test_mps_int8_mlp(ltx_gpu *gpu, char *error,
                             size_t error_size) {
    enum { rows = 2, input_dim = 256, hidden_dim = 256, output_dim = 31 };
    enum {
        input_count = rows * input_dim,
        fc1_weight_count = hidden_dim * input_dim,
        fc2_weight_count = output_dim * hidden_dim,
        hidden_count = rows * hidden_dim,
        output_count = rows * output_dim,
    };
    uint16_t input[input_count];
    int8_t fc1_weight[fc1_weight_count];
    float fc1_scale[hidden_dim];
    uint16_t fc1_bias[hidden_dim];
    int8_t fc2_weight[fc2_weight_count];
    float fc2_scale[output_dim];
    uint16_t fc2_bias[output_dim];
    uint16_t rotated_input[input_count];
    uint16_t hidden[hidden_count];
    uint16_t rotated_hidden[hidden_count];
    uint16_t expected[output_count];
    uint16_t output[output_count];

    for (unsigned index = 0; index < input_count; index++)
        input[index] = f32_to_bf16(
            sinf((float)index * 0.031f) * 0.75f +
            cosf((float)index * 0.017f) * 0.125f);
    for (unsigned index = 0; index < fc1_weight_count; index++)
        fc1_weight[index] = (int8_t)((int)(index * 17u % 29u) - 14);
    for (unsigned index = 0; index < hidden_dim; index++) {
        fc1_scale[index] = 0.0035f + (float)(index % 11u) * 0.00005f;
        fc1_bias[index] = f32_to_bf16(
            (float)((int)(index % 9u) - 4) * 0.0078125f);
    }
    for (unsigned index = 0; index < fc2_weight_count; index++)
        fc2_weight[index] = (int8_t)((int)(index * 23u % 31u) - 15);
    for (unsigned index = 0; index < output_dim; index++) {
        fc2_scale[index] = 0.004f + (float)(index % 7u) * 0.0000625f;
        fc2_bias[index] = f32_to_bf16(
            (float)((int)(index % 5u) - 2) * 0.015625f);
    }

    convrot_256_cpu(input, rotated_input, rows, input_dim);
    linear_int8_dequant_bf16_cpu(
        rotated_input, fc1_weight, fc1_scale, fc1_bias, hidden,
        rows, input_dim, hidden_dim);
    gelu_tanh_bf16_cpu(hidden, hidden_count);
    convrot_256_cpu(hidden, rotated_hidden, rows, hidden_dim);
    linear_int8_dequant_bf16_cpu(
        rotated_hidden, fc2_weight, fc2_scale, fc2_bias, expected,
        rows, hidden_dim, output_dim);

    ltx_gpu_buffer *x = ltx_gpu_buffer_new(gpu, sizeof(input), error,
                                            error_size);
    ltx_gpu_buffer *w1 = ltx_gpu_buffer_new(gpu, sizeof(fc1_weight), error,
                                             error_size);
    ltx_gpu_buffer *s1 = ltx_gpu_buffer_new(gpu, sizeof(fc1_scale), error,
                                             error_size);
    ltx_gpu_buffer *b1 = ltx_gpu_buffer_new(gpu, sizeof(fc1_bias), error,
                                             error_size);
    ltx_gpu_buffer *w2 = ltx_gpu_buffer_new(gpu, sizeof(fc2_weight), error,
                                             error_size);
    ltx_gpu_buffer *s2 = ltx_gpu_buffer_new(gpu, sizeof(fc2_scale), error,
                                             error_size);
    ltx_gpu_buffer *b2 = ltx_gpu_buffer_new(gpu, sizeof(fc2_bias), error,
                                             error_size);
    ltx_gpu_buffer *y = ltx_gpu_buffer_new(gpu, sizeof(output), error,
                                            error_size);
    int ok = x && w1 && s1 && b1 && w2 && s2 && b2 && y &&
        ltx_gpu_buffer_write(x, input, sizeof(input), error, error_size) &&
        ltx_gpu_buffer_write(w1, fc1_weight, sizeof(fc1_weight),
                             error, error_size) &&
        ltx_gpu_buffer_write(s1, fc1_scale, sizeof(fc1_scale),
                             error, error_size) &&
        ltx_gpu_buffer_write(b1, fc1_bias, sizeof(fc1_bias),
                             error, error_size) &&
        ltx_gpu_buffer_write(w2, fc2_weight, sizeof(fc2_weight),
                             error, error_size) &&
        ltx_gpu_buffer_write(s2, fc2_scale, sizeof(fc2_scale),
                             error, error_size) &&
        ltx_gpu_buffer_write(b2, fc2_bias, sizeof(fc2_bias),
                             error, error_size) &&
        ltx_gpu_mlp_int8_convrot_mps_bf16(
            gpu, y, x, w1, s1, b1, w2, s2, b2,
            rows, input_dim, hidden_dim, output_dim, 256u,
            error, error_size) &&
        ltx_gpu_buffer_read(y, output, sizeof(output), error, error_size);

    double diff2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    for (unsigned index = 0; ok && index < output_count; index++) {
        double reference = bf16_to_f32(expected[index]);
        double candidate = bf16_to_f32(output[index]);
        if (!isfinite(candidate)) {
            snprintf(error, error_size,
                     "MPS INT8 MLP produced non-finite output at %u", index);
            ok = 0;
            break;
        }
        double difference = candidate - reference;
        diff2 += difference * difference;
        reference2 += reference * reference;
        candidate2 += candidate * candidate;
        dot += reference * candidate;
    }
    if (ok) {
        double rel_l2 = reference2 > 0.0 ? sqrt(diff2 / reference2) : 0.0;
        double cosine = reference2 > 0.0 && candidate2 > 0.0 ?
            dot / sqrt(reference2 * candidate2) : 0.0;
        if (rel_l2 > 0.02 || cosine < 0.999) {
            snprintf(error, error_size,
                     "MPS INT8 MLP parity failed: rel_l2=%.9g cosine=%.9g",
                     rel_l2, cosine);
            ok = 0;
        }
    }

    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(w1);
    ltx_gpu_buffer_free(s1);
    ltx_gpu_buffer_free(b1);
    ltx_gpu_buffer_free(w2);
    ltx_gpu_buffer_free(s2);
    ltx_gpu_buffer_free(b2);
    ltx_gpu_buffer_free(y);
    if (!ok && !error[0])
        snprintf(error, error_size, "MPS INT8 MLP parity failed");
    return ok;
}

static int test_mps_int8_qkv(ltx_gpu *gpu, char *error,
                             size_t error_size) {
    enum { rows = 2, input_dim = 256, inner_dim = 256 };
    enum {
        input_count = rows * input_dim,
        weight_count = inner_dim * input_dim,
        output_count = rows * inner_dim,
    };
    uint16_t input[input_count];
    int8_t query_weight[weight_count];
    int8_t key_weight[weight_count];
    int8_t value_weight[weight_count];
    float query_scale[inner_dim];
    float key_scale[inner_dim];
    float value_scale[inner_dim];
    uint16_t query_bias[inner_dim];
    uint16_t key_bias[inner_dim];
    uint16_t value_bias[inner_dim];
    uint16_t query_norm[inner_dim];
    uint16_t key_norm[inner_dim];
    uint16_t rotated[input_count];
    uint16_t expected_query[output_count];
    uint16_t expected_key[output_count];
    uint16_t expected_value[output_count];
    uint16_t output_query[output_count];
    uint16_t output_key[output_count];
    uint16_t output_value[output_count];

    for (unsigned index = 0; index < input_count; index++)
        input[index] = f32_to_bf16(
            sinf((float)index * 0.021f) * 0.625f +
            cosf((float)index * 0.009f) * 0.25f);
    for (unsigned index = 0; index < weight_count; index++) {
        query_weight[index] = (int8_t)((int)(index * 17u % 31u) - 15);
        key_weight[index] = (int8_t)((int)(index * 19u % 29u) - 14);
        value_weight[index] = (int8_t)((int)(index * 23u % 27u) - 13);
    }
    for (unsigned index = 0; index < inner_dim; index++) {
        query_scale[index] = 0.003f + (float)(index % 7u) * 0.00005f;
        key_scale[index] = 0.00325f + (float)(index % 9u) * 0.00004f;
        value_scale[index] = 0.0035f + (float)(index % 11u) * 0.00003f;
        query_bias[index] = f32_to_bf16(
            (float)((int)(index % 5u) - 2) * 0.0078125f);
        key_bias[index] = f32_to_bf16(
            (float)((int)(index % 7u) - 3) * 0.005859375f);
        value_bias[index] = f32_to_bf16(
            (float)((int)(index % 9u) - 4) * 0.00390625f);
        query_norm[index] = f32_to_bf16(
            0.875f + (float)(index % 13u) * 0.0078125f);
        key_norm[index] = f32_to_bf16(
            0.90625f + (float)(index % 11u) * 0.0078125f);
    }
    convrot_256_cpu(input, rotated, rows, input_dim);
    linear_int8_dequant_bf16_cpu(
        rotated, query_weight, query_scale, query_bias, expected_query,
        rows, input_dim, inner_dim);
    linear_int8_dequant_bf16_cpu(
        rotated, key_weight, key_scale, key_bias, expected_key,
        rows, input_dim, inner_dim);
    linear_int8_dequant_bf16_cpu(
        rotated, value_weight, value_scale, value_bias, expected_value,
        rows, input_dim, inner_dim);
    rms_norm_weighted_bf16_cpu(
        expected_query, query_norm, rows, inner_dim, 1e-6f);
    rms_norm_weighted_bf16_cpu(
        expected_key, key_norm, rows, inner_dim, 1e-6f);

    ltx_gpu_buffer *x = ltx_gpu_buffer_new(gpu, sizeof(input), error,
                                            error_size);
    ltx_gpu_buffer *qw = ltx_gpu_buffer_new(gpu, sizeof(query_weight), error,
                                             error_size);
    ltx_gpu_buffer *qs = ltx_gpu_buffer_new(gpu, sizeof(query_scale), error,
                                             error_size);
    ltx_gpu_buffer *qb = ltx_gpu_buffer_new(gpu, sizeof(query_bias), error,
                                             error_size);
    ltx_gpu_buffer *kw = ltx_gpu_buffer_new(gpu, sizeof(key_weight), error,
                                             error_size);
    ltx_gpu_buffer *ks = ltx_gpu_buffer_new(gpu, sizeof(key_scale), error,
                                             error_size);
    ltx_gpu_buffer *kb = ltx_gpu_buffer_new(gpu, sizeof(key_bias), error,
                                             error_size);
    ltx_gpu_buffer *vw = ltx_gpu_buffer_new(gpu, sizeof(value_weight), error,
                                             error_size);
    ltx_gpu_buffer *vs = ltx_gpu_buffer_new(gpu, sizeof(value_scale), error,
                                             error_size);
    ltx_gpu_buffer *vb = ltx_gpu_buffer_new(gpu, sizeof(value_bias), error,
                                             error_size);
    ltx_gpu_buffer *qn = ltx_gpu_buffer_new(gpu, sizeof(query_norm), error,
                                             error_size);
    ltx_gpu_buffer *kn = ltx_gpu_buffer_new(gpu, sizeof(key_norm), error,
                                             error_size);
    ltx_gpu_buffer *qo = ltx_gpu_buffer_new(gpu, sizeof(output_query), error,
                                             error_size);
    ltx_gpu_buffer *ko = ltx_gpu_buffer_new(gpu, sizeof(output_key), error,
                                             error_size);
    ltx_gpu_buffer *vo = ltx_gpu_buffer_new(gpu, sizeof(output_value), error,
                                             error_size);
    int ok = x && qw && qs && qb && kw && ks && kb && vw && vs && vb &&
        qn && kn && qo && ko && vo &&
        ltx_gpu_buffer_write(x, input, sizeof(input), error, error_size) &&
        ltx_gpu_buffer_write(qw, query_weight, sizeof(query_weight),
                             error, error_size) &&
        ltx_gpu_buffer_write(qs, query_scale, sizeof(query_scale),
                             error, error_size) &&
        ltx_gpu_buffer_write(qb, query_bias, sizeof(query_bias),
                             error, error_size) &&
        ltx_gpu_buffer_write(kw, key_weight, sizeof(key_weight),
                             error, error_size) &&
        ltx_gpu_buffer_write(ks, key_scale, sizeof(key_scale),
                             error, error_size) &&
        ltx_gpu_buffer_write(kb, key_bias, sizeof(key_bias),
                             error, error_size) &&
        ltx_gpu_buffer_write(vw, value_weight, sizeof(value_weight),
                             error, error_size) &&
        ltx_gpu_buffer_write(vs, value_scale, sizeof(value_scale),
                             error, error_size) &&
        ltx_gpu_buffer_write(vb, value_bias, sizeof(value_bias),
                             error, error_size) &&
        ltx_gpu_buffer_write(qn, query_norm, sizeof(query_norm),
                             error, error_size) &&
        ltx_gpu_buffer_write(kn, key_norm, sizeof(key_norm),
                             error, error_size) &&
        ltx_gpu_qkv_int8_convrot_mps_bf16(
            gpu, qo, ko, vo, x,
            qw, qs, qb, kw, ks, kb, vw, vs, vb, qn, kn,
            rows, input_dim, inner_dim, 256u, 1e-6f,
            error, error_size) &&
        ltx_gpu_buffer_read(qo, output_query, sizeof(output_query),
                            error, error_size) &&
        ltx_gpu_buffer_read(ko, output_key, sizeof(output_key),
                            error, error_size) &&
        ltx_gpu_buffer_read(vo, output_value, sizeof(output_value),
                            error, error_size);

    double diff2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    const uint16_t *expected_sets[] = {
        expected_query, expected_key, expected_value,
    };
    const uint16_t *output_sets[] = {
        output_query, output_key, output_value,
    };
    for (unsigned set = 0; ok && set < 3u; set++)
        for (unsigned index = 0; index < output_count; index++) {
            double reference = bf16_to_f32(expected_sets[set][index]);
            double candidate = bf16_to_f32(output_sets[set][index]);
            if (!isfinite(candidate)) {
                snprintf(error, error_size,
                         "MPS INT8 QKV produced non-finite output");
                ok = 0;
                break;
            }
            double difference = candidate - reference;
            diff2 += difference * difference;
            reference2 += reference * reference;
            candidate2 += candidate * candidate;
            dot += reference * candidate;
        }
    if (ok) {
        double rel_l2 = reference2 > 0.0 ? sqrt(diff2 / reference2) : 0.0;
        double cosine = reference2 > 0.0 && candidate2 > 0.0 ?
            dot / sqrt(reference2 * candidate2) : 0.0;
        if (rel_l2 > 0.02 || cosine < 0.999) {
            snprintf(error, error_size,
                     "MPS INT8 QKV parity failed: rel_l2=%.9g cosine=%.9g",
                     rel_l2, cosine);
            ok = 0;
        }
    }

    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(qw);
    ltx_gpu_buffer_free(qs);
    ltx_gpu_buffer_free(qb);
    ltx_gpu_buffer_free(kw);
    ltx_gpu_buffer_free(ks);
    ltx_gpu_buffer_free(kb);
    ltx_gpu_buffer_free(vw);
    ltx_gpu_buffer_free(vs);
    ltx_gpu_buffer_free(vb);
    ltx_gpu_buffer_free(qn);
    ltx_gpu_buffer_free(kn);
    ltx_gpu_buffer_free(qo);
    ltx_gpu_buffer_free(ko);
    ltx_gpu_buffer_free(vo);
    if (!ok && !error[0])
        snprintf(error, error_size, "MPS INT8 QKV parity failed");
    return ok;
}

static int test_head_layout_and_sdpa(ltx_gpu *gpu, char *error,
                                     size_t error_size) {
    enum { rows = 4, heads = 2, head_dim = 128, blocks = 1 };
    enum {
        inner_dim = heads * head_dim,
        tensor_count = rows * inner_dim,
        frequency_count = heads * rows * (head_dim / 2),
        gate_count = rows * heads,
        summary_count = heads * blocks * head_dim,
        threshold_count = heads * blocks,
        route_count = heads * blocks * blocks,
    };
    uint16_t query_input[tensor_count];
    uint16_t value_input[tensor_count];
    uint16_t cosine[frequency_count];
    uint16_t sine[frequency_count];
    uint16_t gate[gate_count];
    uint16_t query_expected[tensor_count];
    uint16_t value_expected[tensor_count];
    uint16_t attention_expected[tensor_count];
    uint16_t unpacked_expected[tensor_count];
    uint16_t query_actual[tensor_count];
    uint16_t value_actual[tensor_count];
    uint16_t attention_actual[tensor_count];
    uint16_t unpacked_actual[tensor_count];
    uint16_t core_actual[tensor_count];
    uint16_t sol_actual[tensor_count];
    for (unsigned index = 0; index < tensor_count; index++) {
        query_input[index] = f32_to_bf16(
            sinf((float)index * 0.037f) * 0.5f +
            cosf((float)index * 0.011f) * 0.125f);
        value_input[index] = f32_to_bf16(
            cosf((float)index * 0.029f) * 0.375f -
            sinf((float)index * 0.017f) * 0.0625f);
    }
    for (unsigned index = 0; index < frequency_count; index++) {
        float angle = (float)(index % 97u) * 0.0075f;
        cosine[index] = f32_to_bf16(cosf(angle));
        sine[index] = f32_to_bf16(sinf(angle));
    }
    for (unsigned row = 0; row < rows; row++)
        for (unsigned head = 0; head < heads; head++)
            gate[row * heads + head] = f32_to_bf16(
                0.75f + (float)(row + head) * 0.0625f);

    for (unsigned head = 0; head < heads; head++)
        for (unsigned row = 0; row < rows; row++)
            for (unsigned dimension = 0; dimension < head_dim; dimension++) {
                unsigned destination =
                    (head * rows + row) * head_dim + dimension;
                unsigned source = row * inner_dim +
                    head * head_dim + dimension;
                unsigned half = head_dim / 2u;
                unsigned pair = dimension % half;
                unsigned source_base = row * inner_dim + head * head_dim;
                unsigned frequency =
                    (head * rows + row) * half + pair;
                float first = bf16_to_f32(query_input[source_base + pair]);
                float second = bf16_to_f32(
                    query_input[source_base + pair + half]);
                float c = bf16_to_f32(cosine[frequency]);
                float s = bf16_to_f32(sine[frequency]);
                float rotated = dimension < half ?
                    first * c - second * s : first * s + second * c;
                query_expected[destination] = f32_to_bf16(rotated);
                value_expected[destination] = value_input[source];
            }

    const float scale = 0.125f;
    for (unsigned head = 0; head < heads; head++)
        for (unsigned query_row = 0; query_row < rows; query_row++) {
            float scores[rows];
            float maximum = -INFINITY;
            for (unsigned key_row = 0; key_row < rows; key_row++) {
                float score = 0.0f;
                for (unsigned dimension = 0; dimension < head_dim;
                     dimension++)
                    score = fmaf(
                        bf16_to_f32(query_expected[
                            (head * rows + query_row) * head_dim + dimension]),
                        bf16_to_f32(query_expected[
                            (head * rows + key_row) * head_dim + dimension]),
                        score);
                scores[key_row] = score * scale;
                if (scores[key_row] > maximum) maximum = scores[key_row];
            }
            float denominator = 0.0f;
            for (unsigned key_row = 0; key_row < rows; key_row++) {
                scores[key_row] = expf(scores[key_row] - maximum);
                denominator += scores[key_row];
            }
            for (unsigned dimension = 0; dimension < head_dim;
                 dimension++) {
                float sum = 0.0f;
                for (unsigned key_row = 0; key_row < rows; key_row++)
                    sum = fmaf(scores[key_row] / denominator,
                        bf16_to_f32(value_expected[
                            (head * rows + key_row) * head_dim + dimension]),
                        sum);
                attention_expected[
                    (head * rows + query_row) * head_dim + dimension] =
                    f32_to_bf16(sum);
            }
        }
    for (unsigned row = 0; row < rows; row++)
        for (unsigned head = 0; head < heads; head++)
            for (unsigned dimension = 0; dimension < head_dim; dimension++) {
                unsigned destination = row * inner_dim +
                    head * head_dim + dimension;
                unsigned source =
                    (head * rows + row) * head_dim + dimension;
                unpacked_expected[destination] = f32_to_bf16(
                    bf16_to_f32(attention_expected[source]) *
                    bf16_to_f32(gate[row * heads + head]));
            }

    ltx_gpu_buffer *q_in = ltx_gpu_buffer_new(gpu, sizeof(query_input),
                                               error, error_size);
    ltx_gpu_buffer *v_in = ltx_gpu_buffer_new(gpu, sizeof(value_input),
                                               error, error_size);
    ltx_gpu_buffer *cos_buffer = ltx_gpu_buffer_new(gpu, sizeof(cosine),
                                                    error, error_size);
    ltx_gpu_buffer *sin_buffer = ltx_gpu_buffer_new(gpu, sizeof(sine),
                                                    error, error_size);
    ltx_gpu_buffer *gate_buffer = ltx_gpu_buffer_new(gpu, sizeof(gate),
                                                     error, error_size);
    ltx_gpu_buffer *q = ltx_gpu_buffer_new(gpu, sizeof(query_actual),
                                           error, error_size);
    ltx_gpu_buffer *v = ltx_gpu_buffer_new(gpu, sizeof(value_actual),
                                           error, error_size);
    ltx_gpu_buffer *attention = ltx_gpu_buffer_new(
        gpu, sizeof(attention_actual), error, error_size);
    ltx_gpu_buffer *unpacked = ltx_gpu_buffer_new(
        gpu, sizeof(unpacked_actual), error, error_size);
    ltx_gpu_buffer *packed_query = ltx_gpu_buffer_new(
        gpu, sizeof(query_actual), error, error_size);
    ltx_gpu_buffer *packed_key = ltx_gpu_buffer_new(
        gpu, sizeof(query_actual), error, error_size);
    ltx_gpu_buffer *packed_value = ltx_gpu_buffer_new(
        gpu, sizeof(value_actual), error, error_size);
    ltx_gpu_buffer *packed_output = ltx_gpu_buffer_new(
        gpu, sizeof(attention_actual), error, error_size);
    ltx_gpu_buffer *query_centroids = ltx_gpu_buffer_new(
        gpu, summary_count * sizeof(float), error, error_size);
    ltx_gpu_buffer *key_centroids = ltx_gpu_buffer_new(
        gpu, summary_count * sizeof(uint16_t), error, error_size);
    ltx_gpu_buffer *value_sums = ltx_gpu_buffer_new(
        gpu, summary_count * sizeof(uint16_t), error, error_size);
    ltx_gpu_buffer *thresholds = ltx_gpu_buffer_new(
        gpu, threshold_count * sizeof(float), error, error_size);
    ltx_gpu_buffer *routes = ltx_gpu_buffer_new(
        gpu, route_count * sizeof(float), error, error_size);
    int ok = q_in && v_in && cos_buffer && sin_buffer && gate_buffer &&
        q && v && attention && unpacked && packed_query && packed_key &&
        packed_value && packed_output && query_centroids && key_centroids &&
        value_sums && thresholds && routes &&
        ltx_gpu_buffer_write(q_in, query_input, sizeof(query_input),
                             error, error_size) &&
        ltx_gpu_buffer_write(v_in, value_input, sizeof(value_input),
                             error, error_size) &&
        ltx_gpu_buffer_write(cos_buffer, cosine, sizeof(cosine),
                             error, error_size) &&
        ltx_gpu_buffer_write(sin_buffer, sine, sizeof(sine),
                             error, error_size) &&
        ltx_gpu_buffer_write(gate_buffer, gate, sizeof(gate),
                             error, error_size) &&
        ltx_gpu_pack_rope_split_bf16(
            gpu, q, q_in, cos_buffer, sin_buffer,
            rows, heads, head_dim, error, error_size) &&
        ltx_gpu_pack_heads_bf16(
            gpu, v, v_in, rows, heads, head_dim, error, error_size) &&
        ltx_gpu_buffer_read(q, query_actual, sizeof(query_actual),
                            error, error_size) &&
        ltx_gpu_buffer_read(v, value_actual, sizeof(value_actual),
                            error, error_size);
    for (unsigned index = 0; ok && index < tensor_count; index++) {
        if (!near(bf16_to_f32(query_actual[index]),
                  bf16_to_f32(query_expected[index]), 8e-3f) ||
            value_actual[index] != value_expected[index]) {
            snprintf(error, error_size,
                     "head pack/RoPE mismatch at %u", index);
            ok = 0;
        }
    }

    ok = ok && ltx_gpu_sdpa_mps_bf16(
        gpu, attention, q, q, v, heads, rows, rows, head_dim, scale,
        error, error_size) &&
        ltx_gpu_buffer_read(attention, attention_actual,
                            sizeof(attention_actual), error, error_size) &&
        ltx_gpu_unpack_heads_gate_bf16(
            gpu, unpacked, attention, gate_buffer,
            rows, heads, head_dim, error, error_size) &&
        ltx_gpu_buffer_read(unpacked, unpacked_actual,
                            sizeof(unpacked_actual), error, error_size);
    ok = ok && ltx_gpu_self_attention_core_mps_bf16(
        gpu, unpacked, q_in, q_in, v_in, cos_buffer, sin_buffer, gate_buffer,
        rows, heads, head_dim, scale, error, error_size) &&
        ltx_gpu_buffer_read(unpacked, core_actual,
                            sizeof(core_actual), error, error_size);
    ok = ok && ltx_gpu_self_attention_core_sol_bf16(
        gpu, unpacked, q_in, q_in, v_in, cos_buffer, sin_buffer, gate_buffer,
        packed_query, packed_key, packed_value, packed_output,
        query_centroids, key_centroids, value_sums, thresholds, routes,
        rows, heads, head_dim, scale, 1.0f,
        0u, blocks, 0u, 0u, error, error_size) &&
        ltx_gpu_buffer_read(unpacked, sol_actual,
                            sizeof(sol_actual), error, error_size);

    double diff2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    const uint16_t *attention_results[] = {
        unpacked_actual, core_actual, sol_actual,
    };
    for (unsigned path = 0; ok &&
         path < sizeof(attention_results) / sizeof(attention_results[0]);
         path++)
        for (unsigned index = 0; index < tensor_count; index++) {
            double reference = bf16_to_f32(unpacked_expected[index]);
            double candidate = bf16_to_f32(attention_results[path][index]);
            if (!isfinite(candidate)) {
                snprintf(error, error_size,
                         "SDPA path %u produced non-finite output", path);
                ok = 0;
                break;
            }
            double difference = candidate - reference;
            diff2 += difference * difference;
            reference2 += reference * reference;
            candidate2 += candidate * candidate;
            dot += reference * candidate;
        }
    if (ok) {
        double rel_l2 = reference2 > 0.0 ? sqrt(diff2 / reference2) : 0.0;
        double cosine_similarity = reference2 > 0.0 && candidate2 > 0.0 ?
            dot / sqrt(reference2 * candidate2) : 0.0;
        if (rel_l2 > 0.03 || cosine_similarity < 0.998) {
            snprintf(error, error_size,
                     "BF16 SDPA parity failed: rel_l2=%.9g cosine=%.9g",
                     rel_l2, cosine_similarity);
            ok = 0;
        }
    }

    ltx_gpu_buffer_free(q_in);
    ltx_gpu_buffer_free(v_in);
    ltx_gpu_buffer_free(cos_buffer);
    ltx_gpu_buffer_free(sin_buffer);
    ltx_gpu_buffer_free(gate_buffer);
    ltx_gpu_buffer_free(q);
    ltx_gpu_buffer_free(v);
    ltx_gpu_buffer_free(attention);
    ltx_gpu_buffer_free(unpacked);
    ltx_gpu_buffer_free(packed_query);
    ltx_gpu_buffer_free(packed_key);
    ltx_gpu_buffer_free(packed_value);
    ltx_gpu_buffer_free(packed_output);
    ltx_gpu_buffer_free(query_centroids);
    ltx_gpu_buffer_free(key_centroids);
    ltx_gpu_buffer_free(value_sums);
    ltx_gpu_buffer_free(thresholds);
    ltx_gpu_buffer_free(routes);
    if (!ok && !error[0])
        snprintf(error, error_size, "head layout/SDPA parity failed");
    return ok;
}

static int test_mps_int8_cross_attention(ltx_gpu *gpu, char *error,
                                         size_t error_size) {
    enum {
        query_rows = 3,
        key_value_rows = 3,
        query_dim = 512,
        key_value_dim = 256,
        heads = 4,
        head_dim = 64,
        inner_dim = heads * head_dim,
        output_dim = 512,
        query_input_count = query_rows * query_dim,
        key_value_input_count = key_value_rows * key_value_dim,
        query_weight_count = inner_dim * query_dim,
        key_value_weight_count = inner_dim * key_value_dim,
        output_weight_count = output_dim * inner_dim,
        query_inner_count = query_rows * inner_dim,
        key_value_inner_count = key_value_rows * inner_dim,
        output_count = query_rows * output_dim,
        query_frequency_count = heads * query_rows * (head_dim / 2),
        key_frequency_count = heads * key_value_rows * (head_dim / 2),
    };
    uint16_t query_input[query_input_count];
    uint16_t key_value_input[key_value_input_count];
    int8_t query_weight[query_weight_count];
    int8_t key_weight[key_value_weight_count];
    int8_t value_weight[key_value_weight_count];
    int8_t output_weight[output_weight_count];
    float query_scale[inner_dim];
    float key_scale[inner_dim];
    float value_scale[inner_dim];
    float output_scale[output_dim];
    uint16_t query_bias[inner_dim];
    uint16_t key_bias[inner_dim];
    uint16_t value_bias[inner_dim];
    uint16_t output_bias[output_dim];
    uint16_t query_norm[inner_dim];
    uint16_t key_norm[inner_dim];
    uint16_t gate_weight[heads * query_dim];
    uint16_t gate_bias[heads];
    uint16_t query_cosine[query_frequency_count];
    uint16_t query_sine[query_frequency_count];
    uint16_t key_cosine[key_frequency_count];
    uint16_t key_sine[key_frequency_count];
    uint16_t attention_mask[key_value_rows];
    uint16_t rotated_query[query_input_count];
    uint16_t rotated_key_value[key_value_input_count];
    uint16_t query_projection[query_inner_count];
    uint16_t key_projection[key_value_inner_count];
    uint16_t value_projection[key_value_inner_count];
    uint16_t query_heads[query_inner_count];
    uint16_t key_heads[key_value_inner_count];
    uint16_t value_heads[key_value_inner_count];
    uint16_t attention_heads[query_inner_count];
    uint16_t attention_row_major[query_inner_count];
    uint16_t rotated_attention[query_inner_count];
    uint16_t expected[output_count];
    uint16_t actual[output_count];

    for (unsigned index = 0; index < query_input_count; index++)
        query_input[index] = f32_to_bf16(
            sinf((float)index * 0.019f) * 0.5f +
            cosf((float)index * 0.007f) * 0.125f);
    for (unsigned index = 0; index < key_value_input_count; index++)
        key_value_input[index] = f32_to_bf16(
            cosf((float)index * 0.023f) * 0.375f -
            sinf((float)index * 0.011f) * 0.0625f);
    for (unsigned index = 0; index < query_weight_count; index++)
        query_weight[index] = (int8_t)((int)(index * 17u % 29u) - 14);
    for (unsigned index = 0; index < key_value_weight_count; index++) {
        key_weight[index] = (int8_t)((int)(index * 19u % 31u) - 15);
        value_weight[index] = (int8_t)((int)(index * 23u % 27u) - 13);
    }
    for (unsigned index = 0; index < output_weight_count; index++)
        output_weight[index] = (int8_t)((int)(index * 29u % 25u) - 12);
    for (unsigned index = 0; index < inner_dim; index++) {
        query_scale[index] = 0.003f + (float)(index % 7u) * 0.00004f;
        key_scale[index] = 0.00325f + (float)(index % 9u) * 0.000035f;
        value_scale[index] = 0.0035f + (float)(index % 11u) * 0.00003f;
        query_bias[index] = f32_to_bf16(
            (float)((int)(index % 7u) - 3) * 0.00390625f);
        key_bias[index] = f32_to_bf16(
            (float)((int)(index % 5u) - 2) * 0.005859375f);
        value_bias[index] = f32_to_bf16(
            (float)((int)(index % 9u) - 4) * 0.0029296875f);
        query_norm[index] = f32_to_bf16(
            0.875f + (float)(index % 13u) * 0.0078125f);
        key_norm[index] = f32_to_bf16(
            0.90625f + (float)(index % 11u) * 0.0078125f);
    }
    for (unsigned index = 0; index < output_dim; index++) {
        output_scale[index] = 0.00375f +
            (float)(index % 7u) * 0.00003125f;
        output_bias[index] = f32_to_bf16(
            (float)((int)(index % 11u) - 5) * 0.00390625f);
    }
    for (unsigned index = 0; index < heads * query_dim; index++)
        gate_weight[index] = f32_to_bf16(
            sinf((float)index * 0.017f) * 0.015625f);
    for (unsigned head = 0; head < heads; head++)
        gate_bias[head] = f32_to_bf16(
            (float)((int)head - 2) * 0.03125f);
    for (unsigned index = 0; index < query_frequency_count; index++) {
        float angle = (float)(index % 53u) * 0.013f;
        query_cosine[index] = f32_to_bf16(cosf(angle));
        query_sine[index] = f32_to_bf16(sinf(angle));
    }
    for (unsigned index = 0; index < key_frequency_count; index++) {
        float angle = (float)(index % 47u) * 0.011f;
        key_cosine[index] = f32_to_bf16(cosf(angle));
        key_sine[index] = f32_to_bf16(sinf(angle));
    }
    for (unsigned row = 0; row < key_value_rows; row++)
        attention_mask[row] = f32_to_bf16(
            row + 1u == key_value_rows ? -10000.0f : 0.0f);

    convrot_256_cpu(query_input, rotated_query, query_rows, query_dim);
    convrot_256_cpu(key_value_input, rotated_key_value,
                    key_value_rows, key_value_dim);
    linear_int8_dequant_bf16_cpu(
        rotated_query, query_weight, query_scale, query_bias,
        query_projection, query_rows, query_dim, inner_dim);
    linear_int8_dequant_bf16_cpu(
        rotated_key_value, key_weight, key_scale, key_bias,
        key_projection, key_value_rows, key_value_dim, inner_dim);
    linear_int8_dequant_bf16_cpu(
        rotated_key_value, value_weight, value_scale, value_bias,
        value_projection, key_value_rows, key_value_dim, inner_dim);
    rms_norm_weighted_bf16_cpu(
        query_projection, query_norm, query_rows, inner_dim, 1e-6f);
    rms_norm_weighted_bf16_cpu(
        key_projection, key_norm, key_value_rows, inner_dim, 1e-6f);

    for (unsigned head = 0; head < heads; head++) {
        for (unsigned row = 0; row < query_rows; row++)
            for (unsigned dimension = 0; dimension < head_dim; dimension++) {
                unsigned half = head_dim / 2u;
                unsigned pair = dimension % half;
                unsigned source = row * inner_dim + head * head_dim;
                unsigned frequency =
                    (head * query_rows + row) * half + pair;
                float first = bf16_to_f32(query_projection[source + pair]);
                float second = bf16_to_f32(
                    query_projection[source + pair + half]);
                float c = bf16_to_f32(query_cosine[frequency]);
                float s = bf16_to_f32(query_sine[frequency]);
                query_heads[(head * query_rows + row) * head_dim + dimension] =
                    f32_to_bf16(dimension < half ?
                        first * c - second * s : first * s + second * c);
            }
        for (unsigned row = 0; row < key_value_rows; row++)
            for (unsigned dimension = 0; dimension < head_dim; dimension++) {
                unsigned half = head_dim / 2u;
                unsigned pair = dimension % half;
                unsigned source = row * inner_dim + head * head_dim;
                unsigned frequency =
                    (head * key_value_rows + row) * half + pair;
                float first = bf16_to_f32(key_projection[source + pair]);
                float second = bf16_to_f32(
                    key_projection[source + pair + half]);
                float c = bf16_to_f32(key_cosine[frequency]);
                float s = bf16_to_f32(key_sine[frequency]);
                key_heads[(head * key_value_rows + row) * head_dim +
                          dimension] = f32_to_bf16(dimension < half ?
                    first * c - second * s : first * s + second * c);
                value_heads[(head * key_value_rows + row) * head_dim +
                            dimension] = value_projection[source + dimension];
            }
    }

    const float attention_scale = 0.125f;
    for (unsigned head = 0; head < heads; head++)
        for (unsigned query_row = 0; query_row < query_rows; query_row++) {
            float scores[key_value_rows];
            float maximum = -INFINITY;
            for (unsigned key_row = 0; key_row < key_value_rows; key_row++) {
                float score = 0.0f;
                for (unsigned dimension = 0; dimension < head_dim;
                     dimension++)
                    score = fmaf(
                        bf16_to_f32(query_heads[
                            (head * query_rows + query_row) * head_dim +
                            dimension]),
                        bf16_to_f32(key_heads[
                            (head * key_value_rows + key_row) * head_dim +
                            dimension]), score);
                scores[key_row] = score * attention_scale +
                    bf16_to_f32(attention_mask[key_row]);
                if (scores[key_row] > maximum) maximum = scores[key_row];
            }
            float denominator = 0.0f;
            for (unsigned key_row = 0; key_row < key_value_rows; key_row++) {
                scores[key_row] = expf(scores[key_row] - maximum);
                denominator += scores[key_row];
            }
            float gate_sum = bf16_to_f32(gate_bias[head]);
            for (unsigned column = 0; column < query_dim; column++)
                gate_sum = fmaf(
                    bf16_to_f32(query_input[query_row * query_dim + column]),
                    bf16_to_f32(gate_weight[head * query_dim + column]),
                    gate_sum);
            float gate_logit = bf16_to_f32(f32_to_bf16(gate_sum));
            float gate_value = bf16_to_f32(f32_to_bf16(
                2.0f / (1.0f + expf(-gate_logit))));
            for (unsigned dimension = 0; dimension < head_dim;
                 dimension++) {
                float sum = 0.0f;
                for (unsigned key_row = 0; key_row < key_value_rows;
                     key_row++)
                    sum = fmaf(scores[key_row] / denominator,
                        bf16_to_f32(value_heads[
                            (head * key_value_rows + key_row) * head_dim +
                            dimension]), sum);
                attention_heads[
                    (head * query_rows + query_row) * head_dim + dimension] =
                    f32_to_bf16(sum * gate_value);
            }
        }
    for (unsigned row = 0; row < query_rows; row++)
        for (unsigned head = 0; head < heads; head++)
            for (unsigned dimension = 0; dimension < head_dim; dimension++)
                attention_row_major[row * inner_dim + head * head_dim +
                    dimension] = attention_heads[
                        (head * query_rows + row) * head_dim + dimension];
    convrot_256_cpu(attention_row_major, rotated_attention,
                    query_rows, inner_dim);
    linear_int8_dequant_bf16_cpu(
        rotated_attention, output_weight, output_scale, output_bias,
        expected, query_rows, inner_dim, output_dim);

#define LTX_NEW_BUFFER(NAME, DATA) \
    ltx_gpu_buffer *NAME = ltx_gpu_buffer_new( \
        gpu, sizeof(DATA), error, error_size)
    LTX_NEW_BUFFER(qi, query_input);
    LTX_NEW_BUFFER(kvi, key_value_input);
    LTX_NEW_BUFFER(qw, query_weight);
    LTX_NEW_BUFFER(qs, query_scale);
    LTX_NEW_BUFFER(qb, query_bias);
    LTX_NEW_BUFFER(kw, key_weight);
    LTX_NEW_BUFFER(ks, key_scale);
    LTX_NEW_BUFFER(kb, key_bias);
    LTX_NEW_BUFFER(vw, value_weight);
    LTX_NEW_BUFFER(vs, value_scale);
    LTX_NEW_BUFFER(vb, value_bias);
    LTX_NEW_BUFFER(qn, query_norm);
    LTX_NEW_BUFFER(kn, key_norm);
    LTX_NEW_BUFFER(gw, gate_weight);
    LTX_NEW_BUFFER(gb, gate_bias);
    LTX_NEW_BUFFER(ow, output_weight);
    LTX_NEW_BUFFER(os, output_scale);
    LTX_NEW_BUFFER(ob, output_bias);
    LTX_NEW_BUFFER(qc, query_cosine);
    LTX_NEW_BUFFER(qsi, query_sine);
    LTX_NEW_BUFFER(kc, key_cosine);
    LTX_NEW_BUFFER(ksi, key_sine);
    LTX_NEW_BUFFER(mask, attention_mask);
    LTX_NEW_BUFFER(out, actual);
#undef LTX_NEW_BUFFER
    int ok = qi && kvi && qw && qs && qb && kw && ks && kb &&
        vw && vs && vb && qn && kn && gw && gb && ow && os && ob &&
        qc && qsi && kc && ksi && mask && out;
#define LTX_WRITE(BUFFER, DATA) \
    (ok = ok && ltx_gpu_buffer_write( \
        (BUFFER), (DATA), sizeof(DATA), error, error_size))
    LTX_WRITE(qi, query_input);
    LTX_WRITE(kvi, key_value_input);
    LTX_WRITE(qw, query_weight);
    LTX_WRITE(qs, query_scale);
    LTX_WRITE(qb, query_bias);
    LTX_WRITE(kw, key_weight);
    LTX_WRITE(ks, key_scale);
    LTX_WRITE(kb, key_bias);
    LTX_WRITE(vw, value_weight);
    LTX_WRITE(vs, value_scale);
    LTX_WRITE(vb, value_bias);
    LTX_WRITE(qn, query_norm);
    LTX_WRITE(kn, key_norm);
    LTX_WRITE(gw, gate_weight);
    LTX_WRITE(gb, gate_bias);
    LTX_WRITE(ow, output_weight);
    LTX_WRITE(os, output_scale);
    LTX_WRITE(ob, output_bias);
    LTX_WRITE(qc, query_cosine);
    LTX_WRITE(qsi, query_sine);
    LTX_WRITE(kc, key_cosine);
    LTX_WRITE(ksi, key_sine);
    LTX_WRITE(mask, attention_mask);
#undef LTX_WRITE
    ok = ok && ltx_gpu_cross_attention_int8_mps_bf16_masked(
        gpu, out, qi, kvi,
        qw, qs, qb, kw, ks, kb, vw, vs, vb,
        qn, kn, gw, gb, ow, os, ob,
        qc, qsi, kc, ksi, mask, 1u,
        query_rows, key_value_rows, query_dim, key_value_dim,
        heads, head_dim, output_dim, 256u, 1e-6f,
        error, error_size) &&
        ltx_gpu_buffer_read(out, actual, sizeof(actual), error, error_size);

    double diff2 = 0.0;
    double reference2 = 0.0;
    double candidate2 = 0.0;
    double dot = 0.0;
    for (unsigned index = 0; ok && index < output_count; index++) {
        double reference = bf16_to_f32(expected[index]);
        double candidate = bf16_to_f32(actual[index]);
        if (!isfinite(candidate)) {
            snprintf(error, error_size,
                     "cross-attention produced non-finite output");
            ok = 0;
            break;
        }
        double difference = candidate - reference;
        diff2 += difference * difference;
        reference2 += reference * reference;
        candidate2 += candidate * candidate;
        dot += reference * candidate;
    }
    if (ok) {
        double rel_l2 = reference2 > 0.0 ? sqrt(diff2 / reference2) : 0.0;
        double cosine = reference2 > 0.0 && candidate2 > 0.0 ?
            dot / sqrt(reference2 * candidate2) : 0.0;
        if (rel_l2 > 0.04 || cosine < 0.997) {
            snprintf(error, error_size,
                     "cross-attention parity failed: rel_l2=%.9g cosine=%.9g",
                     rel_l2, cosine);
            ok = 0;
        }
    }

    ltx_gpu_buffer_free(qi);
    ltx_gpu_buffer_free(kvi);
    ltx_gpu_buffer_free(qw);
    ltx_gpu_buffer_free(qs);
    ltx_gpu_buffer_free(qb);
    ltx_gpu_buffer_free(kw);
    ltx_gpu_buffer_free(ks);
    ltx_gpu_buffer_free(kb);
    ltx_gpu_buffer_free(vw);
    ltx_gpu_buffer_free(vs);
    ltx_gpu_buffer_free(vb);
    ltx_gpu_buffer_free(qn);
    ltx_gpu_buffer_free(kn);
    ltx_gpu_buffer_free(gw);
    ltx_gpu_buffer_free(gb);
    ltx_gpu_buffer_free(ow);
    ltx_gpu_buffer_free(os);
    ltx_gpu_buffer_free(ob);
    ltx_gpu_buffer_free(qc);
    ltx_gpu_buffer_free(qsi);
    ltx_gpu_buffer_free(kc);
    ltx_gpu_buffer_free(ksi);
    ltx_gpu_buffer_free(mask);
    ltx_gpu_buffer_free(out);
    if (!ok && !error[0])
        snprintf(error, error_size, "MPS INT8 cross-attention parity failed");
    return ok;
}

static int test_latent_stats(ltx_gpu *gpu, char *error,
                             size_t error_size) {
    enum { rows = 3, channels = 5, count = rows * channels };
    uint16_t input[count];
    uint16_t mean[channels];
    uint16_t standard_deviation[channels];
    uint16_t denormalized[count];
    uint16_t normalized[count];
    for (unsigned channel = 0; channel < channels; channel++) {
        mean[channel] = f32_to_bf16(
            (float)((int)channel - 2) * 0.125f);
        standard_deviation[channel] = f32_to_bf16(
            0.75f + (float)channel * 0.125f);
    }
    for (unsigned index = 0; index < count; index++)
        input[index] = f32_to_bf16(
            sinf((float)index * 0.31f) * 1.5f);

    ltx_gpu_buffer *x = ltx_gpu_buffer_new_copy(
        gpu, input, sizeof(input), error, error_size);
    ltx_gpu_buffer *m = ltx_gpu_buffer_new_copy(
        gpu, mean, sizeof(mean), error, error_size);
    ltx_gpu_buffer *s = ltx_gpu_buffer_new_copy(
        gpu, standard_deviation, sizeof(standard_deviation),
        error, error_size);
    ltx_gpu_buffer *y = ltx_gpu_buffer_new(
        gpu, sizeof(denormalized), error, error_size);
    ltx_gpu_buffer *z = ltx_gpu_buffer_new(
        gpu, sizeof(normalized), error, error_size);
    int ok = x && m && s && y && z &&
        ltx_gpu_latent_stats_bf16(
            gpu, y, x, m, s, rows, channels, 0,
            error, error_size) &&
        ltx_gpu_buffer_read(y, denormalized, sizeof(denormalized),
                            error, error_size) &&
        ltx_gpu_latent_stats_bf16(
            gpu, z, y, m, s, rows, channels, 1,
            error, error_size) &&
        ltx_gpu_buffer_read(z, normalized, sizeof(normalized),
                            error, error_size);
    for (unsigned index = 0; ok && index < count; index++) {
        unsigned channel = index % channels;
        float expected_denormalized = fmaf(
            bf16_to_f32(input[index]),
            bf16_to_f32(standard_deviation[channel]),
            bf16_to_f32(mean[channel]));
        if (denormalized[index] !=
            f32_to_bf16(expected_denormalized)) {
            snprintf(error, error_size,
                     "latent denormalize mismatch at %u", index);
            ok = 0;
            break;
        }
        if (!near(bf16_to_f32(normalized[index]),
                  bf16_to_f32(input[index]), 0.01f)) {
            snprintf(error, error_size,
                     "latent normalize round-trip mismatch at %u", index);
            ok = 0;
        }
    }
    ltx_gpu_buffer_free(x);
    ltx_gpu_buffer_free(m);
    ltx_gpu_buffer_free(s);
    ltx_gpu_buffer_free(y);
    ltx_gpu_buffer_free(z);
    return ok;
}

int main(void) {
    char error[1024] = {0};
    ltx_gpu *gpu = ltx_gpu_create("ltx_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("test_gpu: SKIP (%s)\n", error);
        return 0;
    }

    int ok = test_elementwise(gpu, error, sizeof(error)) &&
        test_bf16_slice_and_add(gpu, error, sizeof(error)) &&
        test_bf16_f16_row_partition(gpu, error, sizeof(error)) &&
        test_diffusion_steps(gpu, error, sizeof(error)) &&
        test_split_conditioning(gpu, error, sizeof(error)) &&
        test_latent_stats(gpu, error, sizeof(error)) &&
        test_adaln_and_residual_bf16(gpu, error, sizeof(error)) &&
        test_cast_and_gelu(gpu, error, sizeof(error)) &&
        test_rms_norm(gpu, error, sizeof(error)) &&
        test_linear(gpu, error, sizeof(error)) &&
        test_output_head(gpu, error, sizeof(error)) &&
        test_mps_adaln_single(gpu, error, sizeof(error)) &&
        test_convrot_int8_linear(gpu, error, sizeof(error)) &&
        test_mps_int8_mlp(gpu, error, sizeof(error)) &&
        test_mps_int8_qkv(gpu, error, sizeof(error)) &&
        test_head_layout_and_sdpa(gpu, error, sizeof(error)) &&
        test_mps_int8_cross_attention(gpu, error, sizeof(error));
    ltx_gpu_free(gpu);
    if (!ok) {
        fprintf(stderr, "test_gpu: FAIL (%s)\n",
                error[0] ? error : "unknown error");
        return 1;
    }
    puts("test_gpu: PASS");
    return 0;
}
