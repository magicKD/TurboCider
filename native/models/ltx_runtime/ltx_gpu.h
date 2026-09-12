#ifndef LTX_GPU_H
#define LTX_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct ltx_gpu ltx_gpu;
typedef struct ltx_gpu_buffer ltx_gpu_buffer;

typedef struct {
    char name[256];
    uint64_t recommended_working_set_bytes;
    uint64_t max_buffer_bytes;
    int unified_memory;
    int supports_apple7;
    int supports_apple8;
    int supports_apple9;
} ltx_gpu_info;

ltx_gpu *ltx_gpu_create(const char *shader_source_path,
                        char *error, size_t error_size);
void ltx_gpu_free(ltx_gpu *gpu);
int ltx_gpu_get_info(const ltx_gpu *gpu, ltx_gpu_info *info);
/* Release cached MPSGraph objects while keeping the Metal device/queue alive. */
void ltx_gpu_clear_graph_cache(ltx_gpu *gpu);
/* Queue multiple ordered Metal/MPSGraph command buffers and wait once at the
 * end.  A batch is local to one ltx_gpu command queue and is not nestable. */
int ltx_gpu_batch_begin(ltx_gpu *gpu, char *error, size_t error_size);
int ltx_gpu_batch_end(ltx_gpu *gpu, char *error, size_t error_size);

ltx_gpu_buffer *ltx_gpu_buffer_new(ltx_gpu *gpu, size_t bytes,
                                   char *error, size_t error_size);
ltx_gpu_buffer *ltx_gpu_buffer_new_copy(ltx_gpu *gpu, const void *data,
                                        size_t bytes,
                                        char *error, size_t error_size);
void ltx_gpu_buffer_free(ltx_gpu_buffer *buffer);
size_t ltx_gpu_buffer_bytes(const ltx_gpu_buffer *buffer);
void *ltx_gpu_buffer_contents(ltx_gpu_buffer *buffer);
int ltx_gpu_buffer_write(ltx_gpu_buffer *buffer, const void *data,
                         size_t bytes, char *error, size_t error_size);
int ltx_gpu_buffer_read(const ltx_gpu_buffer *buffer, void *data,
                        size_t bytes, char *error, size_t error_size);

int ltx_gpu_add_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                    const ltx_gpu_buffer *left,
                    const ltx_gpu_buffer *right, uint32_t elements,
                    char *error, size_t error_size);
int ltx_gpu_add_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                     const ltx_gpu_buffer *left,
                     const ltx_gpu_buffer *right, uint32_t elements,
                     char *error, size_t error_size);
int ltx_gpu_scale_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                      const ltx_gpu_buffer *input, float scale,
                      uint32_t elements, char *error, size_t error_size);
int ltx_gpu_velocity_to_denoised_bf16(
                      ltx_gpu *gpu, ltx_gpu_buffer *output,
                      const ltx_gpu_buffer *sample,
                      const ltx_gpu_buffer *velocity,
                      uint32_t elements, float sigma,
                      char *error, size_t error_size);
int ltx_gpu_velocity_to_denoised_bf16_split(
                      ltx_gpu *gpu, ltx_gpu_buffer *output,
                      const ltx_gpu_buffer *sample,
                      const ltx_gpu_buffer *velocity,
                      uint32_t rows, uint32_t columns,
                      uint32_t conditioned_prefix_rows,
                      float generated_sigma, float conditioned_sigma,
                      char *error, size_t error_size);
int ltx_gpu_euler_step_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *sample,
                            const ltx_gpu_buffer *denoised,
                            uint32_t elements, float sigma,
                            float sigma_next,
                            char *error, size_t error_size);
int ltx_gpu_euler_ancestral_step_bf16(
                            ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *sample,
                            const ltx_gpu_buffer *denoised,
                            const ltx_gpu_buffer *noise_f32,
                            uint32_t elements, float sigma,
                            float sigma_next, float eta, float s_noise,
                            char *error, size_t error_size);
int ltx_gpu_renoise_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *clean,
                         const ltx_gpu_buffer *noise,
                         uint32_t elements, float sigma,
                         char *error, size_t error_size);
/* Blend only the leading elements in-place: output = output * mask +
 * clean_prefix * (1 - mask). */
int ltx_gpu_condition_prefix_bf16(
                         ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *clean_prefix,
                         uint32_t prefix_elements, float mask,
                         char *error, size_t error_size);
int ltx_gpu_latent_stats_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                              const ltx_gpu_buffer *input,
                              const ltx_gpu_buffer *mean,
                              const ltx_gpu_buffer *standard_deviation,
                              uint32_t rows, uint32_t channels,
                              int normalize,
                              char *error, size_t error_size);
int ltx_gpu_residual_gate_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                              const ltx_gpu_buffer *residual,
                              const ltx_gpu_buffer *branch,
                              const ltx_gpu_buffer *gate,
                              uint32_t elements,
                              char *error, size_t error_size);
int ltx_gpu_residual_gate_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                               const ltx_gpu_buffer *residual,
                               const ltx_gpu_buffer *branch,
                               const ltx_gpu_buffer *gate,
                               uint32_t rows, uint32_t columns,
                               uint32_t gate_rows,
                               char *error, size_t error_size);
int ltx_gpu_residual_gate_bf16_split(
                               ltx_gpu *gpu, ltx_gpu_buffer *output,
                               const ltx_gpu_buffer *residual,
                               const ltx_gpu_buffer *branch,
                               const ltx_gpu_buffer *generated_gate,
                               const ltx_gpu_buffer *conditioned_gate,
                               uint32_t rows, uint32_t columns,
                               uint32_t conditioned_prefix_rows,
                               char *error, size_t error_size);
int ltx_gpu_affine_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *scale,
                        const ltx_gpu_buffer *shift,
                        uint32_t rows, uint32_t columns,
                        uint32_t parameter_rows,
                        char *error, size_t error_size);
int ltx_gpu_affine_bf16_split(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *generated_scale,
                        const ltx_gpu_buffer *generated_shift,
                        const ltx_gpu_buffer *conditioned_scale,
                        const ltx_gpu_buffer *conditioned_shift,
                        uint32_t rows, uint32_t columns,
                        uint32_t conditioned_prefix_rows,
                        char *error, size_t error_size);
int ltx_gpu_slice_columns_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                               const ltx_gpu_buffer *input,
                               uint32_t rows, uint32_t input_columns,
                               uint32_t start_column,
                               uint32_t output_columns,
                               char *error, size_t error_size);
int ltx_gpu_cast_f32_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *input, uint32_t elements,
                         char *error, size_t error_size);
int ltx_gpu_cast_f16_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *input, uint32_t elements,
                         char *error, size_t error_size);
int ltx_gpu_cast_f32_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size);
int ltx_gpu_cast_bf16_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size);
int ltx_gpu_cast_bf16_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size);
int ltx_gpu_cast_f16_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size);
int ltx_gpu_slice_rows_bf16_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                                const ltx_gpu_buffer *input,
                                uint32_t input_rows, uint32_t columns,
                                uint32_t start_row, uint32_t output_rows,
                                char *error, size_t error_size);
int ltx_gpu_concat_rows_bf16_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                                 const ltx_gpu_buffer *bf16_prefix,
                                 const ltx_gpu_buffer *f16_suffix,
                                 uint32_t prefix_rows,
                                 uint32_t suffix_rows,
                                 uint32_t columns,
                                 char *error, size_t error_size);
int ltx_gpu_join_bf16_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *gpu_partial,
                          const ltx_gpu_buffer *ane_partial,
                          uint32_t elements,
                          char *error, size_t error_size);
int ltx_gpu_join_residual_gate_bf16_f16(
                          ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *residual,
                          const ltx_gpu_buffer *gpu_partial,
                          const ltx_gpu_buffer *ane_partial,
                          const ltx_gpu_buffer *gate,
                          uint32_t rows, uint32_t columns,
                          uint32_t gate_rows,
                          char *error, size_t error_size);
int ltx_gpu_gelu_tanh_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size);
int ltx_gpu_gelu_tanh_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                           const ltx_gpu_buffer *input, uint32_t elements,
                           char *error, size_t error_size);
int ltx_gpu_silu_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                      const ltx_gpu_buffer *input, uint32_t elements,
                      char *error, size_t error_size);
int ltx_gpu_sigmoid2_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size);
int ltx_gpu_rms_norm_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *input,
                         uint32_t rows, uint32_t columns, float epsilon,
                         char *error, size_t error_size);
int ltx_gpu_rms_norm_weighted_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                                  const ltx_gpu_buffer *input,
                                  const ltx_gpu_buffer *weight,
                                  uint32_t rows, uint32_t columns,
                                  float epsilon,
                                  char *error, size_t error_size);
int ltx_gpu_rms_norm_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input,
                          uint32_t rows, uint32_t columns, float epsilon,
                          char *error, size_t error_size);
int ltx_gpu_rms_norm_weighted_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                                   const ltx_gpu_buffer *input,
                                   const ltx_gpu_buffer *weight,
                                   uint32_t rows, uint32_t columns,
                                   float epsilon,
                                   char *error, size_t error_size);
int ltx_gpu_adaln_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                       const ltx_gpu_buffer *input,
                       const ltx_gpu_buffer *scale,
                       const ltx_gpu_buffer *shift,
                       uint32_t rows, uint32_t columns,
                       uint32_t parameter_rows, float epsilon,
                       char *error, size_t error_size);
int ltx_gpu_adaln_bf16_split(
                       ltx_gpu *gpu, ltx_gpu_buffer *output,
                       const ltx_gpu_buffer *input,
                       const ltx_gpu_buffer *generated_scale,
                       const ltx_gpu_buffer *generated_shift,
                       const ltx_gpu_buffer *conditioned_scale,
                       const ltx_gpu_buffer *conditioned_shift,
                       uint32_t rows, uint32_t columns,
                       uint32_t conditioned_prefix_rows, float epsilon,
                       char *error, size_t error_size);
int ltx_gpu_adaln_bf16_f16(ltx_gpu *gpu,
                       ltx_gpu_buffer *output_bf16,
                       ltx_gpu_buffer *output_f16,
                       const ltx_gpu_buffer *input,
                       const ltx_gpu_buffer *scale,
                       const ltx_gpu_buffer *shift,
                       uint32_t rows, uint32_t columns,
                       uint32_t parameter_rows, float epsilon,
                       char *error, size_t error_size);
int ltx_gpu_linear_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                       const ltx_gpu_buffer *input,
                       const ltx_gpu_buffer *weight,
                       const ltx_gpu_buffer *bias,
                       uint32_t rows, uint32_t input_dim,
                       uint32_t output_dim,
                       char *error, size_t error_size);
int ltx_gpu_linear_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *weight,
                        const ltx_gpu_buffer *bias,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t output_dim,
                        char *error, size_t error_size);
int ltx_gpu_linear_mps_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *input,
                            const ltx_gpu_buffer *weight,
                            const ltx_gpu_buffer *bias,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            char *error, size_t error_size);
int ltx_gpu_output_adaln_bf16(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *embedded_timestep,
                        const ltx_gpu_buffer *shift_table,
                        const ltx_gpu_buffer *scale_table,
                        uint32_t rows, uint32_t columns, float epsilon,
                        char *error, size_t error_size);
int ltx_gpu_output_adaln_bf16_split(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *generated_embedded_timestep,
                        const ltx_gpu_buffer *conditioned_embedded_timestep,
                        const ltx_gpu_buffer *shift_table,
                        const ltx_gpu_buffer *scale_table,
                        uint32_t rows, uint32_t columns,
                        uint32_t conditioned_prefix_rows, float epsilon,
                        char *error, size_t error_size);
int ltx_gpu_adaln_single_mps_bf16(
                        ltx_gpu *gpu,
                        ltx_gpu_buffer *parameters,
                        ltx_gpu_buffer *embedded_timestep,
                        const ltx_gpu_buffer *timestep_embedding,
                        const ltx_gpu_buffer *linear1_weight,
                        const ltx_gpu_buffer *linear1_bias,
                        const ltx_gpu_buffer *linear2_weight,
                        const ltx_gpu_buffer *linear2_bias,
                        const ltx_gpu_buffer *parameter_weight,
                        const ltx_gpu_buffer *parameter_bias,
                        uint32_t rows, uint32_t timestep_dim,
                        uint32_t hidden_dim, uint32_t parameter_count,
                        char *error, size_t error_size);
int ltx_gpu_convrot_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *input,
                         uint32_t rows, uint32_t columns,
                         uint32_t group_size,
                         char *error, size_t error_size);
int ltx_gpu_pack_heads_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *input,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim,
                            char *error, size_t error_size);
int ltx_gpu_pack_rope_split_bf16(
                            ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *input,
                            const ltx_gpu_buffer *cosine,
                            const ltx_gpu_buffer *sine,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim,
                            char *error, size_t error_size);
int ltx_gpu_unpack_heads_gate_bf16(
                            ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *input,
                            const ltx_gpu_buffer *gate,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim,
                            char *error, size_t error_size);
int ltx_gpu_sdpa_mps_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *query,
                          const ltx_gpu_buffer *key,
                          const ltx_gpu_buffer *value,
                          uint32_t heads, uint32_t query_rows,
                          uint32_t key_value_rows, uint32_t head_dim,
                          float scale,
                          char *error, size_t error_size);
/* Gemma4 self-attention over already projected/normed row-major Q/K/V.
 * Q has query_heads*head_dim columns; K/V have kv_heads*head_dim columns.
 * The graph applies split RoPE, GQA repeat, and a causal mask. window=0
 * means full causal attention; otherwise keys older than query-window are
 * masked. Gemma4 uses scale=1.0 at this boundary. */
int ltx_gpu_gemma_attention_mps_bf16(
                          ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *query,
                          const ltx_gpu_buffer *key,
                          const ltx_gpu_buffer *value,
                          const ltx_gpu_buffer *cosine,
                          const ltx_gpu_buffer *sine,
                          uint32_t rows, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim,
                          uint32_t window, char *error, size_t error_size);
int ltx_gpu_self_attention_core_mps_bf16(
                          ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *query,
                          const ltx_gpu_buffer *key,
                          const ltx_gpu_buffer *value,
                          const ltx_gpu_buffer *cosine,
                          const ltx_gpu_buffer *sine,
                          const ltx_gpu_buffer *gate,
                          uint32_t rows, uint32_t heads,
                          uint32_t head_dim, float scale,
                          char *error, size_t error_size);
int ltx_gpu_self_attention_core_sol_bf16(
                          ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *query,
                          const ltx_gpu_buffer *key,
                          const ltx_gpu_buffer *value,
                          const ltx_gpu_buffer *cosine,
                          const ltx_gpu_buffer *sine,
                          const ltx_gpu_buffer *gate,
                          ltx_gpu_buffer *packed_query,
                          ltx_gpu_buffer *packed_key,
                          ltx_gpu_buffer *packed_value,
                          ltx_gpu_buffer *packed_output,
                          ltx_gpu_buffer *query_centroids,
                          ltx_gpu_buffer *key_centroids,
                          ltx_gpu_buffer *value_sums,
                          ltx_gpu_buffer *thresholds,
                          ltx_gpu_buffer *routes,
                          uint32_t rows, uint32_t heads,
                          uint32_t head_dim, float scale, float tau,
                          uint32_t sink_start, uint32_t sink_end,
                          uint32_t sink_query_start,
                          uint32_t sink_query_end,
                          char *error, size_t error_size);
int ltx_gpu_self_attention_int8_mps_bf16(
                          ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input,
                          const ltx_gpu_buffer *query_weight,
                          const ltx_gpu_buffer *query_scale,
                          const ltx_gpu_buffer *query_bias,
                          const ltx_gpu_buffer *key_weight,
                          const ltx_gpu_buffer *key_scale,
                          const ltx_gpu_buffer *key_bias,
                          const ltx_gpu_buffer *value_weight,
                          const ltx_gpu_buffer *value_scale,
                          const ltx_gpu_buffer *value_bias,
                          const ltx_gpu_buffer *query_norm_weight,
                          const ltx_gpu_buffer *key_norm_weight,
                          const ltx_gpu_buffer *gate_weight,
                          const ltx_gpu_buffer *gate_bias,
                          const ltx_gpu_buffer *output_weight,
                          const ltx_gpu_buffer *output_scale,
                          const ltx_gpu_buffer *output_bias,
                          const ltx_gpu_buffer *cosine,
                          const ltx_gpu_buffer *sine,
                          uint32_t rows, uint32_t heads,
                          uint32_t head_dim, uint32_t convrot_group_size,
                          float norm_epsilon,
                          char *error, size_t error_size);
int ltx_gpu_cross_attention_int8_mps_bf16(
                          ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *query_input,
                          const ltx_gpu_buffer *key_value_input,
                          const ltx_gpu_buffer *query_weight,
                          const ltx_gpu_buffer *query_scale,
                          const ltx_gpu_buffer *query_bias,
                          const ltx_gpu_buffer *key_weight,
                          const ltx_gpu_buffer *key_scale,
                          const ltx_gpu_buffer *key_bias,
                          const ltx_gpu_buffer *value_weight,
                          const ltx_gpu_buffer *value_scale,
                          const ltx_gpu_buffer *value_bias,
                          const ltx_gpu_buffer *query_norm_weight,
                          const ltx_gpu_buffer *key_norm_weight,
                          const ltx_gpu_buffer *gate_weight,
                          const ltx_gpu_buffer *gate_bias,
                          const ltx_gpu_buffer *output_weight,
                          const ltx_gpu_buffer *output_scale,
                          const ltx_gpu_buffer *output_bias,
                          const ltx_gpu_buffer *query_cosine,
                          const ltx_gpu_buffer *query_sine,
                          const ltx_gpu_buffer *key_cosine,
                          const ltx_gpu_buffer *key_sine,
                          uint32_t query_rows,
                          uint32_t key_value_rows,
                          uint32_t query_dim,
                          uint32_t key_value_dim,
                          uint32_t heads, uint32_t head_dim,
                          uint32_t output_dim,
                          uint32_t convrot_group_size,
                          float norm_epsilon,
                          char *error, size_t error_size);
int ltx_gpu_cross_attention_int8_mps_bf16_masked(
                          ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *query_input,
                          const ltx_gpu_buffer *key_value_input,
                          const ltx_gpu_buffer *query_weight,
                          const ltx_gpu_buffer *query_scale,
                          const ltx_gpu_buffer *query_bias,
                          const ltx_gpu_buffer *key_weight,
                          const ltx_gpu_buffer *key_scale,
                          const ltx_gpu_buffer *key_bias,
                          const ltx_gpu_buffer *value_weight,
                          const ltx_gpu_buffer *value_scale,
                          const ltx_gpu_buffer *value_bias,
                          const ltx_gpu_buffer *query_norm_weight,
                          const ltx_gpu_buffer *key_norm_weight,
                          const ltx_gpu_buffer *gate_weight,
                          const ltx_gpu_buffer *gate_bias,
                          const ltx_gpu_buffer *output_weight,
                          const ltx_gpu_buffer *output_scale,
                          const ltx_gpu_buffer *output_bias,
                          const ltx_gpu_buffer *query_cosine,
                          const ltx_gpu_buffer *query_sine,
                          const ltx_gpu_buffer *key_cosine,
                          const ltx_gpu_buffer *key_sine,
                          const ltx_gpu_buffer *attention_mask,
                          uint32_t attention_mask_rows,
                          uint32_t query_rows,
                          uint32_t key_value_rows,
                          uint32_t query_dim,
                          uint32_t key_value_dim,
                          uint32_t heads, uint32_t head_dim,
                          uint32_t output_dim,
                          uint32_t convrot_group_size,
                          float norm_epsilon,
                          char *error, size_t error_size);
int ltx_gpu_cross_attention_kv_int8_mps_bf16(
                          ltx_gpu *gpu,
                          ltx_gpu_buffer *key_output,
                          ltx_gpu_buffer *value_output,
                          const ltx_gpu_buffer *key_value_input,
                          const ltx_gpu_buffer *key_weight,
                          const ltx_gpu_buffer *key_scale,
                          const ltx_gpu_buffer *key_bias,
                          const ltx_gpu_buffer *value_weight,
                          const ltx_gpu_buffer *value_scale,
                          const ltx_gpu_buffer *value_bias,
                          const ltx_gpu_buffer *key_norm_weight,
                          uint32_t key_value_rows,
                          uint32_t key_value_dim,
                          uint32_t heads, uint32_t head_dim,
                          uint32_t convrot_group_size,
                          float norm_epsilon,
                          char *error, size_t error_size);
int ltx_gpu_cross_attention_query_int8_mps_bf16_masked(
                          ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *query_input,
                          const ltx_gpu_buffer *key,
                          const ltx_gpu_buffer *value,
                          const ltx_gpu_buffer *query_weight,
                          const ltx_gpu_buffer *query_scale,
                          const ltx_gpu_buffer *query_bias,
                          const ltx_gpu_buffer *query_norm_weight,
                          const ltx_gpu_buffer *gate_weight,
                          const ltx_gpu_buffer *gate_bias,
                          const ltx_gpu_buffer *output_weight,
                          const ltx_gpu_buffer *output_scale,
                          const ltx_gpu_buffer *output_bias,
                          const ltx_gpu_buffer *attention_mask,
                          uint32_t attention_mask_rows,
                          uint32_t query_rows,
                          uint32_t key_value_rows,
                          uint32_t query_dim,
                          uint32_t heads, uint32_t head_dim,
                          uint32_t output_dim,
                          uint32_t convrot_group_size,
                          float norm_epsilon,
                          char *error, size_t error_size);
int ltx_gpu_linear_int8_weight_bf16(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *rotated_input,
                        const ltx_gpu_buffer *weight,
                        const ltx_gpu_buffer *weight_scale,
                        const ltx_gpu_buffer *bias,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t output_dim,
                        char *error, size_t error_size);
int ltx_gpu_linear_int8_weight_mps_bf16(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *rotated_input,
                        const ltx_gpu_buffer *weight,
                        const ltx_gpu_buffer *weight_scale,
                        const ltx_gpu_buffer *bias,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t output_dim,
                        char *error, size_t error_size);
int ltx_gpu_linear_int8_convrot_mps_bf16(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *weight,
                        const ltx_gpu_buffer *weight_scale,
                        const ltx_gpu_buffer *bias,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t output_dim,
                        uint32_t convrot_group_size,
                        char *error, size_t error_size);
int ltx_gpu_mlp_int8_convrot_mps_bf16(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *fc1_weight,
                        const ltx_gpu_buffer *fc1_scale,
                        const ltx_gpu_buffer *fc1_bias,
                        const ltx_gpu_buffer *fc2_weight,
                        const ltx_gpu_buffer *fc2_scale,
                        const ltx_gpu_buffer *fc2_bias,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t hidden_dim, uint32_t output_dim,
                        uint32_t convrot_group_size,
                        char *error, size_t error_size);
int ltx_gpu_qkv_int8_convrot_mps_bf16(
                        ltx_gpu *gpu,
                        ltx_gpu_buffer *query_output,
                        ltx_gpu_buffer *key_output,
                        ltx_gpu_buffer *value_output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *query_weight,
                        const ltx_gpu_buffer *query_scale,
                        const ltx_gpu_buffer *query_bias,
                        const ltx_gpu_buffer *key_weight,
                        const ltx_gpu_buffer *key_scale,
                        const ltx_gpu_buffer *key_bias,
                        const ltx_gpu_buffer *value_weight,
                        const ltx_gpu_buffer *value_scale,
                        const ltx_gpu_buffer *value_bias,
                        const ltx_gpu_buffer *query_norm_weight,
                        const ltx_gpu_buffer *key_norm_weight,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t inner_dim, uint32_t convrot_group_size,
                        float norm_epsilon,
                        char *error, size_t error_size);
int ltx_gpu_qkv_packed_int8_convrot_mps_bf16(
                        ltx_gpu *gpu,
                        ltx_gpu_buffer *query_output,
                        ltx_gpu_buffer *key_output,
                        ltx_gpu_buffer *value_output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *packed_weight,
                        const ltx_gpu_buffer *packed_scale,
                        const ltx_gpu_buffer *packed_bias,
                        const ltx_gpu_buffer *query_norm_weight,
                        const ltx_gpu_buffer *key_norm_weight,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t inner_dim, uint32_t convrot_group_size,
                        float norm_epsilon,
                        char *error, size_t error_size);

#endif
