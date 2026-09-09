#ifndef H3_GPU_H
#define H3_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct h3_gpu h3_gpu;
typedef struct h3_gpu_tensor h3_gpu_tensor;

typedef enum {
    H3_GPU_F32 = 0,
    H3_GPU_F16,
    H3_GPU_BF16,
    H3_GPU_I8,
    H3_GPU_U32
} h3_gpu_dtype;

typedef struct {
    uint64_t allocated_bytes;
    uint64_t live_bytes;
    uint64_t peak_live_bytes;
    uint64_t tensor_allocations;
    uint64_t direct_dispatches;
    uint64_t mps_linear_dispatches;
    uint64_t mps_conv_dispatches;
    uint64_t mps_sdpa_dispatches;
    uint64_t blit_copies;
    uint64_t submissions;
    double command_encode_seconds;
    double command_wait_seconds;
    /* Root MTLCommandBuffer timestamps; MPSGraph may schedule child buffers,
     * so command_wait_seconds is the complete turnaround measurement. */
    double gpu_seconds;
} h3_gpu_stats;

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size);
void h3_gpu_free(h3_gpu *gpu);
int h3_gpu_is_m5(const h3_gpu *gpu);
int h3_gpu_has_nax_mlp(const h3_gpu *gpu);
int h3_gpu_has_int8_mlp(const h3_gpu *gpu);

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_new_f16(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements);
/* Allocate shared Metal storage and pread BF16 payload directly into it. */
h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_load_i8(h3_gpu *gpu, const char *path,
                                     uint64_t file_offset, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements);
/* Map a page-aligned BF16 file range into a GPU-readable shared buffer using
 * a read-only MAP_PRIVATE mapping. This always maps, independently of
 * H3_ZERO_COPY_WEIGHTS, and is intended for immutable fallback weights. */
h3_gpu_tensor *h3_gpu_tensor_map_bf16(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements);
/* Fill an existing shared BF16 buffer from a file. The tensor and its
 * accounting are unchanged, so this may run on an I/O thread while another
 * tensor is in flight on the GPU. */
int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size);
/* As above, but ask Darwin to avoid retaining a second copy in the file cache.
 * Intended for large sequential weight streams whose destination is the only
 * useful resident copy. */
int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size);
int h3_gpu_tensor_stream_file_i8(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size);
int h3_gpu_tensor_stream_file_f32(h3_gpu_tensor *tensor, const char *path,
                                  uint64_t file_offset, size_t elements,
                                  char *error, size_t error_size);
void h3_gpu_tensor_free(h3_gpu_tensor *tensor);
size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor);
size_t h3_gpu_tensor_bytes(const h3_gpu_tensor *tensor);
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor);
int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements);
int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements);
int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements);
int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements);
int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements);
int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements);
int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements);
void *h3_gpu_tensor_host_pointer(h3_gpu_tensor *tensor);

/* Experimental private-ANE staging. IOSurface planes use channel-major F32
 * because that is the stable request format of the private ANE bridge. */
h3_gpu_tensor *h3_gpu_tensor_wrap_f32(h3_gpu *gpu, void *base,
                                      size_t elements);
int h3_gpu_pack_ane_input_bf16(h3_gpu *gpu, h3_gpu_tensor *plane,
                               const h3_gpu_tensor *input, uint32_t rows,
                               uint32_t input_dim, uint32_t base,
                               uint32_t chunk_dim, uint32_t plane_rows);
int h3_gpu_pack_ane_input_bf16_rows(h3_gpu *gpu, h3_gpu_tensor *plane,
                                    const h3_gpu_tensor *input,
                                    uint32_t input_row_offset,
                                    uint32_t rows, uint32_t input_dim,
                                    uint32_t base, uint32_t chunk_dim,
                                    uint32_t plane_rows);
int h3_gpu_unpack_ane_output_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *plane, uint32_t rows,
                                  uint32_t output_dim, uint32_t plane_rows,
                                  float scale);
int h3_gpu_unpack_ane_output_bf16_rows_checked(
                                  h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *plane,
                                  h3_gpu_tensor *nonfinite_flag,
                                  uint32_t output_row_offset, uint32_t rows,
                                  uint32_t output_dim, uint32_t plane_rows,
                                  float scale, uint32_t block_index);
/* Row-split unpack with a two-element F32 stats tensor. Element zero stores
 * the raw uint32 nonfinite flag bits; element one stores the raw uint32 bits
 * of the maximum finite absolute ANE value before the caller's output scale.
 * The latter lets the runtime preserve FP16 accumulator headroom before an
 * actual NaN/Inf occurs. */
int h3_gpu_unpack_ane_output_bf16_rows_checked_range(
                                  h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *plane,
                                  h3_gpu_tensor *range_stats,
                                  uint32_t output_row_offset, uint32_t rows,
                                  uint32_t output_dim, uint32_t plane_rows,
                                  float scale, uint32_t block_index);
int h3_gpu_join_ane_linear_output_bf16_checked(
                                  h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *ane_prefix,
                                  const h3_gpu_tensor *gpu_suffix,
                                  h3_gpu_tensor *nonfinite_flag,
                                  uint32_t rows, uint32_t ane_width,
                                  uint32_t gpu_width, uint32_t plane_rows,
                                  uint32_t block_index);

int h3_gpu_begin(h3_gpu *gpu);
/* Commit the current command buffer without waiting, then continue encoding on
 * the same ordered queue. h3_gpu_submit() waits and validates the whole chain. */
int h3_gpu_continue(h3_gpu *gpu);
/* Asynchronously release tensor ownership after the current command buffer
 * completes, then continue encoding on the same ordered queue. On success this
 * consumes every non-NULL tensor reference in the array; callers must clear
 * their corresponding fields. This avoids a CPU synchronization point while
 * allowing final-pass model weights to leave the working set progressively. */
int h3_gpu_continue_releasing(h3_gpu *gpu,
                              h3_gpu_tensor *const *tensors,
                              size_t tensor_count);
int h3_gpu_submit(h3_gpu *gpu);
const char *h3_gpu_error(const h3_gpu *gpu);
int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats);
/* Optional benchmark labels. With H3_PROFILE set, marks and context teardown
 * print wall time alongside command-buffer GPU time and allocation counters. */
void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label);
void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase);

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements);
int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements);
int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon);
int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon);
int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale);
int h3_gpu_sdpa_batch_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *query,
                          const h3_gpu_tensor *key,
                          const h3_gpu_tensor *value, uint32_t batch,
                          uint32_t sequence, uint32_t heads,
                          uint32_t head_dim, float scale);
int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width);
int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width);
int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon);
int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon);

/* H3 AudioVAE uses time-major [batch,length,channels] activations and stores
 * Conv1d/ConvTranspose1d weights in PyTorch OIK/IOK order respectively. */
int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation);
int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation);
int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding);
/* TAEH3 uses channels-last [batch,height,width,channels] BF16 activations and
 * PyTorch OIHW weights. The current decoder needs only stride-1 convolution. */
int h3_gpu_conv2d_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t batch,
                       uint32_t height, uint32_t width,
                       uint32_t input_channels,
                       uint32_t output_channels, uint32_t kernel,
                       uint32_t padding, int relu);
int h3_gpu_taeh3_input_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t time,
                            uint32_t height, uint32_t width,
                            uint32_t channels);
int h3_gpu_taeh3_temporal_concat_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t time,
                            uint32_t height, uint32_t width,
                            uint32_t channels);
int h3_gpu_taeh3_add_relu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *residual,
                               const h3_gpu_tensor *branch,
                               uint32_t elements);
int h3_gpu_taeh3_upsample2_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input, uint32_t time,
                                uint32_t height, uint32_t width,
                                uint32_t channels);
int h3_gpu_taeh3_tgrow_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t time,
                            uint32_t height, uint32_t width,
                            uint32_t channels, uint32_t stride);
int h3_gpu_taeh3_output_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t raw_time,
                            uint32_t height, uint32_t width,
                            uint32_t frames);
int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner);
int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements);
int h3_gpu_alias_free_snake_f32(
                          h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels);
int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels);
int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim);
int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale);
int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim);
int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements);
int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum);

/* Visual-VAE encoder tensors use channels-last [B,T,H,W,C] storage. Spatial
 * padding reflects pixels while temporal front padding is zero-filled. */
int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after);
int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width);
int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon);

/* Portable BF16 storage path. Arithmetic accumulates in F32 and rounds at
 * operation boundaries, matching the released checkpoint's compute dtype. */
int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim);
int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim);
/* Experimental M5 Metal 4 paired FC1/SwiGLU plus direct FC2 path. Available
 * only when the context was created with H3_NAX=mlp. */
int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim);
/* Experimental M5 Metal 4 int8 MLP. Weights use one F32 scale per output
 * channel; activations are quantized dynamically with one F32 scale per row. */
int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns);
int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales);
/* Same projection when a preceding fused epilogue already produced the
 * row-wise INT8 activation and scales for `input`. */
int h3_gpu_linear_int8_bf16_prequantized(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *quantized_input,
                            const h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales);
/* Consume SDPA's native [head,row,dimension] BF16 layout without a full
 * BF16 transpose, gathering directly into the projection's row-major int8. */
int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim);
int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized);
int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon);
int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon);
int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate);
int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half);
int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3_gpu_adaln_linear_bf16(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon);
int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3_gpu_gate_adaln_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon);
int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon);
/* H3 checkpoint QKV rows are [head, q/k/v, dimension], unlike the
 * conventional [q/k/v, head, dimension] layout accepted above. */
int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon);
/* Consume one complete-head QKV segment and write it into a larger Q/K/V
 * tensor at head_offset. The BF16 form accepts the GPU complement in native
 * row-major grouped layout. The Core ML form consumes its channel-major FP16
 * output directly, avoiding a full transpose/concat before attention. */
int h3_gpu_grouped_qkv_rope_bf16_segment(
                                 h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t segment_heads,
                                 uint32_t total_heads, uint32_t head_offset,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon);
int h3_gpu_grouped_qkv_rope_coreml_f16_segment(
                                 h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 h3_gpu_tensor *nonfinite_flag,
                                 uint32_t sequence, uint32_t segment_heads,
                                 uint32_t total_heads, uint32_t head_offset,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon, uint32_t block_index);
int h3_gpu_grouped_qkv_rope_ane_f32_segment(
                                 h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 h3_gpu_tensor *nonfinite_flag,
                                 uint32_t sequence, uint32_t plane_rows,
                                 uint32_t segment_heads,
                                 uint32_t total_heads, uint32_t head_offset,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon, uint32_t block_index);
/* Project grouped H3 QKV and apply its exact Q/K norm/RoPE boundary. Metal 4
 * may route projections directly into the attention layout; other devices
 * retain the ordinary two calls. */
int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon);
int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales);
int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Evaluate dense SDPA and its bias-free BF16 output projection in one
 * fixed-shape MPSGraph. The graph accepts either row-major or the internal
 * head-major Q/K/V layout selected by the preceding grouped QKV operation. */
int h3_gpu_sdpa_output_linear_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value,
                     const h3_gpu_tensor *weight, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t output_dim, float scale);
/* Public-Metal Sol-Attn prototype. Scratch tensors are reusable across
 * layers: QC/thresholds/routes are F32, KC/VS are BF16. Output layout may be
 * row-major or head-major for the following projection. */
int h3_gpu_sol_sdpa_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value,
                     h3_gpu_tensor *query_centroids,
                     h3_gpu_tensor *key_centroids,
                     h3_gpu_tensor *value_sums,
                     h3_gpu_tensor *thresholds,
                     h3_gpu_tensor *routes,
                     uint32_t rows, uint32_t heads, uint32_t head_dim,
                     float scale, float tau,
                     uint32_t sink_start, uint32_t sink_end,
                     uint32_t sink_q_start, uint32_t sink_q_end,
                     int output_head_major);
/* TensorOps BQ64 all-exact bring-up. Q/K/V and output are head-major
 * [head,row,dimension]; head_dim is currently fixed at 128. */
int h3_gpu_sdpa_bf16_nax_bq64_all_exact(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t rows,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Tiled sparse Sol-Attn: summary/routing scratch matches h3_gpu_sol_sdpa_bf16,
 * while routed exact blocks and summary PV use the BQ64 TensorOps kernel.
 * Q/K/V and output are head-major. */
int h3_gpu_sol_sdpa_bf16_nax_bq64(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value,
                     h3_gpu_tensor *query_centroids,
                     h3_gpu_tensor *key_centroids,
                     h3_gpu_tensor *value_sums,
                     h3_gpu_tensor *thresholds,
                     h3_gpu_tensor *routes,
                     uint32_t rows, uint32_t heads, uint32_t head_dim,
                     float scale, float tau,
                     uint32_t sink_start, uint32_t sink_end,
                     uint32_t sink_q_start, uint32_t sink_q_end);
/* Preserve SDPA's native [head,row,dimension] output for an immediately
 * following layout-aware projection. */
int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width);
int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width);
int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon);
int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon);
int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim);
int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale);
int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements);
int h3_gpu_pack_coreml_f16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_pack_coreml_f16_transpose(h3_gpu *gpu, h3_gpu_tensor *output,
                                     const h3_gpu_tensor *input,
                                     uint32_t rows, uint32_t hidden);
int h3_gpu_add_coreml_f16_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *gpu_partial,
                               const h3_gpu_tensor *ane_partial,
                               uint32_t elements);
int h3_gpu_add_coreml_f16_bf16_transpose(
                               h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *gpu_partial,
                               const h3_gpu_tensor *ane_partial,
                               uint32_t rows, uint32_t hidden,
                               float ane_scale);
/* Join the ANE partial while replacing non-finite FP16 values with zero and
 * atomically recording block_index + 1 in a one-element F32 flag tensor.
 * The flag is read as raw uint32 bits after the enclosing command completes. */
int h3_gpu_add_coreml_f16_bf16_transpose_checked(
                               h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *gpu_partial,
                               const h3_gpu_tensor *ane_partial,
                               h3_gpu_tensor *nonfinite_flag,
                               uint32_t rows, uint32_t hidden,
                               float ane_scale, uint32_t block_index);
int h3_gpu_add_ane_f32_bf16_transpose_checked(
                               h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *gpu_partial,
                               const h3_gpu_tensor *ane_partial,
                               h3_gpu_tensor *nonfinite_flag,
                               uint32_t rows, uint32_t hidden,
                               uint32_t plane_rows, float ane_scale,
                               uint32_t block_index);
/* Preserve the standalone join and gate BF16 rounding boundaries while
 * fusing the ANE/GPU MLP join, residual gate, and the next block's AdaLN into
 * one dispatch. This removes a full-width branch reread on the private-ANE
 * path without changing the mathematical graph. */
int h3_gpu_add_ane_f32_bf16_transpose_gate_adaln_checked(
                               h3_gpu *gpu,
                               h3_gpu_tensor *branch_output,
                               h3_gpu_tensor *gated_residual,
                               h3_gpu_tensor *norm_output,
                               const h3_gpu_tensor *gpu_partial,
                               const h3_gpu_tensor *ane_partial,
                               h3_gpu_tensor *nonfinite_flag,
                               const h3_gpu_tensor *residual,
                               const h3_gpu_tensor *norm_weight,
                               const h3_gpu_tensor *gate_modulation,
                               const h3_gpu_tensor *norm_modulation,
                               const h3_gpu_tensor *row_map,
                               uint32_t rows, uint32_t hidden,
                               uint32_t plane_rows, uint32_t slots,
                               uint32_t gate_slot, uint32_t shift_slot,
                               uint32_t scale_slot, float ane_scale,
                               float epsilon, uint32_t block_index);
/* As above, but preserve the existing fused gate/AdaLN/int8-quantization
 * boundary used by the next GPU int8 QKV projection. */
int h3_gpu_add_ane_f32_bf16_transpose_gate_adaln_quantize_int8_checked(
                               h3_gpu *gpu,
                               h3_gpu_tensor *branch_output,
                               h3_gpu_tensor *gated_residual,
                               h3_gpu_tensor *quantized_output,
                               h3_gpu_tensor *quantized_scales,
                               const h3_gpu_tensor *gpu_partial,
                               const h3_gpu_tensor *ane_partial,
                               h3_gpu_tensor *nonfinite_flag,
                               const h3_gpu_tensor *residual,
                               const h3_gpu_tensor *norm_weight,
                               const h3_gpu_tensor *gate_modulation,
                               const h3_gpu_tensor *norm_modulation,
                               const h3_gpu_tensor *row_map,
                               uint32_t rows, uint32_t padded_rows,
                               uint32_t hidden, uint32_t plane_rows,
                               uint32_t slots, uint32_t gate_slot,
                               uint32_t shift_slot, uint32_t scale_slot,
                               float ane_scale, float epsilon,
                               uint32_t block_index);
int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements);
/* FirstBlockCache probe. Each F32 output pair contains partial sums of
 * abs((head_output - head_input) - previous_residual) and
 * abs(previous_residual). */
uint32_t h3_gpu_relative_l1_partial_count(uint32_t elements);
int h3_gpu_relative_l1_residual_bf16(
                    h3_gpu *gpu, h3_gpu_tensor *partials,
                    const h3_gpu_tensor *head_output,
                    const h3_gpu_tensor *head_input,
                    const h3_gpu_tensor *previous_residual,
                    uint32_t elements);
/* TeaCache probe. Each F32 output pair contains partial sums of
 * abs(current - previous) and abs(previous). */
int h3_gpu_relative_l1_bf16(h3_gpu *gpu, h3_gpu_tensor *partials,
                            const h3_gpu_tensor *current,
                            const h3_gpu_tensor *previous,
                            uint32_t elements);
/* Encode a direct relative-L1 reduction over a contiguous tensor span. The
 * output begins at partial_group_offset float2 groups in partials. */
int h3_gpu_relative_l1_bf16_range(
                            h3_gpu *gpu, h3_gpu_tensor *partials,
                            uint32_t partial_group_offset,
                            const h3_gpu_tensor *current,
                            const h3_gpu_tensor *previous,
                            uint32_t element_offset, uint32_t elements);
uint32_t h3_gpu_compare_bf16_partial_count(uint32_t elements);
int h3_gpu_compare_bf16(h3_gpu *gpu, h3_gpu_tensor *partials,
                        const h3_gpu_tensor *candidate,
                        const h3_gpu_tensor *reference,
                        uint32_t elements);
/* Diagnostic-only comparison between private-ANE channel-major F32 output
 * and a row-major BF16 prefix of a wider GPU projection. Each partial group
 * stores two float4 values: {error2, reference2, candidate2, dot} and
 * {max_error, max_reference, reference_nonfinite, candidate_nonfinite}. */
uint32_t h3_gpu_compare_ane_f32_bf16_partial_count(uint32_t elements);
int h3_gpu_compare_ane_f32_bf16_prefix(
                        h3_gpu *gpu, h3_gpu_tensor *partials,
                        const h3_gpu_tensor *candidate,
                        const h3_gpu_tensor *reference,
                        uint32_t rows, uint32_t candidate_width,
                        uint32_t reference_width, uint32_t plane_rows);
int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width);
int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale);
int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
/* Apply one Euler step to an F32 sample range from BF16 velocity caches:
 * sample += delta * (last + ratio * (last - previous)). */
int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio);
int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements);

#endif
