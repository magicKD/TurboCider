#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
/* Non-owning Metal objects; the C runtime owns all scratch and queue lifetime. */
int ltx_w8a8_supported(void *device);
void *ltx_w8a8_create(void *device, char *error, size_t error_size);
void ltx_w8a8_free(void *context);
int ltx_w8a8_fused_output_enabled(void);
/* Compile the optional activated epilogues before encoding a projection. */
int ltx_w8a8_prepare_gelu_output(void *context, char *error, size_t error_size);
void ltx_w8a8_quantize(void *context, void *encoder, void *rotated,
                     void *codes, void *token_scales, uint32_t rows, uint32_t columns);
void ltx_w8a8_gated_product(void *context, void *encoder, void *gate,
                           void *up, void *output, uint32_t count);
void ltx_w8a8_project(void *context, void *encoder, void *codes,
                    void *weight, void *weight_scales, void *token_scales,
                    void *fp32_scratch, void *output, void *bias,
                    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                    int fused_output);
/* gelu_output requires fused_output and a successful prepare call. */
void ltx_w8a8_project_activation(void *context, void *encoder, void *codes,
                    void *weight, void *weight_scales, void *token_scales,
                    void *fp32_scratch, void *output, void *bias,
                    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                    int fused_output, int gelu_output);
#ifdef __cplusplus
}
#endif
