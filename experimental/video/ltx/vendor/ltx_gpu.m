#include <dlfcn.h>
#include "ltx_gpu.h"
#include "ltx_gpu_internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    LTX_PIPELINE_ADD_F32 = 0,
    LTX_PIPELINE_ADD_BF16,
    LTX_PIPELINE_SCALE_F32,
    LTX_PIPELINE_VELOCITY_TO_DENOISED_BF16,
    LTX_PIPELINE_EULER_STEP_BF16,
    LTX_PIPELINE_EULER_ANCESTRAL_STEP_BF16,
    LTX_PIPELINE_RENOISE_BF16,
    LTX_PIPELINE_LATENT_STATS_BF16,
    LTX_PIPELINE_RESIDUAL_GATE_F32,
    LTX_PIPELINE_RESIDUAL_GATE_BF16,
    LTX_PIPELINE_AFFINE_BF16,
    LTX_PIPELINE_SLICE_COLUMNS_BF16,
    LTX_PIPELINE_CAST_F32_F16,
    LTX_PIPELINE_CAST_F16_F32,
    LTX_PIPELINE_CAST_F32_BF16,
    LTX_PIPELINE_CAST_BF16_F32,
    LTX_PIPELINE_CAST_BF16_F16,
    LTX_PIPELINE_CAST_F16_BF16,
    LTX_PIPELINE_SLICE_ROWS_BF16_F16,
    LTX_PIPELINE_CONCAT_ROWS_BF16_F16,
    LTX_PIPELINE_JOIN_BF16_F16,
    LTX_PIPELINE_JOIN_RESIDUAL_GATE_BF16_F16,
    LTX_PIPELINE_GELU_TANH_F32,
    LTX_PIPELINE_GELU_TANH_BF16,
    LTX_PIPELINE_SILU_BF16,
    LTX_PIPELINE_SIGMOID2_BF16,
    LTX_PIPELINE_RMS_NORM_F32,
    LTX_PIPELINE_RMS_NORM_WEIGHTED_F32,
    LTX_PIPELINE_RMS_NORM_BF16,
    LTX_PIPELINE_RMS_NORM_WEIGHTED_BF16,
    LTX_PIPELINE_ADALN_BF16,
    LTX_PIPELINE_ADALN_BF16_F16,
    LTX_PIPELINE_OUTPUT_ADALN_BF16,
    LTX_PIPELINE_LINEAR_F32,
    LTX_PIPELINE_LINEAR_BF16,
    LTX_PIPELINE_CONVROT_BF16,
    LTX_PIPELINE_PACK_HEADS_BF16,
    LTX_PIPELINE_PACK_ROPE_SPLIT_BF16,
    LTX_PIPELINE_UNPACK_HEADS_GATE_BF16,
    LTX_PIPELINE_LINEAR_INT8_WEIGHT_BF16,
    LTX_PIPELINE_SOL_REDUCE_SUMMARIES_BF16,
    LTX_PIPELINE_SOL_THRESHOLDS_DIAG_BF16,
    LTX_PIPELINE_SOL_KEY_STATS_BF16,
    LTX_PIPELINE_SOL_THRESHOLDS_STATS_BF16,
    LTX_PIPELINE_SOL_ROUTE_MASK_BF16,
    LTX_PIPELINE_SOL_ATTENTION_BF16,
    LTX_PIPELINE_SOL_ATTENTION_TILED_BF16,
    LTX_PIPELINE_VELOCITY_TO_DENOISED_BF16_SPLIT,
    LTX_PIPELINE_CONDITION_PREFIX_BF16,
    LTX_PIPELINE_RESIDUAL_GATE_BF16_SPLIT,
    LTX_PIPELINE_AFFINE_BF16_SPLIT,
    LTX_PIPELINE_ADALN_BF16_SPLIT,
    LTX_PIPELINE_OUTPUT_ADALN_BF16_SPLIT,
    LTX_PIPELINE_COUNT
} ltx_pipeline_kind;

typedef struct {
    uint32_t rows;
    uint32_t columns;
    uint32_t parameter_rows;
} ltx_broadcast_args;

typedef struct {
    uint32_t rows;
    uint32_t columns;
    uint32_t conditioned_prefix_rows;
} ltx_split_args;

typedef struct {
    uint32_t rows;
    uint32_t input_columns;
    uint32_t start_column;
    uint32_t output_columns;
} ltx_slice_columns_args;

typedef struct {
    uint32_t input_rows;
    uint32_t columns;
    uint32_t start_row;
    uint32_t output_rows;
} ltx_row_partition_args;

typedef struct {
    uint32_t prefix_rows;
    uint32_t suffix_rows;
    uint32_t columns;
} ltx_concat_rows_args;

typedef struct {
    uint32_t elements;
    float sigma;
    float sigma_next;
    float eta;
    float s_noise;
} ltx_diffusion_args;

typedef struct {
    uint32_t rows;
    uint32_t channels;
    uint32_t normalize;
} ltx_latent_stats_args;

@interface LTXInt8LinearGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *weight;
@property(nonatomic, strong) MPSGraphTensor *scale;
@property(nonatomic, strong) MPSGraphTensor *bias;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_shape;
@end

@implementation LTXInt8LinearGraph
@end

@interface LTXAdaLNSingleGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *linear1_weight;
@property(nonatomic, strong) MPSGraphTensor *linear1_bias;
@property(nonatomic, strong) MPSGraphTensor *linear2_weight;
@property(nonatomic, strong) MPSGraphTensor *linear2_bias;
@property(nonatomic, strong) MPSGraphTensor *parameter_weight;
@property(nonatomic, strong) MPSGraphTensor *parameter_bias;
@property(nonatomic, strong) MPSGraphTensor *parameters;
@property(nonatomic, strong) MPSGraphTensor *embedded;
@property(nonatomic, strong) NSArray<NSNumber *> *input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *linear1_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *hidden_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *hidden_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *parameter_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *parameter_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *embedded_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *parameter_shape;
@end

@implementation LTXAdaLNSingleGraph
@end

@interface LTXInt8MLPGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *fc1_weight;
@property(nonatomic, strong) MPSGraphTensor *fc1_scale;
@property(nonatomic, strong) MPSGraphTensor *fc1_bias;
@property(nonatomic, strong) MPSGraphTensor *fc2_weight;
@property(nonatomic, strong) MPSGraphTensor *fc2_scale;
@property(nonatomic, strong) MPSGraphTensor *fc2_bias;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc1_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc1_scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc1_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc2_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc2_scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc2_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_shape;
@end

@implementation LTXInt8MLPGraph
@end

@interface LTXInt8QKVGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *query_weight;
@property(nonatomic, strong) MPSGraphTensor *query_scale;
@property(nonatomic, strong) MPSGraphTensor *query_bias;
@property(nonatomic, strong) MPSGraphTensor *key_weight;
@property(nonatomic, strong) MPSGraphTensor *key_scale;
@property(nonatomic, strong) MPSGraphTensor *key_bias;
@property(nonatomic, strong) MPSGraphTensor *value_weight;
@property(nonatomic, strong) MPSGraphTensor *value_scale;
@property(nonatomic, strong) MPSGraphTensor *value_bias;
@property(nonatomic, strong) MPSGraphTensor *packed_weight;
@property(nonatomic, strong) MPSGraphTensor *packed_scale;
@property(nonatomic, strong) MPSGraphTensor *packed_bias;
@property(nonatomic, strong) MPSGraphTensor *query_norm_weight;
@property(nonatomic, strong) MPSGraphTensor *key_norm_weight;
@property(nonatomic, strong) MPSGraphTensor *query_output;
@property(nonatomic, strong) MPSGraphTensor *key_output;
@property(nonatomic, strong) MPSGraphTensor *value_output;
@property(nonatomic, strong) NSArray<NSNumber *> *input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *packed_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *packed_scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *packed_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *norm_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_shape;
@end

@implementation LTXInt8QKVGraph
@end

@interface LTXSDPAGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query;
@property(nonatomic, strong) MPSGraphTensor *key;
@property(nonatomic, strong) MPSGraphTensor *value;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *query_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *key_value_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_shape;
@end

@implementation LTXSDPAGraph
@end

@interface LTXSelfAttentionCoreGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query;
@property(nonatomic, strong) MPSGraphTensor *key;
@property(nonatomic, strong) MPSGraphTensor *value;
@property(nonatomic, strong) MPSGraphTensor *cosine;
@property(nonatomic, strong) MPSGraphTensor *sine;
@property(nonatomic, strong) MPSGraphTensor *gate;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *frequency_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *gate_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_shape;
@end

@implementation LTXSelfAttentionCoreGraph
@end

@interface LTXInt8SelfAttentionGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *query_weight;
@property(nonatomic, strong) MPSGraphTensor *query_scale;
@property(nonatomic, strong) MPSGraphTensor *query_bias;
@property(nonatomic, strong) MPSGraphTensor *key_weight;
@property(nonatomic, strong) MPSGraphTensor *key_scale;
@property(nonatomic, strong) MPSGraphTensor *key_bias;
@property(nonatomic, strong) MPSGraphTensor *value_weight;
@property(nonatomic, strong) MPSGraphTensor *value_scale;
@property(nonatomic, strong) MPSGraphTensor *value_bias;
@property(nonatomic, strong) MPSGraphTensor *query_norm_weight;
@property(nonatomic, strong) MPSGraphTensor *key_norm_weight;
@property(nonatomic, strong) MPSGraphTensor *gate_weight;
@property(nonatomic, strong) MPSGraphTensor *gate_bias;
@property(nonatomic, strong) MPSGraphTensor *output_weight;
@property(nonatomic, strong) MPSGraphTensor *output_scale;
@property(nonatomic, strong) MPSGraphTensor *output_bias;
@property(nonatomic, strong) MPSGraphTensor *cosine;
@property(nonatomic, strong) MPSGraphTensor *sine;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *projection_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *projection_scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *projection_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *norm_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *gate_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *gate_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *frequency_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_shape;
@end

@implementation LTXInt8SelfAttentionGraph
@end

@interface LTXInt8CrossAttentionGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query_input;
@property(nonatomic, strong) MPSGraphTensor *key_value_input;
@property(nonatomic, strong) MPSGraphTensor *query_weight;
@property(nonatomic, strong) MPSGraphTensor *query_scale;
@property(nonatomic, strong) MPSGraphTensor *query_bias;
@property(nonatomic, strong) MPSGraphTensor *key_weight;
@property(nonatomic, strong) MPSGraphTensor *key_scale;
@property(nonatomic, strong) MPSGraphTensor *key_bias;
@property(nonatomic, strong) MPSGraphTensor *value_weight;
@property(nonatomic, strong) MPSGraphTensor *value_scale;
@property(nonatomic, strong) MPSGraphTensor *value_bias;
@property(nonatomic, strong) MPSGraphTensor *query_norm_weight;
@property(nonatomic, strong) MPSGraphTensor *key_norm_weight;
@property(nonatomic, strong) MPSGraphTensor *gate_weight;
@property(nonatomic, strong) MPSGraphTensor *gate_bias;
@property(nonatomic, strong) MPSGraphTensor *output_weight;
@property(nonatomic, strong) MPSGraphTensor *output_scale;
@property(nonatomic, strong) MPSGraphTensor *output_bias;
@property(nonatomic, strong) MPSGraphTensor *query_cosine;
@property(nonatomic, strong) MPSGraphTensor *query_sine;
@property(nonatomic, strong) MPSGraphTensor *key_cosine;
@property(nonatomic, strong) MPSGraphTensor *key_sine;
@property(nonatomic, strong) MPSGraphTensor *attention_mask;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *query_input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *key_value_input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *query_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *key_value_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *projection_scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *projection_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *norm_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *gate_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *gate_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *query_frequency_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *key_frequency_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *attention_mask_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_shape;
@end

@implementation LTXInt8CrossAttentionGraph
@end

@interface LTXInt8CrossKVGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *key_weight;
@property(nonatomic, strong) MPSGraphTensor *key_scale;
@property(nonatomic, strong) MPSGraphTensor *key_bias;
@property(nonatomic, strong) MPSGraphTensor *value_weight;
@property(nonatomic, strong) MPSGraphTensor *value_scale;
@property(nonatomic, strong) MPSGraphTensor *value_bias;
@property(nonatomic, strong) MPSGraphTensor *key_norm_weight;
@property(nonatomic, strong) MPSGraphTensor *key_output;
@property(nonatomic, strong) MPSGraphTensor *value_output;
@property(nonatomic, strong) NSArray<NSNumber *> *input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *norm_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_shape;
@end

@implementation LTXInt8CrossKVGraph
@end

@interface LTXInt8CrossQueryGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query_input;
@property(nonatomic, strong) MPSGraphTensor *key;
@property(nonatomic, strong) MPSGraphTensor *value;
@property(nonatomic, strong) MPSGraphTensor *query_weight;
@property(nonatomic, strong) MPSGraphTensor *query_scale;
@property(nonatomic, strong) MPSGraphTensor *query_bias;
@property(nonatomic, strong) MPSGraphTensor *query_norm_weight;
@property(nonatomic, strong) MPSGraphTensor *gate_weight;
@property(nonatomic, strong) MPSGraphTensor *gate_bias;
@property(nonatomic, strong) MPSGraphTensor *output_weight;
@property(nonatomic, strong) MPSGraphTensor *output_scale;
@property(nonatomic, strong) MPSGraphTensor *output_bias;
@property(nonatomic, strong) MPSGraphTensor *attention_mask;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *query_input_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *key_value_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *query_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *projection_scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *projection_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *norm_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *gate_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *gate_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_weight_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_scale_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_bias_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *attention_mask_shape;
@property(nonatomic, strong) NSArray<NSNumber *> *output_shape;
@end

@implementation LTXInt8CrossQueryGraph
@end

static const char *const ltx_pipeline_names[LTX_PIPELINE_COUNT] = {
    "ltx_add_f32",
    "ltx_add_bf16",
    "ltx_scale_f32",
    "ltx_velocity_to_denoised_bf16",
    "ltx_euler_step_bf16",
    "ltx_euler_ancestral_step_bf16",
    "ltx_renoise_bf16",
    "ltx_latent_stats_bf16",
    "ltx_residual_gate_f32",
    "ltx_residual_gate_bf16",
    "ltx_affine_bf16",
    "ltx_slice_columns_bf16",
    "ltx_cast_f32_f16",
    "ltx_cast_f16_f32",
    "ltx_cast_f32_bf16",
    "ltx_cast_bf16_f32",
    "ltx_cast_bf16_f16",
    "ltx_cast_f16_bf16",
    "ltx_slice_rows_bf16_f16",
    "ltx_concat_rows_bf16_f16",
    "ltx_join_bf16_f16",
    "ltx_join_residual_gate_bf16_f16",
    "ltx_gelu_tanh_f32",
    "ltx_gelu_tanh_bf16",
    "ltx_silu_bf16",
    "ltx_sigmoid2_bf16",
    "ltx_rms_norm_f32",
    "ltx_rms_norm_weighted_f32",
    "ltx_rms_norm_bf16",
    "ltx_rms_norm_weighted_bf16",
    "ltx_adaln_bf16",
    "ltx_adaln_bf16_f16",
    "ltx_output_adaln_bf16",
    "ltx_linear_f32",
    "ltx_linear_bf16",
    "ltx_convrot_bf16",
    "ltx_pack_heads_bf16",
    "ltx_pack_rope_split_bf16",
    "ltx_unpack_heads_gate_bf16",
    "ltx_linear_int8_weight_bf16",
    "ltx_sol_reduce_summaries_bf16",
    "ltx_sol_thresholds_diag_bf16",
    "ltx_sol_key_stats_bf16",
    "ltx_sol_thresholds_stats_bf16",
    "ltx_sol_route_mask_bf16",
    "ltx_sol_attention_bf16",
    "ltx_sol_attention_tiled_bf16",
    "ltx_velocity_to_denoised_bf16_split",
    "ltx_condition_prefix_bf16",
    "ltx_residual_gate_bf16_split",
    "ltx_affine_bf16_split",
    "ltx_adaln_bf16_split",
    "ltx_output_adaln_bf16_split",
};

struct ltx_gpu {
    void *device;
    void *queue;
    void *deferred_commands;
    void *deferred_operations;
    int batch_active;
    void *int8_linear_graphs;
    void *adaln_single_graphs;
    void *int8_mlp_graphs;
    void *int8_qkv_graphs;
    void *sdpa_graphs;
    void *self_attention_core_graphs;
    void *int8_self_attention_graphs;
    void *int8_cross_attention_graphs;
    void *int8_cross_kv_graphs;
    void *int8_cross_query_graphs;
    void *pipelines[LTX_PIPELINE_COUNT];
};

struct ltx_gpu_buffer {
    void *buffer;
    size_t bytes;
};

static int ltx_gpu_fail(char *error, size_t error_size,
                        const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return 0;
}

static const char *ltx_error_description(NSError *error) {
    if (!error) return "unknown error";
    const char *description = error.localizedDescription.UTF8String;
    return description ? description : "unknown error";
}

static id<MTLDevice> ltx_device(const ltx_gpu *gpu) {
    return gpu ? (__bridge id<MTLDevice>)gpu->device : nil;
}

static id<MTLCommandQueue> ltx_queue(const ltx_gpu *gpu) {
    return gpu ? (__bridge id<MTLCommandQueue>)gpu->queue : nil;
}

static NSMutableArray<id<MTLCommandBuffer>> *ltx_deferred_commands(
        const ltx_gpu *gpu) {
    return gpu ?
        (__bridge NSMutableArray<id<MTLCommandBuffer>> *)
            gpu->deferred_commands : nil;
}

static NSMutableArray<NSString *> *ltx_deferred_operations(
        const ltx_gpu *gpu) {
    return gpu ?
        (__bridge NSMutableArray<NSString *> *)gpu->deferred_operations : nil;
}

static id<MTLComputePipelineState> ltx_pipeline(
        const ltx_gpu *gpu, ltx_pipeline_kind kind) {
    if (!gpu || (unsigned)kind >= LTX_PIPELINE_COUNT) return nil;
    return (__bridge id<MTLComputePipelineState>)gpu->pipelines[kind];
}

static id<MTLBuffer> ltx_buffer(const ltx_gpu_buffer *buffer) {
    return buffer ? (__bridge id<MTLBuffer>)buffer->buffer : nil;
}

void *ltx_gpu_native_device(const ltx_gpu *gpu) {
    return gpu ? gpu->device : NULL;
}

void *ltx_gpu_native_queue(const ltx_gpu *gpu) {
    return gpu ? gpu->queue : NULL;
}

void *ltx_gpu_buffer_native(const ltx_gpu_buffer *buffer) {
    return buffer ? buffer->buffer : NULL;
}

static NSMutableDictionary<NSString *, LTXInt8LinearGraph *> *
ltx_int8_linear_graphs(const ltx_gpu *gpu) {
    return gpu ? (__bridge NSMutableDictionary *)gpu->int8_linear_graphs : nil;
}

static NSMutableDictionary<NSString *, LTXAdaLNSingleGraph *> *
ltx_adaln_single_graphs(const ltx_gpu *gpu) {
    return gpu ? (__bridge NSMutableDictionary *)gpu->adaln_single_graphs : nil;
}

static NSMutableDictionary<NSString *, LTXInt8MLPGraph *> *
ltx_int8_mlp_graphs(const ltx_gpu *gpu) {
    return gpu ? (__bridge NSMutableDictionary *)gpu->int8_mlp_graphs : nil;
}

static NSMutableDictionary<NSString *, LTXInt8QKVGraph *> *
ltx_int8_qkv_graphs(const ltx_gpu *gpu) {
    return gpu ? (__bridge NSMutableDictionary *)gpu->int8_qkv_graphs : nil;
}

static NSMutableDictionary<NSString *, LTXSDPAGraph *> *
ltx_sdpa_graphs(const ltx_gpu *gpu) {
    return gpu ? (__bridge NSMutableDictionary *)gpu->sdpa_graphs : nil;
}

static NSMutableDictionary<NSString *, LTXSelfAttentionCoreGraph *> *
ltx_self_attention_core_graphs(const ltx_gpu *gpu) {
    return gpu ?
        (__bridge NSMutableDictionary *)gpu->self_attention_core_graphs : nil;
}

static NSMutableDictionary<NSString *, LTXInt8SelfAttentionGraph *> *
ltx_int8_self_attention_graphs(const ltx_gpu *gpu) {
    return gpu ?
        (__bridge NSMutableDictionary *)gpu->int8_self_attention_graphs : nil;
}

static NSMutableDictionary<NSString *, LTXInt8CrossAttentionGraph *> *
ltx_int8_cross_attention_graphs(const ltx_gpu *gpu) {
    return gpu ?
        (__bridge NSMutableDictionary *)gpu->int8_cross_attention_graphs : nil;
}

static NSMutableDictionary<NSString *, LTXInt8CrossKVGraph *> *
ltx_int8_cross_kv_graphs(const ltx_gpu *gpu) {
    return gpu ?
        (__bridge NSMutableDictionary *)gpu->int8_cross_kv_graphs : nil;
}

static NSMutableDictionary<NSString *, LTXInt8CrossQueryGraph *> *
ltx_int8_cross_query_graphs(const ltx_gpu *gpu) {
    return gpu ?
        (__bridge NSMutableDictionary *)gpu->int8_cross_query_graphs : nil;
}

static int ltx_required_bytes(uint64_t elements, size_t element_size,
                              uint64_t *bytes) {
    if (!bytes || (element_size && elements > UINT64_MAX / element_size))
        return 0;
    *bytes = elements * element_size;
    return 1;
}

static uint16_t ltx_host_f32_to_bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static NSData *ltx_hadamard_256_bf16(void) {
    NSMutableData *data = [NSMutableData dataWithLength:
        256u * 256u * sizeof(uint16_t)];
    uint16_t *values = data.mutableBytes;
    static const int h4[4][4] = {
        { 1,  1,  1, -1},
        { 1,  1, -1,  1},
        { 1, -1,  1,  1},
        {-1,  1,  1,  1},
    };
    for (uint32_t row = 0; row < 256u; row++)
        for (uint32_t column = 0; column < 256u; column++) {
            uint32_t r = row;
            uint32_t c = column;
            int sign = 1;
            for (unsigned digit = 0; digit < 4u; digit++) {
                sign *= h4[r & 3u][c & 3u];
                r >>= 2u;
                c >>= 2u;
            }
            values[row * 256u + column] =
                ltx_host_f32_to_bf16((float)sign * 0.0625f);
        }
    return data;
}

static int ltx_buffer_fits(const ltx_gpu_buffer *buffer, uint64_t bytes) {
    return buffer && bytes <= SIZE_MAX && (size_t)bytes <= buffer->bytes;
}

static void ltx_dispatch_elements(id<MTLComputeCommandEncoder> encoder,
                                  id<MTLComputePipelineState> pipeline,
                                  uint32_t elements) {
    NSUInteger width = MIN((NSUInteger)256,
                           pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(elements, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

static int ltx_finish_command(ltx_gpu *gpu,
                              id<MTLCommandBuffer> command,
                              const char *operation,
                              char *error, size_t error_size) {
    [command commit];
    if (gpu && gpu->batch_active) {
        [ltx_deferred_commands(gpu) addObject:command];
        [ltx_deferred_operations(gpu) addObject:
            [NSString stringWithUTF8String:
                operation ? operation : "deferred Metal operation"]];
        return 1;
    }
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        char message[1024];
        snprintf(message, sizeof(message), "Metal %s failed: %s", operation,
                 ltx_error_description(command.error));
        return ltx_gpu_fail(error, error_size, message);
    }
    return 1;
}

static int ltx_finish_mps_command(ltx_gpu *gpu,
                                  MPSCommandBuffer *command,
                                  const char *operation,
                                  char *error, size_t error_size) {
    id<MTLCommandBuffer> root = command.rootCommandBuffer;
    if (!root)
        return ltx_gpu_fail(error, error_size,
                            "MPS command has no root command buffer");
    if (root.status == MTLCommandBufferStatusNotEnqueued) [root commit];
    if (gpu && gpu->batch_active) {
        [ltx_deferred_commands(gpu) addObject:root];
        [ltx_deferred_operations(gpu) addObject:
            [NSString stringWithUTF8String:
                operation ? operation : "deferred MPSGraph operation"]];
        return 1;
    }
    [root waitUntilCompleted];
    if (root.status == MTLCommandBufferStatusError) {
        char message[1024];
        snprintf(message, sizeof(message), "Metal %s failed: %s", operation,
                 ltx_error_description(root.error));
        return ltx_gpu_fail(error, error_size, message);
    }
    return 1;
}

ltx_gpu *ltx_gpu_create(const char *shader_source_path,
                        char *error, size_t error_size) {
    if (!shader_source_path) {
        ltx_gpu_fail(error, error_size, "missing Metal shader source path");
        return NULL;
    }

    ltx_gpu *gpu = NULL;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            ltx_gpu_fail(error, error_size, "no Metal device is available");
            return NULL;
        }
        NSString *path = [NSString stringWithUTF8String:shader_source_path];
        if (![path isAbsolutePath]) {Dl_info location;if(dladdr((void*)&ltx_gpu_create,&location)&&location.dli_fname)path=[[@(location.dli_fname) stringByDeletingLastPathComponent] stringByAppendingPathComponent:path];}
        if (!path) {
            ltx_gpu_fail(error, error_size,
                         "Metal shader path is not valid UTF-8");
            return NULL;
        }
        NSError *source_error = nil;
        NSString *source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&source_error];
        if (!source) {
            char message[512];
            snprintf(message, sizeof(message), "read Metal source: %s",
                     ltx_error_description(source_error));
            ltx_gpu_fail(error, error_size, message);
            return NULL;
        }

        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeSafe;
        NSError *library_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                     options:options
                                                       error:&library_error];
        if (!library) {
            char message[1024];
            snprintf(message, sizeof(message), "compile Metal source: %s",
                     ltx_error_description(library_error));
            ltx_gpu_fail(error, error_size, message);
            return NULL;
        }

        id<MTLComputePipelineState> pipelines[LTX_PIPELINE_COUNT] = {nil};
        for (unsigned index = 0; index < LTX_PIPELINE_COUNT; index++) {
            NSString *name = [NSString stringWithUTF8String:
                ltx_pipeline_names[index]];
            id<MTLFunction> function = [library newFunctionWithName:name];
            if (!function) {
                char message[512];
                snprintf(message, sizeof(message),
                         "Metal source does not define %s",
                         ltx_pipeline_names[index]);
                ltx_gpu_fail(error, error_size, message);
                return NULL;
            }
            NSError *pipeline_error = nil;
            pipelines[index] =
                [device newComputePipelineStateWithFunction:function
                                                       error:&pipeline_error];
            if (!pipelines[index]) {
                char message[1024];
                snprintf(message, sizeof(message),
                         "create Metal pipeline %s: %s",
                         ltx_pipeline_names[index],
                         ltx_error_description(pipeline_error));
                ltx_gpu_fail(error, error_size, message);
                return NULL;
            }
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            ltx_gpu_fail(error, error_size,
                         "create Metal command queue failed");
            return NULL;
        }
        gpu = calloc(1, sizeof(*gpu));
        if (!gpu) {
            ltx_gpu_fail(error, error_size,
                         "out of memory creating GPU context");
            return NULL;
        }
        gpu->device = (__bridge_retained void *)device;
        gpu->queue = (__bridge_retained void *)queue;
        gpu->deferred_commands = (__bridge_retained void *)
            [[NSMutableArray alloc] init];
        gpu->deferred_operations = (__bridge_retained void *)
            [[NSMutableArray alloc] init];
        gpu->int8_linear_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        gpu->adaln_single_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        gpu->int8_mlp_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        gpu->int8_qkv_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        gpu->sdpa_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        gpu->self_attention_core_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        gpu->int8_self_attention_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        gpu->int8_cross_attention_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        gpu->int8_cross_kv_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        gpu->int8_cross_query_graphs = (__bridge_retained void *)
            [[NSMutableDictionary alloc] init];
        for (unsigned index = 0; index < LTX_PIPELINE_COUNT; index++)
            gpu->pipelines[index] =
                (__bridge_retained void *)pipelines[index];
    }
    return gpu;
}

void ltx_gpu_free(ltx_gpu *gpu) {
    if (!gpu) return;
    @autoreleasepool {
        NSArray<id<MTLCommandBuffer>> *commands =
            [ltx_deferred_commands(gpu) copy];
        if (commands.count) [commands.lastObject waitUntilCompleted];
        [ltx_deferred_commands(gpu) removeAllObjects];
        [ltx_deferred_operations(gpu) removeAllObjects];
    }
    for (unsigned index = 0; index < LTX_PIPELINE_COUNT; index++)
        if (gpu->pipelines[index])
            (void)CFBridgingRelease(gpu->pipelines[index]);
    if (gpu->int8_linear_graphs)
        (void)CFBridgingRelease(gpu->int8_linear_graphs);
    if (gpu->adaln_single_graphs)
        (void)CFBridgingRelease(gpu->adaln_single_graphs);
    if (gpu->int8_mlp_graphs)
        (void)CFBridgingRelease(gpu->int8_mlp_graphs);
    if (gpu->int8_qkv_graphs)
        (void)CFBridgingRelease(gpu->int8_qkv_graphs);
    if (gpu->sdpa_graphs)
        (void)CFBridgingRelease(gpu->sdpa_graphs);
    if (gpu->self_attention_core_graphs)
        (void)CFBridgingRelease(gpu->self_attention_core_graphs);
    if (gpu->int8_self_attention_graphs)
        (void)CFBridgingRelease(gpu->int8_self_attention_graphs);
    if (gpu->int8_cross_attention_graphs)
        (void)CFBridgingRelease(gpu->int8_cross_attention_graphs);
    if (gpu->int8_cross_kv_graphs)
        (void)CFBridgingRelease(gpu->int8_cross_kv_graphs);
    if (gpu->int8_cross_query_graphs)
        (void)CFBridgingRelease(gpu->int8_cross_query_graphs);
    if (gpu->deferred_operations)
        (void)CFBridgingRelease(gpu->deferred_operations);
    if (gpu->deferred_commands)
        (void)CFBridgingRelease(gpu->deferred_commands);
    if (gpu->queue) (void)CFBridgingRelease(gpu->queue);
    if (gpu->device) (void)CFBridgingRelease(gpu->device);
    free(gpu);
}

int ltx_gpu_get_info(const ltx_gpu *gpu, ltx_gpu_info *info) {
    if (!gpu || !info) return 0;
    memset(info, 0, sizeof(*info));
    @autoreleasepool {
        id<MTLDevice> device = ltx_device(gpu);
        snprintf(info->name, sizeof(info->name), "%s", device.name.UTF8String);
        info->recommended_working_set_bytes =
            (uint64_t)device.recommendedMaxWorkingSetSize;
        info->max_buffer_bytes = (uint64_t)device.maxBufferLength;
        info->unified_memory = device.hasUnifiedMemory;
        info->supports_apple7 = [device supportsFamily:MTLGPUFamilyApple7];
        info->supports_apple8 = [device supportsFamily:MTLGPUFamilyApple8];
        info->supports_apple9 = [device supportsFamily:MTLGPUFamilyApple9];
    }
    return 1;
}

int ltx_gpu_batch_begin(ltx_gpu *gpu, char *error, size_t error_size) {
    if (!gpu || gpu->batch_active)
        return ltx_gpu_fail(error, error_size,
                            "invalid or nested GPU command batch");
    @autoreleasepool {
        NSMutableArray *commands = ltx_deferred_commands(gpu);
        NSMutableArray *operations = ltx_deferred_operations(gpu);
        if (!commands || !operations || commands.count || operations.count)
            return ltx_gpu_fail(error, error_size,
                                "GPU command batch has pending work");
        gpu->batch_active = 1;
    }
    return 1;
}

int ltx_gpu_batch_end(ltx_gpu *gpu, char *error, size_t error_size) {
    if (!gpu || !gpu->batch_active)
        return ltx_gpu_fail(error, error_size,
                            "GPU command batch is not active");
    int ok = 1;
    @autoreleasepool {
        NSMutableArray<id<MTLCommandBuffer>> *commands =
            ltx_deferred_commands(gpu);
        NSMutableArray<NSString *> *operations =
            ltx_deferred_operations(gpu);
        gpu->batch_active = 0;
        if (commands.count) [commands.lastObject waitUntilCompleted];
        for (NSUInteger index = 0; index < commands.count; index++) {
            id<MTLCommandBuffer> command = commands[index];
            if (command.status != MTLCommandBufferStatusError) continue;
            NSString *operation = index < operations.count ?
                operations[index] : @"deferred GPU operation";
            char message[1024];
            snprintf(message, sizeof(message), "Metal %s failed: %s",
                     operation.UTF8String,
                     ltx_error_description(command.error));
            ok = ltx_gpu_fail(error, error_size, message);
            break;
        }
        [commands removeAllObjects];
        [operations removeAllObjects];
    }
    return ok;
}

void ltx_gpu_clear_graph_cache(ltx_gpu *gpu) {
    if (!gpu) return;
    @autoreleasepool {
        [ltx_int8_linear_graphs(gpu) removeAllObjects];
        [ltx_adaln_single_graphs(gpu) removeAllObjects];
        [ltx_int8_mlp_graphs(gpu) removeAllObjects];
        [ltx_int8_qkv_graphs(gpu) removeAllObjects];
        [ltx_sdpa_graphs(gpu) removeAllObjects];
        [ltx_self_attention_core_graphs(gpu) removeAllObjects];
        [ltx_int8_self_attention_graphs(gpu) removeAllObjects];
        [ltx_int8_cross_attention_graphs(gpu) removeAllObjects];
        [ltx_int8_cross_kv_graphs(gpu) removeAllObjects];
        [ltx_int8_cross_query_graphs(gpu) removeAllObjects];
    }
}

ltx_gpu_buffer *ltx_gpu_buffer_new(ltx_gpu *gpu, size_t bytes,
                                   char *error, size_t error_size) {
    if (!gpu || !bytes) {
        ltx_gpu_fail(error, error_size, "invalid Metal buffer size/context");
        return NULL;
    }
    ltx_gpu_buffer *result = NULL;
    @autoreleasepool {
        id<MTLBuffer> buffer = [ltx_device(gpu)
            newBufferWithLength:(NSUInteger)bytes
                        options:MTLResourceStorageModeShared];
        if (!buffer) {
            ltx_gpu_fail(error, error_size, "Metal buffer allocation failed");
            return NULL;
        }
        result = calloc(1, sizeof(*result));
        if (!result) {
            ltx_gpu_fail(error, error_size,
                         "out of memory tracking Metal buffer");
            return NULL;
        }
        result->buffer = (__bridge_retained void *)buffer;
        result->bytes = bytes;
    }
    return result;
}

ltx_gpu_buffer *ltx_gpu_buffer_new_copy(ltx_gpu *gpu, const void *data,
                                        size_t bytes,
                                        char *error, size_t error_size) {
    if (!data || !bytes) {
        ltx_gpu_fail(error, error_size, "invalid Metal buffer copy source");
        return NULL;
    }
    ltx_gpu_buffer *buffer = ltx_gpu_buffer_new(gpu, bytes, error,
                                                error_size);
    if (!buffer) return NULL;
    if (!ltx_gpu_buffer_write(buffer, data, bytes, error, error_size)) {
        ltx_gpu_buffer_free(buffer);
        return NULL;
    }
    return buffer;
}

void ltx_gpu_buffer_free(ltx_gpu_buffer *buffer) {
    if (!buffer) return;
    if (buffer->buffer) (void)CFBridgingRelease(buffer->buffer);
    free(buffer);
}

size_t ltx_gpu_buffer_bytes(const ltx_gpu_buffer *buffer) {
    return buffer ? buffer->bytes : 0;
}

int ltx_gpu_buffer_write(ltx_gpu_buffer *buffer, const void *data,
                         size_t bytes, char *error, size_t error_size) {
    if (!buffer || (!data && bytes) || bytes > buffer->bytes)
        return ltx_gpu_fail(error, error_size, "invalid Metal buffer write");
    memcpy(ltx_buffer(buffer).contents, data, bytes);
    return 1;
}

int ltx_gpu_buffer_read(const ltx_gpu_buffer *buffer, void *data,
                        size_t bytes, char *error, size_t error_size) {
    if (!buffer || (!data && bytes) || bytes > buffer->bytes)
        return ltx_gpu_fail(error, error_size, "invalid Metal buffer read");
    memcpy(data, ltx_buffer(buffer).contents, bytes);
    return 1;
}

int ltx_gpu_add_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                    const ltx_gpu_buffer *left,
                    const ltx_gpu_buffer *right, uint32_t elements,
                    char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(float), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(left, required) || !ltx_buffer_fits(right, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal add buffers/element count");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal add command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal add encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_ADD_F32);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(left) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(right) offset:0 atIndex:2];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "add", error, error_size);
    }
}

int ltx_gpu_add_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                     const ltx_gpu_buffer *left,
                     const ltx_gpu_buffer *right, uint32_t elements,
                     char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(left, required) || !ltx_buffer_fits(right, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16 add buffers/element count");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal BF16 add command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal BF16 add encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_ADD_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(left) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(right) offset:0 atIndex:2];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "BF16 add", error, error_size);
    }
}

int ltx_gpu_scale_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                      const ltx_gpu_buffer *input, float scale,
                      uint32_t elements, char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(float), &required) ||
        !gpu || !elements || !isfinite(scale) ||
        !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(input, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal scale buffers/arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal scale command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal scale encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_SCALE_F32);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:2];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "scale", error, error_size);
    }
}

static int ltx_gpu_valid_diffusion_step(float sigma, float sigma_next) {
    return isfinite(sigma) && isfinite(sigma_next) && sigma > 0.0f &&
        sigma <= 1.0f && sigma_next >= 0.0f && sigma_next <= sigma;
}

int ltx_gpu_velocity_to_denoised_bf16(
                      ltx_gpu *gpu, ltx_gpu_buffer *output,
                      const ltx_gpu_buffer *sample,
                      const ltx_gpu_buffer *velocity,
                      uint32_t elements, float sigma,
                      char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements || !isfinite(sigma) ||
        sigma < 0.0f || sigma > 1.0f ||
        !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(sample, required) ||
        !ltx_buffer_fits(velocity, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal velocity-to-denoised arguments");
    ltx_diffusion_args args = {elements, sigma, 0.0f, 0.0f, 0.0f};
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(
                error, error_size,
                "create Metal velocity-to-denoised command failed");
        id<MTLComputePipelineState> pipeline = ltx_pipeline(
            gpu, LTX_PIPELINE_VELOCITY_TO_DENOISED_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(sample) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(velocity) offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "velocity-to-denoised",
                                  error, error_size);
    }
}

int ltx_gpu_velocity_to_denoised_bf16_split(
                      ltx_gpu *gpu, ltx_gpu_buffer *output,
                      const ltx_gpu_buffer *sample,
                      const ltx_gpu_buffer *velocity,
                      uint32_t rows, uint32_t columns,
                      uint32_t conditioned_prefix_rows,
                      float generated_sigma, float conditioned_sigma,
                      char *error, size_t error_size) {
    uint64_t elements = (uint64_t)rows * columns;
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !rows || !columns || elements > UINT32_MAX ||
        conditioned_prefix_rows > rows ||
        !isfinite(generated_sigma) || !isfinite(conditioned_sigma) ||
        generated_sigma < 0.0f || generated_sigma > 1.0f ||
        conditioned_sigma < 0.0f || conditioned_sigma > generated_sigma ||
        !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(sample, required) ||
        !ltx_buffer_fits(velocity, required))
        return ltx_gpu_fail(
            error, error_size,
            "invalid split velocity-to-denoised arguments");
    ltx_split_args args = {rows, columns, conditioned_prefix_rows};
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(
                error, error_size,
                "create split velocity-to-denoised command failed");
        id<MTLComputePipelineState> pipeline = ltx_pipeline(
            gpu, LTX_PIPELINE_VELOCITY_TO_DENOISED_BF16_SPLIT);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(sample) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(velocity) offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder setBytes:&generated_sigma
                   length:sizeof(generated_sigma) atIndex:4];
        [encoder setBytes:&conditioned_sigma
                   length:sizeof(conditioned_sigma) atIndex:5];
        ltx_dispatch_elements(encoder, pipeline, (uint32_t)elements);
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "split velocity-to-denoised", error, error_size);
    }
}

int ltx_gpu_euler_step_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *sample,
                            const ltx_gpu_buffer *denoised,
                            uint32_t elements, float sigma,
                            float sigma_next,
                            char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements ||
        !ltx_gpu_valid_diffusion_step(sigma, sigma_next) ||
        !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(sample, required) ||
        !ltx_buffer_fits(denoised, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal Euler step arguments");
    ltx_diffusion_args args = {
        elements, sigma, sigma_next, 0.0f, 0.0f,
    };
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal Euler command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_EULER_STEP_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(sample) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(denoised) offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "Euler step", error, error_size);
    }
}

int ltx_gpu_euler_ancestral_step_bf16(
                            ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *sample,
                            const ltx_gpu_buffer *denoised,
                            const ltx_gpu_buffer *noise_f32,
                            uint32_t elements, float sigma,
                            float sigma_next, float eta, float s_noise,
                            char *error, size_t error_size) {
    uint64_t bf16_bytes = 0;
    uint64_t noise_bytes = 0;
    int needs_noise = sigma_next != 0.0f && eta > 0.0f;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &bf16_bytes) ||
        !ltx_required_bytes(elements, sizeof(float), &noise_bytes) ||
        !gpu || !elements ||
        !ltx_gpu_valid_diffusion_step(sigma, sigma_next) ||
        !isfinite(eta) || eta < 0.0f || eta > 1.0f ||
        !isfinite(s_noise) ||
        !ltx_buffer_fits(output, bf16_bytes) ||
        !ltx_buffer_fits(sample, bf16_bytes) ||
        !ltx_buffer_fits(denoised, bf16_bytes) ||
        (needs_noise && !ltx_buffer_fits(noise_f32, noise_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal ancestral Euler arguments");
    ltx_diffusion_args args = {
        elements, sigma, sigma_next, eta, s_noise,
    };
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(
                error, error_size,
                "create Metal ancestral Euler command failed");
        id<MTLComputePipelineState> pipeline = ltx_pipeline(
            gpu, LTX_PIPELINE_EULER_ANCESTRAL_STEP_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(sample) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(denoised) offset:0 atIndex:2];
        [encoder setBuffer:noise_f32 ? ltx_buffer(noise_f32) :
                           ltx_buffer(output) offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "ancestral Euler step",
                                  error, error_size);
    }
}

int ltx_gpu_renoise_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *clean,
                         const ltx_gpu_buffer *noise,
                         uint32_t elements, float sigma,
                         char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements || !isfinite(sigma) ||
        sigma < 0.0f || sigma > 1.0f ||
        !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(clean, required) ||
        !ltx_buffer_fits(noise, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal latent renoise arguments");
    ltx_diffusion_args args = {elements, sigma, 0.0f, 0.0f, 0.0f};
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal renoise command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_RENOISE_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(clean) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(noise) offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "latent renoise",
                                  error, error_size);
    }
}

int ltx_gpu_condition_prefix_bf16(
                         ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *clean_prefix,
                         uint32_t prefix_elements, float mask,
                         char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(prefix_elements, sizeof(uint16_t), &required) ||
        !gpu || !prefix_elements || !isfinite(mask) ||
        mask < 0.0f || mask > 1.0f ||
        !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(clean_prefix, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid conditioning-prefix arguments");
    ltx_diffusion_args args = {
        prefix_elements, mask, 0.0f, 0.0f, 0.0f,
    };
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(
                error, error_size,
                "create conditioning-prefix command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_CONDITION_PREFIX_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(clean_prefix) offset:0 atIndex:1];
        [encoder setBytes:&args length:sizeof(args) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, prefix_elements);
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "conditioning-prefix", error, error_size);
    }
}

int ltx_gpu_latent_stats_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                              const ltx_gpu_buffer *input,
                              const ltx_gpu_buffer *mean,
                              const ltx_gpu_buffer *standard_deviation,
                              uint32_t rows, uint32_t channels,
                              int normalize,
                              char *error, size_t error_size) {
    uint64_t elements64 = (uint64_t)rows * channels;
    uint64_t tensor_bytes = 0;
    uint64_t stats_bytes = 0;
    if (!ltx_required_bytes(elements64, sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes(channels, sizeof(uint16_t), &stats_bytes) ||
        !gpu || !rows || !channels || elements64 > UINT32_MAX ||
        (normalize != 0 && normalize != 1) ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(mean, stats_bytes) ||
        !ltx_buffer_fits(standard_deviation, stats_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal latent-stat arguments");
    ltx_latent_stats_args args = {
        rows, channels, (uint32_t)normalize,
    };
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal latent-stat command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_LATENT_STATS_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(mean) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(standard_deviation)
                     offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        ltx_dispatch_elements(encoder, pipeline, (uint32_t)elements64);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "latent normalization",
                                  error, error_size);
    }
}

int ltx_gpu_residual_gate_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                              const ltx_gpu_buffer *residual,
                              const ltx_gpu_buffer *branch,
                              const ltx_gpu_buffer *gate,
                              uint32_t elements,
                              char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(float), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(residual, required) ||
        !ltx_buffer_fits(branch, required) ||
        !ltx_buffer_fits(gate, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal residual gate buffers/arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal residual gate command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal residual gate encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_RESIDUAL_GATE_F32);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(residual) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(branch) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(gate) offset:0 atIndex:3];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:4];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "residual gate", error,
                                  error_size);
    }
}

int ltx_gpu_residual_gate_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                               const ltx_gpu_buffer *residual,
                               const ltx_gpu_buffer *branch,
                               const ltx_gpu_buffer *gate,
                               uint32_t rows, uint32_t columns,
                               uint32_t gate_rows,
                               char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t gate_bytes = 0;
    uint64_t elements = (uint64_t)rows * columns;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes((uint64_t)gate_rows * columns,
                            sizeof(uint16_t), &gate_bytes) ||
        !gpu || !rows || !columns || (gate_rows != 1u && gate_rows != rows) ||
        elements > UINT32_MAX ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(residual, tensor_bytes) ||
        !ltx_buffer_fits(branch, tensor_bytes) ||
        !ltx_buffer_fits(gate, gate_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16 residual-gate arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal BF16 residual-gate command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_RESIDUAL_GATE_BF16);
        ltx_broadcast_args args = {rows, columns, gate_rows};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(residual) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(branch) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(gate) offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        ltx_dispatch_elements(encoder, pipeline, (uint32_t)elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "BF16 residual gate",
                                  error, error_size);
    }
}

int ltx_gpu_residual_gate_bf16_split(
                               ltx_gpu *gpu, ltx_gpu_buffer *output,
                               const ltx_gpu_buffer *residual,
                               const ltx_gpu_buffer *branch,
                               const ltx_gpu_buffer *generated_gate,
                               const ltx_gpu_buffer *conditioned_gate,
                               uint32_t rows, uint32_t columns,
                               uint32_t conditioned_prefix_rows,
                               char *error, size_t error_size) {
    uint64_t elements = (uint64_t)rows * columns;
    uint64_t tensor_bytes = 0;
    uint64_t parameter_bytes = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes(columns, sizeof(uint16_t), &parameter_bytes) ||
        !gpu || !rows || !columns || elements > UINT32_MAX ||
        conditioned_prefix_rows > rows ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(residual, tensor_bytes) ||
        !ltx_buffer_fits(branch, tensor_bytes) ||
        !ltx_buffer_fits(generated_gate, parameter_bytes) ||
        !ltx_buffer_fits(conditioned_gate, parameter_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid split residual-gate arguments");
    ltx_split_args args = {rows, columns, conditioned_prefix_rows};
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(
                error, error_size,
                "create split residual-gate command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_RESIDUAL_GATE_BF16_SPLIT);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(residual) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(branch) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(generated_gate) offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(conditioned_gate) offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        ltx_dispatch_elements(encoder, pipeline, (uint32_t)elements);
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "split residual gate", error, error_size);
    }
}

int ltx_gpu_affine_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *scale,
                        const ltx_gpu_buffer *shift,
                        uint32_t rows, uint32_t columns,
                        uint32_t parameter_rows,
                        char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t parameter_bytes = 0;
    uint64_t elements = (uint64_t)rows * columns;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes((uint64_t)parameter_rows * columns,
                            sizeof(uint16_t), &parameter_bytes) ||
        !gpu || !rows || !columns ||
        (parameter_rows != 1u && parameter_rows != rows) ||
        elements > UINT32_MAX ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(scale, parameter_bytes) ||
        !ltx_buffer_fits(shift, parameter_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16 affine arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal BF16 affine command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_AFFINE_BF16);
        ltx_broadcast_args args = {rows, columns, parameter_rows};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(scale) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(shift) offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        ltx_dispatch_elements(encoder, pipeline, (uint32_t)elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "BF16 affine",
                                  error, error_size);
    }
}

int ltx_gpu_affine_bf16_split(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *generated_scale,
                        const ltx_gpu_buffer *generated_shift,
                        const ltx_gpu_buffer *conditioned_scale,
                        const ltx_gpu_buffer *conditioned_shift,
                        uint32_t rows, uint32_t columns,
                        uint32_t conditioned_prefix_rows,
                        char *error, size_t error_size) {
    uint64_t elements = (uint64_t)rows * columns;
    uint64_t tensor_bytes = 0;
    uint64_t parameter_bytes = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes(columns, sizeof(uint16_t), &parameter_bytes) ||
        !gpu || !rows || !columns || elements > UINT32_MAX ||
        conditioned_prefix_rows > rows ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(generated_scale, parameter_bytes) ||
        !ltx_buffer_fits(generated_shift, parameter_bytes) ||
        !ltx_buffer_fits(conditioned_scale, parameter_bytes) ||
        !ltx_buffer_fits(conditioned_shift, parameter_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid split affine arguments");
    ltx_split_args args = {rows, columns, conditioned_prefix_rows};
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create split affine command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_AFFINE_BF16_SPLIT);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(generated_scale) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(generated_shift) offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(conditioned_scale) offset:0 atIndex:4];
        [encoder setBuffer:ltx_buffer(conditioned_shift) offset:0 atIndex:5];
        [encoder setBytes:&args length:sizeof(args) atIndex:6];
        ltx_dispatch_elements(encoder, pipeline, (uint32_t)elements);
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "split affine", error, error_size);
    }
}

int ltx_gpu_slice_columns_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                               const ltx_gpu_buffer *input,
                               uint32_t rows, uint32_t input_columns,
                               uint32_t start_column,
                               uint32_t output_columns,
                               char *error, size_t error_size) {
    uint64_t input_elements = (uint64_t)rows * input_columns;
    uint64_t output_elements = (uint64_t)rows * output_columns;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes(input_elements, sizeof(uint16_t), &input_bytes) ||
        !ltx_required_bytes(output_elements, sizeof(uint16_t),
                            &output_bytes) ||
        !gpu || !rows || !input_columns || !output_columns ||
        start_column >= input_columns ||
        output_columns > input_columns - start_column ||
        output_elements > UINT32_MAX ||
        !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(output, output_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16 column-slice arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal BF16 column-slice command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_SLICE_COLUMNS_BF16);
        ltx_slice_columns_args args = {
            rows, input_columns, start_column, output_columns
        };
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&args length:sizeof(args) atIndex:2];
        ltx_dispatch_elements(
            encoder, pipeline, (uint32_t)output_elements);
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "BF16 column slice", error, error_size);
    }
}

int ltx_gpu_cast_f32_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *input, uint32_t elements,
                         char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes(elements, sizeof(float), &input_bytes) ||
        !ltx_required_bytes(elements, sizeof(uint16_t), &output_bytes) ||
        !gpu || !elements || !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(output, output_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal F32-to-F16 cast arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_CAST_F32_F16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "F32-to-F16 cast", error,
                                  error_size);
    }
}

int ltx_gpu_cast_f16_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *input, uint32_t elements,
                         char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &input_bytes) ||
        !ltx_required_bytes(elements, sizeof(float), &output_bytes) ||
        !gpu || !elements || !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(output, output_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal F16-to-F32 cast arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_CAST_F16_F32);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "F16-to-F32 cast", error,
                                  error_size);
    }
}

int ltx_gpu_cast_f32_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes(elements, sizeof(float), &input_bytes) ||
        !ltx_required_bytes(elements, sizeof(uint16_t), &output_bytes) ||
        !gpu || !elements || !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(output, output_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal F32-to-BF16 cast arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_CAST_F32_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "F32-to-BF16 cast", error,
                                  error_size);
    }
}

int ltx_gpu_cast_bf16_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &input_bytes) ||
        !ltx_required_bytes(elements, sizeof(float), &output_bytes) ||
        !gpu || !elements || !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(output, output_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16-to-F32 cast arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_CAST_BF16_F32);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "BF16-to-F32 cast", error,
                                  error_size);
    }
}

int ltx_gpu_cast_bf16_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(input, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16-to-F16 cast arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_CAST_BF16_F16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "BF16-to-F16 cast", error,
                                  error_size);
    }
}

int ltx_gpu_cast_f16_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(input, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal F16-to-BF16 cast arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal cast encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_CAST_F16_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "F16-to-BF16 cast", error,
                                  error_size);
    }
}

int ltx_gpu_slice_rows_bf16_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                                const ltx_gpu_buffer *input,
                                uint32_t input_rows, uint32_t columns,
                                uint32_t start_row, uint32_t output_rows,
                                char *error, size_t error_size) {
    uint64_t input_elements = (uint64_t)input_rows * columns;
    uint64_t output_elements = (uint64_t)output_rows * columns;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes(input_elements, sizeof(uint16_t), &input_bytes) ||
        !ltx_required_bytes(output_elements, sizeof(uint16_t),
                            &output_bytes) ||
        !gpu || !input_rows || !columns || !output_rows ||
        start_row >= input_rows || output_rows > input_rows - start_row ||
        output_elements > UINT32_MAX ||
        !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(output, output_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16/F16 row-slice arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal row-slice command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_SLICE_ROWS_BF16_F16);
        ltx_row_partition_args args = {
            input_rows, columns, start_row, output_rows
        };
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&args length:sizeof(args) atIndex:2];
        ltx_dispatch_elements(
            encoder, pipeline, (uint32_t)output_elements);
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "BF16/F16 row slice", error, error_size);
    }
}

int ltx_gpu_concat_rows_bf16_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                                 const ltx_gpu_buffer *bf16_prefix,
                                 const ltx_gpu_buffer *f16_suffix,
                                 uint32_t prefix_rows,
                                 uint32_t suffix_rows,
                                 uint32_t columns,
                                 char *error, size_t error_size) {
    uint64_t prefix_elements = (uint64_t)prefix_rows * columns;
    uint64_t suffix_elements = (uint64_t)suffix_rows * columns;
    uint64_t output_elements = prefix_elements + suffix_elements;
    uint64_t prefix_bytes = 0;
    uint64_t suffix_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes(prefix_elements, sizeof(uint16_t),
                            &prefix_bytes) ||
        !ltx_required_bytes(suffix_elements, sizeof(uint16_t),
                            &suffix_bytes) ||
        !ltx_required_bytes(output_elements, sizeof(uint16_t),
                            &output_bytes) ||
        !gpu || !prefix_rows || !suffix_rows || !columns ||
        output_elements > UINT32_MAX ||
        !ltx_buffer_fits(bf16_prefix, prefix_bytes) ||
        !ltx_buffer_fits(f16_suffix, suffix_bytes) ||
        !ltx_buffer_fits(output, output_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16/F16 row-concat arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal row-concat command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_CONCAT_ROWS_BF16_F16);
        ltx_concat_rows_args args = {prefix_rows, suffix_rows, columns};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(bf16_prefix) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(f16_suffix) offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        ltx_dispatch_elements(
            encoder, pipeline, (uint32_t)output_elements);
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "BF16/F16 row concat", error, error_size);
    }
}

int ltx_gpu_join_bf16_f16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *gpu_partial,
                          const ltx_gpu_buffer *ane_partial,
                          uint32_t elements,
                          char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(gpu_partial, required) ||
        !ltx_buffer_fits(ane_partial, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16/F16 join arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal join command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal join encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_JOIN_BF16_F16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(gpu_partial) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(ane_partial) offset:0 atIndex:2];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "BF16/F16 join", error,
                                  error_size);
    }
}

int ltx_gpu_join_residual_gate_bf16_f16(
                          ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *residual,
                          const ltx_gpu_buffer *gpu_partial,
                          const ltx_gpu_buffer *ane_partial,
                          const ltx_gpu_buffer *gate,
                          uint32_t rows, uint32_t columns,
                          uint32_t gate_rows,
                          char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t gate_bytes = 0;
    uint64_t elements = (uint64_t)rows * columns;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes((uint64_t)gate_rows * columns,
                            sizeof(uint16_t), &gate_bytes) ||
        !gpu || !rows || !columns || (gate_rows != 1u && gate_rows != rows) ||
        elements > UINT32_MAX ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(residual, tensor_bytes) ||
        !ltx_buffer_fits(gpu_partial, tensor_bytes) ||
        !ltx_buffer_fits(ane_partial, tensor_bytes) ||
        !ltx_buffer_fits(gate, gate_bytes))
        return ltx_gpu_fail(
            error, error_size,
            "invalid Metal BF16/F16 join-residual arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(
                error, error_size,
                "create Metal BF16/F16 join-residual command failed");
        id<MTLComputePipelineState> pipeline = ltx_pipeline(
            gpu, LTX_PIPELINE_JOIN_RESIDUAL_GATE_BF16_F16);
        ltx_broadcast_args args = {rows, columns, gate_rows};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(residual) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(gpu_partial) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(ane_partial) offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(gate) offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        ltx_dispatch_elements(encoder, pipeline, (uint32_t)elements);
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "BF16/F16 join residual gate", error, error_size);
    }
}

int ltx_gpu_gelu_tanh_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(float), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(input, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal GELU buffers/arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal GELU command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal GELU encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_GELU_TANH_F32);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "GELU", error, error_size);
    }
}

int ltx_gpu_gelu_tanh_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                           const ltx_gpu_buffer *input, uint32_t elements,
                           char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(input, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16 GELU buffers/arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal BF16 GELU command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal BF16 GELU encoder failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_GELU_TANH_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "BF16 GELU", error,
                                  error_size);
    }
}

int ltx_gpu_silu_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                      const ltx_gpu_buffer *input, uint32_t elements,
                      char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(input, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16 SiLU buffers/arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal BF16 SiLU command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_SILU_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "BF16 SiLU", error, error_size);
    }
}

int ltx_gpu_sigmoid2_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input, uint32_t elements,
                          char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        !gpu || !elements || !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(input, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16 sigmoid-gate arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal sigmoid-gate command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_SIGMOID2_BF16);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, elements);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "BF16 sigmoid gate",
                                  error, error_size);
    }
}

static int ltx_gpu_rms_norm_impl(ltx_gpu *gpu, ltx_gpu_buffer *output,
                                 const ltx_gpu_buffer *input,
                                 const ltx_gpu_buffer *weight,
                                 uint32_t rows, uint32_t columns,
                                 float epsilon, size_t element_size,
                                 ltx_pipeline_kind plain_kind,
                                 ltx_pipeline_kind weighted_kind,
                                 char *error, size_t error_size) {
    uint64_t elements = (uint64_t)rows * columns;
    uint64_t required = 0;
    uint64_t weight_bytes = 0;
    if (!ltx_required_bytes(elements, element_size, &required) ||
        !ltx_required_bytes(columns, element_size, &weight_bytes) ||
        !gpu || !rows || !columns || !isfinite(epsilon) || epsilon < 0.0f ||
        !ltx_buffer_fits(output, required) ||
        !ltx_buffer_fits(input, required) ||
        (weight && !ltx_buffer_fits(weight, weight_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal RMSNorm buffers/arguments");
    @autoreleasepool {
        ltx_pipeline_kind kind = weight ? weighted_kind : plain_kind;
        id<MTLComputePipelineState> pipeline = ltx_pipeline(gpu, kind);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
            return ltx_gpu_fail(error, error_size,
                                "RMSNorm requires a 256-thread threadgroup");
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal RMSNorm command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal RMSNorm encoder failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        NSUInteger constant_index = 2u;
        if (weight) {
            [encoder setBuffer:ltx_buffer(weight) offset:0 atIndex:2];
            constant_index = 3u;
        }
        [encoder setBytes:&columns length:sizeof(columns)
                   atIndex:constant_index];
        [encoder setBytes:&epsilon length:sizeof(epsilon)
                   atIndex:constant_index + 1u];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "RMSNorm", error, error_size);
    }
}

int ltx_gpu_rms_norm_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *input,
                         uint32_t rows, uint32_t columns, float epsilon,
                         char *error, size_t error_size) {
    return ltx_gpu_rms_norm_impl(gpu, output, input, NULL, rows, columns,
                                 epsilon, sizeof(float),
                                 LTX_PIPELINE_RMS_NORM_F32,
                                 LTX_PIPELINE_RMS_NORM_WEIGHTED_F32,
                                 error, error_size);
}

int ltx_gpu_rms_norm_weighted_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                                  const ltx_gpu_buffer *input,
                                  const ltx_gpu_buffer *weight,
                                  uint32_t rows, uint32_t columns,
                                  float epsilon,
                                  char *error, size_t error_size) {
    if (!weight)
        return ltx_gpu_fail(error, error_size,
                            "weighted RMSNorm requires weights");
    return ltx_gpu_rms_norm_impl(gpu, output, input, weight, rows, columns,
                                 epsilon, sizeof(float),
                                 LTX_PIPELINE_RMS_NORM_F32,
                                 LTX_PIPELINE_RMS_NORM_WEIGHTED_F32,
                                 error, error_size);
}

int ltx_gpu_rms_norm_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *input,
                          uint32_t rows, uint32_t columns, float epsilon,
                          char *error, size_t error_size) {
    return ltx_gpu_rms_norm_impl(gpu, output, input, NULL, rows, columns,
                                 epsilon, sizeof(uint16_t),
                                 LTX_PIPELINE_RMS_NORM_BF16,
                                 LTX_PIPELINE_RMS_NORM_WEIGHTED_BF16,
                                 error, error_size);
}

int ltx_gpu_rms_norm_weighted_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                                   const ltx_gpu_buffer *input,
                                   const ltx_gpu_buffer *weight,
                                   uint32_t rows, uint32_t columns,
                                   float epsilon,
                                   char *error, size_t error_size) {
    if (!weight)
        return ltx_gpu_fail(error, error_size,
                            "weighted BF16 RMSNorm requires weights");
    return ltx_gpu_rms_norm_impl(gpu, output, input, weight, rows, columns,
                                 epsilon, sizeof(uint16_t),
                                 LTX_PIPELINE_RMS_NORM_BF16,
                                 LTX_PIPELINE_RMS_NORM_WEIGHTED_BF16,
                                 error, error_size);
}

int ltx_gpu_adaln_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                       const ltx_gpu_buffer *input,
                       const ltx_gpu_buffer *scale,
                       const ltx_gpu_buffer *shift,
                       uint32_t rows, uint32_t columns,
                       uint32_t parameter_rows, float epsilon,
                       char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t parameter_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * columns,
                            sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes((uint64_t)parameter_rows * columns,
                            sizeof(uint16_t), &parameter_bytes) ||
        !gpu || !rows || !columns ||
        (parameter_rows != 1u && parameter_rows != rows) ||
        !isfinite(epsilon) || epsilon < 0.0f ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(scale, parameter_bytes) ||
        !ltx_buffer_fits(shift, parameter_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16 AdaLN arguments");
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_ADALN_BF16);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
            return ltx_gpu_fail(error, error_size,
                                "AdaLN requires a 256-thread threadgroup");
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal BF16 AdaLN command failed");
        ltx_broadcast_args args = {rows, columns, parameter_rows};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(scale) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(shift) offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "BF16 AdaLN", error, error_size);
    }
}

int ltx_gpu_adaln_bf16_split(
                       ltx_gpu *gpu, ltx_gpu_buffer *output,
                       const ltx_gpu_buffer *input,
                       const ltx_gpu_buffer *generated_scale,
                       const ltx_gpu_buffer *generated_shift,
                       const ltx_gpu_buffer *conditioned_scale,
                       const ltx_gpu_buffer *conditioned_shift,
                       uint32_t rows, uint32_t columns,
                       uint32_t conditioned_prefix_rows, float epsilon,
                       char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t parameter_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * columns,
                            sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes(columns, sizeof(uint16_t), &parameter_bytes) ||
        !gpu || !rows || !columns || conditioned_prefix_rows > rows ||
        !isfinite(epsilon) || epsilon < 0.0f ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(generated_scale, parameter_bytes) ||
        !ltx_buffer_fits(generated_shift, parameter_bytes) ||
        !ltx_buffer_fits(conditioned_scale, parameter_bytes) ||
        !ltx_buffer_fits(conditioned_shift, parameter_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid split AdaLN arguments");
    ltx_split_args args = {rows, columns, conditioned_prefix_rows};
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_ADALN_BF16_SPLIT);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
            return ltx_gpu_fail(error, error_size,
                                "split AdaLN requires 256 threads");
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create split AdaLN command failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(generated_scale) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(generated_shift) offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(conditioned_scale) offset:0 atIndex:4];
        [encoder setBuffer:ltx_buffer(conditioned_shift) offset:0 atIndex:5];
        [encoder setBytes:&args length:sizeof(args) atIndex:6];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "split AdaLN", error, error_size);
    }
}

int ltx_gpu_adaln_bf16_f16(ltx_gpu *gpu,
                       ltx_gpu_buffer *output_bf16,
                       ltx_gpu_buffer *output_f16,
                       const ltx_gpu_buffer *input,
                       const ltx_gpu_buffer *scale,
                       const ltx_gpu_buffer *shift,
                       uint32_t rows, uint32_t columns,
                       uint32_t parameter_rows, float epsilon,
                       char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t parameter_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * columns,
                            sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes((uint64_t)parameter_rows * columns,
                            sizeof(uint16_t), &parameter_bytes) ||
        !gpu || !rows || !columns ||
        (parameter_rows != 1u && parameter_rows != rows) ||
        !isfinite(epsilon) || epsilon < 0.0f ||
        !ltx_buffer_fits(output_bf16, tensor_bytes) ||
        !ltx_buffer_fits(output_f16, tensor_bytes) ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(scale, parameter_bytes) ||
        !ltx_buffer_fits(shift, parameter_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid dual-output BF16/F16 AdaLN arguments");
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_ADALN_BF16_F16);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
            return ltx_gpu_fail(
                error, error_size,
                "dual-output AdaLN requires a 256-thread threadgroup");
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(
                error, error_size,
                "create dual-output BF16/F16 AdaLN command failed");
        ltx_broadcast_args args = {rows, columns, parameter_rows};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output_bf16) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(output_f16) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(scale) offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(shift) offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "dual-output BF16/F16 AdaLN", error, error_size);
    }
}

int ltx_gpu_output_adaln_bf16(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *embedded_timestep,
                        const ltx_gpu_buffer *shift_table,
                        const ltx_gpu_buffer *scale_table,
                        uint32_t rows, uint32_t columns, float epsilon,
                        char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t parameter_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * columns,
                            sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes(columns, sizeof(uint16_t), &parameter_bytes) ||
        !gpu || !rows || !columns ||
        !isfinite(epsilon) || epsilon < 0.0f ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(embedded_timestep, parameter_bytes) ||
        !ltx_buffer_fits(shift_table, parameter_bytes) ||
        !ltx_buffer_fits(scale_table, parameter_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal BF16 output-AdaLN arguments");
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_OUTPUT_ADALN_BF16);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
            return ltx_gpu_fail(
                error, error_size,
                "output AdaLN requires a 256-thread threadgroup");
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(
                error, error_size,
                "create Metal BF16 output-AdaLN command failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(embedded_timestep)
                     offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(shift_table) offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(scale_table) offset:0 atIndex:4];
        [encoder setBytes:&columns length:sizeof(columns) atIndex:5];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "BF16 output AdaLN", error, error_size);
    }
}

int ltx_gpu_output_adaln_bf16_split(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *generated_embedded_timestep,
                        const ltx_gpu_buffer *conditioned_embedded_timestep,
                        const ltx_gpu_buffer *shift_table,
                        const ltx_gpu_buffer *scale_table,
                        uint32_t rows, uint32_t columns,
                        uint32_t conditioned_prefix_rows, float epsilon,
                        char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t parameter_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * columns,
                            sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes(columns, sizeof(uint16_t), &parameter_bytes) ||
        !gpu || !rows || !columns || conditioned_prefix_rows > rows ||
        !isfinite(epsilon) || epsilon < 0.0f ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(generated_embedded_timestep, parameter_bytes) ||
        !ltx_buffer_fits(conditioned_embedded_timestep, parameter_bytes) ||
        !ltx_buffer_fits(shift_table, parameter_bytes) ||
        !ltx_buffer_fits(scale_table, parameter_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid split output-AdaLN arguments");
    ltx_split_args args = {rows, columns, conditioned_prefix_rows};
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_OUTPUT_ADALN_BF16_SPLIT);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
            return ltx_gpu_fail(error, error_size,
                                "split output-AdaLN requires 256 threads");
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(
                error, error_size,
                "create split output-AdaLN command failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(generated_embedded_timestep)
                     offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(conditioned_embedded_timestep)
                     offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(shift_table) offset:0 atIndex:4];
        [encoder setBuffer:ltx_buffer(scale_table) offset:0 atIndex:5];
        [encoder setBytes:&args length:sizeof(args) atIndex:6];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        return ltx_finish_command(
            gpu, command, "split output AdaLN", error, error_size);
    }
}

typedef struct {
    uint32_t rows;
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t has_bias;
} ltx_linear_args;

typedef struct {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t has_gate;
} ltx_heads_args;

typedef struct {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t blocks;
    uint32_t sink_start;
    uint32_t sink_end;
    uint32_t sink_query_start;
    uint32_t sink_query_end;
    uint32_t head_major_input;
    uint32_t head_major_output;
    float scale_log2;
    float tau;
} ltx_sol_args;

static int ltx_gpu_linear_impl(ltx_gpu *gpu, ltx_gpu_buffer *output,
                               const ltx_gpu_buffer *input,
                               const ltx_gpu_buffer *weight,
                               const ltx_gpu_buffer *bias,
                               uint32_t rows, uint32_t input_dim,
                               uint32_t output_dim, size_t element_size,
                               ltx_pipeline_kind kind,
                               char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t bias_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * input_dim, element_size,
                            &input_bytes) ||
        !ltx_required_bytes((uint64_t)output_dim * input_dim, element_size,
                            &weight_bytes) ||
        !ltx_required_bytes((uint64_t)rows * output_dim, element_size,
                            &output_bytes) ||
        !ltx_required_bytes(output_dim, element_size, &bias_bytes) ||
        !gpu || !rows || !input_dim || !output_dim ||
        !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(weight, weight_bytes) ||
        !ltx_buffer_fits(output, output_bytes) ||
        (bias && !ltx_buffer_fits(bias, bias_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal linear buffers/arguments");
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = ltx_pipeline(gpu, kind);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
            return ltx_gpu_fail(error, error_size,
                                "linear requires a 16x16 threadgroup");
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal linear command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal linear encoder failed");
        ltx_linear_args args = {
            rows, input_dim, output_dim, bias ? 1u : 0u
        };
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(weight) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(bias ? bias : input)
                     offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:
            MTLSizeMake(((NSUInteger)output_dim + 15u) / 16u,
                        ((NSUInteger)rows + 15u) / 16u, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "linear", error, error_size);
    }
}

int ltx_gpu_linear_f32(ltx_gpu *gpu, ltx_gpu_buffer *output,
                       const ltx_gpu_buffer *input,
                       const ltx_gpu_buffer *weight,
                       const ltx_gpu_buffer *bias,
                       uint32_t rows, uint32_t input_dim,
                       uint32_t output_dim,
                       char *error, size_t error_size) {
    return ltx_gpu_linear_impl(gpu, output, input, weight, bias, rows,
                               input_dim, output_dim, sizeof(float),
                               LTX_PIPELINE_LINEAR_F32,
                               error, error_size);
}

int ltx_gpu_linear_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *weight,
                        const ltx_gpu_buffer *bias,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t output_dim,
                        char *error, size_t error_size) {
    return ltx_gpu_linear_impl(gpu, output, input, weight, bias, rows,
                               input_dim, output_dim, sizeof(uint16_t),
                               LTX_PIPELINE_LINEAR_BF16,
                               error, error_size);
}

static MPSGraphTensor *ltx_graph_bf16_linear(
        MPSGraph *graph, MPSGraphTensor *input,
        MPSGraphTensor *weight, MPSGraphTensor *bias) {
    MPSGraphTensor *transposed = [graph transposeTensor:weight
                                             dimension:1 withDimension:2
                                                  name:nil];
    MPSGraphTensor *output =
        [graph matrixMultiplicationWithPrimaryTensor:input
                                     secondaryTensor:transposed name:nil];
    if (bias)
        output = [graph additionWithPrimaryTensor:output
                                   secondaryTensor:bias name:nil];
    return [graph castTensor:output toType:MPSDataTypeBFloat16 name:nil];
}

static MPSGraphTensor *ltx_graph_silu_bf16(
        MPSGraph *graph, MPSGraphTensor *input) {
    MPSGraphTensor *value = [graph castTensor:input
                                      toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *sigmoid = [graph sigmoidWithTensor:value name:nil];
    MPSGraphTensor *output = [graph multiplicationWithPrimaryTensor:value
                                                    secondaryTensor:sigmoid
                                                               name:nil];
    return [graph castTensor:output toType:MPSDataTypeBFloat16 name:nil];
}

static LTXAdaLNSingleGraph *ltx_adaln_single_graph(
        ltx_gpu *gpu, uint32_t rows, uint32_t timestep_dim,
        uint32_t hidden_dim, uint32_t parameter_count) {
    NSMutableDictionary<NSString *, LTXAdaLNSingleGraph *> *cache =
        ltx_adaln_single_graphs(gpu);
    NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%u",
        rows, timestep_dim, hidden_dim, parameter_count];
    LTXAdaLNSingleGraph *cached = cache[key];
    if (cached) return cached;

    uint64_t parameter_dim = (uint64_t)hidden_dim * parameter_count;
    LTXAdaLNSingleGraph *adaln = [[LTXAdaLNSingleGraph alloc] init];
    adaln.graph = [[MPSGraph alloc] init];
    adaln.input_shape = @[@1, @(rows), @(timestep_dim)];
    adaln.linear1_weight_shape = @[@1, @(hidden_dim), @(timestep_dim)];
    adaln.hidden_weight_shape = @[@1, @(hidden_dim), @(hidden_dim)];
    adaln.hidden_bias_shape = @[@1, @1, @(hidden_dim)];
    adaln.parameter_weight_shape =
        @[@1, @(parameter_dim), @(hidden_dim)];
    adaln.parameter_bias_shape = @[@1, @1, @(parameter_dim)];
    adaln.embedded_shape = @[@1, @(rows), @(hidden_dim)];
    adaln.parameter_shape = @[@1, @(rows), @(parameter_dim)];

#define LTX_ADALN_PLACEHOLDER(PROPERTY, SHAPE) \
    adaln.PROPERTY = [adaln.graph placeholderWithShape:(SHAPE) \
        dataType:MPSDataTypeBFloat16 name:nil]
    LTX_ADALN_PLACEHOLDER(input, adaln.input_shape);
    LTX_ADALN_PLACEHOLDER(linear1_weight, adaln.linear1_weight_shape);
    LTX_ADALN_PLACEHOLDER(linear1_bias, adaln.hidden_bias_shape);
    LTX_ADALN_PLACEHOLDER(linear2_weight, adaln.hidden_weight_shape);
    LTX_ADALN_PLACEHOLDER(linear2_bias, adaln.hidden_bias_shape);
    LTX_ADALN_PLACEHOLDER(parameter_weight, adaln.parameter_weight_shape);
    LTX_ADALN_PLACEHOLDER(parameter_bias, adaln.parameter_bias_shape);
#undef LTX_ADALN_PLACEHOLDER

    MPSGraphTensor *hidden = ltx_graph_bf16_linear(
        adaln.graph, adaln.input,
        adaln.linear1_weight, adaln.linear1_bias);
    hidden = ltx_graph_silu_bf16(adaln.graph, hidden);
    adaln.embedded = ltx_graph_bf16_linear(
        adaln.graph, hidden,
        adaln.linear2_weight, adaln.linear2_bias);
    MPSGraphTensor *activated =
        ltx_graph_silu_bf16(adaln.graph, adaln.embedded);
    adaln.parameters = ltx_graph_bf16_linear(
        adaln.graph, activated,
        adaln.parameter_weight, adaln.parameter_bias);
    cache[key] = adaln;
    return adaln;
}

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
                        char *error, size_t error_size) {
    uint64_t parameter_dim = (uint64_t)hidden_dim * parameter_count;
    uint64_t input_bytes = 0;
    uint64_t linear1_weight_bytes = 0;
    uint64_t hidden_weight_bytes = 0;
    uint64_t hidden_bias_bytes = 0;
    uint64_t parameter_weight_bytes = 0;
    uint64_t parameter_bias_bytes = 0;
    uint64_t embedded_bytes = 0;
    uint64_t parameter_bytes = 0;
    if (parameter_dim > UINT32_MAX ||
        !ltx_required_bytes((uint64_t)rows * timestep_dim,
                            sizeof(uint16_t), &input_bytes) ||
        !ltx_required_bytes((uint64_t)hidden_dim * timestep_dim,
                            sizeof(uint16_t), &linear1_weight_bytes) ||
        !ltx_required_bytes((uint64_t)hidden_dim * hidden_dim,
                            sizeof(uint16_t), &hidden_weight_bytes) ||
        !ltx_required_bytes(hidden_dim, sizeof(uint16_t),
                            &hidden_bias_bytes) ||
        !ltx_required_bytes(parameter_dim * hidden_dim,
                            sizeof(uint16_t), &parameter_weight_bytes) ||
        !ltx_required_bytes(parameter_dim, sizeof(uint16_t),
                            &parameter_bias_bytes) ||
        !ltx_required_bytes((uint64_t)rows * hidden_dim,
                            sizeof(uint16_t), &embedded_bytes) ||
        !ltx_required_bytes((uint64_t)rows * parameter_dim,
                            sizeof(uint16_t), &parameter_bytes) ||
        !gpu || !rows || !timestep_dim || !hidden_dim || !parameter_count ||
        !ltx_buffer_fits(timestep_embedding, input_bytes) ||
        !ltx_buffer_fits(linear1_weight, linear1_weight_bytes) ||
        !ltx_buffer_fits(linear1_bias, hidden_bias_bytes) ||
        !ltx_buffer_fits(linear2_weight, hidden_weight_bytes) ||
        !ltx_buffer_fits(linear2_bias, hidden_bias_bytes) ||
        !ltx_buffer_fits(parameter_weight, parameter_weight_bytes) ||
        !ltx_buffer_fits(parameter_bias, parameter_bias_bytes) ||
        !ltx_buffer_fits(embedded_timestep, embedded_bytes) ||
        !ltx_buffer_fits(parameters, parameter_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS AdaLN-single arguments");

    @autoreleasepool {
        LTXAdaLNSingleGraph *adaln = ltx_adaln_single_graph(
            gpu, rows, timestep_dim, hidden_dim, parameter_count);
        if (!adaln)
            return ltx_gpu_fail(error, error_size,
                                "create MPS AdaLN-single graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS AdaLN-single command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
#define LTX_ADALN_FEED(TENSOR, BUFFER, SHAPE) \
        feeds[(TENSOR)] = [[MPSGraphTensorData alloc] \
            initWithMTLBuffer:ltx_buffer((BUFFER)) shape:(SHAPE) \
                     dataType:MPSDataTypeBFloat16]
        LTX_ADALN_FEED(adaln.input, timestep_embedding, adaln.input_shape);
        LTX_ADALN_FEED(adaln.linear1_weight, linear1_weight,
                       adaln.linear1_weight_shape);
        LTX_ADALN_FEED(adaln.linear1_bias, linear1_bias,
                       adaln.hidden_bias_shape);
        LTX_ADALN_FEED(adaln.linear2_weight, linear2_weight,
                       adaln.hidden_weight_shape);
        LTX_ADALN_FEED(adaln.linear2_bias, linear2_bias,
                       adaln.hidden_bias_shape);
        LTX_ADALN_FEED(adaln.parameter_weight, parameter_weight,
                       adaln.parameter_weight_shape);
        LTX_ADALN_FEED(adaln.parameter_bias, parameter_bias,
                       adaln.parameter_bias_shape);
#undef LTX_ADALN_FEED
        MPSGraphTensorData *parameter_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(parameters)
                        shape:adaln.parameter_shape
                     dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *embedded_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(embedded_timestep)
                        shape:adaln.embedded_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [adaln.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{
                    adaln.parameters: parameter_data,
                    adaln.embedded: embedded_data
                }
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message), "MPSGraph AdaLN-single: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(
            gpu, command, "MPSGraph AdaLN-single", error, error_size);
    }
}

int ltx_gpu_convrot_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                         const ltx_gpu_buffer *input,
                         uint32_t rows, uint32_t columns,
                         uint32_t group_size,
                         char *error, size_t error_size) {
    uint64_t required = 0;
    if (!ltx_required_bytes((uint64_t)rows * columns, sizeof(uint16_t),
                            &required) ||
        !gpu || !rows || !columns || group_size != 256u ||
        columns % group_size || !ltx_buffer_fits(input, required) ||
        !ltx_buffer_fits(output, required))
        return ltx_gpu_fail(error, error_size,
                            "ConvRot requires BF16 rows with K divisible by 256");
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_CONVROT_BF16);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
            return ltx_gpu_fail(error, error_size,
                                "ConvRot requires a 256-thread threadgroup");
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal ConvRot command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal ConvRot encoder failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&columns length:sizeof(columns) atIndex:2];
        [encoder dispatchThreadgroups:
            MTLSizeMake(columns / group_size, rows, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "ConvRot", error, error_size);
    }
}

int ltx_gpu_pack_heads_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *input,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim,
                            char *error, size_t error_size) {
    uint64_t required = 0;
    uint64_t elements = (uint64_t)rows * heads * head_dim;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &required) ||
        elements > UINT32_MAX || !gpu || !rows || !heads || !head_dim ||
        !ltx_buffer_fits(input, required) ||
        !ltx_buffer_fits(output, required))
        return ltx_gpu_fail(error, error_size,
                            "invalid BF16 head-pack arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal head-pack command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_PACK_HEADS_BF16);
        ltx_heads_args args = {rows, heads, head_dim, 0u};
        uint32_t count = (uint32_t)elements;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBytes:&args length:sizeof(args) atIndex:2];
        ltx_dispatch_elements(encoder, pipeline, count);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "head pack", error, error_size);
    }
}

int ltx_gpu_pack_rope_split_bf16(
                            ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *input,
                            const ltx_gpu_buffer *cosine,
                            const ltx_gpu_buffer *sine,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim,
                            char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t frequency_bytes = 0;
    uint64_t elements = (uint64_t)rows * heads * head_dim;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes(elements / 2u, sizeof(uint16_t),
                            &frequency_bytes) ||
        elements > UINT32_MAX || !gpu || !rows || !heads ||
        !head_dim || head_dim % 2u ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(cosine, frequency_bytes) ||
        !ltx_buffer_fits(sine, frequency_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid BF16 split-RoPE pack arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal split-RoPE command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_PACK_ROPE_SPLIT_BF16);
        ltx_heads_args args = {rows, heads, head_dim, 0u};
        uint32_t count = (uint32_t)elements;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(cosine) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(sine) offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        ltx_dispatch_elements(encoder, pipeline, count);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "split-RoPE head pack",
                                  error, error_size);
    }
}

int ltx_gpu_unpack_heads_gate_bf16(
                            ltx_gpu *gpu, ltx_gpu_buffer *output,
                            const ltx_gpu_buffer *input,
                            const ltx_gpu_buffer *gate,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim,
                            char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t gate_bytes = 0;
    uint64_t elements = (uint64_t)rows * heads * head_dim;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes((uint64_t)rows * heads, sizeof(uint16_t),
                            &gate_bytes) ||
        elements > UINT32_MAX || !gpu || !rows || !heads || !head_dim ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        (gate && !ltx_buffer_fits(gate, gate_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid BF16 head-unpack arguments");
    @autoreleasepool {
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal head-unpack command failed");
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_UNPACK_HEADS_GATE_BF16);
        ltx_heads_args args = {rows, heads, head_dim, gate ? 1u : 0u};
        uint32_t count = (uint32_t)elements;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(input) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(gate ? gate : input)
                     offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        ltx_dispatch_elements(encoder, pipeline, count);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "head unpack/gate",
                                  error, error_size);
    }
}

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
                          char *error, size_t error_size) {
    uint32_t blocks = rows ? (rows + 63u) / 64u : 0u;
    uint64_t elements = (uint64_t)rows * heads * head_dim;
    uint64_t summary_elements = (uint64_t)heads * blocks * head_dim;
    uint64_t threshold_elements = (uint64_t)heads * blocks;
    uint64_t route_elements = threshold_elements * blocks;
    uint64_t tensor_bytes = 0;
    uint64_t frequency_bytes = 0;
    uint64_t gate_bytes = 0;
    uint64_t query_summary_bytes = 0;
    uint64_t bf16_summary_bytes = 0;
    uint64_t threshold_bytes = 0;
    uint64_t route_bytes = 0;
    if (!ltx_required_bytes(elements, sizeof(uint16_t), &tensor_bytes) ||
        !ltx_required_bytes(elements / 2u, sizeof(uint16_t),
                            &frequency_bytes) ||
        !ltx_required_bytes((uint64_t)rows * heads, sizeof(uint16_t),
                            &gate_bytes) ||
        !ltx_required_bytes(summary_elements, sizeof(float),
                            &query_summary_bytes) ||
        !ltx_required_bytes(summary_elements, sizeof(uint16_t),
                            &bf16_summary_bytes) ||
        !ltx_required_bytes(threshold_elements, sizeof(float),
                            &threshold_bytes) ||
        !ltx_required_bytes(route_elements, sizeof(float), &route_bytes) ||
        elements > UINT32_MAX || !gpu || !rows || !heads ||
        head_dim != 128u || !blocks || !(scale > 0.0f) ||
        !isfinite(scale) || !isfinite(tau) ||
        sink_start > sink_end || sink_end > blocks ||
        sink_query_start > sink_query_end || sink_query_end > blocks ||
        !ltx_buffer_fits(query, tensor_bytes) ||
        !ltx_buffer_fits(key, tensor_bytes) ||
        !ltx_buffer_fits(value, tensor_bytes) ||
        !ltx_buffer_fits(cosine, frequency_bytes) ||
        !ltx_buffer_fits(sine, frequency_bytes) ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(packed_query, tensor_bytes) ||
        !ltx_buffer_fits(packed_key, tensor_bytes) ||
        !ltx_buffer_fits(packed_value, tensor_bytes) ||
        !ltx_buffer_fits(packed_output, tensor_bytes) ||
        !ltx_buffer_fits(query_centroids, query_summary_bytes) ||
        !ltx_buffer_fits(key_centroids, bf16_summary_bytes) ||
        !ltx_buffer_fits(value_sums, bf16_summary_bytes) ||
        !ltx_buffer_fits(thresholds, threshold_bytes) ||
        !ltx_buffer_fits(routes, route_bytes) ||
        (gate && !ltx_buffer_fits(gate, gate_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid BF16 Sol attention arguments");

    @autoreleasepool {
        id<MTLComputePipelineState> rope =
            ltx_pipeline(gpu, LTX_PIPELINE_PACK_ROPE_SPLIT_BF16);
        id<MTLComputePipelineState> pack =
            ltx_pipeline(gpu, LTX_PIPELINE_PACK_HEADS_BF16);
        id<MTLComputePipelineState> reduce =
            ltx_pipeline(gpu, LTX_PIPELINE_SOL_REDUCE_SUMMARIES_BF16);
        id<MTLComputePipelineState> threshold =
            ltx_pipeline(gpu, LTX_PIPELINE_SOL_THRESHOLDS_DIAG_BF16);
        id<MTLComputePipelineState> key_stats =
            ltx_pipeline(gpu, LTX_PIPELINE_SOL_KEY_STATS_BF16);
        id<MTLComputePipelineState> threshold_stats =
            ltx_pipeline(gpu, LTX_PIPELINE_SOL_THRESHOLDS_STATS_BF16);
        id<MTLComputePipelineState> route =
            ltx_pipeline(gpu, LTX_PIPELINE_SOL_ROUTE_MASK_BF16);
        id<MTLComputePipelineState> attention =
            ltx_pipeline(gpu, LTX_PIPELINE_SOL_ATTENTION_TILED_BF16);
        id<MTLComputePipelineState> unpack =
            ltx_pipeline(gpu, LTX_PIPELINE_UNPACK_HEADS_GATE_BF16);
        if (!rope || !pack || !reduce || !threshold || !key_stats ||
            !threshold_stats || !route || !attention || !unpack ||
            reduce.maxTotalThreadsPerThreadgroup < 128u ||
            threshold.maxTotalThreadsPerThreadgroup < 128u ||
            key_stats.maxTotalThreadsPerThreadgroup < 128u ||
            threshold_stats.maxTotalThreadsPerThreadgroup < 128u ||
            route.maxTotalThreadsPerThreadgroup < 128u ||
            attention.maxTotalThreadsPerThreadgroup < 256u || blocks > 64u)
            return ltx_gpu_fail(error, error_size,
                                "device cannot dispatch BF16 Sol attention");

        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal Sol attention command failed");
        ltx_heads_args heads_args = {rows, heads, head_dim, 0u};
        ltx_sol_args sol_args = {
            rows, heads, head_dim, blocks,
            sink_start, sink_end, sink_query_start, sink_query_end,
            1u, 1u, scale * 1.4426950408889634f, tau
        };
        uint32_t count = (uint32_t)elements;
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal Sol Q encoder failed");
        [encoder setComputePipelineState:rope];
        [encoder setBuffer:ltx_buffer(packed_query) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(query) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(cosine) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(sine) offset:0 atIndex:3];
        [encoder setBytes:&heads_args length:sizeof(heads_args) atIndex:4];
        ltx_dispatch_elements(encoder, rope, count);
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal Sol K encoder failed");
        [encoder setComputePipelineState:rope];
        [encoder setBuffer:ltx_buffer(packed_key) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(key) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(cosine) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(sine) offset:0 atIndex:3];
        [encoder setBytes:&heads_args length:sizeof(heads_args) atIndex:4];
        ltx_dispatch_elements(encoder, rope, count);
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal Sol V encoder failed");
        [encoder setComputePipelineState:pack];
        [encoder setBuffer:ltx_buffer(packed_value) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(value) offset:0 atIndex:1];
        [encoder setBytes:&heads_args length:sizeof(heads_args) atIndex:2];
        ltx_dispatch_elements(encoder, pack, count);
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal Sol summary encoder failed");
        [encoder setComputePipelineState:reduce];
        [encoder setBuffer:ltx_buffer(packed_query) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(packed_key) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(packed_value) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(query_centroids) offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(key_centroids) offset:0 atIndex:4];
        [encoder setBuffer:ltx_buffer(value_sums) offset:0 atIndex:5];
        [encoder setBytes:&sol_args length:sizeof(sol_args) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(blocks, heads, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];

        uint64_t stats_bytes =
            (uint64_t)heads * head_dim * 2u * sizeof(float);
        if (route_bytes >= stats_bytes) {
            encoder = [command computeCommandEncoder];
            if (!encoder)
                return ltx_gpu_fail(error, error_size,
                                    "create Metal Sol key-stats encoder failed");
            [encoder setComputePipelineState:key_stats];
            [encoder setBuffer:ltx_buffer(key_centroids) offset:0 atIndex:0];
            [encoder setBuffer:ltx_buffer(routes) offset:0 atIndex:1];
            [encoder setBytes:&sol_args length:sizeof(sol_args) atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(heads, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];

            encoder = [command computeCommandEncoder];
            if (!encoder)
                return ltx_gpu_fail(
                    error, error_size,
                    "create Metal Sol threshold-stats encoder failed");
            [encoder setComputePipelineState:threshold_stats];
            [encoder setBuffer:ltx_buffer(query_centroids) offset:0 atIndex:0];
            [encoder setBuffer:ltx_buffer(routes) offset:0 atIndex:1];
            [encoder setBuffer:ltx_buffer(thresholds) offset:0 atIndex:2];
            [encoder setBytes:&sol_args length:sizeof(sol_args) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(blocks, heads, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        } else {
            encoder = [command computeCommandEncoder];
            if (!encoder)
                return ltx_gpu_fail(error, error_size,
                                    "create Metal Sol threshold encoder failed");
            [encoder setComputePipelineState:threshold];
            [encoder setBuffer:ltx_buffer(query_centroids) offset:0 atIndex:0];
            [encoder setBuffer:ltx_buffer(key_centroids) offset:0 atIndex:1];
            [encoder setBuffer:ltx_buffer(thresholds) offset:0 atIndex:2];
            [encoder setBytes:&sol_args length:sizeof(sol_args) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(blocks, heads, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        }

        encoder = [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal Sol route encoder failed");
        [encoder setComputePipelineState:route];
        [encoder setBuffer:ltx_buffer(query_centroids) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(key_centroids) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(thresholds) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(routes) offset:0 atIndex:3];
        [encoder setBytes:&sol_args length:sizeof(sol_args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(blocks, heads, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal Sol attention encoder failed");
        [encoder setComputePipelineState:attention];
        [encoder setBuffer:ltx_buffer(packed_query) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(packed_key) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(packed_value) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(key_centroids) offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(value_sums) offset:0 atIndex:4];
        [encoder setBuffer:ltx_buffer(routes) offset:0 atIndex:5];
        [encoder setBuffer:ltx_buffer(packed_output) offset:0 atIndex:6];
        [encoder setBytes:&sol_args length:sizeof(sol_args) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(blocks, heads, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal Sol unpack encoder failed");
        heads_args.has_gate = gate ? 1u : 0u;
        [encoder setComputePipelineState:unpack];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(packed_output) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(gate ? gate : packed_output)
                     offset:0 atIndex:2];
        [encoder setBytes:&heads_args length:sizeof(heads_args) atIndex:3];
        ltx_dispatch_elements(encoder, unpack, count);
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "Sol attention", error,
                                  error_size);
    }
}

int ltx_gpu_linear_int8_weight_bf16(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *rotated_input,
                        const ltx_gpu_buffer *weight,
                        const ltx_gpu_buffer *weight_scale,
                        const ltx_gpu_buffer *bias,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t output_dim,
                        char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t bias_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * input_dim, sizeof(uint16_t),
                            &input_bytes) ||
        !ltx_required_bytes((uint64_t)output_dim * input_dim, sizeof(int8_t),
                            &weight_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(float), &scale_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(uint16_t), &bias_bytes) ||
        !ltx_required_bytes((uint64_t)rows * output_dim, sizeof(uint16_t),
                            &output_bytes) ||
        !gpu || !rows || !input_dim || !output_dim ||
        !ltx_buffer_fits(rotated_input, input_bytes) ||
        !ltx_buffer_fits(weight, weight_bytes) ||
        !ltx_buffer_fits(weight_scale, scale_bytes) ||
        !ltx_buffer_fits(output, output_bytes) ||
        (bias && !ltx_buffer_fits(bias, bias_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid Metal INT8-weight linear arguments");
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ltx_pipeline(gpu, LTX_PIPELINE_LINEAR_INT8_WEIGHT_BF16);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
            return ltx_gpu_fail(error, error_size,
                                "INT8-weight linear requires a 16x16 threadgroup");
        id<MTLCommandBuffer> command = [ltx_queue(gpu) commandBuffer];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create Metal INT8 linear command failed");
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!encoder)
            return ltx_gpu_fail(error, error_size,
                                "create Metal INT8 linear encoder failed");
        ltx_linear_args args = {
            rows, input_dim, output_dim, bias ? 1u : 0u
        };
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:ltx_buffer(rotated_input) offset:0 atIndex:0];
        [encoder setBuffer:ltx_buffer(weight) offset:0 atIndex:1];
        [encoder setBuffer:ltx_buffer(weight_scale) offset:0 atIndex:2];
        [encoder setBuffer:ltx_buffer(bias ? bias : rotated_input)
                     offset:0 atIndex:3];
        [encoder setBuffer:ltx_buffer(output) offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        [encoder dispatchThreadgroups:
            MTLSizeMake(((NSUInteger)output_dim + 15u) / 16u,
                        ((NSUInteger)rows + 15u) / 16u, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        return ltx_finish_command(gpu, command, "INT8-weight linear", error,
                                  error_size);
    }
}

static LTXInt8LinearGraph *ltx_int8_linear_graph(
        ltx_gpu *gpu, uint32_t rows, uint32_t input_dim,
        uint32_t output_dim, int has_bias) {
    NSMutableDictionary<NSString *, LTXInt8LinearGraph *> *cache =
        ltx_int8_linear_graphs(gpu);
    NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%d", rows,
                     input_dim, output_dim, has_bias];
    LTXInt8LinearGraph *cached = cache[key];
    if (cached) return cached;

    LTXInt8LinearGraph *linear = [[LTXInt8LinearGraph alloc] init];
    linear.graph = [[MPSGraph alloc] init];
    linear.input_shape = @[@1, @(rows), @(input_dim)];
    linear.weight_shape = @[@1, @(output_dim), @(input_dim)];
    linear.scale_shape = @[@1, @(output_dim), @1];
    linear.bias_shape = @[@1, @1, @(output_dim)];
    linear.output_shape = @[@1, @(rows), @(output_dim)];
    linear.input = [linear.graph placeholderWithShape:linear.input_shape
                                              dataType:MPSDataTypeBFloat16
                                                  name:nil];
    linear.weight = [linear.graph placeholderWithShape:linear.weight_shape
                                               dataType:MPSDataTypeInt8
                                                   name:nil];
    linear.scale = [linear.graph placeholderWithShape:linear.scale_shape
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
    MPSGraphTensor *weight_f32 =
        [linear.graph castTensor:linear.weight
                          toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *scaled_f32 =
        [linear.graph multiplicationWithPrimaryTensor:weight_f32
                                       secondaryTensor:linear.scale name:nil];
    MPSGraphTensor *scaled_bf16 =
        [linear.graph castTensor:scaled_f32
                          toType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *transposed =
        [linear.graph transposeTensor:scaled_bf16 dimension:1
                         withDimension:2 name:nil];
    MPSGraphTensor *output =
        [linear.graph matrixMultiplicationWithPrimaryTensor:linear.input
                                            secondaryTensor:transposed name:nil];
    if (has_bias) {
        linear.bias = [linear.graph placeholderWithShape:linear.bias_shape
                                                 dataType:MPSDataTypeBFloat16
                                                     name:nil];
        output = [linear.graph additionWithPrimaryTensor:output
                                         secondaryTensor:linear.bias name:nil];
    }
    linear.output = [linear.graph castTensor:output
                                      toType:MPSDataTypeBFloat16 name:nil];
    cache[key] = linear;
    return linear;
}

int ltx_gpu_linear_int8_weight_mps_bf16(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *rotated_input,
                        const ltx_gpu_buffer *weight,
                        const ltx_gpu_buffer *weight_scale,
                        const ltx_gpu_buffer *bias,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t output_dim,
                        char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t bias_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * input_dim, sizeof(uint16_t),
                            &input_bytes) ||
        !ltx_required_bytes((uint64_t)output_dim * input_dim, sizeof(int8_t),
                            &weight_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(float), &scale_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(uint16_t), &bias_bytes) ||
        !ltx_required_bytes((uint64_t)rows * output_dim, sizeof(uint16_t),
                            &output_bytes) ||
        !gpu || !rows || !input_dim || !output_dim ||
        !ltx_buffer_fits(rotated_input, input_bytes) ||
        !ltx_buffer_fits(weight, weight_bytes) ||
        !ltx_buffer_fits(weight_scale, scale_bytes) ||
        !ltx_buffer_fits(output, output_bytes) ||
        (bias && !ltx_buffer_fits(bias, bias_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS INT8-weight linear arguments");
    @autoreleasepool {
        LTXInt8LinearGraph *linear = ltx_int8_linear_graph(
            gpu, rows, input_dim, output_dim, bias != NULL);
        if (!linear)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 linear graph failed");
        MPSCommandBuffer *mps_command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!mps_command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 linear command failed");
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(rotated_input)
                        shape:linear.input_shape
                     dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *weight_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(weight)
                        shape:linear.weight_shape
                     dataType:MPSDataTypeInt8];
        MPSGraphTensorData *scale_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(weight_scale)
                        shape:linear.scale_shape
                     dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(output)
                        shape:linear.output_shape
                     dataType:MPSDataTypeBFloat16];
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [@{linear.input: input_data,
               linear.weight: weight_data,
               linear.scale: scale_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:ltx_buffer(bias)
                            shape:linear.bias_shape
                         dataType:MPSDataTypeBFloat16];
            feeds[linear.bias] = bias_data;
        }
        NSDictionary *results = @{linear.output: output_data};
        @try {
            [linear.graph encodeToCommandBuffer:mps_command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message), "MPSGraph INT8 linear: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(gpu, mps_command, "MPSGraph INT8 linear",
                                      error, error_size);
    }
}

static MPSGraphTensor *ltx_graph_convrot_256(
        MPSGraph *graph, MPSGraphTensor *input,
        uint32_t rows, uint32_t columns,
        MPSGraphTensor *hadamard) {
    NSArray<NSNumber *> *flat_shape = @[
        @((uint64_t)rows * (columns / 256u)), @256
    ];
    MPSGraphTensor *flat = [graph reshapeTensor:input
                                      withShape:flat_shape name:nil];
    MPSGraphTensor *rotated =
        [graph matrixMultiplicationWithPrimaryTensor:flat
                                     secondaryTensor:hadamard name:nil];
    return [graph reshapeTensor:rotated
                      withShape:@[@1, @(rows), @(columns)] name:nil];
}

static MPSGraphTensor *ltx_graph_int8_weight_linear(
        MPSGraph *graph, MPSGraphTensor *input,
        MPSGraphTensor *weight, MPSGraphTensor *scale,
        MPSGraphTensor *bias) {
    MPSGraphTensor *weight_f32 = [graph castTensor:weight
                                            toType:MPSDataTypeFloat32
                                              name:nil];
    MPSGraphTensor *scaled_f32 =
        [graph multiplicationWithPrimaryTensor:weight_f32
                                secondaryTensor:scale name:nil];
    MPSGraphTensor *scaled_bf16 = [graph castTensor:scaled_f32
                                             toType:MPSDataTypeBFloat16
                                               name:nil];
    MPSGraphTensor *transposed =
        [graph transposeTensor:scaled_bf16 dimension:1
                 withDimension:2 name:nil];
    MPSGraphTensor *output =
        [graph matrixMultiplicationWithPrimaryTensor:input
                                     secondaryTensor:transposed name:nil];
    if (bias)
        output = [graph additionWithPrimaryTensor:output
                                  secondaryTensor:bias name:nil];
    return [graph castTensor:output toType:MPSDataTypeBFloat16 name:nil];
}

static MPSGraphTensor *ltx_graph_gelu_tanh_bf16(
        MPSGraph *graph, MPSGraphTensor *input) {
    MPSGraphTensor *x = [graph castTensor:input
                                  toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *x2 =
        [graph multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
    MPSGraphTensor *x3 =
        [graph multiplicationWithPrimaryTensor:x2 secondaryTensor:x name:nil];
    MPSGraphTensor *cubic_scale = [graph constantWithScalar:0.044715
                                                   dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cubic =
        [graph multiplicationWithPrimaryTensor:x3
                                secondaryTensor:cubic_scale name:nil];
    MPSGraphTensor *sum =
        [graph additionWithPrimaryTensor:x secondaryTensor:cubic name:nil];
    MPSGraphTensor *sqrt_scale = [graph constantWithScalar:0.7978845608028654
                                                  dataType:MPSDataTypeFloat32];
    MPSGraphTensor *inner =
        [graph multiplicationWithPrimaryTensor:sum
                                secondaryTensor:sqrt_scale name:nil];
    MPSGraphTensor *tanh_value = [graph tanhWithTensor:inner name:nil];
    MPSGraphTensor *one = [graph constantWithScalar:1.0
                                           dataType:MPSDataTypeFloat32];
    MPSGraphTensor *one_plus =
        [graph additionWithPrimaryTensor:one
                          secondaryTensor:tanh_value name:nil];
    MPSGraphTensor *half = [graph constantWithScalar:0.5
                                            dataType:MPSDataTypeFloat32];
    MPSGraphTensor *scaled_x =
        [graph multiplicationWithPrimaryTensor:x secondaryTensor:half name:nil];
    MPSGraphTensor *gelu =
        [graph multiplicationWithPrimaryTensor:scaled_x
                                secondaryTensor:one_plus name:nil];
    return [graph castTensor:gelu toType:MPSDataTypeBFloat16 name:nil];
}

static MPSGraphTensor *ltx_graph_rms_norm_weighted_bf16(
        MPSGraph *graph, MPSGraphTensor *input,
        MPSGraphTensor *weight, float epsilon) {
    MPSGraphTensor *x = [graph castTensor:input
                                  toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *squared = [graph squareWithTensor:x name:nil];
    MPSGraphTensor *mean = [graph meanOfTensor:squared axes:@[@2] name:nil];
    MPSGraphTensor *epsilon_tensor =
        [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
    MPSGraphTensor *mean_plus_epsilon =
        [graph additionWithPrimaryTensor:mean
                          secondaryTensor:epsilon_tensor name:nil];
    MPSGraphTensor *inverse_rms =
        [graph reciprocalSquareRootWithTensor:mean_plus_epsilon name:nil];
    MPSGraphTensor *normalized =
        [graph multiplicationWithPrimaryTensor:x
                                secondaryTensor:inverse_rms name:nil];
    MPSGraphTensor *weight_f32 = [graph castTensor:weight
                                            toType:MPSDataTypeFloat32
                                              name:nil];
    MPSGraphTensor *weighted =
        [graph multiplicationWithPrimaryTensor:normalized
                                secondaryTensor:weight_f32 name:nil];
    return [graph castTensor:weighted toType:MPSDataTypeBFloat16 name:nil];
}

static LTXInt8LinearGraph *ltx_int8_convrot_linear_graph(
        ltx_gpu *gpu, uint32_t rows, uint32_t input_dim,
        uint32_t output_dim, int has_bias) {
    NSMutableDictionary<NSString *, LTXInt8LinearGraph *> *cache =
        ltx_int8_linear_graphs(gpu);
    NSString *key = [NSString stringWithFormat:@"convrot:%u:%u:%u:%d",
                     rows, input_dim, output_dim, has_bias];
    LTXInt8LinearGraph *cached = cache[key];
    if (cached) return cached;

    LTXInt8LinearGraph *linear = [[LTXInt8LinearGraph alloc] init];
    linear.graph = [[MPSGraph alloc] init];
    linear.input_shape = @[@1, @(rows), @(input_dim)];
    linear.weight_shape = @[@1, @(output_dim), @(input_dim)];
    linear.scale_shape = @[@1, @(output_dim), @1];
    linear.bias_shape = @[@1, @1, @(output_dim)];
    linear.output_shape = @[@1, @(rows), @(output_dim)];
    linear.input = [linear.graph placeholderWithShape:linear.input_shape
                                              dataType:MPSDataTypeBFloat16
                                                  name:nil];
    linear.weight = [linear.graph placeholderWithShape:linear.weight_shape
                                               dataType:MPSDataTypeInt8
                                                   name:nil];
    linear.scale = [linear.graph placeholderWithShape:linear.scale_shape
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
    if (has_bias)
        linear.bias = [linear.graph placeholderWithShape:linear.bias_shape
                                                 dataType:MPSDataTypeBFloat16
                                                     name:nil];

    MPSGraphTensor *hadamard = [linear.graph constantWithData:
        ltx_hadamard_256_bf16() shape:@[@256, @256]
                                dataType:MPSDataTypeBFloat16];
    MPSGraphTensor *rotated = ltx_graph_convrot_256(
        linear.graph, linear.input, rows, input_dim, hadamard);
    linear.output = ltx_graph_int8_weight_linear(
        linear.graph, rotated, linear.weight, linear.scale, linear.bias);
    cache[key] = linear;
    return linear;
}

int ltx_gpu_linear_int8_convrot_mps_bf16(
                        ltx_gpu *gpu, ltx_gpu_buffer *output,
                        const ltx_gpu_buffer *input,
                        const ltx_gpu_buffer *weight,
                        const ltx_gpu_buffer *weight_scale,
                        const ltx_gpu_buffer *bias,
                        uint32_t rows, uint32_t input_dim,
                        uint32_t output_dim,
                        uint32_t convrot_group_size,
                        char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t bias_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * input_dim, sizeof(uint16_t),
                            &input_bytes) ||
        !ltx_required_bytes((uint64_t)output_dim * input_dim, sizeof(int8_t),
                            &weight_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(float), &scale_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(uint16_t), &bias_bytes) ||
        !ltx_required_bytes((uint64_t)rows * output_dim, sizeof(uint16_t),
                            &output_bytes) ||
        !gpu || !rows || !input_dim || !output_dim ||
        convrot_group_size != 256u || input_dim % 256u ||
        !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(weight, weight_bytes) ||
        !ltx_buffer_fits(weight_scale, scale_bytes) ||
        !ltx_buffer_fits(output, output_bytes) ||
        (bias && !ltx_buffer_fits(bias, bias_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS ConvRot INT8 linear arguments");

    @autoreleasepool {
        LTXInt8LinearGraph *linear = ltx_int8_convrot_linear_graph(
            gpu, rows, input_dim, output_dim, bias != NULL);
        if (!linear)
            return ltx_gpu_fail(
                error, error_size,
                "create MPS ConvRot INT8 linear graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(
                error, error_size,
                "create MPS ConvRot INT8 linear command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
        feeds[linear.input] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(input) shape:linear.input_shape
                     dataType:MPSDataTypeBFloat16];
        feeds[linear.weight] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(weight) shape:linear.weight_shape
                     dataType:MPSDataTypeInt8];
        feeds[linear.scale] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(weight_scale) shape:linear.scale_shape
                     dataType:MPSDataTypeFloat32];
        if (bias)
            feeds[linear.bias] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:ltx_buffer(bias) shape:linear.bias_shape
                         dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(output) shape:linear.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [linear.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{linear.output: output_data}
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message),
                     "MPSGraph ConvRot INT8 linear: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(
            gpu, command, "MPSGraph ConvRot INT8 linear",
            error, error_size);
    }
}

static LTXInt8MLPGraph *ltx_int8_mlp_graph(
        ltx_gpu *gpu, uint32_t rows, uint32_t input_dim,
        uint32_t hidden_dim, uint32_t output_dim,
        int fc1_has_bias, int fc2_has_bias) {
    NSMutableDictionary<NSString *, LTXInt8MLPGraph *> *cache =
        ltx_int8_mlp_graphs(gpu);
    NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%u:%d:%d",
                     rows, input_dim, hidden_dim, output_dim,
                     fc1_has_bias, fc2_has_bias];
    LTXInt8MLPGraph *cached = cache[key];
    if (cached) return cached;

    LTXInt8MLPGraph *mlp = [[LTXInt8MLPGraph alloc] init];
    mlp.graph = [[MPSGraph alloc] init];
    mlp.input_shape = @[@1, @(rows), @(input_dim)];
    mlp.fc1_weight_shape = @[@1, @(hidden_dim), @(input_dim)];
    mlp.fc1_scale_shape = @[@1, @(hidden_dim), @1];
    mlp.fc1_bias_shape = @[@1, @1, @(hidden_dim)];
    mlp.fc2_weight_shape = @[@1, @(output_dim), @(hidden_dim)];
    mlp.fc2_scale_shape = @[@1, @(output_dim), @1];
    mlp.fc2_bias_shape = @[@1, @1, @(output_dim)];
    mlp.output_shape = @[@1, @(rows), @(output_dim)];
    mlp.input = [mlp.graph placeholderWithShape:mlp.input_shape
                                        dataType:MPSDataTypeBFloat16 name:nil];
    mlp.fc1_weight = [mlp.graph placeholderWithShape:mlp.fc1_weight_shape
                                             dataType:MPSDataTypeInt8 name:nil];
    mlp.fc1_scale = [mlp.graph placeholderWithShape:mlp.fc1_scale_shape
                                            dataType:MPSDataTypeFloat32 name:nil];
    mlp.fc2_weight = [mlp.graph placeholderWithShape:mlp.fc2_weight_shape
                                             dataType:MPSDataTypeInt8 name:nil];
    mlp.fc2_scale = [mlp.graph placeholderWithShape:mlp.fc2_scale_shape
                                            dataType:MPSDataTypeFloat32 name:nil];
    if (fc1_has_bias)
        mlp.fc1_bias = [mlp.graph placeholderWithShape:mlp.fc1_bias_shape
                                               dataType:MPSDataTypeBFloat16
                                                   name:nil];
    if (fc2_has_bias)
        mlp.fc2_bias = [mlp.graph placeholderWithShape:mlp.fc2_bias_shape
                                               dataType:MPSDataTypeBFloat16
                                                   name:nil];

    MPSGraphTensor *hadamard = [mlp.graph constantWithData:
        ltx_hadamard_256_bf16() shape:@[@256, @256]
                                dataType:MPSDataTypeBFloat16];
    MPSGraphTensor *rotated_input = ltx_graph_convrot_256(
        mlp.graph, mlp.input, rows, input_dim, hadamard);
    MPSGraphTensor *hidden = ltx_graph_int8_weight_linear(
        mlp.graph, rotated_input, mlp.fc1_weight, mlp.fc1_scale,
        mlp.fc1_bias);
    hidden = ltx_graph_gelu_tanh_bf16(mlp.graph, hidden);
    MPSGraphTensor *rotated_hidden = ltx_graph_convrot_256(
        mlp.graph, hidden, rows, hidden_dim, hadamard);
    mlp.output = ltx_graph_int8_weight_linear(
        mlp.graph, rotated_hidden, mlp.fc2_weight, mlp.fc2_scale,
        mlp.fc2_bias);
    cache[key] = mlp;
    return mlp;
}

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
                        char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t fc1_weight_bytes = 0;
    uint64_t fc1_scale_bytes = 0;
    uint64_t fc1_bias_bytes = 0;
    uint64_t fc2_weight_bytes = 0;
    uint64_t fc2_scale_bytes = 0;
    uint64_t fc2_bias_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * input_dim, sizeof(uint16_t),
                            &input_bytes) ||
        !ltx_required_bytes((uint64_t)hidden_dim * input_dim, sizeof(int8_t),
                            &fc1_weight_bytes) ||
        !ltx_required_bytes(hidden_dim, sizeof(float), &fc1_scale_bytes) ||
        !ltx_required_bytes(hidden_dim, sizeof(uint16_t), &fc1_bias_bytes) ||
        !ltx_required_bytes((uint64_t)output_dim * hidden_dim, sizeof(int8_t),
                            &fc2_weight_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(float), &fc2_scale_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(uint16_t), &fc2_bias_bytes) ||
        !ltx_required_bytes((uint64_t)rows * output_dim, sizeof(uint16_t),
                            &output_bytes) ||
        !gpu || !rows || !input_dim || !hidden_dim || !output_dim ||
        convrot_group_size != 256u || input_dim % 256u || hidden_dim % 256u ||
        !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(fc1_weight, fc1_weight_bytes) ||
        !ltx_buffer_fits(fc1_scale, fc1_scale_bytes) ||
        !ltx_buffer_fits(fc2_weight, fc2_weight_bytes) ||
        !ltx_buffer_fits(fc2_scale, fc2_scale_bytes) ||
        !ltx_buffer_fits(output, output_bytes) ||
        (fc1_bias && !ltx_buffer_fits(fc1_bias, fc1_bias_bytes)) ||
        (fc2_bias && !ltx_buffer_fits(fc2_bias, fc2_bias_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS ConvRot INT8 MLP arguments");
    @autoreleasepool {
        LTXInt8MLPGraph *mlp = ltx_int8_mlp_graph(
            gpu, rows, input_dim, hidden_dim, output_dim,
            fc1_bias != NULL, fc2_bias != NULL);
        if (!mlp)
            return ltx_gpu_fail(error, error_size,
                                "create MPS ConvRot INT8 MLP graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS ConvRot INT8 MLP command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
        feeds[mlp.input] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(input) shape:mlp.input_shape
                     dataType:MPSDataTypeBFloat16];
        feeds[mlp.fc1_weight] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(fc1_weight)
                        shape:mlp.fc1_weight_shape dataType:MPSDataTypeInt8];
        feeds[mlp.fc1_scale] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(fc1_scale)
                        shape:mlp.fc1_scale_shape dataType:MPSDataTypeFloat32];
        feeds[mlp.fc2_weight] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(fc2_weight)
                        shape:mlp.fc2_weight_shape dataType:MPSDataTypeInt8];
        feeds[mlp.fc2_scale] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(fc2_scale)
                        shape:mlp.fc2_scale_shape dataType:MPSDataTypeFloat32];
        if (fc1_bias)
            feeds[mlp.fc1_bias] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:ltx_buffer(fc1_bias)
                            shape:mlp.fc1_bias_shape
                         dataType:MPSDataTypeBFloat16];
        if (fc2_bias)
            feeds[mlp.fc2_bias] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:ltx_buffer(fc2_bias)
                            shape:mlp.fc2_bias_shape
                         dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(output) shape:mlp.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [mlp.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{mlp.output: output_data}
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message), "MPSGraph INT8 MLP: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(gpu, command, "MPSGraph INT8 MLP",
                                      error, error_size);
    }
}

static LTXInt8QKVGraph *ltx_int8_qkv_graph(
        ltx_gpu *gpu, uint32_t rows, uint32_t input_dim,
        uint32_t inner_dim, int query_has_bias,
        int key_has_bias, int value_has_bias,
        float norm_epsilon) {
    NSMutableDictionary<NSString *, LTXInt8QKVGraph *> *cache =
        ltx_int8_qkv_graphs(gpu);
    NSString *key = [NSString stringWithFormat:
        @"%u:%u:%u:%d:%d:%d:%.9g", rows, input_dim, inner_dim,
        query_has_bias, key_has_bias, value_has_bias, norm_epsilon];
    LTXInt8QKVGraph *cached = cache[key];
    if (cached) return cached;

    LTXInt8QKVGraph *qkv = [[LTXInt8QKVGraph alloc] init];
    qkv.graph = [[MPSGraph alloc] init];
    qkv.input_shape = @[@1, @(rows), @(input_dim)];
    qkv.weight_shape = @[@1, @(inner_dim), @(input_dim)];
    qkv.scale_shape = @[@1, @(inner_dim), @1];
    qkv.bias_shape = @[@1, @1, @(inner_dim)];
    qkv.norm_shape = @[@1, @1, @(inner_dim)];
    qkv.output_shape = @[@1, @(rows), @(inner_dim)];
    qkv.input = [qkv.graph placeholderWithShape:qkv.input_shape
                                        dataType:MPSDataTypeBFloat16 name:nil];
    qkv.query_weight = [qkv.graph placeholderWithShape:qkv.weight_shape
                                               dataType:MPSDataTypeInt8 name:nil];
    qkv.query_scale = [qkv.graph placeholderWithShape:qkv.scale_shape
                                              dataType:MPSDataTypeFloat32 name:nil];
    qkv.key_weight = [qkv.graph placeholderWithShape:qkv.weight_shape
                                             dataType:MPSDataTypeInt8 name:nil];
    qkv.key_scale = [qkv.graph placeholderWithShape:qkv.scale_shape
                                            dataType:MPSDataTypeFloat32 name:nil];
    qkv.value_weight = [qkv.graph placeholderWithShape:qkv.weight_shape
                                               dataType:MPSDataTypeInt8 name:nil];
    qkv.value_scale = [qkv.graph placeholderWithShape:qkv.scale_shape
                                              dataType:MPSDataTypeFloat32 name:nil];
    if (query_has_bias)
        qkv.query_bias = [qkv.graph placeholderWithShape:qkv.bias_shape
                                                dataType:MPSDataTypeBFloat16
                                                    name:nil];
    if (key_has_bias)
        qkv.key_bias = [qkv.graph placeholderWithShape:qkv.bias_shape
                                              dataType:MPSDataTypeBFloat16
                                                  name:nil];
    if (value_has_bias)
        qkv.value_bias = [qkv.graph placeholderWithShape:qkv.bias_shape
                                                dataType:MPSDataTypeBFloat16
                                                    name:nil];
    qkv.query_norm_weight = [qkv.graph placeholderWithShape:qkv.norm_shape
                                                    dataType:MPSDataTypeBFloat16
                                                        name:nil];
    qkv.key_norm_weight = [qkv.graph placeholderWithShape:qkv.norm_shape
                                                  dataType:MPSDataTypeBFloat16
                                                      name:nil];

    MPSGraphTensor *hadamard = [qkv.graph constantWithData:
        ltx_hadamard_256_bf16() shape:@[@256, @256]
                                dataType:MPSDataTypeBFloat16];
    MPSGraphTensor *rotated = ltx_graph_convrot_256(
        qkv.graph, qkv.input, rows, input_dim, hadamard);
    MPSGraphTensor *query = ltx_graph_int8_weight_linear(
        qkv.graph, rotated, qkv.query_weight, qkv.query_scale,
        qkv.query_bias);
    MPSGraphTensor *key_projection = ltx_graph_int8_weight_linear(
        qkv.graph, rotated, qkv.key_weight, qkv.key_scale,
        qkv.key_bias);
    qkv.value_output = ltx_graph_int8_weight_linear(
        qkv.graph, rotated, qkv.value_weight, qkv.value_scale,
        qkv.value_bias);
    qkv.query_output = ltx_graph_rms_norm_weighted_bf16(
        qkv.graph, query, qkv.query_norm_weight, norm_epsilon);
    qkv.key_output = ltx_graph_rms_norm_weighted_bf16(
        qkv.graph, key_projection, qkv.key_norm_weight, norm_epsilon);
    cache[key] = qkv;
    return qkv;
}

static LTXInt8QKVGraph *ltx_int8_packed_qkv_graph(
        ltx_gpu *gpu, uint32_t rows, uint32_t input_dim,
        uint32_t inner_dim, int has_bias, float norm_epsilon) {
    NSMutableDictionary<NSString *, LTXInt8QKVGraph *> *cache =
        ltx_int8_qkv_graphs(gpu);
    NSString *key = [NSString stringWithFormat:
        @"packed:%u:%u:%u:%d:%.9g", rows, input_dim, inner_dim,
        has_bias, norm_epsilon];
    LTXInt8QKVGraph *cached = cache[key];
    if (cached) return cached;

    LTXInt8QKVGraph *qkv = [[LTXInt8QKVGraph alloc] init];
    qkv.graph = [[MPSGraph alloc] init];
    qkv.input_shape = @[@1, @(rows), @(input_dim)];
    qkv.packed_weight_shape = @[@1, @(inner_dim * 3u), @(input_dim)];
    qkv.packed_scale_shape = @[@1, @(inner_dim * 3u), @1];
    qkv.packed_bias_shape = @[@1, @1, @(inner_dim * 3u)];
    qkv.norm_shape = @[@1, @1, @(inner_dim)];
    qkv.output_shape = @[@1, @(rows), @(inner_dim)];
    qkv.input = [qkv.graph placeholderWithShape:qkv.input_shape
                                        dataType:MPSDataTypeBFloat16 name:nil];
    qkv.packed_weight = [qkv.graph
        placeholderWithShape:qkv.packed_weight_shape
                  dataType:MPSDataTypeInt8 name:nil];
    qkv.packed_scale = [qkv.graph
        placeholderWithShape:qkv.packed_scale_shape
                  dataType:MPSDataTypeFloat32 name:nil];
    if (has_bias)
        qkv.packed_bias = [qkv.graph
            placeholderWithShape:qkv.packed_bias_shape
                      dataType:MPSDataTypeBFloat16 name:nil];
    qkv.query_norm_weight = [qkv.graph placeholderWithShape:qkv.norm_shape
                                                    dataType:MPSDataTypeBFloat16
                                                        name:nil];
    qkv.key_norm_weight = [qkv.graph placeholderWithShape:qkv.norm_shape
                                                  dataType:MPSDataTypeBFloat16
                                                      name:nil];

    MPSGraphTensor *hadamard = [qkv.graph constantWithData:
        ltx_hadamard_256_bf16() shape:@[@256, @256]
                                dataType:MPSDataTypeBFloat16];
    MPSGraphTensor *rotated = ltx_graph_convrot_256(
        qkv.graph, qkv.input, rows, input_dim, hadamard);
    MPSGraphTensor *packed = ltx_graph_int8_weight_linear(
        qkv.graph, rotated, qkv.packed_weight, qkv.packed_scale,
        qkv.packed_bias);
    NSArray<MPSGraphTensor *> *parts = [qkv.graph splitTensor:packed
        splitSizes:@[@(inner_dim), @(inner_dim), @(inner_dim)]
        axis:2 name:nil];
    qkv.query_output = ltx_graph_rms_norm_weighted_bf16(
        qkv.graph, parts[0], qkv.query_norm_weight, norm_epsilon);
    qkv.key_output = ltx_graph_rms_norm_weighted_bf16(
        qkv.graph, parts[1], qkv.key_norm_weight, norm_epsilon);
    qkv.value_output = parts[2];
    cache[key] = qkv;
    return qkv;
}

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
                        char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t vector_bytes = 0;
    uint64_t output_bytes = 0;
    if (!ltx_required_bytes((uint64_t)rows * input_dim, sizeof(uint16_t),
                            &input_bytes) ||
        !ltx_required_bytes((uint64_t)inner_dim * input_dim, sizeof(int8_t),
                            &weight_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(float), &scale_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(uint16_t), &vector_bytes) ||
        !ltx_required_bytes((uint64_t)rows * inner_dim, sizeof(uint16_t),
                            &output_bytes) ||
        !gpu || !rows || !input_dim || !inner_dim ||
        convrot_group_size != 256u || input_dim % 256u ||
        !(norm_epsilon > 0.0f) ||
        !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(query_weight, weight_bytes) ||
        !ltx_buffer_fits(query_scale, scale_bytes) ||
        !ltx_buffer_fits(key_weight, weight_bytes) ||
        !ltx_buffer_fits(key_scale, scale_bytes) ||
        !ltx_buffer_fits(value_weight, weight_bytes) ||
        !ltx_buffer_fits(value_scale, scale_bytes) ||
        !ltx_buffer_fits(query_norm_weight, vector_bytes) ||
        !ltx_buffer_fits(key_norm_weight, vector_bytes) ||
        !ltx_buffer_fits(query_output, output_bytes) ||
        !ltx_buffer_fits(key_output, output_bytes) ||
        !ltx_buffer_fits(value_output, output_bytes) ||
        (query_bias && !ltx_buffer_fits(query_bias, vector_bytes)) ||
        (key_bias && !ltx_buffer_fits(key_bias, vector_bytes)) ||
        (value_bias && !ltx_buffer_fits(value_bias, vector_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS ConvRot INT8 QKV arguments");

    @autoreleasepool {
        LTXInt8QKVGraph *qkv = ltx_int8_qkv_graph(
            gpu, rows, input_dim, inner_dim,
            query_bias != NULL, key_bias != NULL, value_bias != NULL,
            norm_epsilon);
        if (!qkv)
            return ltx_gpu_fail(error, error_size,
                                "create MPS ConvRot INT8 QKV graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS ConvRot INT8 QKV command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
        feeds[qkv.input] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(input) shape:qkv.input_shape
                     dataType:MPSDataTypeBFloat16];
#define LTX_QKV_FEED(TENSOR, BUFFER, SHAPE, TYPE) \
        feeds[(TENSOR)] = [[MPSGraphTensorData alloc] \
            initWithMTLBuffer:ltx_buffer((BUFFER)) shape:(SHAPE) \
                     dataType:(TYPE)]
        LTX_QKV_FEED(qkv.query_weight, query_weight, qkv.weight_shape,
                     MPSDataTypeInt8);
        LTX_QKV_FEED(qkv.query_scale, query_scale, qkv.scale_shape,
                     MPSDataTypeFloat32);
        LTX_QKV_FEED(qkv.key_weight, key_weight, qkv.weight_shape,
                     MPSDataTypeInt8);
        LTX_QKV_FEED(qkv.key_scale, key_scale, qkv.scale_shape,
                     MPSDataTypeFloat32);
        LTX_QKV_FEED(qkv.value_weight, value_weight, qkv.weight_shape,
                     MPSDataTypeInt8);
        LTX_QKV_FEED(qkv.value_scale, value_scale, qkv.scale_shape,
                     MPSDataTypeFloat32);
        LTX_QKV_FEED(qkv.query_norm_weight, query_norm_weight, qkv.norm_shape,
                     MPSDataTypeBFloat16);
        LTX_QKV_FEED(qkv.key_norm_weight, key_norm_weight, qkv.norm_shape,
                     MPSDataTypeBFloat16);
        if (query_bias)
            LTX_QKV_FEED(qkv.query_bias, query_bias, qkv.bias_shape,
                         MPSDataTypeBFloat16);
        if (key_bias)
            LTX_QKV_FEED(qkv.key_bias, key_bias, qkv.bias_shape,
                         MPSDataTypeBFloat16);
        if (value_bias)
            LTX_QKV_FEED(qkv.value_bias, value_bias, qkv.bias_shape,
                         MPSDataTypeBFloat16);
#undef LTX_QKV_FEED

        MPSGraphTensorData *query_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(query_output) shape:qkv.output_shape
                     dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *key_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(key_output) shape:qkv.output_shape
                     dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *value_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(value_output) shape:qkv.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [qkv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{
                    qkv.query_output: query_data,
                    qkv.key_output: key_data,
                    qkv.value_output: value_data,
                }
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message), "MPSGraph INT8 QKV: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(gpu, command, "MPSGraph INT8 QKV",
                                      error, error_size);
    }
}

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
                        char *error, size_t error_size) {
    uint64_t input_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t vector_bytes = 0;
    uint64_t packed_vector_bytes = 0;
    uint64_t output_bytes = 0;
    if (inner_dim > UINT32_MAX / 3u ||
        !ltx_required_bytes((uint64_t)rows * input_dim, sizeof(uint16_t),
                            &input_bytes) ||
        !ltx_required_bytes((uint64_t)inner_dim * 3u * input_dim,
                            sizeof(int8_t), &weight_bytes) ||
        !ltx_required_bytes((uint64_t)inner_dim * 3u, sizeof(float),
                            &scale_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(uint16_t), &vector_bytes) ||
        !ltx_required_bytes((uint64_t)inner_dim * 3u, sizeof(uint16_t),
                            &packed_vector_bytes) ||
        !ltx_required_bytes((uint64_t)rows * inner_dim, sizeof(uint16_t),
                            &output_bytes) ||
        !gpu || !rows || !input_dim || !inner_dim ||
        convrot_group_size != 256u || input_dim % 256u ||
        !(norm_epsilon > 0.0f) ||
        !ltx_buffer_fits(input, input_bytes) ||
        !ltx_buffer_fits(packed_weight, weight_bytes) ||
        !ltx_buffer_fits(packed_scale, scale_bytes) ||
        !ltx_buffer_fits(query_norm_weight, vector_bytes) ||
        !ltx_buffer_fits(key_norm_weight, vector_bytes) ||
        !ltx_buffer_fits(query_output, output_bytes) ||
        !ltx_buffer_fits(key_output, output_bytes) ||
        !ltx_buffer_fits(value_output, output_bytes) ||
        (packed_bias && !ltx_buffer_fits(packed_bias,
                                        packed_vector_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid packed MPS ConvRot INT8 QKV arguments");

    @autoreleasepool {
        LTXInt8QKVGraph *qkv = ltx_int8_packed_qkv_graph(
            gpu, rows, input_dim, inner_dim, packed_bias != NULL,
            norm_epsilon);
        if (!qkv)
            return ltx_gpu_fail(error, error_size,
                                "create packed MPS ConvRot INT8 QKV graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create packed MPS ConvRot INT8 QKV command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
        feeds[qkv.input] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(input) shape:qkv.input_shape
                     dataType:MPSDataTypeBFloat16];
        feeds[qkv.packed_weight] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(packed_weight)
                        shape:qkv.packed_weight_shape
                     dataType:MPSDataTypeInt8];
        feeds[qkv.packed_scale] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(packed_scale)
                        shape:qkv.packed_scale_shape
                     dataType:MPSDataTypeFloat32];
        feeds[qkv.query_norm_weight] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(query_norm_weight)
                        shape:qkv.norm_shape
                     dataType:MPSDataTypeBFloat16];
        feeds[qkv.key_norm_weight] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(key_norm_weight)
                        shape:qkv.norm_shape
                     dataType:MPSDataTypeBFloat16];
        if (packed_bias)
            feeds[qkv.packed_bias] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:ltx_buffer(packed_bias)
                            shape:qkv.packed_bias_shape
                         dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *query_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(query_output) shape:qkv.output_shape
                     dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *key_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(key_output) shape:qkv.output_shape
                     dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *value_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(value_output) shape:qkv.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [qkv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{
                    qkv.query_output: query_data,
                    qkv.key_output: key_data,
                    qkv.value_output: value_data,
                }
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message), "packed MPSGraph INT8 QKV: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(
            gpu, command, "packed MPSGraph INT8 QKV", error, error_size);
    }
}

static LTXSDPAGraph *ltx_sdpa_graph(
        ltx_gpu *gpu, uint32_t heads,
        uint32_t query_rows, uint32_t key_value_rows,
        uint32_t head_dim, float scale) {
    NSMutableDictionary<NSString *, LTXSDPAGraph *> *cache =
        ltx_sdpa_graphs(gpu);
    NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%u:%.9g",
                     heads, query_rows, key_value_rows, head_dim, scale];
    LTXSDPAGraph *cached = cache[key];
    if (cached) return cached;

    LTXSDPAGraph *sdpa = [[LTXSDPAGraph alloc] init];
    sdpa.graph = [[MPSGraph alloc] init];
    sdpa.query_shape = @[@1, @(heads), @(query_rows), @(head_dim)];
    sdpa.key_value_shape =
        @[@1, @(heads), @(key_value_rows), @(head_dim)];
    sdpa.output_shape = sdpa.query_shape;
    sdpa.query = [sdpa.graph placeholderWithShape:sdpa.query_shape
                                           dataType:MPSDataTypeBFloat16
                                               name:nil];
    sdpa.key = [sdpa.graph placeholderWithShape:sdpa.key_value_shape
                                         dataType:MPSDataTypeBFloat16
                                             name:nil];
    sdpa.value = [sdpa.graph placeholderWithShape:sdpa.key_value_shape
                                           dataType:MPSDataTypeBFloat16
                                               name:nil];
    MPSGraphTensor *output =
        [sdpa.graph scaledDotProductAttentionWithQueryTensor:sdpa.query
                                                   keyTensor:sdpa.key
                                                 valueTensor:sdpa.value
                                                       scale:scale
                                                        name:nil];
    sdpa.output = [sdpa.graph castTensor:output
                                  toType:MPSDataTypeBFloat16 name:nil];
    cache[key] = sdpa;
    return sdpa;
}

int ltx_gpu_sdpa_mps_bf16(ltx_gpu *gpu, ltx_gpu_buffer *output,
                          const ltx_gpu_buffer *query,
                          const ltx_gpu_buffer *key,
                          const ltx_gpu_buffer *value,
                          uint32_t heads, uint32_t query_rows,
                          uint32_t key_value_rows, uint32_t head_dim,
                          float scale,
                          char *error, size_t error_size) {
    uint64_t query_bytes = 0;
    uint64_t key_value_bytes = 0;
    if (!ltx_required_bytes(
            (uint64_t)heads * query_rows * head_dim,
            sizeof(uint16_t), &query_bytes) ||
        !ltx_required_bytes(
            (uint64_t)heads * key_value_rows * head_dim,
            sizeof(uint16_t), &key_value_bytes) ||
        !gpu || !heads || !query_rows || !key_value_rows || !head_dim ||
        !(scale > 0.0f) || !isfinite(scale) ||
        !ltx_buffer_fits(query, query_bytes) ||
        !ltx_buffer_fits(key, key_value_bytes) ||
        !ltx_buffer_fits(value, key_value_bytes) ||
        !ltx_buffer_fits(output, query_bytes))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS BF16 SDPA arguments");
    @autoreleasepool {
        LTXSDPAGraph *sdpa = ltx_sdpa_graph(
            gpu, heads, query_rows, key_value_rows, head_dim, scale);
        if (!sdpa)
            return ltx_gpu_fail(error, error_size,
                                "create MPS BF16 SDPA graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS BF16 SDPA command failed");
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
            sdpa.query: [[MPSGraphTensorData alloc]
                initWithMTLBuffer:ltx_buffer(query) shape:sdpa.query_shape
                         dataType:MPSDataTypeBFloat16],
            sdpa.key: [[MPSGraphTensorData alloc]
                initWithMTLBuffer:ltx_buffer(key) shape:sdpa.key_value_shape
                         dataType:MPSDataTypeBFloat16],
            sdpa.value: [[MPSGraphTensorData alloc]
                initWithMTLBuffer:ltx_buffer(value) shape:sdpa.key_value_shape
                         dataType:MPSDataTypeBFloat16],
        };
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(output) shape:sdpa.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [sdpa.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{sdpa.output: output_data}
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message), "MPSGraph BF16 SDPA: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(gpu, command, "MPSGraph BF16 SDPA",
                                      error, error_size);
    }
}

static MPSGraphTensor *ltx_graph_to_head_major(
        MPSGraph *graph, MPSGraphTensor *input,
        uint32_t rows, uint32_t heads, uint32_t head_dim) {
    MPSGraphTensor *reshaped = [graph reshapeTensor:input
        withShape:@[@1, @(rows), @(heads), @(head_dim)] name:nil];
    return [graph transposeTensor:reshaped
                       permutation:@[@0, @2, @1, @3] name:nil];
}

static MPSGraphTensor *ltx_graph_apply_rope_split(
        MPSGraph *graph, MPSGraphTensor *input,
        MPSGraphTensor *cosine, MPSGraphTensor *sine,
        uint32_t head_dim) {
    NSArray<MPSGraphTensor *> *parts = [graph splitTensor:input
        splitSizes:@[@(head_dim / 2u), @(head_dim / 2u)] axis:3 name:nil];
    MPSGraphTensor *first_cos =
        [graph multiplicationWithPrimaryTensor:parts[0]
                                secondaryTensor:cosine name:nil];
    MPSGraphTensor *second_sine =
        [graph multiplicationWithPrimaryTensor:parts[1]
                                secondaryTensor:sine name:nil];
    MPSGraphTensor *first =
        [graph subtractionWithPrimaryTensor:first_cos
                              secondaryTensor:second_sine name:nil];
    MPSGraphTensor *first_sine =
        [graph multiplicationWithPrimaryTensor:parts[0]
                                secondaryTensor:sine name:nil];
    MPSGraphTensor *second_cos =
        [graph multiplicationWithPrimaryTensor:parts[1]
                                secondaryTensor:cosine name:nil];
    MPSGraphTensor *second =
        [graph additionWithPrimaryTensor:first_sine
                          secondaryTensor:second_cos name:nil];
    return [graph concatTensors:@[first, second] dimension:3 name:nil];
}

static LTXSelfAttentionCoreGraph *ltx_self_attention_core_graph(
        ltx_gpu *gpu, uint32_t rows, uint32_t heads,
        uint32_t head_dim, float scale, int has_gate) {
    NSMutableDictionary<NSString *, LTXSelfAttentionCoreGraph *> *cache =
        ltx_self_attention_core_graphs(gpu);
    NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%.9g:%d",
                     rows, heads, head_dim, scale, has_gate];
    LTXSelfAttentionCoreGraph *cached = cache[key];
    if (cached) return cached;

    uint64_t inner_dim = (uint64_t)heads * head_dim;
    LTXSelfAttentionCoreGraph *core =
        [[LTXSelfAttentionCoreGraph alloc] init];
    core.graph = [[MPSGraph alloc] init];
    core.input_shape = @[@1, @(rows), @(inner_dim)];
    core.frequency_shape =
        @[@1, @(heads), @(rows), @(head_dim / 2u)];
    core.gate_shape = @[@1, @(rows), @(heads)];
    core.output_shape = core.input_shape;
    core.query = [core.graph placeholderWithShape:core.input_shape
                                          dataType:MPSDataTypeBFloat16
                                              name:nil];
    core.key = [core.graph placeholderWithShape:core.input_shape
                                        dataType:MPSDataTypeBFloat16
                                            name:nil];
    core.value = [core.graph placeholderWithShape:core.input_shape
                                          dataType:MPSDataTypeBFloat16
                                              name:nil];
    core.cosine = [core.graph placeholderWithShape:core.frequency_shape
                                           dataType:MPSDataTypeBFloat16
                                               name:nil];
    core.sine = [core.graph placeholderWithShape:core.frequency_shape
                                         dataType:MPSDataTypeBFloat16
                                             name:nil];
    if (has_gate)
        core.gate = [core.graph placeholderWithShape:core.gate_shape
                                             dataType:MPSDataTypeBFloat16
                                                 name:nil];

    MPSGraphTensor *query = ltx_graph_to_head_major(
        core.graph, core.query, rows, heads, head_dim);
    MPSGraphTensor *key_projection = ltx_graph_to_head_major(
        core.graph, core.key, rows, heads, head_dim);
    MPSGraphTensor *value = ltx_graph_to_head_major(
        core.graph, core.value, rows, heads, head_dim);
    query = ltx_graph_apply_rope_split(
        core.graph, query, core.cosine, core.sine, head_dim);
    key_projection = ltx_graph_apply_rope_split(
        core.graph, key_projection, core.cosine, core.sine, head_dim);
    MPSGraphTensor *attention =
        [core.graph scaledDotProductAttentionWithQueryTensor:query
                                                   keyTensor:key_projection
                                                 valueTensor:value
                                                       scale:scale
                                                        name:nil];
    if (has_gate) {
        MPSGraphTensor *gate_head_major = [core.graph transposeTensor:core.gate
            permutation:@[@0, @2, @1] name:nil];
        gate_head_major = [core.graph reshapeTensor:gate_head_major
            withShape:@[@1, @(heads), @(rows), @1] name:nil];
        attention = [core.graph multiplicationWithPrimaryTensor:attention
                                                secondaryTensor:gate_head_major
                                                          name:nil];
    }
    MPSGraphTensor *row_major = [core.graph transposeTensor:attention
        permutation:@[@0, @2, @1, @3] name:nil];
    row_major = [core.graph reshapeTensor:row_major
        withShape:core.output_shape name:nil];
    core.output = [core.graph castTensor:row_major
                                  toType:MPSDataTypeBFloat16 name:nil];
    cache[key] = core;
    return core;
}

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
                          char *error, size_t error_size) {
    uint64_t tensor_bytes = 0;
    uint64_t frequency_bytes = 0;
    uint64_t gate_bytes = 0;
    uint64_t inner_dim = (uint64_t)heads * head_dim;
    if (!ltx_required_bytes((uint64_t)rows * inner_dim, sizeof(uint16_t),
                            &tensor_bytes) ||
        !ltx_required_bytes(
            (uint64_t)heads * rows * (head_dim / 2u), sizeof(uint16_t),
            &frequency_bytes) ||
        !ltx_required_bytes((uint64_t)rows * heads, sizeof(uint16_t),
                            &gate_bytes) ||
        !gpu || !rows || !heads || !head_dim || head_dim % 2u ||
        !(scale > 0.0f) || !isfinite(scale) ||
        !ltx_buffer_fits(query, tensor_bytes) ||
        !ltx_buffer_fits(key, tensor_bytes) ||
        !ltx_buffer_fits(value, tensor_bytes) ||
        !ltx_buffer_fits(cosine, frequency_bytes) ||
        !ltx_buffer_fits(sine, frequency_bytes) ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        (gate && !ltx_buffer_fits(gate, gate_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS BF16 self-attention arguments");
    @autoreleasepool {
        LTXSelfAttentionCoreGraph *core = ltx_self_attention_core_graph(
            gpu, rows, heads, head_dim, scale, gate != NULL);
        if (!core)
            return ltx_gpu_fail(error, error_size,
                                "create MPS BF16 self-attention graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS BF16 self-attention command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
#define LTX_ATTN_FEED(TENSOR, BUFFER, SHAPE) \
        feeds[(TENSOR)] = [[MPSGraphTensorData alloc] \
            initWithMTLBuffer:ltx_buffer((BUFFER)) shape:(SHAPE) \
                     dataType:MPSDataTypeBFloat16]
        LTX_ATTN_FEED(core.query, query, core.input_shape);
        LTX_ATTN_FEED(core.key, key, core.input_shape);
        LTX_ATTN_FEED(core.value, value, core.input_shape);
        LTX_ATTN_FEED(core.cosine, cosine, core.frequency_shape);
        LTX_ATTN_FEED(core.sine, sine, core.frequency_shape);
        if (gate) LTX_ATTN_FEED(core.gate, gate, core.gate_shape);
#undef LTX_ATTN_FEED
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(output) shape:core.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [core.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{core.output: output_data}
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message),
                     "MPSGraph BF16 self-attention: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(
            gpu, command, "MPSGraph BF16 self-attention", error, error_size);
    }
}

static LTXInt8SelfAttentionGraph *ltx_int8_self_attention_graph(
        ltx_gpu *gpu, uint32_t rows, uint32_t heads,
        uint32_t head_dim, int query_has_bias,
        int key_has_bias, int value_has_bias,
        int gate_has_bias, int output_has_bias,
        float norm_epsilon) {
    NSMutableDictionary<NSString *, LTXInt8SelfAttentionGraph *> *cache =
        ltx_int8_self_attention_graphs(gpu);
    NSString *key = [NSString stringWithFormat:
        @"%u:%u:%u:%d:%d:%d:%d:%d:%.9g",
        rows, heads, head_dim, query_has_bias, key_has_bias,
        value_has_bias, gate_has_bias, output_has_bias, norm_epsilon];
    LTXInt8SelfAttentionGraph *cached = cache[key];
    if (cached) return cached;

    uint64_t inner_dim = (uint64_t)heads * head_dim;
    LTXInt8SelfAttentionGraph *attention =
        [[LTXInt8SelfAttentionGraph alloc] init];
    attention.graph = [[MPSGraph alloc] init];
    attention.input_shape = @[@1, @(rows), @(inner_dim)];
    attention.projection_weight_shape = @[@1, @(inner_dim), @(inner_dim)];
    attention.projection_scale_shape = @[@1, @(inner_dim), @1];
    attention.projection_bias_shape = @[@1, @1, @(inner_dim)];
    attention.norm_shape = @[@1, @1, @(inner_dim)];
    attention.gate_weight_shape = @[@1, @(heads), @(inner_dim)];
    attention.gate_bias_shape = @[@1, @1, @(heads)];
    attention.frequency_shape =
        @[@1, @(heads), @(rows), @(head_dim / 2u)];
    attention.output_shape = attention.input_shape;
    attention.input = [attention.graph
        placeholderWithShape:attention.input_shape
                  dataType:MPSDataTypeBFloat16 name:nil];
#define LTX_ATTN_PLACEHOLDER(PROPERTY, SHAPE, TYPE) \
    attention.PROPERTY = [attention.graph placeholderWithShape:(SHAPE) \
        dataType:(TYPE) name:nil]
    LTX_ATTN_PLACEHOLDER(query_weight, attention.projection_weight_shape,
                         MPSDataTypeInt8);
    LTX_ATTN_PLACEHOLDER(query_scale, attention.projection_scale_shape,
                         MPSDataTypeFloat32);
    LTX_ATTN_PLACEHOLDER(key_weight, attention.projection_weight_shape,
                         MPSDataTypeInt8);
    LTX_ATTN_PLACEHOLDER(key_scale, attention.projection_scale_shape,
                         MPSDataTypeFloat32);
    LTX_ATTN_PLACEHOLDER(value_weight, attention.projection_weight_shape,
                         MPSDataTypeInt8);
    LTX_ATTN_PLACEHOLDER(value_scale, attention.projection_scale_shape,
                         MPSDataTypeFloat32);
    LTX_ATTN_PLACEHOLDER(output_weight, attention.projection_weight_shape,
                         MPSDataTypeInt8);
    LTX_ATTN_PLACEHOLDER(output_scale, attention.projection_scale_shape,
                         MPSDataTypeFloat32);
    LTX_ATTN_PLACEHOLDER(query_norm_weight, attention.norm_shape,
                         MPSDataTypeBFloat16);
    LTX_ATTN_PLACEHOLDER(key_norm_weight, attention.norm_shape,
                         MPSDataTypeBFloat16);
    LTX_ATTN_PLACEHOLDER(gate_weight, attention.gate_weight_shape,
                         MPSDataTypeBFloat16);
    LTX_ATTN_PLACEHOLDER(cosine, attention.frequency_shape,
                         MPSDataTypeBFloat16);
    LTX_ATTN_PLACEHOLDER(sine, attention.frequency_shape,
                         MPSDataTypeBFloat16);
    if (query_has_bias)
        LTX_ATTN_PLACEHOLDER(query_bias, attention.projection_bias_shape,
                             MPSDataTypeBFloat16);
    if (key_has_bias)
        LTX_ATTN_PLACEHOLDER(key_bias, attention.projection_bias_shape,
                             MPSDataTypeBFloat16);
    if (value_has_bias)
        LTX_ATTN_PLACEHOLDER(value_bias, attention.projection_bias_shape,
                             MPSDataTypeBFloat16);
    if (gate_has_bias)
        LTX_ATTN_PLACEHOLDER(gate_bias, attention.gate_bias_shape,
                             MPSDataTypeBFloat16);
    if (output_has_bias)
        LTX_ATTN_PLACEHOLDER(output_bias, attention.projection_bias_shape,
                             MPSDataTypeBFloat16);
#undef LTX_ATTN_PLACEHOLDER

    MPSGraphTensor *hadamard = [attention.graph constantWithData:
        ltx_hadamard_256_bf16() shape:@[@256, @256]
                                dataType:MPSDataTypeBFloat16];
    MPSGraphTensor *rotated_input = ltx_graph_convrot_256(
        attention.graph, attention.input, rows, (uint32_t)inner_dim,
        hadamard);
    MPSGraphTensor *query = ltx_graph_int8_weight_linear(
        attention.graph, rotated_input,
        attention.query_weight, attention.query_scale,
        attention.query_bias);
    MPSGraphTensor *key_projection = ltx_graph_int8_weight_linear(
        attention.graph, rotated_input,
        attention.key_weight, attention.key_scale,
        attention.key_bias);
    MPSGraphTensor *value = ltx_graph_int8_weight_linear(
        attention.graph, rotated_input,
        attention.value_weight, attention.value_scale,
        attention.value_bias);
    query = ltx_graph_rms_norm_weighted_bf16(
        attention.graph, query, attention.query_norm_weight, norm_epsilon);
    key_projection = ltx_graph_rms_norm_weighted_bf16(
        attention.graph, key_projection,
        attention.key_norm_weight, norm_epsilon);

    query = ltx_graph_to_head_major(
        attention.graph, query, rows, heads, head_dim);
    key_projection = ltx_graph_to_head_major(
        attention.graph, key_projection, rows, heads, head_dim);
    value = ltx_graph_to_head_major(
        attention.graph, value, rows, heads, head_dim);
    query = ltx_graph_apply_rope_split(
        attention.graph, query, attention.cosine, attention.sine, head_dim);
    key_projection = ltx_graph_apply_rope_split(
        attention.graph, key_projection,
        attention.cosine, attention.sine, head_dim);
    MPSGraphTensor *head_output =
        [attention.graph scaledDotProductAttentionWithQueryTensor:query
            keyTensor:key_projection valueTensor:value
            scale:1.0f / sqrtf((float)head_dim) name:nil];

    MPSGraphTensor *gate_weight_transposed =
        [attention.graph transposeTensor:attention.gate_weight
                               dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *gate_logits =
        [attention.graph matrixMultiplicationWithPrimaryTensor:attention.input
            secondaryTensor:gate_weight_transposed name:nil];
    if (gate_has_bias)
        gate_logits = [attention.graph additionWithPrimaryTensor:gate_logits
            secondaryTensor:attention.gate_bias name:nil];
    MPSGraphTensor *gate =
        [attention.graph sigmoidWithTensor:gate_logits name:nil];
    MPSGraphTensor *two = [attention.graph constantWithScalar:2.0
        dataType:MPSDataTypeBFloat16];
    gate = [attention.graph multiplicationWithPrimaryTensor:gate
        secondaryTensor:two name:nil];
    gate = [attention.graph transposeTensor:gate
        permutation:@[@0, @2, @1] name:nil];
    gate = [attention.graph reshapeTensor:gate
        withShape:@[@1, @(heads), @(rows), @1] name:nil];
    head_output = [attention.graph multiplicationWithPrimaryTensor:head_output
        secondaryTensor:gate name:nil];
    MPSGraphTensor *row_major = [attention.graph transposeTensor:head_output
        permutation:@[@0, @2, @1, @3] name:nil];
    row_major = [attention.graph reshapeTensor:row_major
        withShape:attention.output_shape name:nil];
    MPSGraphTensor *rotated_output = ltx_graph_convrot_256(
        attention.graph, row_major, rows, (uint32_t)inner_dim, hadamard);
    attention.output = ltx_graph_int8_weight_linear(
        attention.graph, rotated_output,
        attention.output_weight, attention.output_scale,
        attention.output_bias);
    cache[key] = attention;
    return attention;
}

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
                          char *error, size_t error_size) {
    uint64_t inner_dim = (uint64_t)heads * head_dim;
    uint64_t tensor_bytes = 0;
    uint64_t projection_weight_bytes = 0;
    uint64_t projection_scale_bytes = 0;
    uint64_t projection_bias_bytes = 0;
    uint64_t gate_weight_bytes = 0;
    uint64_t gate_bias_bytes = 0;
    uint64_t frequency_bytes = 0;
    if (inner_dim > UINT32_MAX ||
        !ltx_required_bytes((uint64_t)rows * inner_dim, sizeof(uint16_t),
                            &tensor_bytes) ||
        !ltx_required_bytes(inner_dim * inner_dim, sizeof(int8_t),
                            &projection_weight_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(float),
                            &projection_scale_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(uint16_t),
                            &projection_bias_bytes) ||
        !ltx_required_bytes((uint64_t)heads * inner_dim, sizeof(uint16_t),
                            &gate_weight_bytes) ||
        !ltx_required_bytes(heads, sizeof(uint16_t), &gate_bias_bytes) ||
        !ltx_required_bytes(
            (uint64_t)heads * rows * (head_dim / 2u), sizeof(uint16_t),
            &frequency_bytes) ||
        !gpu || !rows || !heads || !head_dim || head_dim % 2u ||
        convrot_group_size != 256u || inner_dim % 256u ||
        !(norm_epsilon > 0.0f) ||
        !ltx_buffer_fits(input, tensor_bytes) ||
        !ltx_buffer_fits(output, tensor_bytes) ||
        !ltx_buffer_fits(query_weight, projection_weight_bytes) ||
        !ltx_buffer_fits(query_scale, projection_scale_bytes) ||
        !ltx_buffer_fits(key_weight, projection_weight_bytes) ||
        !ltx_buffer_fits(key_scale, projection_scale_bytes) ||
        !ltx_buffer_fits(value_weight, projection_weight_bytes) ||
        !ltx_buffer_fits(value_scale, projection_scale_bytes) ||
        !ltx_buffer_fits(output_weight, projection_weight_bytes) ||
        !ltx_buffer_fits(output_scale, projection_scale_bytes) ||
        !ltx_buffer_fits(query_norm_weight, projection_bias_bytes) ||
        !ltx_buffer_fits(key_norm_weight, projection_bias_bytes) ||
        !ltx_buffer_fits(gate_weight, gate_weight_bytes) ||
        !ltx_buffer_fits(cosine, frequency_bytes) ||
        !ltx_buffer_fits(sine, frequency_bytes) ||
        (query_bias && !ltx_buffer_fits(query_bias, projection_bias_bytes)) ||
        (key_bias && !ltx_buffer_fits(key_bias, projection_bias_bytes)) ||
        (value_bias && !ltx_buffer_fits(value_bias, projection_bias_bytes)) ||
        (gate_bias && !ltx_buffer_fits(gate_bias, gate_bias_bytes)) ||
        (output_bias && !ltx_buffer_fits(output_bias, projection_bias_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS INT8 self-attention arguments");

    @autoreleasepool {
        LTXInt8SelfAttentionGraph *attention =
            ltx_int8_self_attention_graph(
                gpu, rows, heads, head_dim,
                query_bias != NULL, key_bias != NULL,
                value_bias != NULL, gate_bias != NULL,
                output_bias != NULL, norm_epsilon);
        if (!attention)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 self-attention graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 self-attention command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
#define LTX_FULL_ATTN_FEED(TENSOR, BUFFER, SHAPE, TYPE) \
        feeds[(TENSOR)] = [[MPSGraphTensorData alloc] \
            initWithMTLBuffer:ltx_buffer((BUFFER)) shape:(SHAPE) \
                     dataType:(TYPE)]
        LTX_FULL_ATTN_FEED(attention.input, input, attention.input_shape,
                           MPSDataTypeBFloat16);
#define LTX_FULL_ATTN_PROJECTION(PREFIX, WEIGHT, SCALE, BIAS) \
        LTX_FULL_ATTN_FEED(attention.PREFIX##_weight, WEIGHT, \
                           attention.projection_weight_shape, MPSDataTypeInt8); \
        LTX_FULL_ATTN_FEED(attention.PREFIX##_scale, SCALE, \
                           attention.projection_scale_shape, MPSDataTypeFloat32); \
        if (BIAS) LTX_FULL_ATTN_FEED(attention.PREFIX##_bias, BIAS, \
                           attention.projection_bias_shape, MPSDataTypeBFloat16)
        LTX_FULL_ATTN_PROJECTION(query, query_weight, query_scale, query_bias);
        LTX_FULL_ATTN_PROJECTION(key, key_weight, key_scale, key_bias);
        LTX_FULL_ATTN_PROJECTION(value, value_weight, value_scale, value_bias);
        LTX_FULL_ATTN_PROJECTION(output, output_weight, output_scale,
                                 output_bias);
#undef LTX_FULL_ATTN_PROJECTION
        LTX_FULL_ATTN_FEED(attention.query_norm_weight, query_norm_weight,
                           attention.norm_shape, MPSDataTypeBFloat16);
        LTX_FULL_ATTN_FEED(attention.key_norm_weight, key_norm_weight,
                           attention.norm_shape, MPSDataTypeBFloat16);
        LTX_FULL_ATTN_FEED(attention.gate_weight, gate_weight,
                           attention.gate_weight_shape, MPSDataTypeBFloat16);
        if (gate_bias)
            LTX_FULL_ATTN_FEED(attention.gate_bias, gate_bias,
                               attention.gate_bias_shape,
                               MPSDataTypeBFloat16);
        LTX_FULL_ATTN_FEED(attention.cosine, cosine,
                           attention.frequency_shape,
                           MPSDataTypeBFloat16);
        LTX_FULL_ATTN_FEED(attention.sine, sine,
                           attention.frequency_shape,
                           MPSDataTypeBFloat16);
#undef LTX_FULL_ATTN_FEED
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(output) shape:attention.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [attention.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{attention.output: output_data}
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message),
                     "MPSGraph INT8 self-attention: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(
            gpu, command, "MPSGraph INT8 self-attention", error, error_size);
    }
}

static LTXInt8CrossKVGraph *ltx_int8_cross_kv_graph(
        ltx_gpu *gpu, uint32_t rows, uint32_t input_dim,
        uint32_t heads, uint32_t head_dim,
        int key_has_bias, int value_has_bias, float norm_epsilon) {
    NSMutableDictionary<NSString *, LTXInt8CrossKVGraph *> *cache =
        ltx_int8_cross_kv_graphs(gpu);
    NSString *cache_key = [NSString stringWithFormat:
        @"%u:%u:%u:%u:%d:%d:%.9g", rows, input_dim, heads, head_dim,
        key_has_bias, value_has_bias, norm_epsilon];
    LTXInt8CrossKVGraph *cached = cache[cache_key];
    if (cached) return cached;

    uint64_t inner_dim = (uint64_t)heads * head_dim;
    LTXInt8CrossKVGraph *kv = [[LTXInt8CrossKVGraph alloc] init];
    kv.graph = [[MPSGraph alloc] init];
    kv.input_shape = @[@1, @(rows), @(input_dim)];
    kv.weight_shape = @[@1, @(inner_dim), @(input_dim)];
    kv.scale_shape = @[@1, @(inner_dim), @1];
    kv.bias_shape = @[@1, @1, @(inner_dim)];
    kv.norm_shape = @[@1, @1, @(inner_dim)];
    kv.output_shape = @[@1, @(heads), @(rows), @(head_dim)];
#define LTX_KV_PLACEHOLDER(PROPERTY, SHAPE, TYPE) \
    kv.PROPERTY = [kv.graph placeholderWithShape:(SHAPE) \
        dataType:(TYPE) name:nil]
    LTX_KV_PLACEHOLDER(input, kv.input_shape, MPSDataTypeBFloat16);
    LTX_KV_PLACEHOLDER(key_weight, kv.weight_shape, MPSDataTypeInt8);
    LTX_KV_PLACEHOLDER(key_scale, kv.scale_shape, MPSDataTypeFloat32);
    LTX_KV_PLACEHOLDER(value_weight, kv.weight_shape, MPSDataTypeInt8);
    LTX_KV_PLACEHOLDER(value_scale, kv.scale_shape, MPSDataTypeFloat32);
    LTX_KV_PLACEHOLDER(key_norm_weight, kv.norm_shape,
                       MPSDataTypeBFloat16);
    if (key_has_bias)
        LTX_KV_PLACEHOLDER(key_bias, kv.bias_shape, MPSDataTypeBFloat16);
    if (value_has_bias)
        LTX_KV_PLACEHOLDER(value_bias, kv.bias_shape, MPSDataTypeBFloat16);
#undef LTX_KV_PLACEHOLDER

    MPSGraphTensor *hadamard = [kv.graph constantWithData:
        ltx_hadamard_256_bf16() shape:@[@256, @256]
                                dataType:MPSDataTypeBFloat16];
    MPSGraphTensor *rotated = ltx_graph_convrot_256(
        kv.graph, kv.input, rows, input_dim, hadamard);
    MPSGraphTensor *key = ltx_graph_int8_weight_linear(
        kv.graph, rotated, kv.key_weight, kv.key_scale, kv.key_bias);
    key = ltx_graph_rms_norm_weighted_bf16(
        kv.graph, key, kv.key_norm_weight, norm_epsilon);
    key = ltx_graph_to_head_major(kv.graph, key, rows, heads, head_dim);
    MPSGraphTensor *value = ltx_graph_int8_weight_linear(
        kv.graph, rotated, kv.value_weight, kv.value_scale, kv.value_bias);
    value = ltx_graph_to_head_major(
        kv.graph, value, rows, heads, head_dim);
    kv.key_output = [kv.graph castTensor:key
                                  toType:MPSDataTypeBFloat16 name:nil];
    kv.value_output = [kv.graph castTensor:value
                                    toType:MPSDataTypeBFloat16 name:nil];
    cache[cache_key] = kv;
    return kv;
}

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
                          char *error, size_t error_size) {
    uint64_t inner_dim = (uint64_t)heads * head_dim;
    uint64_t input_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t bias_bytes = 0;
    uint64_t output_bytes = 0;
    if (inner_dim > UINT32_MAX ||
        !ltx_required_bytes((uint64_t)key_value_rows * key_value_dim,
                            sizeof(uint16_t), &input_bytes) ||
        !ltx_required_bytes(inner_dim * key_value_dim, sizeof(int8_t),
                            &weight_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(float), &scale_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(uint16_t), &bias_bytes) ||
        !ltx_required_bytes((uint64_t)key_value_rows * inner_dim,
                            sizeof(uint16_t), &output_bytes) ||
        !gpu || !key_value_rows || !key_value_dim || !heads || !head_dim ||
        convrot_group_size != 256u || key_value_dim % 256u ||
        inner_dim % 256u || !(norm_epsilon > 0.0f) ||
        !ltx_buffer_fits(key_value_input, input_bytes) ||
        !ltx_buffer_fits(key_weight, weight_bytes) ||
        !ltx_buffer_fits(key_scale, scale_bytes) ||
        !ltx_buffer_fits(value_weight, weight_bytes) ||
        !ltx_buffer_fits(value_scale, scale_bytes) ||
        !ltx_buffer_fits(key_norm_weight, bias_bytes) ||
        !ltx_buffer_fits(key_output, output_bytes) ||
        !ltx_buffer_fits(value_output, output_bytes) ||
        (key_bias && !ltx_buffer_fits(key_bias, bias_bytes)) ||
        (value_bias && !ltx_buffer_fits(value_bias, bias_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS INT8 cross-attention K/V arguments");

    @autoreleasepool {
        LTXInt8CrossKVGraph *kv = ltx_int8_cross_kv_graph(
            gpu, key_value_rows, key_value_dim, heads, head_dim,
            key_bias != NULL, value_bias != NULL, norm_epsilon);
        if (!kv)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 cross K/V graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 cross K/V command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
#define LTX_KV_FEED(TENSOR, BUFFER, SHAPE, TYPE) \
        feeds[(TENSOR)] = [[MPSGraphTensorData alloc] \
            initWithMTLBuffer:ltx_buffer((BUFFER)) shape:(SHAPE) \
                     dataType:(TYPE)]
        LTX_KV_FEED(kv.input, key_value_input, kv.input_shape,
                    MPSDataTypeBFloat16);
        LTX_KV_FEED(kv.key_weight, key_weight, kv.weight_shape,
                    MPSDataTypeInt8);
        LTX_KV_FEED(kv.key_scale, key_scale, kv.scale_shape,
                    MPSDataTypeFloat32);
        LTX_KV_FEED(kv.value_weight, value_weight, kv.weight_shape,
                    MPSDataTypeInt8);
        LTX_KV_FEED(kv.value_scale, value_scale, kv.scale_shape,
                    MPSDataTypeFloat32);
        LTX_KV_FEED(kv.key_norm_weight, key_norm_weight, kv.norm_shape,
                    MPSDataTypeBFloat16);
        if (key_bias)
            LTX_KV_FEED(kv.key_bias, key_bias, kv.bias_shape,
                        MPSDataTypeBFloat16);
        if (value_bias)
            LTX_KV_FEED(kv.value_bias, value_bias, kv.bias_shape,
                        MPSDataTypeBFloat16);
#undef LTX_KV_FEED
        MPSGraphTensorData *key_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(key_output) shape:kv.output_shape
                     dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *value_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(value_output) shape:kv.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [kv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{
                    kv.key_output: key_data,
                    kv.value_output: value_data,
                }
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message), "MPSGraph INT8 cross K/V: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(
            gpu, command, "MPSGraph INT8 cross K/V", error, error_size);
    }
}

static LTXInt8CrossQueryGraph *ltx_int8_cross_query_graph(
        ltx_gpu *gpu, uint32_t query_rows, uint32_t key_value_rows,
        uint32_t query_dim, uint32_t heads, uint32_t head_dim,
        uint32_t output_dim, int query_has_bias, int gate_has_bias,
        int output_has_bias, uint32_t attention_mask_rows,
        float norm_epsilon) {
    NSMutableDictionary<NSString *, LTXInt8CrossQueryGraph *> *cache =
        ltx_int8_cross_query_graphs(gpu);
    NSString *cache_key = [NSString stringWithFormat:
        @"%u:%u:%u:%u:%u:%u:%d:%d:%d:%u:%.9g", query_rows,
        key_value_rows, query_dim, heads, head_dim, output_dim,
        query_has_bias, gate_has_bias, output_has_bias,
        attention_mask_rows, norm_epsilon];
    LTXInt8CrossQueryGraph *cached = cache[cache_key];
    if (cached) return cached;

    uint64_t inner_dim = (uint64_t)heads * head_dim;
    LTXInt8CrossQueryGraph *query_graph =
        [[LTXInt8CrossQueryGraph alloc] init];
    query_graph.graph = [[MPSGraph alloc] init];
    query_graph.query_input_shape = @[@1, @(query_rows), @(query_dim)];
    query_graph.key_value_shape =
        @[@1, @(heads), @(key_value_rows), @(head_dim)];
    query_graph.query_weight_shape = @[@1, @(inner_dim), @(query_dim)];
    query_graph.projection_scale_shape = @[@1, @(inner_dim), @1];
    query_graph.projection_bias_shape = @[@1, @1, @(inner_dim)];
    query_graph.norm_shape = @[@1, @1, @(inner_dim)];
    query_graph.gate_weight_shape = @[@1, @(heads), @(query_dim)];
    query_graph.gate_bias_shape = @[@1, @1, @(heads)];
    query_graph.output_weight_shape = @[@1, @(output_dim), @(inner_dim)];
    query_graph.output_scale_shape = @[@1, @(output_dim), @1];
    query_graph.output_bias_shape = @[@1, @1, @(output_dim)];
    if (attention_mask_rows)
        query_graph.attention_mask_shape =
            @[@1, @1, @(attention_mask_rows), @(key_value_rows)];
    query_graph.output_shape = @[@1, @(query_rows), @(output_dim)];
#define LTX_QUERY_PLACEHOLDER(PROPERTY, SHAPE, TYPE) \
    query_graph.PROPERTY = [query_graph.graph placeholderWithShape:(SHAPE) \
        dataType:(TYPE) name:nil]
    LTX_QUERY_PLACEHOLDER(query_input, query_graph.query_input_shape,
                          MPSDataTypeBFloat16);
    LTX_QUERY_PLACEHOLDER(key, query_graph.key_value_shape,
                          MPSDataTypeBFloat16);
    LTX_QUERY_PLACEHOLDER(value, query_graph.key_value_shape,
                          MPSDataTypeBFloat16);
    LTX_QUERY_PLACEHOLDER(query_weight, query_graph.query_weight_shape,
                          MPSDataTypeInt8);
    LTX_QUERY_PLACEHOLDER(query_scale, query_graph.projection_scale_shape,
                          MPSDataTypeFloat32);
    LTX_QUERY_PLACEHOLDER(query_norm_weight, query_graph.norm_shape,
                          MPSDataTypeBFloat16);
    LTX_QUERY_PLACEHOLDER(gate_weight, query_graph.gate_weight_shape,
                          MPSDataTypeBFloat16);
    LTX_QUERY_PLACEHOLDER(output_weight, query_graph.output_weight_shape,
                          MPSDataTypeInt8);
    LTX_QUERY_PLACEHOLDER(output_scale, query_graph.output_scale_shape,
                          MPSDataTypeFloat32);
    if (query_has_bias)
        LTX_QUERY_PLACEHOLDER(query_bias,
                              query_graph.projection_bias_shape,
                              MPSDataTypeBFloat16);
    if (gate_has_bias)
        LTX_QUERY_PLACEHOLDER(gate_bias, query_graph.gate_bias_shape,
                              MPSDataTypeBFloat16);
    if (output_has_bias)
        LTX_QUERY_PLACEHOLDER(output_bias, query_graph.output_bias_shape,
                              MPSDataTypeBFloat16);
    if (attention_mask_rows)
        LTX_QUERY_PLACEHOLDER(attention_mask,
                              query_graph.attention_mask_shape,
                              MPSDataTypeBFloat16);
#undef LTX_QUERY_PLACEHOLDER

    MPSGraphTensor *hadamard = [query_graph.graph constantWithData:
        ltx_hadamard_256_bf16() shape:@[@256, @256]
                                dataType:MPSDataTypeBFloat16];
    MPSGraphTensor *rotated_query = ltx_graph_convrot_256(
        query_graph.graph, query_graph.query_input,
        query_rows, query_dim, hadamard);
    MPSGraphTensor *query = ltx_graph_int8_weight_linear(
        query_graph.graph, rotated_query, query_graph.query_weight,
        query_graph.query_scale, query_graph.query_bias);
    query = ltx_graph_rms_norm_weighted_bf16(
        query_graph.graph, query, query_graph.query_norm_weight,
        norm_epsilon);
    query = ltx_graph_to_head_major(
        query_graph.graph, query, query_rows, heads, head_dim);
    MPSGraphTensor *head_output = attention_mask_rows ?
        [query_graph.graph scaledDotProductAttentionWithQueryTensor:query
            keyTensor:query_graph.key valueTensor:query_graph.value
            maskTensor:query_graph.attention_mask
            scale:1.0f / sqrtf((float)head_dim) name:nil] :
        [query_graph.graph scaledDotProductAttentionWithQueryTensor:query
            keyTensor:query_graph.key valueTensor:query_graph.value
            scale:1.0f / sqrtf((float)head_dim) name:nil];
    MPSGraphTensor *gate_weight = [query_graph.graph
        transposeTensor:query_graph.gate_weight
        dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *gate = [query_graph.graph
        matrixMultiplicationWithPrimaryTensor:query_graph.query_input
        secondaryTensor:gate_weight name:nil];
    if (gate_has_bias)
        gate = [query_graph.graph additionWithPrimaryTensor:gate
            secondaryTensor:query_graph.gate_bias name:nil];
    gate = [query_graph.graph sigmoidWithTensor:gate name:nil];
    MPSGraphTensor *two = [query_graph.graph constantWithScalar:2.0
        dataType:MPSDataTypeBFloat16];
    gate = [query_graph.graph multiplicationWithPrimaryTensor:gate
        secondaryTensor:two name:nil];
    gate = [query_graph.graph transposeTensor:gate
        permutation:@[@0, @2, @1] name:nil];
    gate = [query_graph.graph reshapeTensor:gate
        withShape:@[@1, @(heads), @(query_rows), @1] name:nil];
    head_output = [query_graph.graph
        multiplicationWithPrimaryTensor:head_output
        secondaryTensor:gate name:nil];
    MPSGraphTensor *row_major = [query_graph.graph
        transposeTensor:head_output permutation:@[@0, @2, @1, @3] name:nil];
    row_major = [query_graph.graph reshapeTensor:row_major
        withShape:@[@1, @(query_rows), @(inner_dim)] name:nil];
    MPSGraphTensor *rotated_output = ltx_graph_convrot_256(
        query_graph.graph, row_major, query_rows,
        (uint32_t)inner_dim, hadamard);
    query_graph.output = ltx_graph_int8_weight_linear(
        query_graph.graph, rotated_output, query_graph.output_weight,
        query_graph.output_scale, query_graph.output_bias);
    cache[cache_key] = query_graph;
    return query_graph;
}

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
                          char *error, size_t error_size) {
    uint64_t inner_dim = (uint64_t)heads * head_dim;
    uint64_t query_bytes = 0;
    uint64_t key_value_bytes = 0;
    uint64_t query_weight_bytes = 0;
    uint64_t projection_scale_bytes = 0;
    uint64_t projection_bias_bytes = 0;
    uint64_t gate_weight_bytes = 0;
    uint64_t gate_bias_bytes = 0;
    uint64_t output_weight_bytes = 0;
    uint64_t output_scale_bytes = 0;
    uint64_t output_bias_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t attention_mask_bytes = 0;
    if (inner_dim > UINT32_MAX ||
        !ltx_required_bytes((uint64_t)query_rows * query_dim,
                            sizeof(uint16_t), &query_bytes) ||
        !ltx_required_bytes((uint64_t)key_value_rows * inner_dim,
                            sizeof(uint16_t), &key_value_bytes) ||
        !ltx_required_bytes(inner_dim * query_dim, sizeof(int8_t),
                            &query_weight_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(float),
                            &projection_scale_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(uint16_t),
                            &projection_bias_bytes) ||
        !ltx_required_bytes((uint64_t)heads * query_dim, sizeof(uint16_t),
                            &gate_weight_bytes) ||
        !ltx_required_bytes(heads, sizeof(uint16_t), &gate_bias_bytes) ||
        !ltx_required_bytes((uint64_t)output_dim * inner_dim, sizeof(int8_t),
                            &output_weight_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(float), &output_scale_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(uint16_t),
                            &output_bias_bytes) ||
        !ltx_required_bytes((uint64_t)query_rows * output_dim,
                            sizeof(uint16_t), &output_bytes) ||
        !ltx_required_bytes((uint64_t)attention_mask_rows * key_value_rows,
                            sizeof(uint16_t), &attention_mask_bytes) ||
        !gpu || !query_rows || !key_value_rows || !query_dim || !heads ||
        !head_dim || head_dim % 2u || !output_dim ||
        convrot_group_size != 256u || query_dim % 256u ||
        inner_dim % 256u || !(norm_epsilon > 0.0f) ||
        !ltx_buffer_fits(query_input, query_bytes) ||
        !ltx_buffer_fits(key, key_value_bytes) ||
        !ltx_buffer_fits(value, key_value_bytes) ||
        !ltx_buffer_fits(query_weight, query_weight_bytes) ||
        !ltx_buffer_fits(query_scale, projection_scale_bytes) ||
        !ltx_buffer_fits(query_norm_weight, projection_bias_bytes) ||
        !ltx_buffer_fits(gate_weight, gate_weight_bytes) ||
        !ltx_buffer_fits(output_weight, output_weight_bytes) ||
        !ltx_buffer_fits(output_scale, output_scale_bytes) ||
        !ltx_buffer_fits(output, output_bytes) ||
        (query_bias && !ltx_buffer_fits(query_bias,
                                        projection_bias_bytes)) ||
        (gate_bias && !ltx_buffer_fits(gate_bias, gate_bias_bytes)) ||
        (output_bias && !ltx_buffer_fits(output_bias,
                                         output_bias_bytes)) ||
        ((attention_mask_rows != 0u) != (attention_mask != NULL)) ||
        (attention_mask_rows && attention_mask_rows != 1u &&
         attention_mask_rows != query_rows) ||
        (attention_mask &&
         !ltx_buffer_fits(attention_mask, attention_mask_bytes)))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS INT8 cross query arguments");

    @autoreleasepool {
        LTXInt8CrossQueryGraph *query_graph = ltx_int8_cross_query_graph(
            gpu, query_rows, key_value_rows, query_dim, heads, head_dim,
            output_dim, query_bias != NULL, gate_bias != NULL,
            output_bias != NULL, attention_mask_rows, norm_epsilon);
        if (!query_graph)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 cross query graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 cross query command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
#define LTX_QUERY_FEED(TENSOR, BUFFER, SHAPE, TYPE) \
        feeds[(TENSOR)] = [[MPSGraphTensorData alloc] \
            initWithMTLBuffer:ltx_buffer((BUFFER)) shape:(SHAPE) \
                     dataType:(TYPE)]
        LTX_QUERY_FEED(query_graph.query_input, query_input,
                       query_graph.query_input_shape, MPSDataTypeBFloat16);
        LTX_QUERY_FEED(query_graph.key, key, query_graph.key_value_shape,
                       MPSDataTypeBFloat16);
        LTX_QUERY_FEED(query_graph.value, value, query_graph.key_value_shape,
                       MPSDataTypeBFloat16);
        LTX_QUERY_FEED(query_graph.query_weight, query_weight,
                       query_graph.query_weight_shape, MPSDataTypeInt8);
        LTX_QUERY_FEED(query_graph.query_scale, query_scale,
                       query_graph.projection_scale_shape,
                       MPSDataTypeFloat32);
        LTX_QUERY_FEED(query_graph.query_norm_weight, query_norm_weight,
                       query_graph.norm_shape, MPSDataTypeBFloat16);
        LTX_QUERY_FEED(query_graph.gate_weight, gate_weight,
                       query_graph.gate_weight_shape, MPSDataTypeBFloat16);
        LTX_QUERY_FEED(query_graph.output_weight, output_weight,
                       query_graph.output_weight_shape, MPSDataTypeInt8);
        LTX_QUERY_FEED(query_graph.output_scale, output_scale,
                       query_graph.output_scale_shape, MPSDataTypeFloat32);
        if (query_bias)
            LTX_QUERY_FEED(query_graph.query_bias, query_bias,
                           query_graph.projection_bias_shape,
                           MPSDataTypeBFloat16);
        if (gate_bias)
            LTX_QUERY_FEED(query_graph.gate_bias, gate_bias,
                           query_graph.gate_bias_shape,
                           MPSDataTypeBFloat16);
        if (output_bias)
            LTX_QUERY_FEED(query_graph.output_bias, output_bias,
                           query_graph.output_bias_shape,
                           MPSDataTypeBFloat16);
        if (attention_mask)
            LTX_QUERY_FEED(query_graph.attention_mask, attention_mask,
                           query_graph.attention_mask_shape,
                           MPSDataTypeBFloat16);
#undef LTX_QUERY_FEED
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(output) shape:query_graph.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [query_graph.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{query_graph.output: output_data}
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message),
                     "MPSGraph INT8 cross query: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(
            gpu, command, "MPSGraph INT8 cross query", error, error_size);
    }
}

static LTXInt8CrossAttentionGraph *ltx_int8_cross_attention_graph(
        ltx_gpu *gpu,
        uint32_t query_rows, uint32_t key_value_rows,
        uint32_t query_dim, uint32_t key_value_dim,
        uint32_t heads, uint32_t head_dim, uint32_t output_dim,
        int query_has_bias, int key_has_bias, int value_has_bias,
        int gate_has_bias, int output_has_bias, int use_rope,
        uint32_t attention_mask_rows,
        float norm_epsilon) {
    NSMutableDictionary<NSString *, LTXInt8CrossAttentionGraph *> *cache =
        ltx_int8_cross_attention_graphs(gpu);
    NSString *cache_key = [NSString stringWithFormat:
        @"%u:%u:%u:%u:%u:%u:%u:%d:%d:%d:%d:%d:%d:%u:%.9g",
        query_rows, key_value_rows, query_dim, key_value_dim,
        heads, head_dim, output_dim, query_has_bias, key_has_bias,
        value_has_bias, gate_has_bias, output_has_bias, use_rope,
        attention_mask_rows, norm_epsilon];
    LTXInt8CrossAttentionGraph *cached = cache[cache_key];
    if (cached) return cached;

    uint64_t inner_dim = (uint64_t)heads * head_dim;
    LTXInt8CrossAttentionGraph *cross =
        [[LTXInt8CrossAttentionGraph alloc] init];
    cross.graph = [[MPSGraph alloc] init];
    cross.query_input_shape = @[@1, @(query_rows), @(query_dim)];
    cross.key_value_input_shape =
        @[@1, @(key_value_rows), @(key_value_dim)];
    cross.query_weight_shape = @[@1, @(inner_dim), @(query_dim)];
    cross.key_value_weight_shape =
        @[@1, @(inner_dim), @(key_value_dim)];
    cross.projection_scale_shape = @[@1, @(inner_dim), @1];
    cross.projection_bias_shape = @[@1, @1, @(inner_dim)];
    cross.norm_shape = @[@1, @1, @(inner_dim)];
    cross.gate_weight_shape = @[@1, @(heads), @(query_dim)];
    cross.gate_bias_shape = @[@1, @1, @(heads)];
    cross.output_weight_shape = @[@1, @(output_dim), @(inner_dim)];
    cross.output_scale_shape = @[@1, @(output_dim), @1];
    cross.output_bias_shape = @[@1, @1, @(output_dim)];
    cross.query_frequency_shape =
        @[@1, @(heads), @(query_rows), @(head_dim / 2u)];
    cross.key_frequency_shape =
        @[@1, @(heads), @(key_value_rows), @(head_dim / 2u)];
    if (attention_mask_rows)
        cross.attention_mask_shape =
            @[@1, @1, @(attention_mask_rows), @(key_value_rows)];
    cross.output_shape = @[@1, @(query_rows), @(output_dim)];

#define LTX_CROSS_PLACEHOLDER(PROPERTY, SHAPE, TYPE) \
    cross.PROPERTY = [cross.graph placeholderWithShape:(SHAPE) \
        dataType:(TYPE) name:nil]
    LTX_CROSS_PLACEHOLDER(query_input, cross.query_input_shape,
                          MPSDataTypeBFloat16);
    LTX_CROSS_PLACEHOLDER(key_value_input, cross.key_value_input_shape,
                          MPSDataTypeBFloat16);
    LTX_CROSS_PLACEHOLDER(query_weight, cross.query_weight_shape,
                          MPSDataTypeInt8);
    LTX_CROSS_PLACEHOLDER(query_scale, cross.projection_scale_shape,
                          MPSDataTypeFloat32);
    LTX_CROSS_PLACEHOLDER(key_weight, cross.key_value_weight_shape,
                          MPSDataTypeInt8);
    LTX_CROSS_PLACEHOLDER(key_scale, cross.projection_scale_shape,
                          MPSDataTypeFloat32);
    LTX_CROSS_PLACEHOLDER(value_weight, cross.key_value_weight_shape,
                          MPSDataTypeInt8);
    LTX_CROSS_PLACEHOLDER(value_scale, cross.projection_scale_shape,
                          MPSDataTypeFloat32);
    LTX_CROSS_PLACEHOLDER(query_norm_weight, cross.norm_shape,
                          MPSDataTypeBFloat16);
    LTX_CROSS_PLACEHOLDER(key_norm_weight, cross.norm_shape,
                          MPSDataTypeBFloat16);
    LTX_CROSS_PLACEHOLDER(gate_weight, cross.gate_weight_shape,
                          MPSDataTypeBFloat16);
    LTX_CROSS_PLACEHOLDER(output_weight, cross.output_weight_shape,
                          MPSDataTypeInt8);
    LTX_CROSS_PLACEHOLDER(output_scale, cross.output_scale_shape,
                          MPSDataTypeFloat32);
    if (query_has_bias)
        LTX_CROSS_PLACEHOLDER(query_bias, cross.projection_bias_shape,
                              MPSDataTypeBFloat16);
    if (key_has_bias)
        LTX_CROSS_PLACEHOLDER(key_bias, cross.projection_bias_shape,
                              MPSDataTypeBFloat16);
    if (value_has_bias)
        LTX_CROSS_PLACEHOLDER(value_bias, cross.projection_bias_shape,
                              MPSDataTypeBFloat16);
    if (gate_has_bias)
        LTX_CROSS_PLACEHOLDER(gate_bias, cross.gate_bias_shape,
                              MPSDataTypeBFloat16);
    if (output_has_bias)
        LTX_CROSS_PLACEHOLDER(output_bias, cross.output_bias_shape,
                              MPSDataTypeBFloat16);
    if (use_rope) {
        LTX_CROSS_PLACEHOLDER(query_cosine, cross.query_frequency_shape,
                              MPSDataTypeBFloat16);
        LTX_CROSS_PLACEHOLDER(query_sine, cross.query_frequency_shape,
                              MPSDataTypeBFloat16);
        LTX_CROSS_PLACEHOLDER(key_cosine, cross.key_frequency_shape,
                              MPSDataTypeBFloat16);
        LTX_CROSS_PLACEHOLDER(key_sine, cross.key_frequency_shape,
                              MPSDataTypeBFloat16);
    }
    if (attention_mask_rows)
        LTX_CROSS_PLACEHOLDER(attention_mask, cross.attention_mask_shape,
                              MPSDataTypeBFloat16);
#undef LTX_CROSS_PLACEHOLDER

    MPSGraphTensor *hadamard = [cross.graph constantWithData:
        ltx_hadamard_256_bf16() shape:@[@256, @256]
                                dataType:MPSDataTypeBFloat16];
    MPSGraphTensor *rotated_query = ltx_graph_convrot_256(
        cross.graph, cross.query_input, query_rows, query_dim, hadamard);
    MPSGraphTensor *rotated_key_value = ltx_graph_convrot_256(
        cross.graph, cross.key_value_input, key_value_rows,
        key_value_dim, hadamard);
    MPSGraphTensor *query = ltx_graph_int8_weight_linear(
        cross.graph, rotated_query,
        cross.query_weight, cross.query_scale, cross.query_bias);
    MPSGraphTensor *key_projection = ltx_graph_int8_weight_linear(
        cross.graph, rotated_key_value,
        cross.key_weight, cross.key_scale, cross.key_bias);
    MPSGraphTensor *value = ltx_graph_int8_weight_linear(
        cross.graph, rotated_key_value,
        cross.value_weight, cross.value_scale, cross.value_bias);
    query = ltx_graph_rms_norm_weighted_bf16(
        cross.graph, query, cross.query_norm_weight, norm_epsilon);
    key_projection = ltx_graph_rms_norm_weighted_bf16(
        cross.graph, key_projection, cross.key_norm_weight, norm_epsilon);
    query = ltx_graph_to_head_major(
        cross.graph, query, query_rows, heads, head_dim);
    key_projection = ltx_graph_to_head_major(
        cross.graph, key_projection, key_value_rows, heads, head_dim);
    value = ltx_graph_to_head_major(
        cross.graph, value, key_value_rows, heads, head_dim);
    if (use_rope) {
        query = ltx_graph_apply_rope_split(
            cross.graph, query, cross.query_cosine,
            cross.query_sine, head_dim);
        key_projection = ltx_graph_apply_rope_split(
            cross.graph, key_projection, cross.key_cosine,
            cross.key_sine, head_dim);
    }
    MPSGraphTensor *head_output = attention_mask_rows ?
        [cross.graph scaledDotProductAttentionWithQueryTensor:query
            keyTensor:key_projection valueTensor:value
            maskTensor:cross.attention_mask
            scale:1.0f / sqrtf((float)head_dim) name:nil] :
        [cross.graph scaledDotProductAttentionWithQueryTensor:query
            keyTensor:key_projection valueTensor:value
            scale:1.0f / sqrtf((float)head_dim) name:nil];

    MPSGraphTensor *gate_weight_transposed =
        [cross.graph transposeTensor:cross.gate_weight
                           dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *gate_logits =
        [cross.graph matrixMultiplicationWithPrimaryTensor:cross.query_input
            secondaryTensor:gate_weight_transposed name:nil];
    if (gate_has_bias)
        gate_logits = [cross.graph additionWithPrimaryTensor:gate_logits
            secondaryTensor:cross.gate_bias name:nil];
    MPSGraphTensor *gate =
        [cross.graph sigmoidWithTensor:gate_logits name:nil];
    MPSGraphTensor *two = [cross.graph constantWithScalar:2.0
        dataType:MPSDataTypeBFloat16];
    gate = [cross.graph multiplicationWithPrimaryTensor:gate
        secondaryTensor:two name:nil];
    gate = [cross.graph transposeTensor:gate
        permutation:@[@0, @2, @1] name:nil];
    gate = [cross.graph reshapeTensor:gate
        withShape:@[@1, @(heads), @(query_rows), @1] name:nil];
    head_output = [cross.graph multiplicationWithPrimaryTensor:head_output
        secondaryTensor:gate name:nil];
    MPSGraphTensor *row_major = [cross.graph transposeTensor:head_output
        permutation:@[@0, @2, @1, @3] name:nil];
    row_major = [cross.graph reshapeTensor:row_major
        withShape:@[@1, @(query_rows), @(inner_dim)] name:nil];
    MPSGraphTensor *rotated_output = ltx_graph_convrot_256(
        cross.graph, row_major, query_rows, (uint32_t)inner_dim, hadamard);
    cross.output = ltx_graph_int8_weight_linear(
        cross.graph, rotated_output,
        cross.output_weight, cross.output_scale, cross.output_bias);
    cache[cache_key] = cross;
    return cross;
}

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
                          char *error, size_t error_size) {
    uint64_t inner_dim = (uint64_t)heads * head_dim;
    uint64_t query_input_bytes = 0;
    uint64_t key_value_input_bytes = 0;
    uint64_t query_weight_bytes = 0;
    uint64_t key_value_weight_bytes = 0;
    uint64_t projection_scale_bytes = 0;
    uint64_t projection_bias_bytes = 0;
    uint64_t gate_weight_bytes = 0;
    uint64_t gate_bias_bytes = 0;
    uint64_t output_weight_bytes = 0;
    uint64_t output_scale_bytes = 0;
    uint64_t output_bias_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t query_frequency_bytes = 0;
    uint64_t key_frequency_bytes = 0;
    uint64_t attention_mask_bytes = 0;
    int use_rope = query_cosine || query_sine || key_cosine || key_sine;
    if (inner_dim > UINT32_MAX ||
        !ltx_required_bytes((uint64_t)query_rows * query_dim,
                            sizeof(uint16_t), &query_input_bytes) ||
        !ltx_required_bytes((uint64_t)key_value_rows * key_value_dim,
                            sizeof(uint16_t), &key_value_input_bytes) ||
        !ltx_required_bytes(inner_dim * query_dim, sizeof(int8_t),
                            &query_weight_bytes) ||
        !ltx_required_bytes(inner_dim * key_value_dim, sizeof(int8_t),
                            &key_value_weight_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(float),
                            &projection_scale_bytes) ||
        !ltx_required_bytes(inner_dim, sizeof(uint16_t),
                            &projection_bias_bytes) ||
        !ltx_required_bytes((uint64_t)heads * query_dim, sizeof(uint16_t),
                            &gate_weight_bytes) ||
        !ltx_required_bytes(heads, sizeof(uint16_t), &gate_bias_bytes) ||
        !ltx_required_bytes((uint64_t)output_dim * inner_dim, sizeof(int8_t),
                            &output_weight_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(float), &output_scale_bytes) ||
        !ltx_required_bytes(output_dim, sizeof(uint16_t), &output_bias_bytes) ||
        !ltx_required_bytes((uint64_t)query_rows * output_dim,
                            sizeof(uint16_t), &output_bytes) ||
        !ltx_required_bytes(
            (uint64_t)heads * query_rows * (head_dim / 2u),
            sizeof(uint16_t), &query_frequency_bytes) ||
        !ltx_required_bytes(
            (uint64_t)heads * key_value_rows * (head_dim / 2u),
            sizeof(uint16_t), &key_frequency_bytes) ||
        !ltx_required_bytes(
            (uint64_t)attention_mask_rows * key_value_rows,
            sizeof(uint16_t), &attention_mask_bytes) ||
        !gpu || !query_rows || !key_value_rows || !query_dim ||
        !key_value_dim || !heads || !head_dim || head_dim % 2u ||
        !output_dim || convrot_group_size != 256u ||
        query_dim % 256u || key_value_dim % 256u || inner_dim % 256u ||
        !(norm_epsilon > 0.0f) ||
        !ltx_buffer_fits(query_input, query_input_bytes) ||
        !ltx_buffer_fits(key_value_input, key_value_input_bytes) ||
        !ltx_buffer_fits(query_weight, query_weight_bytes) ||
        !ltx_buffer_fits(query_scale, projection_scale_bytes) ||
        !ltx_buffer_fits(key_weight, key_value_weight_bytes) ||
        !ltx_buffer_fits(key_scale, projection_scale_bytes) ||
        !ltx_buffer_fits(value_weight, key_value_weight_bytes) ||
        !ltx_buffer_fits(value_scale, projection_scale_bytes) ||
        !ltx_buffer_fits(query_norm_weight, projection_bias_bytes) ||
        !ltx_buffer_fits(key_norm_weight, projection_bias_bytes) ||
        !ltx_buffer_fits(gate_weight, gate_weight_bytes) ||
        !ltx_buffer_fits(output_weight, output_weight_bytes) ||
        !ltx_buffer_fits(output_scale, output_scale_bytes) ||
        !ltx_buffer_fits(output, output_bytes) ||
        (query_bias && !ltx_buffer_fits(query_bias, projection_bias_bytes)) ||
        (key_bias && !ltx_buffer_fits(key_bias, projection_bias_bytes)) ||
        (value_bias && !ltx_buffer_fits(value_bias, projection_bias_bytes)) ||
        (gate_bias && !ltx_buffer_fits(gate_bias, gate_bias_bytes)) ||
        (output_bias && !ltx_buffer_fits(output_bias, output_bias_bytes)) ||
        ((attention_mask_rows != 0u) != (attention_mask != NULL)) ||
        (attention_mask_rows && attention_mask_rows != 1u &&
         attention_mask_rows != query_rows) ||
        (attention_mask &&
         !ltx_buffer_fits(attention_mask, attention_mask_bytes)) ||
        (use_rope &&
         (!query_cosine || !query_sine || !key_cosine || !key_sine ||
          !ltx_buffer_fits(query_cosine, query_frequency_bytes) ||
          !ltx_buffer_fits(query_sine, query_frequency_bytes) ||
          !ltx_buffer_fits(key_cosine, key_frequency_bytes) ||
          !ltx_buffer_fits(key_sine, key_frequency_bytes))))
        return ltx_gpu_fail(error, error_size,
                            "invalid MPS INT8 cross-attention arguments");

    @autoreleasepool {
        LTXInt8CrossAttentionGraph *cross =
            ltx_int8_cross_attention_graph(
                gpu, query_rows, key_value_rows,
                query_dim, key_value_dim, heads, head_dim, output_dim,
                query_bias != NULL, key_bias != NULL, value_bias != NULL,
                gate_bias != NULL, output_bias != NULL, use_rope,
                attention_mask_rows,
                norm_epsilon);
        if (!cross)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 cross-attention graph failed");
        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:ltx_queue(gpu)];
        if (!command)
            return ltx_gpu_fail(error, error_size,
                                "create MPS INT8 cross-attention command failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
#define LTX_CROSS_FEED(TENSOR, BUFFER, SHAPE, TYPE) \
        feeds[(TENSOR)] = [[MPSGraphTensorData alloc] \
            initWithMTLBuffer:ltx_buffer((BUFFER)) shape:(SHAPE) \
                     dataType:(TYPE)]
        LTX_CROSS_FEED(cross.query_input, query_input,
                       cross.query_input_shape, MPSDataTypeBFloat16);
        LTX_CROSS_FEED(cross.key_value_input, key_value_input,
                       cross.key_value_input_shape, MPSDataTypeBFloat16);
        LTX_CROSS_FEED(cross.query_weight, query_weight,
                       cross.query_weight_shape, MPSDataTypeInt8);
        LTX_CROSS_FEED(cross.query_scale, query_scale,
                       cross.projection_scale_shape, MPSDataTypeFloat32);
        LTX_CROSS_FEED(cross.key_weight, key_weight,
                       cross.key_value_weight_shape, MPSDataTypeInt8);
        LTX_CROSS_FEED(cross.key_scale, key_scale,
                       cross.projection_scale_shape, MPSDataTypeFloat32);
        LTX_CROSS_FEED(cross.value_weight, value_weight,
                       cross.key_value_weight_shape, MPSDataTypeInt8);
        LTX_CROSS_FEED(cross.value_scale, value_scale,
                       cross.projection_scale_shape, MPSDataTypeFloat32);
        LTX_CROSS_FEED(cross.query_norm_weight, query_norm_weight,
                       cross.norm_shape, MPSDataTypeBFloat16);
        LTX_CROSS_FEED(cross.key_norm_weight, key_norm_weight,
                       cross.norm_shape, MPSDataTypeBFloat16);
        LTX_CROSS_FEED(cross.gate_weight, gate_weight,
                       cross.gate_weight_shape, MPSDataTypeBFloat16);
        LTX_CROSS_FEED(cross.output_weight, output_weight,
                       cross.output_weight_shape, MPSDataTypeInt8);
        LTX_CROSS_FEED(cross.output_scale, output_scale,
                       cross.output_scale_shape, MPSDataTypeFloat32);
        if (query_bias)
            LTX_CROSS_FEED(cross.query_bias, query_bias,
                           cross.projection_bias_shape, MPSDataTypeBFloat16);
        if (key_bias)
            LTX_CROSS_FEED(cross.key_bias, key_bias,
                           cross.projection_bias_shape, MPSDataTypeBFloat16);
        if (value_bias)
            LTX_CROSS_FEED(cross.value_bias, value_bias,
                           cross.projection_bias_shape, MPSDataTypeBFloat16);
        if (gate_bias)
            LTX_CROSS_FEED(cross.gate_bias, gate_bias,
                           cross.gate_bias_shape, MPSDataTypeBFloat16);
        if (output_bias)
            LTX_CROSS_FEED(cross.output_bias, output_bias,
                           cross.output_bias_shape, MPSDataTypeBFloat16);
        if (use_rope) {
            LTX_CROSS_FEED(cross.query_cosine, query_cosine,
                           cross.query_frequency_shape, MPSDataTypeBFloat16);
            LTX_CROSS_FEED(cross.query_sine, query_sine,
                           cross.query_frequency_shape, MPSDataTypeBFloat16);
            LTX_CROSS_FEED(cross.key_cosine, key_cosine,
                           cross.key_frequency_shape, MPSDataTypeBFloat16);
            LTX_CROSS_FEED(cross.key_sine, key_sine,
                           cross.key_frequency_shape, MPSDataTypeBFloat16);
        }
        if (attention_mask)
            LTX_CROSS_FEED(cross.attention_mask, attention_mask,
                           cross.attention_mask_shape,
                           MPSDataTypeBFloat16);
#undef LTX_CROSS_FEED
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:ltx_buffer(output) shape:cross.output_shape
                     dataType:MPSDataTypeBFloat16];
        @try {
            [cross.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{cross.output: output_data}
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            char message[1024];
            const char *reason = exception.reason.UTF8String;
            snprintf(message, sizeof(message),
                     "MPSGraph INT8 cross-attention: %s",
                     reason ? reason : "unknown exception");
            return ltx_gpu_fail(error, error_size, message);
        }
        return ltx_finish_mps_command(
            gpu, command, "MPSGraph INT8 cross-attention", error, error_size);
    }
}

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
                          char *error, size_t error_size) {
    return ltx_gpu_cross_attention_int8_mps_bf16_masked(
        gpu, output, query_input, key_value_input,
        query_weight, query_scale, query_bias,
        key_weight, key_scale, key_bias,
        value_weight, value_scale, value_bias,
        query_norm_weight, key_norm_weight,
        gate_weight, gate_bias,
        output_weight, output_scale, output_bias,
        query_cosine, query_sine, key_cosine, key_sine,
        NULL, 0u,
        query_rows, key_value_rows, query_dim, key_value_dim,
        heads, head_dim, output_dim, convrot_group_size,
        norm_epsilon, error, error_size);
}
