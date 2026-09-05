#include "ltx_runtime_config.h"
#include "ltx_video_vae.h"

#include "ltx_gpu_internal.h"
#include "ltx_safetensors.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <dispatch/dispatch.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LTX_VAE_LATENT_CHANNELS = 128,
    LTX_VAE_OUTPUT_CHANNELS = 3,
    LTX_VAE_GRAPH_STAGES = 5,
};

@interface LTXVideoVAEGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@property(nonatomic, strong) NSMutableArray<NSString *> *weightNames;
@property(nonatomic, strong) NSMutableArray<MPSGraphTensor *> *weightTensors;
@property(nonatomic, strong)
    NSMutableDictionary<NSString *, MPSGraphTensor *> *weightByName;
@end

@implementation LTXVideoVAEGraph
@end

@interface LTXVideoVAEPlan : NSObject
@property(nonatomic, strong) NSArray<LTXVideoVAEGraph *> *stages;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *workspaceBuffers;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end

@implementation LTXVideoVAEPlan
@end

@interface LTXVideoVAEModel : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) NSArray<NSString *> *weightNames;
@property(nonatomic, strong) NSArray<NSArray<NSNumber *> *> *weightShapes;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *weightBuffers;
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *weightIndices;
@property(nonatomic, strong)
    NSMutableDictionary<NSString *, LTXVideoVAEPlan *> *plans;
@property(nonatomic) uint64_t weightBytes;
@end

@implementation LTXVideoVAEModel
@end

struct ltx_video_vae {
    void *model;
};

typedef struct {
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t residual_block_index;
    uint32_t residual_blocks;
    uint32_t upsample_block_index;
    uint32_t spatial_factor;
    uint32_t temporal_factor;
} ltx_vae_stage_spec;

static const ltx_vae_stage_spec ltx_vae_stages[4] = {
    {1024u, 512u, 0u, 2u, 1u, 2u, 2u},
    {512u, 512u, 2u, 2u, 3u, 2u, 2u},
    {512u, 256u, 4u, 4u, 5u, 1u, 2u},
    {256u, 128u, 6u, 6u, 7u, 2u, 1u},
};

static int ltx_vae_fail(char *error, size_t error_size,
                        const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static const char *ltx_vae_error_description(NSError *error) {
    if (!error) return "unknown error";
    const char *description = error.localizedDescription.UTF8String;
    return description ? description : "unknown error";
}

static int ltx_vae_mul_u64(uint64_t left, uint64_t right,
                           uint64_t *result) {
    if (!result || (right && left > UINT64_MAX / right)) return 0;
    *result = left * right;
    return 1;
}

static void ltx_vae_add_res_stage_names(NSMutableArray<NSString *> *names,
                                        uint32_t stage,
                                        uint32_t blocks) {
    for (uint32_t block = 0; block < blocks; block++) {
        for (NSString *convolution in @[@"conv1", @"conv2"]) {
            [names addObject:[NSString stringWithFormat:
                @"decoder.up_blocks.%u.res_blocks.%u.%@.conv.weight",
                stage, block, convolution]];
            [names addObject:[NSString stringWithFormat:
                @"decoder.up_blocks.%u.res_blocks.%u.%@.conv.bias",
                stage, block, convolution]];
        }
    }
}

static NSArray<NSString *> *ltx_vae_weight_names(void) {
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    [names addObject:@"per_channel_statistics.mean-of-means"];
    [names addObject:@"per_channel_statistics.std-of-means"];
    [names addObject:@"decoder.conv_in.conv.weight"];
    [names addObject:@"decoder.conv_in.conv.bias"];
    ltx_vae_add_res_stage_names(names, 0u, 2u);
    [names addObject:@"decoder.up_blocks.1.conv.conv.weight"];
    [names addObject:@"decoder.up_blocks.1.conv.conv.bias"];
    ltx_vae_add_res_stage_names(names, 2u, 2u);
    [names addObject:@"decoder.up_blocks.3.conv.conv.weight"];
    [names addObject:@"decoder.up_blocks.3.conv.conv.bias"];
    ltx_vae_add_res_stage_names(names, 4u, 4u);
    [names addObject:@"decoder.up_blocks.5.conv.conv.weight"];
    [names addObject:@"decoder.up_blocks.5.conv.conv.bias"];
    ltx_vae_add_res_stage_names(names, 6u, 6u);
    [names addObject:@"decoder.up_blocks.7.conv.conv.weight"];
    [names addObject:@"decoder.up_blocks.7.conv.conv.bias"];
    ltx_vae_add_res_stage_names(names, 8u, 4u);
    [names addObject:@"decoder.conv_out.conv.weight"];
    [names addObject:@"decoder.conv_out.conv.bias"];
    return names;
}

static int ltx_vae_expect(const ltx_st_header *header, const char *name,
                          uint32_t ndim, const uint64_t *shape,
                          char *error, size_t error_size) {
    const ltx_st_tensor *tensor = ltx_st_find(header, name);
    if (!tensor)
        return ltx_vae_fail(error, error_size,
                            "video VAE is missing tensor %s", name);
    if (tensor->dtype != LTX_DTYPE_BF16)
        return ltx_vae_fail(error, error_size,
                            "video VAE tensor %s must be BF16, got %s",
                            name, ltx_dtype_name(tensor->dtype));
    if (tensor->ndim != ndim)
        return ltx_vae_fail(error, error_size,
                            "video VAE tensor %s rank is %u, expected %u",
                            name, tensor->ndim, ndim);
    for (uint32_t index = 0; index < ndim; index++) {
        if (tensor->shape[index] != shape[index])
            return ltx_vae_fail(
                error, error_size,
                "video VAE tensor %s shape mismatch at axis %u: "
                "%llu != %llu",
                name, index,
                (unsigned long long)tensor->shape[index],
                (unsigned long long)shape[index]);
    }
    return 1;
}

static int ltx_vae_expect_conv(const ltx_st_header *header,
                               NSString *prefix,
                               uint32_t output_channels,
                               uint32_t input_channels,
                               char *error, size_t error_size) {
    NSString *weight = [prefix stringByAppendingString:@".weight"];
    NSString *bias = [prefix stringByAppendingString:@".bias"];
    const uint64_t weight_shape[] = {
        output_channels, input_channels, 3u, 3u, 3u};
    const uint64_t bias_shape[] = {output_channels};
    return ltx_vae_expect(header, weight.UTF8String, 5u, weight_shape,
                          error, error_size) &&
           ltx_vae_expect(header, bias.UTF8String, 1u, bias_shape,
                          error, error_size);
}

static int ltx_vae_validate_res_stage(const ltx_st_header *header,
                                      uint32_t stage,
                                      uint32_t channels,
                                      uint32_t blocks,
                                      char *error, size_t error_size) {
    for (uint32_t block = 0; block < blocks; block++) {
        for (uint32_t convolution = 1; convolution <= 2u; convolution++) {
            NSString *prefix = [NSString stringWithFormat:
                @"decoder.up_blocks.%u.res_blocks.%u.conv%u.conv",
                stage, block, convolution];
            if (!ltx_vae_expect_conv(header, prefix, channels, channels,
                                     error, error_size))
                return 0;
        }
    }
    return 1;
}

static int ltx_vae_validate(const ltx_st_header *header,
                            char *error, size_t error_size) {
    const uint64_t statistics_shape[] = {LTX_VAE_LATENT_CHANNELS};
    if (!ltx_vae_expect(header,
                        "per_channel_statistics.mean-of-means", 1u,
                        statistics_shape, error, error_size) ||
        !ltx_vae_expect(header,
                        "per_channel_statistics.std-of-means", 1u,
                        statistics_shape, error, error_size) ||
        !ltx_vae_expect_conv(header, @"decoder.conv_in.conv", 1024u,
                             LTX_VAE_LATENT_CHANNELS, error, error_size))
        return 0;

    for (uint32_t index = 0; index < 4u; index++) {
        const ltx_vae_stage_spec *stage = &ltx_vae_stages[index];
        if (!ltx_vae_validate_res_stage(
                header, stage->residual_block_index,
                stage->input_channels, stage->residual_blocks,
                error, error_size))
            return 0;
        NSString *prefix = [NSString stringWithFormat:
            @"decoder.up_blocks.%u.conv.conv", stage->upsample_block_index];
        uint64_t factor = (uint64_t)stage->spatial_factor *
                          stage->spatial_factor * stage->temporal_factor;
        uint64_t convolution_channels =
            (uint64_t)stage->output_channels * factor;
        if (convolution_channels > UINT32_MAX ||
            !ltx_vae_expect_conv(
                header, prefix, (uint32_t)convolution_channels,
                stage->input_channels, error, error_size))
            return 0;
    }

    return ltx_vae_validate_res_stage(
               header, 8u, 128u, 4u, error, error_size) &&
           ltx_vae_expect_conv(header, @"decoder.conv_out.conv", 48u,
                               128u, error, error_size);
}

static NSArray<NSNumber *> *ltx_vae_shape(const ltx_st_tensor *tensor) {
    NSMutableArray<NSNumber *> *shape =
        [NSMutableArray arrayWithCapacity:tensor->ndim];
    for (uint32_t index = 0; index < tensor->ndim; index++)
        [shape addObject:@(tensor->shape[index])];
    return shape;
}

static NSArray<NSNumber *> *ltx_vae_graph_weight_shape(
        const ltx_st_tensor *tensor) {
    if (tensor->ndim != 5u) return ltx_vae_shape(tensor);
    return @[@(tensor->shape[2]), @(tensor->shape[3]),
             @(tensor->shape[4]), @(tensor->shape[1]),
             @(tensor->shape[0])];
}

static void ltx_vae_copy_weight_dhwio(const ltx_st_tensor *tensor,
                                      const void *source,
                                      void *destination,
                                      size_t bytes) {
    if (tensor->ndim != 5u) {
        memcpy(destination, source, bytes);
        return;
    }
    const uint64_t output_channels = tensor->shape[0];
    const uint64_t input_channels = tensor->shape[1];
    const uint64_t depth = tensor->shape[2];
    const uint64_t height = tensor->shape[3];
    const uint64_t width = tensor->shape[4];
    const uint64_t source_columns =
        input_channels * depth * height * width;
    const uint64_t target_rows = source_columns;
    const uint16_t *input = source;
    uint16_t *output = destination;
    const uint64_t rows_per_task = 16u;
    size_t tasks = (size_t)((target_rows + rows_per_task - 1u) /
                            rows_per_task);
    dispatch_apply(tasks,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t task) {
        uint64_t begin = (uint64_t)task * rows_per_task;
        uint64_t end = begin + rows_per_task;
        if (end > target_rows) end = target_rows;
        for (uint64_t target = begin; target < end; target++) {
            uint64_t remainder = target;
            uint64_t input_channel = remainder % input_channels;
            remainder /= input_channels;
            uint64_t x = remainder % width;
            remainder /= width;
            uint64_t y = remainder % height;
            uint64_t z = remainder / height;
            uint64_t source_column =
                ((input_channel * depth + z) * height + y) * width + x;
            uint64_t target_base = target * output_channels;
            for (uint64_t output_channel = 0;
                 output_channel < output_channels; output_channel++) {
                output[target_base + output_channel] =
                    input[output_channel * source_columns + source_column];
            }
        }
    });
}

static NSUInteger ltx_vae_weight_index(LTXVideoVAEModel *model,
                                       NSString *name) {
    NSNumber *index = model.weightIndices[name];
    return index ? index.unsignedIntegerValue : NSNotFound;
}

static MPSGraphTensor *ltx_vae_weight(LTXVideoVAEModel *model,
                                     LTXVideoVAEGraph *state,
                                     NSString *name) {
    MPSGraphTensor *existing = state.weightByName[name];
    if (existing) return existing;
    NSUInteger index = ltx_vae_weight_index(model, name);
    if (index == NSNotFound || index >= model.weightShapes.count) return nil;
    MPSGraphTensor *tensor =
        [state.graph placeholderWithShape:model.weightShapes[index]
                                 dataType:MPSDataTypeBFloat16
                                     name:name];
    if (!tensor) return nil;
    state.weightByName[name] = tensor;
    [state.weightNames addObject:name];
    [state.weightTensors addObject:tensor];
    return tensor;
}

static MPSGraphTensor *ltx_vae_pixel_norm(LTXVideoVAEGraph *state,
                                          MPSGraphTensor *input,
                                          uint32_t channels) {
    MPSGraph *graph = state.graph;
    MPSGraphTensor *flat =
        [graph reshapeTensor:input
                   withShape:@[@(-1), @(channels)] name:nil];
    MPSGraphTensor *value =
        [graph castTensor:flat toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *squared = [graph squareWithTensor:value name:nil];
    MPSGraphTensor *mean =
        [graph meanOfTensor:squared axes:@[@1] name:nil];
    MPSGraphTensor *epsilon =
        [graph constantWithScalar:1.0e-8 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *inverse =
        [graph reciprocalSquareRootWithTensor:
            [graph additionWithPrimaryTensor:mean
                              secondaryTensor:epsilon name:nil]
                                         name:nil];
    MPSGraphTensor *normalized =
        [graph multiplicationWithPrimaryTensor:value
                                secondaryTensor:inverse name:nil];
    MPSGraphTensor *result =
        [graph castTensor:normalized
                   toType:MPSDataTypeBFloat16 name:nil];
    return [graph reshapeTensor:result withShape:input.shape name:nil];
}

static MPSGraphTensor *ltx_vae_silu(LTXVideoVAEGraph *state,
                                    MPSGraphTensor *input) {
    MPSGraph *graph = state.graph;
    MPSGraphTensor *value =
        [graph castTensor:input toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *sigmoid = [graph sigmoidWithTensor:value name:nil];
    MPSGraphTensor *result =
        [graph multiplicationWithPrimaryTensor:value
                                secondaryTensor:sigmoid name:nil];
    return [graph castTensor:result
                      toType:MPSDataTypeBFloat16 name:nil];
}

static MPSGraphTensor *ltx_vae_conv3d(LTXVideoVAEModel *model,
                                      LTXVideoVAEGraph *state,
                                      MPSGraphTensor *input,
                                      NSString *prefix,
                                      uint32_t frames,
                                      uint32_t output_channels) {
    MPSGraph *graph = state.graph;
    MPSGraphTensor *first =
        [graph sliceTensor:input dimension:1 start:0 length:1 name:nil];
    MPSGraphTensor *last =
        [graph sliceTensor:input dimension:1
                     start:(NSInteger)frames - 1 length:1 name:nil];
    MPSGraphTensor *padded =
        [graph concatTensors:@[first, input, last] dimension:1 name:nil];
    MPSGraphTensor *weight = ltx_vae_weight(
        model, state, [prefix stringByAppendingString:@".weight"]);
    MPSGraphTensor *bias = ltx_vae_weight(
        model, state, [prefix stringByAppendingString:@".bias"]);
    MPSGraphConvolution3DOpDescriptor *descriptor =
        [MPSGraphConvolution3DOpDescriptor
            descriptorWithStrideInX:1u strideInY:1u strideInZ:1u
            dilationRateInX:1u dilationRateInY:1u dilationRateInZ:1u
            groups:1u paddingLeft:1u paddingRight:1u
            paddingTop:1u paddingBottom:1u
            paddingFront:0u paddingBack:0u
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNDHWC
            weightsLayout:MPSGraphTensorNamedDataLayoutDHWIO];
    MPSGraphTensor *convolution =
        [graph convolution3DWithSourceTensor:padded
                              weightsTensor:weight
                                 descriptor:descriptor name:nil];
    MPSGraphTensor *reshaped_bias =
        [graph reshapeTensor:bias
                   withShape:@[@1, @1, @1, @1, @(output_channels)]
                        name:nil];
    return [graph additionWithPrimaryTensor:convolution
                             secondaryTensor:reshaped_bias name:nil];
}

static MPSGraphTensor *ltx_vae_res_block(LTXVideoVAEModel *model,
                                         LTXVideoVAEGraph *state,
                                         MPSGraphTensor *input,
                                         uint32_t stage,
                                         uint32_t block,
                                         uint32_t channels,
                                         uint32_t frames) {
    NSString *base = [NSString stringWithFormat:
        @"decoder.up_blocks.%u.res_blocks.%u", stage, block];
    MPSGraphTensor *value = ltx_vae_pixel_norm(state, input, channels);
    value = ltx_vae_silu(state, value);
    value = ltx_vae_conv3d(
        model, state, value,
        [base stringByAppendingString:@".conv1.conv"],
        frames, channels);
    value = ltx_vae_pixel_norm(state, value, channels);
    value = ltx_vae_silu(state, value);
    value = ltx_vae_conv3d(
        model, state, value,
        [base stringByAppendingString:@".conv2.conv"],
        frames, channels);
    return [state.graph additionWithPrimaryTensor:value
                                   secondaryTensor:input name:nil];
}

static MPSGraphTensor *ltx_vae_depth_to_space(
        LTXVideoVAEModel *model, LTXVideoVAEGraph *state,
        MPSGraphTensor *input, uint32_t block,
        uint32_t batch,
        uint32_t input_channels, uint32_t output_channels,
        uint32_t frames, uint32_t height, uint32_t width,
        uint32_t spatial_factor, uint32_t temporal_factor) {
    MPSGraph *graph = state.graph;
    uint64_t factor = (uint64_t)spatial_factor * spatial_factor *
                      temporal_factor;
    uint32_t convolution_channels = (uint32_t)(output_channels * factor);
    NSString *prefix = [NSString stringWithFormat:
        @"decoder.up_blocks.%u.conv.conv", block];
    MPSGraphTensor *value = ltx_vae_conv3d(
        model, state, input, prefix, frames, convolution_channels);
    value = [graph reshapeTensor:value
                      withShape:@[@(batch), @(frames), @(height),
                                  @(width), @(output_channels),
                                  @(temporal_factor), @(spatial_factor),
                                  @(spatial_factor)]
                           name:nil];
    value = [graph transposeTensor:value
                       permutation:@[@0, @1, @5, @2, @6, @3, @7, @4]
                              name:nil];
    uint32_t up_frames = frames * temporal_factor;
    value = [graph reshapeTensor:value
                      withShape:@[@(batch), @(up_frames),
                                  @((uint64_t)height * spatial_factor),
                                  @((uint64_t)width * spatial_factor),
                                  @(output_channels)]
                           name:nil];
    if (temporal_factor > 1u) {
        value = [graph sliceTensor:value dimension:1 start:1
                            length:(NSInteger)up_frames - 1 name:nil];
    }
    (void)input_channels;
    return value;
}

static MPSGraphTensor *ltx_vae_unpatchify(LTXVideoVAEGraph *state,
                                          MPSGraphTensor *input,
                                          uint32_t batch,
                                          uint32_t frames,
                                          uint32_t height,
                                          uint32_t width) {
    MPSGraph *graph = state.graph;
    MPSGraphTensor *value =
        [graph reshapeTensor:input
                   withShape:@[@(batch), @(frames), @(height), @(width),
                               @3, @4, @4]
                        name:nil];
    value = [graph transposeTensor:value
                       permutation:@[@0, @4, @1, @2, @6, @3, @5]
                              name:nil];
    return [graph reshapeTensor:value
                     withShape:@[@(batch), @3, @(frames),
                                 @((uint64_t)height * 4u),
                                 @((uint64_t)width * 4u)]
                          name:nil];
}

static LTXVideoVAEGraph *ltx_vae_build_stage(
        LTXVideoVAEModel *model, uint32_t stage_index,
        uint32_t batch, uint32_t frames, uint32_t height, uint32_t width,
        int token_layout) {
    LTXVideoVAEGraph *state = [[LTXVideoVAEGraph alloc] init];
    state.graph = [[MPSGraph alloc] init];
    state.weightNames = [NSMutableArray array];
    state.weightTensors = [NSMutableArray array];
    state.weightByName = [NSMutableDictionary dictionary];

    uint32_t input_channels = stage_index == 0u ?
        LTX_VAE_LATENT_CHANNELS : ltx_vae_stages[stage_index - 1u].output_channels;
    if (stage_index == 0u) {
        state.inputShape = token_layout ?
            @[@(batch), @(frames), @(height), @(width), @(input_channels)] :
            @[@(batch), @(input_channels), @(frames), @(height), @(width)];
    } else {
        state.inputShape =
            @[@(batch), @(frames), @(height), @(width), @(input_channels)];
    }
    state.input = [state.graph placeholderWithShape:state.inputShape
                                           dataType:MPSDataTypeBFloat16
                                               name:@"input"];
    MPSGraphTensor *value = stage_index == 0u && !token_layout ?
        [state.graph transposeTensor:state.input
                         permutation:@[@0, @2, @3, @4, @1] name:nil] :
        state.input;

    if (stage_index == 0u) {
        MPSGraphTensor *mean = ltx_vae_weight(
            model, state, @"per_channel_statistics.mean-of-means");
        MPSGraphTensor *standard_deviation = ltx_vae_weight(
            model, state, @"per_channel_statistics.std-of-means");
        mean = [state.graph reshapeTensor:mean
                               withShape:@[@1, @1, @1, @1, @128]
                                    name:nil];
        standard_deviation = [state.graph
            reshapeTensor:standard_deviation
                withShape:@[@1, @1, @1, @1, @128] name:nil];
        value = [state.graph multiplicationWithPrimaryTensor:value
                                             secondaryTensor:standard_deviation
                                                        name:nil];
        value = [state.graph additionWithPrimaryTensor:value
                                       secondaryTensor:mean name:nil];
        value = ltx_vae_conv3d(model, state, value,
                               @"decoder.conv_in.conv", frames, 1024u);
    }

    if (stage_index < 4u) {
        const ltx_vae_stage_spec *spec = &ltx_vae_stages[stage_index];
        for (uint32_t block = 0; block < spec->residual_blocks; block++) {
            value = ltx_vae_res_block(
                model, state, value, spec->residual_block_index, block,
                spec->input_channels, frames);
        }
        value = ltx_vae_depth_to_space(
            model, state, value, spec->upsample_block_index,
            batch,
            spec->input_channels, spec->output_channels,
            frames, height, width,
            spec->spatial_factor, spec->temporal_factor);
        uint32_t output_frames = frames * spec->temporal_factor -
                                 (spec->temporal_factor > 1u ? 1u : 0u);
        state.outputShape =
            @[@(batch), @(output_frames),
              @((uint64_t)height * spec->spatial_factor),
              @((uint64_t)width * spec->spatial_factor),
              @(spec->output_channels)];
    } else {
        for (uint32_t block = 0; block < 4u; block++) {
            value = ltx_vae_res_block(model, state, value, 8u, block,
                                      128u, frames);
        }
        value = ltx_vae_pixel_norm(state, value, 128u);
        value = ltx_vae_silu(state, value);
        value = ltx_vae_conv3d(model, state, value,
                               @"decoder.conv_out.conv", frames, 48u);
        value = ltx_vae_unpatchify(
            state, value, batch, frames, height, width);
        state.outputShape =
            @[@(batch), @3, @(frames), @((uint64_t)height * 4u),
              @((uint64_t)width * 4u)];
    }
    state.output = value;
    return state;
}

static LTXVideoVAEGraph *ltx_vae_build_full_graph(
        LTXVideoVAEModel *model, uint32_t batch,
        uint32_t frames, uint32_t height, uint32_t width,
        int token_layout) {
    LTXVideoVAEGraph *state = [[LTXVideoVAEGraph alloc] init];
    state.graph = [[MPSGraph alloc] init];
    state.weightNames = [NSMutableArray array];
    state.weightTensors = [NSMutableArray array];
    state.weightByName = [NSMutableDictionary dictionary];
    state.inputShape = token_layout ?
        @[@(batch), @(frames), @(height), @(width), @128] :
        @[@(batch), @128, @(frames), @(height), @(width)];
    state.input = [state.graph placeholderWithShape:state.inputShape
                                           dataType:MPSDataTypeBFloat16
                                               name:@"input"];
    MPSGraphTensor *value = token_layout ? state.input :
        [state.graph transposeTensor:state.input
                         permutation:@[@0, @2, @3, @4, @1] name:nil];

    MPSGraphTensor *mean = ltx_vae_weight(
        model, state, @"per_channel_statistics.mean-of-means");
    MPSGraphTensor *standard_deviation = ltx_vae_weight(
        model, state, @"per_channel_statistics.std-of-means");
    mean = [state.graph reshapeTensor:mean
                           withShape:@[@1, @1, @1, @1, @128] name:nil];
    standard_deviation = [state.graph reshapeTensor:standard_deviation
                                         withShape:@[@1, @1, @1, @1, @128]
                                              name:nil];
    value = [state.graph multiplicationWithPrimaryTensor:value
                                         secondaryTensor:standard_deviation
                                                    name:nil];
    value = [state.graph additionWithPrimaryTensor:value
                                   secondaryTensor:mean name:nil];
    value = ltx_vae_conv3d(
        model, state, value, @"decoder.conv_in.conv", frames, 1024u);

    uint32_t current_frames = frames;
    uint32_t current_height = height;
    uint32_t current_width = width;
    for (uint32_t stage_index = 0; stage_index < 4u; stage_index++) {
        const ltx_vae_stage_spec *spec = &ltx_vae_stages[stage_index];
        for (uint32_t block = 0; block < spec->residual_blocks; block++) {
            value = ltx_vae_res_block(
                model, state, value, spec->residual_block_index, block,
                spec->input_channels, current_frames);
        }
        value = ltx_vae_depth_to_space(
            model, state, value, spec->upsample_block_index,
            batch, spec->input_channels, spec->output_channels,
            current_frames, current_height, current_width,
            spec->spatial_factor, spec->temporal_factor);
        current_frames = current_frames * spec->temporal_factor -
            (spec->temporal_factor > 1u ? 1u : 0u);
        current_height *= spec->spatial_factor;
        current_width *= spec->spatial_factor;
    }

    for (uint32_t block = 0; block < 4u; block++) {
        value = ltx_vae_res_block(
            model, state, value, 8u, block, 128u, current_frames);
    }
    value = ltx_vae_pixel_norm(state, value, 128u);
    value = ltx_vae_silu(state, value);
    value = ltx_vae_conv3d(
        model, state, value, @"decoder.conv_out.conv",
        current_frames, 48u);
    value = ltx_vae_unpatchify(
        state, value, batch, current_frames,
        current_height, current_width);

    state.outputShape =
        @[@(batch), @3, @(current_frames),
          @((uint64_t)current_height * 4u),
          @((uint64_t)current_width * 4u)];
    state.output = value;
    return state;
}

static int ltx_vae_shape_bytes(NSArray<NSNumber *> *shape,
                               uint64_t *bytes) {
    uint64_t elements = 1;
    for (NSNumber *dimension in shape) {
        if (!ltx_vae_mul_u64(elements, dimension.unsignedLongLongValue,
                             &elements))
            return 0;
    }
    return ltx_vae_mul_u64(elements, sizeof(uint16_t), bytes);
}

static LTXVideoVAEPlan *ltx_vae_build_plan(
        LTXVideoVAEModel *model, uint32_t batch,
        uint32_t frames, uint32_t height, uint32_t width,
        int token_layout, int split_graphs,
        char *error, size_t error_size) {
    if (!split_graphs) {
        LTXVideoVAEGraph *graph = ltx_vae_build_full_graph(
            model, batch, frames, height, width, token_layout);
        if (!graph || !graph.input || !graph.output) {
            ltx_vae_fail(error, error_size,
                         "build full video VAE graph failed");
            return nil;
        }
        LTXVideoVAEPlan *plan = [[LTXVideoVAEPlan alloc] init];
        plan.stages = @[graph];
        plan.workspaceBuffers = @[];
        plan.inputShape = graph.inputShape;
        plan.outputShape = graph.outputShape;
        return plan;
    }

    NSMutableArray<LTXVideoVAEGraph *> *stages =
        [NSMutableArray arrayWithCapacity:LTX_VAE_GRAPH_STAGES];
    NSMutableArray<id<MTLBuffer>> *workspace =
        [NSMutableArray arrayWithCapacity:LTX_VAE_GRAPH_STAGES - 1u];
    uint32_t current_frames = frames;
    uint32_t current_height = height;
    uint32_t current_width = width;
    for (uint32_t stage = 0; stage < LTX_VAE_GRAPH_STAGES; stage++) {
        LTXVideoVAEGraph *graph = ltx_vae_build_stage(
            model, stage, batch, current_frames,
            current_height, current_width, token_layout);
        if (!graph || !graph.input || !graph.output) {
            ltx_vae_fail(error, error_size,
                         "build video VAE stage %u failed", stage);
            return nil;
        }
        [stages addObject:graph];
        if (stage < LTX_VAE_GRAPH_STAGES - 1u) {
            uint64_t bytes = 0;
            if (!ltx_vae_shape_bytes(graph.outputShape, &bytes) ||
                bytes > NSUIntegerMax || bytes > model.device.maxBufferLength) {
                ltx_vae_fail(error, error_size,
                             "video VAE stage %u workspace is too large",
                             stage);
                return nil;
            }
            id<MTLBuffer> buffer = [model.device
                newBufferWithLength:(NSUInteger)bytes
                            options:MTLResourceStorageModePrivate];
            if (!buffer) {
                ltx_vae_fail(error, error_size,
                             "allocate video VAE stage %u workspace "
                             "(%llu bytes)", stage,
                             (unsigned long long)bytes);
                return nil;
            }
            [workspace addObject:buffer];
            const ltx_vae_stage_spec *spec = &ltx_vae_stages[stage];
            current_frames = current_frames * spec->temporal_factor -
                (spec->temporal_factor > 1u ? 1u : 0u);
            current_height *= spec->spatial_factor;
            current_width *= spec->spatial_factor;
        }
        token_layout = 0;
    }

    LTXVideoVAEPlan *plan = [[LTXVideoVAEPlan alloc] init];
    plan.stages = stages;
    plan.workspaceBuffers = workspace;
    plan.inputShape = stages.firstObject.inputShape;
    plan.outputShape = stages.lastObject.outputShape;
    return plan;
}

static LTXVideoVAEPlan *ltx_vae_plan(
        LTXVideoVAEModel *model, uint32_t batch,
        uint32_t frames, uint32_t height, uint32_t width,
        int token_layout, char *error, size_t error_size) {
    const char *full_value = ltx_runtime_getenv("LTX_VAE_FULL_GRAPH");
    int split_graphs = !(full_value && full_value[0] &&
                         strcmp(full_value, "0") != 0);
    NSString *key = [NSString stringWithFormat:
        @"%u:%u:%u:%u:%d:%d", batch, frames, height, width,
        token_layout, split_graphs];
    @synchronized (model.plans) {
        LTXVideoVAEPlan *plan = model.plans[key];
        if (!plan) {
            @try {
                plan = ltx_vae_build_plan(
                    model, batch, frames, height, width,
                    token_layout, split_graphs, error, error_size);
            } @catch (NSException *exception) {
                const char *reason = exception.reason.UTF8String;
                ltx_vae_fail(error, error_size,
                             "build MPSGraph video VAE: %s",
                             reason ? reason : "unknown exception");
                return nil;
            }
            if (plan) model.plans[key] = plan;
        }
        return plan;
    }
}

ltx_video_vae *ltx_video_vae_create(ltx_gpu *gpu,
                                    const char *checkpoint_path,
                                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !checkpoint_path || !checkpoint_path[0]) {
        ltx_vae_fail(error, error_size,
                     "missing video VAE GPU context/checkpoint");
        return NULL;
    }

    ltx_video_vae *result = NULL;
    ltx_st_header header;
    ltx_st_mapping mapping;
    memset(&header, 0, sizeof(header));
    memset(&mapping, 0, sizeof(mapping));
    if (!ltx_st_read_header(checkpoint_path, &header,
                            error, error_size))
        return NULL;
    if (!ltx_vae_validate(&header, error, error_size) ||
        !ltx_st_map_open(&header, &mapping, error, error_size)) {
        ltx_st_free_header(&header);
        return NULL;
    }

    @autoreleasepool {
        id<MTLDevice> device =
            (__bridge id<MTLDevice>)ltx_gpu_native_device(gpu);
        id<MTLCommandQueue> queue =
            (__bridge id<MTLCommandQueue>)ltx_gpu_native_queue(gpu);
        if (!device || !queue) {
            ltx_vae_fail(error, error_size,
                         "video VAE GPU native handles are unavailable");
            goto cleanup;
        }

        LTXVideoVAEModel *model = [[LTXVideoVAEModel alloc] init];
        model.device = device;
        model.queue = queue;
        model.plans = [NSMutableDictionary dictionary];
        model.weightNames = ltx_vae_weight_names();
        NSMutableArray<NSArray<NSNumber *> *> *shapes =
            [NSMutableArray arrayWithCapacity:model.weightNames.count];
        NSMutableArray<id<MTLBuffer>> *buffers =
            [NSMutableArray arrayWithCapacity:model.weightNames.count];
        NSMutableDictionary<NSString *, NSNumber *> *indices =
            [NSMutableDictionary dictionaryWithCapacity:model.weightNames.count];
        uint64_t total_bytes = 0;
        for (NSUInteger index = 0; index < model.weightNames.count; index++) {
            NSString *name = model.weightNames[index];
            const ltx_st_tensor *tensor =
                ltx_st_find(&header, name.UTF8String);
            size_t bytes = 0;
            const void *data = ltx_st_map_tensor(
                &mapping, tensor, &bytes, error, error_size);
            if (!data) goto cleanup;
            id<MTLBuffer> buffer =
                [device newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared];
            if (!buffer) {
                ltx_vae_fail(error, error_size,
                             "allocate video VAE weight %s (%zu bytes)",
                             name.UTF8String, bytes);
                goto cleanup;
            }
            ltx_vae_copy_weight_dhwio(
                tensor, data, buffer.contents, bytes);
            [shapes addObject:ltx_vae_graph_weight_shape(tensor)];
            [buffers addObject:buffer];
            indices[name] = @(index);
            if (UINT64_MAX - total_bytes < bytes) {
                ltx_vae_fail(error, error_size,
                             "video VAE weight byte count overflow");
                goto cleanup;
            }
            total_bytes += bytes;
        }
        model.weightShapes = shapes;
        model.weightBuffers = buffers;
        model.weightIndices = indices;
        model.weightBytes = total_bytes;

        result = calloc(1, sizeof(*result));
        if (!result) {
            ltx_vae_fail(error, error_size,
                         "out of memory creating video VAE");
            goto cleanup;
        }
        result->model = (__bridge_retained void *)model;
    }

cleanup:
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return result;
}

void ltx_video_vae_free(ltx_video_vae *vae) {
    if (!vae) return;
    if (vae->model) (void)CFBridgingRelease(vae->model);
    free(vae);
}

int ltx_video_vae_get_info(const ltx_video_vae *vae,
                           ltx_video_vae_info *info) {
    if (!vae || !vae->model || !info) return 0;
    @autoreleasepool {
        LTXVideoVAEModel *model =
            (__bridge LTXVideoVAEModel *)vae->model;
        memset(info, 0, sizeof(*info));
        info->latent_channels = LTX_VAE_LATENT_CHANNELS;
        info->output_channels = LTX_VAE_OUTPUT_CHANNELS;
        info->temporal_scale = 8u;
        info->temporal_offset = 7u;
        info->spatial_scale = 32u;
        info->graph_stages = LTX_VAE_GRAPH_STAGES;
        info->weight_tensors = (uint32_t)model.weightNames.count;
        info->weight_bytes = model.weightBytes;
    }
    return 1;
}

int ltx_video_vae_output_shape(uint32_t latent_frames,
                               uint32_t latent_height,
                               uint32_t latent_width,
                               uint32_t *frames,
                               uint32_t *height,
                               uint32_t *width) {
    if (!latent_frames || !latent_height || !latent_width ||
        latent_frames > (UINT32_MAX + 7ull) / 8ull ||
        latent_height > UINT32_MAX / 32u ||
        latent_width > UINT32_MAX / 32u)
        return 0;
    if (frames) *frames = latent_frames * 8u - 7u;
    if (height) *height = latent_height * 32u;
    if (width) *width = latent_width * 32u;
    return 1;
}

static int ltx_vae_run(ltx_video_vae *vae,
                       ltx_gpu_buffer *output,
                       const ltx_gpu_buffer *input,
                       uint32_t batch, uint32_t frames,
                       uint32_t height, uint32_t width,
                       int token_layout,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    uint32_t output_frames = 0;
    uint32_t output_height = 0;
    uint32_t output_width = 0;
    if (!vae || !vae->model || !input || !output || !batch ||
        !ltx_video_vae_output_shape(frames, height, width,
                                    &output_frames, &output_height,
                                    &output_width))
        return ltx_vae_fail(error, error_size,
                            "invalid video VAE decode arguments");

    uint64_t input_elements = batch;
    uint64_t output_elements = batch;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    int valid =
        ltx_vae_mul_u64(input_elements, LTX_VAE_LATENT_CHANNELS,
                        &input_elements) &&
        ltx_vae_mul_u64(input_elements, frames, &input_elements) &&
        ltx_vae_mul_u64(input_elements, height, &input_elements) &&
        ltx_vae_mul_u64(input_elements, width, &input_elements) &&
        ltx_vae_mul_u64(input_elements, sizeof(uint16_t), &input_bytes) &&
        ltx_vae_mul_u64(output_elements, LTX_VAE_OUTPUT_CHANNELS,
                        &output_elements) &&
        ltx_vae_mul_u64(output_elements, output_frames, &output_elements) &&
        ltx_vae_mul_u64(output_elements, output_height, &output_elements) &&
        ltx_vae_mul_u64(output_elements, output_width, &output_elements) &&
        ltx_vae_mul_u64(output_elements, sizeof(uint16_t), &output_bytes);
    if (!valid || input_bytes > SIZE_MAX || output_bytes > SIZE_MAX ||
        ltx_gpu_buffer_bytes(input) < (size_t)input_bytes ||
        ltx_gpu_buffer_bytes(output) < (size_t)output_bytes)
        return ltx_vae_fail(error, error_size,
                            "video VAE input/output buffer is too small");

    @autoreleasepool {
        LTXVideoVAEModel *model =
            (__bridge LTXVideoVAEModel *)vae->model;
        id<MTLBuffer> input_buffer =
            (__bridge id<MTLBuffer>)ltx_gpu_buffer_native(input);
        id<MTLBuffer> output_buffer =
            (__bridge id<MTLBuffer>)ltx_gpu_buffer_native(output);
        if (!input_buffer || !output_buffer)
            return ltx_vae_fail(error, error_size,
                                "video VAE native buffer is unavailable");
        LTXVideoVAEPlan *plan = ltx_vae_plan(
            model, batch, frames, height, width,
            token_layout, error, error_size);
        if (!plan) return 0;

        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:model.queue];
        if (!command)
            return ltx_vae_fail(error, error_size,
                                "create video VAE command buffer failed");
        for (NSUInteger stage_index = 0;
             stage_index < plan.stages.count; stage_index++) {
            LTXVideoVAEGraph *stage = plan.stages[stage_index];
            id<MTLBuffer> stage_input = stage_index == 0u ?
                input_buffer : plan.workspaceBuffers[stage_index - 1u];
            id<MTLBuffer> stage_output =
                stage_index + 1u == plan.stages.count ?
                    output_buffer : plan.workspaceBuffers[stage_index];
            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
                [NSMutableDictionary dictionaryWithCapacity:
                    stage.weightNames.count + 1u];
            feeds[stage.input] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:stage_input shape:stage.inputShape
                         dataType:MPSDataTypeBFloat16];
            for (NSUInteger weight_index = 0;
                 weight_index < stage.weightNames.count; weight_index++) {
                NSString *name = stage.weightNames[weight_index];
                NSUInteger model_index = ltx_vae_weight_index(model, name);
                feeds[stage.weightTensors[weight_index]] =
                    [[MPSGraphTensorData alloc]
                        initWithMTLBuffer:model.weightBuffers[model_index]
                                    shape:model.weightShapes[model_index]
                                 dataType:MPSDataTypeBFloat16];
            }
            MPSGraphTensorData *result = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:stage_output shape:stage.outputShape
                         dataType:MPSDataTypeBFloat16];
            @try {
                [stage.graph encodeToCommandBuffer:command feeds:feeds
                    targetOperations:nil
                    resultsDictionary:@{stage.output: result}
                    executionDescriptor:nil];
            } @catch (NSException *exception) {
                const char *reason = exception.reason.UTF8String;
                return ltx_vae_fail(error, error_size,
                                    "encode MPSGraph video VAE stage %lu: %s",
                                    (unsigned long)stage_index,
                                    reason ? reason : "unknown exception");
            }
        }

        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        if (!root)
            return ltx_vae_fail(error, error_size,
                                "video VAE MPS command has no root buffer");
        if (root.status == MTLCommandBufferStatusNotEnqueued) [root commit];
        [root waitUntilCompleted];
        if (root.status == MTLCommandBufferStatusError)
            return ltx_vae_fail(error, error_size,
                                "MPSGraph video VAE failed: %s",
                                ltx_vae_error_description(root.error));
    }
    return 1;
}

int ltx_video_vae_decode_bf16(ltx_video_vae *vae,
                              ltx_gpu_buffer *output,
                              const ltx_gpu_buffer *input,
                              uint32_t batch,
                              uint32_t frames,
                              uint32_t height,
                              uint32_t width,
                              char *error, size_t error_size) {
    return ltx_vae_run(vae, output, input, batch, frames, height, width, 0,
                       error, error_size);
}

int ltx_video_vae_decode_tokens_bf16(ltx_video_vae *vae,
                                     ltx_gpu_buffer *output,
                                     const ltx_gpu_buffer *input,
                                     uint32_t batch,
                                     uint32_t frames,
                                     uint32_t height,
                                     uint32_t width,
                                     char *error, size_t error_size) {
    return ltx_vae_run(vae, output, input, batch, frames, height, width, 1,
                       error, error_size);
}
