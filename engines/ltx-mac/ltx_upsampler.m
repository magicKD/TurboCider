#include "ltx_upsampler.h"

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

@interface LTXUpsamplerGraph : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<MPSGraphTensor *> *weightTensors;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end

@implementation LTXUpsamplerGraph
@end

@interface LTXUpsamplerModel : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) NSArray<NSString *> *weightNames;
@property(nonatomic, strong) NSArray<NSArray<NSNumber *> *> *weightShapes;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *weightBuffers;
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *weightIndices;
@property(nonatomic, strong)
    NSMutableDictionary<NSString *, LTXUpsamplerGraph *> *graphs;
@property(nonatomic) uint32_t inputChannels;
@property(nonatomic) uint32_t hiddenChannels;
@property(nonatomic) uint32_t blocksPerStage;
@property(nonatomic) uint64_t weightBytes;
@end

@implementation LTXUpsamplerModel
@end

struct ltx_upsampler {
    void *model;
};

static int ltx_up_fail(char *error, size_t error_size,
                       const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static const char *ltx_up_error_description(NSError *error) {
    if (!error) return "unknown error";
    const char *description = error.localizedDescription.UTF8String;
    return description ? description : "unknown error";
}

static int ltx_up_mul_u64(uint64_t left, uint64_t right,
                          uint64_t *result) {
    if (!result || (right && left > UINT64_MAX / right)) return 0;
    *result = left * right;
    return 1;
}

static NSArray<NSString *> *ltx_up_weight_names(void) {
    NSMutableArray<NSString *> *names = [NSMutableArray arrayWithObjects:
        @"initial_conv.weight",
        @"initial_conv.bias",
        @"initial_norm.weight",
        @"initial_norm.bias",
        nil];
    for (NSString *stage in @[@"res_blocks",
                               @"post_upsample_res_blocks"]) {
        for (NSUInteger block = 0; block < 4u; block++) {
            for (NSString *suffix in @[
                    @"conv1.weight", @"conv1.bias",
                    @"norm1.weight", @"norm1.bias",
                    @"conv2.weight", @"conv2.bias",
                    @"norm2.weight", @"norm2.bias"]) {
                [names addObject:[NSString stringWithFormat:
                    @"%@.%lu.%@", stage, (unsigned long)block, suffix]];
            }
        }
        if ([stage isEqualToString:@"res_blocks"]) {
            [names addObject:@"upsampler.0.weight"];
            [names addObject:@"upsampler.0.bias"];
        }
    }
    [names addObject:@"final_conv.weight"];
    [names addObject:@"final_conv.bias"];
    return names;
}

static int ltx_up_expect(const ltx_st_header *header, const char *name,
                         uint32_t ndim, const uint64_t *shape,
                         char *error, size_t error_size) {
    const ltx_st_tensor *tensor = ltx_st_find(header, name);
    if (!tensor)
        return ltx_up_fail(error, error_size,
                           "upsampler is missing tensor %s", name);
    if (tensor->dtype != LTX_DTYPE_BF16)
        return ltx_up_fail(error, error_size,
                           "upsampler tensor %s must be BF16, got %s",
                           name, ltx_dtype_name(tensor->dtype));
    if (tensor->ndim != ndim)
        return ltx_up_fail(error, error_size,
                           "upsampler tensor %s rank is %u, expected %u",
                           name, tensor->ndim, ndim);
    for (uint32_t index = 0; index < ndim; index++) {
        if (tensor->shape[index] != shape[index])
            return ltx_up_fail(
                error, error_size,
                "upsampler tensor %s shape mismatch at axis %u: "
                "%llu != %llu",
                name, index,
                (unsigned long long)tensor->shape[index],
                (unsigned long long)shape[index]);
    }
    return 1;
}

static int ltx_up_validate(const ltx_st_header *header,
                           uint32_t *input_channels,
                           uint32_t *hidden_channels,
                           char *error, size_t error_size) {
    const ltx_st_tensor *initial =
        ltx_st_find(header, "initial_conv.weight");
    if (!initial || initial->dtype != LTX_DTYPE_BF16 ||
        initial->ndim != 5u ||
        initial->shape[0] > UINT32_MAX ||
        initial->shape[1] > UINT32_MAX)
        return ltx_up_fail(error, error_size,
                           "invalid upsampler initial_conv.weight");
    uint32_t hidden = (uint32_t)initial->shape[0];
    uint32_t channels = (uint32_t)initial->shape[1];
    if (!channels || !hidden || hidden % 32u)
        return ltx_up_fail(error, error_size,
                           "invalid upsampler channel dimensions");

    const uint64_t initial_weight[] = {
        hidden, channels, 3u, 3u, 3u};
    const uint64_t hidden_vector[] = {hidden};
    const uint64_t block_weight[] = {
        hidden, hidden, 3u, 3u, 3u};
    const uint64_t up_weight[] = {
        (uint64_t)hidden * 4u, hidden, 3u, 3u};
    const uint64_t up_bias[] = {(uint64_t)hidden * 4u};
    const uint64_t final_weight[] = {
        channels, hidden, 3u, 3u, 3u};
    const uint64_t final_bias[] = {channels};

    if (!ltx_up_expect(header, "initial_conv.weight", 5u,
                       initial_weight, error, error_size) ||
        !ltx_up_expect(header, "initial_conv.bias", 1u,
                       hidden_vector, error, error_size) ||
        !ltx_up_expect(header, "initial_norm.weight", 1u,
                       hidden_vector, error, error_size) ||
        !ltx_up_expect(header, "initial_norm.bias", 1u,
                       hidden_vector, error, error_size))
        return 0;

    for (unsigned stage_index = 0; stage_index < 2u; stage_index++) {
        const char *stage = stage_index ?
            "post_upsample_res_blocks" : "res_blocks";
        for (unsigned block = 0; block < 4u; block++) {
            static const char *const suffixes[] = {
                "conv1.weight", "conv1.bias",
                "norm1.weight", "norm1.bias",
                "conv2.weight", "conv2.bias",
                "norm2.weight", "norm2.bias",
            };
            for (unsigned suffix = 0; suffix < 8u; suffix++) {
                char name[128];
                snprintf(name, sizeof(name), "%s.%u.%s",
                         stage, block, suffixes[suffix]);
                int is_conv_weight =
                    suffix == 0u || suffix == 4u;
                if (!ltx_up_expect(
                        header, name, is_conv_weight ? 5u : 1u,
                        is_conv_weight ? block_weight : hidden_vector,
                        error, error_size))
                    return 0;
            }
        }
    }

    if (!ltx_up_expect(header, "upsampler.0.weight", 4u,
                       up_weight, error, error_size) ||
        !ltx_up_expect(header, "upsampler.0.bias", 1u,
                       up_bias, error, error_size) ||
        !ltx_up_expect(header, "final_conv.weight", 5u,
                       final_weight, error, error_size) ||
        !ltx_up_expect(header, "final_conv.bias", 1u,
                       final_bias, error, error_size))
        return 0;

    *input_channels = channels;
    *hidden_channels = hidden;
    return 1;
}

static NSArray<NSNumber *> *ltx_up_shape(const ltx_st_tensor *tensor) {
    NSMutableArray<NSNumber *> *shape =
        [NSMutableArray arrayWithCapacity:tensor->ndim];
    for (uint32_t index = 0; index < tensor->ndim; index++)
        [shape addObject:@(tensor->shape[index])];
    return shape;
}

static NSArray<NSNumber *> *ltx_up_graph_weight_shape(
        const ltx_st_tensor *tensor) {
    if (tensor->ndim == 5u) {
        return @[@(tensor->shape[2]), @(tensor->shape[3]),
                 @(tensor->shape[4]), @(tensor->shape[1]),
                 @(tensor->shape[0])];
    }
    if (tensor->ndim == 4u) {
        return @[@(tensor->shape[2]), @(tensor->shape[3]),
                 @(tensor->shape[1]), @(tensor->shape[0])];
    }
    return ltx_up_shape(tensor);
}

static void ltx_up_copy_weight_channel_last(const ltx_st_tensor *tensor,
                                            const void *source,
                                            void *destination,
                                            size_t bytes) {
    if (tensor->ndim != 4u && tensor->ndim != 5u) {
        memcpy(destination, source, bytes);
        return;
    }
    const uint64_t output_channels = tensor->shape[0];
    const uint64_t input_channels = tensor->shape[1];
    const uint64_t depth = tensor->ndim == 5u ? tensor->shape[2] : 1u;
    const uint64_t height = tensor->shape[tensor->ndim - 2u];
    const uint64_t width = tensor->shape[tensor->ndim - 1u];
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

static NSUInteger ltx_up_weight_index(LTXUpsamplerModel *model,
                                      NSString *name) {
    NSNumber *index = model.weightIndices[name];
    return index ? index.unsignedIntegerValue : NSNotFound;
}

static MPSGraphTensor *ltx_up_weight(
        LTXUpsamplerModel *model, LTXUpsamplerGraph *state,
        NSString *name) {
    NSUInteger index = ltx_up_weight_index(model, name);
    if (index == NSNotFound || index >= state.weightTensors.count)
        return nil;
    return state.weightTensors[index];
}

static MPSGraphTensor *ltx_up_silu_bf16(
        MPSGraph *graph, MPSGraphTensor *input) {
    MPSGraphTensor *value = [graph castTensor:input
                                      toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *sigmoid = [graph sigmoidWithTensor:value name:nil];
    MPSGraphTensor *result =
        [graph multiplicationWithPrimaryTensor:value
                                secondaryTensor:sigmoid name:nil];
    return [graph castTensor:result toType:MPSDataTypeBFloat16 name:nil];
}

static MPSGraphTensor *ltx_up_conv3d(
        LTXUpsamplerModel *model, LTXUpsamplerGraph *state,
        MPSGraphTensor *input, NSString *prefix, uint32_t output_channels) {
    MPSGraph *graph = state.graph;
    MPSGraphTensor *weight = ltx_up_weight(
        model, state, [prefix stringByAppendingString:@".weight"]);
    MPSGraphTensor *bias = ltx_up_weight(
        model, state, [prefix stringByAppendingString:@".bias"]);
    MPSGraphConvolution3DOpDescriptor *descriptor =
        [MPSGraphConvolution3DOpDescriptor
            descriptorWithStrideInX:1u strideInY:1u strideInZ:1u
            dilationRateInX:1u dilationRateInY:1u dilationRateInZ:1u
            groups:1u paddingLeft:1u paddingRight:1u
            paddingTop:1u paddingBottom:1u
            paddingFront:1u paddingBack:1u
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNDHWC
            weightsLayout:MPSGraphTensorNamedDataLayoutDHWIO];
    MPSGraphTensor *convolution =
        [graph convolution3DWithSourceTensor:input
                              weightsTensor:weight
                                 descriptor:descriptor name:nil];
    MPSGraphTensor *reshaped_bias =
        [graph reshapeTensor:bias
                   withShape:@[@1, @1, @1, @1, @(output_channels)]
                        name:nil];
    return [graph additionWithPrimaryTensor:convolution
                            secondaryTensor:reshaped_bias name:nil];
}

static MPSGraphTensor *ltx_up_group_norm(
        LTXUpsamplerModel *model, LTXUpsamplerGraph *state,
        MPSGraphTensor *input, NSString *prefix,
        uint32_t batch, uint32_t channels,
        uint32_t frames, uint32_t height, uint32_t width) {
    MPSGraph *graph = state.graph;
    const uint32_t groups = 32u;
    MPSGraphTensor *weight = ltx_up_weight(
        model, state, [prefix stringByAppendingString:@".weight"]);
    MPSGraphTensor *bias = ltx_up_weight(
        model, state, [prefix stringByAppendingString:@".bias"]);
    MPSGraphTensor *value = [graph castTensor:input
                                      toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *grouped =
        [graph reshapeTensor:value
                   withShape:@[@(batch), @(frames), @(height), @(width),
                               @(groups), @(channels / groups)]
                        name:nil];
    MPSGraphTensor *group_major =
        [graph transposeTensor:grouped
                   permutation:@[@0, @4, @1, @2, @3, @5] name:nil];
    MPSGraphTensor *group_flat =
        [graph reshapeTensor:group_major
                   withShape:@[@(batch), @(groups),
                               @((uint64_t)(channels / groups) * frames *
                                 height * width)]
                        name:nil];
    NSArray<NSNumber *> *axes = @[@2];
    MPSGraphTensor *mean =
        [graph meanOfTensor:group_flat axes:axes name:nil];
    MPSGraphTensor *variance =
        [graph varianceOfTensor:group_flat
                     meanTensor:mean axes:axes name:nil];
    MPSGraphTensor *normalized =
        [graph normalizationWithTensor:group_flat
                            meanTensor:mean
                        varianceTensor:variance
                           gammaTensor:nil betaTensor:nil
                               epsilon:1.0e-5f name:nil];
    MPSGraphTensor *normalized_group_major =
        [graph reshapeTensor:normalized
                   withShape:@[@(batch), @(groups), @(frames),
                               @(height), @(width), @(channels / groups)]
                        name:nil];
    MPSGraphTensor *normalized_channel_last =
        [graph transposeTensor:normalized_group_major
                   permutation:@[@0, @2, @3, @4, @1, @5] name:nil];
    MPSGraphTensor *flat =
        [graph reshapeTensor:normalized_channel_last
                   withShape:@[@(batch), @(frames), @(height),
                               @(width), @(channels)]
                        name:nil];
    MPSGraphTensor *weight_f32 =
        [graph castTensor:weight toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *bias_f32 =
        [graph castTensor:bias toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *reshaped_weight =
        [graph reshapeTensor:weight_f32
                   withShape:@[@1, @1, @1, @1, @(channels)]
                        name:nil];
    MPSGraphTensor *reshaped_bias =
        [graph reshapeTensor:bias_f32
                   withShape:@[@1, @1, @1, @1, @(channels)]
                        name:nil];
    MPSGraphTensor *scaled =
        [graph multiplicationWithPrimaryTensor:flat
                                secondaryTensor:reshaped_weight name:nil];
    MPSGraphTensor *shifted =
        [graph additionWithPrimaryTensor:scaled
                          secondaryTensor:reshaped_bias name:nil];
    return [graph castTensor:shifted
                      toType:MPSDataTypeBFloat16 name:nil];
}

static MPSGraphTensor *ltx_up_res_block(
        LTXUpsamplerModel *model, LTXUpsamplerGraph *state,
        MPSGraphTensor *input, NSString *stage, NSUInteger block,
        uint32_t batch, uint32_t channels,
        uint32_t frames, uint32_t height, uint32_t width) {
    NSString *base = [NSString stringWithFormat:
        @"%@.%lu", stage, (unsigned long)block];
    MPSGraphTensor *conv1 = ltx_up_conv3d(
        model, state, input,
        [base stringByAppendingString:@".conv1"], channels);
    MPSGraphTensor *norm1 = ltx_up_group_norm(
        model, state, conv1,
        [base stringByAppendingString:@".norm1"],
        batch, channels, frames, height, width);
    MPSGraphTensor *activated1 =
        ltx_up_silu_bf16(state.graph, norm1);
    MPSGraphTensor *conv2 = ltx_up_conv3d(
        model, state, activated1,
        [base stringByAppendingString:@".conv2"], channels);
    MPSGraphTensor *norm2 = ltx_up_group_norm(
        model, state, conv2,
        [base stringByAppendingString:@".norm2"],
        batch, channels, frames, height, width);
    MPSGraphTensor *residual =
        [state.graph additionWithPrimaryTensor:norm2
                                secondaryTensor:input name:nil];
    return ltx_up_silu_bf16(state.graph, residual);
}

static MPSGraphTensor *ltx_up_spatial_x2(
        LTXUpsamplerModel *model, LTXUpsamplerGraph *state,
        MPSGraphTensor *input, uint32_t batch, uint32_t channels,
        uint32_t frames, uint32_t height, uint32_t width) {
    MPSGraph *graph = state.graph;
    MPSGraphTensor *frames_input =
        [graph reshapeTensor:input
                   withShape:@[@((uint64_t)batch * frames), @(height),
                               @(width), @(channels)]
                        name:nil];
    MPSGraphTensor *weight =
        ltx_up_weight(model, state, @"upsampler.0.weight");
    MPSGraphTensor *bias =
        ltx_up_weight(model, state, @"upsampler.0.bias");
    MPSGraphConvolution2DOpDescriptor *descriptor =
        [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1u strideInY:1u
            dilationRateInX:1u dilationRateInY:1u groups:1u
            paddingLeft:1u paddingRight:1u
            paddingTop:1u paddingBottom:1u
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNHWC
            weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
    MPSGraphTensor *convolution =
        [graph convolution2DWithSourceTensor:frames_input
                              weightsTensor:weight
                                 descriptor:descriptor name:nil];
    MPSGraphTensor *reshaped_bias =
        [graph reshapeTensor:bias
                   withShape:@[@1, @1, @1,
                               @((uint64_t)channels * 4u)]
                        name:nil];
    MPSGraphTensor *biased =
        [graph additionWithPrimaryTensor:convolution
                          secondaryTensor:reshaped_bias name:nil];

    MPSGraphTensor *split_channels =
        [graph reshapeTensor:biased
                   withShape:@[@((uint64_t)batch * frames), @(height),
                               @(width), @(channels), @2, @2]
                        name:nil];
    MPSGraphTensor *interleaved =
        [graph transposeTensor:split_channels
                   permutation:@[@0, @1, @4, @2, @5, @3] name:nil];
    MPSGraphTensor *up_frames =
        [graph reshapeTensor:interleaved
                   withShape:@[@((uint64_t)batch * frames),
                               @((uint64_t)height * 2u),
                               @((uint64_t)width * 2u), @(channels)]
                        name:nil];
    return [graph reshapeTensor:up_frames
                     withShape:@[@(batch), @(frames),
                                 @((uint64_t)height * 2u),
                                 @((uint64_t)width * 2u), @(channels)]
                          name:nil];
}

static LTXUpsamplerGraph *ltx_up_build_graph(
        LTXUpsamplerModel *model, uint32_t batch, uint32_t frames,
        uint32_t height, uint32_t width, int token_layout) {
    LTXUpsamplerGraph *state = [[LTXUpsamplerGraph alloc] init];
    state.graph = [[MPSGraph alloc] init];
    state.inputShape = token_layout ?
        @[@(batch), @(frames), @(height), @(width),
          @(model.inputChannels)] :
        @[@(batch), @(model.inputChannels), @(frames), @(height), @(width)];
    state.outputShape = token_layout ?
        @[@(batch), @(frames), @((uint64_t)height * 2u),
          @((uint64_t)width * 2u), @(model.inputChannels)] :
        @[@(batch), @(model.inputChannels), @(frames),
          @((uint64_t)height * 2u), @((uint64_t)width * 2u)];
    state.input = [state.graph placeholderWithShape:state.inputShape
                                          dataType:MPSDataTypeBFloat16
                                              name:@"latent"];

    NSMutableArray<MPSGraphTensor *> *weight_tensors =
        [NSMutableArray arrayWithCapacity:model.weightNames.count];
    for (NSUInteger index = 0; index < model.weightNames.count; index++) {
        MPSGraphTensor *tensor =
            [state.graph placeholderWithShape:model.weightShapes[index]
                                      dataType:MPSDataTypeBFloat16
                                          name:model.weightNames[index]];
        [weight_tensors addObject:tensor];
    }
    state.weightTensors = weight_tensors;

    MPSGraphTensor *value = token_layout ? state.input :
        [state.graph transposeTensor:state.input
                         permutation:@[@0, @2, @3, @4, @1] name:nil];
    value = ltx_up_conv3d(
        model, state, value, @"initial_conv", model.hiddenChannels);
    value = ltx_up_group_norm(
        model, state, value, @"initial_norm",
        batch, model.hiddenChannels, frames, height, width);
    value = ltx_up_silu_bf16(state.graph, value);
    for (NSUInteger block = 0; block < model.blocksPerStage; block++)
        value = ltx_up_res_block(
            model, state, value, @"res_blocks", block,
            batch, model.hiddenChannels, frames, height, width);

    value = ltx_up_spatial_x2(
        model, state, value, batch, model.hiddenChannels,
        frames, height, width);
    uint32_t up_height = height * 2u;
    uint32_t up_width = width * 2u;
    for (NSUInteger block = 0; block < model.blocksPerStage; block++)
        value = ltx_up_res_block(
            model, state, value, @"post_upsample_res_blocks", block,
            batch, model.hiddenChannels, frames, up_height, up_width);
    value = ltx_up_conv3d(
        model, state, value, @"final_conv", model.inputChannels);
    state.output = token_layout ? value :
        [state.graph transposeTensor:value
                         permutation:@[@0, @4, @1, @2, @3] name:nil];
    return state;
}

static LTXUpsamplerGraph *ltx_up_graph(
        LTXUpsamplerModel *model, uint32_t batch, uint32_t frames,
        uint32_t height, uint32_t width, int token_layout) {
    NSString *key = [NSString stringWithFormat:
        @"%u:%u:%u:%u:%d", batch, frames, height, width, token_layout];
    @synchronized (model.graphs) {
        LTXUpsamplerGraph *state = model.graphs[key];
        if (!state) {
            state = ltx_up_build_graph(
                model, batch, frames, height, width, token_layout);
            model.graphs[key] = state;
        }
        return state;
    }
}

ltx_upsampler *ltx_upsampler_create(ltx_gpu *gpu,
                                    const char *checkpoint_path,
                                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !checkpoint_path || !checkpoint_path[0]) {
        ltx_up_fail(error, error_size,
                    "missing upsampler GPU context/checkpoint");
        return NULL;
    }

    ltx_upsampler *result = NULL;
    ltx_st_header header;
    ltx_st_mapping mapping;
    memset(&header, 0, sizeof(header));
    memset(&mapping, 0, sizeof(mapping));
    if (!ltx_st_read_header(checkpoint_path, &header,
                            error, error_size))
        return NULL;

    uint32_t input_channels = 0;
    uint32_t hidden_channels = 0;
    if (!ltx_up_validate(&header, &input_channels, &hidden_channels,
                         error, error_size) ||
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
            ltx_up_fail(error, error_size,
                        "upsampler GPU native handles are unavailable");
            goto cleanup;
        }

        LTXUpsamplerModel *model = [[LTXUpsamplerModel alloc] init];
        model.device = device;
        model.queue = queue;
        model.inputChannels = input_channels;
        model.hiddenChannels = hidden_channels;
        model.blocksPerStage = 4u;
        model.graphs = [NSMutableDictionary dictionary];
        model.weightNames = ltx_up_weight_names();

        NSMutableArray<NSArray<NSNumber *> *> *weight_shapes =
            [NSMutableArray arrayWithCapacity:model.weightNames.count];
        NSMutableArray<id<MTLBuffer>> *weight_buffers =
            [NSMutableArray arrayWithCapacity:model.weightNames.count];
        NSMutableDictionary<NSString *, NSNumber *> *indices =
            [NSMutableDictionary dictionaryWithCapacity:
                model.weightNames.count];
        uint64_t total_bytes = 0;

        for (NSUInteger index = 0;
             index < model.weightNames.count; index++) {
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
                ltx_up_fail(error, error_size,
                            "allocate upsampler weight %s (%zu bytes)",
                            name.UTF8String, bytes);
                goto cleanup;
            }
            ltx_up_copy_weight_channel_last(
                tensor, data, buffer.contents, bytes);
            [weight_shapes addObject:ltx_up_graph_weight_shape(tensor)];
            [weight_buffers addObject:buffer];
            indices[name] = @(index);
            if (UINT64_MAX - total_bytes < bytes) {
                ltx_up_fail(error, error_size,
                            "upsampler weight byte count overflow");
                goto cleanup;
            }
            total_bytes += bytes;
        }
        model.weightShapes = weight_shapes;
        model.weightBuffers = weight_buffers;
        model.weightIndices = indices;
        model.weightBytes = total_bytes;

        result = calloc(1, sizeof(*result));
        if (!result) {
            ltx_up_fail(error, error_size,
                        "out of memory creating upsampler");
            goto cleanup;
        }
        result->model = (__bridge_retained void *)model;
    }

cleanup:
    ltx_st_map_close(&mapping);
    ltx_st_free_header(&header);
    return result;
}

void ltx_upsampler_free(ltx_upsampler *upsampler) {
    if (!upsampler) return;
    if (upsampler->model)
        (void)CFBridgingRelease(upsampler->model);
    free(upsampler);
}

int ltx_upsampler_get_info(const ltx_upsampler *upsampler,
                           ltx_upsampler_info *info) {
    if (!upsampler || !upsampler->model || !info) return 0;
    @autoreleasepool {
        LTXUpsamplerModel *model =
            (__bridge LTXUpsamplerModel *)upsampler->model;
        memset(info, 0, sizeof(*info));
        info->input_channels = model.inputChannels;
        info->hidden_channels = model.hiddenChannels;
        info->residual_blocks_per_stage = model.blocksPerStage;
        info->weight_bytes = model.weightBytes;
        info->weight_tensors = (uint32_t)model.weightNames.count;
    }
    return 1;
}

static int ltx_up_run_bf16(ltx_upsampler *upsampler,
                           ltx_gpu_buffer *output,
                           const ltx_gpu_buffer *input,
                           uint32_t batch, uint32_t frames,
                           uint32_t height, uint32_t width,
                           int token_layout,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!upsampler || !upsampler->model || !input || !output ||
        !batch || !frames || !height || !width ||
        height > UINT32_MAX / 2u || width > UINT32_MAX / 2u)
        return ltx_up_fail(error, error_size,
                           "invalid upsampler run arguments");

    @autoreleasepool {
        LTXUpsamplerModel *model =
            (__bridge LTXUpsamplerModel *)upsampler->model;
        uint64_t input_elements = batch;
        uint64_t output_elements = batch;
        int valid =
            ltx_up_mul_u64(input_elements, model.inputChannels,
                           &input_elements) &&
            ltx_up_mul_u64(input_elements, frames, &input_elements) &&
            ltx_up_mul_u64(input_elements, height, &input_elements) &&
            ltx_up_mul_u64(input_elements, width, &input_elements) &&
            ltx_up_mul_u64(output_elements, model.inputChannels,
                           &output_elements) &&
            ltx_up_mul_u64(output_elements, frames, &output_elements) &&
            ltx_up_mul_u64(output_elements, (uint64_t)height * 2u,
                           &output_elements) &&
            ltx_up_mul_u64(output_elements, (uint64_t)width * 2u,
                           &output_elements);
        uint64_t input_bytes = 0;
        uint64_t output_bytes = 0;
        valid = valid &&
            ltx_up_mul_u64(input_elements, sizeof(uint16_t),
                           &input_bytes) &&
            ltx_up_mul_u64(output_elements, sizeof(uint16_t),
                           &output_bytes);
        if (!valid || input_bytes > SIZE_MAX || output_bytes > SIZE_MAX ||
            ltx_gpu_buffer_bytes(input) < (size_t)input_bytes ||
            ltx_gpu_buffer_bytes(output) < (size_t)output_bytes)
            return ltx_up_fail(error, error_size,
                               "upsampler input/output buffer is too small");

        id<MTLBuffer> input_buffer =
            (__bridge id<MTLBuffer>)ltx_gpu_buffer_native(input);
        id<MTLBuffer> output_buffer =
            (__bridge id<MTLBuffer>)ltx_gpu_buffer_native(output);
        if (!input_buffer || !output_buffer)
            return ltx_up_fail(error, error_size,
                               "upsampler native buffer is unavailable");

        LTXUpsamplerGraph *state = nil;
        @try {
            state = ltx_up_graph(
                model, batch, frames, height, width, token_layout);
        } @catch (NSException *exception) {
            const char *reason = exception.reason.UTF8String;
            return ltx_up_fail(error, error_size,
                               "build MPSGraph upsampler: %s",
                               reason ? reason : "unknown exception");
        }
        if (!state)
            return ltx_up_fail(error, error_size,
                               "create MPSGraph upsampler failed");

        MPSCommandBuffer *command =
            [MPSCommandBuffer commandBufferFromCommandQueue:model.queue];
        if (!command)
            return ltx_up_fail(error, error_size,
                               "create upsampler command buffer failed");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionaryWithCapacity:
                model.weightNames.count + 1u];
        feeds[state.input] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input_buffer shape:state.inputShape
                     dataType:MPSDataTypeBFloat16];
        for (NSUInteger index = 0;
             index < model.weightNames.count; index++) {
            feeds[state.weightTensors[index]] =
                [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:model.weightBuffers[index]
                                shape:model.weightShapes[index]
                             dataType:MPSDataTypeBFloat16];
        }
        MPSGraphTensorData *output_data =
            [[MPSGraphTensorData alloc]
                initWithMTLBuffer:output_buffer shape:state.outputShape
                         dataType:MPSDataTypeBFloat16];
        @try {
            [state.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil
                resultsDictionary:@{state.output: output_data}
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            const char *reason = exception.reason.UTF8String;
            return ltx_up_fail(error, error_size,
                               "encode MPSGraph upsampler: %s",
                               reason ? reason : "unknown exception");
        }

        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        if (!root)
            return ltx_up_fail(error, error_size,
                               "upsampler MPS command has no root buffer");
        if (root.status == MTLCommandBufferStatusNotEnqueued)
            [root commit];
        [root waitUntilCompleted];
        if (root.status == MTLCommandBufferStatusError)
            return ltx_up_fail(error, error_size,
                               "MPSGraph upsampler failed: %s",
                               ltx_up_error_description(root.error));
    }
    return 1;
}

int ltx_upsampler_run_bf16(ltx_upsampler *upsampler,
                           ltx_gpu_buffer *output,
                           const ltx_gpu_buffer *input,
                           uint32_t batch, uint32_t frames,
                           uint32_t height, uint32_t width,
                           char *error, size_t error_size) {
    return ltx_up_run_bf16(
        upsampler, output, input, batch, frames, height, width, 0,
        error, error_size);
}

int ltx_upsampler_run_tokens_bf16(ltx_upsampler *upsampler,
                                  ltx_gpu_buffer *output,
                                  const ltx_gpu_buffer *input,
                                  uint32_t batch, uint32_t frames,
                                  uint32_t height, uint32_t width,
                                  char *error, size_t error_size) {
    return ltx_up_run_bf16(
        upsampler, output, input, batch, frames, height, width, 1,
        error, error_size);
}
