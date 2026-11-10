#include "ltx_w8a8.h"
#include "qwen21_cider_kernels_generated.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>

@interface LTXW8A8Pipelines : NSObject
@property(nonatomic, strong) id<MTLComputePipelineState> quantize;
@property(nonatomic, strong) id<MTLComputePipelineState> small;
@property(nonatomic, strong) id<MTLComputePipelineState> large;
@property(nonatomic, strong) id<MTLComputePipelineState> output;
@property(nonatomic, strong) id<MTLComputePipelineState> gatedProduct;
@end
@implementation LTXW8A8Pipelines
@end

int ltx_w8a8_supported(void *opaque) {
    if (@available(macOS 26.4, *)) {
        id<MTLDevice> device = (__bridge id<MTLDevice>)opaque;
        unsigned generation = 0;
        // Same architecture generation gate as MLX/Cider; includes future M chips.
        const char *name = device.architecture.name.UTF8String;
        return name && sscanf(name, "applegpu_g%u", &generation) == 1 && generation >= 17;
    }
    return 0;
}

void *ltx_w8a8_create(void *opaque, char *error, size_t error_size) {
    @autoreleasepool {
        if (!ltx_w8a8_supported(opaque)) {
            if (error && error_size) snprintf(error, error_size, "LTX W8A8 requires M5+ and macOS 26.4+");
            return nullptr;
        }
        id<MTLDevice> device = (__bridge id<MTLDevice>)opaque;
        LTXW8A8Pipelines *pipelines = [LTXW8A8Pipelines new];
        // Use the exact original-code GEMM and FP32 absmax/RNE quantizer used
        // by Qwen. No FP16 scale conversion or second weight quantization.
        const std::string quantize = tc::qwen21::embedded_cider_kernel("convrot_quantize.metal") + R"METAL(
kernel void ltx_w8a8_output(const device float *x [[buffer(0)]],
    device bfloat *out [[buffer(1)]], const device bfloat *bias [[buffer(2)]],
    constant uint &count [[buffer(3)]], constant uint &columns [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i < count) {
        bfloat value = bfloat(x[i]);
        if (has_bias) value = bfloat(float(value) + float(bias[i % columns]));
        out[i] = value;
    }
}
kernel void ltx_w8a8_gated_product(const device bfloat *gate [[buffer(0)]],
    const device bfloat *up [[buffer(1)]], device bfloat *out [[buffer(2)]],
    constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < count) {
        float x = float(gate[i]);
        float cubic = (x * x) * x * 0.044715f;
        // Match ltx_gelu_tanh_bf16: Metal tanh can produce NaN for the
        // large finite gates encountered by the real Gemma checkpoint.
        float inner = clamp((x + cubic) * 0.7978845608028654f, -10.0f, 10.0f);
        bfloat activated = bfloat((x * 0.5f) * (1.0f + tanh(inner)));
        out[i] = bfloat(float(activated) * float(up[i]));
    }
}
)METAL";
        const std::string matmul = tc::qwen21::embedded_cider_kernel("w8a8_matmul.metal");
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = static_cast<MTLLanguageVersion>(4U << 16);
        options.mathMode = MTLMathModeSafe;
        NSError *failure = nil;
        id<MTLLibrary> quantize_library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:quantize.c_str()] options:options error:&failure];
        id<MTLLibrary> matmul_library = quantize_library ? [device newLibraryWithSource:
            [NSString stringWithUTF8String:matmul.c_str()] options:options error:&failure] : nil;
        if (matmul_library) {
            pipelines.quantize = [device newComputePipelineStateWithFunction:
                [quantize_library newFunctionWithName:@"tc_convrot_quantize_bf16"] error:&failure];
            pipelines.output = [device newComputePipelineStateWithFunction:
                [quantize_library newFunctionWithName:@"ltx_w8a8_output"] error:&failure];
            pipelines.gatedProduct = [device newComputePipelineStateWithFunction:
                [quantize_library newFunctionWithName:@"ltx_w8a8_gated_product"] error:&failure];
            pipelines.small = [device newComputePipelineStateWithFunction:
                [matmul_library newFunctionWithName:@"tc_convrot_w8a8_f32_small"] error:&failure];
            pipelines.large = [device newComputePipelineStateWithFunction:
                [matmul_library newFunctionWithName:@"tc_convrot_w8a8_f32"] error:&failure];
        }
        if (!pipelines.quantize || !pipelines.output || !pipelines.gatedProduct || !pipelines.small || !pipelines.large) {
            if (error && error_size) snprintf(error, error_size, "create LTX W8A8 pipelines: %s",
                failure ? failure.localizedDescription.UTF8String : "missing Metal function");
            return nullptr;
        }
        return (__bridge_retained void *)pipelines;
    }
}

void ltx_w8a8_free(void *context) {
    if (context) (void)CFBridgingRelease(context);
}

void ltx_w8a8_quantize(void *opaque, void *command_encoder, void *rotated,
                     void *codes, void *token_scales, uint32_t rows, uint32_t columns) {
    LTXW8A8Pipelines *pipelines = (__bridge LTXW8A8Pipelines *)opaque;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
    const uint32_t bm = rows <= 64 ? 32 : 128;
    const uint32_t padded_rows = (rows + bm - 1) / bm * bm;
    [encoder setComputePipelineState:pipelines.quantize];
    [encoder setBuffer:(__bridge id<MTLBuffer>)rotated offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)codes offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)token_scales offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(padded_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void ltx_w8a8_gated_product(void *opaque, void *command_encoder, void *gate,
                           void *up, void *output, uint32_t count) {
    LTXW8A8Pipelines *pipelines = (__bridge LTXW8A8Pipelines *)opaque;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
    [encoder setComputePipelineState:pipelines.gatedProduct];
    [encoder setBuffer:(__bridge id<MTLBuffer>)gate offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)up offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void ltx_w8a8_project(void *opaque, void *command_encoder, void *codes,
                    void *weight, void *weight_scales, void *token_scales,
                    void *fp32_scratch, void *output, void *bias,
                    uint32_t rows, uint32_t input_dim, uint32_t output_dim) {
    LTXW8A8Pipelines *pipelines = (__bridge LTXW8A8Pipelines *)opaque;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
    const bool small = rows <= 64;
    const uint32_t bm = small ? 32 : 128;
    const uint32_t tiles_m = (rows + bm - 1) / bm, tiles_n = (output_dim + 127) / 128;
    const uint32_t swizzle = tiles_m <= 3 ? 0 : tiles_m <= 6 ? 1 : 2;
    const uint32_t tile = 1u << swizzle;
    [encoder setComputePipelineState:small ? pipelines.small : pipelines.large];
    [encoder setBuffer:(__bridge id<MTLBuffer>)codes offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)fp32_scratch offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&output_dim length:sizeof(output_dim) atIndex:4];
    [encoder setBytes:&input_dim length:sizeof(input_dim) atIndex:5];
    [encoder setBuffer:(__bridge id<MTLBuffer>)token_scales offset:0 atIndex:6];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight_scales offset:0 atIndex:7];
    [encoder setBytes:&swizzle length:sizeof(swizzle) atIndex:8];
    [encoder setBytes:&tiles_m length:sizeof(tiles_m) atIndex:9];
    [encoder setBytes:&tiles_n length:sizeof(tiles_n) atIndex:10];
    [encoder dispatchThreadgroups:MTLSizeMake(tiles_n * tile, (tiles_m + tile - 1) / tile, 1)
             threadsPerThreadgroup:MTLSizeMake(small ? 128 : 512, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    const uint32_t count = rows * output_dim, has_bias = bias ? 1 : 0;
    [encoder setComputePipelineState:pipelines.output];
    [encoder setBuffer:(__bridge id<MTLBuffer>)fp32_scratch offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(bias ? bias : output) offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder setBytes:&output_dim length:sizeof(output_dim) atIndex:4];
    [encoder setBytes:&has_bias length:sizeof(has_bias) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
