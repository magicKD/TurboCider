#include "ltx_w8a8.h"
#include "qwen21_cider_kernels_generated.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

@interface LTXW8A8Pipelines : NSObject
@property(nonatomic, strong) id<MTLLibrary> matmulLibrary;
@property(nonatomic, strong) id<MTLComputePipelineState> quantize;
@property(nonatomic, strong) id<MTLComputePipelineState> small;
@property(nonatomic, strong) id<MTLComputePipelineState> large;
@property(nonatomic, strong) id<MTLComputePipelineState> smallBF16;
@property(nonatomic, strong) id<MTLComputePipelineState> mediumBF16;
@property(nonatomic, strong) id<MTLComputePipelineState> largeBF16;
@property(nonatomic, strong) id<MTLComputePipelineState> smallBF16Bias;
@property(nonatomic, strong) id<MTLComputePipelineState> mediumBF16Bias;
@property(nonatomic, strong) id<MTLComputePipelineState> largeBF16Bias;
@property(nonatomic, strong) id<MTLComputePipelineState> smallBF16Gelu;
@property(nonatomic, strong) id<MTLComputePipelineState> mediumBF16Gelu;
@property(nonatomic, strong) id<MTLComputePipelineState> largeBF16Gelu;
@property(nonatomic, strong) id<MTLComputePipelineState> smallBF16BiasGelu;
@property(nonatomic, strong) id<MTLComputePipelineState> mediumBF16BiasGelu;
@property(nonatomic, strong) id<MTLComputePipelineState> largeBF16BiasGelu;
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
        std::string matmul = tc::qwen21::embedded_cider_kernel("w8a8_matmul.metal") + R"METAL(
// Preserve the separate path's BF16 rounding before adding a BF16 bias.
#define LTX_BF16_BIAS(NAME, BM, WM) \
kernel void NAME(const device int8_t* A [[buffer(0)]], const device int8_t* B [[buffer(1)]], \
    device bfloat* C [[buffer(2)]], constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]], \
    const device float* scale_a [[buffer(6)]], const device float* scale_w [[buffer(7)]], \
    constant uint& swizzle_log [[buffer(8)]], constant uint& tiles_m [[buffer(9)]], \
    constant uint& tiles_n [[buffer(10)]], const device bfloat* bias [[buffer(11)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]], \
    uint lid [[thread_index_in_simdgroup]]) { \
    w8a8_gemm_impl<BM,128,512,32,WM,4,bfloat,true,true>(A,B,C,M,N,K,scale_a,scale_w, \
        bias,swizzle_log,tiles_m,tiles_n,tgid,sgid,lid); \
}
LTX_BF16_BIAS(ltx_w8a8_bf16_bias, 128, 4)
LTX_BF16_BIAS(ltx_w8a8_bf16_bias_medium, 64, 2)
LTX_BF16_BIAS(ltx_w8a8_bf16_bias_small, 32, 1)
TC_CONVROT_GEMM(ltx_w8a8_bf16_medium, bfloat, 64, 2, 4)
#define LTX_BF16_GELU(NAME, BM, WM, BIAS) \
kernel void NAME(const device int8_t* A [[buffer(0)]], const device int8_t* B [[buffer(1)]], \
    device bfloat* C [[buffer(2)]], constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]], \
    const device float* scale_a [[buffer(6)]], const device float* scale_w [[buffer(7)]], \
    constant uint& swizzle_log [[buffer(8)]], constant uint& tiles_m [[buffer(9)]], \
    constant uint& tiles_n [[buffer(10)]], const device bfloat* bias [[buffer(11)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]], \
    uint lid [[thread_index_in_simdgroup]]) { \
    w8a8_gemm_impl<BM,128,512,32,WM,4,bfloat,BIAS,BIAS,true>(A,B,C,M,N,K,scale_a,scale_w, \
        bias,swizzle_log,tiles_m,tiles_n,tgid,sgid,lid); \
}
LTX_BF16_GELU(ltx_w8a8_bf16_gelu, 128, 4, false)
LTX_BF16_GELU(ltx_w8a8_bf16_gelu_medium, 64, 2, false)
LTX_BF16_GELU(ltx_w8a8_bf16_gelu_small, 32, 1, false)
LTX_BF16_GELU(ltx_w8a8_bf16_bias_gelu, 128, 4, true)
LTX_BF16_GELU(ltx_w8a8_bf16_bias_gelu_medium, 64, 2, true)
LTX_BF16_GELU(ltx_w8a8_bf16_bias_gelu_small, 32, 1, true)
)METAL";
        const char *barrier_option = std::getenv("TURBOCIDER_LTX_W8A8_NO_TG_BARRIER");
        if (barrier_option && std::strcmp(barrier_option, "1") == 0) {
            const std::string marker = "threadgroup_barrier(mem_flags::mem_none);";
            size_t offset = 0;
            while ((offset = matmul.find(marker, offset)) != std::string::npos)
                matmul.erase(offset, marker.size());
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = static_cast<MTLLanguageVersion>(4U << 16);
        options.mathMode = MTLMathModeSafe;
        NSError *failure = nil;
        id<MTLLibrary> quantize_library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:quantize.c_str()] options:options error:&failure];
        id<MTLLibrary> matmul_library = quantize_library ? [device newLibraryWithSource:
            [NSString stringWithUTF8String:matmul.c_str()] options:options error:&failure] : nil;
        if (matmul_library) {
            pipelines.matmulLibrary = matmul_library;
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
            pipelines.smallBF16 = [device newComputePipelineStateWithFunction:
                [matmul_library newFunctionWithName:@"tc_convrot_w8a8_bf16_small"] error:&failure];
            pipelines.largeBF16 = [device newComputePipelineStateWithFunction:
                [matmul_library newFunctionWithName:@"tc_convrot_w8a8_bf16"] error:&failure];
            pipelines.mediumBF16 = [device newComputePipelineStateWithFunction:
                [matmul_library newFunctionWithName:@"ltx_w8a8_bf16_medium"] error:&failure];
            pipelines.smallBF16Bias = [device newComputePipelineStateWithFunction:
                [matmul_library newFunctionWithName:@"ltx_w8a8_bf16_bias_small"] error:&failure];
            pipelines.largeBF16Bias = [device newComputePipelineStateWithFunction:
                [matmul_library newFunctionWithName:@"ltx_w8a8_bf16_bias"] error:&failure];
            pipelines.mediumBF16Bias = [device newComputePipelineStateWithFunction:
                [matmul_library newFunctionWithName:@"ltx_w8a8_bf16_bias_medium"] error:&failure];
        }
        if (!pipelines.quantize || !pipelines.output || !pipelines.gatedProduct || !pipelines.small || !pipelines.large ||
            !pipelines.smallBF16 || !pipelines.mediumBF16 || !pipelines.largeBF16 ||
            !pipelines.smallBF16Bias || !pipelines.mediumBF16Bias || !pipelines.largeBF16Bias) {
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

int ltx_w8a8_fused_output_enabled(void) {
    const char *option = std::getenv("TURBOCIDER_LTX_W8A8_FUSED_OUTPUT");
    // W8A8 admission already requires M5+ / macOS 26.4+. Preserve the exact
    // BF16-before-bias rounding while avoiding FP32 output/scratch traffic.
    // Keep an explicit off switch for paired regression measurements.
    return !option || std::strcmp(option, "1") == 0;
}

int ltx_w8a8_prepare_gelu_output(void *opaque, char *error, size_t error_size) {
    @autoreleasepool {
        LTXW8A8Pipelines *pipelines = (__bridge LTXW8A8Pipelines *)opaque;
        if (pipelines.largeBF16Gelu && pipelines.mediumBF16Gelu && pipelines.smallBF16Gelu &&
            pipelines.largeBF16BiasGelu && pipelines.mediumBF16BiasGelu && pipelines.smallBF16BiasGelu)
            return 1;
        NSError *failure = nil;
        id<MTLDevice> device = pipelines.matmulLibrary.device;
#define LTX_GELU_PIPELINE(PROPERTY, NAME) \
        pipelines.PROPERTY = [device newComputePipelineStateWithFunction: \
            [pipelines.matmulLibrary newFunctionWithName:@NAME] error:&failure]
        LTX_GELU_PIPELINE(smallBF16Gelu, "ltx_w8a8_bf16_gelu_small");
        LTX_GELU_PIPELINE(mediumBF16Gelu, "ltx_w8a8_bf16_gelu_medium");
        LTX_GELU_PIPELINE(largeBF16Gelu, "ltx_w8a8_bf16_gelu");
        LTX_GELU_PIPELINE(smallBF16BiasGelu, "ltx_w8a8_bf16_bias_gelu_small");
        LTX_GELU_PIPELINE(mediumBF16BiasGelu, "ltx_w8a8_bf16_bias_gelu_medium");
        LTX_GELU_PIPELINE(largeBF16BiasGelu, "ltx_w8a8_bf16_bias_gelu");
#undef LTX_GELU_PIPELINE
        if (!pipelines.smallBF16Gelu || !pipelines.mediumBF16Gelu || !pipelines.largeBF16Gelu ||
            !pipelines.smallBF16BiasGelu || !pipelines.mediumBF16BiasGelu || !pipelines.largeBF16BiasGelu) {
            if (error && error_size) snprintf(error, error_size, "create LTX GELU epilogues: %s",
                failure ? failure.localizedDescription.UTF8String : "missing Metal function");
            return 0;
        }
        return 1;
    }
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
                    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                    int fused_output) {
    ltx_w8a8_project_activation(opaque, command_encoder, codes, weight, weight_scales,
        token_scales, fp32_scratch, output, bias, rows, input_dim, output_dim, fused_output, 0);
}

void ltx_w8a8_project_activation(void *opaque, void *command_encoder, void *codes,
                    void *weight, void *weight_scales, void *token_scales,
                    void *fp32_scratch, void *output, void *bias,
                    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                    int fused_output, int gelu_output) {
    LTXW8A8Pipelines *pipelines = (__bridge LTXW8A8Pipelines *)opaque;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)command_encoder;
    const bool small = rows <= 64;
    const bool fused = fused_output != 0;
    uint32_t bm = small ? 32 : 128;
    const char *tile_option = std::getenv("TURBOCIDER_LTX_W8A8_TILE_M");
    if (fused && !small && tile_option) {
        if (std::strcmp(tile_option, "32") == 0) bm = 32;
        else if (std::strcmp(tile_option, "64") == 0) bm = 64;
    }
    const uint32_t tiles_m = (rows + bm - 1) / bm, tiles_n = (output_dim + 127) / 128;
    const uint32_t swizzle = tiles_m <= 3 ? 0 : tiles_m <= 6 ? 1 : 2;
    const uint32_t tile = 1u << swizzle;
    id<MTLComputePipelineState> pipeline = small ? pipelines.small : pipelines.large;
    if (fused) pipeline = bias ?
        (bm == 32 ? pipelines.smallBF16Bias : bm == 64 ? pipelines.mediumBF16Bias : pipelines.largeBF16Bias) :
        (bm == 32 ? pipelines.smallBF16 : bm == 64 ? pipelines.mediumBF16 : pipelines.largeBF16);
    if (gelu_output) pipeline = bias ?
        (bm == 32 ? pipelines.smallBF16BiasGelu : bm == 64 ? pipelines.mediumBF16BiasGelu : pipelines.largeBF16BiasGelu) :
        (bm == 32 ? pipelines.smallBF16Gelu : bm == 64 ? pipelines.mediumBF16Gelu : pipelines.largeBF16Gelu);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)codes offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(fused ? output : fp32_scratch) offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&output_dim length:sizeof(output_dim) atIndex:4];
    [encoder setBytes:&input_dim length:sizeof(input_dim) atIndex:5];
    [encoder setBuffer:(__bridge id<MTLBuffer>)token_scales offset:0 atIndex:6];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight_scales offset:0 atIndex:7];
    [encoder setBytes:&swizzle length:sizeof(swizzle) atIndex:8];
    [encoder setBytes:&tiles_m length:sizeof(tiles_m) atIndex:9];
    [encoder setBytes:&tiles_n length:sizeof(tiles_n) atIndex:10];
    if (fused && (bias || gelu_output))
        [encoder setBuffer:(__bridge id<MTLBuffer>)(bias ? bias : output) offset:0 atIndex:11];
    [encoder dispatchThreadgroups:MTLSizeMake(tiles_n * tile, (tiles_m + tile - 1) / tile, 1)
             threadsPerThreadgroup:MTLSizeMake(bm * 4, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    if (fused) return;
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
