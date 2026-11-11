#include "ltx_gpu.h"
extern "C" {
#include "ltx_gpu_internal.h"
}
#include "ltx_w8a8.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <stdexcept>
#include <vector>

static char error[2048];
static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(error[0] ? error : message);
}
using Buffer = std::unique_ptr<ltx_gpu_buffer, decltype(&ltx_gpu_buffer_free)>;
static Buffer tensor(ltx_gpu *gpu, size_t bytes, const void *data = nullptr) {
    auto *p = data ? ltx_gpu_buffer_new_copy(gpu, data, bytes, error, sizeof(error)) :
        ltx_gpu_buffer_new(gpu, bytes, error, sizeof(error));
    require(p, "allocate tensor");
    return Buffer(p, ltx_gpu_buffer_free);
}
static void complete(id<MTLCommandBuffer> command) {
    [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        std::snprintf(error, sizeof(error), "%s", command.error.localizedDescription.UTF8String);
        require(false, "GPU command failed");
    }
}
int main(int argc, char **argv) {
    @autoreleasepool { try {
        require(argc == 3, "usage: correctness BASELINE_SHADER CANDIDATE_SHADER");
        auto gpu = std::unique_ptr<ltx_gpu, decltype(&ltx_gpu_free)>(
            ltx_gpu_create(argv[1], error, sizeof(error)), ltx_gpu_free);
        require(bool(gpu), "GPU create");
        if (!ltx_gpu_convrot_w8a8_info(gpu.get(),nullptr,nullptr)) {
            std::puts("SKIP fused quantization oracle: W8A8 hardware unavailable");return 0;
        }
        require(ltx_gpu_convrot_simd_enabled(gpu.get()), "SIMD route required");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *failure = nil;
        NSString *source = [NSString stringWithContentsOfFile:
            [NSString stringWithUTF8String:argv[2]] encoding:NSUTF8StringEncoding error:&failure];
        MTLCompileOptions *options = [MTLCompileOptions new]; options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&failure];
        if (!library) { std::fprintf(stderr,"%s\n",failure.localizedDescription.UTF8String); return 1; }
        void *w8 = ltx_w8a8_create((__bridge void *)device,error,sizeof(error));
        require(w8, "create original FP32 quantizer");
        id<MTLComputePipelineState> pipelines[2][3];
        for (unsigned gelu = 0; gelu < 2; ++gelu) for (unsigned index = 0; index < 3; ++index) {
            NSString *name = [NSString stringWithFormat:@"%s_s%u",
                gelu ? "ltx_gelu_rot_quant_bf16" : "ltx_rot_quant_bf16", 1u << index];
            pipelines[gelu][index] = [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:name] error:&failure];
            if (!pipelines[gelu][index]) { std::fprintf(stderr,"%s\n",failure.localizedDescription.UTF8String); return 1; }
        }
        auto choose = [&](unsigned gelu, uint32_t columns, uint32_t &threads) -> id<MTLComputePipelineState> {
            for (uint32_t cap : {32u,16u}) {
                uint32_t groups = std::min(cap,columns/256), segments = (columns/256+groups-1)/groups;
                unsigned index = segments <= 1 ? 0 : segments <= 2 ? 1 : 2;
                id<MTLComputePipelineState> p = pipelines[gelu][index];
                if (p.threadExecutionWidth == 32 && p.maxTotalThreadsPerThreadgroup >= groups*32) {
                    threads = groups*32;return p;
                }
            }
            throw std::runtime_error("no compatible fused geometry");
        };
        uint32_t random = 73;
        for (auto [rows, columns] : {std::pair{1u,256u}, std::pair{7u,512u},
                std::pair{33u,768u}, std::pair{65u,3584u}, std::pair{129u,4096u},
                std::pair{65u,8192u}, std::pair{129u,16384u}}) {
            const uint32_t bm = rows <= 64 ? 32 : 128;
            const uint32_t padded = (rows + bm - 1) / bm * bm;
            size_t count = size_t(rows) * columns, padded_count = size_t(padded) * columns;
            std::vector<uint16_t> input(count);
            for (size_t i = 0; i < count; ++i) {
                random ^= random << 13; random ^= random >> 17; random ^= random << 5;
                input[i] = uint16_t((random & 0x807fu) | ((random >> 8) % 243u) << 7);
            }
            std::fill(input.begin(),input.begin()+columns,uint16_t(0));
            if (rows > 1) std::fill(input.begin()+columns,input.begin()+2*columns,uint16_t(0x8000));
            if (rows > 3) input[3*columns+3] = 0x7fc1;
            if (rows > 4) input[4*columns] = 0x7f80;
            if (rows > 5) input[5*columns+1] = 0xff80;
            auto x = tensor(gpu.get(),count*2,input.data()), activated = tensor(gpu.get(),count*2);
            auto rotated = tensor(gpu.get(),count*2);
            auto codes = tensor(gpu.get(),padded_count), scales = tensor(gpu.get(),size_t(padded)*4);
            id<MTLBuffer> actual_codes = [device newBufferWithLength:padded_count options:MTLResourceStorageModeShared];
            id<MTLBuffer> actual_scales = [device newBufferWithLength:size_t(padded)*4 options:MTLResourceStorageModeShared];
            for (unsigned gelu = 0; gelu < 2; ++gelu) {
                memset(ltx_gpu_buffer_contents(codes.get()),0xa5,padded_count);
                memset(ltx_gpu_buffer_contents(scales.get()),0xa5,size_t(padded)*4);
                memset(actual_codes.contents,0xa5,padded_count);
                memset(actual_scales.contents,0xa5,size_t(padded)*4);
                if (gelu) require(ltx_gpu_gelu_tanh_bf16(gpu.get(),activated.get(),x.get(),uint32_t(count),error,sizeof(error)),"reference GELU");
                require(ltx_gpu_convrot_bf16(gpu.get(),rotated.get(),gelu ? activated.get() : x.get(),
                    rows,columns,256,error,sizeof(error)),"reference rotation");
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                ltx_w8a8_quantize(w8,(__bridge void *)encoder,ltx_gpu_buffer_native(rotated.get()),
                    ltx_gpu_buffer_native(codes.get()),ltx_gpu_buffer_native(scales.get()),rows,columns);
                [encoder endEncoding]; complete(command);
                command = [queue commandBuffer]; encoder = [command computeCommandEncoder];
                uint32_t threads = 0;
                id<MTLComputePipelineState> selected = choose(gelu,columns,threads);
                [encoder setComputePipelineState:selected];
                [encoder setBuffer:(__bridge id<MTLBuffer>)ltx_gpu_buffer_native(x.get()) offset:0 atIndex:0];
                [encoder setBuffer:actual_codes offset:0 atIndex:1];
                [encoder setBuffer:actual_scales offset:0 atIndex:2];
                [encoder setBytes:&rows length:4 atIndex:3];
                [encoder setBytes:&columns length:4 atIndex:4];
                require(threads <= selected.maxTotalThreadsPerThreadgroup,"pipeline thread count");
                [encoder dispatchThreadgroups:MTLSizeMake(padded,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
                [encoder endEncoding]; complete(command);
                require(!memcmp(ltx_gpu_buffer_contents(codes.get()),actual_codes.contents,padded_count),"INT8 codes mismatch");
                if (memcmp(ltx_gpu_buffer_contents(scales.get()),actual_scales.contents,size_t(padded)*4)) {
                    auto *a = (const uint32_t *)ltx_gpu_buffer_contents(scales.get());
                    auto *b = (const uint32_t *)actual_scales.contents;
                    for (uint32_t j = 0; j < padded; ++j) if (a[j]!=b[j]) {
                        std::snprintf(error,sizeof(error),"scale mismatch row%u reference%08x actual%08x",j,a[j],b[j]);break;
                    }
                    require(false,"FP32 scales mismatch");
                }
                std::printf("PASS rows=%u columns=%u padded=%u gelu=%u codes/scales byte-exact\n",rows,columns,padded,gelu);
            }
        }
        id<MTLComputePipelineState> adaln_pipeline = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"ltx_adaln_rot_quant_bf16"] error:&failure];
        require(adaln_pipeline && adaln_pipeline.threadExecutionWidth == 32 &&
            adaln_pipeline.maxTotalThreadsPerThreadgroup >= 512, "AdaLN pipeline geometry");
        for (uint32_t rows : {1u, 7u, 17u, 33u, 65u, 129u}) {
            const uint32_t columns = 4096, bm = rows <= 64 ? 32 : 128;
            const uint32_t padded = (rows + bm - 1) / bm * bm;
            const float epsilon = 1e-6f;
            size_t count = size_t(rows) * columns, padded_count = size_t(padded) * columns;
            std::vector<uint16_t> input(count), norm_scale(columns), norm_shift(columns);
            for (size_t i = 0; i < count; ++i) {
                random = random * 1664525u + 1013904223u;
                float value = float(random % 2048) / 1024 - 1;
                uint32_t bits; memcpy(&bits, &value, 4);
                input[i] = uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
            }
            for (uint32_t i = 0; i < columns; ++i) {
                norm_scale[i] = uint16_t(0x3b00 + (i * 17) % 1024);
                norm_shift[i] = uint16_t(0xbb00 + (i * 11) % 1024);
            }
            if (rows == 1) {
                std::fill(input.begin(), input.end(), uint16_t(0x3f80));
                std::fill(norm_scale.begin(), norm_scale.end(), uint16_t(0));
                std::fill(norm_shift.begin(), norm_shift.end(), uint16_t(0));
            }
            if (rows == 17) {
                size_t index = 0;
                for (uint32_t pattern = 0; pattern < 65536; ++pattern)
                    if ((pattern & 0x7f80u) != 0x7f80u) input[index++] = uint16_t(pattern);
                require(index == 65280, "all finite BF16 patterns");
                input[16 * columns] = 0x7fc1;
            }
            if (rows >= 33) {
                std::fill(input.begin(), input.begin() + columns, uint16_t(0));
                std::fill(input.begin() + columns, input.begin() + 2 * columns, uint16_t(0x8000));
                input[2 * columns] = 0x7f80; input[3 * columns] = 0xff80;
                input[4 * columns] = 0x7fc1;
            }
            auto x = tensor(gpu.get(), count * 2, input.data()), normalized = tensor(gpu.get(), count * 2);
            auto ns = tensor(gpu.get(), columns * 2, norm_scale.data()), nt = tensor(gpu.get(), columns * 2, norm_shift.data());
            auto rotated = tensor(gpu.get(), count * 2), codes = tensor(gpu.get(), padded_count), scales = tensor(gpu.get(), padded * 4);
            require(ltx_gpu_adaln_bf16(gpu.get(), normalized.get(), x.get(), ns.get(), nt.get(),
                rows, columns, 1, epsilon, error, sizeof(error)), "separate AdaLN reference");
            require(ltx_gpu_convrot_bf16(gpu.get(), rotated.get(), normalized.get(), rows, columns,
                256, error, sizeof(error)), "separate ConvRot reference");
            id<MTLBuffer> actual_codes = [device newBufferWithLength:padded_count options:MTLResourceStorageModeShared];
            id<MTLBuffer> actual_scales = [device newBufferWithLength:padded * 4 options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            ltx_w8a8_quantize(w8, (__bridge void*)encoder, ltx_gpu_buffer_native(rotated.get()),
                ltx_gpu_buffer_native(codes.get()), ltx_gpu_buffer_native(scales.get()), rows, columns);
            [encoder endEncoding]; complete(command);
            command = [queue commandBuffer]; encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:adaln_pipeline];
            [encoder setBuffer:(__bridge id<MTLBuffer>)ltx_gpu_buffer_native(x.get()) offset:0 atIndex:0];
            [encoder setBuffer:actual_codes offset:0 atIndex:1]; [encoder setBuffer:actual_scales offset:0 atIndex:2];
            [encoder setBytes:&rows length:4 atIndex:3]; [encoder setBytes:&columns length:4 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)ltx_gpu_buffer_native(ns.get()) offset:0 atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)ltx_gpu_buffer_native(nt.get()) offset:0 atIndex:6];
            [encoder setBytes:&epsilon length:4 atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(padded, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
            [encoder endEncoding]; complete(command);
            require(!memcmp(ltx_gpu_buffer_contents(codes.get()), actual_codes.contents, padded_count), "AdaLN fused INT8 codes");
            require(!memcmp(ltx_gpu_buffer_contents(scales.get()), actual_scales.contents, padded * 4), "AdaLN fused FP32 scales");
            if (rows == 1) {
                const auto* values = static_cast<const int8_t*>(actual_codes.contents);
                // This rotation uses four H4 rows whose signs sum to two.
                // H4^4 therefore maps ones to ones after its 1/16 scaling.
                for (uint32_t column = 0; column < columns; ++column)
                    require(values[column] == 127, "independent constant-input H4 witness");
                require(static_cast<const float*>(actual_scales.contents)[0] == 1.0f / 127.0f, "independent scale witness");
            }
            std::printf("PASS AdaLN/rotation/quantization rows=%u codes/scales byte-exact, finite patterns/tails/nonfinite flags\n", rows);
        }
        ltx_w8a8_free(w8);
        std::puts("PASS exact rotation/quantization: signed zero, BF16 tails, NaN/Inf flags, padded rows");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what());return 1; } }
}
