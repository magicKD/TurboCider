// Independent teaching sample. Public frameworks only, no TurboCider headers.
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {
constexpr unsigned M = 33, K = 128, N = 64;

void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

void finish(id<MTLCommandBuffer> command) {
    require(command != nil, "command allocation failed");
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error(command.error.localizedDescription.UTF8String ?: "Metal failed");
    }
}

struct SharedArray {
    IOSurfaceRef surface = nullptr;
    CVPixelBufferRef pixel = nullptr;
    id<MTLBuffer> buffer = nil;
    MLMultiArray *array = nil;
    unsigned pitch = 0;

    SharedArray() = default;
    SharedArray(const SharedArray &) = delete;
    SharedArray &operator=(const SharedArray &) = delete;

    ~SharedArray() {
        array = nil;
        buffer = nil;
        if (pixel) CVPixelBufferRelease(pixel);
        if (surface) CFRelease(surface);
    }

    void create(id<MTLDevice> device, unsigned rows, unsigned columns, unsigned item_bytes) {
        pitch = ((columns * item_bytes + 63) / 64) * 64;
        const size_t page = size_t(getpagesize());
        const size_t bytes = ((size_t(rows) * pitch + page - 1) / page) * page;
        surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth: @(columns), (id)kIOSurfaceHeight: @(rows),
            (id)kIOSurfaceBytesPerElement: @(item_bytes), (id)kIOSurfaceBytesPerRow: @(pitch),
            (id)kIOSurfaceAllocSize: @(bytes),
            (id)kIOSurfacePixelFormat: @(item_bytes == 1 ? 0x4c303038 : 0x4c303068)
        });
        require(surface && IOSurfaceGetBytesPerRow(surface) == pitch, "shared surface layout failed");
        buffer = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(surface)
            length:IOSurfaceGetAllocSize(surface) options:MTLResourceStorageModeShared deallocator:nil];
        require(buffer != nil, "Metal no-copy surface binding failed");
        require(CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, surface, nullptr, &pixel) == kCVReturnSuccess,
                "CVPixelBuffer wrapping failed");
        array = [[MLMultiArray alloc] initWithPixelBuffer:pixel shape:@[@(rows), @(columns)]];
        require(array != nil, "MLMultiArray surface binding failed");
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
        if (@available(macOS 26.0, *)) {
            require(array.dataType == (item_bytes == 1 ? MLMultiArrayDataTypeInt8 : MLMultiArrayDataTypeFloat16),
                    "unexpected shared feature type");
        }
#endif
    }
};

const char *kernels = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint rows, columns, pitch, generation, kind; };
kernel void produce(device char *dst [[buffer(0)]], constant Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.columns) return;
    uint r = i / p.columns, c = i % p.columns;
    int value = p.kind == 0 ? int((r*13+c*7+p.generation*11)%63)-31
                            : int((r*5+c*3+p.generation*17)%61)-30;
    dst[r*p.pitch+c] = char(value);
}
kernel void consume(device const half *src [[buffer(0)]], device float *dst [[buffer(1)]],
                    constant Params &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.rows * p.columns) dst[i] = float(src[(i/p.columns)*p.pitch+i%p.columns]);
}
)MSL";

struct Params { uint32_t rows, columns, pitch, generation, kind; };

void dispatch(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, unsigned count) {
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

int sample(const char *compiled_path) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "no Metal device");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError *error = nil;
    MTLCompileOptions *compile_options = [MTLCompileOptions new];
    compile_options.fastMathEnabled = NO;
    id<MTLLibrary> library = [device newLibraryWithSource:@(kernels) options:compile_options error:&error];
    require(library != nil, "teaching kernel compilation failed");
    id<MTLComputePipelineState> producer = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"produce"] error:&error];
    id<MTLComputePipelineState> consumer = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"consume"] error:&error];
    require(producer && consumer, "pipeline creation failed");

    SharedArray x, w, y;
    x.create(device, K, M, 1);
    w.create(device, N, K, 1);
    y.create(device, N, M, 2);
    id<MTLBuffer> restored = [device newBufferWithLength:(N * M + 1) * sizeof(float) options:MTLResourceStorageModeShared];
    require(restored != nil, "consumer output allocation failed");
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLModel *model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(compiled_path)]
                                    configuration:configuration error:&error];
    if (!model) throw std::runtime_error(error.localizedDescription.UTF8String ?: "public model load failed");
    MLPredictionOptions *options = [MLPredictionOptions new];
    options.outputBackings = @{@"y": y.array};
    double max_error = 0;
    unsigned copies = 0;
    for (unsigned generation = 1; generation <= 2; ++generation) {
        // The producer is physically complete BEFORE Core ML reads either input.
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        Params xp{K, M, x.pitch, generation, 0}, wp{N, K, w.pitch, generation, 1};
        [encoder setBuffer:x.buffer offset:0 atIndex:0];
        [encoder setBytes:&xp length:sizeof(xp) atIndex:1];
        dispatch(encoder, producer, K * M);
        [encoder setBuffer:w.buffer offset:0 atIndex:0];
        [encoder setBytes:&wp length:sizeof(wp) atIndex:1];
        dispatch(encoder, producer, N * K);
        [encoder endEncoding];
        finish(command);
        MLDictionaryFeatureProvider *features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
            @"x": [MLFeatureValue featureValueWithMultiArray:x.array],
            @"w": [MLFeatureValue featureValueWithMultiArray:w.array]
        } error:&error];
        require(features != nil, "feature provider failed");
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:features options:options error:&error];
        if (!prediction) throw std::runtime_error(error.localizedDescription.UTF8String ?: "public prediction failed");
        MLMultiArray *actual = [prediction featureValueForName:@"y"].multiArrayValue;
        require(actual && actual.dataType == MLMultiArrayDataTypeFloat16 && [actual.shape isEqual:y.array.shape],
                "prediction output ABI mismatch");
        // outputBackings is a request, not a promise. Consume the ACTUAL result.
        if (actual.pixelBuffer != y.pixel) {
            const size_t rs = actual.strides[0].unsignedLongLongValue;
            const size_t cs = actual.strides[1].unsignedLongLongValue;
            const size_t destination_stride = y.pitch / 2;
            auto *destination = static_cast<uint16_t *>(y.buffer.contents);
            require(rs && cs, "invalid actual strides");
            __block bool copied = false;
            require(IOSurfaceLock(y.surface, 0, nullptr) == kIOReturnSuccess, "output CPU lock failed");
            [actual getBytesWithHandler:^(const void *bytes, NSInteger size) {
                if (!bytes || size < 0 || size_t(size) / 2 < (N - 1) * rs + (M - 1) * cs + 1) return;
                const auto *source = static_cast<const uint16_t *>(bytes);
                for (unsigned row = 0; row < N; ++row)
                    for (unsigned column = 0; column < M; ++column)
                        destination[row * destination_stride + column] = source[row * rs + column * cs];
                copied = true;
            }];
            IOSurfaceUnlock(y.surface, 0, nullptr);
            require(copied, "copy of non-shared actual output failed");
            ++copies;
        }
        auto *host_output = static_cast<float *>(restored.contents);
        host_output[N * M] = 12345.f;
        command = [queue commandBuffer];
        encoder = [command computeCommandEncoder];
        Params yp{N, M, y.pitch / 2, generation, 0};
        [encoder setBuffer:y.buffer offset:0 atIndex:0];
        [encoder setBuffer:restored offset:0 atIndex:1];
        [encoder setBytes:&yp length:sizeof(yp) atIndex:2];
        dispatch(encoder, consumer, N * M);
        [encoder endEncoding];
        finish(command);  // All owners outlive the last Metal consumer, even on error.
        for (unsigned row = 0; row < N; ++row) {
            for (unsigned column = 0; column < M; ++column) {
                double expected = 0;
                for (unsigned index = 0; index < K; ++index) {
                    const int a = int((index * 13 + column * 7 + generation * 11) % 63) - 31;
                    const int b = int((row * 5 + index * 3 + generation * 17) % 61) - 30;
                    expected += double(a * b) / 16384.;
                }
                require(std::isfinite(host_output[row * M + column]), "nonfinite output");
                max_error = std::max(max_error, std::abs(double(host_output[row * M + column]) - expected));
            }
        }
        require(host_output[N * M] == 12345.f, "consumer tail guard overwritten");
    }
    require(max_error < .003, "normalized INT8 matmul parity gate failed");
    NSDictionary *result = @{
        @"passed": @YES, @"generations": @2, @"shape_mkn": @[@(M), @(K), @(N)],
        @"maximum_abs_error": @(max_error), @"abs_error_gate": @0.003,
        @"output_fallback_copies": @(copies), @"tail_guard_intact": @YES,
        @"input_sharing": @"same IOSurface backing in Metal buffer and native MLMultiArray",
        @"compute_units_policy": @"CPU_AND_NE; actual placement unverified",
        @"synchronization": @"serial producer-complete -> prediction-return -> consumer-complete"
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:result options:0 error:&error];
    require(json != nil, "JSON report failed");
    std::cout.write(static_cast<const char *>(json.bytes), std::streamsize(json.length));
    std::cout << '\n';
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
        if (@available(macOS 26.0, *)) {
            try {
                return sample(argv[1]);
            } catch (const std::exception &error) {
                std::cerr << error.what() << '\n';
                return 1;
            }
        }
#endif
        std::cerr << "Public INT8 shared-feature sample requires macOS 26 and SDK 26\n";
        return 3;
    }
}
