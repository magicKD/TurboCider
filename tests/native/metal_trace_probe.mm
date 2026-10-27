#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 77;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            @"#include <metal_stdlib>\nusing namespace metal;\nkernel void trace_probe(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = float(i); }"
            options:nil error:&error];
        if (!library) return 1;
        id<MTLFunction> function = [library newFunctionWithName:@"trace_probe"];
        auto descriptor = [MTLFunctionDescriptor functionDescriptor];
        descriptor.name = @"trace_probe";
        id<MTLFunction> specialized = [library newFunctionWithDescriptor:descriptor error:&error];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!specialized || !pipeline) return 2;
        auto compute = [[MTLComputePipelineDescriptor alloc] init]; compute.computeFunction = specialized;
        id<MTLComputePipelineState> pipeline2 = [device newComputePipelineStateWithDescriptor:compute options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!pipeline2) return 3;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> buffer = [device newBufferWithLength:4 * sizeof(float) options:MTLResourceStorageModeShared];
        auto command = [queue commandBuffer]; auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline]; [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        auto result = static_cast<const float *>(buffer.contents);
        for (int i = 0; i < 4; ++i) if (result[i] != float(i)) return 4;
        id<MTLLibrary> invalid = [device newLibraryWithSource:@"not valid metal" options:nil error:&error];
        if (invalid || !error) return 5;
        [buffer release]; [queue release]; [pipeline2 release]; [compute release];
        [pipeline release]; [specialized release]; [function release]; [library release]; [device release];
        std::puts("PASS original new-method ownership, GPU execution and compile-error forwarding");
    }
}
