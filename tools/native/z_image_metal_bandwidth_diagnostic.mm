// Standalone Metal control: intentionally independent of MLX and model code.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "z_image_gpu_benchmark_lock.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <mach/mach.h>
#include <mach/task_policy.h>
#include <pthread.h>
#include <sys/resource.h>

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::stoi(argv[1]) : 1;
    const bool private_storage = argc > 2 && std::string(argv[2]) == "private";
    if (repeats < 1 || repeats > 64) return 2;
    ZImageGpuBenchmarkLock lock;
    @autoreleasepool {
        qos_class_t initial_qos = QOS_CLASS_UNSPECIFIED;
        int initial_rel = 0;
        pthread_get_qos_class_np(pthread_self(), &initial_qos, &initial_rel);
        task_category_policy_data_t policy{};
        mach_msg_type_number_t policy_count = TASK_CATEGORY_POLICY_COUNT;
        boolean_t default_policy = false;
        auto policy_status = task_policy_get(mach_task_self(),TASK_CATEGORY_POLICY,
            reinterpret_cast<task_policy_t>(&policy),&policy_count,&default_policy);
        std::cerr << "task_policy_status=" << policy_status << " role=" << policy.role
                  << " darwin_bg=" << getpriority(PRIO_DARWIN_BG,0)
                  << " qos=" << int(initial_qos) << " rel=" << initial_rel << '\n';
        id<NSObject> activity = nil;
        if (argc > 2 && std::string(argv[2]) == "active") {
            const int bg_result = setpriority(PRIO_DARWIN_BG,0,0);
            const int qos_result = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED,0);
            activity = [NSProcessInfo.processInfo beginActivityWithOptions:
                NSActivityUserInitiatedAllowingIdleSystemSleep reason:@"Short GPU scheduling diagnostic"];
            qos_class_t active_qos = QOS_CLASS_UNSPECIFIED;
            int active_rel = 0;
            pthread_get_qos_class_np(pthread_self(), &active_qos, &active_rel);
            std::cerr << "foreground_bg_result=" << bg_result << " qos_result=" << qos_result
                      << " qos=" << int(active_qos) << " rel=" << active_rel << '\n';
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* error = nil;
        NSString* source = @"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void add(device const float* x [[buffer(0)]],"
            "device float* y [[buffer(1)]], uint i [[thread_position_in_grid]])"
            "{ y[i] = x[i] + 1.0f; }"
            "kernel void add4(device const float4* x [[buffer(0)]],"
            "device float4* y [[buffer(1)]], uint i [[thread_position_in_grid]])"
            "{ y[i] = x[i] + 1.0f; }";
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) { std::cerr << error.description.UTF8String << '\n'; return 1; }
        const size_t count = 256 * 256 * 128;
        id<MTLBuffer> staging = [device newBufferWithLength:count*sizeof(float)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> input = private_storage ? [device newBufferWithLength:count*sizeof(float)
            options:MTLResourceStorageModePrivate] : staging;
        id<MTLBuffer> output = [device newBufferWithLength:count*sizeof(float)
            options:private_storage ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared];
        id<MTLBuffer> readback = private_storage ? [device newBufferWithLength:count*sizeof(float)
            options:MTLResourceStorageModeShared] : output;
        if (!staging || !input || !output || !readback) return 1;
        float* data = static_cast<float*>(staging.contents);
        for (size_t i = 0; i < count; ++i) data[i] = float(i % 100);
        auto copy = [&](id<MTLBuffer> from, id<MTLBuffer> to) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromBuffer:from sourceOffset:0 toBuffer:to destinationOffset:0 size:count*sizeof(float)];
            [blit endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return cb.status == MTLCommandBufferStatusCompleted;
        };
        if (private_storage && !copy(staging,input)) return 1;
        for (NSUInteger vector_width : {1ul, 4ul}) {
          id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
              [library newFunctionWithName:vector_width == 1 ? @"add" : @"add4"] error:&error];
          if (!pipeline) { std::cerr << error.description.UTF8String << '\n'; return 1; }
          for (NSUInteger threads : {128ul, 256ul, 512ul}) {
            std::vector<double> wall, gpu;
            for (int i = 0; i < 8; ++i) {
                const auto start = std::chrono::steady_clock::now();
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc setBuffer:input offset:0 atIndex:0];
                [enc setBuffer:output offset:0 atIndex:1];
                for (int dispatch = 0; dispatch < repeats; ++dispatch) {
                    [enc dispatchThreads:MTLSizeMake(count/vector_width,1,1)
                        threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) return 1;
                if (i >= 3) {
                    wall.push_back(std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-start).count());
                    gpu.push_back(1000.0*(cb.GPUEndTime-cb.GPUStartTime));
                }
            }
            std::sort(wall.begin(),wall.end()); std::sort(gpu.begin(),gpu.end());
            // Upload/readback are deliberately outside kernel execution timing.
            if (private_storage && !copy(output,readback)) return 1;
            const float* result = static_cast<float*>(readback.contents);
            for (size_t i = 0; i < count; ++i)
                if (result[i] != data[i]+1.0f) return 1;
            std::cout << "{\"threads\":" << threads << ",\"vector_width\":" << vector_width
                      << ",\"storage\":\"" << (private_storage ? "private" : "shared") << "\""
                      << ",\"dispatches\":" << repeats << ",\"wall_ms\":" << wall[2]
                      << ",\"gpu_ms\":" << gpu[2] << ",\"gpu_ms_per_dispatch\":" << gpu[2]/repeats
                      << ",\"effective_GB_s\":"
                      << (2.0*count*sizeof(float)*repeats/1e6/gpu[2]) << "}" << std::endl;
          }
        }
        if (activity) [NSProcessInfo.processInfo endActivity:activity];
    }
}
