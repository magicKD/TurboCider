// Original explicit IOSurface/Metal -> MLX adoption/lifetime probe.
// Synthetic values only. No private API, real model, security changes or benchmark score.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <mlx/array.h>
#include <mlx/allocator.h>
#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/version.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace mx = mlx::core;
namespace {
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
uint64_t now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
NSString *sha(const void *data, size_t size) {
    check(size <= UINT32_MAX, "bounded digest required");
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(data, CC_LONG(size), bytes);
    NSMutableString *result = [NSMutableString string];
    for (auto byte : bytes) [result appendFormat:@"%02x", byte];
    return result;
}
void save(NSDictionary *report, const char *path) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    check(data && [data writeToFile:[NSString stringWithUTF8String:path] options:NSDataWritingAtomic error:&error], "report write failed");
}
size_t rounded_allocation(size_t value) { const size_t page = size_t(getpagesize()); return (value + page - 1) / page * page; }
double source(size_t i, uint32_t generation, uint32_t seed) {
    return (int((i * 7 + generation * 5 + seed * 11) % 37) - 18) / 16.;
}
struct Parameters { uint32_t rows, cols, pitch, generation, seed; };
const char *shader = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct P { uint rows, cols, pitch, generation, seed; };
kernel void produce(device half *x [[buffer(0)]], constant P &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.cols) return;
    x[(i / p.cols) * p.pitch + i % p.cols] = half((int((i * 7 + p.generation * 5 + p.seed * 11) % 37) - 18) / 16.0f);
}
)METAL";

void produce(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, id<MTLBuffer> buffer,
             const Parameters &parameters, bool &pending) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    check(command && encoder, "Metal command unavailable");
    [encoder setComputePipelineState:pipeline]; [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(parameters.rows * parameters.cols, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min(NSUInteger(128), pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding]; pending = true; [command commit]; [command waitUntilCompleted];
    check(command.status == MTLCommandBufferStatusCompleted, "owned Metal producer failed"); pending = false;
}
bool tail(IOSurfaceRef surface, size_t payload) {
    const auto *bytes = static_cast<const unsigned char *>(IOSurfaceGetBaseAddress(surface));
    return std::all_of(bytes + payload, bytes + IOSurfaceGetAllocSize(surface), [](unsigned char value) { return value == 0x6b; });
}
NSDictionary *alias(const mx::array &array, IOSurfaceRef surface, id<MTLDevice> device) {
    // Version-bound Metal backend Buffer::ptr() is MTL::Buffer*, not an arbitrary CPU pointer.
    // Its identity is also checked against raw_ptr(), storageMode, length and the actual device.
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)array.buffer().ptr();
    check(buffer && [buffer respondsToSelector:@selector(contents)] && [buffer respondsToSelector:@selector(device)], "unexpected MLX buffer object");
    NSDictionary *value = @{@"buffer_contents_matches_surface": @(buffer.contents == IOSurfaceGetBaseAddress(surface)),
        @"allocator_raw_pointer_matches_surface": @(const_cast<mx::array &>(array).buffer().raw_ptr() == IOSurfaceGetBaseAddress(surface)),
        @"same_metal_device": @(buffer.device.registryID == device.registryID),
        @"shared_storage": @(buffer.storageMode == MTLStorageModeShared),
        @"zero_view_offset": @(array.offset() == 0), @"buffer_covers_surface": @(buffer.length == IOSurfaceGetAllocSize(surface)),
        @"mlx_buffer_hazard_tracking_untracked": @(buffer.hazardTrackingMode == MTLHazardTrackingModeUntracked)};
    for (NSString *key in @[@"buffer_contents_matches_surface", @"allocator_raw_pointer_matches_surface", @"same_metal_device",
                           @"shared_storage", @"zero_view_offset", @"buffer_covers_surface"])
        check([value[key] boolValue], "actual MLX adoption/view alias proof failed");
    return value;
}
mx::array consume(const mx::array &input) {
    return mx::astype(mx::add(mx::multiply(input, mx::array(2.f, mx::float16)), mx::array(.25f, mx::float16)), mx::float32);
}
NSDictionary *metrics(mx::array &value, uint32_t rows, uint32_t cols, uint32_t generation, uint32_t seed) {
    value.eval(); mx::synchronize();
    check(value.dtype() == mx::float32 && value.size() == size_t(rows) * cols, "complete FP32 consumer required");
    const auto *data = value.data<float>(); size_t mismatches = 0, nonfinite = 0; double maximum = 0;
    for (size_t i = 0; i < value.size(); ++i) {
        double actual = data[i], expected = 2 * source(i, generation, seed) + .25;
        nonfinite += !std::isfinite(actual); mismatches += !std::isfinite(actual) || actual != expected;
        if (std::isfinite(actual)) maximum = std::max(maximum, std::abs(actual - expected));
    }
    return @{@"mismatched_elements": @(mismatches), @"nonfinite_elements": @(nonfinite),
        @"maximum_finite_abs_error": @(maximum), @"consumer_sha256": sha(data, value.nbytes())};
}

void exercise(NSString *mode, uint32_t rows, uint32_t cols, uint32_t pitch, uint32_t seed,
              IOSurfaceRef surface, id<MTLDevice> device, id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> pipeline, id<MTLBuffer> producer_buffer,
              bool &gpu_pending, NSMutableDictionary *record, const char *path, NSMutableDictionary *report) {
    const size_t payload = size_t(rows) * pitch * 2, allocation = IOSurfaceGetAllocSize(surface);
    void *base = IOSurfaceGetBaseAddress(surface);
    check(IOSurfaceLock(surface, 0, nullptr) == kIOReturnSuccess, "surface CPU lock failed");
    std::memset(base, 0xff, payload); std::memset(static_cast<char *>(base) + payload, 0x6b, allocation - payload);
    check(IOSurfaceUnlock(surface, 0, nullptr) == kIOReturnSuccess, "surface CPU unlock failed");
    uint32_t calls = 0; bool deleter_pointer_matches = true;
    std::optional<mx::array> live, packed;
    auto observe = [&](void *pointer) { ++calls; deleter_pointer_matches &= pointer == base; };
    record[@"mode"] = mode; record[@"samples"] = [NSMutableArray array];
    {
        std::optional<mx::array> flat;
        // Adopt the WHOLE page-aligned region, then expose the actual logical view.
        // This avoids pretending that a short, padded logical span is a valid page-sized import.
        const mx::Shape flat_shape{int32_t(allocation / 2)};
        if ([mode isEqual:@"raw-pointer"]) flat.emplace(base, flat_shape, mx::float16, observe);
        else {
            auto buffer = mx::allocator::make_buffer(base, allocation);
            check(buffer.ptr(), "strict allocator wrapping failed; no copy fallback");
            try {
                flat.emplace(buffer, flat_shape, mx::float16, [observe](mx::allocator::Buffer value) {
                    void *pointer = value.raw_ptr(); mx::allocator::release(value); observe(pointer);
                });
            } catch (...) { mx::allocator::release(buffer); throw; }
        }
        record[@"deleter_calls_after_construct"] = @(calls);
        check(calls == 0, "raw-pointer constructor used its copy fallback");
        record[@"flat_alias"] = alias(*flat, surface, device);
        live.emplace(mx::as_strided(*flat, {int32_t(rows), int32_t(cols)}, {int64_t(pitch), 1}, 0)); live->eval();
        record[@"view_alias"] = alias(*live, surface, device);
        if (pitch != cols) { packed.emplace(mx::as_strided(*flat, {int32_t(rows), int32_t(cols)}, {int64_t(cols), 1}, 0)); packed->eval(); }
    }
    record[@"deleter_calls_after_flat_scope"] = @(calls);
    check(calls == 0, "view failed to retain its owner");
    {
        std::optional<mx::array> cached, snapshot, named_copy;
        for (uint32_t generation = 0; generation < 3; ++generation) {
            Parameters parameters{rows, cols, pitch, generation, seed};
            NSMutableArray *events = [NSMutableArray array]; const uint64_t start = now();
            auto event = [&](NSString *name, uint64_t begin) {
                [events addObject:@{@"name": name, @"start_ns": @(begin - start), @"duration_ns": @(now() - begin),
                                    @"clock_domain": @"same-process-uptime-raw-host"}];
            };
            uint64_t t = now(); mx::synchronize(); event(@"mlx-drain-before-external-write", t);
            t = now(); produce(queue, pipeline, producer_buffer, parameters, gpu_pending); event(@"external-metal-producer-complete", t);
            t = now(); auto output = consume(*live); output.eval(); mx::synchronize(); event(@"new-mlx-consumer-complete", t);
            const uint64_t boundary = now() - start;
            auto result = metrics(output, rows, cols, generation, seed);
            check(IOSurfaceLock(surface, 0, nullptr) == kIOReturnSuccess, "surface verification lock failed");
            std::vector<_Float16> logical(size_t(rows) * cols);
            for (size_t i = 0; i < logical.size(); ++i) {
                std::memcpy(&logical[i], static_cast<char *>(base) + (i / cols * pitch + i % cols) * 2, 2);
                check(double(logical[i]) == source(i, generation, seed), "full actual producer data differs");
            }
            const bool guard = tail(surface, payload);
            check(IOSurfaceUnlock(surface, 0, nullptr) == kIOReturnSuccess, "surface verification unlock failed");
            NSDictionary *sample = @{@"generation": @(generation), @"input_sha256": sha(logical.data(), logical.size() * 2),
                @"consumer": result, @"actual_alias": alias(*live, surface, device), @"tail_guard_pass": @(guard),
                @"owner_deleter_calls": @(calls), @"owned_producer_completed": @YES, @"mlx_stream_drained": @YES,
                @"host_events": events, @"diagnostic_boundary_ns": @(boundary), @"formal_performance_score": @NO};
            [record[@"samples"] addObject:sample]; save(report, path);
            check([result[@"mismatched_elements"] unsignedLongLongValue] == 0 && guard && calls == 0, "full MLX consumer/lifetime/guard gate failed");
            if (generation == 0) {
                cached.emplace(output);
                named_copy.emplace(mx::copy(*live)); named_copy->eval(); mx::synchronize();
                record[@"same_dtype_copy_alias"] = alias(*named_copy, surface, device);
                // MLX 0.32.0 Copy::eval is copy_shared_buffer, NOT a physical snapshot.
                // A different-dtype materialization is the separate-backing control.
                snapshot.emplace(mx::astype(*live, mx::float32, true)); snapshot->eval(); mx::synchronize();
                record[@"snapshot_kind"] = @"explicit-FP16-to-FP32-materialization";
                record[@"snapshot_has_distinct_backing"] = @(snapshot->buffer().raw_ptr() != base);
                check([record[@"snapshot_has_distinct_backing"] boolValue], "different-dtype materialization did not create distinct backing");
            }
            if (generation == 1) {
                // Evaluated arrays are not invalidated by an external writer. Never label these as refreshed.
                auto copy_consumer = consume(*named_copy);
                record[@"new_consumer_of_same_dtype_copy"] = metrics(copy_consumer, rows, cols, generation, seed);
                check([record[@"new_consumer_of_same_dtype_copy"][@"mismatched_elements"] unsignedLongLongValue] == 0,
                      "new consumer of aliasing same-dtype copy did not observe external write");
                auto copied_consumer = consume(*snapshot);
                record[@"stale_controls"] = @{@"cached_evaluated_output": metrics(*cached, rows, cols, generation, seed),
                    @"new_consumer_of_materialized_snapshot": metrics(copied_consumer, rows, cols, generation, seed),
                    @"actual_cached_generation": @0, @"expected_generation": @1, @"used_for_speed_ratio": @NO};
                for (NSString *key in @[@"cached_evaluated_output", @"new_consumer_of_materialized_snapshot"])
                    check([record[@"stale_controls"][key][@"mismatched_elements"] unsignedLongLongValue] == size_t(rows) * cols, "stale/copy control was not rejected");
                if (packed) {
                    auto wrong = consume(*packed); auto wrong_result = metrics(wrong, rows, cols, generation, seed);
                    record[@"packed_stride_control"] = wrong_result;
                    check([wrong_result[@"mismatched_elements"] unsignedLongLongValue] > 0, "incorrect packed view did not fail");
                }
            }
        }
    }
    mx::synchronize(); record[@"deleter_calls_before_last_view_release"] = @(calls);
    check(calls == 0, "owner released before last view");
    packed.reset(); live.reset(); mx::synchronize();
    record[@"deleter_calls_after_last_view_release"] = @(calls);
    record[@"deleter_pointer_matches_surface"] = @(deleter_pointer_matches);
    record[@"released_after_owned_completions"] = @YES;
    check(calls == 1 && deleter_pointer_matches, "owner deleter did not run exactly once after completion");
    record[@"status"] = @"actual-adoption-and-lifetime-validated"; save(report, path);
}
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 5) { std::fprintf(stderr, "usage: mlx-adoption padded|port seed new-report.json reserved-zero\n"); return 2; }
        NSMutableDictionary *report = [@{@"schema": @"tc-mlx-adoption-native-v1", @"status": @"starting", @"errors": [NSMutableArray array],
            @"records": [NSMutableArray array], @"synthetic": @YES, @"real_qwen": @NO, @"production_qualified": @NO,
            @"formal_performance_score": @NO, @"physical_overlap_qualified": @NO, @"runtime_internal_copy_behavior": @"unknown"} mutableCopy];
        IOSurfaceRef surface = nullptr; bool pending = false, success = false;
        @try { try {
            std::string kind = argv[1]; const auto parsed_seed = std::stoul(argv[2]);
            check((kind == "padded" || kind == "port") && parsed_seed <= 10000 && std::string(argv[4]) == "0", "bounded case/seed required");
            const uint32_t seed = uint32_t(parsed_seed), rows = kind == "padded" ? 32 : 1024;
            const uint32_t cols = kind == "padded" ? 33 : 4096, pitch = kind == "padded" ? 64 : 4096;
            const size_t payload = size_t(rows) * pitch * 2, bytes = rounded_allocation(payload + getpagesize());
            report[@"case"] = [NSString stringWithUTF8String:kind.c_str()]; report[@"seed"] = @(seed);
            report[@"logical_shape"] = @[@(rows), @(cols)]; report[@"pitch_elements"] = @(pitch); report[@"payload_bytes"] = @(payload);
            report[@"page_bytes"] = @(getpagesize()); report[@"mlx_version"] = [NSString stringWithUTF8String:mx::version()];
            mx::set_default_device(mx::Device::gpu); check(mx::is_available(mx::Device::gpu), "MLX GPU unavailable");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice(); check(device && device.hasUnifiedMemory, "unified-memory Metal device required");
            report[@"gpu_name"] = device.name;
            const auto &info = mx::device_info(); auto architecture = info.find("architecture");
            report[@"mlx_architecture"] = architecture == info.end() ? (id)NSNull.null
                : [NSString stringWithUTF8String:std::get<std::string>(architecture->second).c_str()];
            surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
                (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes), (id)kIOSurfaceAllocSize: @(bytes)});
            check(surface && IOSurfaceGetAllocSize(surface) >= bytes && IOSurfaceGetAllocSize(surface) % getpagesize() == 0 &&
                  reinterpret_cast<uintptr_t>(IOSurfaceGetBaseAddress(surface)) % getpagesize() == 0, "page-aligned owned mapping required");
            report[@"allocation_bytes"] = @(IOSurfaceGetAllocSize(surface));
            id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(surface) length:IOSurfaceGetAllocSize(surface)
                options:MTLResourceStorageModeShared deallocator:nil];
            check(buffer && buffer.contents == IOSurfaceGetBaseAddress(surface), "external Metal buffer did not alias surface");
            id<MTLCommandQueue> queue = [device newCommandQueue]; NSError *error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shader] options:nil error:&error];
            check(library && queue, "original producer shader/queue unavailable");
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"produce"] error:&error];
            check(pipeline, "producer pipeline unavailable");
            for (NSString *mode in @[@"raw-pointer", @"allocator-buffer"]) {
                NSMutableDictionary *record = [NSMutableDictionary dictionary]; [report[@"records"] addObject:record];
                exercise(mode, rows, cols, pitch, seed, surface, device, queue, pipeline, buffer, pending, record, argv[3], report);
            }
            mx::synchronize(); success = true; report[@"status"] = @"actual-adoption-and-lifetime-validated";
        } catch (const std::exception &error) { report[@"status"] = @"failed"; [report[@"errors"] addObject:[NSString stringWithUTF8String:error.what()]]; }
        } @catch (NSException *error) { report[@"status"] = @"objc-exception"; [report[@"errors"] addObject:error.name]; }
        // On any failure the original CF mapping is retained until this owned child exits.
        if (success && !pending && surface) CFRelease(surface);
        report[@"surface_released_after_owned_completions"] = @(success && !pending);
        report[@"mapping_quarantined_until_child_exit"] = @(!success || pending);
        report[@"mapping_reuse_after_failure"] = @NO; save(report, argv[3]);
        std::puts([report[@"status"] UTF8String]); return success && !pending ? 0 : 1;
    }
}
