// Original bounded Metal -> private ANE -> Metal backing/lifetime experiment.
// No security changes, third-party implementation, PMU, async API or global drain.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <unistd.h>

@interface NSObject (TCSharedSurfaceResearch)
+ (id)sharedConnection;
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices
                outputs:(NSArray *)outputs outputIndices:(NSArray *)outputIndices
              perfStats:(NSArray *)stats procedureIndex:(NSNumber *)procedure;
- (BOOL)compileModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)loadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary *)options request:(id)request qos:(unsigned)qos error:(NSError **)error;
- (BOOL)validate;
- (NSDictionary *)modelAttributes;
- (NSArray *)inputArray;
- (NSArray *)outputArray;
- (IOSurfaceRef)ioSurface;
@end

namespace {
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
uint64_t now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
NSString *sha(const void *data, size_t size) {
    check(size <= UINT32_MAX, "bounded digest required");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(data, CC_LONG(size), digest);
    NSMutableString *result = [NSMutableString string];
    for (auto byte : digest) [result appendFormat:@"%02x", byte];
    return result;
}
void save(NSDictionary *report, const char *path) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    check(data && [data writeToFile:[NSString stringWithUTF8String:path] options:NSDataWritingAtomic error:&error], "report write failed");
}
void guard_method(Class cls, NSString *name, bool meta, NSUInteger count, char kind, NSMutableDictionary *signatures) {
    Method method = meta ? class_getClassMethod(cls, NSSelectorFromString(name)) : class_getInstanceMethod(cls, NSSelectorFromString(name));
    check(method, "required private selector absent; no alternative ABI guessed");
    auto signature = [NSMethodSignature signatureWithObjCTypes:method_getTypeEncoding(method)];
    check(signature.numberOfArguments == count, "private argument count changed");
    const char actual = signature.methodReturnType[0];
    check(actual == kind || (kind == 'B' && actual == 'c'), "private return ABI changed");
    for (NSUInteger i = 2; i < count; ++i) {
        const char expected = [name containsString:@"qos:"] && i == count - 2 ? 'I'
            : [name containsString:@"qos:"] && i == count - 1 ? '^'
            : [name isEqual:@"objectWithIOSurface:"] ? '^' : '@';
        check([signature getArgumentTypeAtIndex:i][0] == expected, "private argument ABI changed");
    }
    signatures[[NSString stringWithFormat:@"%@.%@", NSStringFromClass(cls), name]] =
        [NSString stringWithUTF8String:method_getTypeEncoding(method)];
}
struct Layout { uint32_t channels, width, row, plane, total; };
uint32_t number(NSDictionary *value, NSString *key) {
    id item = value[key];
    check([item isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)item) != CFBooleanGetTypeID(), "numeric layout field required");
    const long long result = [item longLongValue];
    check(result > 0 && result <= 8 * 1024 * 1024, "layout value outside cap");
    return uint32_t(result);
}
Layout layout(NSDictionary *value, uint32_t channels, uint32_t width) {
    check(number(value, @"Batches") == 1 && number(value, @"Height") == 1 &&
          number(value, @"Channels") == channels && number(value, @"Width") == width, "unexpected physical NCHW shape");
    check([value[@"Type"] isEqual:@"Float16"], "explicit Float16 port required");
    if (value[@"Offset"]) check([value[@"Offset"] unsignedLongLongValue] == 0, "nonzero offset unsupported");
    if (value[@"Depth"]) check([value[@"Depth"] unsignedLongLongValue] == 1, "depth unsupported");
    if (value[@"PlaneCount"]) check([value[@"PlaneCount"] unsignedLongLongValue] == channels, "plane count differs");
    Layout result{channels, width, number(value, @"RowStride"), number(value, @"PlaneStride"), number(value, @"BatchStride")};
    check(result.row >= width * 2 && result.plane >= result.row && result.total >= channels * result.plane, "invalid byte pitch");
    return result;
}
NSDictionary *descriptor(NSDictionary *value) {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSString *key in @[@"Batches", @"Channels", @"Height", @"Width", @"Type", @"Offset", @"Depth", @"PlaneCount", @"RowStride", @"PlaneStride", @"BatchStride"])
        if (value[key]) result[key] = value[key];
    return result;
}
size_t rounded(size_t value) { const size_t page = size_t(getpagesize()); return (value + page - 1) / page * page; }
IOSurfaceRef make_surface(size_t payload) {
    const size_t bytes = rounded(payload + size_t(getpagesize()));
    check(bytes <= 16 * 1024 * 1024, "surface allocation cap");
    auto result = IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1, (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes), (id)kIOSurfaceAllocSize: @(bytes)});
    check(result && IOSurfaceGetAllocSize(result) >= bytes && IOSurfaceGetAllocSize(result) % getpagesize() == 0 &&
          reinterpret_cast<uintptr_t>(IOSurfaceGetBaseAddress(result)) % getpagesize() == 0, "page-aligned owned surface required");
    return result;
}
void lock(IOSurfaceRef surface) { check(IOSurfaceLock(surface, 0, nullptr) == kIOReturnSuccess, "surface CPU lock failed"); }
void unlock(IOSurfaceRef surface) { check(IOSurfaceUnlock(surface, 0, nullptr) == kIOReturnSuccess, "surface CPU unlock failed"); }
void reset(void *base, size_t payload, size_t bytes, unsigned char fill) {
    std::memset(base, fill, payload); std::memset(static_cast<char *>(base) + payload, 0x6b, bytes - payload);
}
bool tail(const void *base, size_t payload, size_t bytes) {
    const auto *data = static_cast<const unsigned char *>(base);
    return std::all_of(data + payload, data + bytes, [](unsigned char value) { return value == 0x6b; });
}
id<MTLBuffer> wrap(id<MTLDevice> device, IOSurfaceRef surface) {
    id<MTLBuffer> result = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(surface)
        length:IOSurfaceGetAllocSize(surface) options:MTLResourceStorageModeShared deallocator:nil];
    check(result && result.contents == IOSurfaceGetBaseAddress(surface) && result.storageMode == MTLStorageModeShared,
          "Metal did not alias the intended surface");
    return result;
}
NSDictionary *identity(id request, IOSurfaceRef input, IOSurfaceRef output, id<MTLBuffer> in_buffer, id<MTLBuffer> out_buffer) {
    NSArray *ins = [request inputArray], *outs = [request outputArray];
    check([ins isKindOfClass:NSArray.class] && [outs isKindOfClass:NSArray.class] && ins.count == 1 && outs.count == 1, "request ports changed");
    IOSurfaceRef actual_in = [ins[0] ioSurface], actual_out = [outs[0] ioSurface];
    check(actual_in && actual_out, "request did not expose surfaces");
    NSDictionary *result = @{@"request_input_surface_matches": @(IOSurfaceGetID(input) == IOSurfaceGetID(actual_in)),
        @"request_output_surface_matches": @(IOSurfaceGetID(output) == IOSurfaceGetID(actual_out)),
        @"input_metal_contents_matches": @(in_buffer.contents == IOSurfaceGetBaseAddress(actual_in)),
        @"output_metal_contents_matches": @(out_buffer.contents == IOSurfaceGetBaseAddress(actual_out)),
        @"input_output_distinct": @(IOSurfaceGetID(actual_in) != IOSurfaceGetID(actual_out))};
    for (NSNumber *value in result.allValues) check(value.boolValue, "actual request/Metal backing differs");
    return result;
}
struct Parameters { uint32_t channels, width, plane_elements, generation, seed; };
const char *shader = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct P { uint channels, width, plane, generation, seed; };
kernel void produce(device half *x [[buffer(0)]], constant P &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.channels * p.width) return;
    uint c = i / p.width, w = i % p.width;
    x[c * p.plane + w] = half((int((c * 7 + w * 3 + p.generation * 5 + p.seed * 11) % 37) - 18) / 16.0f);
}
kernel void consume(device const half *y [[buffer(0)]], constant P &p [[buffer(1)]],
                    device float *result [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.channels * p.width) return;
    result[i] = 3.0f * float(y[(i / p.width) * p.plane + i % p.width]) - 0.5f;
}
)METAL";
void dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, id<MTLBuffer> buffer,
              const Parameters &parameters, id<MTLBuffer> result, bool &gpu_pending) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    check(command && encoder, "Metal encoder allocation failed");
    [encoder setComputePipelineState:pipeline]; [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
    if (result) [encoder setBuffer:result offset:0 atIndex:2];
    const NSUInteger threads = std::min(NSUInteger(128), pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(parameters.channels * parameters.width, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding]; gpu_pending = true; [command commit]; [command waitUntilCompleted];
    check(command.status == MTLCommandBufferStatusCompleted, "owned GPU command did not complete successfully");
    gpu_pending = false;
}
double source(uint32_t c, uint32_t x, uint32_t generation, uint32_t seed) {
    return (int((c * 7 + x * 3 + generation * 5 + seed * 11) % 37) - 18) / 16.;
}
NSDictionary *consumer_metrics(id<MTLBuffer> buffer, uint32_t channels, uint32_t width, uint32_t generation, uint32_t seed) {
    const auto *data = static_cast<const float *>(buffer.contents);
    size_t mismatches = 0, nonfinite = 0; double maximum = 0;
    for (uint32_t c = 0; c < channels; ++c) for (uint32_t x = 0; x < width; ++x) {
        const double actual = data[c * width + x], expected = 6 * source(c, x, generation, seed) + .25;
        nonfinite += !std::isfinite(actual); mismatches += !std::isfinite(actual) || actual != expected;
        if (std::isfinite(actual)) maximum = std::max(maximum, std::abs(actual - expected));
    }
    return @{@"mismatched_elements": @(mismatches), @"nonfinite_elements": @(nonfinite), @"maximum_finite_abs_error": @(maximum),
             @"consumer_sha256": sha(buffer.contents, size_t(channels) * width * 4)};
}
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 8) { std::fprintf(stderr, "usage: shared-probe model-dir channels width seed repeats new-report.json abi-only\n"); return 2; }
        NSMutableDictionary *report = [@{@"schema": @"tc-private-shared-native-v1", @"status": @"starting",
            @"production_qualified": @NO, @"runtime_internal_copy_behavior": @"unknown", @"physical_overlap_qualified": @NO,
            @"formal_performance_score": @NO, @"samples": [NSMutableArray array], @"errors": [NSMutableArray array]} mutableCopy];
        id client = nil, model = nil;
        bool loaded = false, unsafe_failure = false, gpu_pending = false;
        std::vector<IOSurfaceRef> surfaces;
        NSMutableArray *metal_owners = [NSMutableArray array];
        @try { try {
            uint32_t channels = uint32_t(std::stoul(argv[2])), width = uint32_t(std::stoul(argv[3]));
            uint32_t seed = uint32_t(std::stoul(argv[4])), repeats = uint32_t(std::stoul(argv[5]));
            check((channels == 32 || channels == 64) && (width == 33 || width == 65) && seed <= 10000 &&
                  (repeats == 8 || repeats == 24), "bounded probe arguments required");
            check(std::string(argv[7]) == "0", "no alternate ABI mode supported");
            report[@"shape_nchw"] = @[@1, @(channels), @1, @(width)]; report[@"seed"] = @(seed); report[@"measured_groups"] = @(repeats);
            check(dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL), "private framework unavailable");
            Class cc = NSClassFromString(@"_ANEClient"), mc = NSClassFromString(@"_ANEModel");
            Class ic = NSClassFromString(@"_ANEIOSurfaceObject"), rc = NSClassFromString(@"_ANERequest");
            NSMutableDictionary *signatures = [NSMutableDictionary dictionary]; report[@"private_method_signatures"] = signatures;
            guard_method(cc, @"sharedConnection", true, 2, '@', signatures); guard_method(mc, @"modelAtURL:key:", true, 4, '@', signatures);
            guard_method(ic, @"objectWithIOSurface:", true, 3, '@', signatures); guard_method(ic, @"ioSurface", false, 2, '^', signatures);
            guard_method(rc, @"requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:", true, 8, '@', signatures);
            for (NSString *name in @[@"inputArray", @"outputArray"]) guard_method(rc, name, false, 2, '@', signatures);
            guard_method(rc, @"validate", false, 2, 'B', signatures); guard_method(mc, @"modelAttributes", false, 2, '@', signatures);
            for (NSString *name in @[@"compileModel:options:qos:error:", @"loadModel:options:qos:error:", @"unloadModel:options:qos:error:"])
                guard_method(cc, name, false, 6, 'B', signatures);
            guard_method(cc, @"evaluateWithModel:options:request:qos:error:", false, 7, 'B', signatures);
            client = [cc sharedConnection]; model = [mc modelAtURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] key:@"net"];
            check(client && model, "private client/model construction failed");
            NSError *error = nil; NSDictionary *options = @{@"kANEFModelType": @"kANEFModelMIL", @"kANEFPerformanceStatsMask": @0};
            report[@"stage"] = @"compile"; uint64_t begin = now();
            check([client compileModel:model options:options qos:25 error:&error], "MIL compile failed; no fallback");
            report[@"compile_wall_ns"] = @(now() - begin); report[@"stage"] = @"load"; begin = now();
            loaded = [client loadModel:model options:options qos:25 error:&error];
            check(loaded, "MIL load failed; no fallback"); report[@"load_wall_ns"] = @(now() - begin);
            NSArray *networks = [model modelAttributes][@"NetworkStatusList"];
            check([networks isKindOfClass:NSArray.class] && networks.count == 1, "one network required");
            NSArray *inputs = networks[0][@"LiveInputList"], *outputs = networks[0][@"LiveOutputList"];
            check(inputs.count == 1 && outputs.count == 1, "one input and output required");
            Layout in = layout(inputs[0], channels, width), out = layout(outputs[0], channels, width);
            report[@"input_descriptor"] = descriptor(inputs[0]); report[@"output_descriptor"] = descriptor(outputs[0]);
            check(in.plane % 2 == 0 && out.plane % 2 == 0 && out.plane > width * 2, "padded FP16 plane required for stride negative");
            IOSurfaceRef input = make_surface(in.total); surfaces.push_back(input);
            IOSurfaceRef output = make_surface(out.total); surfaces.push_back(output);
            id<MTLDevice> device = MTLCreateSystemDefaultDevice(); check(device && device.hasUnifiedMemory, "unified-memory Metal device required");
            report[@"gpu_name"] = device.name; report[@"page_bytes"] = @(getpagesize());
            id<MTLCommandQueue> queue = [device newCommandQueue]; check(queue, "Metal queue unavailable");
            id<MTLBuffer> in_buffer = wrap(device, input), out_buffer = wrap(device, output);
            id<MTLBuffer> staging_in = [device newBufferWithLength:in_buffer.length options:MTLResourceStorageModeShared];
            id<MTLBuffer> staging_out = [device newBufferWithLength:out_buffer.length options:MTLResourceStorageModeShared];
            const size_t result_bytes = size_t(channels) * width * 4;
            id<MTLBuffer> result = [device newBufferWithLength:rounded(result_bytes + getpagesize()) options:MTLResourceStorageModeShared];
            check(staging_in && staging_out && result, "bounded control buffers unavailable");
            [metal_owners addObjectsFromArray:@[in_buffer, out_buffer, staging_in, staging_out, result]];
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shader] options:nil error:&error];
            check(library, "original Metal shader compile failed");
            id<MTLComputePipelineState> producer = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"produce"] error:&error];
            id<MTLComputePipelineState> consumer = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"consume"] error:&error];
            check(producer && consumer, "Metal pipeline unavailable");
            id input_object = [ic objectWithIOSurface:input], output_object = [ic objectWithIOSurface:output];
            id request = [rc requestWithInputs:@[input_object] inputIndices:@[@0] outputs:@[output_object] outputIndices:@[@0] perfStats:nil procedureIndex:@0];
            check(request && [request validate], "private request invalid");
            report[@"initial_identity"] = identity(request, input, output, in_buffer, out_buffer);
            report[@"surface_allocation_bytes"] = @[@(in_buffer.length), @(out_buffer.length)];
            NSDictionary *eval_options = @{@"kANEFPerformanceStatsMask": @0}; report[@"stage"] = @"execute";
            for (uint32_t group = 0; group < repeats + 3; ++group) {
                NSString *phase = group == 0 ? @"first" : group < 3 ? @"warmup" : @"measured";
                const uint32_t trial = group == 0 ? 0 : group < 3 ? group - 1 : group - 3;
                for (uint32_t position = 0; position < 2; ++position) {
                    const bool copied = (position + group) % 2 == 1;
                    lock(input); reset(IOSurfaceGetBaseAddress(input), in.total, in_buffer.length, 0xa5); unlock(input);
                    lock(output); reset(IOSurfaceGetBaseAddress(output), out.total, out_buffer.length, 0xff); unlock(output);
                    reset(staging_in.contents, in.total, staging_in.length, 0xa5); reset(staging_out.contents, out.total, staging_out.length, 0xff);
                    reset(result.contents, result_bytes, result.length, 0xff);
                    Parameters ip{channels, width, in.plane / 2, group, seed}, op{channels, width, out.plane / 2, group, seed};
                    id<MTLBuffer> produced = copied ? staging_in : in_buffer, consumed = copied ? staging_out : out_buffer;
                    NSMutableArray *events = [NSMutableArray array]; const uint64_t start = now();
                    auto event = [&](NSString *name, uint64_t t) {
                        [events addObject:@{@"name": name, @"start_ns": @(t - start), @"duration_ns": @(now() - t), @"clock_domain": @"same-process-uptime-raw-host"}];
                    };
                    uint64_t t = now(); dispatch(queue, producer, produced, ip, nil, gpu_pending); event(@"gpu-producer-complete", t);
                    if (copied) { t = now(); lock(input); std::memcpy(IOSurfaceGetBaseAddress(input), staging_in.contents, in.total); unlock(input); event(@"cpu-input-feature-copy", t); }
                    NSDictionary *before = identity(request, input, output, in_buffer, out_buffer);
                    t = now(); unsafe_failure = true;
                    const bool ok = [client evaluateWithModel:model options:eval_options request:request qos:25 error:&error];
                    check(ok, "synchronous evaluate failed; backing quarantined"); unsafe_failure = false; event(@"private-evaluate-returned", t);
                    NSDictionary *after = identity(request, input, output, in_buffer, out_buffer);
                    if (copied) { t = now(); lock(output); std::memcpy(staging_out.contents, IOSurfaceGetBaseAddress(output), out.total); unlock(output); event(@"cpu-output-feature-copy", t); }
                    t = now(); dispatch(queue, consumer, consumed, op, result, gpu_pending); event(@"gpu-consumer-complete", t);
                    const uint64_t wall = now() - start;
                    // Validation/readback are AFTER the measured producer-to-consumer boundary.
                    lock(input); lock(output);
                    std::vector<_Float16> actual_input(size_t(channels) * width), actual_output(actual_input.size());
                    for (uint32_t c = 0; c < channels; ++c) for (uint32_t x = 0; x < width; ++x) {
                        std::memcpy(&actual_input[c * width + x], static_cast<char *>(in_buffer.contents) + c * in.plane + x * 2, 2);
                        std::memcpy(&actual_output[c * width + x], static_cast<char *>(out_buffer.contents) + c * out.plane + x * 2, 2);
                        const double expected = source(c, x, group, seed);
                        check(double(actual_input[c * width + x]) == expected && double(actual_output[c * width + x]) == 2 * expected + .25, "actual logical surface value differs from exact independent formula");
                    }
                    const bool guards = tail(in_buffer.contents, in.total, in_buffer.length) && tail(out_buffer.contents, out.total, out_buffer.length) &&
                        tail(staging_in.contents, in.total, staging_in.length) && tail(staging_out.contents, out.total, staging_out.length) && tail(result.contents, result_bytes, result.length);
                    unlock(output); unlock(input);
                    auto metrics = consumer_metrics(result, channels, width, group, seed);
                    NSMutableDictionary *sample = [@{@"phase": phase, @"trial": @(trial), @"arm": copied ? @"copy" : @"shared", @"generation": @(group),
                        @"wall_ns": @(wall), @"host_events": events, @"external_feature_copy_count": copied ? @2 : @0,
                        @"external_feature_copy_bytes": copied ? @(in.total + out.total) : @0,
                        @"input_sha256": sha(actual_input.data(), actual_input.size() * 2), @"ane_output_sha256": sha(actual_output.data(), actual_output.size() * 2),
                        @"consumer": metrics, @"identity_before": before, @"identity_after": after,
                        @"tail_guards_pass": @(guards), @"gpu_completions_checked": @YES, @"output_poison_overwritten": @([metrics[@"nonfinite_elements"] unsignedLongLongValue] == 0),
                        @"runtime_internal_copy_behavior": @"unknown", @"physical_overlap_qualified": @NO} mutableCopy];
                    [report[@"samples"] addObject:sample]; save(report, argv[6]);
                    check(guards && [metrics[@"mismatched_elements"] unsignedLongLongValue] == 0, "consumer or tail gate failed");
                }
            }
            Parameters correct{channels, width, out.plane / 2, repeats + 2, seed};
            reset(result.contents, result_bytes, result.length, 0xff); dispatch(queue, consumer, out_buffer, correct, result, gpu_pending);
            auto stale = consumer_metrics(result, channels, width, repeats + 3, seed);
            Parameters wrong{channels, width, width, repeats + 2, seed};
            reset(result.contents, result_bytes, result.length, 0xff); dispatch(queue, consumer, out_buffer, wrong, result, gpu_pending);
            auto stride = consumer_metrics(result, channels, width, repeats + 2, seed);
            report[@"negative_controls"] = @{@"stale_generation": stale, @"packed_stride": stride,
                @"last_actual_generation": @(repeats + 2), @"stale_expected_generation": @(repeats + 3),
                @"packed_plane_bytes": @(width * 2), @"actual_plane_bytes": @(out.plane),
                @"consumer_tail_guard_pass": @(tail(result.contents, result_bytes, result.length)), @"used_for_speed_ratio": @NO};
            check([stale[@"mismatched_elements"] unsignedLongLongValue] > 0 && [stride[@"mismatched_elements"] unsignedLongLongValue] > 0 &&
                  tail(result.contents, result_bytes, result.length), "negative control did not reject stale/packed interpretation");
            report[@"final_identity"] = identity(request, input, output, in_buffer, out_buffer);
            report[@"status"] = @"backing-and-consumer-validated";
        } catch (const std::exception &error) {
            report[@"status"] = @"failed"; [report[@"errors"] addObject:[NSString stringWithUTF8String:error.what()]];
        }} @catch (NSException *error) { report[@"status"] = @"private-api-exception"; [report[@"errors"] addObject:error.name]; }
        bool unloaded = !loaded;
        @try { if (loaded) { NSError *error = nil; unloaded = [client unloadModel:model options:@{} qos:25 error:&error]; } }
        @catch (NSException *error) { unloaded = false; [report[@"errors"] addObject:error.name]; }
        report[@"unload_succeeded"] = @(unloaded);
        const bool release = unloaded && !unsafe_failure && !gpu_pending;
        if (release) { [metal_owners removeAllObjects]; for (auto surface : surfaces) CFRelease(surface); }
        report[@"mapping_released_after_owned_completion_and_unload"] = @(release);
        report[@"mapping_quarantined_until_child_exit"] = @(!release);
        report[@"mapping_reuse_after_failure"] = @NO; save(report, argv[6]);
        std::puts([report[@"status"] UTF8String]);
        return [report[@"status"] isEqual:@"backing-and-consumer-validated"] && release ? 0 : 1;
    }
}
