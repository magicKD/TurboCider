// Independent bounded private-ANE diagnostic. No upstream implementation copied.
// No boot arguments, entitlements, kernel interfaces, global logs or cache deletion.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <IOKit/IOKitLib.h>
#import <CommonCrypto/CommonDigest.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Private selectors are version-bound, checked before use, and never shipping APIs.
@interface NSObject (TCPrivatePmuResearch)
+ (id)sharedConnection;
+ (unsigned)driverMaskForANEFMask:(unsigned)mask;
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
+ (id)objectWithIOSurface:(id)object statType:(NSInteger)kind;
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices
                outputs:(NSArray *)outputs outputIndices:(NSArray *)outputIndices
              perfStats:(NSArray *)stats procedureIndex:(NSNumber *)procedure;
- (BOOL)compileModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)loadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary *)options request:(id)request qos:(unsigned)qos error:(NSError **)error;
- (BOOL)validate;
- (NSDictionary *)modelAttributes;
- (id)perfStats;
- (NSData *)perfCounterData;
- (unsigned long long)hwExecutionTime;
- (NSString *)stringForPerfCounter:(int32_t)index;
@end

namespace {
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
uint64_t now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

NSString *hash_bytes(const void *data, size_t size) {
    check(size <= UINT32_MAX, "hash input too large");
    unsigned char value[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data, CC_LONG(size), value);
    NSMutableString *result = [NSMutableString string];
    for (unsigned char byte : value) [result appendFormat:@"%02x", byte];
    return result;
}

NSDictionary *host_ane() {
    for (const char *name : {"H11ANEIn", "AppleH16ANEInterface"}) {
        auto service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching(name));
        if (!service) continue;
        CFTypeRef property = IORegistryEntryCreateCFProperty(service, CFSTR("DeviceProperties"), kCFAllocatorDefault, 0);
        IOObjectRelease(service);
        NSDictionary *properties = CFBridgingRelease(property);
        if (![properties isKindOfClass:NSDictionary.class]) continue;
        return @{@"architecture": properties[@"ANEDevicePropertyTypeANEArchitectureTypeStr"] ?: NSNull.null,
                 @"core_count_reported": properties[@"ANEDevicePropertyNumANECores"] ?: NSNull.null};
    }
    return @{@"architecture": NSNull.null, @"core_count_reported": NSNull.null};
}

void method_guard(Class cls, NSString *name, bool class_method, NSUInteger count, char return_kind,
                  NSMutableDictionary *identities) {
    SEL selector = NSSelectorFromString(name);
    Method method = class_method ? class_getClassMethod(cls, selector) : class_getInstanceMethod(cls, selector);
    check(method != nullptr, "required private selector missing");
    NSMethodSignature *sig = [NSMethodSignature signatureWithObjCTypes:method_getTypeEncoding(method)];
    check(sig.numberOfArguments == count, "private selector argument count changed");
    const char actual = sig.methodReturnType[0];
    check(actual == return_kind || (return_kind == 'B' && actual == 'c'), "private selector return type changed");
    if ([name containsString:@"qos:"]) {
        const NSUInteger qos_index = [name hasPrefix:@"evaluate"] ? 5 : 4;
        check([sig getArgumentTypeAtIndex:qos_index][0] == 'I', "private QoS ABI changed");
        check([sig getArgumentTypeAtIndex:qos_index + 1][0] == '^', "private error pointer ABI changed");
    }
    identities[[NSString stringWithFormat:@"%@.%@", NSStringFromClass(cls), name]] =
        [NSString stringWithUTF8String:method_getTypeEncoding(method)];
}

struct Layout {
    size_t channels, width, row, plane, total;
};

NSDictionary *selected_descriptor(NSDictionary *value) {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSString *key in @[@"Batches", @"Channels", @"Height", @"Width", @"PlaneCount", @"BatchStride",
                            @"PlaneStride", @"RowStride", @"Type", @"Offset", @"Depth"])
        result[key] = value[key] ?: NSNull.null;
    return result;
}

size_t number(NSDictionary *value, NSString *key) {
    id item = value[key];
    check([item isKindOfClass:NSNumber.class], "required numeric layout field missing");
    const long long signed_value = [item longLongValue];
    check(signed_value > 0 && signed_value <= 8 * 1024 * 1024, "layout field outside bounded positive range");
    return size_t(signed_value);
}

Layout layout(NSDictionary *value, size_t channels, size_t width) {
    check(number(value, @"Batches") == 1 && number(value, @"Height") == 1, "unexpected batch/height");
    check(number(value, @"Channels") == channels && number(value, @"Width") == width, "unexpected physical shape");
    NSString *type = value[@"Type"];
    check([type isKindOfClass:NSString.class] &&
          ([type.lowercaseString containsString:@"fp16"] || [type.lowercaseString containsString:@"float16"]),
          "physical port is not explicitly FP16");
    if (value[@"Depth"]) check([value[@"Depth"] unsignedLongLongValue] == 1, "unexpected depth");
    if (value[@"Offset"]) check([value[@"Offset"] unsignedLongLongValue] == 0, "nonzero port offset unsupported");
    size_t row = number(value, @"RowStride"), plane = number(value, @"PlaneStride");
    size_t total = number(value, @"BatchStride");
    check(row >= width * 2 && plane >= row && total >= channels * plane, "invalid physical byte pitch");
    return {channels, width, row, plane, total};
}

IOSurfaceRef surface(size_t bytes) {
    check(bytes > 0 && bytes <= 8 * 1024 * 1024, "surface cap exceeded");
    auto result = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes), (id)kIOSurfaceAllocSize: @(bytes)});
    check(result && IOSurfaceGetAllocSize(result) >= bytes && IOSurfaceGetBaseAddress(result), "surface allocation failed");
    return result;
}

void lock(IOSurfaceRef value) { check(IOSurfaceLock(value, 0, nullptr) == kIOReturnSuccess, "surface lock failed"); }
void unlock(IOSurfaceRef value) { check(IOSurfaceUnlock(value, 0, nullptr) == kIOReturnSuccess, "surface unlock failed"); }

NSDictionary *capture_stats(id stats) {
    if (!stats) return @{@"object_present": @NO, @"payload_bytes": NSNull.null,
                         @"raw_words": NSNull.null, @"hardware_time_raw": NSNull.null};
    NSData *data = nil;
    if ([stats respondsToSelector:@selector(perfCounterData)]) data = [[stats perfCounterData] copy];
    check(!data || [data isKindOfClass:NSData.class], "unexpected stats data type");
    check(!data || data.length <= 4096, "stats payload exceeds bounded buffer");
    NSMutableArray *words = [NSMutableArray array], *labels = [NSMutableArray array];
    if (data.length && data.length % 8 == 0) {
        for (NSUInteger index = 0; index < data.length / 8; ++index) {
            uint64_t word = 0;
            std::memcpy(&word, static_cast<const char *>(data.bytes) + index * 8, 8);
            [words addObject:@(word)];
            NSString *label = nil;
            if (index < 29 && [stats respondsToSelector:@selector(stringForPerfCounter:)])
                label = [stats stringForPerfCounter:int32_t(index)];
            [labels addObject:label ?: (id)NSNull.null];
        }
    }
    return @{@"object_present": @YES, @"payload_bytes": data ? @(data.length) : (id)NSNull.null,
             @"payload_sha256": data ? hash_bytes(data.bytes, data.length) : (id)NSNull.null,
             @"raw_words": words, @"labels_by_index": labels,
             @"hardware_time_raw": [stats respondsToSelector:@selector(hwExecutionTime)] ? @([stats hwExecutionTime]) : (id)NSNull.null,
             @"hardware_time_unit": NSNull.null, @"counter_units_qualified": @NO};
}

void save(NSMutableDictionary *report, const char *path) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    check(data != nil, "report encoding failed");
    check([data writeToFile:[NSString stringWithUTF8String:path] options:NSDataWritingAtomic error:&error], "report write failed");
}
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 8) { std::fprintf(stderr, "usage: pmu-probe owned-model-dir mask channels width new-report.json affine|conv compile-profiling\n"); return 2; }
        NSMutableDictionary *report = [@{@"schema": @"tc-private-ane-pmu-native-v1", @"status": @"starting",
            @"production_qualified": @NO, @"counter_semantics_qualified": @NO,
            @"physical_device_drain": @"not a global-device observation", @"samples": [NSMutableArray array],
            @"errors": [NSMutableArray array], @"host_ane": host_ane()} mutableCopy];
        id client = nil, model = nil;
        bool loaded = false;
        std::vector<IOSurfaceRef> surfaces;
        @try {
        try {
            const int mask = std::stoi(argv[2]);
            const size_t channels = std::stoul(argv[3]), width = std::stoul(argv[4]);
            const std::string kind = argv[6];
            check(kind == "affine" || kind == "conv", "unknown owned recipe");
            const bool compile_profiling = std::stoi(argv[7]) == 1;
            check((mask == 0 || mask == 2 || mask == 15) && channels >= 32 && channels <= 128 && channels % 32 == 0 &&
                  width >= 32 && width <= 512 && width % 32 == 0, "invalid bounded mask/shape");
            report[@"requested_mask"] = @(mask);
            report[@"compile_profiling_requested"] = @(compile_profiling);
            report[@"recipe_kind"] = [NSString stringWithUTF8String:kind.c_str()];
            void *framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL);
            check(framework != nullptr, "private framework unavailable");
            Class client_class = NSClassFromString(@"_ANEClient"), model_class = NSClassFromString(@"_ANEModel");
            Class io_class = NSClassFromString(@"_ANEIOSurfaceObject"), request_class = NSClassFromString(@"_ANERequest");
            Class stats_class = NSClassFromString(@"_ANEPerformanceStatsIOSurface");
            auto signatures = [NSMutableDictionary dictionary]; report[@"private_method_signatures"] = signatures;
            method_guard(client_class, @"sharedConnection", true, 2, '@', signatures);
            method_guard(model_class, @"modelAtURL:key:", true, 4, '@', signatures);
            for (NSString *selector in @[@"compileModel:options:qos:error:", @"loadModel:options:qos:error:", @"unloadModel:options:qos:error:"])
                method_guard(client_class, selector, false, 6, 'B', signatures);
            method_guard(client_class, @"evaluateWithModel:options:request:qos:error:", false, 7, 'B', signatures);
            method_guard(io_class, @"objectWithIOSurface:", true, 3, '@', signatures);
            method_guard(request_class, @"requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:", true, 8, '@', signatures);
            if (mask) method_guard(stats_class, @"objectWithIOSurface:statType:", true, 4, '@', signatures);
            NSMutableDictionary *stats_methods = [NSMutableDictionary dictionary];
            Class raw_stats_class = NSClassFromString(@"_ANEPerformanceStats");
            for (bool meta : {false, true}) {
                unsigned count = 0;
                Method *methods = class_copyMethodList(meta ? object_getClass(raw_stats_class) : raw_stats_class, &count);
                for (unsigned i = 0; i < count; ++i) {
                    NSString *name = NSStringFromSelector(method_getName(methods[i]));
                    if ([name.lowercaseString containsString:@"stat"] || [name.lowercaseString containsString:@"counter"] ||
                        [name.lowercaseString containsString:@"execution"] || [name.lowercaseString containsString:@"mask"])
                        stats_methods[[NSString stringWithFormat:@"%c%@", meta ? '+' : '-', name]] =
                            [NSString stringWithUTF8String:method_getTypeEncoding(methods[i])];
                }
                std::free(methods);
            }
            report[@"stats_selector_metadata_only"] = stats_methods;
            SEL mask_selector = @selector(driverMaskForANEFMask:);
            if ([raw_stats_class respondsToSelector:mask_selector]) {
                method_guard(raw_stats_class, @"driverMaskForANEFMask:", true, 3, 'I', signatures);
                NSMutableArray *translations = [NSMutableArray array];
                for (unsigned value : {0u, 2u, 15u})
                    [translations addObject:@{@"anef_mask": @(value), @"driver_mask_returned": @([raw_stats_class driverMaskForANEFMask:value])}];
                report[@"mask_translation_readonly"] = translations;
            }
            client = [client_class sharedConnection];
            model = [model_class modelAtURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] key:@"net"];
            check(client && model, "private model/client construction failed");
            NSMutableDictionary *options = [@{@"kANEFModelType": @"kANEFModelMIL", @"kANEFPerformanceStatsMask": @(mask)} mutableCopy];
            if (compile_profiling) options[@"enableProfiling"] = @YES;
            NSError *error = nil;
            report[@"stage"] = @"compile";
            const uint64_t compile_start = now();
            const bool compiled = [client compileModel:model options:options qos:25 error:&error];
            report[@"compile_wall_ns"] = @(now() - compile_start);
            if (!compiled) { report[@"native_error"] = @{@"domain": error.domain ?: @"unknown", @"code": @(error.code)}; }
            check(compiled, "private MIL compile failed; no fallback");
            report[@"stage"] = @"load";
            const uint64_t load_start = now();
            loaded = [client loadModel:model options:options qos:25 error:&error];
            report[@"load_wall_ns"] = @(now() - load_start);
            check(loaded, "private MIL load failed; no fallback");
            NSDictionary *attributes = [model modelAttributes];
            NSArray *networks = attributes[@"NetworkStatusList"];
            check([networks isKindOfClass:NSArray.class] && networks.count == 1, "unique network layout required");
            NSArray *inputs = networks[0][@"LiveInputList"], *outputs = networks[0][@"LiveOutputList"];
            check(inputs.count == 1 && outputs.count == 1, "single input/output required");
            report[@"input_descriptor"] = selected_descriptor(inputs[0]);
            report[@"output_descriptor"] = selected_descriptor(outputs[0]);
            report[@"stage"] = @"layout";
            Layout in = layout(inputs[0], channels, width), out = layout(outputs[0], channels, width);
            IOSurfaceRef input = surface(in.total), output = surface(out.total);
            surfaces.push_back(input); surfaces.push_back(output);
            id input_object = [io_class objectWithIOSurface:input], output_object = [io_class objectWithIOSurface:output];
            NSArray *perf = nil;
            IOSurfaceRef pmu = nullptr;
            if (mask) {
                pmu = surface(4096); surfaces.push_back(pmu);
                perf = @[[stats_class objectWithIOSurface:[io_class objectWithIOSurface:pmu] statType:2]];
            }
            id request = [request_class requestWithInputs:@[input_object] inputIndices:@[@0]
                outputs:@[output_object] outputIndices:@[@0] perfStats:perf procedureIndex:@0];
            check(request && [request validate], "private request validation failed");
            report[@"stage"] = @"evaluate";
            NSDictionary *eval_options = mask ? @{@"kANEFPerformanceStatsMask": @(mask), @"enableProfiling": @YES}
                                              : @{@"kANEFPerformanceStatsMask": @0};
            for (int trial = 0; trial < 9; ++trial) {
                const uint64_t full_start = now();
                std::vector<_Float16> source(channels * width);
                lock(input); lock(output);
                std::memset(IOSurfaceGetBaseAddress(input), 0, IOSurfaceGetAllocSize(input));
                std::memset(IOSurfaceGetBaseAddress(output), 0xff, IOSurfaceGetAllocSize(output));
                for (size_t c = 0; c < channels; ++c) for (size_t x = 0; x < width; ++x) {
                    const _Float16 value = _Float16((int((c * 7 + x * 3 + trial * 5) % 17) - 8) / 8.);
                    source[c * width + x] = value;
                    std::memcpy(static_cast<char *>(IOSurfaceGetBaseAddress(input)) + c * in.plane + x * 2, &value, 2);
                }
                unlock(output); unlock(input);
                if (pmu) { lock(pmu); std::memset(IOSurfaceGetBaseAddress(pmu), 0, IOSurfaceGetAllocSize(pmu)); unlock(pmu); }
                error = nil;
                const uint64_t begin = now();
                const bool ok = [client evaluateWithModel:model options:eval_options request:request qos:25 error:&error];
                const uint64_t wall = now() - begin;
                if (!ok) report[@"native_error"] = @{@"domain": error.domain ?: @"unknown", @"code": @(error.code)};
                check(ok, "synchronous evaluate returned failure; no buffers reused");
                lock(output);
                std::vector<_Float16> actual(channels * width);
                for (size_t c = 0; c < channels; ++c) for (size_t x = 0; x < width; ++x)
                    std::memcpy(&actual[c * width + x], static_cast<char *>(IOSurfaceGetBaseAddress(output)) + c * out.plane + x * 2, 2);
                unlock(output);
                const uint64_t full_wall = now() - full_start;
                double max_error = 0, squared_error = 0, energy = 0;
                bool finite = true;
                for (size_t i = 0; i < actual.size(); ++i) {
                    finite = finite && std::isfinite(double(actual[i]));
                    double expected = double(source[i]) * 2. + .25;
                    if (kind == "conv") {
                        expected = 0;
                        const size_t channel = i / width, x = i % width;
                        for (size_t input_channel = 0; input_channel < channels; ++input_channel)
                            expected += double(source[input_channel * width + x]) *
                                        (int((channel * 7 + input_channel * 3) % 11) - 5) / 64.;
                    }
                    const double difference = double(actual[i]) - expected;
                    max_error = std::max(max_error, std::abs(difference));
                    squared_error += difference * difference; energy += expected * expected;
                }
                const double relative_l2 = energy ? std::sqrt(squared_error / energy) : (squared_error == 0 ? 0 : 1);
                const bool passed = finite && (kind == "affine" ? max_error == 0 : max_error <= .001 && relative_l2 <= .001);
                NSMutableDictionary *sample = [@{@"trial": @(trial), @"warmup": @(trial < 3),
                    @"input_generation": @(trial), @"api_wall_ns": @(wall), @"host_prepare_call_readback_ns": @(full_wall),
                    @"input_sha256": hash_bytes(source.data(), source.size() * 2),
                    @"output_sha256": hash_bytes(actual.data(), actual.size() * 2),
                    @"all_outputs_finite": @(finite), @"maximum_abs_error": @(max_error),
                    @"relative_l2": @(relative_l2), @"numeric_pass": @(passed), @"stats": capture_stats([request perfStats])} mutableCopy];
                if (pmu) {
                    lock(pmu);
                    size_t nonzero = 0;
                    const auto *bytes = static_cast<const uint8_t *>(IOSurfaceGetBaseAddress(pmu));
                    for (size_t i = 0; i < 4096; ++i) nonzero += bytes[i] != 0;
                    sample[@"stats_surface"] = @{@"nonzero_bytes": @(nonzero), @"total_bytes": @4096,
                        @"sha256": hash_bytes(bytes, 4096), @"opaque_bytes_exported": @NO};
                    unlock(pmu);
                }
                [report[@"samples"] addObject:sample];
                save(report, argv[5]);
                check(passed, "independent full-output gate failed");
            }
            report[@"status"] = @"executed-and-numerically-validated";
        } catch (const std::exception &error) {
            report[@"status"] = @"failed";
            [report[@"errors"] addObject:[NSString stringWithUTF8String:error.what()]];
        }
        } @catch (NSException *error) {
            report[@"status"] = @"private-api-exception";
            [report[@"errors"] addObject:error.name];
        }
        bool unloaded = !loaded;
        if (loaded) {
            NSError *error = nil;
            unloaded = [client unloadModel:model options:@{} qos:25 error:&error];
            report[@"unload_succeeded"] = @(unloaded);
        }
        // A failed unload quarantines our mapping until child exit, without reuse.
        if (unloaded) for (auto value : surfaces) CFRelease(value);
        report[@"mapping_reuse_after_failure"] = @NO;
        save(report, argv[5]);
        std::puts([report[@"status"] UTF8String]);
        return [report[@"status"] isEqual:@"executed-and-numerically-validated"] && unloaded ? 0 : 1;
    }
}
