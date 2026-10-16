// Original, version-checked private API client for a bounded real-Qwen experiment.
// Not part of shipping targets. No PMU, security changes or system-wide drain.
#pragma once
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>

@interface NSObject (TCRealSharedQwen)
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

namespace tcq {
constexpr uint32_t M = 1024, H = 4096, F = 12288, S = 2048;
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
uint64_t now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
NSString *sha(const void *data, size_t bytes) {
    check(bytes <= UINT32_MAX, "bounded tensor digest required");
    unsigned char value[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(data, CC_LONG(bytes), value);
    NSMutableString *result = [NSMutableString string];
    for (auto byte : value) [result appendFormat:@"%02x", byte];
    return result;
}
void save(NSDictionary *value, const char *path) {
    NSError *error = nil;
    NSData *bytes = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted error:&error];
    check(bytes && [bytes writeToFile:[NSString stringWithUTF8String:path] options:NSDataWritingAtomic error:&error], "report write failed");
}
void guard(Class cls, NSString *name, bool meta, NSUInteger count, char kind, NSMutableDictionary *signatures) {
    Method method = meta ? class_getClassMethod(cls, NSSelectorFromString(name)) : class_getInstanceMethod(cls, NSSelectorFromString(name));
    check(method, "required private selector absent");
    auto signature = [NSMethodSignature signatureWithObjCTypes:method_getTypeEncoding(method)];
    check(signature.numberOfArguments == count, "private argument count changed");
    check(signature.methodReturnType[0] == kind || (kind == 'B' && signature.methodReturnType[0] == 'c'), "private return ABI changed");
    for (NSUInteger index = 2; index < count; ++index) {
        char expected = [name containsString:@"qos:"] && index == count - 2 ? 'I'
            : [name containsString:@"qos:"] && index == count - 1 ? '^' : [name isEqual:@"objectWithIOSurface:"] ? '^' : '@';
        check([signature getArgumentTypeAtIndex:index][0] == expected, "private argument ABI changed");
    }
    signatures[[NSString stringWithFormat:@"%@.%@", NSStringFromClass(cls), name]] = [NSString stringWithUTF8String:method_getTypeEncoding(method)];
}
uint64_t integer(NSDictionary *value, NSString *key, uint64_t maximum = 512ull * 1024 * 1024) {
    id number = value[key];
    check([number isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)number) != CFBooleanGetTypeID(), "typed integer field required");
    check([number longLongValue] >= 0 && [number unsignedLongLongValue] <= maximum, "integer field outside cap");
    return [number unsignedLongLongValue];
}
NSDictionary *descriptor(NSDictionary *value) {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSString *key in @[@"Batches", @"Channels", @"Height", @"Width", @"Type", @"Offset", @"Depth", @"PlaneCount", @"RowStride", @"PlaneStride", @"BatchStride"])
        if (value[key]) result[key] = value[key];
    return result;
}
struct Private {
    id client = nil, model = nil; Class io_class = Nil, request_class = Nil;
    bool loaded = false;
    void load(NSString *directory, NSMutableDictionary *report, bool compile_allowed = true) {
        check(dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL), "private framework unavailable");
        Class cc = NSClassFromString(@"_ANEClient"), mc = NSClassFromString(@"_ANEModel");
        io_class = NSClassFromString(@"_ANEIOSurfaceObject"); request_class = NSClassFromString(@"_ANERequest");
        NSMutableDictionary *signatures = [NSMutableDictionary dictionary]; report[@"private_method_signatures"] = signatures;
        guard(cc, @"sharedConnection", true, 2, '@', signatures); guard(mc, @"modelAtURL:key:", true, 4, '@', signatures);
        guard(mc, @"modelAttributes", false, 2, '@', signatures);
        guard(io_class, @"objectWithIOSurface:", true, 3, '@', signatures); guard(io_class, @"ioSurface", false, 2, '^', signatures);
        guard(request_class, @"requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:", true, 8, '@', signatures);
        guard(request_class, @"validate", false, 2, 'B', signatures);
        for (NSString *name in @[@"inputArray", @"outputArray"]) guard(request_class, name, false, 2, '@', signatures);
        for (NSString *name in @[@"compileModel:options:qos:error:", @"loadModel:options:qos:error:", @"unloadModel:options:qos:error:"])
            guard(cc, name, false, 6, 'B', signatures);
        guard(cc, @"evaluateWithModel:options:request:qos:error:", false, 7, 'B', signatures);
        client = [cc sharedConnection]; model = [mc modelAtURL:[NSURL fileURLWithPath:directory] key:@"net"];
        check(client && model, "private client/model construction failed");
        NSDictionary *options = @{@"kANEFModelType": @"kANEFModelMIL", @"kANEFPerformanceStatsMask": @0}; NSError *error = nil;
        report[@"compile_selector_called"] = @(compile_allowed); uint64_t start = now();
        if (compile_allowed) {
            report[@"stage"] = @"compile";
            bool compiled = [client compileModel:model options:options qos:25 error:&error]; report[@"compile_wall_ns"] = @(now() - start);
            if (!compiled) report[@"private_error"] = @{@"domain": error.domain ?: @"unknown", @"code": @(error.code)};
            check(compiled, "real prefix MIL compile failed; no fallback");
        } else report[@"compile_wall_ns"] = NSNull.null;
        report[@"stage"] = @"load"; start = now(); loaded = [client loadModel:model options:options qos:25 error:&error];
        report[@"load_wall_ns"] = @(now() - start);
        if (!loaded) report[@"private_error"] = @{@"domain": error.domain ?: @"unknown", @"code": @(error.code)};
        check(loaded, "real prefix MIL load failed; no fallback");
        NSArray *networks = [model modelAttributes][@"NetworkStatusList"];
        check([networks isKindOfClass:NSArray.class] && networks.count == 1, "unique real-prefix network required");
        NSArray *inputs = networks[0][@"LiveInputList"], *outputs = networks[0][@"LiveOutputList"];
        check(inputs.count == 1 && outputs.count == 1, "one input/output port required");
        report[@"input_descriptor"] = descriptor(inputs[0]); report[@"output_descriptor"] = descriptor(outputs[0]);
    }
    bool unload() {
        if (!loaded) return true;
        NSError *error = nil; bool value = [client unloadModel:model options:@{} qos:25 error:&error];
        if (value) loaded = false; return value;
    }
};
}
