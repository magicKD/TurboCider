// Development-only synchronous Metal build-API observer. Never linked into
// the App/runtime. Inject into an owned benchmark with DYLD_INSERT_LIBRARIES.
// MRC is deliberate: all intercepted `new` methods return their original +1
// object unchanged, without ARC ownership inference at IMP boundaries.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <map>
#include <mutex>

namespace {
struct State {
    int fd = -1;
    std::mutex methods, output;
    std::map<std::pair<uintptr_t, uintptr_t>, IMP> originals;
    uint64_t count = 0;
};
State &state() { static auto *value = new State; return *value; }
thread_local unsigned depth = 0;
uint64_t now_ns() {
    static const auto scale = [] { mach_timebase_info_data_t v; mach_timebase_info(&v); return v; }();
    return uint64_t(__uint128_t(mach_absolute_time()) * scale.numer / scale.denom);
}
void record(NSDictionary *row) {
    auto &s = state();
    std::lock_guard lock(s.output);
    if (s.fd < 0) return;
    // A bounded trace must never silently appear to have complete coverage.
    if (s.count++ > 200000) return;
    if (s.count == 200001) row = @{ @"type": @"truncated", @"pid": @(getpid()) };
    @autoreleasepool {
        auto data = [NSJSONSerialization dataWithJSONObject:row options:0 error:nullptr];
        auto bytes = static_cast<const char *>(data.bytes);
        size_t left = data.length;
        while (left) {
            auto n = write(s.fd, bytes, left);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { s.fd = -1; return; }
            left -= size_t(n); bytes += n;
        }
        if (write(s.fd, "\n", 1) != 1) s.fd = -1;
    }
}
IMP original(id object, SEL selector) {
    std::lock_guard lock(state().methods);
    for (Class cls = object_getClass(object); cls; cls = class_getSuperclass(cls)) {
        auto it = state().originals.find({uintptr_t(cls), uintptr_t(selector)});
        if (it != state().originals.end()) return it->second;
    }
    // A successfully installed wrapper always has its original registered.
    std::abort();
}
bool hook(Class cls, SEL selector, IMP replacement) {
    std::lock_guard lock(state().methods);
    auto key = std::pair{uintptr_t(cls), uintptr_t(selector)};
    if (state().originals.contains(key)) return true;
    Method method = class_getInstanceMethod(cls, selector);
    if (!method) return false;
    IMP previous = method_getImplementation(method);
    if (previous == replacement) return true; // inherited wrapper
    state().originals.emplace(key, previous);
    if (!class_addMethod(cls, selector, replacement, method_getTypeEncoding(method)))
        method_setImplementation(class_getInstanceMethod(cls, selector), replacement);
    return true;
}
struct Call {
    uint64_t start = now_ns();
    unsigned nesting = depth++;
    void finish(NSString *kind, SEL selector, id object, NSString *name, bool success,
                NSDictionary *extra = @{}) {
        const uint64_t end = now_ns();
        --depth;
        uint64_t thread = 0; pthread_threadid_np(nullptr, &thread);
        auto row = [NSMutableDictionary dictionaryWithDictionary:extra];
        [row addEntriesFromDictionary:@{
            @"type": @"metal_api", @"kind": kind, @"selector": NSStringFromSelector(selector),
            @"class": NSStringFromClass(object_getClass(object)), @"name": name ?: @"",
            @"start_ns": @(start), @"end_ns": @(end), @"thread": @(thread),
            @"pid": @(getpid()), @"nesting": @(nesting), @"success": @(success)
        }];
        record(row);
    }
};
NSString *short_name(NSString *name) { return name.length <= 256 ? name : [name substringToIndex:256]; }
void hook_library(id library);

id library_source(id self, SEL sel, NSString *source, MTLCompileOptions *options, NSError **error) {
    auto fn = reinterpret_cast<id (*)(id, SEL, NSString *, MTLCompileOptions *, NSError **)>(original(self, sel));
    // Do not record source/prompt text. This non-cryptographic digest identifies
    // repeated source strings only, and is not used as a provenance guarantee.
    const char *bytes = source.UTF8String;
    uint64_t hash = 14695981039346656037ULL, size = 0;
    if (bytes) for (; bytes[size]; ++size) { hash ^= uint8_t(bytes[size]); hash *= 1099511628211ULL; }
    Call call;
    id value = fn(self, sel, source, options, error);
    call.finish(@"library_source", sel, self, @"", value != nil,
                @{ @"source_bytes": @(size), @"source_fnv1a64": [NSString stringWithFormat:@"%016llx", (unsigned long long)hash] });
    if (value) hook_library(value);
    return value;
}
id library_url(id self, SEL sel, NSURL *url, NSError **error) {
    auto fn = reinterpret_cast<id (*)(id, SEL, NSURL *, NSError **)>(original(self, sel));
    Call call; id value = fn(self, sel, url, error);
    call.finish(@"library_load", sel, self, short_name(url.lastPathComponent), value != nil);
    if (value) hook_library(value);
    return value;
}
id library_data(id self, SEL sel, dispatch_data_t data, NSError **error) {
    auto fn = reinterpret_cast<id (*)(id, SEL, dispatch_data_t, NSError **)>(original(self, sel));
    Call call; id value = fn(self, sel, data, error);
    call.finish(@"library_load", sel, self, @"data", value != nil);
    if (value) hook_library(value);
    return value;
}
id function_name(id self, SEL sel, NSString *name) {
    auto fn = reinterpret_cast<id (*)(id, SEL, NSString *)>(original(self, sel));
    Call call; id value = fn(self, sel, name);
    call.finish(@"function", sel, self, short_name(name), value != nil); return value;
}
id function_constants(id self, SEL sel, NSString *name, MTLFunctionConstantValues *constants, NSError **error) {
    auto fn = reinterpret_cast<id (*)(id, SEL, NSString *, MTLFunctionConstantValues *, NSError **)>(original(self, sel));
    Call call; id value = fn(self, sel, name, constants, error);
    call.finish(@"function_specialize", sel, self, short_name(name), value != nil); return value;
}
id function_descriptor(id self, SEL sel, MTLFunctionDescriptor *descriptor, NSError **error) {
    auto fn = reinterpret_cast<id (*)(id, SEL, MTLFunctionDescriptor *, NSError **)>(original(self, sel));
    Call call; id value = fn(self, sel, descriptor, error);
    call.finish(@"function_specialize", sel, self, short_name(descriptor.name), value != nil); return value;
}
void hook_library(id library) {
    Class cls = object_getClass(library);
    hook(cls, @selector(newFunctionWithName:), reinterpret_cast<IMP>(function_name));
    hook(cls, @selector(newFunctionWithName:constantValues:error:), reinterpret_cast<IMP>(function_constants));
    hook(cls, @selector(newFunctionWithDescriptor:error:), reinterpret_cast<IMP>(function_descriptor));
}
id pipeline_function(id self, SEL sel, id<MTLFunction> function, NSError **error) {
    auto fn = reinterpret_cast<id (*)(id, SEL, id<MTLFunction>, NSError **)>(original(self, sel));
    Call call; id value = fn(self, sel, function, error);
    call.finish(@"pipeline", sel, self, short_name(function.name), value != nil); return value;
}
id pipeline_options(id self, SEL sel, id<MTLFunction> function, MTLPipelineOption options,
                    MTLComputePipelineReflection **reflection, NSError **error) {
    auto fn = reinterpret_cast<id (*)(id, SEL, id<MTLFunction>, MTLPipelineOption, MTLComputePipelineReflection **, NSError **)>(original(self, sel));
    Call call; id value = fn(self, sel, function, options, reflection, error);
    call.finish(@"pipeline", sel, self, short_name(function.name), value != nil); return value;
}
id pipeline_descriptor(id self, SEL sel, MTLComputePipelineDescriptor *descriptor, MTLPipelineOption options,
                       MTLComputePipelineReflection **reflection, NSError **error) {
    auto fn = reinterpret_cast<id (*)(id, SEL, MTLComputePipelineDescriptor *, MTLPipelineOption, MTLComputePipelineReflection **, NSError **)>(original(self, sel));
    Call call; id value = fn(self, sel, descriptor, options, reflection, error);
    call.finish(@"pipeline", sel, self, short_name(descriptor.computeFunction.name), value != nil); return value;
}
__attribute__((constructor)) void install() {
    const char *path = std::getenv("TURBOCIDER_METAL_TRACE_FILE");
    if (!path || !*path) return;
    // Refuse symlinks and pre-existing files, including prior traces.
    state().fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (state().fd < 0) { std::fprintf(stderr, "Metal trace could not create a fresh file: %s\n", strerror(errno)); return; }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        Class cls = object_getClass(device);
        const bool source = cls && hook(cls, @selector(newLibraryWithSource:options:error:), reinterpret_cast<IMP>(library_source));
        const bool url = cls && hook(cls, @selector(newLibraryWithURL:error:), reinterpret_cast<IMP>(library_url));
        const bool data = cls && hook(cls, @selector(newLibraryWithData:error:), reinterpret_cast<IMP>(library_data));
        const bool plain = cls && hook(cls, @selector(newComputePipelineStateWithFunction:error:), reinterpret_cast<IMP>(pipeline_function));
        const bool options = cls && hook(cls, @selector(newComputePipelineStateWithFunction:options:reflection:error:), reinterpret_cast<IMP>(pipeline_options));
        const bool descriptor = cls && hook(cls, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), reinterpret_cast<IMP>(pipeline_descriptor));
        record(@{ @"type": @"setup", @"schema_version": @1, @"pid": @(getpid()), @"clock": @"mach_absolute_time_ns",
            @"device": device.name ?: @"unavailable", @"device_class": cls ? NSStringFromClass(cls) : @"",
            @"library_source": @(source), @"library_url": @(url), @"library_data": @(data),
            @"pipeline_function": @(plain), @"pipeline_options": @(options), @"pipeline_descriptor": @(descriptor),
            @"coverage": @"synchronous Metal library/function/pipeline host API calls; not GPU execution, driver internals or MLX graph tracing" });
        [device release];
    }
}
} // namespace
