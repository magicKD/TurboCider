// Adapted private selectors and shared-event handoff from Splash #260.
// Modified: checked selectors, named bindings, SHA256/OS/ABI cache identity,
// process locking, actual allocation accounting and retained async ownership.
#include "ane_program.hpp"
#include "ane_ticket_state.hpp"
#include "../ane_runtime.hpp"
#include "../ane_w8a8_math.hpp"
#include "../ane_w8_stage.hpp"
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <functional>
#include <mutex>
#include <tuple>
#include <set>
#include <array>
#include <map>

@protocol TCAneModel
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
- (NSDictionary *)modelAttributes;
@end
@protocol TCAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary *)options request:(id)request qos:(unsigned)qos error:(NSError **)error;
@end
@protocol TCAneSurface
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
@end
@protocol TCAneRequest
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices outputs:(NSArray *)outputs
          outputIndices:(NSArray *)outputIndices weightsBuffer:(id)weights perfStats:(id)stats
         procedureIndex:(NSNumber *)procedure sharedEvents:(id)events transactionHandle:(NSNumber *)transaction;
- (void)setCompletionHandler:(void (^)(BOOL, NSError *))handler;
@end
@protocol TCAneEvents
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event eventType:(uint64_t)type;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(unsigned)symbol eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray *)signals waitEvents:(NSArray *)waits;
@end

namespace tc::ane::private_api {
std::filesystem::path default_cache_directory() {
    @autoreleasepool {
        NSString *directory = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject;
        if (!directory) throw CapabilityError("private ANE cache unavailable");
        return std::filesystem::path(directory.UTF8String) / "TurboCider/ane/private";
    }
}
namespace {
constexpr unsigned qos = QOS_CLASS_DEFAULT;
std::atomic<bool> process_healthy{true};
void require(bool ok, const std::string &error) { if (!ok) throw CapabilityError(error); }
uint64_t next_binding_generation() {
    static std::atomic<uint64_t> next{1};
    auto current = next.load(std::memory_order_relaxed);
    do {
        require(current < std::numeric_limits<uint64_t>::max(), "private ANE binding generation exhausted");
    } while (!next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed));
    return current;
}
std::string description(NSError *error) {
    return error ? std::string(error.localizedDescription.UTF8String) : "no driver diagnostic";
}
Class api_class(NSString *name, SEL selector) {
    static void *framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL);
    Class cls = framework ? NSClassFromString(name) : Nil;
    require(cls && [cls respondsToSelector:selector], "private ANE missing class/selector: " + std::string(name.UTF8String));
    return cls;
}
id<TCAneClient> client() {
    id<TCAneClient> connection = [(Class<TCAneClient>)api_class(@"_ANEClient", @selector(sharedConnection)) sharedConnection];
    for (NSString *selector in @[@"compileModel:options:qos:error:", @"compiledModelExistsFor:",
         @"purgeCompiledModel:", @"loadModel:options:qos:error:", @"unloadModel:options:qos:error:",
         @"evaluateWithModel:options:request:qos:error:"])
        require(connection && [(id)connection respondsToSelector:NSSelectorFromString(selector)],
                "private ANE client selector unavailable: " + std::string(selector.UTF8String));
    return connection;
}
struct FileLock {
    int descriptor = -1;
    explicit FileLock(const std::filesystem::path &path) {
        descriptor = open(path.c_str(), O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (descriptor < 0) throw CapabilityError("cannot open private ANE cache lock");
        struct stat stat{};
        if (fstat(descriptor, &stat) || !S_ISREG(stat.st_mode) || stat.st_uid != getuid() ||
            flock(descriptor, LOCK_EX)) {
            close(descriptor); descriptor = -1;
            throw CapabilityError("cannot lock private ANE cache");
        }
    }
    ~FileLock() { if (descriptor >= 0) { flock(descriptor, LOCK_UN); close(descriptor); } }
};
std::string os_build() {
    char version[256]{};
    size_t size = sizeof(version);
    require(!sysctlbyname("kern.osversion", version, &size, nullptr, 0) && size > 1 && size <= sizeof(version),
            "cannot identify private ANE OS build");
    return std::string(version, size - 1);
}
std::string key_for(std::string_view mil, std::span<const uint8_t> constants, const std::string &device) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    const auto add = [&](const void *bytes, size_t size) {
        const uint64_t length = size;
        CC_SHA256_Update(&context, &length, sizeof(length));
        const auto *at = static_cast<const uint8_t *>(bytes);
        while (size) {
            const auto take = CC_LONG(std::min(size, size_t(UINT32_MAX)));
            CC_SHA256_Update(&context, at, take); at += take; size -= take;
        }
    };
    const std::string abi = "tc-private-ane-program-v1", os = os_build();
    for (const auto &part : {abi, os, device}) add(part.data(), part.size());
    add(mil.data(), mil.size()); add(constants.data(), constants.size());
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(digest, &context);
    char hex[65];
    for (size_t i = 0; i < sizeof(digest); ++i) std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    return hex;
}
void safe_directory(const std::filesystem::path &path) {
    require(path.is_absolute() && !path.empty(), "private ANE cache must be absolute");
    // Reject existing symlinks all the way down, not just the final directory.
    auto current = path.root_path();
    for (const auto &part : path.relative_path()) {
        current /= part;
        require(!std::filesystem::is_symlink(current), "private ANE cache contains symlink");
    }
    std::filesystem::create_directories(path);
    struct stat info{};
    require(!lstat(path.c_str(), &info) && S_ISDIR(info.st_mode) && info.st_uid == getuid(),
            "private ANE cache ownership mismatch");
    require(!chmod(path.c_str(), 0700), "cannot restrict private ANE cache");
}
void source_file(const std::filesystem::path &path, const void *data, size_t size) {
    require(!std::filesystem::is_symlink(path), "private ANE source cache symlink");
    NSData *expected = [NSData dataWithBytes:data length:size];
    if (std::filesystem::exists(path)) {
        require(std::filesystem::is_regular_file(path), "private ANE cached source not regular");
        require([[NSData dataWithContentsOfFile:@(path.c_str())] isEqualToData:expected],
                "private ANE source cache digest mismatch");
    } else {
        require([expected writeToFile:@(path.c_str()) atomically:YES], "cannot write private ANE cached source");
        require(!chmod(path.c_str(), 0600), "cannot restrict private ANE cached source");
    }
}
std::vector<std::string> symbols(NSDictionary *attributes, NSString *key) {
    id array = attributes[@"ANEFModelDescription"][key];
    require([array isKindOfClass:NSArray.class] && [array count] > 0, "private ANE graph symbol list missing");
    std::vector<std::string> result;
    std::set<std::string> unique;
    for (id item in array) {
        require([item isKindOfClass:NSString.class] && [item length] > 0, "private ANE graph symbol invalid");
        std::string name([(NSString *)item UTF8String]);
        // The private compiler decorates tensor-buffer output names, while
        // MIL and caller bindings use the declared name. Indices still follow
        // the driver's original order; never assume alphabetical ordering.
        if ([key isEqualToString:@"kANEFModelOutputSymbolsArrayKey"] && name.ends_with("@output"))
            name.resize(name.size() - 7);
        require(unique.insert(name).second, "private ANE duplicate graph symbol");
        result.push_back(std::move(name));
    }
    return result;
}
} // namespace

struct Program::Impl {
    id<TCAneClient> connection;
    id model;
    std::optional<Device> device;
    std::vector<std::string> inputs, outputs;
    std::string key;
    const uint64_t binding_generation = next_binding_generation();
    bool loaded = false, compiled = false;
    ~Impl() {
        @autoreleasepool { if (loaded) [connection unloadModel:model options:@{} qos:qos error:nil]; }
    }
};
Program::Program(Device &device, std::string_view mil, std::span<const uint8_t> constants,
                 const std::filesystem::path &cache) : impl_(std::make_shared<Impl>()) {
    @autoreleasepool {
      @try {
        require(healthy(), "private ANE disabled after failed/missing completion");
        require(!mil.empty() && mil.size() <= (16u << 20) && constants.size() <= (64u << 20), "private ANE program source exceeds limit");
        impl_->device = device;
        impl_->connection = client();
        impl_->key = key_for(mil, constants, device.name());
        safe_directory(cache);
        FileLock lock(cache / (impl_->key + ".lock"));
        const auto directory = cache / impl_->key;
        safe_directory(directory);
        source_file(directory / "model.mil", mil.data(), mil.size());
        source_file(directory / "weights.bin", constants.data(), constants.size());
        impl_->model = [(Class<TCAneModel>)api_class(@"_ANEModel", @selector(modelAtURL:key:))
            modelAtURL:[NSURL fileURLWithPath:@(directory.c_str()) isDirectory:YES] key:@(impl_->key.c_str())];
        require(impl_->model && [impl_->model respondsToSelector:@selector(modelAttributes)], "private ANE model creation failed");
        auto compile = [&] {
            NSError *error = nil;
            require([impl_->connection compileModel:impl_->model
                options:@{@"kANEFModelType":@"kANEFModelMIL", @"kANEFNetPlistFilenameKey":@"model.mil"}
                qos:qos error:&error], "private ANE compile failed: " + description(error));
            impl_->compiled = true;
        };
        const bool cached = [impl_->connection compiledModelExistsFor:impl_->model];
        if (!cached) compile();
        NSError *error = nil;
        if (![impl_->connection loadModel:impl_->model options:@{} qos:qos error:&error]) {
            require(cached, "private ANE load failed: " + description(error));
            [impl_->connection purgeCompiledModel:impl_->model];
            compile(); error = nil;
            require([impl_->connection loadModel:impl_->model options:@{} qos:qos error:&error], "private ANE reload failed: " + description(error));
        }
        impl_->loaded = true;
        NSDictionary *attributes = [(id<TCAneModel>)impl_->model modelAttributes];
        impl_->inputs = symbols(attributes, @"kANEFModelInputSymbolsArrayKey");
        impl_->outputs = symbols(attributes, @"kANEFModelOutputSymbolsArrayKey");
      } @catch (NSException *exception) {
        throw CapabilityError("private ANE Objective-C exception: " + std::string(exception.reason.UTF8String ?: "unknown"));
      }
    }
}
bool Program::healthy() { return process_healthy.load() && gpu::healthy(); }
const std::vector<std::string> &Program::inputs() const { return impl_->inputs; }
const std::vector<std::string> &Program::outputs() const { return impl_->outputs; }
const std::string &Program::cache_key() const { return impl_->key; }
uint64_t Program::binding_generation() const { return impl_->binding_generation; }
bool Program::compiled_now() const { return impl_->compiled; }

struct Ticket::Impl {
    std::mutex mutex;
    std::condition_variable cv;
    detail::TicketCompletionState completion;
    std::function<void(uint64_t)> release;
    uint64_t signal = 0;
    std::function<void()> failure;
    // The callback owns this state, which owns request/model/event/surfaces.
    // Timeout MUST NOT drop live driver resources or permit their reuse. A
    // missing callback conservatively quarantines this one state until exit.
    std::shared_ptr<void> program;
    std::vector<Surface> surfaces;
    std::optional<AneEventBinding> external_events;
    id<MTLSharedEvent> ready_event, done_event;
    id request;
};
std::optional<AneRequestProvenance> Ticket::provenance() const {
    if (!impl_ || !impl_->external_events) return std::nullopt;
    return impl_->external_events->provenance();
}
Completion Ticket::status() const {
    if (!impl_) return {false, false, "private ANE empty ticket", true, true};
    std::lock_guard lock(impl_->mutex);
    return impl_->completion.status();
}
Completion Ticket::finish(std::chrono::milliseconds timeout) {
    if (!impl_) return status();
    std::unique_lock lock(impl_->mutex);
    if (!impl_->cv.wait_for(lock, timeout, [&] { return impl_->completion.terminal(); })) {
        const bool first_failure = impl_->completion.retire(
            "private ANE callback timeout; resources quarantined, executor must disable", true);
        const auto result = impl_->completion.status();
        lock.unlock();
        process_healthy = false;
        impl_->cv.notify_all();
        if (first_failure) {
            if (impl_->failure) impl_->failure();
            // Legacy waiter-release policy only. It proves neither physical
            // completion nor permission for a native graph to read the output.
            if (impl_->release) impl_->release(impl_->signal);
        }
        return result;
    }
    return impl_->completion.status();
}
void PreparedRequest::discard() noexcept {
    if (!state_) return;
    // No driver work exists until submit consumes state_. A prepared request
    // owns its callback, which owns this state, so explicitly break the cycle.
    {
        std::lock_guard lock(state_->mutex);
        state_->completion.discard_unsubmitted();
        state_->request = nil;
    }
    state_.reset();
}
PreparedRequest::~PreparedRequest() { discard(); }
PreparedRequest::PreparedRequest(PreparedRequest &&other) noexcept
    : state_(std::move(other.state_)) {}
PreparedRequest &PreparedRequest::operator=(PreparedRequest &&other) noexcept {
    if (this != &other) { discard(); state_ = std::move(other.state_); }
    return *this;
}
Ticket PreparedRequest::submit() {
    require(bool(state_), "private ANE prepared request already consumed");
    require(Program::healthy(), "private ANE disabled after failed/missing completion");
    auto program = std::static_pointer_cast<Program::Impl>(state_->program);
    if (state_->external_events) {
        // W0 only. This does not replace the caller's successful physical GPU
        // producer certificate; the epoch owner must exclude other signalers.
        @try {
            state_->external_events->validate_metadata(program->device->native_device(), program->key,
                                                       program->binding_generation);
            state_->external_events->validate_w0_values(state_->ready_event.signaledValue,
                                                       state_->done_event.signaledValue);
        } @catch (NSException *exception) {
            // No request has been submitted; its PreparedRequest still owns it.
            throw CapabilityError("private ANE external event preflight exception: " +
                                  std::string(exception.reason.UTF8String ?: "unknown"));
        }
    }
    auto state = std::exchange(state_, {});
    @autoreleasepool {
      @try {
        NSError *error = nil;
        if (![program->connection evaluateWithModel:program->model options:@{} request:state->request qos:qos error:&error]) {
            process_healthy = false;
            // A failed submission may still have reached the driver. Retain
            // its callback resources; rejection is not physical completion.
            bool first_failure;
            {
                std::lock_guard lock(state->mutex);
                first_failure = state->completion.retire("private ANE enqueue failed: " + description(error));
            }
            state->cv.notify_all();
            if (first_failure) {
                if (state->failure) state->failure();
                if (state->release) state->release(state->signal);
            }
        }
        Ticket ticket; ticket.impl_ = std::move(state); return ticket;
      } @catch (NSException *exception) {
        process_healthy = false;
        const auto message = "private ANE enqueue exception: " + std::string(exception.reason.UTF8String ?: "unknown");
        bool first_failure;
        {
            std::lock_guard lock(state->mutex);
            first_failure = state->completion.retire(message);
        }
        state->cv.notify_all();
        if (first_failure) {
            if (state->failure) state->failure();
            if (state->release) state->release(state->signal);
        }
        throw CapabilityError(message);
      }
    }
}
Ticket Program::enqueue(std::span<const std::pair<std::string, Surface>> inputs,
                        std::span<const std::pair<std::string, Surface>> outputs,
                        uint64_t wait_value, uint64_t signal_value, std::function<void()> failure) {
    return prepare(inputs, outputs, wait_value, signal_value, std::move(failure)).submit();
}
PreparedRequest Program::prepare(std::span<const std::pair<std::string, Surface>> inputs,
                                std::span<const std::pair<std::string, Surface>> outputs,
                                uint64_t wait_value, uint64_t signal_value, std::function<void()> failure) {
    return prepare_impl(inputs, outputs, wait_value, signal_value, std::nullopt, std::move(failure));
}
PreparedRequest Program::prepare(std::span<const std::pair<std::string, Surface>> inputs,
                                std::span<const std::pair<std::string, Surface>> outputs,
                                AneEventBinding events, std::function<void()> failure) {
    const auto ready = events.ready().value, done = events.done().value;
    return prepare_impl(inputs, outputs, ready, done, std::move(events), std::move(failure));
}
PreparedRequest Program::prepare_impl(std::span<const std::pair<std::string, Surface>> inputs,
                                     std::span<const std::pair<std::string, Surface>> outputs,
                                     uint64_t wait_value, uint64_t signal_value,
                                     std::optional<AneEventBinding> external_events,
                                     std::function<void()> failure) {
    @autoreleasepool {
      const bool external = external_events.has_value();
      auto emergency_failure = failure;
      @try {
        require(healthy(), "private ANE disabled after failed/missing completion");
        if (external) {
            external_events->validate_metadata(impl_->device->native_device(), impl_->key, impl_->binding_generation);
        } else {
            require(wait_value && signal_value > wait_value && wait_value >= impl_->device->value(),
                    "private ANE invalid dependency timeline");
        }
        auto state = std::make_shared<Ticket::Impl>();
        state->failure = std::move(failure);
        state->program = impl_; state->signal = signal_value;
        auto device = impl_->device;
        state->external_events = std::move(external_events);
        if (external) {
            state->ready_event = (__bridge id<MTLSharedEvent>)state->external_events->ready().native_event;
            state->done_event = (__bridge id<MTLSharedEvent>)state->external_events->done().native_event;
            for (const auto *endpoint : {&state->external_events->ready(), &state->external_events->done()}) {
                id<MTLSharedEvent> event = (__bridge id<MTLSharedEvent>)endpoint->native_event;
                require([(id)event respondsToSelector:@selector(device)] &&
                        [(id)event respondsToSelector:@selector(signaledValue)],
                        "private ANE external event has wrong Metal type");
                detail::validate_shared_event_device(endpoint->native_device, (__bridge void*)event.device,
                                                     device->native_device());
                // Splash Program::enqueue (local 0d5a035e) checks eventPort:
                // validation-layer wrappers may lack the port ANE must share.
                require([(id)event respondsToSelector:NSSelectorFromString(@"eventPort")],
                        "private ANE cannot share a validation-wrapped Metal event");
            }
            // Deliberately no release callback: failure/timeout must not raise
            // either external event or grant permission to read stale output.
        } else {
            state->ready_event = state->done_event = (__bridge id<MTLSharedEvent>)device->shared_event();
            state->release = [device](uint64_t value) { device->release_after_failure(value); };
        }
        auto objects = [&](auto bindings, const auto &names, bool writes) {
            require(bindings.size() == names.size(), "private ANE binding count mismatch");
            NSMutableArray *result = [NSMutableArray new];
            std::set<std::string> observed;
            Class<TCAneSurface> cls = (Class<TCAneSurface>)api_class(@"_ANEIOSurfaceObject", @selector(objectWithIOSurface:));
            for (const auto &name : names) {
                auto found = std::find_if(bindings.begin(), bindings.end(), [&](const auto &b) { return b.first == name; });
                require(found != bindings.end() && observed.insert(name).second, "private ANE binding names mismatch: " + name);
                require(!found->second.is_view(), "private ANE consumer surface slice cannot be a driver binding");
                if (external) {
                    const auto lease = found->second.metal_buffer_lease();
                    lease.validate(device->native_device());
                    if (writes) for (const auto &previous : state->surfaces)
                        require(previous.metal_buffer_lease().native_buffer != lease.native_buffer,
                                "private ANE W0 output must not alias an input or another output");
                }
                id object = [cls objectWithIOSurface:(IOSurfaceRef)found->second.native_iosurface()];
                require(object != nil, "private ANE surface binding failed");
                [result addObject:object]; state->surfaces.push_back(found->second);
            }
            return result;
        };
        // Build bindings before any async evaluation. No driver work exists
        // if validation throws here.
        NSArray *in = objects(inputs, impl_->inputs, false), *out = objects(outputs, impl_->outputs, true);
        auto indices = [](NSUInteger count) {
            NSMutableArray *result = [NSMutableArray new];
            for (NSUInteger i = 0; i < count; ++i) [result addObject:@(i)];
            return result;
        };
        id wait = [(Class<TCAneEvents>)api_class(@"_ANESharedWaitEvent", @selector(waitEventWithValue:sharedEvent:eventType:))
            waitEventWithValue:wait_value sharedEvent:state->ready_event eventType:0];
        id signal = [(Class<TCAneEvents>)api_class(@"_ANESharedSignalEvent", @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:))
            signalEventWithValue:signal_value symbolIndex:0 eventType:0 sharedEvent:state->done_event];
        require(wait && signal, "private ANE event binding failed");
        id events = [(Class<TCAneEvents>)api_class(@"_ANESharedEvents", @selector(sharedEventsWithSignalEvents:waitEvents:))
            sharedEventsWithSignalEvents:@[signal] waitEvents:@[wait]];
        require(events != nil, "private ANE shared event binding failed");
        state->request = [(Class<TCAneRequest>)api_class(@"_ANERequest", @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:))
            requestWithInputs:in inputIndices:indices(in.count) outputs:out outputIndices:indices(out.count)
            weightsBuffer:nil perfStats:nil procedureIndex:@0 sharedEvents:events transactionHandle:nil];
        require(state->request && [state->request respondsToSelector:@selector(setCompletionHandler:)], "private ANE async request creation failed");
        [(id<TCAneRequest>)state->request setCompletionHandler:^(BOOL ok, NSError *error) {
            bool first_failure;
            {
                std::lock_guard lock(state->mutex);
                first_failure = state->completion.completed(bool(ok), ok ? "" : description(error));
                state->request = nil; // break callback ownership cycle ONLY on driver completion
            }
            if (!ok) process_healthy = false;
            if (first_failure) {
                if (state->failure) state->failure();
                if (state->release) state->release(state->signal);
            }
            state->cv.notify_all();
        }];
        PreparedRequest prepared; prepared.state_ = std::move(state); return prepared;
      } @catch (NSException *exception) {
        process_healthy = false;
        if (emergency_failure) emergency_failure();
        if (!external) impl_->device->release_after_failure(signal_value);
        throw CapabilityError("private ANE enqueue exception: " + std::string(exception.reason.UTF8String ?: "unknown"));
      }
    }
}
} // namespace tc::ane::private_api
