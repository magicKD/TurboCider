// Adapted private selectors and shared-event handoff from Splash #260.
// Modified: checked selectors, named bindings, SHA256/OS/ABI cache identity,
// process locking, actual allocation accounting and retained async ownership.
#include "ane_program.hpp"
#include "ane_transfer_kernels.hpp"
#include "ane_w8_kernels.hpp"
#include "../ane_runtime.hpp"
#include "../ane_w8a8_math.hpp"
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
namespace {
constexpr unsigned qos = QOS_CLASS_DEFAULT;
std::atomic<bool> process_healthy{true};
void require(bool ok, const std::string &error) { if (!ok) throw CapabilityError(error); }
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

namespace {
bool same_generation(const std::weak_ptr<void>&a,const std::weak_ptr<void>&b) {
    return !a.owner_before(b)&&!b.owner_before(a);
}
bool same_scale_matrix(const DeviceMatrixView&a,const DeviceMatrixView&b) {
    return same_generation(a.allocation_identity,b.allocation_identity)&&
        std::tie(a.buffer,a.buffer_bytes,a.offset_bytes,a.rows,a.cols,a.row_stride_bytes,a.dtype)==
        std::tie(b.buffer,b.buffer_bytes,b.offset_bytes,b.rows,b.cols,b.row_stride_bytes,b.dtype);
}
bool same_scale_key(const DeviceWeightRegion&a,const DeviceWeightRegion&b) {
    const auto &x=a.source,&y=b.source;const auto &p=a.selection,&q=b.selection;
    const auto meta=[](const auto&a,const auto&b){return bool(a)==bool(b)&&(!a||same_scale_matrix(*a,*b));};
    return same_generation(x.allocation_identity,y.allocation_identity)&&meta(x.scales,y.scales)&&meta(x.offsets,y.offsets)&&
        std::tie(x.buffer,x.buffer_bytes,x.offset_bytes,x.row_stride_bytes,x.rows,x.cols,x.encoding,x.dense_dtype,x.group_size)==
        std::tie(y.buffer,y.buffer_bytes,y.offset_bytes,y.row_stride_bytes,y.rows,y.cols,y.encoding,y.dense_dtype,y.group_size)&&
        std::tie(p.row_begin,p.rows,p.column_begin,p.columns,p.rotation_block,p.rotation_seed,p.transpose)==
        std::tie(q.row_begin,q.rows,q.column_begin,q.columns,q.rotation_block,q.rotation_seed,q.transpose);
}
bool live_scale_key(const DeviceWeightRegion&key) {
    return !key.source.allocation_identity.expired() &&
        (!key.source.scales||!key.source.scales->allocation_identity.expired()) &&
        (!key.source.offsets||!key.source.offsets->allocation_identity.expired());
}
DeviceWeightRegion weak_scale_key(DeviceWeightView source,W8StageSpec spec) {
    source.owner.reset();if(source.scales)source.scales->owner.reset();if(source.offsets)source.offsets->owner.reset();
    return {std::move(source),spec}; // no strong source/model allocation lease retained in the cache
}
struct ScaleCacheEntry {
    DeviceWeightRegion key;
    id<MTLBuffer> buffer;
    std::atomic<bool> valid{false};
    uint64_t last_use=0;
};
}
struct Device::Impl {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue;
    id<MTLCommandQueue> staging_queue, activation_queue;
    id<MTLSharedEvent> event;
    uint64_t last_signal = 0;
    std::mutex release_mutex;
    id<MTLComputePipelineState> upload, restore;
    id<MTLComputePipelineState> w8_scale_copy;
    id<MTLLibrary> w8_library;
    bool specialize_staging = false;
    struct W8Pipelines { id<MTLComputePipelineState> scales, codes; };
    std::mutex pipelines_mutex;
    // At most (3 dense dtypes + 6 packed encodings) * 2 rotation blocks.
    // Keys retain no source allocation or model/adapter values.
    std::map<std::array<uint32_t,3>,W8Pipelines> w8_pipelines;
    W8Pipelines pipelines(DeviceWeightEncoding encoding,DType dtype,int block) {
        std::lock_guard lock(pipelines_mutex);
        NSError *error=nil;
        if(!w8_library) {
            MTLCompileOptions *options=[MTLCompileOptions new];options.fastMathEnabled=NO;
            const std::string shader=std::string(transfer_source)+w8_source;
            w8_library=[device newLibraryWithSource:@(shader.c_str()) options:options error:&error];
            require(w8_library!=nil,"W8 stager shader compile failed: "+description(error));
        }
        if(!w8_scale_copy) {
            w8_scale_copy=[device newComputePipelineStateWithFunction:[w8_library newFunctionWithName:@"tc_ane_w8_scale_copy"] error:&error];
            require(w8_scale_copy!=nil,"W8 scale metadata copy pipeline unavailable");
        }
        std::array<uint32_t,3> key{0,0,0};
        if(specialize_staging)key={uint32_t(encoding),encoding==DeviceWeightEncoding::Dense?uint32_t(dtype):0u,uint32_t(block)};
        auto found=w8_pipelines.find(key);if(found!=w8_pipelines.end())return found->second;
        require(w8_pipelines.size()<18,"W8 pipeline variant bound exceeded");
        MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
        const bool specialized=specialize_staging;
        [constants setConstantValue:&specialized type:MTLDataTypeBool atIndex:0];
        for(NSUInteger index=0;index<3;++index)[constants setConstantValue:&key[index] type:MTLDataTypeUInt atIndex:index+1];
        W8Pipelines result;
        auto make=[&](NSString *name) {
            id<MTLFunction> function=[w8_library newFunctionWithName:name constantValues:constants error:&error];
            require(function!=nil,"W8 specialized function unavailable: "+description(error));
            id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline!=nil && pipeline.threadExecutionWidth==32 && pipeline.maxTotalThreadsPerThreadgroup>=uint32_t(block),
                    "W8 pipeline rotation/SIMD capacity unsupported");
            return pipeline;
        };
        result.scales=make(@"tc_ane_w8_scales");result.codes=make(@"tc_ane_w8_codes");
        w8_pipelines.emplace(key,result);return result;
    }
    std::mutex scale_cache_mutex;
    std::vector<std::shared_ptr<ScaleCacheEntry>> scale_cache;
    WeightCacheStats scale_stats;
    uint64_t scale_clock=0;
    // One current seed, not an unbounded seed/model cache. Active stage
    // tickets retain previous immutable buffers across a seed change. H128
    // takes the first 128 signs of this same H512 recipe table.
    std::mutex signs_mutex;
    std::optional<uint64_t> signs_seed;
    id<MTLBuffer> rotation_signs;
    id<MTLBuffer> signs(uint64_t seed) {
        std::lock_guard lock(signs_mutex);
        if (!signs_seed || *signs_seed != seed) {
            id<MTLBuffer> next = [device newBufferWithLength:512*sizeof(float) options:MTLResourceStorageModeShared];
            require(next != nil,"W8 rotation-sign metadata allocation failed");
            for (uint32_t i=0;i<512;++i) static_cast<float*>(next.contents)[i]=float(rotation_sign(seed,i));
            rotation_signs=next;signs_seed=seed;
        }
        return rotation_signs;
    }
    void release(uint64_t value) {
        std::lock_guard lock(release_mutex);
        if (event.signaledValue < value) event.signaledValue = value;
    }
};
Device::Device() : impl_(std::make_shared<Impl>()) {
    require(impl_->device != nil, "private ANE Metal unavailable");
    impl_->queue = [impl_->device newCommandQueueWithMaxCommandBufferCount:256];
    impl_->staging_queue = [impl_->device newCommandQueueWithMaxCommandBufferCount:256];
    impl_->activation_queue = [impl_->device newCommandQueueWithMaxCommandBufferCount:256];
    impl_->event = [impl_->device newSharedEvent];
    require(impl_->queue && impl_->staging_queue && impl_->activation_queue && impl_->event, "private ANE shared-event queue unavailable");
    const char* cache=std::getenv("TURBOCIDER_PRIVATE_ANE_SCALE_CACHE");
    require(!cache||std::string(cache)=="0"||std::string(cache)=="1","private ANE scale cache requires 0 or 1");
    impl_->scale_stats.enabled=!cache||std::string(cache)=="1";
    const char *specialize=std::getenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE");
    require(!specialize||std::string(specialize)=="0"||std::string(specialize)=="1","private ANE stage specialization requires 0 or 1");
    impl_->specialize_staging=specialize&&std::string(specialize)=="1";
}
WeightCacheStats Device::scale_cache_stats() const {
    std::lock_guard lock(impl_->scale_cache_mutex);auto stats=impl_->scale_stats;stats.entries=impl_->scale_cache.size();return stats;
}
StagePipelineStats Device::stage_pipeline_stats() const {
    std::lock_guard lock(impl_->pipelines_mutex);
    return {impl_->specialize_staging,impl_->w8_pipelines.size()};
}
std::string Device::name() const { return impl_->device.name.UTF8String; }
uint64_t Device::value() const { return impl_->event.signaledValue; }
void Device::signal(uint64_t value) {
    require(value > impl_->last_signal && value > impl_->event.signaledValue, "private ANE event timeline not monotonic");
    id<MTLCommandBuffer> buffer = [impl_->queue commandBuffer];
    require(buffer != nil, "private ANE signal command buffer unavailable");
    [buffer encodeSignalEvent:impl_->event value:value];
    auto owned = impl_;
    [buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        if (completed.status == MTLCommandBufferStatusError) owned->release(value);
    }];
    impl_->last_signal = value;
    [buffer commit];
}
void Device::release_prepared(uint64_t value) {
    require(value > impl_->last_signal && value > impl_->event.signaledValue,
            "private ANE prepared event timeline not monotonic");
    impl_->last_signal = value;
    impl_->release(value);
}
bool Device::wait(uint64_t value, std::chrono::milliseconds timeout) {
    // Wait is queued AFTER the leading signal has already been committed.
    // A host deadline releases a stuck wait, but it NEVER certifies success.
    id<MTLCommandBuffer> buffer = [impl_->queue commandBuffer];
    require(buffer != nil, "private ANE wait command buffer unavailable");
    struct State { std::mutex mutex; std::condition_variable cv; bool done = false, ok = false; };
    auto state = std::make_shared<State>();
    [buffer encodeWaitForEvent:impl_->event value:value];
    [buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        { std::lock_guard lock(state->mutex); state->ok = completed.status == MTLCommandBufferStatusCompleted; state->done = true; }
        state->cv.notify_all();
    }];
    [buffer commit];
    std::unique_lock lock(state->mutex);
    if (!state->cv.wait_for(lock, timeout, [&] { return state->done; })) { impl_->release(value); return false; }
    return state->ok;
}

struct Surface::Impl {
    IOSurfaceRef surface = nullptr;
    id<MTLBuffer> buffer;
    uint32_t rows = 0, columns = 0;
    Element element;
    ~Impl() { buffer = nil; if (surface) CFRelease(surface); }
};
Surface::Surface(Device &device, uint32_t rows, uint32_t columns, Element element) : impl_(std::make_shared<Impl>()) {
    require(rows && rows <= 32768 && columns && columns <= 32768 &&
            (element == Element::FP16 || element == Element::I8), "private ANE invalid surface geometry");
    const size_t item = element == Element::I8 ? 1 : 2;
    const size_t pitch = (size_t(columns) * item + 63) / 64 * 64;
    const size_t page = size_t(getpagesize());
    const size_t allocation = (size_t(rows) * pitch + page - 1) / page * page;
    impl_->surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(columns), (id)kIOSurfaceHeight:@(rows),
        (id)kIOSurfaceBytesPerElement:@(item), (id)kIOSurfaceBytesPerRow:@(pitch),
        (id)kIOSurfaceAllocSize:@(allocation),
        (id)kIOSurfacePixelFormat:@(element == Element::I8 ? 0x4c303038 : 0x4c303068)});
    require(impl_->surface != nullptr, "private ANE IOSurface allocation failed");
    impl_->rows = rows; impl_->columns = columns; impl_->element = element;
    require(IOSurfaceGetBytesPerRow(impl_->surface) == pitch && IOSurfaceGetAllocSize(impl_->surface) >= allocation,
            "private ANE IOSurface allocation changed layout");
    impl_->buffer = [device.impl_->device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(impl_->surface)
        length:IOSurfaceGetAllocSize(impl_->surface) options:MTLResourceStorageModeShared deallocator:nil];
    require(impl_->buffer != nil, "private ANE cannot share IOSurface with Metal");
}
size_t Surface::bytes() const { return impl_->buffer.allocatedSize; }
size_t Surface::pitch() const { return IOSurfaceGetBytesPerRow(impl_->surface); }
uint32_t Surface::rows() const { return row_count_ ? row_count_ : impl_->rows; }
uint32_t Surface::columns() const { return impl_->columns; }
Element Surface::element() const { return impl_->element; }
void *Surface::data() const { return static_cast<char *>(IOSurfaceGetBaseAddress(impl_->surface)) + size_t(row_begin_) * pitch(); }
Surface Surface::slice_rows(uint32_t begin, uint32_t count) const {
    require(count && begin <= rows() && count <= rows() - begin, "private ANE surface slice extent invalid");
    Surface view = *this; view.row_begin_ += begin; view.row_count_ = count; return view;
}

struct Transfer::Impl {
    std::shared_ptr<Device::Impl> device;
    std::vector<Upload> uploads;
    std::vector<Download> downloads;
    id<MTLBuffer> status, failed;
    uint64_t ready = 0, done_value = 0;
    std::mutex mutex;
    std::condition_variable cv;
    bool submitted = false, done = false, ok = false;
    void fail() {
        std::atomic_ref<uint32_t>(*static_cast<uint32_t *>(failed.contents)).store(1, std::memory_order_release);
    }
};
namespace {
size_t device_pitch(const DeviceMatrixView &v) {
    return v.row_stride_bytes ? v.row_stride_bytes : size_t(v.cols) * (v.dtype == DType::FP32 ? 4 : 2);
}
void validate_device(const DeviceMatrixView &v, id<MTLDevice> device, bool output) {
    require(v.owner && v.buffer && v.rows > 0 && v.rows <= 1048576 && v.cols > 0 && v.cols <= 32768 &&
        (v.dtype == DType::FP16 || v.dtype == DType::BF16 || (!output && v.dtype == DType::FP32)),
        "private ANE invalid device binding/type/owner");
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)v.buffer;
    const size_t item = v.dtype == DType::FP32 ? 4 : 2, pitch = device_pitch(v), row = size_t(v.cols) * item;
    require(buffer.device == device && v.buffer_bytes <= buffer.length && v.offset_bytes % item == 0 &&
        pitch % item == 0 && pitch >= row && pitch <= UINT32_MAX && pitch <= SIZE_MAX / size_t(v.rows) &&
        v.offset_bytes <= v.buffer_bytes && size_t(v.rows - 1) * pitch + row <= v.buffer_bytes - v.offset_bytes,
        "private ANE device binding extent/device mismatch");
}
struct TransferParams { uint32_t rows, cols, source_pitch, target_pitch, dtype, validate_only; float scale; uint32_t row_scale_pitch, scaled, second_scaled; };
}
Transfer Device::prepare_transfer(std::vector<Upload> uploads, std::vector<Download> downloads,
                                  uint64_t ready, uint64_t done) {
    @autoreleasepool {
        require(!downloads.empty() && ready > impl_->last_signal && ready > value() && done >= ready,
                "private ANE invalid transfer timeline/bindings");
        for (const auto &u : uploads) {
            validate_device(u.source, impl_->device, false);
            require(u.destination.element() == Element::FP16 && u.source.cols == int(u.destination.rows()) &&
                u.begin_row >= 0 && u.begin_row <= u.source.rows && int(u.destination.columns()) <= u.source.rows - u.begin_row &&
                std::isfinite(u.scale) && u.scale > 0, "private ANE upload geometry/scale mismatch");
        }
        for (const auto &d : downloads) {
            if (d.second_token_scales) require(d.second_token_scales->element() == Element::FP16 && d.second_token_scales->rows() == 1 &&
                d.second_token_scales->columns() == d.source.columns(), "private ANE secondary scale geometry mismatch");
            require(bool(d.row_scales) == bool(d.token_scales), "private ANE W8 epilogue requires both scales");
            if (d.row_scales) require(d.row_scales->element() == Element::FP16 && d.row_scales->rows() == d.source.rows() &&
                d.row_scales->columns() == 1 && d.token_scales->element() == Element::FP16 && d.token_scales->rows() == 1 &&
                d.token_scales->columns() == d.source.columns(), "private ANE W8 epilogue scale geometry mismatch");
            require(d.source.element() == Element::FP16 && d.begin_row >= 0 && std::isfinite(d.scale) && d.scale > 0 &&
                (d.dtype == DType::FP16 || d.dtype == DType::BF16), "private ANE download geometry/dtype/scale mismatch");
            if (d.destination) {
                validate_device(*d.destination, impl_->device, true);
                require(d.destination->dtype == d.dtype && d.destination->cols == int(d.source.rows()) &&
                    d.begin_row <= d.destination->rows && int(d.source.columns()) <= d.destination->rows - d.begin_row,
                    "private ANE download destination mismatch");
            }
        }
        if (!impl_->upload || !impl_->restore) {
            NSError *error = nil;
            MTLCompileOptions *options = [MTLCompileOptions new]; options.fastMathEnabled = NO;
            id<MTLLibrary> library = [impl_->device newLibraryWithSource:@(transfer_source) options:options error:&error];
            require(library != nil, "private ANE transfer shader compile failed: " + description(error));
            impl_->upload = [impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tc_ane_upload"] error:&error];
            require(impl_->upload != nil, "private ANE upload pipeline failed: " + description(error));
            impl_->restore = [impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tc_ane_restore"] error:&error];
            require(impl_->restore != nil, "private ANE restore pipeline failed: " + description(error));
        }
        auto state = std::make_shared<Transfer::Impl>();
        state->device = impl_; state->uploads = std::move(uploads); state->downloads = std::move(downloads);
        state->ready = ready; state->done_value = done;
        state->status = [impl_->device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        state->failed = [impl_->device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        require(state->status && state->failed, "private ANE transfer status allocation failed");
        *static_cast<uint32_t *>(state->failed.contents) = 0;
        Transfer result; result.impl_ = std::move(state); return result;
    }
}
std::function<void()> Transfer::failure_callback() const {
    auto state = impl_; return [state] { state->fail(); };
}
void Transfer::submit() {
    @autoreleasepool {
      @try {
        auto state = impl_;
        require(state && !state->submitted, "private ANE transfer already submitted");
        auto dev = state->device;
        require(state->ready > dev->last_signal, "private ANE transfer timeline reused");
        id<MTLCommandBuffer> input = [dev->queue commandBuffer], output = [dev->queue commandBuffer];
        require(input && output, "private ANE transfer command buffer unavailable");
        id<MTLBlitCommandEncoder> clear = [input blitCommandEncoder];
        require(clear != nil, "private ANE transfer blit encoder unavailable");
        [clear fillBuffer:state->status range:NSMakeRange(0, 4) value:0]; [clear endEncoding];
        id<MTLComputeCommandEncoder> pack = [input computeCommandEncoder];
        require(pack != nil, "private ANE upload encoder unavailable");
        [pack setComputePipelineState:dev->upload];
        for (const auto &u : state->uploads) {
            const auto &v = u.source;
            TransferParams p{u.destination.columns(), u.destination.rows(), uint32_t(device_pitch(v)),
                uint32_t(u.destination.pitch()), uint32_t(v.dtype), 0, u.scale, 0, 0, 0};
            [pack setBuffer:(__bridge id<MTLBuffer>)v.buffer offset:v.offset_bytes + size_t(u.begin_row) * device_pitch(v) atIndex:0];
            [pack setBuffer:u.destination.impl_->buffer offset:size_t(u.destination.row_begin_) * u.destination.pitch() atIndex:1];
            [pack setBuffer:state->status offset:0 atIndex:2]; [pack setBytes:&p length:sizeof(p) atIndex:3];
            [pack dispatchThreadgroups:MTLSizeMake((p.rows + 31) / 32, (p.cols + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        }
        [pack endEncoding]; [input encodeSignalEvent:dev->event value:state->ready];
        [input addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.status == MTLCommandBufferStatusError) { state->fail(); dev->release(state->done_value); }
        }];
        // Construct both encoders BEFORE committing. After leading commit,
        // no exception may leave an untracked GPU consumer or ANE wait.
        [output encodeWaitForEvent:dev->event value:state->done_value];
        id<MTLComputeCommandEncoder> restore = [output computeCommandEncoder];
        require(restore != nil, "private ANE restore encoder unavailable");
        [restore setComputePipelineState:dev->restore];
        for (const auto &d : state->downloads) {
            TransferParams p{d.source.columns(), d.source.rows(), uint32_t(d.source.pitch()),
                uint32_t(d.destination ? device_pitch(*d.destination) : 2), uint32_t(d.dtype), uint32_t(!d.destination), d.scale,
                uint32_t(d.row_scales ? d.row_scales->pitch() : 0), uint32_t(d.row_scales.has_value()), uint32_t(d.second_token_scales.has_value())};
            [restore setBuffer:d.source.impl_->buffer offset:size_t(d.source.row_begin_) * d.source.pitch() atIndex:0];
            if (d.destination) {
                const auto &v = *d.destination;
                [restore setBuffer:(__bridge id<MTLBuffer>)v.buffer offset:v.offset_bytes + size_t(d.begin_row) * device_pitch(v) atIndex:1];
            } else [restore setBuffer:state->status offset:0 atIndex:1];
            [restore setBuffer:state->status offset:0 atIndex:2]; [restore setBytes:&p length:sizeof(p) atIndex:3];
            [restore setBuffer:state->failed offset:0 atIndex:4];
            if (d.row_scales) {
                [restore setBuffer:d.row_scales->impl_->buffer offset:size_t(d.row_scales->row_begin_) * d.row_scales->pitch() atIndex:5];
                [restore setBuffer:d.token_scales->impl_->buffer offset:size_t(d.token_scales->row_begin_) * d.token_scales->pitch() atIndex:6];
            } else {
                [restore setBuffer:state->status offset:0 atIndex:5]; [restore setBuffer:state->status offset:0 atIndex:6];
            }
            if (d.second_token_scales) [restore setBuffer:d.second_token_scales->impl_->buffer
                offset:size_t(d.second_token_scales->row_begin_) * d.second_token_scales->pitch() atIndex:7];
            else [restore setBuffer:state->status offset:0 atIndex:7];
            [restore dispatchThreadgroups:MTLSizeMake((p.rows + 31) / 32, (p.cols + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        }
        [restore endEncoding];
        [output addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            { std::lock_guard lock(state->mutex); state->ok = completed.status == MTLCommandBufferStatusCompleted; state->done = true; }
            state->cv.notify_all();
        }];
        state->submitted = true; dev->last_signal = state->ready;
        [input commit]; // leading signal must be submitted before dependent wait
        [output commit];
      } @catch (NSException *exception) {
        if (impl_) { impl_->fail(); impl_->device->release(impl_->done_value); }
        process_healthy = false;
        throw CapabilityError("private ANE GPU transfer exception: " + std::string(exception.reason.UTF8String ?: "unknown"));
      }
    }
}
Completion Transfer::finish(std::chrono::milliseconds timeout) {
    if (!impl_ || !impl_->submitted) return {false, false, "private ANE transfer not submitted"};
    std::unique_lock lock(impl_->mutex);
    if (!impl_->cv.wait_for(lock, timeout, [&] { return impl_->done; })) {
        impl_->fail(); process_healthy = false; impl_->device->release(impl_->done_value);
        return {false, true, "private ANE transfer timeout; resources retained until GPU completion"};
    }
    const bool failed = std::atomic_ref<uint32_t>(*static_cast<uint32_t *>(impl_->failed.contents)).load(std::memory_order_acquire);
    return {impl_->ok && !failed, false, !impl_->ok ? "private ANE GPU transfer failed" : failed ? "private ANE transfer suppressed after failure" : ""};
}
uint32_t Transfer::validation_flags() const {
    require(impl_ && impl_->done, "private ANE transfer status not ready");
    return *static_cast<const uint32_t *>(impl_->status.contents);
}

struct QuantStage::Impl {
    std::shared_ptr<Device::Impl> device;
    DeviceWeightView source;
    W8StageSpec spec;
    Surface codes, scales;
    id<MTLBuffer> status, signs;
    id<MTLSharedEvent> event;
    std::shared_ptr<ScaleCacheEntry> cached_scales;
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false, ok = false;
    Impl(Surface a, Surface b) : codes(std::move(a)), scales(std::move(b)) {}
};
namespace {
struct W8Params {
    uint32_t source_pitch, physical_cols, encoding, dtype, group_size;
    uint32_t meta_pitch, meta_dtype, offset_pitch, offset_dtype, has_offset;
    uint32_t row_begin, rows, column_begin, columns, block, code_pitch, scale_pitch, transpose;
    uint32_t source_aligned, seed_low, seed_high;
    float norm;
};
size_t w8_source_row_bytes(const DeviceWeightView &v) {
    switch (v.encoding) {
    case DeviceWeightEncoding::Dense:
        require(v.dense_dtype == DType::FP16 || v.dense_dtype == DType::BF16 || v.dense_dtype == DType::FP32, "W8 dense dtype unsupported");
        return size_t(v.cols) * (v.dense_dtype == DType::FP32 ? 4 : 2);
    case DeviceWeightEncoding::AffineQ4: case DeviceWeightEncoding::AffineQ8: {
        const int bits = v.encoding == DeviceWeightEncoding::AffineQ4 ? 4 : 8;
        require(v.cols % (32 / bits) == 0, "W8 affine packed width mismatch"); return size_t(v.cols) * bits / 8;
    }
    case DeviceWeightEncoding::GgufQ4_0:
        require(v.cols % 32 == 0, "W8 Q4_0 block geometry mismatch"); return size_t(v.cols) / 32 * 18;
    case DeviceWeightEncoding::GgufQ4_K:
        require(v.cols % 256 == 0, "W8 Q4_K block geometry mismatch"); return size_t(v.cols) / 256 * 144;
    case DeviceWeightEncoding::GgufQ8_0:
        require(v.cols % 32 == 0, "W8 Q8_0 block geometry mismatch"); return size_t(v.cols) / 32 * 34;
    case DeviceWeightEncoding::GgufQ6_K:
        require(v.cols % 256 == 0, "W8 Q6_K block geometry mismatch"); return size_t(v.cols) / 256 * 210;
    }
    throw CapabilityError("W8 source encoding unsupported");
}
}
QuantStage Device::stage_w8(DeviceWeightView source, W8StageSpec spec, Surface codes, Surface scales) {
    @autoreleasepool {
      @try {
        require(source.owner && source.buffer && source.rows > 0 && source.rows <= 1048576 && source.cols > 0 && source.cols <= 32768,
                "W8 source binding invalid");
        const size_t row_bytes = w8_source_row_bytes(source);
        const size_t pitch = source.row_stride_bytes ? source.row_stride_bytes : row_bytes;
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)source.buffer;
        require(buffer.device == impl_->device && source.buffer_bytes <= buffer.length && pitch >= row_bytes && pitch <= UINT32_MAX &&
            pitch <= SIZE_MAX / size_t(source.rows) && source.offset_bytes <= source.buffer_bytes &&
            size_t(source.rows - 1) * pitch + row_bytes <= source.buffer_bytes - source.offset_bytes,
            "W8 source device/extent/pitch mismatch");
        require(spec.rows > 0 && spec.rows <= 32768 && spec.row_begin >= 0 && spec.row_begin <= source.rows && spec.rows <= source.rows - spec.row_begin &&
            spec.columns > 0 && spec.column_begin >= 0 && spec.column_begin <= source.cols && spec.columns <= source.cols - spec.column_begin &&
            (spec.rotation_block == 128 || spec.rotation_block == 512) && spec.columns % spec.rotation_block == 0 &&
            spec.column_begin % spec.rotation_block == 0, "W8 slice/rotation geometry mismatch");
        require(codes.element() == Element::I8 && scales.element() == Element::FP16 &&
            codes.rows() == uint32_t(spec.transpose ? spec.columns : spec.rows) && codes.columns() == uint32_t(spec.transpose ? spec.rows : spec.columns) &&
            scales.rows() == uint32_t(spec.transpose ? 1 : spec.rows) && scales.columns() == uint32_t(spec.transpose ? spec.rows : 1),
            "W8 target surface geometry/type mismatch");
        const bool affine = source.encoding == DeviceWeightEncoding::AffineQ4 || source.encoding == DeviceWeightEncoding::AffineQ8;
        require(affine || (!source.scales && !source.offsets), "W8 non-affine source cannot have affine metadata");
        if (affine) {
            require(source.scales && (source.group_size == 32 || source.group_size == 64 || source.group_size == 128 || source.group_size == 256) &&
                source.cols % source.group_size == 0, "W8 affine group/scale mismatch");
            for (const auto *meta : {&source.scales, &source.offsets}) if (*meta) {
                validate_device(**meta, impl_->device, false);
                require((**meta).rows == source.rows && (**meta).cols == source.cols / source.group_size,
                        "W8 affine metadata physical geometry mismatch");
            }
        }
        // Alias includes unused physical storage: an immutable source/metadata
        // must not be overwritten by either target, even for a logical slice.
        auto disjoint = [&](void *native) {
            require((__bridge id)native != codes.impl_->buffer && (__bridge id)native != scales.impl_->buffer,
                    "W8 source/target aliases");
        };
        disjoint(source.buffer);
        for (const auto *meta : {&source.scales, &source.offsets}) if (*meta) disjoint((**meta).buffer);
        require(codes.impl_->buffer != scales.impl_->buffer, "W8 code/scale target aliases");
        const auto pipelines=impl_->pipelines(source.encoding,source.dense_dtype,spec.rotation_block);
        require(pipelines.scales.maxTotalThreadsPerThreadgroup>=uint32_t(spec.rotation_block) &&
                pipelines.codes.maxTotalThreadsPerThreadgroup>=uint32_t(spec.rotation_block),"W8 cached pipeline rotation capacity unsupported");
        auto state = std::make_shared<QuantStage::Impl>(std::move(codes), std::move(scales));
        state->device = impl_; state->source = std::move(source); state->spec = spec;
        state->signs = impl_->signs(spec.rotation_seed);
        state->status = [impl_->device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        state->event = [impl_->device newSharedEvent];
        require(state->status && state->event, "W8 stage status/event allocation failed");
        bool scale_hit=false;
        const DeviceWeightRegion key=weak_scale_key(state->source,spec);
        if(impl_->scale_stats.enabled && state->source.immutable_generation && !spec.transpose && live_scale_key(key)) {
            std::lock_guard lock(impl_->scale_cache_mutex);
            for(auto it=impl_->scale_cache.begin();it!=impl_->scale_cache.end();) {
                if(!live_scale_key((*it)->key)) {impl_->scale_stats.bytes-=(*it)->buffer.length;it=impl_->scale_cache.erase(it);++impl_->scale_stats.evictions;}
                else ++it;
            }
            for(const auto &entry:impl_->scale_cache)if(entry->valid.load(std::memory_order_acquire)&&same_scale_key(entry->key,key)) {
                state->cached_scales=entry;entry->last_use=++impl_->scale_clock;scale_hit=true;++impl_->scale_stats.hits;break;
            }
            if(!scale_hit) {
                ++impl_->scale_stats.misses;const size_t bytes=size_t(spec.rows)*2;
                while(!impl_->scale_cache.empty() && (impl_->scale_cache.size()>=128 || bytes>scale_cache_budget_bytes-impl_->scale_stats.bytes)) {
                    auto oldest=std::min_element(impl_->scale_cache.begin(),impl_->scale_cache.end(),[](const auto&a,const auto&b){return a->last_use<b->last_use;});
                    impl_->scale_stats.bytes-=(*oldest)->buffer.length;impl_->scale_cache.erase(oldest);++impl_->scale_stats.evictions;
                }
                require(bytes<=scale_cache_budget_bytes,"W8 scale metadata exceeds cache cap");
                auto entry=std::make_shared<ScaleCacheEntry>();entry->key=key;entry->last_use=++impl_->scale_clock;
                entry->buffer=[impl_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                require(entry->buffer!=nil,"W8 scale cache allocation failed");
                impl_->scale_stats.bytes+=entry->buffer.length;impl_->scale_cache.push_back(entry);state->cached_scales=std::move(entry);
            }
        }
        const auto &s = state->source;
        const size_t dense_item = s.dense_dtype == DType::FP32 ? 4 : 2;
        const bool source_aligned = s.encoding == DeviceWeightEncoding::Dense &&
            s.offset_bytes % dense_item == 0 && pitch % dense_item == 0;
        W8Params p{uint32_t(pitch), uint32_t(s.cols), uint32_t(s.encoding), uint32_t(s.dense_dtype), uint32_t(s.group_size),
            uint32_t(s.scales ? device_pitch(*s.scales) : 0), uint32_t(s.scales ? s.scales->dtype : DType::FP16),
            uint32_t(s.offsets ? device_pitch(*s.offsets) : 0), uint32_t(s.offsets ? s.offsets->dtype : DType::FP16), uint32_t(s.offsets.has_value()),
            uint32_t(spec.row_begin), uint32_t(spec.rows), uint32_t(spec.column_begin), uint32_t(spec.columns), uint32_t(spec.rotation_block),
            uint32_t(state->codes.pitch()), uint32_t(spec.transpose ? 2 : state->scales.pitch()), uint32_t(spec.transpose),
            uint32_t(source_aligned), uint32_t(spec.rotation_seed), uint32_t(spec.rotation_seed >> 32), 1.f / std::sqrt(float(spec.rotation_block))};
        // A8 must not queue behind an independently prepared future W bank.
        id<MTLCommandBuffer> command = [(spec.transpose ? impl_->activation_queue : impl_->staging_queue) commandBuffer];
        require(command != nil, "W8 stage command buffer unavailable");
        id<MTLBlitCommandEncoder> clear = [command blitCommandEncoder];
        require(clear != nil, "W8 stage clear encoder unavailable");
        [clear fillBuffer:state->status range:NSMakeRange(0, 4) value:0]; [clear endEncoding];
        auto bind = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)s.buffer offset:s.offset_bytes atIndex:0];
            if (s.scales) [encoder setBuffer:(__bridge id<MTLBuffer>)s.scales->buffer offset:s.scales->offset_bytes atIndex:1];
            else [encoder setBuffer:state->status offset:0 atIndex:1];
            if (s.offsets) [encoder setBuffer:(__bridge id<MTLBuffer>)s.offsets->buffer offset:s.offsets->offset_bytes atIndex:2];
            else [encoder setBuffer:state->status offset:0 atIndex:2];
        };
        auto copy_scales=[&](bool restore) {
            id<MTLComputeCommandEncoder> copy=[command computeCommandEncoder];require(copy!=nil,"W8 scale cache encoder unavailable");
            const uint32_t params[4]{uint32_t(spec.rows),restore?1u:uint32_t(state->scales.pitch()/2),restore?uint32_t(state->scales.pitch()/2):1u,0};
            [copy setComputePipelineState:impl_->w8_scale_copy];
            [copy setBuffer:restore?state->cached_scales->buffer:state->scales.impl_->buffer offset:restore?0:size_t(state->scales.row_begin_)*state->scales.pitch() atIndex:0];
            [copy setBuffer:restore?state->scales.impl_->buffer:state->cached_scales->buffer offset:restore?size_t(state->scales.row_begin_)*state->scales.pitch():0 atIndex:1];
            [copy setBytes:params length:sizeof(params) atIndex:2];
            [copy dispatchThreads:MTLSizeMake(spec.rows,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[copy endEncoding];
        };
        if(scale_hit)copy_scales(true);
        else {
            id<MTLComputeCommandEncoder> scale = [command computeCommandEncoder];
            require(scale != nil, "W8 scale encoder unavailable"); bind(scale);
            [scale setComputePipelineState:pipelines.scales];
            [scale setBuffer:state->scales.impl_->buffer offset:size_t(state->scales.row_begin_) * state->scales.pitch() atIndex:3];
            [scale setBuffer:state->status offset:0 atIndex:4]; [scale setBytes:&p length:sizeof(p) atIndex:5];
            [scale setBuffer:state->signs offset:0 atIndex:6];
            [scale dispatchThreadgroups:MTLSizeMake(spec.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(spec.rotation_block, 1, 1)]; [scale endEncoding];
            if(state->cached_scales)copy_scales(false);
        }
        id<MTLComputeCommandEncoder> quant = [command computeCommandEncoder];
        require(quant != nil, "W8 code encoder unavailable"); bind(quant);
        [quant setComputePipelineState:pipelines.codes];
        [quant setBuffer:state->scales.impl_->buffer offset:size_t(state->scales.row_begin_) * state->scales.pitch() atIndex:3];
        [quant setBuffer:state->codes.impl_->buffer offset:size_t(state->codes.row_begin_) * state->codes.pitch() atIndex:4];
        [quant setBuffer:state->status offset:0 atIndex:5]; [quant setBytes:&p length:sizeof(p) atIndex:6];
        [quant setBuffer:state->signs offset:0 atIndex:7];
        [quant dispatchThreadgroups:MTLSizeMake(spec.rows, spec.columns / spec.rotation_block, 1) threadsPerThreadgroup:MTLSizeMake(spec.rotation_block, 1, 1)]; [quant endEncoding];
        [command encodeSignalEvent:state->event value:1];
        [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.status == MTLCommandBufferStatusError && state->event.signaledValue < 1) state->event.signaledValue = 1;
            if(state->cached_scales && !scale_hit)state->cached_scales->valid.store(
                completed.status==MTLCommandBufferStatusCompleted && !*static_cast<const uint32_t*>(state->status.contents),std::memory_order_release);
            { std::lock_guard lock(state->mutex); state->ok = completed.status == MTLCommandBufferStatusCompleted; state->done = true; }
            state->cv.notify_all();
        }];
        [command commit]; QuantStage ticket; ticket.impl_ = std::move(state); return ticket;
      } @catch (NSException *exception) {
        throw CapabilityError("W8 stager exception: " + std::string(exception.reason.UTF8String ?: "unknown"));
      }
    }
}
Completion QuantStage::finish(std::chrono::milliseconds timeout) {
    if (!impl_) return {false, false, "W8 stage ticket empty"};
    std::unique_lock lock(impl_->mutex);
    if (!impl_->cv.wait_for(lock, timeout, [&] { return impl_->done; })) {
        process_healthy = false;
        return {false, true, "W8 GPU stage timeout; resources retained until completion"};
    }
    const uint32_t flags = *static_cast<const uint32_t *>(impl_->status.contents);
    return {impl_->ok && !flags, false, !impl_->ok ? "W8 GPU staging failed" : flags ? "W8 source/rotation/scale/quantization nonfinite or overflow" : ""};
}
uint32_t QuantStage::validation_flags() const {
    require(impl_ && impl_->done, "W8 stage status not ready"); return *static_cast<const uint32_t *>(impl_->status.contents);
}
void *QuantStage::ready_event() const { return impl_ ? (__bridge void *)impl_->event : nullptr; }

struct Program::Impl {
    id<TCAneClient> connection;
    id model;
    std::shared_ptr<Device::Impl> device;
    std::vector<std::string> inputs, outputs;
    std::string key;
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
        impl_->device = device.impl_;
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
bool Program::healthy() { return process_healthy.load(); }
const std::vector<std::string> &Program::inputs() const { return impl_->inputs; }
const std::vector<std::string> &Program::outputs() const { return impl_->outputs; }
const std::string &Program::cache_key() const { return impl_->key; }
bool Program::compiled_now() const { return impl_->compiled; }

struct Ticket::Impl {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    Completion result;
    std::function<void(uint64_t)> release;
    uint64_t signal = 0;
    std::function<void()> failure;
    // The callback owns this state, which owns request/model/event/surfaces.
    // Timeout MUST NOT drop live driver resources or permit their reuse. A
    // missing callback conservatively quarantines this one state until exit.
    std::shared_ptr<void> program;
    std::vector<Surface> surfaces;
    id request;
};
Completion Ticket::finish(std::chrono::milliseconds timeout) {
    if (!impl_) return {false, false, "private ANE empty ticket"};
    std::unique_lock lock(impl_->mutex);
    if (!impl_->cv.wait_for(lock, timeout, [&] { return impl_->done; })) {
        process_healthy = false;
        if (impl_->failure) impl_->failure();
        impl_->release(impl_->signal);
        return {false, true, "private ANE callback timeout; resources quarantined, executor must disable"};
    }
    return impl_->result;
}
void PreparedRequest::discard() noexcept {
    if (!state_) return;
    // No driver work exists until submit consumes state_. A prepared request
    // owns its callback, which owns this state, so explicitly break the cycle.
    { std::lock_guard lock(state_->mutex); state_->request = nil; }
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
    auto state = std::exchange(state_, {});
    auto program = std::static_pointer_cast<Program::Impl>(state->program);
    @autoreleasepool {
      @try {
        NSError *error = nil;
        if (![program->connection evaluateWithModel:program->model options:@{} request:state->request qos:qos error:&error]) {
            process_healthy = false;
            if (state->failure) state->failure();
            state->release(state->signal);
            // A failed submission may still have reached the driver. Retain
            // its callback resources exactly like the inference enqueue path.
            std::lock_guard lock(state->mutex);
            state->result = {false, false, "private ANE enqueue failed: " + description(error)};
            state->done = true;
        }
        Ticket ticket; ticket.impl_ = std::move(state); return ticket;
      } @catch (NSException *exception) {
        process_healthy = false;
        if (state->failure) state->failure();
        state->release(state->signal);
        throw CapabilityError("private ANE enqueue exception: " + std::string(exception.reason.UTF8String ?: "unknown"));
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
    @autoreleasepool {
      auto emergency_failure = failure;
      @try {
        require(healthy(), "private ANE disabled after failed/missing completion");
        require(wait_value && signal_value > wait_value && wait_value >= impl_->device->event.signaledValue,
                "private ANE invalid dependency timeline");
        auto state = std::make_shared<Ticket::Impl>();
        state->failure = std::move(failure);
        state->program = impl_; state->signal = signal_value;
        auto device = impl_->device;
        state->release = [device](uint64_t value) { device->release(value); };
        auto objects = [&](auto bindings, const auto &names) {
            require(bindings.size() == names.size(), "private ANE binding count mismatch");
            NSMutableArray *result = [NSMutableArray new];
            std::set<std::string> observed;
            Class<TCAneSurface> cls = (Class<TCAneSurface>)api_class(@"_ANEIOSurfaceObject", @selector(objectWithIOSurface:));
            for (const auto &name : names) {
                auto found = std::find_if(bindings.begin(), bindings.end(), [&](const auto &b) { return b.first == name; });
                require(found != bindings.end() && observed.insert(name).second, "private ANE binding names mismatch: " + name);
                require(!found->second.row_begin_ && !found->second.row_count_, "private ANE consumer surface slice cannot be a driver binding");
                id object = [cls objectWithIOSurface:found->second.impl_->surface];
                require(object != nil, "private ANE surface binding failed");
                [result addObject:object]; state->surfaces.push_back(found->second);
            }
            return result;
        };
        // Build bindings before any async evaluation. No driver work exists
        // if validation throws here.
        NSArray *in = objects(inputs, impl_->inputs), *out = objects(outputs, impl_->outputs);
        auto indices = [](NSUInteger count) {
            NSMutableArray *result = [NSMutableArray new];
            for (NSUInteger i = 0; i < count; ++i) [result addObject:@(i)];
            return result;
        };
        id wait = [(Class<TCAneEvents>)api_class(@"_ANESharedWaitEvent", @selector(waitEventWithValue:sharedEvent:eventType:))
            waitEventWithValue:wait_value sharedEvent:device->event eventType:0];
        id signal = [(Class<TCAneEvents>)api_class(@"_ANESharedSignalEvent", @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:))
            signalEventWithValue:signal_value symbolIndex:0 eventType:0 sharedEvent:device->event];
        require(wait && signal, "private ANE event binding failed");
        id events = [(Class<TCAneEvents>)api_class(@"_ANESharedEvents", @selector(sharedEventsWithSignalEvents:waitEvents:))
            sharedEventsWithSignalEvents:@[signal] waitEvents:@[wait]];
        require(events != nil, "private ANE shared event binding failed");
        state->request = [(Class<TCAneRequest>)api_class(@"_ANERequest", @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:))
            requestWithInputs:in inputIndices:indices(in.count) outputs:out outputIndices:indices(out.count)
            weightsBuffer:nil perfStats:nil procedureIndex:@0 sharedEvents:events transactionHandle:nil];
        require(state->request && [state->request respondsToSelector:@selector(setCompletionHandler:)], "private ANE async request creation failed");
        [(id<TCAneRequest>)state->request setCompletionHandler:^(BOOL ok, NSError *error) {
            if (!ok) { process_healthy = false; if (state->failure) state->failure(); state->release(state->signal); }
            {
                std::lock_guard lock(state->mutex);
                if (!state->done) { state->result = {bool(ok), false, ok ? "" : description(error)}; state->done = true; }
                state->request = nil; // break callback ownership cycle ONLY on driver completion
            }
            state->cv.notify_all();
        }];
        PreparedRequest prepared; prepared.state_ = std::move(state); return prepared;
      } @catch (NSException *exception) {
        process_healthy = false;
        if (emergency_failure) emergency_failure();
        impl_->device->release(signal_value);
        throw CapabilityError("private ANE enqueue exception: " + std::string(exception.reason.UTF8String ?: "unknown"));
      }
    }
}
} // namespace tc::ane::private_api
