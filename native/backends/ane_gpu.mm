// Shared GPU implementation extracted from the verified Private route.
// Math, typed storage, weak generations, bounded banks and device ordering
// remain unchanged. Public Core ML can use it without any private selector.
#include "ane_gpu.hpp"
#include "ane_transfer_kernels.hpp"
#include "ane_w8_kernels.hpp"
#include "ane_w8a8_math.hpp"
#include "ane_weight_code_cache.hpp"
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <functional>
#include <mutex>
#include <tuple>
#include <array>
#include <map>
namespace tc::ane::gpu {
namespace {
std::atomic<bool> gpu_process_healthy{true};
void require(bool ok,const std::string &error) { if(!ok)throw CapabilityError(error); }
std::string description(NSError *error) { return error?std::string(error.localizedDescription.UTF8String):"no GPU diagnostic"; }
}
namespace {
bool same_generation(const std::weak_ptr<void>&a,const std::weak_ptr<void>&b) {
    return !a.owner_before(b)&&!b.owner_before(a);
}
bool has_generation(const std::weak_ptr<void> &value) {
    return !same_generation(value,{}); // distinguishes absent from expired
}
bool same_scale_matrix(const DeviceMatrixView&a,const DeviceMatrixView&b) {
    return same_generation(a.allocation_identity,b.allocation_identity)&&
        std::tie(a.buffer,a.buffer_bytes,a.offset_bytes,a.rows,a.cols,a.row_stride_bytes,a.dtype)==
        std::tie(b.buffer,b.buffer_bytes,b.offset_bytes,b.rows,b.cols,b.row_stride_bytes,b.dtype);
}
bool same_scale_key(const DeviceWeightRegion&a,const DeviceWeightRegion&b) {
    const auto &x=a.source,&y=b.source;const auto &p=a.selection,&q=b.selection;
    const auto meta=[](const auto&a,const auto&b){return bool(a)==bool(b)&&(!a||same_scale_matrix(*a,*b));};
    const bool logical=has_generation(x.logical_content_identity);
    if(logical!=has_generation(y.logical_content_identity))return false;
    if(logical) {
        if(!same_generation(x.logical_content_identity,y.logical_content_identity))return false;
    } else if(!same_generation(x.allocation_identity,y.allocation_identity) ||
        std::tie(x.buffer,x.buffer_bytes,x.offset_bytes,x.row_stride_bytes)!=
        std::tie(y.buffer,y.buffer_bytes,y.offset_bytes,y.row_stride_bytes))return false;
    return meta(x.scales,y.scales)&&meta(x.offsets,y.offsets)&&
        std::tie(x.rows,x.cols,x.encoding,x.dense_dtype,x.group_size)==
        std::tie(y.rows,y.cols,y.encoding,y.dense_dtype,y.group_size)&&
        std::tie(p.row_begin,p.rows,p.column_begin,p.columns,p.rotation_block,p.rotation_seed,p.transpose,p.basis,p.activation_group_size)==
        std::tie(q.row_begin,q.rows,q.column_begin,q.columns,q.rotation_block,q.rotation_seed,q.transpose,q.basis,q.activation_group_size);
}
bool live_scale_key(const DeviceWeightRegion&key) {
    return (has_generation(key.source.logical_content_identity) ? !key.source.logical_content_identity.expired() :
        !key.source.allocation_identity.expired()) &&
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
struct CodeCacheEntry {
    DeviceWeightRegion key;
    std::shared_ptr<WeightCodeCacheClaim> claim;
    id<MTLBuffer> codes, scales;
    std::optional<Surface> native_codes, native_scales;
    std::atomic<bool> valid{false}, pending{true};
    ~CodeCacheEntry() {
        codes = nil; scales = nil;
    }
    uint64_t capacity() const { return claim ? claim->bytes() : 0; }
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
    struct W8Pipelines { id<MTLComputePipelineState> scales, codes; bool register_comfy,register_sylvester; };
    std::mutex pipelines_mutex;
    // At most 18 Sylvester + 3 Comfy row-A + 3 group-A + 2 direct W variants.
    // Keys retain no source allocation or model/adapter values.
    std::map<std::array<uint32_t,4>,W8Pipelines> w8_pipelines;
    W8Pipelines pipelines(DeviceWeightEncoding encoding,DType dtype,int block,bool grouped) {
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
        std::array<uint32_t,4> key{0,0,0,uint32_t(grouped)};
        const bool direct = encoding==DeviceWeightEncoding::ConvrotQ8Signed || encoding==DeviceWeightEncoding::ConvrotQ8Packed;
        // Dense H256 uniquely identifies validated Comfy A8 (not W or
        // Sylvester). Existing specialization opt-in keeps generic control.
        const bool register_comfy=specialize_staging && encoding==DeviceWeightEncoding::Dense && block==256;
        // The validated affine/raw GGUF decoder feeds the same FP32
        // butterfly directly; never expand a full dense matrix first.
        // Direct ConvRot W is already rotated and must not enter this path.
        const bool register_sylvester=specialize_staging && !direct && (block==128 || block==512);
        if(specialize_staging || direct)key={uint32_t(encoding),encoding==DeviceWeightEncoding::Dense?uint32_t(dtype):0u,uint32_t(block),uint32_t(grouped)};
        auto found=w8_pipelines.find(key);if(found!=w8_pipelines.end())return found->second;
        require(w8_pipelines.size()<26,"W8 pipeline variant bound exceeded");
        MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
        const bool specialized=specialize_staging;
        [constants setConstantValue:&specialized type:MTLDataTypeBool atIndex:0];
        for(NSUInteger index=0;index<3;++index)[constants setConstantValue:&key[index] type:MTLDataTypeUInt atIndex:index+1];
        W8Pipelines result;result.register_comfy=register_comfy;result.register_sylvester=register_sylvester;
        auto make=[&](NSString *name) {
            id<MTLFunction> function=[w8_library newFunctionWithName:name constantValues:constants error:&error];
            require(function!=nil,"W8 specialized function unavailable: "+description(error));
            id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline!=nil && pipeline.threadExecutionWidth==32 && pipeline.maxTotalThreadsPerThreadgroup>=uint32_t(register_comfy||register_sylvester?32:block),
                    "W8 pipeline rotation/SIMD capacity unsupported");
            return pipeline;
        };
        result.scales=make(direct?@"tc_ane_convrot_scales":register_comfy?@"tc_ane_comfy_register_scales":register_sylvester?@"tc_ane_sylvester_register_scales":grouped?@"tc_ane_w8_group_scales":@"tc_ane_w8_scales");
        result.codes=make(direct?@"tc_ane_convrot_codes":register_comfy?@"tc_ane_comfy_register_codes":register_sylvester?@"tc_ane_sylvester_register_codes":@"tc_ane_w8_codes");
        w8_pipelines.emplace(key,result);return result;
    }
    std::mutex scale_cache_mutex;
    std::vector<std::shared_ptr<ScaleCacheEntry>> scale_cache;
    WeightCacheStats scale_stats;
    uint64_t scale_clock=0;
    std::mutex code_cache_mutex;
    std::vector<std::shared_ptr<CodeCacheEntry>> code_cache;
    std::shared_ptr<WeightCodeCacheLedger> code_ledger;
    WeightCodeCacheReport code_stats;
    std::pair<std::shared_ptr<CodeCacheEntry>,bool> code_binding(const DeviceWeightRegion &key,
                                                               bool eligible,Device &outer) {
        std::lock_guard lock(code_cache_mutex);
        if(!code_stats.enabled)return {{},false};
        // First-admitted live generations stay pinned. A smaller cache must
        // not thrash across a complete 32-layer traversal and have zero hits.
        for(auto it=code_cache.begin();it!=code_cache.end();) {
            const auto &entry=*it;
            if(!live_scale_key(entry->key) || (!entry->pending.load(std::memory_order_acquire) &&
                                              !entry->valid.load(std::memory_order_acquire))) {
                it=code_cache.erase(it);++code_stats.evictions;
            } else ++it;
        }
        if(!eligible) {++code_stats.ineligible;return {{},false};}
        for(const auto &entry:code_cache)if(same_scale_key(entry->key,key)) {
            if(entry->valid.load(std::memory_order_acquire))return {entry,true};
            ++code_stats.misses;++code_stats.declines;return {{},false}; // pending producer, no unsafe hit or duplicate
        }
        ++code_stats.misses;
        const auto plan=weight_code_cache_plan(key.selection.rows,key.selection.columns,uint64_t(getpagesize()),code_stats.native_surface_storage);
        if(!plan || code_cache.size()>=weight_code_cache_max_entries || !code_ledger->reserve(plan->allocation_upper)) {
            ++code_stats.declines;return {{},false};
        }
        std::shared_ptr<CodeCacheEntry> entry;
        try {
            entry=std::make_shared<CodeCacheEntry>();
            entry->claim=std::make_shared<WeightCodeCacheClaim>(code_ledger,plan->allocation_upper);
        }
        catch(...) {code_ledger->release(plan->allocation_upper);throw;}
        entry->key=key;
        uint64_t actual=0;
        if(code_stats.native_surface_storage) {
            entry->native_codes.emplace(outer,key.selection.rows,key.selection.columns,Element::I8);
            entry->native_scales.emplace(outer,key.selection.rows,1,Element::FP16);
            actual=entry->native_codes->cache_capacity_bytes()+entry->native_scales->cache_capacity_bytes();
            entry->native_codes->retain_cache_claim(entry->claim);
            entry->native_scales->retain_cache_claim(entry->claim);
        } else {
            entry->codes=[device newBufferWithLength:plan->codes_bytes options:MTLResourceStorageModeShared];
            entry->scales=[device newBufferWithLength:plan->scales_bytes options:MTLResourceStorageModeShared];
            if(!entry->codes || !entry->scales) {++code_stats.declines;return {{},false};}
            actual=entry->codes.allocatedSize+entry->scales.allocatedSize;
        }
        if(actual<plan->codes_bytes+plan->scales_bytes || actual>plan->allocation_upper) {
            ++code_stats.declines;return {{},false};
        }
        entry->claim->commit(actual);
        code_cache.push_back(entry);return {entry,false};
    }
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
namespace {
bool configured_flag(const char *name,bool fallback) {
    const char *raw=std::getenv(name);
    require(!raw || std::string(raw)=="0" || std::string(raw)=="1",std::string(name)+" requires 0 or 1");
    return raw?std::string(raw)=="1":fallback;
}
}
Device::Device() : Device(configured_flag("TURBOCIDER_PRIVATE_ANE_SCALE_CACHE",true),
                         configured_flag("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE",false)) {}
uint64_t configured_weight_code_cache_bytes() {
    return parse_weight_code_cache_bytes(std::getenv("TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES"));
}
Device::Device(bool scale_cache,bool specialize) : Device(scale_cache,specialize,configured_weight_code_cache_bytes(),
    parse_weight_code_cache_storage(std::getenv("TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_MODE"))) {}
Device::Device(bool scale_cache,bool specialize,uint64_t code_bytes) : Device(scale_cache,specialize,code_bytes,WeightCodeCacheStorage::CompactCopy) {}
Device::Device(bool scale_cache,bool specialize,uint64_t code_bytes,WeightCodeCacheStorage storage) : impl_(std::make_shared<Impl>()) {
    impl_->code_ledger=std::make_shared<WeightCodeCacheLedger>(code_bytes);
    impl_->code_stats.enabled=code_bytes>0;impl_->code_stats.budget_bytes=code_bytes;
    impl_->code_stats.native_surface_storage=storage==WeightCodeCacheStorage::NativeSurface;
    require(impl_->device != nil, "ANE GPU Metal unavailable");
    impl_->queue = [impl_->device newCommandQueueWithMaxCommandBufferCount:256];
    impl_->staging_queue = [impl_->device newCommandQueueWithMaxCommandBufferCount:256];
    impl_->activation_queue = [impl_->device newCommandQueueWithMaxCommandBufferCount:256];
    impl_->event = [impl_->device newSharedEvent];
    require(impl_->queue && impl_->staging_queue && impl_->activation_queue && impl_->event, "ANE GPU shared-event queue unavailable");
    impl_->scale_stats.enabled=scale_cache;impl_->specialize_staging=specialize;
}
bool healthy() { return gpu_process_healthy.load(); }
void *Device::shared_event() const { return (__bridge void*)impl_->event; }
void Device::release_after_failure(uint64_t value) const { impl_->release(value); }
WeightCacheStats Device::scale_cache_stats() const {
    std::lock_guard lock(impl_->scale_cache_mutex);auto stats=impl_->scale_stats;stats.entries=impl_->scale_cache.size();return stats;
}
WeightCodeCacheReport Device::weight_code_cache_stats() const {
    std::lock_guard lock(impl_->code_cache_mutex);
    auto stats=impl_->code_stats;
    stats.entries=impl_->code_cache.size();
    for(const auto &entry:impl_->code_cache) {
        stats.ready_entries+=entry->valid.load(std::memory_order_acquire);
        stats.retained_bytes+=entry->capacity();
    }
    stats.live_capacity_bytes=impl_->code_ledger->live();
    stats.peak_capacity_bytes=impl_->code_ledger->peak();
    return stats;
}
uint64_t Device::weight_code_cache_budget_bytes() const { return impl_->code_stats.budget_bytes; }
void Device::clear_weight_code_cache() {
    std::lock_guard lock(impl_->code_cache_mutex);
    impl_->code_stats.evictions+=impl_->code_cache.size();impl_->code_cache.clear();
}
StagePipelineStats Device::stage_pipeline_stats() const {
    std::lock_guard lock(impl_->pipelines_mutex);
    return {impl_->specialize_staging,impl_->w8_pipelines.size()};
}
std::string Device::name() const { return impl_->device.name.UTF8String; }
uint64_t Device::value() const { return impl_->event.signaledValue; }
void Device::signal(uint64_t value) {
    require(value > impl_->last_signal && value > impl_->event.signaledValue, "ANE GPU event timeline not monotonic");
    id<MTLCommandBuffer> buffer = [impl_->queue commandBuffer];
    require(buffer != nil, "ANE GPU signal command buffer unavailable");
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
            "ANE GPU prepared event timeline not monotonic");
    impl_->last_signal = value;
    impl_->release(value);
}
bool Device::wait(uint64_t value, std::chrono::milliseconds timeout) {
    // Wait is queued AFTER the leading signal has already been committed.
    // A host deadline releases a stuck wait, but it NEVER certifies success.
    id<MTLCommandBuffer> buffer = [impl_->queue commandBuffer];
    require(buffer != nil, "ANE GPU wait command buffer unavailable");
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
    std::shared_ptr<WeightCodeCacheClaim> cache_claim;
    IOSurfaceRef surface = nullptr;
    id<MTLBuffer> buffer;
    uint32_t rows = 0, columns = 0;
    Element element;
    ~Impl() { buffer = nil; if (surface) CFRelease(surface); }
};
Surface::Surface(Device &device, uint32_t rows, uint32_t columns, Element element) : impl_(std::make_shared<Impl>()) {
    require(rows && rows <= 32768 && columns && columns <= 32768 &&
            (element == Element::FP16 || element == Element::I8), "ANE GPU invalid surface geometry");
    const size_t item = element == Element::I8 ? 1 : 2;
    const size_t pitch = (size_t(columns) * item + 63) / 64 * 64;
    const size_t page = size_t(getpagesize());
    const size_t allocation = (size_t(rows) * pitch + page - 1) / page * page;
    impl_->surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(columns), (id)kIOSurfaceHeight:@(rows),
        (id)kIOSurfaceBytesPerElement:@(item), (id)kIOSurfaceBytesPerRow:@(pitch),
        (id)kIOSurfaceAllocSize:@(allocation),
        (id)kIOSurfacePixelFormat:@(element == Element::I8 ? 0x4c303038 : 0x4c303068)});
    require(impl_->surface != nullptr, "ANE GPU IOSurface allocation failed");
    impl_->rows = rows; impl_->columns = columns; impl_->element = element;
    require(IOSurfaceGetBytesPerRow(impl_->surface) == pitch && IOSurfaceGetAllocSize(impl_->surface) >= allocation,
            "ANE GPU IOSurface allocation changed layout");
    impl_->buffer = [device.impl_->device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(impl_->surface)
        length:IOSurfaceGetAllocSize(impl_->surface) options:MTLResourceStorageModeShared deallocator:nil];
    require(impl_->buffer != nil, "ANE GPU cannot share IOSurface with Metal");
}
size_t Surface::bytes() const { return impl_->buffer.allocatedSize; }
size_t Surface::cache_capacity_bytes() const {
    return std::max(size_t(impl_->buffer.allocatedSize),size_t(IOSurfaceGetAllocSize(impl_->surface)));
}
void Surface::retain_cache_claim(std::shared_ptr<WeightCodeCacheClaim> claim) { impl_->cache_claim=std::move(claim); }
size_t Surface::pitch() const { return IOSurfaceGetBytesPerRow(impl_->surface); }
uint32_t Surface::rows() const { return row_count_ ? row_count_ : impl_->rows; }
uint32_t Surface::columns() const { return impl_->columns; }
Element Surface::element() const { return impl_->element; }
void *Surface::data() const { return static_cast<char *>(IOSurfaceGetBaseAddress(impl_->surface)) + size_t(row_begin_) * pitch(); }
void *Surface::native_iosurface() const { return impl_->surface; }
Surface Surface::slice_rows(uint32_t begin, uint32_t count) const {
    require(count && begin <= rows() && count <= rows() - begin, "ANE GPU surface slice extent invalid");
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
    bool submitted = false, done = false, ok = false, independent_gpu = false;
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
        (v.dtype == DType::FP16 || v.dtype == DType::BF16 || v.dtype == DType::FP32),
        output ? "ANE GPU invalid device output binding/type/owner" : "ANE GPU invalid device binding/type/owner");
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)v.buffer;
    const size_t item = v.dtype == DType::FP32 ? 4 : 2, pitch = device_pitch(v), row = size_t(v.cols) * item;
    require(buffer.device == device && v.buffer_bytes <= buffer.length && v.offset_bytes % item == 0 &&
        pitch % item == 0 && pitch >= row && pitch <= UINT32_MAX && pitch <= SIZE_MAX / size_t(v.rows) &&
        v.offset_bytes <= v.buffer_bytes && size_t(v.rows - 1) * pitch + row <= v.buffer_bytes - v.offset_bytes,
        "ANE GPU device binding extent/device mismatch");
}
struct TransferParams { uint32_t rows, cols, source_pitch, target_pitch, dtype, validate_only; float scale; uint32_t row_scale_pitch, scaled, second_scaled, signed_row_scale; };
}
Transfer Device::prepare_transfer(std::vector<Upload> uploads, std::vector<Download> downloads,
                                  uint64_t ready, uint64_t done) {
    return prepare_transfer_impl(std::move(uploads), std::move(downloads), std::pair{ready, done});
}
Transfer Device::prepare_gpu_transfer(std::vector<Upload> uploads, std::vector<Download> downloads) {
    return prepare_transfer_impl(std::move(uploads), std::move(downloads), std::nullopt);
}
Transfer Device::prepare_upload(std::vector<Upload> uploads) {
    return prepare_transfer_impl(std::move(uploads),{},std::nullopt,true);
}
Transfer Device::prepare_transfer_impl(std::vector<Upload> uploads, std::vector<Download> downloads,
                                      std::optional<std::pair<uint64_t, uint64_t>> dependency,bool upload_only) {
    @autoreleasepool {
        require((!downloads.empty() || (upload_only && !dependency && !uploads.empty())) && (!dependency ||
                (dependency->first > impl_->last_signal && dependency->first > value() &&
                 dependency->second >= dependency->first)),
                "ANE GPU invalid transfer timeline/bindings");
        for (const auto &u : uploads) {
            validate_device(u.source, impl_->device, false);
            require(!u.destination.impl_->cache_claim,"cached weight surface cannot be an upload destination");
            require(u.destination.element() == Element::FP16 && u.source.cols == int(u.destination.rows()) &&
                u.begin_row >= 0 && u.begin_row <= u.source.rows && int(u.destination.columns()) <= u.source.rows - u.begin_row &&
                std::isfinite(u.scale) && u.scale > 0, "ANE GPU upload geometry/scale mismatch");
        }
        for (const auto &d : downloads) {
            if (d.second_token_scales) require(d.second_token_scales->element() == Element::FP16 && d.second_token_scales->rows() == 1 &&
                d.second_token_scales->columns() == d.source.columns(), "ANE GPU secondary scale geometry mismatch");
            require(bool(d.row_scales) == bool(d.token_scales), "ANE GPU W8 epilogue requires both scales");
            require(d.row_scale_policy==RowScalePolicy::Positive ||
                (d.row_scale_policy==RowScalePolicy::SignedFinite && d.row_scales),
                "ANE GPU signed row-scale policy requires explicit row/token scales");
            if (d.row_scales) require(d.row_scales->element() == Element::FP16 && d.row_scales->rows() == d.source.rows() &&
                d.row_scales->columns() == 1 && d.token_scales->element() == Element::FP16 && d.token_scales->rows() == 1 &&
                d.token_scales->columns() == d.source.columns(), "ANE GPU W8 epilogue scale geometry mismatch");
            require(d.source.element() == Element::FP16 && d.begin_row >= 0 && std::isfinite(d.scale) && d.scale > 0 &&
                (d.dtype == DType::FP16 || d.dtype == DType::BF16 || d.dtype == DType::FP32), "ANE GPU download geometry/dtype/scale mismatch");
            if (d.destination) {
                validate_device(*d.destination, impl_->device, true);
                require(d.destination->dtype == d.dtype && d.destination->cols == int(d.source.rows()) &&
                    d.begin_row <= d.destination->rows && int(d.source.columns()) <= d.destination->rows - d.begin_row,
                    "ANE GPU download destination mismatch");
            }
        }
        if (!impl_->upload || !impl_->restore) {
            NSError *error = nil;
            MTLCompileOptions *options = [MTLCompileOptions new]; options.fastMathEnabled = NO;
            id<MTLLibrary> library = [impl_->device newLibraryWithSource:@(transfer_source) options:options error:&error];
            require(library != nil, "ANE GPU transfer shader compile failed: " + description(error));
            impl_->upload = [impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tc_ane_upload"] error:&error];
            require(impl_->upload != nil, "ANE GPU upload pipeline failed: " + description(error));
            impl_->restore = [impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tc_ane_restore"] error:&error];
            require(impl_->restore != nil, "ANE GPU restore pipeline failed: " + description(error));
        }
        auto state = std::make_shared<Transfer::Impl>();
        state->device = impl_; state->uploads = std::move(uploads); state->downloads = std::move(downloads);
        state->independent_gpu = !dependency;
        if (dependency) { state->ready = dependency->first; state->done_value = dependency->second; }
        state->status = [impl_->device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        state->failed = [impl_->device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        require(state->status && state->failed, "ANE GPU transfer status allocation failed");
        *static_cast<uint32_t *>(state->failed.contents) = 0;
        Transfer result; result.impl_ = std::move(state); return result;
    }
}
std::function<void()> Transfer::failure_callback() const {
    auto state = impl_; return [state] { state->fail(); };
}
bool Transfer::independent_gpu() const {
    require(bool(impl_), "ANE GPU empty transfer");
    return impl_->independent_gpu;
}
void Transfer::submit() {
    @autoreleasepool {
      @try {
        auto state = impl_;
        require(state && !state->submitted, "ANE GPU transfer already submitted");
        auto dev = state->device;
        require(state->independent_gpu || state->ready > dev->last_signal, "ANE GPU transfer timeline reused");
        id<MTLCommandBuffer> input = [dev->queue commandBuffer];
        id<MTLCommandBuffer> output = state->independent_gpu ? input : [dev->queue commandBuffer];
        require(input && output, "ANE GPU transfer command buffer unavailable");
        id<MTLBlitCommandEncoder> clear = [input blitCommandEncoder];
        require(clear != nil, "ANE GPU transfer blit encoder unavailable");
        [clear fillBuffer:state->status range:NSMakeRange(0, 4) value:0]; [clear endEncoding];
        id<MTLComputeCommandEncoder> pack = [input computeCommandEncoder];
        require(pack != nil, "ANE GPU upload encoder unavailable");
        [pack setComputePipelineState:dev->upload];
        for (const auto &u : state->uploads) {
            const auto &v = u.source;
            TransferParams p{u.destination.columns(), u.destination.rows(), uint32_t(device_pitch(v)),
                uint32_t(u.destination.pitch()), uint32_t(v.dtype), 0, u.scale, 0, 0, 0, 0};
            [pack setBuffer:(__bridge id<MTLBuffer>)v.buffer offset:v.offset_bytes + size_t(u.begin_row) * device_pitch(v) atIndex:0];
            [pack setBuffer:u.destination.impl_->buffer offset:size_t(u.destination.row_begin_) * u.destination.pitch() atIndex:1];
            [pack setBuffer:state->status offset:0 atIndex:2]; [pack setBytes:&p length:sizeof(p) atIndex:3];
            [pack dispatchThreadgroups:MTLSizeMake((p.rows + 31) / 32, (p.cols + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        }
        [pack endEncoding];
        if (!state->independent_gpu) [input encodeSignalEvent:dev->event value:state->ready];
        [input addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.status == MTLCommandBufferStatusError) {
                state->fail();
                if (!state->independent_gpu) dev->release(state->done_value);
            }
        }];
        // Construct both encoders BEFORE committing. After leading commit,
        // no exception may leave an untracked GPU consumer or ANE wait.
        if (!state->independent_gpu) [output encodeWaitForEvent:dev->event value:state->done_value];
        id<MTLComputeCommandEncoder> restore = [output computeCommandEncoder];
        require(restore != nil, "ANE GPU restore encoder unavailable");
        [restore setComputePipelineState:dev->restore];
        for (const auto &d : state->downloads) {
            TransferParams p{d.source.columns(), d.source.rows(), uint32_t(d.source.pitch()),
                uint32_t(d.destination ? device_pitch(*d.destination) : 2), uint32_t(d.dtype), uint32_t(!d.destination), d.scale,
                uint32_t(d.row_scales ? d.row_scales->pitch() : 0), uint32_t(d.row_scales.has_value()), uint32_t(d.second_token_scales.has_value()),
                uint32_t(d.row_scale_policy==RowScalePolicy::SignedFinite)};
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
        state->submitted = true;
        if (!state->independent_gpu) dev->last_signal = state->ready;
        [input commit]; // leading signal must be submitted before dependent wait
        if (!state->independent_gpu) [output commit];
      } @catch (NSException *exception) {
        if (impl_) { impl_->fail(); if (!impl_->independent_gpu) impl_->device->release(impl_->done_value); }
        gpu_process_healthy = false;
        throw CapabilityError("ANE GPU GPU transfer exception: " + std::string(exception.reason.UTF8String ?: "unknown"));
      }
    }
}
Completion Transfer::finish(std::chrono::milliseconds timeout) {
    if (!impl_ || !impl_->submitted) return {false, false, "ANE GPU transfer not submitted"};
    std::unique_lock lock(impl_->mutex);
    if (!impl_->cv.wait_for(lock, timeout, [&] { return impl_->done; })) {
        impl_->fail(); gpu_process_healthy = false;
        if (!impl_->independent_gpu) impl_->device->release(impl_->done_value);
        return {false, true, "ANE GPU transfer timeout; resources retained until GPU completion"};
    }
    const bool failed = std::atomic_ref<uint32_t>(*static_cast<uint32_t *>(impl_->failed.contents)).load(std::memory_order_acquire);
    return {impl_->ok && !failed, false, !impl_->ok ? "ANE GPU GPU transfer failed" : failed ? "ANE GPU transfer suppressed after failure" : ""};
}
uint32_t Transfer::validation_flags() const {
    require(impl_ && impl_->done, "ANE GPU transfer status not ready");
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
    std::shared_ptr<CodeCacheEntry> cached_codes;
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
    uint32_t source_aligned, seed_low, seed_high, basis, activation_group;
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
    case DeviceWeightEncoding::ConvrotQ8Signed: case DeviceWeightEncoding::ConvrotQ8Packed:
        require(v.cols % 256 == 0, "W8 ConvRot requires complete H256 groups"); return size_t(v.cols);
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
QuantStage Device::stage_w8(DeviceWeightView source,W8StageSpec spec,Surface codes,Surface scales) {
    return stage_w8_impl(std::move(source),spec,std::move(codes),std::move(scales),nullptr,nullptr);
}
QuantStage Device::bind_w8(DeviceWeightView source,W8StageSpec spec,Surface &codes,Surface &scales) {
    return stage_w8_impl(std::move(source),spec,codes,scales,&codes,&scales);
}
QuantStage Device::stage_w8_impl(DeviceWeightView source, W8StageSpec spec, Surface codes, Surface scales,
                               Surface *bound_codes,Surface *bound_scales) {
    @autoreleasepool {
      @try {
        require(source.owner && source.buffer && source.rows > 0 && source.rows <= 1048576 && source.cols > 0 && source.cols <= 32768,
                "W8 source binding invalid");
        const size_t row_bytes = w8_source_row_bytes(source);
        const size_t pitch = source.row_stride_bytes ? source.row_stride_bytes : row_bytes;
        const bool comfy = spec.basis==W8Basis::ComfyH256;
        const bool grouped=spec.activation_group_size==256;
        require(spec.activation_group_size==0 || (grouped && comfy && spec.transpose && source.encoding==DeviceWeightEncoding::Dense),
                "group A8 requires explicit Comfy dense activation staging");
        const bool direct = source.encoding==DeviceWeightEncoding::ConvrotQ8Signed || source.encoding==DeviceWeightEncoding::ConvrotQ8Packed;
        require((spec.basis==W8Basis::SylvesterDH || comfy) &&
            (!comfy ? !direct : spec.rotation_block==256 && spec.rotation_seed==0 &&
                (direct ? !spec.transpose : spec.transpose && source.encoding==DeviceWeightEncoding::Dense)),
            "W8 Comfy direct-W/activation basis mismatch");
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)source.buffer;
        require(buffer.device == impl_->device && source.buffer_bytes <= buffer.length && pitch >= row_bytes && pitch <= UINT32_MAX &&
            pitch <= SIZE_MAX / size_t(source.rows) && source.offset_bytes <= source.buffer_bytes &&
            size_t(source.rows - 1) * pitch + row_bytes <= source.buffer_bytes - source.offset_bytes,
            "W8 source device/extent/pitch mismatch");
        require(spec.rows > 0 && spec.rows <= 32768 && spec.row_begin >= 0 && spec.row_begin <= source.rows && spec.rows <= source.rows - spec.row_begin &&
            spec.columns > 0 && spec.column_begin >= 0 && spec.column_begin <= source.cols && spec.columns <= source.cols - spec.column_begin &&
            (comfy ? spec.rotation_block==256 : spec.rotation_block == 128 || spec.rotation_block == 512) && spec.columns % spec.rotation_block == 0 &&
            spec.column_begin % spec.rotation_block == 0, "W8 slice/rotation geometry mismatch");
        require(codes.element() == Element::I8 && scales.element() == Element::FP16 &&
            codes.impl_->buffer.device==impl_->device && scales.impl_->buffer.device==impl_->device &&
            codes.rows() == uint32_t(spec.transpose ? spec.columns : spec.rows) && codes.columns() == uint32_t(spec.transpose ? spec.rows : spec.columns) &&
            scales.rows() == uint32_t(spec.transpose ? (grouped?spec.columns/256:1) : spec.rows) && scales.columns() == uint32_t(spec.transpose ? spec.rows : 1),
            "W8 target surface geometry/type mismatch");
        const bool packed_comfy = source.encoding==DeviceWeightEncoding::ConvrotQ8Packed;
        const bool affine = source.encoding == DeviceWeightEncoding::AffineQ4 || source.encoding == DeviceWeightEncoding::AffineQ8 || packed_comfy;
        const bool raw=source.encoding==DeviceWeightEncoding::GgufQ4_0 || source.encoding==DeviceWeightEncoding::GgufQ4_K ||
            source.encoding==DeviceWeightEncoding::GgufQ8_0 || source.encoding==DeviceWeightEncoding::GgufQ6_K;
        require(!has_generation(source.logical_content_identity) || (raw && source.immutable_generation),
                "W8 logical content identity requires immutable raw GGUF source");
        require(affine || direct || (!source.scales && !source.offsets), "W8 non-affine source cannot have affine metadata");
        if (affine) {
            require(source.scales && (source.group_size == 32 || source.group_size == 64 || source.group_size == 128 || source.group_size == 256) &&
                source.cols % source.group_size == 0, "W8 affine group/scale mismatch");
            for (const auto *meta : {&source.scales, &source.offsets}) if (*meta) {
                validate_device(**meta, impl_->device, false);
                require((**meta).rows == source.rows && (**meta).cols == source.cols / source.group_size,
                        "W8 affine metadata physical geometry mismatch");
            }
        }
        if (packed_comfy) require(source.offsets.has_value(), "W8 packed ConvRot requires signed offsets");
        if (source.encoding==DeviceWeightEncoding::ConvrotQ8Signed) {
            require(source.scales && !source.offsets, "W8 signed ConvRot requires original row scales only");
            validate_device(*source.scales,impl_->device,false);
            require(source.scales->rows==source.rows && source.scales->cols==1 && source.scales->dtype==DType::FP32,
                    "W8 signed ConvRot requires one FP32 scale per physical row");
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
        require(!codes.impl_->cache_claim && !scales.impl_->cache_claim,
                "cached W8 surfaces are immutable readers; reset the bank to fixed scratch before staging");
        const auto pipelines=impl_->pipelines(source.encoding,source.dense_dtype,spec.rotation_block,grouped);
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
        const auto code_binding=impl_->code_binding(key,state->source.immutable_generation &&
            !spec.transpose && live_scale_key(key) && state->codes.pitch()==size_t(spec.columns),*this);
        const bool code_hit=code_binding.second;
        state->cached_codes=code_binding.first;
        const bool bind_native=bound_codes && bound_scales && state->cached_codes && state->cached_codes->native_codes;
        if(bind_native) {
            state->codes=*state->cached_codes->native_codes;state->scales=*state->cached_codes->native_scales;
            if(code_hit) {
                // Source/metadata/target validation above still applies on a
                // hit, including physical raw-refill extents and aliases.
                // These cached surfaces were already produced and validated.
                *static_cast<uint32_t*>(state->status.contents)=0;
                state->event.signaledValue=1;state->ok=true;state->done=true;
                *bound_codes=state->codes;*bound_scales=state->scales;
                {std::lock_guard cache_lock(impl_->code_cache_mutex);++impl_->code_stats.hits;++impl_->code_stats.surface_bind_hits;}
                QuantStage ticket;ticket.impl_=std::move(state);return ticket;
            }
        }
        bool code_submitted=false;
        struct UnsubmittedCodeFill {
            std::shared_ptr<QuantStage::Impl> state;
            bool &submitted;
            bool hit;
            ~UnsubmittedCodeFill() {
                if(submitted || hit || !state->cached_codes)return;
                // A prepare/encode/commit exception must not pin a forever
                // pending entry. A possibly submitted CB still retains the
                // ticket/entry and its capacity until actual completion.
                auto entry=state->cached_codes;
                entry->valid.store(false,std::memory_order_release);
                entry->pending.store(false,std::memory_order_release);
                std::lock_guard cache_lock(state->device->code_cache_mutex);
                auto &entries=state->device->code_cache;
                auto at=std::find(entries.begin(),entries.end(),entry);
                if(at!=entries.end()) {entries.erase(at);++state->device->code_stats.evictions;}
                ++state->device->code_stats.failed_fills;
            }
        } unsubmitted{state,code_submitted,code_hit};
        if(!code_hit && impl_->scale_stats.enabled && state->source.immutable_generation && !spec.transpose && !direct && live_scale_key(key)) {
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
            uint32_t(state->codes.pitch()), uint32_t(spec.transpose && !grouped ? 2 : state->scales.pitch()), uint32_t(spec.transpose),
            uint32_t(source_aligned), uint32_t(spec.rotation_seed), uint32_t(spec.rotation_seed >> 32), uint32_t(spec.basis),uint32_t(spec.activation_group_size), 1.f / std::sqrt(float(spec.rotation_block))};
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
        auto copy_scales=[&](bool restore,id<MTLBuffer> cached,uint32_t cached_pitch=1) {
            id<MTLComputeCommandEncoder> copy=[command computeCommandEncoder];require(copy!=nil,"W8 scale cache encoder unavailable");
            const uint32_t params[4]{uint32_t(spec.rows),restore?cached_pitch:uint32_t(state->scales.pitch()/2),restore?uint32_t(state->scales.pitch()/2):cached_pitch,0};
            [copy setComputePipelineState:impl_->w8_scale_copy];
            [copy setBuffer:restore?cached:state->scales.impl_->buffer offset:restore?0:size_t(state->scales.row_begin_)*state->scales.pitch() atIndex:0];
            [copy setBuffer:restore?state->scales.impl_->buffer:cached offset:restore?size_t(state->scales.row_begin_)*state->scales.pitch():0 atIndex:1];
            [copy setBytes:params length:sizeof(params) atIndex:2];
            [copy dispatchThreads:MTLSizeMake(spec.rows,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[copy endEncoding];
        };
        auto copy_codes=[&](bool restore) {
            id<MTLBlitCommandEncoder> copy=[command blitCommandEncoder];
            require(copy!=nil,"W8 converted-code copy encoder unavailable");
            const size_t offset=size_t(state->codes.row_begin_)*state->codes.pitch();
            const size_t bytes=size_t(spec.rows)*spec.columns;
            id<MTLBuffer> cached=state->cached_codes->native_codes ? state->cached_codes->native_codes->impl_->buffer : state->cached_codes->codes;
            [copy copyFromBuffer:restore?cached:state->codes.impl_->buffer
                sourceOffset:restore?0:offset toBuffer:restore?state->codes.impl_->buffer:cached
                destinationOffset:restore?offset:0 size:bytes];[copy endEncoding];
        };
        auto copy_code_scales=[&](bool restore) {
            if(state->cached_codes->native_scales) {
                const auto &surface=*state->cached_codes->native_scales;
                copy_scales(restore,surface.impl_->buffer,uint32_t(surface.pitch()/2));
            } else copy_scales(restore,state->cached_codes->scales);
        };
        if(code_hit) {
            copy_code_scales(true);copy_codes(true);
        } else {
            if(scale_hit)copy_scales(true,state->cached_scales->buffer);
            else {
            id<MTLComputeCommandEncoder> scale = [command computeCommandEncoder];
            require(scale != nil, "W8 scale encoder unavailable"); bind(scale);
            [scale setComputePipelineState:pipelines.scales];
            [scale setBuffer:state->scales.impl_->buffer offset:size_t(state->scales.row_begin_) * state->scales.pitch() atIndex:3];
            [scale setBuffer:state->status offset:0 atIndex:4]; [scale setBytes:&p length:sizeof(p) atIndex:5];
            [scale setBuffer:state->signs offset:0 atIndex:6];
            if (direct) [scale dispatchThreads:MTLSizeMake(spec.rows,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            // Generic H128/H512/H256 share one pipeline. Its TG extent
            // belongs to THIS operation, never the first cached shape.
            else [scale dispatchThreadgroups:MTLSizeMake(spec.rows, grouped?spec.columns/256:1, 1) threadsPerThreadgroup:MTLSizeMake(pipelines.register_comfy||pipelines.register_sylvester?32:spec.rotation_block, 1, 1)];
            [scale endEncoding];
            if(state->cached_scales)copy_scales(false,state->cached_scales->buffer);
            }
            id<MTLComputeCommandEncoder> quant = [command computeCommandEncoder];
            require(quant != nil, "W8 code encoder unavailable"); bind(quant);
            [quant setComputePipelineState:pipelines.codes];
            [quant setBuffer:state->scales.impl_->buffer offset:size_t(state->scales.row_begin_) * state->scales.pitch() atIndex:3];
            [quant setBuffer:state->codes.impl_->buffer offset:size_t(state->codes.row_begin_) * state->codes.pitch() atIndex:4];
            [quant setBuffer:state->status offset:0 atIndex:5]; [quant setBytes:&p length:sizeof(p) atIndex:6];
            [quant setBuffer:state->signs offset:0 atIndex:7];
            if (direct) [quant dispatchThreads:MTLSizeMake(spec.columns,spec.rows,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            else [quant dispatchThreadgroups:MTLSizeMake(spec.rows, spec.columns / spec.rotation_block, 1) threadsPerThreadgroup:MTLSizeMake(pipelines.register_comfy||pipelines.register_sylvester?32:spec.rotation_block, 1, 1)];
            [quant endEncoding];
            if(state->cached_codes && !bind_native) {
                copy_code_scales(false);copy_codes(false);
            }
        }
        [command encodeSignalEvent:state->event value:1];
        [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.status == MTLCommandBufferStatusError && state->event.signaledValue < 1) state->event.signaledValue = 1;
            if(state->cached_scales && !scale_hit)state->cached_scales->valid.store(
                completed.status==MTLCommandBufferStatusCompleted && !*static_cast<const uint32_t*>(state->status.contents),std::memory_order_release);
            if(state->cached_codes) {
                const bool valid=completed.status==MTLCommandBufferStatusCompleted &&
                    !*static_cast<const uint32_t*>(state->status.contents);
                std::lock_guard cache_lock(state->device->code_cache_mutex);
                if(code_hit) {if(valid){++state->device->code_stats.hits;++state->device->code_stats.copy_hits;}}
                else {
                    state->cached_codes->valid.store(valid,std::memory_order_release);
                    state->cached_codes->pending.store(false,std::memory_order_release);
                    if(valid)++state->device->code_stats.fills;
                    else ++state->device->code_stats.failed_fills;
                }
            }
            { std::lock_guard lock(state->mutex); state->ok = completed.status == MTLCommandBufferStatusCompleted; state->done = true; }
            state->cv.notify_all();
        }];
        [command commit];code_submitted=true;
        if(bind_native) {*bound_codes=state->codes;*bound_scales=state->scales;}
        QuantStage ticket; ticket.impl_ = std::move(state); return ticket;
      } @catch (NSException *exception) {
        throw CapabilityError("W8 stager exception: " + std::string(exception.reason.UTF8String ?: "unknown"));
      }
    }
}
Completion QuantStage::finish(std::chrono::milliseconds timeout) {
    if (!impl_) return {false, false, "W8 stage ticket empty"};
    std::unique_lock lock(impl_->mutex);
    if (!impl_->cv.wait_for(lock, timeout, [&] { return impl_->done; })) {
        gpu_process_healthy = false;
        return {false, true, "W8 GPU stage timeout; resources retained until completion"};
    }
    const uint32_t flags = *static_cast<const uint32_t *>(impl_->status.contents);
    return {impl_->ok && !flags, false, !impl_->ok ? "W8 GPU staging failed" : flags ? "W8 source/rotation/scale/quantization nonfinite or overflow" : ""};
}
uint32_t QuantStage::validation_flags() const {
    require(impl_ && impl_->done, "W8 stage status not ready"); return *static_cast<const uint32_t *>(impl_->status.contents);
}
void *QuantStage::ready_event() const { return impl_ ? (__bridge void *)impl_->event : nullptr; }

} // namespace tc::ane::gpu
