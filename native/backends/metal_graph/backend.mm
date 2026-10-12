#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace tc::metal_graph {
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
NSString* ns_string(std::string_view value) {
    NSString* result = [[NSString alloc] initWithBytes:value.data() length:value.size()
                                             encoding:NSUTF8StringEncoding];
    if (!result) throw std::invalid_argument("native Metal string is not UTF-8");
    return result;
}
std::string error_text(NSError* error, const char* fallback) {
    const char* text = error.localizedDescription.UTF8String;
    return text ? text : fallback;
}
size_t checked_product(Grid grid) {
    require(grid.x && grid.y && grid.z, "native Metal dispatch dimensions must be positive");
    require(grid.x <= std::numeric_limits<size_t>::max() / grid.y,
            "native Metal dispatch dimensions overflow");
    size_t xy = grid.x * grid.y;
    require(xy <= std::numeric_limits<size_t>::max() / grid.z,
            "native Metal dispatch dimensions overflow");
    return xy * grid.z;
}
struct ContextIdentity {};
}

struct Buffer::Impl {
    id<MTLBuffer> native;
    std::shared_ptr<void> external_owner;
    std::shared_ptr<const void> external_allocation_identity;
    std::shared_ptr<void> accounting_lease;
    ~Impl() noexcept {
        // Member destruction is reverse declaration order. Explicitly drop our
        // Metal retain and the actual external allocator owner before releasing
        // the accounting claim; otherwise a callback could advertise budget as
        // free while this wrapper still owns the backing allocation.
        native = nil;
        external_owner.reset();
        external_allocation_identity.reset();
        accounting_lease.reset();
    }
};
struct Library::Impl {
    id<MTLLibrary> native;
    std::shared_ptr<ContextIdentity> context;
    uint64_t serial = 0;
};
struct Pipeline::Impl {
    id<MTLComputePipelineState> native;
    std::shared_ptr<Library::Impl> library;
};
struct SharedEvent::Impl {
    id<MTLSharedEvent> native;
    std::shared_ptr<ContextIdentity> context;
    EventSource source = EventSource::ThisQueue;
    uint64_t reserved_signal = 0; // protected by owning backend submission mutex
    std::mutex external_mutex;
};
struct Backend::Impl {
    BackendOptions options;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::shared_ptr<ContextIdentity> context = std::make_shared<ContextIdentity>();
    std::atomic<bool> healthy{true};
    std::atomic<size_t> outstanding_buffers{0};
    std::mutex submission_mutex;
    std::mutex preparation_mutex;
    std::mutex quarantine_mutex;
    // Reserved at construction. An ambiguous partial submission must retain
    // allocations even if Metal never invokes its registered completion.
    std::vector<std::shared_ptr<Ticket::State>> quarantine;
    uint64_t library_serial = 0;
    std::unordered_map<std::string, std::shared_ptr<Pipeline::Impl>> pipelines;
};

struct Ticket::State {
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::shared_ptr<Backend::Impl> backend;
    CommandGraph graph; // retains every buffer owner, parameter payload, PSO/event
    Completion completion;
    bool sealed = false;
    bool failed = false;
    double first_gpu_start = 0;
    double last_gpu_end = 0;

    void fail_locked(std::string_view message) noexcept {
        failed = true;
        backend->healthy = false;
        if (completion.error.empty()) {
            try { completion.error.assign(message); } catch (...) {}
        }
    }
    void settle_locked() {
        if (!sealed || completion.completed_buffers != completion.submitted_buffers) return;
        completion.physically_complete = true;
        completion.ok = !failed;
        if (first_gpu_start > 0 && last_gpu_end >= first_gpu_start)
            completion.gpu_seconds = last_gpu_end - first_gpu_start;
        graph = {}; // only after every registered command physically completed
        changed.notify_all();
    }
};

namespace {
void complete_buffer(const std::shared_ptr<Ticket::State>& state,
                     id<MTLCommandBuffer> command) noexcept {
    @autoreleasepool {
        try {
            std::lock_guard lock(state->mutex);
            if (command.status != MTLCommandBufferStatusCompleted)
                state->fail_locked(error_text(command.error, "native Metal command buffer failed"));
            const double begin = command.GPUStartTime, end = command.GPUEndTime;
            if (begin > 0 && (!state->first_gpu_start || begin < state->first_gpu_start))
                state->first_gpu_start = begin;
            state->last_gpu_end = std::max(state->last_gpu_end, end);
            ++state->completion.completed_buffers;
            state->backend->outstanding_buffers.fetch_sub(1);
            state->settle_locked();
            if (state->completion.physically_complete) {
                std::lock_guard quarantine_lock(state->backend->quarantine_mutex);
                std::erase(state->backend->quarantine, state);
            }
        } catch (...) {
            // A callback failure cannot certify reclamation. Keep owners held
            // by its ticket/state and prohibit new work; the caller times out.
            state->backend->healthy = false;
        }
    }
}
}

size_t Buffer::size_bytes() const noexcept { return impl_ ? impl_->native.length : 0; }
size_t Buffer::allocated_bytes() const noexcept { return impl_ ? impl_->native.allocatedSize : 0; }
std::shared_ptr<const void> Buffer::allocation_identity() const noexcept {
    if (!impl_) return {};
    if (impl_->external_owner) return impl_->external_allocation_identity;
    return impl_;
}
void* Buffer::contents() const noexcept {
    return impl_ && impl_->native.storageMode == MTLStorageModeShared ? impl_->native.contents : nullptr;
}
void* Buffer::native_handle() const noexcept { return impl_ ? (__bridge void*)impl_->native : nullptr; }
BufferView Buffer::view(size_t offset, size_t bytes, ScalarType type) const {
    require(bool(impl_), "native Metal view needs a buffer");
    const size_t item = scalar_bytes(type);
    require(item && offset % item == 0 && bytes % item == 0, "native Metal view scalar alignment mismatch");
    require(bytes && offset <= size_bytes() && bytes <= size_bytes() - offset,
            "native Metal view exceeds its backing allocation");
    return BufferView(*this, offset, bytes, type);
}
void* BufferView::contents() const noexcept {
    auto* base = static_cast<std::byte*>(buffer_.contents());
    return base ? base + offset_ : nullptr;
}
size_t Pipeline::max_threads_per_threadgroup() const noexcept {
    return impl_ ? impl_->native.maxTotalThreadsPerThreadgroup : 0;
}
size_t Pipeline::thread_execution_width() const noexcept {
    return impl_ ? impl_->native.threadExecutionWidth : 0;
}
void* SharedEvent::native_handle() const noexcept { return impl_ ? (__bridge void*)impl_->native : nullptr; }
uint64_t SharedEvent::value() const noexcept { return impl_ ? impl_->native.signaledValue : 0; }
void SharedEvent::signal_external(uint64_t value) const {
    require(impl_ && impl_->source == EventSource::External,
            "host signal requires an explicitly external native Metal event");
    std::lock_guard lock(impl_->external_mutex);
    require(value && value > impl_->native.signaledValue, "native Metal event signal must increase");
    impl_->native.signaledValue = value;
}

void CommandGraph::append(Dispatch item) {
    GraphStep step;
    step.compute = std::move(item);
    steps_.push_back(std::move(step));
}
void CommandGraph::dispatch(Pipeline pipeline, std::vector<BufferView> buffers,
                            Grid groups, Grid threads, size_t threadgroup_memory_bytes) {
    append({std::move(pipeline), std::move(buffers), {}, groups, threads, threadgroup_memory_bytes});
}
void CommandGraph::signal(SharedEvent event, uint64_t value) {
    GraphStep step; step.kind = GraphStep::Kind::Signal;
    step.event = std::move(event); step.value = value; steps_.push_back(std::move(step));
}
void CommandGraph::wait(SharedEvent event, uint64_t value) {
    GraphStep step; step.kind = GraphStep::Kind::Wait;
    step.event = std::move(event); step.value = value; steps_.push_back(std::move(step));
}

bool Ticket::ready() const noexcept {
    if (!state_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->completion.physically_complete;
}
Completion Ticket::status() const {
    if (!state_) return {false, false, false, 0, 0, 0, "empty native Metal ticket"};
    std::lock_guard lock(state_->mutex);
    return state_->completion;
}
Completion Ticket::wait(std::chrono::milliseconds timeout) const {
    require(timeout.count() >= 0, "native Metal wait timeout must be nonnegative");
    if (!state_) return status();
    std::unique_lock lock(state_->mutex);
    if (!state_->changed.wait_for(lock, timeout, [&] { return state_->completion.physically_complete; })) {
        state_->fail_locked("native Metal completion timeout; resources remain in flight");
        state_->completion.timed_out = true;
        // A host timeout never signal-exits GPU/ANE waits or frees backing memory.
    }
    return state_->completion;
}

Backend::Backend(BackendOptions options) : impl_(std::make_shared<Impl>()) {
    require(options.maximum_command_buffers && options.maximum_command_buffers <= 256,
            "native Metal command-buffer bound must be between 1 and 256");
    require(options.maximum_prepared_pipelines && options.maximum_graph_steps,
            "native Metal preparation bounds must be positive");
    require(bool(options.allocation.reserve) == bool(options.allocation.allocated),
            "native Metal allocation hooks must be supplied together");
    impl_->options = std::move(options);
    impl_->quarantine.reserve(impl_->options.maximum_command_buffers);
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) throw std::runtime_error("native Metal device unavailable");
        impl_->queue = [impl_->device newCommandQueueWithMaxCommandBufferCount:impl_->options.maximum_command_buffers];
        if (!impl_->queue) throw std::runtime_error("native Metal command queue unavailable");
        impl_->queue.label = @"TurboCider native Metal graph";
    }
}
Backend::~Backend() = default;
std::string Backend::device_name() const {
    const char* name = impl_->device.name.UTF8String;
    return name ? name : "unknown Metal device";
}
bool Backend::healthy() const noexcept { return impl_->healthy.load(); }
void* Backend::native_device() const noexcept { return (__bridge void*)impl_->device; }

Buffer Backend::allocate(size_t bytes, std::string_view label, StorageMode storage) {
    require(healthy(), "native Metal backend is unhealthy");
    require(bytes && bytes <= impl_->device.maxBufferLength, "native Metal buffer size unsupported");
    // Metal may account at a page granularity. Reserve before creating storage,
    // and fail closed if the driver's actual capacity exceeds this upper bound.
    constexpr size_t alignment = 16384;
    require(bytes <= std::numeric_limits<size_t>::max() - (alignment - 1), "native Metal allocation size overflow");
    const size_t upper = (bytes + alignment - 1) / alignment * alignment;
    std::shared_ptr<void> lease;
    if (impl_->options.allocation.reserve) {
        lease = impl_->options.allocation.reserve(upper, label);
        if (!lease) throw std::runtime_error("native Metal allocation admission denied");
    }
    @autoreleasepool {
        MTLResourceOptions mode = storage == StorageMode::Shared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        id<MTLBuffer> native = [impl_->device newBufferWithLength:bytes options:mode];
        if (!native) throw std::runtime_error("native Metal buffer allocation failed");
        require(native.allocatedSize <= upper, "native Metal allocation exceeds admitted capacity");
        if (!label.empty()) native.label = ns_string(label);
        if (impl_->options.allocation.allocated)
            impl_->options.allocation.allocated(lease, (__bridge void*)native, native.allocatedSize);
        auto backing = std::make_shared<Buffer::Impl>();
        backing->native = native; backing->accounting_lease = std::move(lease);
        return Buffer(std::move(backing));
    }
}

Buffer Backend::import_buffer(void* native_buffer, std::shared_ptr<void> storage_owner,
                              std::shared_ptr<void> existing_lease,
                              std::shared_ptr<const void> allocation_identity) {
    require(healthy(), "native Metal backend is unhealthy");
    require(native_buffer && storage_owner, "native Metal import requires backing storage ownership");
    id<MTLBuffer> native = (__bridge id<MTLBuffer>)native_buffer;
    require([native conformsToProtocol:@protocol(MTLBuffer)], "native Metal import is not an MTLBuffer");
    require(native.device == impl_->device && native.length, "native Metal import device/capacity mismatch");
    auto backing = std::make_shared<Buffer::Impl>();
    backing->native = native; backing->external_owner = std::move(storage_owner);
    backing->external_allocation_identity = std::move(allocation_identity);
    backing->accounting_lease = std::move(existing_lease);
    return Buffer(std::move(backing));
}

Library Backend::compile_library(std::string_view source, std::string_view label,
                                  LibraryOptions requested) {
    require(healthy() && !source.empty(), "native Metal library source unavailable");
    std::lock_guard lock(impl_->preparation_mutex);
    @autoreleasepool {
        MTLCompileOptions* options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        if (requested.language == LanguageVersion::Metal4_0) {
            if (@available(macOS 26.0, *)) {
                // The CLT 15.2 header stops at MSL 3.2, while macOS 26's
                // runtime compiler accepts MSL 4.0. Keep the newer enum local
                // and guarded; verify the SDK's established major/minor ABI.
                static_assert(MTLLanguageVersion3_1 == ((3U << 16) + 1));
                options.languageVersion = static_cast<MTLLanguageVersion>(4U << 16);
            } else {
                throw std::runtime_error("native Metal 4.0 kernels require macOS 26 or later");
            }
        } else {
            require(requested.language == LanguageVersion::Default,
                    "unsupported native Metal language selection");
        }
        NSError* error = nil;
        id<MTLLibrary> library = [impl_->device newLibraryWithSource:ns_string(source) options:options error:&error];
        if (!library) throw std::runtime_error(error_text(error, "native Metal shader compilation failed"));
        if (!label.empty()) library.label = ns_string(label);
        auto result = std::make_shared<Library::Impl>();
        result->native = library; result->context = impl_->context; result->serial = ++impl_->library_serial;
        return Library(std::move(result));
    }
}
Library Backend::load_library(std::string_view path) {
    require(healthy() && !path.empty(), "native Metal library path unavailable");
    std::lock_guard lock(impl_->preparation_mutex);
    @autoreleasepool {
        NSError* error = nil;
        id<MTLLibrary> library = [impl_->device newLibraryWithURL:[NSURL fileURLWithPath:ns_string(path)] error:&error];
        if (!library) throw std::runtime_error(error_text(error, "native Metal library loading failed"));
        auto result = std::make_shared<Library::Impl>();
        result->native = library; result->context = impl_->context; result->serial = ++impl_->library_serial;
        return Library(std::move(result));
    }
}
Pipeline Backend::prepare_pipeline(const Library& library, std::string_view function) {
    require(healthy() && library.impl_ && library.impl_->context == impl_->context && !function.empty(),
            "native Metal pipeline library/function mismatch");
    std::lock_guard lock(impl_->preparation_mutex);
    const std::string key = std::to_string(library.impl_->serial) + ":" + std::string(function);
    if (auto it = impl_->pipelines.find(key); it != impl_->pipelines.end()) return Pipeline(it->second);
    require(impl_->pipelines.size() < impl_->options.maximum_prepared_pipelines,
            "native Metal prepared pipeline capacity exceeded");
    @autoreleasepool {
        id<MTLFunction> fn = [library.impl_->native newFunctionWithName:ns_string(function)];
        if (!fn) throw std::runtime_error("native Metal shader function not found: " + std::string(function));
        NSError* error = nil;
        id<MTLComputePipelineState> native = [impl_->device newComputePipelineStateWithFunction:fn error:&error];
        if (!native) throw std::runtime_error(error_text(error, "native Metal pipeline preparation failed"));
        auto result = std::make_shared<Pipeline::Impl>(); result->native = native; result->library = library.impl_;
        impl_->pipelines.emplace(key, result);
        return Pipeline(std::move(result));
    }
}
SharedEvent Backend::make_event(EventSource source) {
    require(healthy(), "native Metal backend is unhealthy");
    auto event = std::make_shared<SharedEvent::Impl>();
    event->native = [impl_->device newSharedEvent];
    if (!event->native) throw std::runtime_error("native Metal shared event unavailable");
    event->context = impl_->context; event->source = source;
    return SharedEvent(std::move(event));
}

Ticket Backend::submit(CommandGraph graph) {
    std::lock_guard submit_lock(impl_->submission_mutex);
    require(healthy(), "native Metal backend is unhealthy");
    const auto& steps = graph.steps();
    require(steps.size() <= impl_->options.maximum_graph_steps, "native Metal graph step capacity exceeded");
    std::unordered_map<SharedEvent::Impl*, uint64_t> signals;
    size_t segments = steps.empty() ? 0 : 1;
    const MTLSize max_threads = impl_->device.maxThreadsPerThreadgroup;
    // All shape, ownership and queue-local dependency validation precedes any
    // reservation mutation or actual command submission.
    for (size_t i = 0; i < steps.size(); ++i) {
        const auto& step = steps[i];
        if (step.kind == GraphStep::Kind::Dispatch) {
            const auto& d = step.compute;
            require(d.pipeline.impl_ && d.pipeline.impl_->library->context == impl_->context,
                    "native Metal dispatch uses another backend's pipeline");
            require(d.buffers.size() + !d.parameters.empty() <= 31 && d.parameters.size() <= 4096,
                    "native Metal argument table/parameter capacity exceeded");
            require(checked_product(d.threads) <= d.pipeline.max_threads_per_threadgroup() &&
                    d.threads.x <= max_threads.width && d.threads.y <= max_threads.height && d.threads.z <= max_threads.depth,
                    "native Metal threadgroup exceeds pipeline/device capability");
            (void)checked_product(d.groups);
            require(d.threadgroup_memory_bytes % 16 == 0 &&
                    d.threadgroup_memory_bytes <= impl_->device.maxThreadgroupMemoryLength &&
                    d.pipeline.impl_->native.staticThreadgroupMemoryLength <=
                        impl_->device.maxThreadgroupMemoryLength - d.threadgroup_memory_bytes,
                    "native Metal threadgroup memory exceeds capability/alignment");
            for (const auto& view : d.buffers) {
                require(view && view.buffer().impl_->native.device == impl_->device,
                        "native Metal dispatch buffer device mismatch");
                require(view.offset_bytes() <= view.buffer().size_bytes() &&
                        view.size_bytes() <= view.buffer().size_bytes() - view.offset_bytes(),
                        "native Metal dispatch view exceeds backing allocation");
            }
        } else {
            require(step.event.impl_ && step.event.impl_->context == impl_->context && step.value,
                    "native Metal event belongs to another backend or has zero value");
            auto* event = step.event.impl_.get();
            const auto found = signals.find(event);
            uint64_t produced = found == signals.end() ? event->reserved_signal : found->second;
            if (step.kind == GraphStep::Kind::Signal) {
                require(step.value > produced && step.value > event->native.signaledValue,
                        "native Metal graph signals must strictly increase");
                signals[event] = step.value;
                if (i + 1 < steps.size()) ++segments;
            } else {
                require(event->source == EventSource::External || step.value <= produced ||
                        step.value <= event->native.signaledValue,
                        "native Metal queue waits for an unsubmitted future producer");
            }
        }
    }
    const size_t capacity = impl_->options.maximum_command_buffers;
    require(segments <= capacity && impl_->outstanding_buffers.load() <= capacity - segments,
            "native Metal in-flight command-buffer capacity exceeded; retry after completion");
    auto state = std::make_shared<Ticket::State>();
    state->backend = impl_; state->graph = std::move(graph);
    for (const auto& [event, value] : signals) event->reserved_signal = value;
    impl_->outstanding_buffers.fetch_add(segments);
    size_t registered = 0;
    try {
        @autoreleasepool {
          @try {
            size_t begin = 0;
            while (begin < state->graph.steps().size()) {
                if (!healthy()) throw std::runtime_error("native Metal backend failed during bounded submission");
                id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
                if (!command) throw std::runtime_error("native Metal command buffer creation failed");
                id<MTLComputeCommandEncoder> encoder = nil;
                size_t end = begin;
                for (; end < state->graph.steps().size(); ++end) {
                    const auto& step = state->graph.steps()[end];
                    if (step.kind == GraphStep::Kind::Dispatch) {
                        const auto& d = step.compute;
                        if (!encoder) encoder = [command computeCommandEncoder];
                        if (!encoder) throw std::runtime_error("native Metal compute encoder creation failed");
                        [encoder setComputePipelineState:d.pipeline.impl_->native];
                        for (size_t index = 0; index < d.buffers.size(); ++index) {
                            const auto& view = d.buffers[index];
                            [encoder setBuffer:view.buffer().impl_->native offset:view.offset_bytes() atIndex:index];
                        }
                        if (!d.parameters.empty()) [encoder setBytes:d.parameters.data() length:d.parameters.size() atIndex:d.buffers.size()];
                        if (d.threadgroup_memory_bytes) [encoder setThreadgroupMemoryLength:d.threadgroup_memory_bytes atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake(d.groups.x,d.groups.y,d.groups.z)
                                 threadsPerThreadgroup:MTLSizeMake(d.threads.x,d.threads.y,d.threads.z)];
                        // Imported allocations may have untracked Metal hazards.
                        // Conservative, explicit inter-dispatch visibility first.
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    } else {
                        if (encoder) { [encoder endEncoding]; encoder = nil; }
                        if (step.kind == GraphStep::Kind::Wait)
                            [command encodeWaitForEvent:step.event.impl_->native value:step.value];
                        else {
                            [command encodeSignalEvent:step.event.impl_->native value:step.value];
                            ++end;
                            break; // producer is committed before constructing its consumer
                        }
                    }
                }
                if (encoder) [encoder endEncoding];
                [command addCompletedHandler:^(id<MTLCommandBuffer> completed) { complete_buffer(state, completed); }];
                {
                    std::lock_guard lock(state->mutex);
                    ++state->completion.submitted_buffers; // register before possibly immediate callback
                    ++registered;
                }
                [command commit];
                begin = end;
            }
          } @catch (NSException* exception) {
            throw std::runtime_error(std::string("native Metal submission exception: ") +
                                     (exception.reason.UTF8String ?: "unknown"));
          }
        }
    } catch (const std::exception& error) {
        std::lock_guard lock(state->mutex);
        state->fail_locked(error.what());
        if (registered) {
            std::lock_guard quarantine_lock(impl_->quarantine_mutex);
            impl_->quarantine.push_back(state);
        }
    } catch (...) {
        std::lock_guard lock(state->mutex);
        state->fail_locked("unknown native Metal submission failure");
        if (registered) {
            std::lock_guard quarantine_lock(impl_->quarantine_mutex);
            impl_->quarantine.push_back(state);
        }
    }
    impl_->outstanding_buffers.fetch_sub(segments - registered);
    {
        std::lock_guard lock(state->mutex);
        state->sealed = true;
        state->settle_locked();
        if (state->completion.physically_complete) {
            std::lock_guard quarantine_lock(impl_->quarantine_mutex);
            std::erase(impl_->quarantine, state);
        }
    }
    return Ticket(std::move(state));
}

} // namespace tc::metal_graph
