#pragma once

// Independent native Metal execution primitives. No MLX dependency.
// Ordered dispatch/event design informed by Splash PR #260 (Apache-2.0):
// design audit 80827e1c3593f7d10c266394e4d4381321b73ad0; implementation-time
// local API cross-check 0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7.
// This implementation is TurboCider's; no Splash runtime is linked.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace tc::metal_graph {

enum class ScalarType : uint8_t { Bytes, BF16, FP16, FP32, UInt32, Int8 };
constexpr size_t scalar_bytes(ScalarType type) {
    switch (type) {
    case ScalarType::Bytes: case ScalarType::Int8: return 1;
    case ScalarType::BF16: case ScalarType::FP16: return 2;
    case ScalarType::FP32: case ScalarType::UInt32: return 4;
    }
    return 0;
}

struct Grid { size_t x = 1, y = 1, z = 1; };
enum class StorageMode : uint8_t { Shared, Private };
enum class LanguageVersion : uint8_t { Default, Metal4_0 };
struct LibraryOptions {
    LanguageVersion language = LanguageVersion::Default;
};
class Backend;
class BufferView;

class Buffer {
public:
    struct Impl;
    Buffer() = default;
    explicit operator bool() const noexcept { return bool(impl_); }
    size_t size_bytes() const noexcept;
    size_t allocated_bytes() const noexcept;
    void* contents() const noexcept; // Shared only; caller must prove readiness.
    void* native_handle() const noexcept; // borrowed id<MTLBuffer>; retain Buffer.
    // Actual allocation ownership control block, never a bare address. Native
    // allocations use their owning Impl; imports supply their allocator owner
    // separately from an array/view wrapper. Empty imports cannot enter a
    // source-bound cache. This proves lifetime identity, not immutability.
    std::shared_ptr<const void> allocation_identity() const noexcept;
    BufferView view(size_t offset, size_t bytes,
                    ScalarType type = ScalarType::Bytes) const;
private:
    explicit Buffer(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<Impl> impl_;
    friend class Backend;
    friend class BufferView;
};

class BufferView {
public:
    BufferView() = default;
    explicit operator bool() const noexcept { return bool(buffer_); }
    const Buffer& buffer() const noexcept { return buffer_; }
    size_t offset_bytes() const noexcept { return offset_; }
    size_t size_bytes() const noexcept { return size_; }
    ScalarType scalar_type() const noexcept { return type_; }
    void* contents() const noexcept;
private:
    BufferView(Buffer buffer, size_t offset, size_t size, ScalarType type)
        : buffer_(std::move(buffer)), offset_(offset), size_(size), type_(type) {}
    Buffer buffer_;
    size_t offset_ = 0, size_ = 0;
    ScalarType type_ = ScalarType::Bytes;
    friend class Buffer;
};

class Library {
public:
    struct Impl;
    Library() = default;
    explicit operator bool() const noexcept { return bool(impl_); }
private:
    explicit Library(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<Impl> impl_;
    friend class Backend;
};

class Pipeline {
public:
    struct Impl;
    Pipeline() = default;
    explicit operator bool() const noexcept { return bool(impl_); }
    size_t max_threads_per_threadgroup() const noexcept;
    size_t thread_execution_width() const noexcept;
private:
    explicit Pipeline(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<Impl> impl_;
    friend class Backend;
};

enum class EventSource : uint8_t { ThisQueue, External };
class SharedEvent {
public:
    struct Impl;
    SharedEvent() = default;
    explicit operator bool() const noexcept { return bool(impl_); }
    void* native_handle() const noexcept; // borrowed id<MTLSharedEvent>
    uint64_t value() const noexcept;
    // Explicit external producer only. Never called automatically on timeout:
    // releasing a device dependency is NOT a result-validity/reuse permission.
    void signal_external(uint64_t value) const;
private:
    explicit SharedEvent(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<Impl> impl_;
    friend class Backend;
};

struct Dispatch {
    Pipeline pipeline;
    std::vector<BufferView> buffers; // indices 0..N-1
    std::vector<std::byte> parameters; // copied bytes at index N, <=4096 bytes
    Grid groups;
    Grid threads;
    size_t threadgroup_memory_bytes = 0; // dynamic memory at index 0
};
struct GraphStep {
    enum class Kind : uint8_t { Dispatch, Signal, Wait };
    Kind kind = Kind::Dispatch;
    Dispatch compute;
    SharedEvent event;
    uint64_t value = 0;
};

class CommandGraph {
public:
    // Assemble a validated operator/block privately, then append it atomically.
    // A validation or allocation failure leaves this graph unchanged. The
    // draft has not been submitted and owns all copied parameters and views.
    void append_graph(CommandGraph draft) {
        static_assert(std::is_nothrow_move_constructible_v<GraphStep>);
        if (draft.steps_.size() > steps_.max_size()-steps_.size())
            throw std::length_error("native graph append exceeds vector capacity");
        steps_.reserve(steps_.size()+draft.steps_.size());
        for (auto& step : draft.steps_) steps_.push_back(std::move(step));
    }
    void dispatch(Pipeline pipeline, std::vector<BufferView> buffers,
                  Grid groups, Grid threads, size_t threadgroup_memory_bytes = 0);
    template<class Params>
    void dispatch(Pipeline pipeline, std::vector<BufferView> buffers,
                  const Params& params, Grid groups, Grid threads,
                  size_t threadgroup_memory_bytes = 0) {
        static_assert(std::is_trivially_copyable_v<Params>);
        static_assert(sizeof(Params) <= 4096);
        Dispatch item{std::move(pipeline), std::move(buffers), {}, groups, threads,
                      threadgroup_memory_bytes};
        item.parameters.resize(sizeof(Params));
        std::memcpy(item.parameters.data(), &params, sizeof(Params));
        append(std::move(item));
    }
    void signal(SharedEvent event, uint64_t value);
    void wait(SharedEvent event, uint64_t value);
    const std::vector<GraphStep>& steps() const noexcept { return steps_; }
    bool empty() const noexcept { return steps_.empty(); }
private:
    void append(Dispatch item);
    std::vector<GraphStep> steps_;
};

// The reservation token can wrap an existing MemoryReservation/StorageLease.
// No private/global byte budget or second memory ledger exists in this backend.
// Tokens must support destruction from a completion thread (e.g. enqueue an
// owner-thread release), because a ticket may be abandoned while GPU work runs.
struct AllocationHooks {
    std::function<std::shared_ptr<void>(size_t upper_bytes, std::string_view label)> reserve;
    std::function<void(const std::shared_ptr<void>& token, void* native_buffer,
                       size_t allocated_bytes)> allocated;
};

struct BackendOptions {
    size_t maximum_command_buffers = 64;
    size_t maximum_prepared_pipelines = 128;
    size_t maximum_graph_steps = 4096;
    AllocationHooks allocation;
};

struct Completion {
    bool ok = false;
    bool physically_complete = false;
    bool timed_out = false;
    size_t submitted_buffers = 0;
    size_t completed_buffers = 0;
    double gpu_seconds = 0; // first start to last end; includes inter-CB gaps
    std::string error;
};

class Ticket {
public:
    struct State;
    Ticket() = default;
    Ticket(const Ticket&) = delete;
    Ticket& operator=(const Ticket&) = delete;
    Ticket(Ticket&&) noexcept = default;
    Ticket& operator=(Ticket&&) noexcept = default;
    // Nonblocking destruction, including after wait() times out. Completion
    // handlers retain backing/allocator owners until physical completion;
    // ambiguous partial submissions are additionally quarantined by the backend.
    ~Ticket() = default;
    bool ready() const noexcept;
    Completion status() const;
    // A timeout retires this result and marks its backend unhealthy. Use status()
    // to poll; a zero-duration wait is an actual deadline, not a benign poll.
    Completion wait(std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;
private:
    explicit Ticket(std::shared_ptr<State> state) : state_(std::move(state)) {}
    std::shared_ptr<State> state_;
    friend class Backend;
};

class Backend {
public:
    struct Impl;
    explicit Backend(BackendOptions options = {});
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    std::string device_name() const;
    bool healthy() const noexcept;
    void* native_device() const noexcept; // borrowed; Backend/Buffer must outlive use
    Buffer allocate(size_t bytes, std::string_view label = {},
                    StorageMode storage = StorageMode::Shared);
    // Same-device retained import. storage_owner MUST keep external allocator
    // ownership alive, not just retain the id<MTLBuffer>. Existing lease reused;
    // allocation hooks are deliberately NOT invoked for an imported backing.
    Buffer import_buffer(void* native_buffer, std::shared_ptr<void> storage_owner,
                         std::shared_ptr<void> existing_lease = {},
                         std::shared_ptr<const void> allocation_identity = {});
    // Metal4_0 is explicit and requires macOS 26; actual kernel/device support
    // is still validated by library and pipeline preparation.
    Library compile_library(std::string_view source, std::string_view label = {},
                            LibraryOptions options = {});
    Library load_library(std::string_view path);
    Pipeline prepare_pipeline(const Library&, std::string_view function);
    SharedEvent make_event(EventSource source = EventSource::ThisQueue);
    // Prevalidation errors throw before submission. After any command may have
    // reached Metal, errors return a failed Ticket retaining all graph owners.
    // Capacity is reserved before encoding; insufficient capacity throws, never
    // waits while holding an unsubmitted ready producer. No automatic event
    // force-signals on failure: external handoff recovery belongs to its domain.
    Ticket submit(CommandGraph graph);
private:
    std::shared_ptr<Impl> impl_;
};

} // namespace tc::metal_graph
