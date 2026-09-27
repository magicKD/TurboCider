#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <vector>

namespace tc {

enum class MemoryTraceEventKind : uint8_t {
    State,
    Epoch,
    Reserve,
    Commit,
    Retire,
    Completion,
    Schedule,
    Checkpoint,
    Pressure,
    Failure,
};

const char *memory_trace_event_kind_name(MemoryTraceEventKind kind);

struct MemoryTraceEvent {
    uint64_t sequence = 0;
    uint64_t monotonic_ns = 0;
    MemoryTraceEventKind kind = MemoryTraceEventKind::Checkpoint;
    uint64_t phase_id = 0;
    uint64_t site_id = 0;
    uint64_t epoch = 0;
    uint64_t upper_bytes = 0;
    uint64_t actual_bytes = 0;
    uint64_t committed_bytes = 0;
    uint64_t reserved_bytes = 0;
    uint64_t pending_bytes = 0;
    uint32_t stage_id = 0;
    uint32_t slot_id = 0;
    int32_t status = 0;
};

struct MemoryTraceStatus {
    uint64_t event_count = 0;
    uint64_t dropped_events = 0;
    bool overflowed = false;
};

class MemoryTraceBuffer {
  public:
    explicit MemoryTraceBuffer(size_t capacity = 1024);
    MemoryTraceBuffer(const MemoryTraceBuffer &) = delete;
    MemoryTraceBuffer &operator=(const MemoryTraceBuffer &) = delete;

    void record(MemoryTraceEventKind kind, std::string_view phase,
                std::string_view site, uint64_t epoch,
                uint64_t upper_bytes = 0, uint64_t actual_bytes = 0,
                uint64_t committed_bytes = 0, uint64_t reserved_bytes = 0,
                uint64_t pending_bytes = 0, uint32_t stage_id = 0,
                uint32_t slot_id = 0, int32_t status = 0) noexcept;
    MemoryTraceStatus status() const;
    std::vector<MemoryTraceEvent> drain();

  private:
    static uint64_t hash(std::string_view value);
    static uint64_t monotonic_ns();

    mutable std::mutex mutex_;
    std::vector<MemoryTraceEvent> ring_;
    size_t head_ = 0;
    size_t size_ = 0;
    bool drained_ = false;
    MemoryTraceStatus status_;
};

} // namespace tc
