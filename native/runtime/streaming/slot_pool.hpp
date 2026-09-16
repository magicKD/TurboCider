#pragma once
#include "../../core/stream_slot_c.h"
#include "layout.hpp"
#include <array>
#include <span>
#include <thread>

namespace tc::streaming {

enum class ContentState { Vacant, Loading, Ready, InUse, AwaitingFence };

// Owner-only content safety. Does NOT own/free model storage or end its lease.
// The model adapter owns backings until drain and explicit pool destruction.
class SlotSafetyTracker {
public:
    SlotSafetyTracker(uint32_t pool, uint64_t request_generation,
                      std::span<const uint64_t> capacities);
    tc_stream_slot_ticket_v1 begin_fill(uint32_t slot, tc_stream_work_item_v1,
                                       uint64_t expected_bytes);
    void accept_ready(const tc_stream_slot_ticket_v1 &, uint64_t actual_bytes);
    void begin_use(const tc_stream_slot_ticket_v1 &);
    void seal_readers(const tc_stream_slot_ticket_v1 &,
                      std::span<const tc_stream_reader_fence_v1>);
    void complete_reader(const tc_stream_slot_ticket_v1 &, tc_stream_reader_fence_v1);
    ContentState state(uint32_t slot) const;
    uint64_t capacity_bytes() const noexcept { return capacity_bytes_; }
    bool quiescent() const;
    bool poisoned() const noexcept { return poisoned_; }

private:
    struct Slot {
        uint64_t capacity = 0, expected = 0, generation = 0;
        tc_stream_slot_ticket_v1 ticket{};
        ContentState state = ContentState::Vacant;
        std::array<tc_stream_reader_fence_v1, TC_STREAM_MAX_READER_QUEUES> readers{};
        std::array<bool, TC_STREAM_MAX_READER_QUEUES> complete{};
        uint32_t reader_count = 0;
    };
    uint32_t pool_;
    uint64_t request_, capacity_bytes_ = 0;
    std::thread::id owner_;
    std::vector<Slot> slots_;
    bool poisoned_ = false;
    void owner() const;
    void check(bool, const char *);
    Slot &match(const tc_stream_slot_ticket_v1 &);
};
} // namespace tc::streaming
