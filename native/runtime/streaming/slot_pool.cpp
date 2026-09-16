#include "slot_pool.hpp"
#include <limits>
#include <stdexcept>

namespace tc::streaming {
namespace {
bool same(const tc_stream_slot_ticket_v1 &a, const tc_stream_slot_ticket_v1 &b) {
    return a.pool == b.pool && a.slot == b.slot &&
        a.request_generation == b.request_generation && a.content_generation == b.content_generation &&
        a.item.stage == b.item.stage && a.item.pass == b.item.pass &&
        a.item.step == b.item.step && a.item.group == b.item.group;
}
}
SlotSafetyTracker::SlotSafetyTracker(uint32_t pool, uint64_t request,
                                    std::span<const uint64_t> capacities)
    : pool_(pool), request_(request), owner_(std::this_thread::get_id()) {
    check(request != 0 && !capacities.empty() && capacities.size() <= max_slots,
          "invalid pool construction");
    slots_.resize(capacities.size());
    for (size_t i=0; i<capacities.size(); ++i) {
        check(capacities[i] && capacities[i] <= UINT64_MAX-capacity_bytes_, "capacity overflow");
        slots_[i].capacity = capacities[i]; capacity_bytes_ += capacities[i];
    }
}
void SlotSafetyTracker::owner() const {
    if (owner_ != std::this_thread::get_id())
        throw std::logic_error("streaming_owner_violation");
}
void SlotSafetyTracker::check(bool value, const char *why) {
    owner();
    if (poisoned_ || !value) {
        poisoned_ = true;
        throw std::logic_error(std::string("streaming_slot_violation: ") + why);
    }
}
SlotSafetyTracker::Slot &SlotSafetyTracker::match(const tc_stream_slot_ticket_v1 &ticket) {
    check(ticket.struct_size == sizeof(ticket) && ticket.version == TC_STREAM_SLOT_ABI_V1 &&
          ticket.slot < slots_.size(), "invalid ticket ABI/index");
    auto &slot = slots_[ticket.slot];
    check(same(slot.ticket,ticket), "stale or mismatched ticket");
    return slot;
}
tc_stream_slot_ticket_v1 SlotSafetyTracker::begin_fill(uint32_t index, tc_stream_work_item_v1 item,
                                                       uint64_t expected) {
    check(index < slots_.size(), "invalid slot index");
    auto &s = slots_[index];
    check(s.state == ContentState::Vacant, "overwrite before last reader");
    check(expected && expected <= s.capacity, "fill exceeds capacity");
    check(s.generation != UINT64_MAX, "content generation overflow");
    s.expected = expected;
    s.ticket = {sizeof(tc_stream_slot_ticket_v1), TC_STREAM_SLOT_ABI_V1, pool_, index,
                request_, ++s.generation, item};
    s.state = ContentState::Loading;
    s.reader_count = 0; s.complete.fill(false);
    return s.ticket;
}
void SlotSafetyTracker::accept_ready(const tc_stream_slot_ticket_v1 &t, uint64_t bytes) {
    auto &s = match(t);
    check(s.state == ContentState::Loading && bytes == s.expected, "invalid fill completion");
    s.state = ContentState::Ready;
}
void SlotSafetyTracker::begin_use(const tc_stream_slot_ticket_v1 &t) {
    auto &s = match(t); check(s.state == ContentState::Ready, "use before ready");
    s.state = ContentState::InUse;
}
void SlotSafetyTracker::seal_readers(const tc_stream_slot_ticket_v1 &t,
                                    std::span<const tc_stream_reader_fence_v1> readers) {
    auto &s = match(t);
    check(s.state == ContentState::InUse && !readers.empty() &&
          readers.size() <= TC_STREAM_MAX_READER_QUEUES, "invalid reader set");
    for (size_t i=0; i<readers.size(); ++i) {
        check(readers[i].queue && readers[i].sequence, "invalid reader identity");
        for (size_t j=0; j<i; ++j)
            check(readers[i].queue != readers[j].queue, "duplicate queue in reader set");
        s.readers[i] = readers[i];
    }
    s.reader_count = uint32_t(readers.size()); s.state = ContentState::AwaitingFence;
}
void SlotSafetyTracker::complete_reader(const tc_stream_slot_ticket_v1 &t, tc_stream_reader_fence_v1 f) {
    auto &s = match(t);
    check(s.state == ContentState::AwaitingFence, "reader completion outside sealed use");
    uint32_t found = s.reader_count;
    for (uint32_t i=0; i<s.reader_count; ++i)
        if (s.readers[i].queue == f.queue && s.readers[i].sequence == f.sequence) found=i;
    check(found < s.reader_count, "unknown reader completion");
    check(!s.complete[found], "duplicate reader completion");
    s.complete[found] = true;
    bool all = true;
    for (uint32_t i=0; i<s.reader_count; ++i) all &= s.complete[i];
    if (all) s.state = ContentState::Vacant;
}
ContentState SlotSafetyTracker::state(uint32_t slot) const {
    owner();
    if (slot >= slots_.size()) throw std::out_of_range("streaming slot index");
    return slots_[slot].state;
}
bool SlotSafetyTracker::quiescent() const {
    owner();
    for (const auto &s : slots_) if (s.state != ContentState::Vacant) return false;
    return true;
}
} // namespace tc::streaming
