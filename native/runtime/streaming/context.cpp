#include "context.hpp"
#include "audit.hpp"
#include <exception>
#include <limits>
#include <stdexcept>

namespace tc::streaming {
struct StageExecutor::State {
    StageLayout layout;
    std::unique_ptr<SlotSafetyTracker> safety;
    std::unique_ptr<IoExecutor> io;
    std::vector<tc_stream_slot_ticket_v1> tickets;
    ExecutionCounters counters;
    uint32_t passes = 0;
    uint32_t active_pool_index = std::numeric_limits<uint32_t>::max();
    bool pool_drained = false;
    bool failed = false, finished = false;
    std::thread::id owner = std::this_thread::get_id();
};
StageExecutor::StageExecutor(uint32_t stage, uint64_t request,
                             std::shared_ptr<ModelSlotAdapter> adapter)
    : stage_(stage), request_(request), adapter_(std::move(adapter)) {
    if (!request_ || !adapter_) throw std::invalid_argument("invalid streaming executor identity");
}
StageExecutor::~StageExecutor() {
    if (pool_live_ && !retry_drain()) std::terminate();
}
void StageExecutor::destroy_active_pool() noexcept {
    if (!pool_live_) return;
    adapter_->destroy_pool();
    pool_live_ = false;
    if (state_) {
        state_->safety.reset();
        state_->tickets.clear();
        state_->active_pool_index = std::numeric_limits<uint32_t>::max();
        state_->pool_drained = false;
    }
}
bool StageExecutor::retry_drain() noexcept {
    if (state_ && state_->owner!=std::this_thread::get_id()) return false;
    if (state_ && state_->io) state_->io->shutdown_and_join();
    if (pool_live_) {
        if ((!state_ || !state_->pool_drained) && !adapter_->drain()) {
            quarantined_ = true;
            return false;
        }
        destroy_active_pool();
    }
    quarantined_ = false;
    return true;
}
void StageExecutor::begin(const StageLayout &layout) {
    if (used_ || layout.resident || layout.pools.empty() || layout.groups.empty() ||
        !layout.pass_count || layout.pass_count>max_passes || !layout.slot_count || layout.slot_count>max_slots ||
        layout.pools.size()>layout.groups.size() || layout.groups.size()>max_blocks ||
        !layout.workers || layout.workers>layout.slot_count || layout.distance>=layout.slot_count)
        throw std::invalid_argument("streaming executor requires a fresh streamed stage");
    used_ = true;
    std::vector<uint32_t> group_counts(layout.pools.size());
    std::vector<uint32_t> ordinals(layout.pools.size());
    uint32_t expected_pool_index = 0;
    for (size_t pi=0; pi<layout.pools.size(); ++pi) {
        const auto &pool=layout.pools[pi];
        if (pool.slots.size()!=layout.slot_count)
            throw std::invalid_argument("streaming pool layout mismatch");
        for (size_t previous=0; previous<pi; ++previous)
            if (layout.pools[previous].id==pool.id)
                throw std::invalid_argument("streaming duplicate pool id");
        for (const auto &slot : pool.slots)
            if (!slot.capacity_bytes)
                throw std::invalid_argument("streaming pool has empty slot capacity");
    }
    for (size_t i=0; i<layout.groups.size(); ++i) {
        const auto &g=layout.groups[i];
        if (expected_pool_index>=layout.pools.size() ||
            g.pool!=layout.pools[expected_pool_index].id)
            throw std::invalid_argument("streaming group pool order mismatch");
        const auto &pool=layout.pools[expected_pool_index];
        if (g.id!=i || g.slot!=ordinals[expected_pool_index]++%layout.slot_count || !g.bytes ||
            g.bytes>pool.slots[g.slot].capacity_bytes)
            throw std::invalid_argument("invalid streaming group mapping/capacity");
        ++group_counts[expected_pool_index];
        if (i+1<layout.groups.size() && layout.groups[i+1].pool!=g.pool)
            ++expected_pool_index;
    }
    if (expected_pool_index+1!=layout.pools.size())
        throw std::invalid_argument("streaming pool has no group range");
    for (auto count : group_counts)
        if (count<layout.slot_count)
            throw std::invalid_argument("streaming pool has fewer groups than slots");
    state_=std::make_unique<State>(); state_->layout=layout;
    mailbox_=std::make_unique<CompletionMailbox>(layout.slot_count*(1+TC_STREAM_MAX_READER_QUEUES));
    try {
        activate_pool(0);
        state_->io=std::make_unique<IoExecutor>(layout.workers,layout.slot_count,*mailbox_);
    } catch (...) { state_->failed=true; retry_drain(); throw; }
}
void StageExecutor::activate_pool(uint32_t pool_index) {
    if (!state_)
        throw std::logic_error("streaming invalid pool activation");
    if (pool_index>=state_->layout.pools.size())
        throw std::logic_error("streaming invalid pool activation");
    if (pool_live_ && state_->active_pool_index==pool_index) return;
    if (pool_live_) {
        if (!state_->pool_drained)
            throw std::logic_error("streaming pool switch before drain");
        destroy_active_pool();
    }
    const auto &pool=state_->layout.pools[pool_index];
    std::vector<uint64_t> capacities;
    capacities.reserve(pool.slots.size());
    for (const auto &slot : pool.slots) capacities.push_back(slot.capacity_bytes);
    state_->safety=std::make_unique<SlotSafetyTracker>(pool.id,request_,capacities);
    state_->tickets.assign(pool.slots.size(),{});
    state_->active_pool_index=pool_index;
    state_->pool_drained=false;
    pool_live_=true;
    adapter_->create_pool(pool);
    audit_increment(AuditCounter::PoolAllocations);
    ++state_->counters.pool_creates;
    state_->counters.slot_bundles+=pool.slots.size();
}
void StageExecutor::check_cancel(const std::atomic<bool> &cancel) const {
    if (!state_ || state_->owner!=std::this_thread::get_id())
        throw std::logic_error("streaming_owner_violation");
    if (cancel.load(std::memory_order_acquire)) throw std::runtime_error("streaming_cancelled");
    if (mailbox_->overflowed()) throw std::runtime_error("streaming_mailbox_overflow");
}
bool StageExecutor::consume() {
    bool progress=false;
    tc_stream_completion_v1 record;
    while (mailbox_->pop(record)) {
        if (record.struct_size!=sizeof(record) || record.version!=TC_STREAM_SLOT_ABI_V1 || record.status)
            throw std::runtime_error("streaming_completion_failed");
        if (!pool_live_ || !state_->safety ||
            record.ticket.pool!=state_->layout.pools[state_->active_pool_index].id)
            throw std::runtime_error("streaming completion for inactive pool");
        if (record.kind==TC_STREAM_FILL_COMPLETE) {
            state_->safety->accept_ready(record.ticket,record.bytes);
            if (record.bytes>UINT64_MAX-state_->counters.bytes_loaded)
                throw std::overflow_error("streaming byte counter overflow");
            state_->counters.bytes_loaded+=record.bytes; ++state_->counters.fills;
        } else if (record.kind==TC_STREAM_READER_COMPLETE)
            state_->safety->complete_reader(record.ticket,record.fence);
        else throw std::runtime_error("streaming unknown completion kind");
        progress=true;
    }
    return progress;
}
void StageExecutor::drain_active_pool() {
    if (!pool_live_ || !state_ || !state_->safety)
        throw std::logic_error("streaming missing active pool");
    if (!adapter_->drain()) throw std::runtime_error("streaming_pass_drain_failed");
    consume();
    if (!state_->safety->quiescent())
        throw std::runtime_error("streaming drain left active slot content");
    state_->pool_drained=true;
}
void StageExecutor::run_pass(uint32_t pass, uint32_t step, std::atomic<bool> &cancel,
                            std::chrono::milliseconds timeout) {
    if (!state_ || state_->owner!=std::this_thread::get_id())
        throw std::logic_error("streaming_owner_violation");
    try {
        if (state_->failed || state_->finished || !pool_live_ ||
            pass!=state_->passes || pass>=state_->layout.pass_count || timeout.count()<=0)
            throw std::logic_error("streaming invalid pass/lifecycle");
        const auto &layout=state_->layout;
        bool prefix=false;
        size_t segment_begin=0;
        uint32_t pool_index=0;
        while (segment_begin<layout.groups.size()) {
            const uint32_t pool_id=layout.groups[segment_begin].pool;
            size_t segment_end=segment_begin+1;
            while (segment_end<layout.groups.size() && layout.groups[segment_end].pool==pool_id)
                ++segment_end;
            activate_pool(pool_index);
            state_->pool_drained=false;
            auto &safety=*state_->safety;
            size_t next=segment_begin, dispatch=segment_begin;
            auto last_progress=std::chrono::steady_clock::now();
            while (next<segment_end || !safety.quiescent()) {
                check_cancel(cancel);
                bool progress=consume();
                while (next<segment_end && dispatch<segment_end &&
                       dispatch<=next+layout.distance) {
                    const auto &g=layout.groups[dispatch];
                    if (safety.state(g.slot)!=ContentState::Vacant) break;
                    auto t=safety.begin_fill(g.slot,{stage_,pass,step,g.id},g.bytes);
                    auto job=adapter_->make_fill_job(g,t); job.ticket=t;
                    if (!state_->io->enqueue(job))
                        throw std::logic_error("streaming slot and I/O credits diverged");
                    state_->tickets[g.slot]=t; ++dispatch; progress=true;
                }
                if (!prefix) { adapter_->encode_prefix(pass); prefix=true; progress=true; }
                if (next<segment_end) {
                    const auto &g=layout.groups[next];
                    if (safety.state(g.slot)==ContentState::Ready) {
                        check_cancel(cancel);
                        const auto &t=state_->tickets[g.slot];
                        adapter_->prepare_group(g,t);
                        check_cancel(cancel);
                        safety.begin_use(t);
                        const auto readers=adapter_->encode_group(g,t,*mailbox_);
                        if (readers.count>readers.fences.size())
                            throw std::runtime_error("streaming reader count exceeds capacity");
                        safety.seal_readers(t,{readers.fences.data(),readers.count});
                        ++next; ++state_->counters.groups_submitted; progress=true;
                    }
                }
                if (progress) last_progress=std::chrono::steady_clock::now();
                else {
                    if (std::chrono::steady_clock::now()-last_progress>timeout)
                        throw std::runtime_error("streaming_stall_timeout");
                    mailbox_->wait_for(std::chrono::milliseconds(5));
                }
            }
            drain_active_pool();
            segment_begin=segment_end;
            ++pool_index;
        }
        check_cancel(cancel); ++state_->passes;
    } catch (...) { state_->failed=true; retry_drain(); throw; }
}
ExecutionCounters StageExecutor::finish() {
    if (!state_ || state_->owner!=std::this_thread::get_id())
        throw std::logic_error("streaming_owner_violation");
    try {
        if (state_->failed || state_->finished || state_->passes!=state_->layout.pass_count)
            throw std::logic_error("streaming incomplete/failed execution");
        state_->io->shutdown_and_join(); consume();
        if (mailbox_->overflowed()) throw std::runtime_error("streaming_mailbox_overflow");
        if (!retry_drain()) throw std::runtime_error("streaming_drain_failed");
        state_->finished=true; return state_->counters;
    } catch (...) { state_->failed=true; retry_drain(); throw; }
}
ExecutionCounters StageExecutor::counters() const {
    if (!state_ || state_->owner!=std::this_thread::get_id())
        throw std::logic_error("streaming_owner_violation");
    return state_->counters;
}
ExecutionCounters StageExecutor::run(const StageLayout &layout, std::atomic<bool> &cancel,
                                     std::chrono::milliseconds timeout) {
    if (cancel.load()) throw std::runtime_error("streaming_cancelled");
    begin(layout);
    for (uint32_t p=0;p<layout.pass_count;++p) run_pass(p,p,cancel,timeout);
    return finish();
}
} // namespace tc::streaming
