#include "context.hpp"
#include <exception>
#include <stdexcept>

namespace tc::streaming {
struct StageExecutor::State {
    StageLayout layout;
    std::unique_ptr<SlotSafetyTracker> safety;
    std::unique_ptr<IoExecutor> io;
    std::vector<tc_stream_slot_ticket_v1> tickets;
    ExecutionCounters counters;
    uint32_t passes = 0;
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
bool StageExecutor::retry_drain() noexcept {
    if (state_ && state_->owner!=std::this_thread::get_id()) return false;
    if (state_ && state_->io) state_->io->shutdown_and_join();
    if (pool_live_) {
        if (!adapter_->drain()) { quarantined_ = true; return false; }
        adapter_->destroy_pool(); pool_live_ = false;
    }
    quarantined_ = false;
    return true;
}
void StageExecutor::begin(const StageLayout &layout) {
    if (used_ || layout.resident || layout.pools.size()!=1 || layout.groups.empty() ||
        !layout.pass_count || layout.pass_count>max_passes || !layout.slot_count || layout.slot_count>max_slots ||
        layout.slot_count>layout.groups.size() ||
        !layout.workers || layout.workers>layout.slot_count || layout.distance>=layout.slot_count)
        throw std::invalid_argument("streaming executor requires a fresh, single-class streamed stage");
    used_ = true;
    const auto &pool = layout.pools[0];
    if (pool.slots.size()!=layout.slot_count || layout.groups.size()>max_blocks)
        throw std::invalid_argument("streaming pool layout mismatch");
    for (size_t i=0; i<layout.groups.size(); ++i) {
        const auto &g=layout.groups[i];
        if (g.id!=i || g.pool!=pool.id || g.slot!=i%layout.slot_count || !g.bytes ||
            g.bytes>pool.slots[g.slot].capacity_bytes)
            throw std::invalid_argument("invalid streaming group mapping/capacity");
    }
    state_=std::make_unique<State>(); state_->layout=layout;
    std::vector<uint64_t> capacities;
    for (const auto &slot : pool.slots) capacities.push_back(slot.capacity_bytes);
    state_->safety=std::make_unique<SlotSafetyTracker>(pool.id,request_,capacities);
    state_->tickets.resize(layout.slot_count);
    mailbox_=std::make_unique<CompletionMailbox>(layout.slot_count*(1+TC_STREAM_MAX_READER_QUEUES));
    try {
        pool_live_=true; adapter_->create_pool(pool);
        ++state_->counters.pool_creates; state_->counters.slot_bundles=pool.slots.size();
        state_->io=std::make_unique<IoExecutor>(layout.workers,layout.slot_count,*mailbox_);
    } catch (...) { state_->failed=true; retry_drain(); throw; }
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
void StageExecutor::run_pass(uint32_t pass, uint32_t step, std::atomic<bool> &cancel,
                            std::chrono::milliseconds timeout) {
    if (!state_ || state_->owner!=std::this_thread::get_id())
        throw std::logic_error("streaming_owner_violation");
    try {
        if (state_->failed || state_->finished || !pool_live_ ||
            pass!=state_->passes || pass>=state_->layout.pass_count || timeout.count()<=0)
            throw std::logic_error("streaming invalid pass/lifecycle");
        auto &safety=*state_->safety;
        const auto &layout=state_->layout;
        size_t next=0, dispatch=0;
        bool prefix=false;
        auto last_progress=std::chrono::steady_clock::now();
        while (next<layout.groups.size() || !safety.quiescent()) {
            check_cancel(cancel);
            bool progress=consume();
            while (next<layout.groups.size() && dispatch<layout.groups.size() &&
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
            if (next<layout.groups.size()) {
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
        if (!adapter_->drain()) throw std::runtime_error("streaming_pass_drain_failed");
        consume(); check_cancel(cancel); ++state_->passes;
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
