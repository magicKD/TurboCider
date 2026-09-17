#include "streaming/context.hpp"
#include "streaming/audit.hpp"
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>

using namespace tc::streaming;

template<class F> static void rejects(F fn) {
    bool failed=false;
    try { fn(); } catch (const std::exception &) { failed=true; }
    assert(failed);
}
static StageLayout layout(uint32_t k, uint32_t d, uint32_t q) {
    StageLayout s;
    s.id="fake"; s.prefix=1; s.group_size=1; s.slot_count=k; s.distance=d; s.workers=q;
    s.pass_count=3;
    PoolLayout p; p.id=0; p.layout_class="u64";
    for (uint32_t i=0; i<k; ++i) p.slots.push_back({{sizeof(uint64_t)},sizeof(uint64_t)});
    s.pools.push_back(p);
    for (uint32_t i=0; i<13; ++i) s.groups.push_back({i,0,i%k,{i+1},{8},8});
    return s;
}
static StageLayout multi_layout(uint32_t k, uint32_t d, uint32_t q) {
    StageLayout s;
    s.id="multi"; s.prefix=1; s.group_size=1; s.slot_count=k; s.distance=d; s.workers=q;
    s.pass_count=3;
    for (uint32_t pool_id=0; pool_id<2; ++pool_id) {
        PoolLayout p; p.id=pool_id; p.layout_class=pool_id ? "q" : "k";
        for (uint32_t i=0; i<k; ++i)
            p.slots.push_back({{sizeof(uint64_t)},sizeof(uint64_t)});
        s.pools.push_back(std::move(p));
        for (uint32_t i=0; i<3; ++i) {
            const uint32_t id=static_cast<uint32_t>(s.groups.size());
            s.groups.push_back({id,pool_id,i%k,{id+1},{8},8});
        }
    }
    return s;
}

static StageLayout retained_multi_layout(uint32_t k, uint32_t d, uint32_t q) {
    auto result = multi_layout(k, d, q);
    result.multi_pool_policy = MultiPoolPolicy::retain_all;
    return result;
}

static StageLayout carry_layout(uint32_t d = 1, uint32_t q = 2) {
    StageLayout s;
    s.id = "carry";
    s.prefix = 1;
    s.group_size = 1;
    s.slot_count = 2;
    s.distance = d;
    s.workers = q;
    s.pass_count = 3;
    s.pass_transition = PassTransition::carry_first_group;
    PoolLayout p;
    p.id = 0;
    p.layout_class = "u64";
    p.slots.push_back({{sizeof(uint64_t)}, sizeof(uint64_t)});
    p.slots.push_back({{sizeof(uint64_t)}, sizeof(uint64_t)});
    s.pools.push_back(std::move(p));
    // An odd suffix forces the physical slot mapping to rotate between passes:
    // pass 0 group 0 uses slot 0, while pass 1 group 0 is carried in slot 1.
    for (uint32_t i = 0; i < 13; ++i)
        s.groups.push_back({i, 0, i % 2, {i + 1}, {8}, 8});
    return s;
}

// Independent readers sample real backing contents before AND after a delay.
// Two queues must both finish before the CPU is allowed to rewrite a slot.
class FakeModel final : public ModelSlotAdapter {
    struct Read { uint32_t slot; uint64_t expected; tc_stream_slot_ticket_v1 ticket;
                  tc_stream_reader_fence_v1 fence; CompletionMailbox *mailbox; };
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Read> queues[2];
    std::thread gpu[2];
    bool stop=false;
    uint32_t pending=0;
    uint64_t sequence=0;
    void reader(unsigned q) {
        while (true) {
            Read r;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock,[&]{return stop || !queues[q].empty();});
                if (stop && queues[q].empty()) return;
                r=queues[q].front(); queues[q].pop_front();
            }
            assert(value(r.ticket.pool, r.slot) == r.expected);
            std::this_thread::sleep_for(std::chrono::microseconds(q?150:20));
            assert(value(r.ticket.pool, r.slot) == r.expected);
            tc_stream_completion_v1 event{};
            event.struct_size=sizeof(event); event.version=TC_STREAM_SLOT_ABI_V1;
            event.kind=TC_STREAM_READER_COMPLETE; event.ticket=r.ticket; event.fence=r.fence;
            assert(r.mailbox->post(event));
            { std::lock_guard lock(mutex); --pending; }
            changed.notify_all();
        }
    }
public:
    std::vector<uint64_t> values;
    std::map<uint32_t, std::vector<uint64_t>> retained_values;
    unsigned creates=0, destroys=0, prefixes=0;
    std::atomic<int> fills{0};
    std::mutex event_mutex;
    std::vector<std::string> events;
    bool fail_fill=false, fail_create=false, fake_bad_drain=false, short_fill=false;
    bool claim_overlap=false, immediate_readers=false;
    bool retain_pools=false;
    unsigned fail_create_at=0;
    std::atomic<bool> *cancel_on_prepare=nullptr;
    static uint64_t tag(const tc_stream_slot_ticket_v1 &t) {return 1+t.item.pass*1000+t.item.group;}
    FakeModel() {
        for(unsigned q=0;q<2;++q) gpu[q]=std::thread([this,q]{reader(q);});
    }
    ~FakeModel() {
        {std::lock_guard lock(mutex); stop=true;}
        changed.notify_all();
        for(auto &t:gpu)t.join();
    }
    uint64_t &value(uint32_t pool, uint32_t slot) {
        return retain_pools ? retained_values.at(pool).at(slot) : values.at(slot);
    }
    bool supports_multi_pool_policy(MultiPoolPolicy policy) const noexcept override {
        return policy == MultiPoolPolicy::serial ||
            (retain_pools && policy == MultiPoolPolicy::retain_all);
    }
    void create_pool(const PoolLayout &p) override {
        ++creates;
        if (retain_pools)
            retained_values[p.id].resize(p.slots.size());
        else
            values.resize(p.slots.size());
        if(fail_create || (fail_create_at && creates==fail_create_at))
            throw std::runtime_error("injected partial create");
    }
    FillJob make_fill_job(const Group &, const tc_stream_slot_ticket_v1 &t) override {
        {
            std::lock_guard lock(event_mutex);
            events.push_back("dispatch:" + std::to_string(t.item.pass) +
                             ":" + std::to_string(t.item.group));
        }
        return {t,this,[](void *opaque,const tc_stream_slot_ticket_v1 *ticket,
                          const std::atomic<bool> *cancel,uint64_t *bytes) {
            auto &m=*static_cast<FakeModel *>(opaque);
            if(m.fail_fill || cancel->load())return -1;
            std::this_thread::sleep_for(std::chrono::microseconds(ticket->item.group%3*35));
            m.value(ticket->pool,ticket->slot)=tag(*ticket); *bytes=m.short_fill?4:8; ++m.fills;
            {
                std::lock_guard lock(m.event_mutex);
                m.events.push_back("fill:" + std::to_string(ticket->item.pass) +
                                   ":" + std::to_string(ticket->item.group));
            }
            return 0;
        }};
    }
    void encode_prefix(uint32_t) override {++prefixes;}
    void prepare_group(const Group &,const tc_stream_slot_ticket_v1 &t) override {
        assert(value(t.pool,t.slot)==tag(t));
        if(cancel_on_prepare)cancel_on_prepare->store(true);
    }
    bool overlap_next_fill_after_claim() const noexcept override {
        return claim_overlap;
    }
    ReaderSet encode_group(const Group &,const tc_stream_slot_ticket_v1 &t,
                            CompletionMailbox &mailbox) override {
        ReaderSet set;
        {
            std::lock_guard lock(event_mutex);
            events.push_back("encode:" + std::to_string(t.item.pass) +
                             ":" + std::to_string(t.item.group));
        }
        if (immediate_readers) {
            set.count=1;
            set.fences[0]={1,++sequence};
            set.already_complete=true;
            return set;
        }
        set.count=2;
        std::lock_guard lock(mutex);
        for(unsigned q=0;q<2;++q){
            set.fences[q]={q+1,++sequence};
            queues[q].push_back({t.slot,tag(t),t,set.fences[q],&mailbox}); ++pending;
        }
        changed.notify_all(); return set;
    }
    bool drain() noexcept override {
        std::unique_lock lock(mutex); changed.wait(lock,[&]{return pending==0;});
        return !fake_bad_drain;
    }
    void destroy_pool() noexcept override {
        std::lock_guard lock(mutex); assert(pending==0); values.clear(); ++destroys;
    }
    void destroy_pool(uint32_t pool) noexcept override {
        std::lock_guard lock(mutex);
        assert(pending==0);
        retained_values.erase(pool);
        ++destroys;
    }
};

int main() {
    const uint64_t cap=8;
    {
        SlotSafetyTracker pool(2,5,{&cap,1});
        auto t=pool.begin_fill(0,{1,2,3,4},8);
        pool.accept_ready(t,8); pool.begin_use(t);
        tc_stream_reader_fence_v1 fs[]={{1,5},{2,6}};
        pool.seal_readers(t,fs); pool.complete_reader(t,fs[1]);
        assert(pool.state(0)==ContentState::AwaitingFence);
        pool.complete_reader(t,fs[0]); assert(pool.quiescent());
        assert(pool.capacity_bytes()==8); // Contents vacant, backing still live.
        auto next=pool.begin_fill(0,{1,2,3,5},8);
        assert(next.content_generation==t.content_generation+1);
        rejects([&]{pool.accept_ready(t,8);}); assert(pool.poisoned());
        rejects([&]{pool.accept_ready(next,8);});
    }
    {
        SlotSafetyTracker pool(1,1,{&cap,1});
        auto t=pool.begin_fill(0,{},8);
        rejects([&]{pool.begin_fill(0,{},8);}); assert(pool.poisoned());
        (void)t;
    }
    {
        SlotSafetyTracker pool(1,1,{&cap,1});
        auto t=pool.begin_fill(0,{},8); pool.accept_ready(t,8); pool.begin_use(t);
        rejects([&]{pool.complete_reader(t,{1,1});}); // unsealed/early callback
    }
    {
        CompletionMailbox mailbox(1);
        tc_stream_completion_v1 e{}; assert(mailbox.post(e)); assert(!mailbox.post(e));
        assert(mailbox.overflowed()); assert(mailbox.pop(e)); assert(mailbox.overflowed());
    }
    {
        SlotSafetyTracker pool(1,1,{&cap,1});
        std::thread intruder([&]{rejects([&]{pool.begin_fill(0,{},8);});});
        intruder.join(); assert(pool.quiescent());
    }
    unsigned runs=0;
    for(uint32_t k=1;k<=3;++k)for(uint32_t d=0;d<k;++d)for(uint32_t q=1;q<=k;++q){
        auto model=std::make_shared<FakeModel>();
        StageExecutor exec(3,7,model); std::atomic<bool> cancel{false};
        const auto result=exec.run(layout(k,d,q),cancel);
        assert(result.pool_creates==1 && result.slot_bundles==k);
        assert(result.fills==39 && result.groups_submitted==39 && result.bytes_loaded==39*8);
        assert(model->creates==1 && model->destroys==1 && model->prefixes==3);
        assert(!exec.quarantined());
        rejects([&]{exec.run(layout(k,d,q),cancel);});
        ++runs;
    }
    for (uint32_t k=1; k<=3; ++k) {
        auto model=std::make_shared<FakeModel>(); std::atomic<bool> cancel{false};
        StageExecutor exec(3,7,model);
        const auto result=exec.run(multi_layout(k,k-1,k),cancel);
        // Two ordered pools are switched at a barrier.  Each pass visits both
        // pools exactly once, so pool creation is deterministic and bounded.
        assert(result.pool_creates==6 && result.slot_bundles==6*k);
        assert(result.fills==18 && result.groups_submitted==18 && result.bytes_loaded==18*8);
        assert(model->creates==6 && model->destroys==6 && model->prefixes==3);
        assert(!exec.quarantined());
    }
    for (uint32_t k=1; k<=3; ++k) {
        auto model=std::make_shared<FakeModel>();
        model->retain_pools=true;
        std::atomic<bool> cancel{false};
        StageExecutor exec(3,7,model);
        const auto result=exec.run(retained_multi_layout(k,k-1,k),cancel);
        assert(result.pool_creates==2 && result.slot_bundles==2*k);
        assert(result.fills==18 && result.groups_submitted==18 && result.bytes_loaded==18*8);
        assert(model->creates==2 && model->destroys==2 && model->prefixes==3);
        assert(model->retained_values.empty() && !exec.quarantined());
    }
    {
        auto model=std::make_shared<FakeModel>();
        model->claim_overlap=true;
        model->immediate_readers=true;
        std::atomic<bool> cancel{false};
        StageExecutor exec(3,7,model);
        const auto result=exec.run(layout(2,0,1),cancel);
        assert(result.fills==39 && result.groups_submitted==39);
        const auto next_dispatch=std::find(
            model->events.begin(),model->events.end(),"dispatch:0:1");
        const auto current_encode=std::find(
            model->events.begin(),model->events.end(),"encode:0:0");
        assert(next_dispatch!=model->events.end() &&
               current_encode!=model->events.end() &&
               next_dispatch<current_encode);
    }
    {
        auto model=std::make_shared<FakeModel>(); std::atomic<bool> cancel{false};
        model->fail_create_at=2;
        StageExecutor exec(3,7,model);
        rejects([&]{exec.run(multi_layout(2,1,1),cancel);});
        assert(model->creates==2 && model->destroys==2 && !exec.quarantined());
    }
    {
        auto model=std::make_shared<FakeModel>();
        StageExecutor exec(3,7,model);
        auto invalid=multi_layout(2,1,1);
        invalid.groups[3].pool=invalid.pools[0].id;
        rejects([&]{exec.begin(invalid);});
        assert(model->creates==0 && model->destroys==0);
    }
    for(unsigned failure=0;failure<5;++failure){
        auto model=std::make_shared<FakeModel>(); std::atomic<bool> cancel{false};
        model->fail_create=failure==0; model->fail_fill=failure==1;
        if(failure==2)model->cancel_on_prepare=&cancel;
        if(failure==3)model->fake_bad_drain=true;
        if(failure==4)model->short_fill=true;
        StageExecutor exec(3,7,model);
        rejects([&]{exec.run(layout(3,2,2),cancel);});
        if(failure==3){
            assert(exec.quarantined() && model->destroys==0);
            model->fake_bad_drain=false;
            assert(exec.retry_drain());
        }
        assert(model->destroys==1);
    }
    {
        auto model=std::make_shared<FakeModel>(); std::atomic<bool> cancel{false};
        StageExecutor exec(3,7,model);
        const auto result=exec.run(carry_layout(),cancel);
        assert(result.pool_creates==1 && result.slot_bundles==2);
        assert(result.fills==39 && result.groups_submitted==39 &&
               result.bytes_loaded==39*8);
        assert(model->creates==1 && model->destroys==1 && model->prefixes==3);
        auto carry_fill = std::find(model->events.begin(), model->events.end(),
                                    "dispatch:1:0");
        auto previous_last = std::find(model->events.begin(), model->events.end(),
                                       "encode:0:12");
        assert(carry_fill != model->events.end() &&
               previous_last != model->events.end() &&
               carry_fill < previous_last);
        assert(!exec.quarantined());
    }
    {
        auto model=std::make_shared<FakeModel>(); std::atomic<bool> cancel{false};
        StageExecutor exec(3,7,model);
        exec.begin(carry_layout());
        exec.run_pass(0,10,cancel);
        // Carry tickets are created before the caller supplies the next pass.
        // Requiring consecutive steps keeps the prefetched ticket identity
        // equal to the actual pass identity instead of silently accepting a
        // different scheduler step.
        rejects([&]{exec.run_pass(1,12,cancel);});
        assert(model->destroys==1 && !exec.quarantined());
    }
    {
        auto model=std::make_shared<FakeModel>();
        StageExecutor exec(3,7,model);
        auto invalid=carry_layout();
        invalid.groups[0].blocks.push_back(99);
        rejects([&]{exec.begin(invalid);});
        assert(model->creates==0 && model->destroys==0);
    }
#ifdef TURBOCIDER_ENABLE_AUDIT_COUNTERS
    // The audit build must observe the framework only when this explicit
    // streaming executor is entered.  Keep the assertion independent of the
    // larger matrix above so it remains stable if that matrix grows.
    audit_reset();
    {
        auto model=std::make_shared<FakeModel>(); std::atomic<bool> cancel{false};
        StageExecutor exec(3,7,model);
        exec.run(layout(3,2,2),cancel);
    }
    const auto audit = audit_snapshot();
    assert(audit.pool_allocations == 1);
    assert(audit.worker_threads == 2);
    assert(audit.steady_framework_allocations == 0);
    assert(audit.steady_framework_thread_creates == 0);
    assert(audit.framework_hooks == 0);
    assert(audit.memory_probes == 0);
    assert(audit.cache_clear_or_unload_calls == 0);
    audit_reset();
    {
        auto model=std::make_shared<FakeModel>(); std::atomic<bool> cancel{false};
        StageExecutor exec(3,7,model);
        exec.run(multi_layout(2,1,1),cancel);
    }
    const auto multi_audit = audit_snapshot();
    assert(multi_audit.steady_framework_allocations > 0);
    assert(multi_audit.steady_framework_thread_creates == 0);
    audit_reset();
    {
        auto model=std::make_shared<FakeModel>();
        model->retain_pools=true;
        std::atomic<bool> cancel{false};
        StageExecutor exec(3,7,model);
        exec.run(retained_multi_layout(2,1,1),cancel);
    }
    const auto retained_multi_audit = audit_snapshot();
    assert(retained_multi_audit.pool_allocations == 2);
    assert(retained_multi_audit.steady_framework_allocations == 0);
    assert(retained_multi_audit.steady_framework_thread_creates == 0);
#endif
    std::cout<<"PASS streaming executor: "<<runs
             <<" K/D/Q combinations, multi-class barriers, two independent readers, faults and cleanup\n";
}
