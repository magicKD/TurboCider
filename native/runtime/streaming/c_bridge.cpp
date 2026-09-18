#include "context.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
using namespace tc::streaming;
int fail(char *error, size_t size, const char *reason) {
    if (error && size) std::snprintf(error,size,"%s",reason);
    return 0;
}
std::string fixed_string(const char *value, size_t capacity,
                         const char *detail) {
    const size_t length = ::strnlen(value, capacity);
    if (!length || length == capacity)
        throw std::invalid_argument(detail);
    return std::string(value, length);
}
template <size_t N>
void copy_fixed(char (&target)[N], const std::string &source,
                const char *detail) {
    if (source.empty() || source.size() >= N)
        throw std::invalid_argument(detail);
    std::memcpy(target, source.data(), source.size());
    target[source.size()] = '\0';
}
class CAdapter final : public ModelSlotAdapter {
public:
    tc_stream_adapter_v1 ops{};
    tc_stream_adapter_v2 ops_v2{};
    bool version_2 = false;
    uint32_t active_pool = 0;
    struct Job {
        CAdapter *self = nullptr;
        tc_stream_group_v1 group{};
        char error[1024]{};
    };
    std::vector<Job> jobs;
    char error[1024]{};
    explicit CAdapter(const tc_stream_adapter_v1 &o,uint32_t slots):ops(o),jobs(slots){}
    explicit CAdapter(const tc_stream_adapter_v2 &o,uint32_t slots)
        :ops_v2(o),version_2(true),jobs(slots){}
    void checked(int success) {if (!success)throw std::runtime_error(error[0]?error:"streaming adapter failed");}
    void clear_error() noexcept {error[0]=0;}
    static tc_stream_group_v1 view(const Group &g) {
        return {g.id,g.slot,uint32_t(g.blocks.size()),g.blocks.data(),g.bytes};
    }
    void create_pool(const PoolLayout &pool) override {
        active_pool=pool.id;
        for(uint32_t i=0;i<pool.slots.size();++i) {
            clear_error();
            checked(version_2 ?
                ops_v2.allocate_slot(ops_v2.user,pool.id,i,pool.slots[i].capacity_bytes,error,sizeof(error)) :
                ops.allocate_slot(ops.user,i,pool.slots[i].capacity_bytes,error,sizeof(error)));
        }
    }
    FillJob make_fill_job(const Group &g,const tc_stream_slot_ticket_v1 &t) override {
        auto &job=jobs.at(t.slot);job.self=this;
        // Group block spans are owned by StageExecutor::State::layout and stay
        // stable until every fill worker is joined.  Borrow them directly so
        // the steady refill path never copies or grows a vector.
        job.group={g.id,g.slot,uint32_t(g.blocks.size()),g.blocks.data(),g.bytes};
        job.error[0]=0;
        return {t,&job,[](void *raw,const tc_stream_slot_ticket_v1 *ticket,
                         const std::atomic<bool> *cancel,uint64_t *bytes){
            auto &j=*static_cast<Job *>(raw);
            auto query=[](const void *v){return static_cast<const std::atomic<bool> *>(v)->load()?1:0;};
            const int success=j.self->version_2 ?
                j.self->ops_v2.fill(j.self->ops_v2.user,ticket,&j.group,query,cancel,bytes,j.error,sizeof(j.error)) :
                j.self->ops.fill(j.self->ops.user,ticket,&j.group,query,cancel,bytes,j.error,sizeof(j.error));
            return success?0:-1;
        }};
    }
    void encode_prefix(uint32_t pass) override {
        clear_error();
        checked(version_2 ? ops_v2.prefix(ops_v2.user,pass,error,sizeof(error)) :
                            ops.prefix(ops.user,pass,error,sizeof(error)));
    }
    void prepare_group(const Group &g,const tc_stream_slot_ticket_v1 &t) override {
        auto v=view(g);clear_error();
        checked(version_2 ? ops_v2.prepare(ops_v2.user,&t,&v,error,sizeof(error)) :
                            ops.prepare(ops.user,&t,&v,error,sizeof(error)));
    }
    ReaderSet encode_group(const Group &g,const tc_stream_slot_ticket_v1 &t,CompletionMailbox &mailbox) override {
        tc_stream_completion_sink_v1 sink{&mailbox,[](void *m,const tc_stream_slot_ticket_v1 *ticket,
                                                    tc_stream_reader_fence_v1 fence,int32_t status){
            if(!ticket)return 0;
            tc_stream_completion_v1 event{};event.struct_size=sizeof(event);event.version=TC_STREAM_SLOT_ABI_V1;
            event.kind=TC_STREAM_READER_COMPLETE;event.status=status;event.ticket=*ticket;event.fence=fence;
            return static_cast<CompletionMailbox *>(m)->post(event)?1:0;
        }};
        auto v=view(g);tc_stream_reader_set_v1 output{};clear_error();
        checked(version_2 ?
            ops_v2.encode(ops_v2.user,&t,&v,&sink,&output,error,sizeof(error)) :
            ops.encode(ops.user,&t,&v,&sink,&output,error,sizeof(error)));
        if(output.count>TC_STREAM_MAX_READER_QUEUES)throw std::runtime_error("streaming reader count overflow");
        ReaderSet result;result.count=output.count;
        std::copy_n(output.fences,output.count,result.fences.begin());return result;
    }
    bool drain() noexcept override {
        char detail[1024]{};
        try {
            return version_2 ? ops_v2.drain(ops_v2.user,detail,sizeof(detail))!=0 :
                               ops.drain(ops.user,detail,sizeof(detail))!=0;
        } catch(...) {return false;}
    }
    void destroy_pool() noexcept override {
        if(version_2)ops_v2.destroy_pool(ops_v2.user,active_pool);
        else ops.destroy_pool(ops.user);
    }
};
}
struct tc_stream_executor {
    std::shared_ptr<CAdapter> adapter;
    std::unique_ptr<StageExecutor> executor;
    std::atomic<bool> cancel{false};
    std::thread::id owner=std::this_thread::get_id();
};
namespace {
void owner(tc_stream_executor *h) {
    if(!h || h->owner!=std::this_thread::get_id())throw std::logic_error("streaming_owner_violation");
}
int failure(tc_stream_executor *h,char *error,size_t size,const std::exception &e) {
    if(h && h->adapter && h->owner==std::this_thread::get_id()){
        if(h->adapter->error[0])return fail(error,size,h->adapter->error);
        // run_pass has joined workers on failure before inspecting job errors.
        for(const auto &j:h->adapter->jobs)if(j.error[0])return fail(error,size,j.error);
    }
    return fail(error,size,e.what());
}
}
extern "C" int tc_stream_executor_create_v1(const tc_stream_stage_plan_v1 *p,const tc_stream_adapter_v1 *a,
                                             tc_stream_executor **out,char *error,size_t size) {
    if(out)*out=nullptr;
    if(!out || !p || !a || p->struct_size!=sizeof(*p) || p->version!=TC_STREAM_SLOT_ABI_V1 ||
       a->struct_size!=sizeof(*a) || a->version!=TC_STREAM_SLOT_ABI_V1 || !p->request_generation ||
       !p->slot_count || p->slot_count>max_slots || !p->group_count || p->group_count>max_blocks ||
       !p->slot_capacity_bytes || !p->groups || !a->allocate_slot || !a->destroy_pool || !a->fill ||
       !a->prefix || !a->prepare || !a->encode || !a->drain)
        return fail(error,size,"streaming invalid plan/adapter ABI");
    std::unique_ptr<tc_stream_executor> h;
    try {
        StageLayout layout;layout.slot_count=p->slot_count;layout.distance=p->prefetch_distance;
        layout.workers=p->io_workers;layout.pass_count=p->pass_count;
        PoolLayout pool;pool.id=p->pool;
        for(uint32_t i=0;i<p->slot_count;++i)pool.slots.push_back({{},p->slot_capacity_bytes[i]});
        layout.pools.push_back(std::move(pool));
        uint64_t blocks=0;
        for(uint32_t i=0;i<p->group_count;++i){
            const auto &g=p->groups[i];blocks+=g.block_count;
            if(!g.blocks || !g.block_count || blocks>max_blocks)throw std::invalid_argument("streaming invalid group blocks");
            Group group;group.id=g.group;group.slot=g.slot;group.pool=p->pool;group.bytes=g.content_bytes;
            group.blocks.assign(g.blocks,g.blocks+g.block_count);layout.groups.push_back(std::move(group));
        }
        h=std::make_unique<tc_stream_executor>();h->adapter=std::make_shared<CAdapter>(*a,p->slot_count);
        h->executor=std::make_unique<StageExecutor>(p->stage,p->request_generation,h->adapter);
        h->executor->begin(layout);*out=h.release();return 1;
    } catch(const std::exception &e){
        const int result=failure(h.get(),error,size,e);
        if(h && h->executor && h->executor->quarantined())*out=h.release();
        return result;
    } catch (...) {
        // Non-standard adapter exceptions need the same ownership handoff as
        // std::exception. Otherwise the local handle's destructor can terminate
        // while a partially created pool still has unproven readers.
        if (h && h->executor && h->executor->quarantined())
            *out = h.release();
        return fail(error, size, "streaming unknown create error");
    }
}
extern "C" int tc_stream_executor_create_v3(
        const tc_stream_stage_plan_v3 *p, const tc_stream_adapter_v1 *a,
        tc_stream_executor **out, char *error, size_t size) {
    if (out) *out = nullptr;
    if (!out || !p || !a || p->struct_size != sizeof(*p) ||
        p->version != TC_STREAM_SLOT_ABI_V3 ||
        a->struct_size != sizeof(*a) || a->version != TC_STREAM_SLOT_ABI_V1 ||
        !p->request_generation || !p->slot_count ||
        p->slot_count > max_slots || !p->group_count ||
        p->group_count > max_blocks || !p->slot_capacity_bytes || !p->groups ||
        (p->pass_transition != TC_STREAM_PASS_RELOAD_V3 &&
         p->pass_transition != TC_STREAM_PASS_CARRY_FIRST_GROUP_V3) ||
        !a->allocate_slot || !a->destroy_pool || !a->fill || !a->prefix ||
        !a->prepare || !a->encode || !a->drain)
        return fail(error, size, "streaming invalid v3 plan/adapter ABI");
    std::unique_ptr<tc_stream_executor> h;
    try {
        StageLayout layout;
        layout.slot_count = p->slot_count;
        layout.distance = p->prefetch_distance;
        layout.workers = p->io_workers;
        layout.pass_count = p->pass_count;
        layout.pass_transition =
            p->pass_transition == TC_STREAM_PASS_CARRY_FIRST_GROUP_V3 ?
                PassTransition::carry_first_group : PassTransition::reload;
        PoolLayout pool;
        pool.id = p->pool;
        for (uint32_t index = 0; index < p->slot_count; ++index)
            pool.slots.push_back({{}, p->slot_capacity_bytes[index]});
        layout.pools.push_back(std::move(pool));
        uint64_t blocks = 0;
        for (uint32_t index = 0; index < p->group_count; ++index) {
            const auto &source = p->groups[index];
            blocks += source.block_count;
            if (!source.blocks || !source.block_count || blocks > max_blocks)
                throw std::invalid_argument("streaming invalid v3 group blocks");
            Group group;
            group.id = source.group;
            group.slot = source.slot;
            group.pool = p->pool;
            group.bytes = source.content_bytes;
            group.blocks.assign(source.blocks,
                                source.blocks + source.block_count);
            layout.groups.push_back(std::move(group));
        }
        h = std::make_unique<tc_stream_executor>();
        h->adapter = std::make_shared<CAdapter>(*a, p->slot_count);
        h->executor = std::make_unique<StageExecutor>(
            p->stage, p->request_generation, h->adapter);
        h->executor->begin(layout);
        *out = h.release();
        return 1;
    } catch (const std::exception &e) {
        const int result = failure(h.get(), error, size, e);
        if (h && h->executor && h->executor->quarantined()) *out = h.release();
        return result;
    } catch (...) {
        if (h && h->executor && h->executor->quarantined()) *out = h.release();
        return fail(error, size, "streaming unknown v3 create error");
    }
}
extern "C" int tc_stream_executor_create_v2(const tc_stream_stage_plan_v2 *p,
                                             const tc_stream_adapter_v2 *a,
                                             tc_stream_executor **out,
                                             char *error, size_t size) {
    if (out) *out = nullptr;
    if (!out || !p || !a || p->struct_size != sizeof(*p) ||
        p->version != TC_STREAM_SLOT_ABI_V2 || a->struct_size != sizeof(*a) ||
        a->version != TC_STREAM_SLOT_ABI_V2 || !p->request_generation ||
        !p->pool_count || p->pool_count > max_blocks || !p->pools ||
        !p->slot_count || p->slot_count > max_slots || !p->group_count ||
        p->group_count > max_blocks || !p->groups || !a->allocate_slot ||
        !a->destroy_pool || !a->fill || !a->prefix || !a->prepare ||
        !a->encode || !a->drain)
        return fail(error, size, "streaming invalid v2 plan/adapter ABI");
    std::unique_ptr<tc_stream_executor> h;
    try {
        StageLayout layout;
        layout.slot_count = p->slot_count;
        layout.distance = p->prefetch_distance;
        layout.workers = p->io_workers;
        layout.pass_count = p->pass_count;
        layout.pools.reserve(p->pool_count);
        for (uint32_t index = 0; index < p->pool_count; ++index) {
            const auto &source = p->pools[index];
            if (source.struct_size != sizeof(source) ||
                source.version != TC_STREAM_SLOT_ABI_V2 ||
                source.slot_count != p->slot_count ||
                !source.slot_capacity_bytes)
                throw std::invalid_argument("streaming invalid v2 pool plan");
            PoolLayout pool;
            pool.id = source.pool;
            pool.slots.reserve(source.slot_count);
            for (uint32_t slot = 0; slot < source.slot_count; ++slot)
                pool.slots.push_back({{}, source.slot_capacity_bytes[slot]});
            layout.pools.push_back(std::move(pool));
        }
        uint64_t blocks = 0;
        for (uint32_t index = 0; index < p->group_count; ++index) {
            const auto &source = p->groups[index];
            blocks += source.block_count;
            if (!source.blocks || !source.block_count || blocks > max_blocks)
                throw std::invalid_argument("streaming invalid v2 group blocks");
            Group group;
            group.id = source.group;
            group.pool = source.pool;
            group.slot = source.slot;
            group.bytes = source.content_bytes;
            group.blocks.assign(source.blocks, source.blocks + source.block_count);
            layout.groups.push_back(std::move(group));
        }
        h = std::make_unique<tc_stream_executor>();
        h->adapter = std::make_shared<CAdapter>(*a, p->slot_count);
        h->executor = std::make_unique<StageExecutor>(p->stage,
                                                       p->request_generation,
                                                       h->adapter);
        h->executor->begin(layout);
        *out = h.release();
        return 1;
    } catch (const std::exception &e) {
        const int result = failure(h.get(), error, size, e);
        if (h && h->executor && h->executor->quarantined()) *out = h.release();
        return result;
    } catch (...) {
        if (h && h->executor && h->executor->quarantined()) *out = h.release();
        return fail(error, size, "streaming unknown v2 create error");
    }
}
extern "C" int tc_stream_executor_run_pass(tc_stream_executor *h,uint32_t pass,uint32_t step,char *error,size_t size){
    try {owner(h);h->executor->run_pass(pass,step,h->cancel);return 1;}
    catch(const std::exception &e){return failure(h,error,size,e);}
    catch(...){return fail(error,size,"streaming unknown pass error");}
}
extern "C" int tc_stream_executor_finish(tc_stream_executor *h,char *error,size_t size){
    try {owner(h);h->executor->finish();return 1;}
    catch(const std::exception &e){return failure(h,error,size,e);}
    catch(...){return fail(error,size,"streaming unknown finish error");}
}
extern "C" int tc_stream_executor_counters(tc_stream_executor *h,tc_stream_counters_v1 *out,char *error,size_t size){
    try {owner(h);if(!out)throw std::invalid_argument("missing counters output");auto c=h->executor->counters();
        *out={c.pool_creates,c.slot_bundles,c.fills,c.bytes_loaded,
             c.groups_submitted,c.wait_seconds};return 1;}
    catch(const std::exception &e){return fail(error,size,e.what());}
}
extern "C" int tc_stream_executor_enable_receipt_v1(
        tc_stream_executor *h, const tc_stream_receipt_config_v1 *config,
        char *error, size_t size) {
    try {
        owner(h);
        if (!config || config->struct_size != sizeof(*config) ||
            config->version != TC_STREAM_RECEIPT_ABI_V1 ||
            !config->source_generation)
            throw std::invalid_argument("streaming invalid receipt ABI");
        auto layout_digest = fixed_string(
            config->layout_digest, sizeof(config->layout_digest),
            "streaming invalid receipt layout digest");
        auto implementation = fixed_string(
            config->implementation, sizeof(config->implementation),
            "streaming invalid receipt implementation");
        if (layout_digest.size() != 64)
            throw std::invalid_argument(
                "streaming invalid receipt layout digest");
        h->executor->enable_receipt({
            std::move(layout_digest), std::move(implementation),
            config->source_generation});
        return 1;
    } catch (const std::exception &e) {
        return fail(error, size, e.what());
    }
}
extern "C" int tc_stream_executor_receipt_v1(
        tc_stream_executor *h, tc_stream_receipt_v1 *out,
        char *error, size_t size) {
    try {
        owner(h);
        if (!out || out->struct_size != sizeof(*out) ||
            out->version != TC_STREAM_RECEIPT_ABI_V1)
            throw std::invalid_argument("streaming invalid receipt output ABI");
        const auto receipt = h->executor->receipt();
        tc_stream_receipt_v1 value{};
        value.struct_size = sizeof(value);
        value.version = TC_STREAM_RECEIPT_ABI_V1;
        value.stage_index = receipt->stage_index;
        value.completed_passes = receipt->completed_passes;
        value.completed_groups = receipt->completed_groups;
        value.fills = receipt->fills;
        value.groups_submitted = receipt->groups_submitted;
        value.logical_read_bytes = receipt->logical_read_bytes;
        value.reader_fences_issued = receipt->reader_fences_issued;
        value.reader_fences_completed = receipt->reader_fences_completed;
        value.source_generation = receipt->source_generation;
        value.drained = receipt->drain_completed ? 1 : 0;
        value.verified = 1;
        std::snprintf(value.event_digest, sizeof(value.event_digest), "%s",
                      receipt->event_digest.c_str());
        std::snprintf(value.canonical_digest,
                      sizeof(value.canonical_digest), "%s",
                      receipt->canonical_digest.c_str());
        *out = value;
        return 1;
    } catch (const std::exception &e) {
        return fail(error, size, e.what());
    }
}
extern "C" int tc_stream_executor_receipt_v2(
        tc_stream_executor *h, tc_stream_receipt_v2 *out,
        char *error, size_t size) {
    try {
        owner(h);
        if (!out || out->struct_size != sizeof(*out) ||
            out->version != TC_STREAM_RECEIPT_ABI_V2)
            throw std::invalid_argument("streaming invalid receipt output ABI");
        const auto receipt = h->executor->receipt();
        const bool query_only =
            out->group_capacity == 0 && !out->groups &&
            out->pool_selection_capacity == 0 && !out->pool_selections &&
            out->carry_capacity == 0 && !out->carries;
        if (!query_only) {
            if (receipt->groups.size() > out->group_capacity ||
                (receipt->groups.size() && !out->groups) ||
                receipt->pool_selections.size() >
                    out->pool_selection_capacity ||
                (receipt->pool_selections.size() &&
                 !out->pool_selections) ||
                receipt->carries.size() > out->carry_capacity ||
                (receipt->carries.size() && !out->carries))
                throw std::invalid_argument(
                    "streaming receipt output capacity is insufficient");
        }

        const uint32_t group_capacity = out->group_capacity;
        tc_stream_group_receipt_v2 *groups = out->groups;
        const uint32_t pool_capacity = out->pool_selection_capacity;
        tc_stream_pool_selection_receipt_v2 *pools = out->pool_selections;
        const uint32_t carry_capacity = out->carry_capacity;
        tc_stream_carry_receipt_v2 *carries = out->carries;

        tc_stream_receipt_v2 value{};
        value.struct_size = sizeof(value);
        value.version = TC_STREAM_RECEIPT_ABI_V2;
        value.stage_index = receipt->stage_index;
        value.completed_passes = receipt->completed_passes;
        value.completed_groups = receipt->completed_groups;
        value.fills = receipt->fills;
        value.groups_submitted = receipt->groups_submitted;
        value.logical_read_bytes = receipt->logical_read_bytes;
        value.reader_fences_issued = receipt->reader_fences_issued;
        value.reader_fences_completed = receipt->reader_fences_completed;
        value.source_generation = receipt->source_generation;
        value.drained = receipt->drain_completed ? 1 : 0;
        value.verified = 1;
        copy_fixed(value.stage_id, receipt->stage_id,
                   "streaming invalid receipt stage id");
        copy_fixed(value.layout_digest, receipt->layout_digest,
                   "streaming invalid receipt layout digest");
        copy_fixed(value.implementation, receipt->implementation,
                   "streaming invalid receipt implementation");
        copy_fixed(value.event_digest, receipt->event_digest,
                   "streaming invalid receipt event digest");
        copy_fixed(value.canonical_digest, receipt->canonical_digest,
                   "streaming invalid receipt canonical digest");
        if (receipt->groups.size() > UINT32_MAX ||
            receipt->pool_selections.size() > UINT32_MAX ||
            receipt->carries.size() > UINT32_MAX)
            throw std::overflow_error("streaming receipt output is too large");
        value.group_capacity = group_capacity;
        value.group_count = static_cast<uint32_t>(receipt->groups.size());
        value.groups = groups;
        value.pool_selection_capacity = pool_capacity;
        value.pool_selection_count = static_cast<uint32_t>(
            receipt->pool_selections.size());
        value.pool_selections = pools;
        value.carry_capacity = carry_capacity;
        value.carry_count = static_cast<uint32_t>(receipt->carries.size());
        value.carries = carries;

        if (!query_only) {
            for (size_t index = 0; index < receipt->groups.size(); ++index) {
                const auto &source = receipt->groups[index];
                tc_stream_group_receipt_v2 target{};
                target.pass = source.pass;
                target.group = source.group;
                target.pool = source.pool;
                target.slot = source.slot;
                target.step = source.step;
                target.fill_count = source.fill_count;
                target.reader_count = source.reader_count;
                target.flags =
                    (source.fill_completed ?
                         TC_STREAM_GROUP_FILL_COMPLETED_V2 : 0u) |
                    (source.group_submitted ?
                         TC_STREAM_GROUP_SUBMITTED_V2 : 0u);
                target.request_generation = source.request_generation;
                target.content_generation = source.content_generation;
                target.expected_bytes = source.expected_bytes;
                target.actual_bytes = source.actual_bytes;
                target.source_generation = source.source_generation;
                for (uint32_t reader = 0;
                     reader < source.reader_count; ++reader) {
                    target.readers[reader].fence =
                        source.readers[reader].fence;
                    target.readers[reader].completed =
                        source.readers[reader].completed ? 1 : 0;
                }
                groups[index] = target;
            }
            for (size_t index = 0;
                 index < receipt->pool_selections.size(); ++index) {
                const auto &source = receipt->pool_selections[index];
                pools[index] = {source.pass, source.ordinal, source.pool};
            }
            for (size_t index = 0;
                 index < receipt->carries.size(); ++index) {
                const auto &source = receipt->carries[index];
                carries[index] = {
                    source.from_pass, source.to_pass, source.group,
                    source.pool, source.slot, source.content_generation};
            }
        }
        *out = value;
        return 1;
    } catch (const std::exception &e) {
        return fail(error, size, e.what());
    }
}
extern "C" void tc_stream_executor_cancel(tc_stream_executor *h){if(h)h->cancel.store(true);}
extern "C" int tc_stream_executor_destroy(tc_stream_executor **h,char *error,size_t size){
    if(!h || !*h)return 1;
    try {owner(*h);if(!(*h)->executor->retry_drain())return fail(error,size,"streaming quarantined: drain incomplete");
        delete *h;*h=nullptr;return 1;}
    catch(const std::exception &e){return fail(error,size,e.what());}
}
