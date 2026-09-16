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
class CAdapter final : public ModelSlotAdapter {
public:
    tc_stream_adapter_v1 ops;
    struct Job { CAdapter *self; tc_stream_group_v1 group; char error[1024]{}; };
    std::vector<Job> jobs;
    char error[1024]{};
    explicit CAdapter(const tc_stream_adapter_v1 &o,uint32_t slots):ops(o),jobs(slots){}
    void checked(int success) {if (!success)throw std::runtime_error(error[0]?error:"streaming adapter failed");}
    static tc_stream_group_v1 view(const Group &g) {
        return {g.id,g.slot,uint32_t(g.blocks.size()),g.blocks.data(),g.bytes};
    }
    void create_pool(const PoolLayout &pool) override {
        for(uint32_t i=0;i<pool.slots.size();++i)
            checked(ops.allocate_slot(ops.user,i,pool.slots[i].capacity_bytes,error,sizeof(error)));
    }
    FillJob make_fill_job(const Group &g,const tc_stream_slot_ticket_v1 &t) override {
        auto &job=jobs.at(t.slot);job.self=this;job.group=view(g);job.error[0]=0;
        return {t,&job,[](void *raw,const tc_stream_slot_ticket_v1 *ticket,
                         const std::atomic<bool> *cancel,uint64_t *bytes){
            auto &j=*static_cast<Job *>(raw);
            auto query=[](const void *v){return static_cast<const std::atomic<bool> *>(v)->load()?1:0;};
            return j.self->ops.fill(j.self->ops.user,ticket,&j.group,query,cancel,bytes,j.error,sizeof(j.error))?0:-1;
        }};
    }
    void encode_prefix(uint32_t pass) override {checked(ops.prefix(ops.user,pass,error,sizeof(error)));}
    void prepare_group(const Group &g,const tc_stream_slot_ticket_v1 &t) override {
        auto v=view(g);checked(ops.prepare(ops.user,&t,&v,error,sizeof(error)));
    }
    ReaderSet encode_group(const Group &g,const tc_stream_slot_ticket_v1 &t,CompletionMailbox &mailbox) override {
        tc_stream_completion_sink_v1 sink{&mailbox,[](void *m,const tc_stream_slot_ticket_v1 *ticket,
                                                    tc_stream_reader_fence_v1 fence,int32_t status){
            if(!ticket)return 0;
            tc_stream_completion_v1 event{};event.struct_size=sizeof(event);event.version=TC_STREAM_SLOT_ABI_V1;
            event.kind=TC_STREAM_READER_COMPLETE;event.status=status;event.ticket=*ticket;event.fence=fence;
            return static_cast<CompletionMailbox *>(m)->post(event)?1:0;
        }};
        auto v=view(g);tc_stream_reader_set_v1 output{};
        checked(ops.encode(ops.user,&t,&v,&sink,&output,error,sizeof(error)));
        if(output.count>TC_STREAM_MAX_READER_QUEUES)throw std::runtime_error("streaming reader count overflow");
        ReaderSet result;result.count=output.count;
        std::copy_n(output.fences,output.count,result.fences.begin());return result;
    }
    bool drain() noexcept override {
        char detail[1024]{};
        try {return ops.drain(ops.user,detail,sizeof(detail))!=0;}catch(...){return false;}
    }
    void destroy_pool() noexcept override {ops.destroy_pool(ops.user);}
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
        *out={c.pool_creates,c.slot_bundles,c.fills,c.bytes_loaded,c.groups_submitted};return 1;}
    catch(const std::exception &e){return fail(error,size,e.what());}
}
extern "C" void tc_stream_executor_cancel(tc_stream_executor *h){if(h)h->cancel.store(true);}
extern "C" int tc_stream_executor_destroy(tc_stream_executor **h,char *error,size_t size){
    if(!h || !*h)return 1;
    try {owner(*h);if(!(*h)->executor->retry_drain())return fail(error,size,"streaming quarantined: drain incomplete");
        delete *h;*h=nullptr;return 1;}
    catch(const std::exception &e){return fail(error,size,e.what());}
}
