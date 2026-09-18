#include "stream_slot_c.h"
#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    uint64_t slots[3];
    unsigned allocations, destroys, prefix;
    uint32_t active_pool;
    _Atomic unsigned fills, encodes;
} model;
static int allocate(void *user,uint32_t slot,uint64_t capacity,char *e,size_t n){
    (void)e;(void)n;assert(slot<3 && capacity==8);++((model *)user)->allocations;return 1;
}
static int allocate_v2(void *user,uint32_t pool,uint32_t slot,uint64_t capacity,char *e,size_t n){
    model *m=(model *)user;
    assert(pool==3 || pool==9);
    if(slot==0)m->active_pool=pool;else assert(m->active_pool==pool);
    return allocate(user,slot,capacity,e,n);
}
static void destroy(void *user){++((model *)user)->destroys;}
static void destroy_v2(void *user,uint32_t pool){assert(((model *)user)->active_pool==pool);destroy(user);}
static int fill(void *user,const tc_stream_slot_ticket_v1 *t,const tc_stream_group_v1 *g,
                tc_stream_cancel_query_v1 cancelled,const void *cu,uint64_t *bytes,char *e,size_t n){
    (void)e;(void)n;if(cancelled(cu))return 0;
    ((model *)user)->slots[t->slot]=g->blocks[0]+t->item.pass*100;
    atomic_fetch_add(&((model *)user)->fills,1);*bytes=8;return 1;
}
static int prefix(void *user,uint32_t pass,char *e,size_t n){
    (void)e;(void)n;assert(((model *)user)->prefix++==pass);return 1;
}
static int prepare(void *user,const tc_stream_slot_ticket_v1 *t,const tc_stream_group_v1 *g,char *e,size_t n){
    (void)e;(void)n;assert(((model *)user)->slots[t->slot]==g->blocks[0]+t->item.pass*100);return 1;
}
static int encode(void *user,const tc_stream_slot_ticket_v1 *t,const tc_stream_group_v1 *g,
                  const tc_stream_completion_sink_v1 *sink,tc_stream_reader_set_v1 *readers,char *e,size_t n){
    assert(prepare(user,t,g,e,n));readers->count=1;
    atomic_fetch_add(&((model *)user)->encodes,1);
    readers->fences[0]=(tc_stream_reader_fence_v1){1,1+t->item.pass*100+t->item.group};
    /* Synchronous test backend: callback is delivered before seal_readers. */
    assert(sink->post(sink->user,t,readers->fences[0],0));return 1;
}
static int drain(void *user,char *e,size_t n){(void)user;(void)e;(void)n;return 1;}
int main(void){
    model m={0};char error[1024]={0};
    uint32_t blocks[]={2,3,4,5};uint64_t capacity[]={8,8,8};
    tc_stream_group_v1 groups[4];
    for(uint32_t i=0;i<4;++i)groups[i]=(tc_stream_group_v1){i,i%3,1,&blocks[i],8};
    tc_stream_stage_plan_v1 plan={sizeof(plan),TC_STREAM_SLOT_ABI_V1,1,7,3,2,1,2,1,capacity,4,groups};
    tc_stream_adapter_v1 ops={sizeof(ops),TC_STREAM_SLOT_ABI_V1,&m,allocate,destroy,fill,prefix,prepare,encode,drain};
    tc_stream_executor *executor=NULL;
    assert(tc_stream_executor_create_v1(&plan,&ops,&executor,error,sizeof(error)));
    tc_stream_receipt_v1 receipt={0};
    receipt.struct_size=sizeof(receipt);
    receipt.version=TC_STREAM_RECEIPT_ABI_V1;
    assert(!tc_stream_executor_receipt_v1(
        executor,&receipt,error,sizeof(error)));
    assert(strstr(error,"streaming receipt unavailable"));
    memset(error,0,sizeof(error));
    tc_stream_receipt_config_v1 receipt_config={0};
    receipt_config.struct_size=sizeof(receipt_config);
    receipt_config.version=TC_STREAM_RECEIPT_ABI_V1;
    receipt_config.source_generation=41;
    memset(receipt_config.layout_digest,'a',64);
    strcpy(receipt_config.implementation,"c_bridge_generic_v2");
    assert(tc_stream_executor_enable_receipt_v1(
        executor,&receipt_config,error,sizeof(error)));
    assert(!tc_stream_executor_enable_receipt_v1(
        executor,&receipt_config,error,sizeof(error)));
    assert(strstr(error,"invalid receipt lifecycle"));
    memset(error,0,sizeof(error));
    assert(m.allocations==3 && m.destroys==0);
    assert(tc_stream_executor_run_pass(executor,0,10,error,sizeof(error)));
    assert(m.allocations==3 && m.destroys==0);
    assert(tc_stream_executor_run_pass(executor,1,11,error,sizeof(error)));
    tc_stream_counters_v1 counters;
    assert(tc_stream_executor_counters(executor,&counters,error,sizeof(error)));
    assert(counters.fills==8 && counters.slot_bundles==3);
    assert(tc_stream_executor_finish(executor,error,sizeof(error)));
    receipt.struct_size=sizeof(receipt);
    receipt.version=TC_STREAM_RECEIPT_ABI_V1;
    assert(tc_stream_executor_receipt_v1(
        executor,&receipt,error,sizeof(error)));
    assert(receipt.stage_index==1 && receipt.completed_passes==2 &&
           receipt.completed_groups==8 && receipt.fills==8 &&
           receipt.groups_submitted==8 && receipt.logical_read_bytes==64 &&
           receipt.reader_fences_issued==8 &&
           receipt.reader_fences_completed==8 &&
           receipt.source_generation==41 && receipt.drained &&
           receipt.verified && strlen(receipt.event_digest)==64 &&
           strlen(receipt.canonical_digest)==64);
    assert(m.destroys==1);
    assert(tc_stream_executor_destroy(&executor,error,sizeof(error)) && !executor);
    assert(m.destroys==1);
    memset(&m, 0, sizeof(m)); memset(error, 0, sizeof(error)); executor=NULL;
    uint32_t pool_blocks[]={10,11,20,21};
    tc_stream_group_v2 multi_groups[4];
    for (uint32_t i=0; i<4; ++i) {
        const uint32_t pool = i < 2 ? 3u : 9u;
        multi_groups[i]=(tc_stream_group_v2){i,pool,i%2,1,&pool_blocks[i],8};
    }
    tc_stream_pool_plan_v2 pools[2]={
        {sizeof(tc_stream_pool_plan_v2),TC_STREAM_SLOT_ABI_V2,3,2,capacity},
        {sizeof(tc_stream_pool_plan_v2),TC_STREAM_SLOT_ABI_V2,9,2,capacity}};
    tc_stream_stage_plan_v2 plan_v2={sizeof(plan_v2),TC_STREAM_SLOT_ABI_V2,1,2,2,1,1,2,1,
                                     pools,4,multi_groups};
    tc_stream_adapter_v2 ops_v2={sizeof(ops_v2),TC_STREAM_SLOT_ABI_V2,&m,
                                 allocate_v2,destroy_v2,fill,prefix,prepare,encode,drain};
    assert(tc_stream_executor_create_v2(&plan_v2,&ops_v2,&executor,error,sizeof(error)));
    assert(m.allocations==2 && m.destroys==0);
    assert(tc_stream_executor_run_pass(executor,0,30,error,sizeof(error)));
    assert(m.allocations==4 && m.destroys==1);
    assert(tc_stream_executor_run_pass(executor,1,31,error,sizeof(error)));
    assert(tc_stream_executor_finish(executor,error,sizeof(error)));
    tc_stream_counters_v1 v2_counters;
    assert(tc_stream_executor_counters(executor,&v2_counters,error,sizeof(error)));
    assert(v2_counters.fills==8 && v2_counters.groups_submitted==8 &&
           v2_counters.pool_creates==4 && v2_counters.slot_bundles==8);
    assert(tc_stream_executor_destroy(&executor,error,sizeof(error)) && !executor);
    assert(m.allocations==8 && m.destroys==4);

    model carry={0}; memset(error,0,sizeof(error)); executor=NULL;
    uint32_t carry_blocks[]={30,31,32,33};
    tc_stream_group_v1 carry_groups[4];
    for(uint32_t i=0;i<4;++i)
        carry_groups[i]=(tc_stream_group_v1){i,i%2,1,&carry_blocks[i],8};
    tc_stream_stage_plan_v3 plan_v3={
        sizeof(plan_v3),TC_STREAM_SLOT_ABI_V3,1,7,2,1,1,3,
        TC_STREAM_PASS_CARRY_FIRST_GROUP_V3,9,capacity,4,carry_groups};
    tc_stream_adapter_v1 carry_ops={sizeof(carry_ops),TC_STREAM_SLOT_ABI_V1,
        &carry,allocate,destroy,fill,prefix,prepare,encode,drain};
    assert(tc_stream_executor_create_v3(
        &plan_v3,&carry_ops,&executor,error,sizeof(error)));
    assert(tc_stream_executor_run_pass(executor,0,0,error,sizeof(error)));
    assert(atomic_load(&carry.fills)==5 && atomic_load(&carry.encodes)==4);
    assert(tc_stream_executor_run_pass(executor,1,1,error,sizeof(error)));
    assert(atomic_load(&carry.fills)==9 && atomic_load(&carry.encodes)==8);
    assert(tc_stream_executor_run_pass(executor,2,2,error,sizeof(error)));
    assert(tc_stream_executor_finish(executor,error,sizeof(error)));
    tc_stream_counters_v1 v3_counters;
    assert(tc_stream_executor_counters(executor,&v3_counters,error,sizeof(error)));
    assert(v3_counters.fills==12 && v3_counters.groups_submitted==12 &&
           v3_counters.pool_creates==1 && v3_counters.slot_bundles==2);
    assert(atomic_load(&carry.fills)==12 && atomic_load(&carry.encodes)==12 &&
           carry.prefix==3);
    assert(tc_stream_executor_destroy(&executor,error,sizeof(error)) && !executor);
    assert(carry.allocations==2 && carry.destroys==1);

    puts("PASS: C11 persistent slot bridge, v3 cross-pass carry, separate pass/step, exact pool retention and release");
}
