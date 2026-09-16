#include "stream_slot_c.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct { uint64_t slots[3]; unsigned allocations, destroys, prefix; } model;
static int allocate(void *user,uint32_t slot,uint64_t capacity,char *e,size_t n){
    (void)e;(void)n;assert(slot<3 && capacity==8);++((model *)user)->allocations;return 1;
}
static void destroy(void *user){++((model *)user)->destroys;}
static int fill(void *user,const tc_stream_slot_ticket_v1 *t,const tc_stream_group_v1 *g,
                tc_stream_cancel_query_v1 cancelled,const void *cu,uint64_t *bytes,char *e,size_t n){
    (void)e;(void)n;if(cancelled(cu))return 0;
    ((model *)user)->slots[t->slot]=g->blocks[0]+t->item.pass*100;*bytes=8;return 1;
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
    tc_stream_stage_plan_v1 plan={sizeof(plan),TC_STREAM_SLOT_ABI_V1,1,0,3,2,1,2,1,capacity,4,groups};
    tc_stream_adapter_v1 ops={sizeof(ops),TC_STREAM_SLOT_ABI_V1,&m,allocate,destroy,fill,prefix,prepare,encode,drain};
    tc_stream_executor *executor=NULL;
    assert(tc_stream_executor_create_v1(&plan,&ops,&executor,error,sizeof(error)));
    assert(m.allocations==3 && m.destroys==0);
    assert(tc_stream_executor_run_pass(executor,0,10,error,sizeof(error)));
    assert(m.allocations==3 && m.destroys==0);
    assert(tc_stream_executor_run_pass(executor,1,11,error,sizeof(error)));
    tc_stream_counters_v1 counters;
    assert(tc_stream_executor_counters(executor,&counters,error,sizeof(error)));
    assert(counters.fills==8 && counters.slot_bundles==3);
    assert(tc_stream_executor_finish(executor,error,sizeof(error)));
    assert(m.destroys==1);
    assert(tc_stream_executor_destroy(&executor,error,sizeof(error)) && !executor);
    assert(m.destroys==1);
    puts("PASS: C11 persistent slot bridge, separate pass/step, exact pool retention and release");
}
