// Diagnostic subprocess only. Forward original selectors exactly once.
// Public Metal completion timestamps; Private ANE API host spans, NOT ANE
// hardware execution/placement. No modification to installed libraries/apps.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TC_PROFILE_SOURCE_SHA
#define TC_PROFILE_SOURCE_SHA "unbound"
#endif
namespace {
// A four-request joint FFN+QKV sequence matrix exceeds100k CB/API spans.
// Still bounded; dropped events remain a hard validation failure.
constexpr size_t capacity=200000;
struct Row {const char *name;uint64_t begin,end,thread,model=0,queue=0,gpu_begin=0,gpu_end=0;int request;bool ok=true;};
struct Hook {Class cls;SEL sel;IMP previous;};
std::mutex mutex;std::condition_variable changed;
std::vector<Row> rows;std::vector<Hook> hooks;std::map<void*,uint64_t> models,queue_ids;
NSMutableArray *retained_queues;
uint64_t next_model=0,next_queue=0,pending=0,dropped=0;
std::atomic<int> scope{-1};bool installed=false,enabled=false;
using ModelFn=BOOL(*)(id,SEL,id,NSDictionary*,unsigned,NSError**);
using EvalFn=BOOL(*)(id,SEL,id,NSDictionary*,id,unsigned,NSError**);
using CommitFn=void(*)(id,SEL);
ModelFn original_compile=nullptr,original_load=nullptr,original_unload=nullptr;
EvalFn original_evaluate=nullptr;CommitFn original_commit=nullptr;
uint64_t now(){return uint64_t(CACurrentMediaTime()*1e9);}
uint64_t thread(){uint64_t value=0;pthread_threadid_np(nullptr,&value);return value;}
uint64_t model_id(id model){std::lock_guard lock(mutex);auto key=(__bridge void*)model;auto [it,inserted]=models.emplace(key,0);if(inserted)it->second=++next_model;return it->second;}
void record(Row row){std::lock_guard lock(mutex);if(rows.size()<capacity)rows.push_back(row);else ++dropped;}
BOOL compile_call(id self,SEL sel,id model,NSDictionary *options,unsigned qos,NSError **error){
    Row row{"ane.compile_api",now(),0,thread(),model_id(model),0,0,0,scope.load()};
    const BOOL ok=original_compile(self,sel,model,options,qos,error);row.end=now();row.ok=ok;record(row);return ok;}
BOOL load_call(id self,SEL sel,id model,NSDictionary *options,unsigned qos,NSError **error){
    Row row{"ane.load_api",now(),0,thread(),model_id(model),0,0,0,scope.load()};
    const BOOL ok=original_load(self,sel,model,options,qos,error);row.end=now();row.ok=ok;record(row);return ok;}
BOOL unload_call(id self,SEL sel,id model,NSDictionary *options,unsigned qos,NSError **error){
    Row row{"ane.unload_api",now(),0,thread(),model_id(model),0,0,0,scope.load()};
    const BOOL ok=original_unload(self,sel,model,options,qos,error);row.end=now();row.ok=ok;record(row);
    {std::lock_guard lock(mutex);models.erase((__bridge void*)model);}return ok;}
BOOL evaluate_call(id self,SEL sel,id model,NSDictionary *options,id request,unsigned qos,NSError **error){
    Row row{"ane.evaluate_api",now(),0,thread(),model_id(model),0,0,0,scope.load()};
    const BOOL ok=original_evaluate(self,sel,model,options,request,qos,error);row.end=now();row.ok=ok;record(row);return ok;}
void commit_call(id self,SEL sel){
    id<MTLCommandBuffer> buffer=self;uint64_t queue;
    {std::lock_guard lock(mutex);auto key=(__bridge void*)buffer.commandQueue;auto [it,inserted]=queue_ids.emplace(key,0);
        if(inserted){it->second=++next_queue;[retained_queues addObject:buffer.commandQueue];}queue=it->second;++pending;}
    const Row row{"metal.command_buffer",now(),0,thread(),0,queue,0,0,scope.load()};
    [buffer addCompletedHandler:^(id<MTLCommandBuffer> done){
        Row complete=row;complete.end=now();complete.ok=done.status==MTLCommandBufferStatusCompleted;
        if(done.GPUStartTime>0&&done.GPUEndTime>=done.GPUStartTime){complete.gpu_begin=uint64_t(done.GPUStartTime*1e9);complete.gpu_end=uint64_t(done.GPUEndTime*1e9);}
        {std::lock_guard lock(mutex);if(rows.size()<capacity)rows.push_back(complete);else ++dropped;--pending;}changed.notify_all();
    }];
    original_commit(self,sel);
}
IMP hook(Class cls,const char *name,IMP implementation,unsigned arguments){
    SEL sel=sel_registerName(name);Method method=class_getInstanceMethod(cls,sel);
    if(!method||method_getNumberOfArguments(method)!=arguments)throw std::runtime_error("observer selector ABI unavailable");
    IMP previous=method_getImplementation(method);const char *encoding=method_getTypeEncoding(method);
    if(!class_addMethod(cls,sel,implementation,encoding))method_setImplementation(class_getInstanceMethod(cls,sel),implementation);
    hooks.push_back({cls,sel,previous});return previous;
}
void restore(){for(auto it=hooks.rbegin();it!=hooks.rend();++it)method_setImplementation(class_getInstanceMethod(it->cls,it->sel),it->previous);hooks.clear();}
char *encode(id value){NSData *data=[NSJSONSerialization dataWithJSONObject:value options:0 error:nil];if(!data)return nullptr;
    char *result=static_cast<char*>(std::malloc(data.length+1));if(!result)return nullptr;std::memcpy(result,data.bytes,data.length);result[data.length]=0;return result;}
NSMutableDictionary *snapshot(bool include_rows){
    std::lock_guard lock(mutex);NSMutableArray *events=[NSMutableArray array];NSMutableDictionary *counts=[NSMutableDictionary dictionary];
    for(const auto &row:rows){NSString *name=@(row.name);counts[name]=@([counts[name] unsignedLongLongValue]+1);
        if(include_rows)[events addObject:@{@"name":name,@"begin_ns":@(row.begin),@"end_ns":@(row.end),@"thread":@(row.thread),@"request":@(row.request),
            @"model":@(row.model),@"queue":@(row.queue),@"gpu_begin_ns":@(row.gpu_begin),@"gpu_end_ns":@(row.gpu_end),@"ok":@(row.ok)}];}
    return [@{@"schema":@"tc-qwen-base-observer-v1",@"enabled":@(enabled),@"source_sha256":@TC_PROFILE_SOURCE_SHA,
        @"pending":@(pending),@"dropped":@(dropped),@"counts":counts,@"events":events,@"queues_seen":@(next_queue),@"models_seen":@(next_model),
        @"scope":@"same-process API forwarding/Metal CB completion; not kernel counters, all Metal classes, ANE hardware timing or zero overhead"} mutableCopy];
}
}
extern "C" uint64_t tc_qwen_profile_now_ns(){return now();}
extern "C" void tc_qwen_profile_scope(int value){scope.store(value);}
extern "C" void tc_qwen_profile_string_free(char *value){std::free(value);}
extern "C" int tc_qwen_profile_install(int active){@autoreleasepool{
    if(installed)return -1;
    try{
        rows.reserve(capacity);retained_queues=[NSMutableArray array];enabled=active;
        if(active){
            static void *framework=dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW|RTLD_LOCAL);
            Class ane=framework?NSClassFromString(@"_ANEClient"):Nil;if(!ane)throw std::runtime_error("private ANE runtime unavailable");
            original_compile=reinterpret_cast<ModelFn>(hook(ane,"compileModel:options:qos:error:",reinterpret_cast<IMP>(compile_call),6));
            original_load=reinterpret_cast<ModelFn>(hook(ane,"loadModel:options:qos:error:",reinterpret_cast<IMP>(load_call),6));
            original_unload=reinterpret_cast<ModelFn>(hook(ane,"unloadModel:options:qos:error:",reinterpret_cast<IMP>(unload_call),6));
            original_evaluate=reinterpret_cast<EvalFn>(hook(ane,"evaluateWithModel:options:request:qos:error:",reinterpret_cast<IMP>(evaluate_call),7));
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> buffer=[queue commandBuffer];
            if(!buffer)throw std::runtime_error("Metal class discovery unavailable");
            original_commit=reinterpret_cast<CommitFn>(hook(object_getClass(buffer),"commit",reinterpret_cast<IMP>(commit_call),2));
        }
        installed=true;return 0;
    }catch(...){restore();enabled=false;return -1;}
}}
extern "C" int tc_qwen_profile_wait(unsigned timeout_ms){
    std::unique_lock lock(mutex);return changed.wait_for(lock,std::chrono::milliseconds(timeout_ms),[]{return pending==0;})?0:-1;
}
extern "C" char *tc_qwen_profile_counters(){@autoreleasepool{return encode(snapshot(false));}}
extern "C" char *tc_qwen_profile_finish(){@autoreleasepool{
    const bool drained=tc_qwen_profile_wait(10000)==0;restore();auto result=snapshot(true);result[@"drained"]=@(drained);
    // Restored IMPs, but a subclass method slot added during discovery remains
    // until subprocess exit. The helper stays loaded through all callbacks.
    result[@"restored_imps"]=@YES;retained_queues=nil;return encode(result);
}}
extern "C" int tc_qwen_profile_self_test(){@autoreleasepool{
    if(!installed||!enabled)return -1;scope.store(-2);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> a=[device newBufferWithLength:64 options:MTLResourceStorageModeShared],b=[device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if(!a||!b)return -1;std::memset(a.contents,0x5a,64);std::memset(b.contents,0,64);
    id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
    [blit copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:64];[blit endEncoding];[cb commit];[cb waitUntilCompleted];
    if(tc_qwen_profile_wait(1000)||std::memcmp(a.contents,b.contents,64))return -1;
    std::lock_guard lock(mutex);
    for(const auto &row:rows)if(row.request==-2&&row.name==std::string("metal.command_buffer")){
        return row.ok&&row.gpu_begin>=row.begin&&row.gpu_end>=row.gpu_begin&&row.gpu_end<=row.end?0:-1;}
    return -1;
}}
