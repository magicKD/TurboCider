// CPU-only wrapper/control-flow oracle. Stubs cannot load a model or use ANE/GPU.
#include "../../apps/cli/runtime_worker_protocol.hpp"
#include "../../apps/cli/startup_gate.hpp"
#include <cstring>
struct tc_engine {std::atomic<bool> stop{false};};
static std::string mode;
static NSString *png;
static bool planned=false,hardware_checked=false;
namespace tc {const char *runtime_build_identity() noexcept {return "test-runtime";}}
extern "C" {
void tc_string_free(char *value){std::free(value);}
void tc_engine_free(tc_engine *engine){delete engine;}
void tc_engine_cancel(tc_engine *engine){engine->stop=true;}
char *tc_system_json() {
    hardware_checked=true;
    NSDictionary *system=@{@"gpu":mode=="bad_hardware"?@"Apple M4 Max":@"Apple M4 Pro",
        @"physical_memory_bytes":mode=="bad_memory"?@(uint64_t(24)<<30):mode=="boolean_memory"?@YES:@(uint64_t(48)<<30)};
    return strdup([[[NSString alloc] initWithData:tc_worker::canonical_request(system) encoding:NSUTF8StringEncoding] UTF8String]);
}
int tc_plan_json(const char *,char **result,char **error) {
    if(!hardware_checked || mode=="rejected" || mode=="bad_hardware" || mode=="bad_memory" || mode=="boolean_memory")std::abort();
    planned=true;
    if(mode=="plan_error"){*error=strdup("normal native gate rejected request");return 1;}
    *result=strdup("{\"executable\":true}");return 0;
}
int tc_engine_create_model_worker(const char *,const char *,tc_engine **engine,char **error) {
    if(!planned || mode=="plan_error" || mode=="rejected")std::abort();
    if(mode=="create_error"){*error=strdup("creation rejected");return 1;}
    *engine=new tc_engine;return 0;
}
int tc_engine_generate(tc_engine *engine,const char *request,tc_event_callback callback,void *context,char **result,char **error) {
    NSDictionary *input=[NSJSONSerialization JSONObjectWithData:[@(request) dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
    if(input[@"execution"][@"streaming"] || ![input[@"execution"][@"hybrid_mlp_mode"] isEqual:@"runtime"])std::abort();
    callback("{\"sequence\":1,\"phase\":\"route_gpu\",\"completed\":0,\"total\":1,\"elapsed_seconds\":0}",context);
    for(int i=0;i<10000;++i)callback("{\"sequence\":2,\"phase\":\"denoise\",\"completed\":1,\"total\":8,\"elapsed_seconds\":1}",context);
    NSDictionary *output=input[@"outputs"][0];NSString *path=output[@"path"];
    if(mode!="missing_artifact") {
        if(mode=="symlink_artifact"){if(symlink(png.fileSystemRepresentation,path.fileSystemRepresentation))std::abort();}
        else if(![[NSData dataWithContentsOfFile:png] writeToFile:path atomically:NO])std::abort();
    }
    if(mode=="generate_error"){*error=strdup("failed after partial output");return 1;}
    if(mode=="cancel") {
        std::raise(SIGTERM);
        for(int i=0;i<200 && !engine->stop;++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if(!engine->stop)std::abort();*error=strdup("cancelled after partial output");return 2;
    }
    const bool gpu=mode=="gpu_fallback";
    const bool failed=gpu || mode=="partial_fallback" || mode=="all_fallback";
    id backend=gpu?NSNull.null:@"private_ane";
    NSString *data=gpu?@"":@"w8a8_hadamard",*axis=gpu?@"rows":@"intermediate_channels";
    const int width=[input[@"model"] isEqual:@"qwen-image-2.1"]?12288:10240;
    NSMutableDictionary *metrics=[@{@"executor_backend":backend,@"data_path":data,@"partition_axis":axis,
        @"ane_channels":gpu?@0:@4096,@"gpu_channels":gpu?@0:@(width-4096),
        @"hybrid_blocks_session_total":(gpu || mode=="all_fallback")?@0:@10,@"gpu_blocks_session_total":@2,
        @"fallback_blocks_session_total":(mode=="partial_fallback" || mode=="all_fallback")?@1:@0,
        @"backend_fallback_reason":gpu?@"private capability unavailable":@"",
        @"failure_reason":failed?@"runtime capability failed":@""} mutableCopy];
    NSMutableDictionary *contract=[@{@"executor_backend":backend,@"data_path":data,@"partition_axis":axis} mutableCopy];
    NSMutableDictionary *plan=[@{@"runtime_weight_contract":contract,@"execution":gpu?@"gpu":@"gpu_ane_experimental"} mutableCopy];
    // Match results.mm: failure_reason belongs to runtime_weight, while
    // runtime_failed is exported at hybrid's top level.
    NSMutableDictionary *hybrid=[@{@"runtime_weight":metrics,@"runtime_failed":@(failed)} mutableCopy];
    NSMutableDictionary *value=[@{@"schema_version":@1,@"warmup":@NO,@"model":input[@"model"],@"operation":input[@"operation"],
        @"output":path,@"width":output[@"width"],@"height":output[@"height"],@"seed":input[@"sampling"][@"seed"],@"steps":input[@"sampling"][@"steps"],
        @"plan":plan,@"hybrid":hybrid} mutableCopy];
    if(mode=="wrong_output")value[@"output"]=@"/tmp/wrong.png";
    if(mode=="wrong_dimensions")value[@"width"]=@17;
    if(mode=="wrong_steps")value[@"steps"]=@9;
    if(mode=="wrong_contract")contract[@"data_path"]=@"fp16";
    if(mode=="wrong_channels")metrics[@"ane_channels"]=@2048;
    if(mode=="public_backend"){metrics[@"executor_backend"]=@"public_coreml";contract[@"executor_backend"]=@"public_coreml";}
    if(mode=="wrong_axis"){metrics[@"partition_axis"]=@"rows";contract[@"partition_axis"]=@"rows";}
    if(mode=="boolean_count")metrics[@"hybrid_blocks_session_total"]=@YES;
    if(mode=="negative_count")metrics[@"gpu_blocks_session_total"]=@(-1);
    if(mode=="missing_metrics")[hybrid removeObjectForKey:@"runtime_weight"];
    if(mode=="unexpected_streaming")value[@"public_streaming"]=@{};
    if(mode=="wrong_execution")plan[@"execution"]=@"gpu";
    if(mode=="reason_missing") {
        metrics[@"hybrid_blocks_session_total"]=@0;metrics[@"fallback_blocks_session_total"]=@1;
        hybrid[@"runtime_failed"]=@YES;metrics[@"failure_reason"]=@"";
    }
    if(mode=="nested_reason_missing" || mode=="top_only_reason")[metrics removeObjectForKey:@"failure_reason"];
    if(mode=="nested_reason_bool")metrics[@"failure_reason"]=@YES;
    if(mode=="top_only_reason")hybrid[@"failure_reason"]=@"invented top-level reason must not grant authority";
    *result=strdup([[[NSString alloc] initWithData:tc_worker::canonical_request(value) encoding:NSUTF8StringEncoding] UTF8String]);return 0;
}
}
int main(int argc,char **argv){@autoreleasepool{
    if(argc!=4 && argc!=5)return 3;mode=argv[2];png=@(argv[3]);
    if(argc==5 && !tc_worker::await_admission()){std::cerr<<"worker_start_not_admitted\n";return 1;}
    return tc_worker::runtime::generate(argv[1]);
}}
