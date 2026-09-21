// Isolated envelope/control-flow oracle. Does not represent native model execution.
#include "../../apps/cli/query_worker.hpp"
#include <cstring>
struct tc_engine { std::atomic<bool> stop{false}; };
static std::string mode;
namespace tc { const char *runtime_build_identity() noexcept {return "test-runtime";} }
extern "C" {
void tc_string_free(char *value) {std::free(value);}
void tc_engine_free(tc_engine *engine) {delete engine;}
void tc_engine_cancel(tc_engine *engine) {engine->stop=true;}
int tc_engine_create_model_worker(const char *,const char *,tc_engine **engine,char **error) {
    if(mode=="create_error"){*error=strdup("creation rejected");return 1;}
    *engine=new tc_engine;return 0;
}
int tc_engine_verify_streaming_sources_json(tc_engine *engine,char **result,char **error) {
    if(mode=="cancel") {
        std::raise(SIGTERM);
        for(int i=0;i<200 && !engine->stop;++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if(!engine->stop)std::abort();
        *error=strdup("source verification cancelled");return 2;
    }
    if(mode=="source_error"){*error=strdup("source rejected");return 1;}
    *result=strdup("{\"status\":\"verified\"}");return 0;
}
static NSDictionary *resolution_for(NSDictionary *input) {
    NSDictionary *selector=input[@"execution"][@"streaming"];
    return @{@"schema_version":@1,@"status":@"resolved",@"requested_selector":selector,
        @"request_digest":@"workload",@"resolution_digest":@"resolution",@"catalog_revision":@"catalog",
        @"exact_selector":@{@"schema_version":@2,@"enabled":@YES,@"selection":@"preset",@"retention":@"request",
            @"target_request_memory_bytes":selector[@"target_request_memory_bytes"],@"preset_id":@"preset",@"preset_revision":@1,
            @"catalog_revision":@"catalog",@"expected_resolution_digest":@"resolution"},
        @"identity":@{@"source_digest":@"source",@"runtime_digest":@"runtime",@"device_digest":@"device"},
        @"selection":@{@"release_channel":@"public",@"record_digest":@"record",@"layout_digest":@"layout",@"preset_id":@"preset",@"preset_revision":@1,
            @"target_request_memory_bytes":selector[@"target_request_memory_bytes"],@"calibrated_request_bytes":@1024,
            @"component_policy_revision":@"policy",@"memory_scope":@"request",@"execution_container":mode=="wrong_container"?@"embedded_app":@"cli_worker"}};
}
int tc_engine_resolve_streaming_json(tc_engine *,const char *request,char **result,char **error) {
    if(mode=="resolve_error"){*error=strdup("unvalidated_workload");return 1;}
    NSDictionary *input=[NSJSONSerialization JSONObjectWithData:[@(request) dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
    NSMutableDictionary *resolution=[resolution_for(input) mutableCopy];
    if(mode=="bad_exact") {
        NSMutableDictionary *exact=[resolution[@"exact_selector"] mutableCopy];exact[@"expected_resolution_digest"]=@"wrong";resolution[@"exact_selector"]=exact;
    }
    NSData *data=tc_worker::canonical_request(resolution);
    *result=strdup([[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] UTF8String]);return 0;
}
int tc_engine_generate(tc_engine *engine,const char *request,tc_event_callback,void *,char **result,char **error) {
    if(mode=="existing" || mode=="bad_exact")std::abort(); // Must reject before generation.
    NSDictionary *input=[NSJSONSerialization JSONObjectWithData:[@(request) dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
    NSDictionary *selector=input[@"execution"][@"streaming"],*output=input[@"outputs"][0];
    if(![selector[@"selection"] isEqual:@"preset"] || ![selector[@"expected_resolution_digest"] isEqual:@"resolution"])std::abort();
    NSString *path=output[@"path"];
    if(mode!="missing_artifact") {
        if(mode=="symlink_artifact") {if(symlink("/dev/null",path.fileSystemRepresentation))std::abort();}
        else if(![[@"stub artifact\n" dataUsingEncoding:NSUTF8StringEncoding] writeToFile:path atomically:NO])std::abort();
    }
    if(mode=="generate_error"){*error=strdup("generation failed after partial output");return 1;}
    if(mode=="generate_cancel") {
        std::raise(SIGTERM);
        for(int i=0;i<200 && !engine->stop;++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if(!engine->stop)std::abort();*error=strdup("cancelled after partial output");return 2;
    }
    NSDictionary *resolution=resolution_for(input);
    NSMutableDictionary *summary=[resolution[@"selection"] mutableCopy];
    [summary removeObjectForKey:@"layout_digest"];
    [summary addEntriesFromDictionary:resolution[@"identity"]];
    [summary addEntriesFromDictionary:@{@"schema_version":@1,@"actual_plan_verified":@YES,@"catalog_revision":@"catalog",
        @"resolution_digest":@"resolution",@"workload_digest":@"workload",@"authorized_layout_digest":@"layout",@"actual_layout_digest":@"layout",
        @"receipt_schema_version":@3,@"receipt_source_generation":@1,@"receipt_digest":@"receipt",@"receipt_verifier_revision":@"verifier"}];
    NSMutableDictionary *value=[@{@"schema_version":@1,@"warmup":@NO,@"model":input[@"model"],@"operation":input[@"operation"],
        @"output":path,@"width":output[@"width"],@"height":output[@"height"],@"seed":input[@"sampling"][@"seed"],@"steps":input[@"sampling"][@"steps"],@"public_streaming":summary} mutableCopy];
    if(mode=="empty_result")value=[NSMutableDictionary dictionary];
    if(mode=="unverified")summary[@"actual_plan_verified"]=@NO;
    if(mode=="wrong_layout")summary[@"actual_layout_digest"]=@"wrong";
    if(mode=="wrong_record")summary[@"record_digest"]=@"wrong";
    if(mode=="wrong_source")summary[@"source_digest"]=@"wrong";
    if(mode=="wrong_output")value[@"output"]=@"/tmp/wrong.png";
    if(mode=="wrong_dimensions")value[@"width"]=@17;
    if(mode=="missing_receipt")[summary removeObjectForKey:@"receipt_digest"];
    NSData *data=tc_worker::canonical_request(value);
    *result=strdup([[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] UTF8String]);return 0;
}
}
int main(int argc,char **argv) { @autoreleasepool {
    if(argc!=3 && argc!=4)return 3;mode=argv[2];return argc==4?tc_worker::generate(argv[1]):tc_worker::query(argv[1]);
} }
