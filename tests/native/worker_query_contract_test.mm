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
int tc_engine_resolve_streaming_json(tc_engine *,const char *request,char **result,char **error) {
    if(mode=="resolve_error"){*error=strdup("unvalidated_workload");return 1;}
    NSDictionary *input=[NSJSONSerialization JSONObjectWithData:[@(request) dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
    auto resolution=@{@"status":@"resolved",@"requested_selector":input[@"execution"][@"streaming"],
        @"resolution_digest":@"resolution",@"selection":@{@"record_digest":@"record",@"layout_digest":@"layout",
        @"execution_container":mode=="wrong_container"?@"embedded_app":@"cli_worker"}};
    NSData *data=tc_worker::canonical_request(resolution);
    *result=strdup([[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] UTF8String]);return 0;
}
}
int main(int argc,char **argv) { @autoreleasepool {
    if(argc!=3)return 3;mode=argv[2];return tc_worker::query(argv[1]);
} }
