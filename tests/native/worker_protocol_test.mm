#include "../../apps/cli/worker_protocol.hpp"
#include <iostream>
using namespace tc_worker;
template<class F> void rejects(F work) {
    bool rejected=false;try{work();}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"expected worker protocol rejection");
}
int main(int argc, char **argv) { try { @autoreleasepool {
    if(argc==2) {
        auto value=read_input(argv[1]);
        require([value[@"request_digest"] isEqual:@"802fd8d904d4e5421f73382626dc03c5bb20b433c16aa82c9816bb559b927292"],"wire digest mismatch");
        std::cout<<"Swift envelope accepted by native PASS\n";return 0;
    }
    NSDictionary *request=@{@"schema_version":@2,@"model":@"z-image-turbo",@"operation":@"image.generate",
        @"inputs":@[@{@"kind":@"text",@"role":@"prompt",@"text":@"雪/fox \"gold\""}],
        @"outputs":@[@{@"kind":@"image",@"path":@"/tmp/staging/output.png",@"width":@512,@"height":@512}],
        @"sampling":@{@"seed":@42,@"steps":@9},
        @"execution":@{@"policy":@"gpu",@"streaming":@{@"schema_version":@2,@"enabled":@YES,@"selection":@"memory_tier",@"retention":@"request",@"target_request_memory_bytes":@(10ull<<30)}}};
    NSMutableDictionary *input=[@{@"protocol_version":@1,@"job_id":@"00000000-0000-4000-8000-000000000001",@"request_id":@"00000000-0000-4000-8000-000000000002",@"model_installation_ref":@"/tmp/model",@"native_request_v2":request,@"request_digest":request_digest(request)} mutableCopy];
    validate_input(input);
    require([request_digest(request) isEqual:@"802fd8d904d4e5421f73382626dc03c5bb20b433c16aa82c9816bb559b927292"],"canonical bytes changed");
    auto original=request_digest(request);NSMutableDictionary *changed=[request mutableCopy];
    changed[@"outputs"]=@[@{@"kind":@"image",@"path":@"/tmp/staging/other.png",@"width":@512,@"height":@512}];
    require(![request_digest(changed) isEqual:original],"output path not bound");
    NSMutableDictionary *bad=[input mutableCopy];bad[@"native_request_v2"]=changed;rejects([&]{validate_input(bad);});
    for (id version in @[@YES,@0,@2,@"1"]) {bad=[input mutableCopy];bad[@"protocol_version"]=version;rejects([&]{validate_input(bad);});}
    for (NSString *path in @[@"relative",@"/tmp/../model",@"/tmp/\0model"]) {bad=[input mutableCopy];bad[@"model_installation_ref"]=path;rejects([&]{validate_input(bad);});}
    bad=[input mutableCopy];bad[@"extra"]=@1;rejects([&]{validate_input(bad);});
    bad=[input mutableCopy];[bad removeObjectForKey:@"job_id"];rejects([&]{validate_input(bad);});
    bad=[input mutableCopy];bad[@"request_id"]=@"not-a-uuid";rejects([&]{validate_input(bad);});
    changed=[request mutableCopy];changed[@"model"]=@"unsupported";bad=[input mutableCopy];bad[@"native_request_v2"]=changed;bad[@"request_digest"]=request_digest(changed);rejects([&]{validate_input(bad);});
    auto reply=terminal(input,"test-runtime");
    require([reply[@"job_id"] isEqual:input[@"job_id"]] && [reply[@"request_digest"] isEqual:original] &&
        [reply[@"status"] isEqual:@"error"] && reply[@"artifact"]==[NSNull null],"terminal correlation/defaults wrong");
    std::cout<<original.UTF8String<<"\nworker protocol validation PASS\n";
} } catch(const std::invalid_argument &error) {std::cerr<<error.what()<<"\n";return 2;} }
