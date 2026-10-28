#include "../../../native/models/qwen21/runtime_private_qkv.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

#ifndef TC_QKV_LOOKAHEAD_CPP_SHA
#define TC_QKV_LOOKAHEAD_CPP_SHA "unbound"
#endif
#ifndef TC_QKV_LOOKAHEAD_NATIVE_ID
#define TC_QKV_LOOKAHEAD_NATIVE_ID "unbound"
#endif
namespace {
using namespace tc;using Scope=void(*)(int);
NSString *ns(const std::string&s){return [NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor &a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);
    return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
NSDictionary *prefetch(const QwenPrivateQkvPrefetchMetrics &s){return @{@"requested":@(s.requested),@"enabled":@(s.enabled),@"future_pending":@(s.future_pending),
    @"attempts":@(s.attempts),@"submissions":@(s.submissions),@"producer_submissions":@(s.producer_submissions),@"claims":@(s.claims),
    @"weight_completions":@(s.weight_completions),@"completed_hits":@(s.completed_hits),@"discards":@(s.discards),@"retired_claims":@(s.retired_claims),@"failures":@(s.failures)};}
void write(const std::filesystem::path &p,NSDictionary*v){std::ofstream f(p);f<<json(v)<<'\n';require(bool(f),"QKV probe report write failed");}
}
extern "C" __attribute__((visibility("default")))
int tc_qkv_lookahead_probe(const char *output,void(*scope)(int)) {@autoreleasepool{
    using namespace tc;const std::filesystem::path out(output?output:"");
    NSMutableDictionary *report=[@{@"schema":@"tc-qkv-lookahead-state-probe-v1",@"status":@"starting",@"synthetic_only":@YES,
        @"complete_request":@NO,@"image_export":@NO,@"cpp_sha256":@TC_QKV_LOOKAHEAD_CPP_SHA,@"compiled_native_id":@TC_QKV_LOOKAHEAD_NATIVE_ID} mutableCopy];
    try{
        require(output&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh QKV state fixture required");
        char *raw=tc_runtime_build_identity();require(raw,"native identity missing");std::string identity(raw);tc_string_free(raw);
        report[@"loaded_runtime_id"]=ns(identity);require(identity==TC_QKV_LOOKAHEAD_NATIVE_ID,"probe/native C++ ABI identity mismatch");
        configure_streams();auto diagonal=mx::eye(4096,mx::bfloat16);
        std::array<Tensor,3> a{diagonal*Tensor(.125f,mx::bfloat16),diagonal*Tensor(.25f,mx::bfloat16),diagonal*Tensor(.5f,mx::bfloat16)};
        std::array<Tensor,3> b{diagonal*Tensor(.1875f,mx::bfloat16),diagonal*Tensor(.375f,mx::bfloat16),diagonal*Tensor(.75f,mx::bfloat16)};
        auto x=mx::full({1,1024,4096},.5f,mx::bfloat16),z=mx::full({1,1024,4096},.375f,mx::bfloat16);
        mx::eval({x,z,a[0],a[1],a[2],b[0],b[1],b[2]});
        std::array<std::string,6> before{hash(a[0]),hash(a[1]),hash(a[2]),hash(b[0]),hash(b[1]),hash(b[2])};
        auto compiled=mx::compile([](const std::vector<Tensor>&v){return std::vector<Tensor>{mx::concatenate({
            mx::matmul(v[0],mx::transpose(v[1])),mx::matmul(v[0],mx::transpose(v[2])),mx::matmul(v[0],mx::transpose(v[3]))},-1)};});
        std::array<Tensor,3> current=a;auto gpu=[&](const Tensor &v){return compiled({v,current[0],current[1],current[2]})[0];};
        std::atomic<bool> cancelled{false};if(scope)scope(0);
        qwen21::RuntimePrivateQkv owner(512ull<<20,cancelled);require(owner.available(),owner.metrics().failure_reason);
        owner.begin_request("synthetic-generation-a",true);const auto key=owner.metrics().program_key;
        require(!owner.eligible(1048)&&!owner.prefetch(1,1048,b)&&owner.metrics().gpu_blocks==0,"peek/zero-share boundary consumed credits or producers");
        if(scope)scope(1);owner.stage(0,1024,a);auto original=owner.run(0,x,gpu,cancelled);mx::eval(original);const auto original_hash=hash(original);
        if(scope)scope(2);require(owner.prefetch(1,1024,b),"future B did not submit");
        require(owner.prefetch_metrics().future_pending&&!owner.prefetch(2,1024,a),"more than one future was admitted");
        owner.stage(1,1024,b);current=b;auto changed=owner.run(1,z,gpu,cancelled);mx::eval(changed);require(hash(changed)!=original_hash,"weight/input switch returned stale output");
        if(scope)scope(3);require(owner.prefetch(0,1024,a),"future A did not submit");owner.stage(0,1024,a);current=a;
        auto restored=owner.run(0,x,gpu,cancelled);mx::eval(restored);require(hash(restored)==original_hash,"same-program A/B/A did not restore output");
        if(scope)scope(4);require(owner.prefetch(0,1024,b),"mismatch control did not submit");owner.stage(0,1024,a);
        auto mismatch=owner.run(0,x,gpu,cancelled);mx::eval(mismatch);require(hash(mismatch)==original_hash,"same layer/rows accepted wrong future Tensor IDs");
        if(scope)scope(5);require(owner.prefetch(1,1024,b),"generation-change future missing");owner.begin_request("synthetic-generation-b",true);
        require(!owner.prefetch_metrics().future_pending&&!owner.metrics().validation_passed,"source generation reused future or quality credit");
        owner.stage(0,1024,a);auto generation=owner.run(0,x,gpu,cancelled);mx::eval(generation);require(hash(generation)==original_hash,"generation reset changed the original values");
        if(scope)scope(6);require(owner.prefetch(1,1024,b),"claimed-drain future missing");owner.stage(1,1024,b);owner.drain();
        if(scope)scope(7);require(owner.prefetch(1,1024,b),"disabled future missing");owner.disable("synthetic retirement with committed future");
        require(!owner.available()&&!owner.prefetch(0,1024,a),"disabled owner retried/retained a live model");
        const auto s=owner.prefetch_metrics();const auto &m=owner.metrics();
        require(s.submissions==6&&s.producer_submissions==18&&s.claims==3&&s.weight_completions==2&&s.completed_hits==2&&
            s.discards==3&&s.retired_claims==1&&s.failures==0&&!s.enabled&&!s.future_pending,"actual producer/claim/drain state accounting differs");
        require(m.predictions==5&&m.completed_blocks==5&&m.weight_stages==5&&m.validation_calls==2&&m.fallback_blocks==0&&m.program_key==key,
            "actual same-program projection/numeric coverage differs");
        for(int i=0;i<3;++i)require(hash(a[i])==before[i]&&hash(b[i])==before[i+3],"original synthetic weights mutated");
        require(hash(original)==original_hash,"old owned output was overwritten by lookahead or retirement");
        mx::save_safetensors((out/"arrays.safetensors").string(),{{"input_a",x},{"input_b",z},{"output_a",original},{"output_b",changed},
            {"restored_a",restored},{"mismatch_a",mismatch},{"generation_a",generation}});
        report[@"prefetch"]=prefetch(s);report[@"predictions"]=@(m.predictions);report[@"validation_calls"]=@(m.validation_calls);
        report[@"no_fallback"]=@YES;report[@"same_program_key"]=ns(key);report[@"source_unchanged"]=@YES;
        report[@"old_output_unchanged"]=@YES;report[@"status"]=@"complete";
    }catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);write(out/"report.json",report);return [report[@"status"] isEqual:@"complete"]?0:1;
}}
