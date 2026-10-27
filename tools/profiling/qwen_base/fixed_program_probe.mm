#include "../../../native/backends/private/ane_w8_executor.hpp"
#include "../../../native/backends/ane_ffn.hpp"
#include "../../../native/backends/ane_w8_activation.hpp"
#include "../../../native/models/qwen21/runtime_private_qkv.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

#ifndef TC_FIXED_PROGRAM_CPP_SHA
#define TC_FIXED_PROGRAM_CPP_SHA "unbound"
#endif

namespace {
using namespace tc;
constexpr int M=32,H=128,C=512,F=1024;
using Scope=void(*)(int);
NSString *ns(const std::string&s){return [NSString stringWithUTF8String:s.c_str()];}
void write(const std::filesystem::path&p,NSDictionary*v){std::ofstream f(p);f<<json(v)<<'\n';require(bool(f),"fixed probe report write failed");}
std::string hash(const Tensor &a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);
    return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
Tensor diagonal(int rows,int cols,float scale,int second=-1){
    std::vector<float> values(size_t(rows)*cols,0);
    for(int r=0;r<rows;++r){values[size_t(r)*cols+r%cols]=scale;if(second>=0)values[size_t(r)*cols+second+r]=scale;}
    auto a=mx::astype(Tensor(values.data(),{rows,cols},mx::float32),mx::bfloat16);mx::eval(a);return a;
}
ane::DeviceMatrixView view(const Tensor&a,int rows,int cols){
    require(a.is_available()&&a.dtype()==mx::bfloat16&&a.flags().row_contiguous,"ready BF16 source required");
    return {const_cast<void*>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),rows,cols,size_t(cols)*2,
            ane::DType::BF16,std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
ane::DeviceWeightView weight(const Tensor&a){const auto v=view(a,a.shape(0),a.shape(1));
    return {v.buffer,v.buffer_bytes,v.offset_bytes,v.row_stride_bytes,v.rows,v.cols,ane::DeviceWeightEncoding::Dense,
            ane::DType::BF16,32,{},{},v.owner,v.allocation_identity};}
NSDictionary *stats(const ane::ProgramPolicyStats&s){return @{@"fixed":@(s.fixed),@"instances":@(s.instances),
    @"blocked_rebuilds":@(s.blocked_rebuilds),@"disabled":@(s.disabled)};}
NSDictionary *run(const ane::RunResult&r){return @{@"ok":@(r.ok),@"error":ns(r.error),@"calls":@(r.calls),
    @"overflow_retries":@(r.overflow_retries),@"headroom_start":@(r.headroom_start_scale),@"headroom_end":@(r.headroom_scale)};}
double relative(const Tensor &a,const Tensor &b){auto x=mx::astype(a,mx::float32),y=mx::astype(b,mx::float32);
    auto value=mx::sqrt(mx::sum(mx::square(x-y))/mx::maximum(mx::sum(mx::square(y)),Tensor(1e-30f)));
    return value.item<float>();}
}

extern "C" __attribute__((visibility("default")))
int tc_fixed_program_probe(const char *output,const char *descriptor,int strict,void(*scope)(int)) {
 @autoreleasepool {
    const std::filesystem::path out(output?output:"");
    NSMutableDictionary *report=[@{@"schema":@"tc-fixed-program-negative-v1",@"status":@"starting",
        @"synthetic_only":@YES,@"complete_request":@NO,@"image_export":@NO,@"strict":@(bool(strict)),
        @"cpp_sha256":@TC_FIXED_PROGRAM_CPP_SHA,@"rows":@(M),@"hidden":@(H),@"ane_channels":@(C),@"full_width":@(F)} mutableCopy];
    try {
        require(output&&descriptor&&(strict==0||strict==1)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh bounded probe output required");
        const auto activation=ane::configured_w8_activation();
        require(activation==ane::W8Activation::ExpDivide || strict,"non-default activation requires fixed program");
        const auto activation_name=std::string(ane::w8_activation_name(activation));report[@"activation"]=ns(activation_name);
        char *id=tc_runtime_build_identity();require(id,"runtime identity unavailable");report[@"loaded_runtime_id"]=ns(id);tc_string_free(id);
        setenv("TURBOCIDER_ANE_BACKEND","private",1);setenv("TURBOCIDER_ALLOW_PRIVATE_ANE","1",1);
        setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","w8a8",1);setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","512",1);
        setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM",strict?"1":"0",1);
        setenv("TURBOCIDER_PRIVATE_ANE_PREFETCH","0",1);setenv("TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD","0",1);
        setenv("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE","1",1);setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","1",1);
        setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);setenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1",1);
        configure_streams();
        auto x=mx::full({1,M,H},8.f,mx::bfloat16);mx::eval(x);const auto input_hash=hash(x);
        auto safe_g=diagonal(C,H,.125f),safe_u=diagonal(C,H,.125f),direct_d=diagonal(H,C,.25f);
        auto high_g=diagonal(C,H,32.f),high_u=diagonal(C,H,32.f);
        auto direct_y=mx::zeros({1,M,H},mx::bfloat16),invalid_y=mx::zeros({1,M,H},mx::bfloat16);mx::eval(direct_y,invalid_y);
        if(scope)scope(0);
        ane::PrivateW8Graph graph({ane::Kind::SwiGLU,M,H,C,128,512,false},512u<<20,out/"direct-cache");
        std::string error;require(graph.self_test(error),error);
        require(graph.program_policy_stats().instances==1,"self-test created another program");
        const auto captured_recipe=graph.weight_recipe();
        // Deliberately change ambient policy: the existing owner must retain
        // its captured choice, rather than reading mutable env on overflow.
        setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM",strict?"0":"1",1);
        setenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION",activation==ane::W8Activation::Sigmoid?"silu":"sigmoid",1);
        if(scope)scope(1);
        graph.stage_device_weights({weight(safe_g),weight(safe_u),weight(direct_d)});
        require(graph.wait_stage().ok,"safe direct stage failed");graph.launch_device(view(x,M,H),view(direct_y,M,H));
        const auto safe=graph.finish();require(safe.ok,safe.error);const auto safe_hash=hash(direct_y);
        const auto before=graph.program_policy_stats();
        if(scope)scope(2);
        graph.stage_device_weights({weight(high_g),weight(high_u),weight(direct_d)});
        require(graph.wait_stage().ok,"overflow direct stage failed");graph.launch_device(view(x,M,H),view(invalid_y,M,H));
        const auto overflow=graph.finish();const auto after=graph.program_policy_stats();
        require(strict?(!overflow.ok&&overflow.calls==1&&after.instances==1&&after.blocked_rebuilds==1&&after.disabled):
            (overflow.ok&&overflow.overflow_retries>0&&after.instances==1+overflow.overflow_retries&&!after.disabled),
            "actual direct overflow policy outcome differs");
        if(scope)scope(3);
        bool rejected=false;ane::RunResult repeated;
        try {graph.stage_device_weights({weight(high_g),weight(high_u),weight(direct_d)});
            require(graph.wait_stage().ok,"repeat stage failed");graph.launch_device(view(x,M,H),view(invalid_y,M,H));repeated=graph.finish();}
        catch(const std::exception&){rejected=true;}
        require(strict?rejected:(!rejected&&repeated.ok&&repeated.calls==1&&repeated.overflow_retries==0),"disabled executor retried or legacy headroom was lost");
        require(hash(direct_y)==safe_hash&&hash(x)==input_hash,"old output/input changed after failed projection");
        require(graph.weight_recipe()==captured_recipe,"existing owner changed its captured activation recipe");
        report[@"direct"]=@{@"safe":run(safe),@"overflow":run(overflow),@"before":stats(before),@"after":stats(after),
            @"repeat_stage_rejected":@(rejected),@"repeat":run(repeated),@"after_repeat":stats(graph.program_policy_stats()),
            @"captured_policy_survived_env_change":@YES,@"captured_activation_survived_env_change":@YES,
            @"captured_recipe":ns(captured_recipe),@"old_output_unchanged":@YES};
        setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM",strict?"1":"0",1);
        setenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION",activation_name.c_str(),1);
        auto g=diagonal(F,H,32.f),u=diagonal(F,H,32.f),d=diagonal(H,F,.25f,C);
        const auto ghash=hash(g),uhash=hash(u),dhash=hash(d);
        auto gpu=[&](const Tensor &input,int first,int count){
            auto gv=mx::matmul(input,mx::transpose(slice_axis(g,0,first,first+count)));
            auto uv=mx::matmul(input,mx::transpose(slice_axis(u,0,first,first+count)));
            auto hidden=silu(gv)*uv;
            auto result=mx::matmul(hidden,mx::transpose(slice_axis(d,1,first,first+count)));
            return std::make_pair(result,hidden);};
        int full_calls=0;
        auto full=[&](const Tensor &input){++full_calls;return gpu(input,0,F).first;};
        auto channel=[&](const Tensor &input,int first,int count){auto p=gpu(input,first,count);
            return std::make_pair(mx::astype(p.first,mx::float32),p.second);};
        const std::vector<ane::FfnWeight> sources{{g},{u},{d}};
        if(scope)scope(4);
        std::atomic<bool> cancelled{false};
        ane::HybridFfn runtime(descriptor,H,F,512u<<20,cancelled);
        require(runtime.available()&&runtime.channel_split(),runtime.reason());
        ane::HybridFfn::SourceScope source_scope(&runtime);
        if(scope)scope(5);
        runtime.begin_request("synthetic-fixed-program");auto plan=runtime.plan_block(0,M);require(plan.split(),"missing first hybrid plan");
        runtime.stage_weights(0,M,sources);auto first=runtime.run(0,x,full,cancelled,nullptr,channel);mx::eval(first);
        auto reference=gpu(x,0,F).first;mx::eval(reference);
        const auto first_metrics=runtime.metrics();
        const bool first_retained=runtime.retains_resources();const bool sticky=runtime.sticky_fixed_program_failure();
        require(mx::all(mx::isfinite(first)).item<bool>()&&relative(first,reference)<.06,"invalid/incorrect hybrid result");
        if(strict)require(mx::all(first==reference).item<bool>()&&first_metrics.runtime_failed&&
            first_metrics.runtime_weight_fallback_blocks==1&&!first_retained&&sticky&&runtime.usable_configuration(),"complete GPU fallback/resource/sticky contract failed");
        else require(!first_metrics.runtime_failed&&first_metrics.runtime_weight_overflow_retries>0&&!sticky,"legacy hybrid did not exercise recovery");
        if(scope)scope(6);
        runtime.begin_request("synthetic-fixed-program");plan=runtime.plan_block(1,M);
        runtime.stage_weights(1,M,sources);auto second=runtime.run(1,x,full,cancelled,nullptr,channel);mx::eval(second);
        const auto second_metrics=runtime.metrics();source_scope.finish();
        if(strict)require(!plan.split()&&mx::all(second==reference).item<bool>()&&full_calls==2&&
            second_metrics.runtime_calls==first_metrics.runtime_calls&&!runtime.retains_resources(),"next request retried or failed full GPU fallback");
        else require(plan.split()&&relative(second,reference)<.06&&full_calls==0&&
            second_metrics.runtime_weight_overflow_retries==first_metrics.runtime_weight_overflow_retries,"legacy repeat did not reuse its new headroom");
        require(hash(x)==input_hash&&hash(g)==ghash&&hash(u)==uhash&&hash(d)==dhash,"synthetic source mutated");
        // The disabled peer constructor must not allocate/load an ANE model.
        if(scope)scope(7);
        qwen21::RuntimePrivateQkv peer("synthetic peer FFN disabled");
        require(!peer.available()&&!peer.plan(1024)&&peer.metrics().slot_bytes==0,"disabled peer allocated a live graph");
        peer.disable("do not overwrite first failure");require(peer.metrics().failure_reason=="synthetic peer FFN disabled","first failure provenance lost");
        report[@"hybrid"]=@{@"first_metrics":to_dictionary(first_metrics),@"second_metrics":to_dictionary(second_metrics),
            @"first_relative_l2":@(relative(first,reference)),@"second_relative_l2":@(relative(second,reference)),
            @"first_retained_resources":@(first_retained),@"sticky_failure":@(sticky),@"full_gpu_calls":@(full_calls),
            @"complete_gpu_bits_exact":@(mx::all(first==reference).item<bool>()&&mx::all(second==reference).item<bool>()),
            @"disabled_peer_no_model":@YES,@"source_unchanged":@YES};
        mx::save_safetensors((out/"arrays.safetensors").string(),{{"input",x},{"first",first},{"second",second},{"gpu_reference",reference},{"safe_direct",direct_y}});
        if(scope)scope(-3);
        report[@"status"]=@"complete";write(out/"report.json",report);return 0;
    } catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=ns(e.what());
        if(std::filesystem::is_directory(out))write(out/"report.json",report);std::cerr<<e.what()<<'\n';return 1;}
 }
}
