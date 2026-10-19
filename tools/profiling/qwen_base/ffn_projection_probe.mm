#include "../../../native/models/qwen21/pipeline.hpp"
#include "../../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "../../../native/backends/ane_ffn.hpp"
#include "../../../native/backends/ane_w8_projection.hpp"
#include "../../../native/backends/ane_w8_activation.hpp"
#include "../../../native/backends/dense_gpu_projection.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#ifdef TC_FFN_DUAL_SCREEN
#include "dual_gup.hpp"
#endif

#ifndef TC_FFN_PROJECTION_CPP_SHA
#define TC_FFN_PROJECTION_CPP_SHA "unbound"
#endif
#ifndef TC_FFN_REPLAY_CHANNELS
#define TC_FFN_REPLAY_CHANNELS 6144
#endif
#ifdef TC_FFN_DOWN_SCREEN
#ifndef TC_FFN_DOWN_BM
#error explicit down tile required
#endif
static_assert(TC_FFN_DOWN_BM==16 || TC_FFN_DOWN_BM==32 || TC_FFN_DOWN_BM==64);
static_assert(TC_FFN_DOWN_BN==64 || TC_FFN_DOWN_BN==128);
static_assert(TC_FFN_DOWN_STATIC==0 || TC_FFN_DOWN_STATIC==1);
#endif

namespace {
using namespace tc;
using Fn=qwen21::DiagnosticGpuBlock;
using Scope=void(*)(int);
constexpr int M=1024,H=4096,F=12288,C=TC_FFN_REPLAY_CHANNELS;
#ifdef TC_FFN_SHARE_SCREEN
static_assert(C==5632 || C==6144 || C==6656,"bounded neighboring-share diagnostic only");
#else
static_assert(C==6144 || C==5120,"bounded fixed FFN handoff shares only");
#endif
NSString *ns(const std::string&s){return [NSString stringWithUTF8String:s.c_str()];}
void write(const std::filesystem::path&p,NSDictionary*v){std::ofstream f(p);f<<json(v)<<'\n';require(bool(f),"FFN probe report write failed");}
std::string hash(const Tensor &x){auto bytes=mx::view(mx::contiguous(x),mx::uint8);mx::eval(bytes);
    return memory_sha256_hex({reinterpret_cast<const char*>(bytes.data<uint8_t>()),bytes.nbytes()});}
NSDictionary *quality(const Tensor&a,const Tensor&b){
    require(a.shape()==b.shape(),"full block output geometry differs");
    auto x=mx::contiguous(mx::astype(a,mx::float32)),y=mx::contiguous(mx::astype(b,mx::float32));mx::eval(x,y);
    double energy=0,error=0,max_abs=0;uint64_t changed=0;
    for(size_t i=0;i<x.size();++i){const double v=x.data<float>()[i],w=y.data<float>()[i],d=v-w;
        require(std::isfinite(v)&&std::isfinite(w),"nonfinite FFN/block output");energy+=v*v;error+=d*d;max_abs=std::max(max_abs,std::abs(d));changed+=v!=w;}
    return @{@"relative_l2":@(std::sqrt(error/std::max(energy,1e-30))),@"max_abs":@(max_abs),@"changed":@(changed),@"elements":@(x.size())};
}
Tensor rope(const Tensor&x,const Tensor&c,const Tensor&s){auto shape=x.shape(),pairs=shape;pairs.back()/=2;pairs.push_back(2);
    auto p=mx::reshape(mx::astype(x,mx::float32),pairs),a=mx::squeeze(slice_axis(p,-1,0,1),-1),b=mx::squeeze(slice_axis(p,-1,1,2),-1);
    auto co=mx::reshape(c,{1,1,shape[2],64}),si=mx::reshape(s,{1,1,shape[2],64});
    return mx::astype(mx::reshape(mx::stack({a*co-b*si,a*si+b*co},-1),shape),x.dtype());
}

void measure(const qwen21::DiagnosticBlockObservation&o,const std::filesystem::path&out,const char *descriptor,
    const std::string&projection,int repeats,NSMutableDictionary *report,Scope scope){
    require(o.layer==0&&o.steady&&o.arguments.size()==9&&o.arguments[0].shape()==mx::Shape{1,M,H}&&o.outputs.size()==1,
        "real base40 decode block0 required");
    const auto &w=o.weights;require(!w.has_runtime_loras(),"base FFN probe cannot carry adapters");
    std::array<std::vector<ane::FfnWeight>,2> sources;
    std::array<qwen21::runtime_ffn::Function,2> full,head;
#ifdef TC_FFN_DUAL_SCREEN
    std::array<qwen21::runtime_ffn::Function,2> ordinary_head;
#endif
#ifdef TC_FFN_DOWN_SCREEN
    std::array<qwen21::runtime_ffn::Function,2> gate_up,down_partial,control_partial;
#endif
    uint64_t packed_gpu_weight_bytes=0;double packed_gpu_setup_seconds=0;
    std::map<std::string,std::string> source_hash;
    for(int layer=0;layer<2;++layer){const auto p="transformer_blocks."+std::to_string(layer)+".img_mlp.";
        const auto &gu=w.at(p+"gate_up.weight"),&down=w.at(p+"out.weight");
        require(gu.dtype()==mx::bfloat16&&down.dtype()==mx::bfloat16&&gu.shape()==mx::Shape{2*F,H}&&down.shape()==mx::Shape{H,F},"original BF16 base FFN matrices required");
        auto halves=mx::split(gu,2,0);mx::eval(halves);
        sources[layer]={{halves[0]},{halves[1]},{down}};
        source_hash[p+"gate_up"]=hash(gu);source_hash[p+"out"]=hash(down);
        full[layer]=qwen21::runtime_ffn::full(w,p);
        head[layer]=qwen21::runtime_ffn::channels(w,p,0,F-C,H,F,true);
#ifdef TC_FFN_DUAL_SCREEN
        ordinary_head[layer]=head[layer];
        head[layer]=mx::compile([&w,p](const std::vector<Tensor>&a){
            auto hidden=dual_gup_lab::prepare(a[0],w.at(p+"gate_up.weight"),0,F-C,bool(TC_FFN_DUAL_SHARED),TC_FFN_DUAL_BM,TC_FFN_DUAL_BN);
            auto base=w.project_base_slice_fp32(hidden,p+"out",0,H,0,F-C);
            return std::vector<Tensor>{base,hidden};});
#endif
#ifdef TC_FFN_DOWN_SCREEN
        require(!w.quantized(p+"out")&&!w.convrot(p+"out"),"original dense down required");
        auto gate_up_body=[&w,p](const std::vector<Tensor>&a){
            auto gate=w.project_slice(a[0],p+"gate_up",0,F-C,0,H,false);
            auto up=w.project_slice(a[0],p+"gate_up",F,2*F-C,0,H,false);
            return std::vector<Tensor>{silu(gate)*up};};
        auto down_body=[&w,p](const std::vector<Tensor>&a){return std::vector<Tensor>{
            dense_gpu::projection_range(a[0],w.at(p+"out.weight"),0,H,0,F-C,TC_FFN_DOWN_BM,true,TC_FFN_DOWN_BN,TC_FFN_DOWN_STATIC)};};
        gate_up[layer]=mx::compile(gate_up_body);down_partial[layer]=mx::compile(down_body);
        control_partial[layer]=mx::compile([&w,p](const std::vector<Tensor>&a){return std::vector<Tensor>{w.project_base_slice_fp32(a[0],p+"out",0,H,0,F-C)};});
        head[layer]=mx::compile([gate_up_body,down_body](const std::vector<Tensor>&a){auto hidden=gate_up_body(a)[0];
            return std::vector<Tensor>{down_body({hidden})[0],hidden};});
#ifdef TC_FFN_GATE_FP16
        auto half_gate_up=[](const Tensor &x,const Tensor &weight){
            auto input=mx::astype(x,mx::float16);
            auto gate=mx::astype(mx::matmul(input,mx::transpose(mx::astype(slice_axis(weight,0,0,F-C),mx::float16))),mx::bfloat16);
            auto up=mx::astype(mx::matmul(input,mx::transpose(mx::astype(slice_axis(weight,0,F,2*F-C),mx::float16))),mx::bfloat16);
            return silu(gate)*up;};
        // W is an explicit compiled input, not a captured cast constant. Each
        // measured call includes the BF16->FP16 conversions; no retained half
        // bank is prepared outside the timer. Original W remains unchanged.
        auto half_gu=mx::compile([half_gate_up](const std::vector<Tensor>&a){return std::vector<Tensor>{half_gate_up(a[0],a[1])};});
        gate_up[layer]=[half_gu,gu](const std::vector<Tensor>&a){return half_gu({a[0],gu});};
        auto half_head=mx::compile([half_gate_up](const std::vector<Tensor>&a){auto hidden=half_gate_up(a[0],a[1]);
            auto base=dense_gpu::projection_range(hidden,a[2],0,H,0,F-C,TC_FFN_DOWN_BM,true,TC_FFN_DOWN_BN,TC_FFN_DOWN_STATIC);
            return std::vector<Tensor>{base,hidden};});
        head[layer]=[half_head,gu,down](const std::vector<Tensor>&a){return half_head({a[0],gu,down});};
#endif
#endif
#ifdef TC_FFN_HANDOFF_PACKED_GPU
        const auto packing_start=Clock::now();
        auto packed=mx::contiguous(mx::concatenate({slice_axis(gu,0,0,F-C),slice_axis(gu,0,F,2*F-C)},0));mx::eval(packed);
        packed_gpu_setup_seconds+=std::chrono::duration<double>(Clock::now()-packing_start).count();packed_gpu_weight_bytes+=packed.nbytes();
        // Base-only diagnostic: preserve BF16 gate/up and the production F32
        // down partial. One request-local packed allocation per captured layer.
        head[layer]=mx::compile([packed,&w,p](const std::vector<Tensor>&a){
            auto halves=mx::split(mx::matmul(a[0],mx::transpose(packed)),2,-1);
            auto hidden=silu(halves[0])*halves[1];
            auto base=w.project_base_slice_fp32(hidden,p+"out",0,H,0,F-C);
            return std::vector<Tensor>{base,hidden};});
#endif
    }
    report[@"gpu_complement_strategy"]=
#ifdef TC_FFN_DUAL_SCREEN
        TC_FFN_DUAL_SHARED?@"dual-shared":@"dual-register";
    report[@"dual_gup"]=@{@"shared":@(bool(TC_FFN_DUAL_SHARED)),@"bm":@(TC_FFN_DUAL_BM),@"bn":@(TC_FFN_DUAL_BN),
        @"header_sha256":@TC_FFN_DUAL_HEADER_SHA,@"first":@0,@"count":@(F-C),@"input_dtype":@"bfloat16",@"weight_copies":@0};
#elif defined(TC_FFN_HANDOFF_PACKED_GPU)
        @"packed";
#else
        @"separate";
#endif
    report[@"gpu_packed_weight_bytes"]=@(packed_gpu_weight_bytes);report[@"gpu_packed_setup_seconds"]=@(packed_gpu_setup_seconds);
#ifdef TC_FFN_DOWN_SCREEN
    report[@"gpu_down_policy"]=@{@"name":@TC_FFN_DOWN_POLICY,@"bm":@(TC_FFN_DOWN_BM),@"bn":@(TC_FFN_DOWN_BN),
        @"static_tiles":@(bool(TC_FFN_DOWN_STATIC)),@"physical_weight_pitch":@(F),@"input_width":@(F-C),@"output_dtype":@"float32",@"weight_copies":@0};
    report[@"gpu_gate_up_precision"]=
#ifdef TC_FFN_GATE_FP16
        @"fp16";report[@"gpu_gate_up_weight_binding"]=@"runtime-arguments-per-call-cast";
#else
        @"bf16";
#endif
#endif
    auto prepare=mx::compile(Fn([&w](const auto&a){const std::string p="transformer_blocks.0.";
        auto x=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);
        auto q=heads(linear(x,w,p+"attn.to_q"),32,128),k=heads(linear(x,w,p+"attn.to_k"),32,128),v=heads(linear(x,w,p+"attn.to_v"),32,128);
        q=rope(mx::fast::rms_norm(q,w.at(p+"attn.norm_q.weight"),1e-6f),a[5],a[6]);
        k=rope(mx::fast::rms_norm(k,w.at(p+"attn.norm_k.weight"),1e-6f),a[5],a[6]);
        auto attended=attend(q,mx::concatenate({a[7],k},2),mx::concatenate({a[8],v},2));
        auto hidden=a[0]+mx::tanh(a[2])*linear(attended,w,p+"attn.to_out.0");
        auto ffn=mx::fast::layer_norm(hidden,{},{},1e-6f)*(Tensor(1.f,hidden.dtype())+a[3]);
        return std::vector<Tensor>{hidden,ffn};}));
    auto finish=mx::compile(Fn([](const auto&a){return std::vector<Tensor>{a[0]+mx::tanh(a[1])*a[2]};}));
    auto original=o.arguments,alternate=o.arguments;
    alternate[0]=mx::astype(alternate[0]*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16),mx::bfloat16);mx::eval(alternate[0]);
    std::vector<std::string> input_hashes;for(const auto&a:original)input_hashes.push_back(hash(a));
    auto reference=o.gpu_block(original)[0],reference_b=o.gpu_block(alternate)[0];mx::eval(reference,reference_b);
    require(hash(reference)==hash(o.outputs[0]),"ordinary GPU block replay changed");
    auto parts=prepare(original);mx::eval(parts);
#ifdef TC_FFN_DUAL_SCREEN
    auto alternate_parts_dual=prepare(alternate);mx::eval(alternate_parts_dual);
    auto ordinary_a=ordinary_head[0]({parts[1]}),dual_a=head[0]({parts[1]});mx::eval(ordinary_a);mx::eval(dual_a);
    auto ordinary_b=ordinary_head[0]({alternate_parts_dual[1]}),dual_b=head[0]({alternate_parts_dual[1]});mx::eval(ordinary_b);mx::eval(dual_b);
    report[@"dual_hidden_quality_a"]=quality(ordinary_a[1],dual_a[1]);report[@"dual_hidden_quality_b"]=quality(ordinary_b[1],dual_b[1]);
    report[@"dual_hidden_bits_exact_a"]=@(hash(ordinary_a[1])==hash(dual_a[1]));report[@"dual_hidden_bits_exact_b"]=@(hash(ordinary_b[1])==hash(dual_b[1]));
    mx::save_safetensors((out/"gpu-complement-arrays.safetensors").string(),{{"input_a",parts[1]},{"input_b",alternate_parts_dual[1]},
        {"hidden_original_a",ordinary_a[1]},{"hidden_fused_a",dual_a[1]},{"partial_original_a",ordinary_a[0]},{"partial_fused_a",dual_a[0]},
        {"hidden_original_b",ordinary_b[1]},{"hidden_fused_b",dual_b[1]},{"partial_original_b",ordinary_b[0]},{"partial_fused_b",dual_b[0]}});
    require([report[@"dual_hidden_quality_a"][@"relative_l2"] doubleValue]<.005 && [report[@"dual_hidden_quality_b"][@"relative_l2"] doubleValue]<.005,
        "paired dense GPU hidden exceeds explicit predeclared screening bound");
#endif
    std::array<std::shared_ptr<void>,2> prepared_owners{parts[0].data_shared_ptr(),parts[1].data_shared_ptr()};
    auto staged=finish({parts[0],original[4],full[0]({parts[1]})[0]})[0];mx::eval(staged);
    require(hash(staged)==hash(reference),"staged GPU control changed complete block arithmetic");
    setenv("TURBOCIDER_ANE_BACKEND","private",1);setenv("TURBOCIDER_ALLOW_PRIVATE_ANE","1",1);
    setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","w8a8",1);setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS",std::to_string(C).c_str(),1);
    setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1",1);setenv("TURBOCIDER_PRIVATE_ANE_W8_PROJECTION",projection.c_str(),1);
    setenv("TURBOCIDER_PRIVATE_ANE_SCALE_CACHE","1",1);setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","1",1);
    setenv("TURBOCIDER_PRIVATE_ANE_PREFETCH","0",1);setenv("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE","1",1);
    setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","1",1);
    setenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1",1);setenv("TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES","0",1);
    setenv("TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_MODE","copy",1);
    if(scope)scope(0);
    std::atomic<bool> cancelled{false};
    ane::HybridFfn runtime(descriptor,H,F,2ull<<30,cancelled);require(runtime.available()&&runtime.channel_split(),runtime.reason());
    ane::HybridFfn::SourceScope source_scope(&runtime);
    auto candidate=[&](const std::vector<Tensor>&args,int layer,bool use_original=false){
#ifndef TC_FFN_DUAL_SCREEN
        (void)use_original;
#endif
        auto plan=runtime.plan_block(layer,M);require(plan.split(),"fixed block candidate declined its share");
        runtime.stage_weights(layer,M,sources[layer]);auto p=prepare(args);
        auto feed=runtime.run(layer,p[1],[&](const Tensor&x){return full[layer]({x})[0];},cancelled,nullptr,
            [&](const Tensor&x,int first,int count){require(first==0&&count==F-C,"GPU complement range changed");
#ifdef TC_FFN_DUAL_SCREEN
                auto r=(use_original?ordinary_head[layer]:head[layer])({x});
#else
                auto r=head[layer]({x});
#endif
                return std::make_pair(r[0],r[1]);});
        auto y=finish({p[0],args[4],feed})[0];mx::eval(y);return y;};
    if(scope)scope(1);auto first=candidate(original,0);const auto first_hash=hash(first);
#ifdef TC_FFN_DUAL_SCREEN
    if(scope)scope(101);auto private_reference_a=candidate(original,0,true);
    report[@"dual_incremental_quality_a"]=quality(private_reference_a,first);
#endif
    if(scope)scope(2);auto different=candidate(alternate,0);
#ifdef TC_FFN_DUAL_SCREEN
    if(scope)scope(102);auto private_reference_b=candidate(alternate,0,true);
    report[@"dual_incremental_quality_b"]=quality(private_reference_b,different);
#endif
    require(hash(different)!=first_hash&&hash(first)==first_hash,"changed activation or old owned output failed");
    if(scope)scope(3);auto restored=candidate(original,0);require(hash(restored)==first_hash,"activation A/B/A failed");
    if(scope)scope(4);auto other_parts=prepare(original);auto other_reference=finish({other_parts[0],original[4],full[1]({other_parts[1]})[0]})[0];mx::eval(other_reference);
    auto other=candidate(original,1);
#ifdef TC_FFN_DUAL_SCREEN
    if(scope)scope(104);auto private_reference_w=candidate(original,1,true);
    report[@"dual_incremental_quality_w"]=quality(private_reference_w,other);
    for(NSString *key:{@"dual_incremental_quality_a",@"dual_incremental_quality_b",@"dual_incremental_quality_w"})
        require([report[key][@"relative_l2"] doubleValue]<.003,"paired GPU complete hybrid increment exceeds explicit screening bound");
#endif
    if(scope)scope(5);auto restored_w=candidate(original,0);require(hash(restored_w)==first_hash&&hash(first)==first_hash,"same program weight A/B/A or independent output failed");
    const auto q=quality(reference,first),qb=quality(reference_b,different),qw=quality(other_reference,other);
    require([q[@"relative_l2"] doubleValue]<.03&&[qb[@"relative_l2"] doubleValue]<.03&&[qw[@"relative_l2"] doubleValue]<.03,"real full-block approximation exceeds screening bound");
#ifdef TC_FFN_DUAL_SCREEN
    constexpr int arm_count=4;const char *names[]{"gpu-original","gpu-staged","private-original","private-fused"};
#else
    constexpr int arm_count=3;const char *names[]{"gpu-original","gpu-staged","private-restaged"};
#endif
    NSMutableArray *samples=[NSMutableArray array];std::vector<Tensor> last(arm_count,Tensor(0.f));
    for(int rep=-2;rep<repeats;++rep)for(int j=0;j<arm_count;++j){const int arm=(j+rep+arm_count)%arm_count;if(scope)scope(6+(rep+2)*arm_count+j);
        const auto &args=(rep&1)?alternate:original;mx::eval(args);
#ifdef TC_FFN_HANDOFF_SCREEN
        const auto before=runtime.metrics();
#endif
        const auto start=Clock::now();Tensor value(0.f);
        if(arm==0)value=o.gpu_block(args)[0];
        else if(arm==1){auto p=prepare(args);value=finish({p[0],args[4],full[0]({p[1]})[0]})[0];}
        else value=candidate(args,0,
#ifdef TC_FFN_DUAL_SCREEN
            arm==2
#else
            false
#endif
        );
        mx::eval(value);const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();last[arm]=value;
        NSMutableDictionary *sample=[@{@"arm":@(names[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(rep&1)),@"seconds":@(elapsed)} mutableCopy];
#ifdef TC_FFN_HANDOFF_SCREEN
        if(arm>=2){const auto after=runtime.metrics();sample[@"completion_deltas"]=@{
            @"prediction_wait_seconds":@(after.model_prediction_seconds-before.model_prediction_seconds),
            @"output_wait_seconds":@(after.output_handling_seconds-before.output_handling_seconds),
            @"caller_wait_seconds":@(after.runtime_weight_async_ane_wait_seconds-before.runtime_weight_async_ane_wait_seconds)};}
#endif
        [samples addObject:sample];
    }
#ifdef TC_FFN_HANDOFF_SCREEN
    // Independent GPU-complement boundary: same weights and ready captured FFN
    // input, no ANE work in flight. Not a complete block or speed denominator.
    auto alternate_parts=prepare(alternate);mx::eval(alternate_parts);
    NSMutableArray *complement_samples=[NSMutableArray array];
    for(int rep=-2;rep<repeats;++rep)
#ifdef TC_FFN_DUAL_SCREEN
      for(int visit=0;visit<2;++visit){const int arm=(visit+rep+2)%2;if(scope)scope(1000+(rep+2)*2+visit);
#else
      {if(scope)scope(1000+rep+2);
#endif
        const auto &feed=(rep&1)?alternate_parts[1]:parts[1];const auto start=Clock::now();
#ifdef TC_FFN_DUAL_SCREEN
        auto values=(arm==0?ordinary_head[0]:head[0])({feed});
#else
        auto values=head[0]({feed});
#endif
        mx::eval(values);const auto seconds=std::chrono::duration<double>(Clock::now()-start).count();
        NSMutableDictionary *s=[@{@"rep":@(rep),@"warmup":@(rep<0),@"seconds":@(seconds),@"input_variant":@(bool(rep&1))} mutableCopy];
#ifdef TC_FFN_DUAL_SCREEN
        s[@"arm"]=arm==0?@"original":@"fused";
#endif
        [complement_samples addObject:s];}
    report[@"gpu_complement_alone_samples"]=complement_samples;
    report[@"handoff_screen"]=@YES;
#ifdef TC_FFN_DOWN_SCREEN
    // Component-only attribution, with all intermediate producers ready.
    // Keep ordinary F32 down as a numerical control, not a weakened full-GPU
    // block denominator. No permanent compact/widened weight bank is created.
    auto hidden_a=gate_up[0]({parts[1]})[0],hidden_b=gate_up[0]({alternate_parts[1]})[0];mx::eval(hidden_a,hidden_b);
    auto down_a=down_partial[0]({hidden_a})[0],down_b=down_partial[0]({hidden_b})[0];
    auto control_a=control_partial[0]({hidden_a})[0],control_b=control_partial[0]({hidden_b})[0];mx::eval(down_a,down_b,control_a,control_b);
    report[@"down_quality_a"]=quality(control_a,down_a);report[@"down_quality_b"]=quality(control_b,down_b);
    require([report[@"down_quality_a"][@"relative_l2"] doubleValue]<1e-4 && [report[@"down_quality_b"][@"relative_l2"] doubleValue]<1e-4,
        "F32 down candidate changed original physical-pitch projection");
    NSMutableArray *component_samples=[NSMutableArray array];
    const char *component_names[]{"gate-up-alone","down-alone","control-down-alone"};
    for(int rep=-2;rep<repeats;++rep)for(int j=0;j<3;++j){const int arm=(j+rep+3)%3;if(scope)scope(2000+(rep+2)*3+j);
        const auto &feed=(rep&1)?alternate_parts[1]:parts[1];const auto &hidden=(rep&1)?hidden_b:hidden_a;
        const auto start=Clock::now();auto result=arm==0?gate_up[0]({feed})[0]:arm==1?down_partial[0]({hidden})[0]:control_partial[0]({hidden})[0];
        mx::eval(result);const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
        [component_samples addObject:@{@"arm":@(component_names[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(rep&1)),@"seconds":@(seconds)}];}
    report[@"gpu_down_component_samples"]=component_samples;
    std::unordered_map<std::string,Tensor> down_arrays{{"hidden_a",hidden_a},{"hidden_b",hidden_b},
        {"down_a",down_a},{"down_b",down_b},{"control_a",control_a},{"control_b",control_b}};
#ifdef TC_FFN_GATE_FP16
    const std::string gp="transformer_blocks.0.img_mlp.";
    auto ordinary_gu=mx::compile([&w,gp](const std::vector<Tensor>&a){auto g=w.project_slice(a[0],gp+"gate_up",0,F-C,0,H,false);
        auto u=w.project_slice(a[0],gp+"gate_up",F,2*F-C,0,H,false);return std::vector<Tensor>{silu(g)*u};});
    auto ordinary_a=ordinary_gu({parts[1]})[0],ordinary_b=ordinary_gu({alternate_parts[1]})[0];mx::eval(ordinary_a,ordinary_b);
    report[@"gate_up_quality_a"]=quality(ordinary_a,hidden_a);report[@"gate_up_quality_b"]=quality(ordinary_b,hidden_b);
    require([report[@"gate_up_quality_a"][@"relative_l2"] doubleValue]<.01 && [report[@"gate_up_quality_b"][@"relative_l2"] doubleValue]<.01,
        "FP16 GPU gate/up exceeds explicit approximate screen bound");
    down_arrays.emplace("ordinary_hidden_a",ordinary_a);down_arrays.emplace("ordinary_hidden_b",ordinary_b);
#endif
    mx::save_safetensors((out/"down-arrays.safetensors").string(),down_arrays);
#endif
#endif
    if(scope)scope(-3);source_scope.finish();auto metrics=runtime.metrics();
    const uint64_t workload_calls=
#ifdef TC_FFN_DUAL_SCREEN
        2*repeats+12;
#else
        repeats+7;
#endif
    require(!metrics.runtime_failed&&metrics.runtime_weight_fallback_blocks==0&&metrics.runtime_weight_overflow_retries==0&&
        metrics.runtime_weight_headroom==1&&metrics.runtime_weight_program_instances==1&&metrics.runtime_weight_blocked_rebuilds==0&&
        metrics.runtime_weight_hybrid_blocks==workload_calls&&metrics.runtime_calls==workload_calls,"candidate silently fell back/retried/recreated or lost complete blocks");
    for(size_t i=0;i<original.size();++i)require(hash(original[i])==input_hashes[i],"borrowed block input changed");
    NSMutableDictionary *hashes=[NSMutableDictionary dictionary];
    for(int layer=0;layer<2;++layer){const auto p="transformer_blocks."+std::to_string(layer)+".img_mlp.";
        require(hash(w.at(p+"gate_up.weight"))==source_hash.at(p+"gate_up")&&hash(w.at(p+"out.weight"))==source_hash.at(p+"out"),"original FFN weights changed");
        hashes[ns(p+"gate_up")]=ns(source_hash.at(p+"gate_up"));hashes[ns(p+"out")]=ns(source_hash.at(p+"out"));}
    std::unordered_map<std::string,Tensor> payload{{"ffn_input",parts[1]},{"block_reference",reference},{"block_candidate",first},
        {"alternate_reference",reference_b},{"alternate_candidate",different},{"other_weight_reference",other_reference},{"other_weight_candidate",other},
        {"last_gpu_original",last[0]},{"last_gpu_staged",last[1]},{"last_private",last.back()}};
#ifdef TC_FFN_DUAL_SCREEN
    payload.emplace("private_original_a",private_reference_a);payload.emplace("private_original_b",private_reference_b);payload.emplace("private_original_w",private_reference_w);
    payload.emplace("last_private_original",last[2]);
#endif
    mx::save_safetensors((out/"arrays.safetensors").string(),payload);
    report[@"metrics"]=to_dictionary(metrics);report[@"complete_block_quality"]=q;report[@"alternate_quality"]=qb;report[@"other_weight_quality"]=qw;
    report[@"samples"]=samples;report[@"source_arrays_sha256"]=hashes;report[@"source_unchanged"]=@YES;report[@"fresh_weights_inputs_outputs_verified"]=@YES;
    report[@"gpu_staged_exact"]=@YES;report[@"prefix_tokens"]=@(o.prefix_tokens);report[@"status"]=@"complete";write(out/"report.json",report);
}
}

extern "C" __attribute__((visibility("default")))
int tc_ffn_projection_probe(const char *model,const char *reference,const char *descriptor,const char *output,const char *projection,int repeats,void(*scope)(int)){
 @autoreleasepool {
    const std::filesystem::path out(output?output:"");
    NSMutableDictionary *report=[@{@"schema":@"tc-base-ffn-projection-block-v1",@"status":@"starting",@"complete_request":@NO,
        @"image_export":@NO,@"cpp_sha256":@TC_FFN_PROJECTION_CPP_SHA,@"repeats":@(repeats),@"warmups":@2,
        @"completed_full_dit_steps":@1,@"stopped_step":@1,@"stopped_layer":@0,@"ane_channels":@(C),@"gpu_channels":@(F-C)} mutableCopy];
    try{
        require(model&&reference&&descriptor&&projection&&output&&repeats>=4&&repeats<=20&&
            std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"bounded real source/probe and fresh output required");
        const auto policy=ane::parse_w8_projection(projection);report[@"projection"]=ns(std::string(ane::w8_projection_name(policy)));
        const auto activation=ane::configured_w8_activation();report[@"activation"]=ns(std::string(ane::w8_activation_name(activation)));
        require(activation==ane::W8Activation::ExpDivide || policy==ane::W8Projection::Descriptor,"activation experiment requires descriptor projection");
        char *id=tc_runtime_build_identity();require(id,"runtime identity missing");report[@"loaded_runtime_id"]=ns(id);tc_string_free(id);
#ifdef TC_FFN_SHARE_SCREEN
        report[@"compiled_native_id"]=@TC_FFN_SHARE_NATIVE_ID;
        require([report[@"loaded_runtime_id"] isEqual:report[@"compiled_native_id"]],"share helper/native C++ ABI identity mismatch");
        report[@"neighbor_share_screen"]=@YES;
#endif
        configure_streams();qwen21::Session session(model);
        Request r;r.model="qwen-image-2.1";r.operation=std::string_view(reference)=="-"?"image.generate":"image.edit";
        r.width=r.height=512;r.steps=40;r.frames=1;r.audio=false;r.seed=73;r.execution="gpu";r.allow_approximation=false;r.residency="resident";
        r.qwen21_dit_cache="off";r.qwen21_dit_cache_explicit=true;r.qwen21_reference_size=1024;r.output=(out/"not-exported.png").string();
        r.prompt=r.operation=="image.generate"?"A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.":
            "Change the teapot to bright red. Preserve its shape, handle, spout, table and background.";
        if(r.operation=="image.edit")r.inputs.push_back({.kind="image",.role="reference",.path=reference});
        report[@"actual_source_request"]=@{@"model":ns(r.model),@"operation":ns(r.operation),@"steps":@(r.steps),@"width":@(r.width),@"height":@(r.height),
            @"seed":@(r.seed),@"audio":@(r.audio),@"lora_count":@(r.loras.size()),@"reference_count":@(r.inputs.size()),@"reference_size":@(r.qwen21_reference_size)};
        session.set_diagnostic_block_observer([&](const qwen21::DiagnosticBlockObservation&o){if(o.layer||!o.steady)return;
            measure(o,out,descriptor,projection,repeats,report,scope);throw qwen21::DiagnosticBlockReplayComplete();});
        std::atomic<bool> cancelled{false};try{(void)session.generate(r,[](const auto&,int,int){},cancelled);throw std::runtime_error("bounded capture did not stop");}
            catch(const qwen21::DiagnosticBlockReplayComplete&){require([report[@"status"] isEqual:@"complete"],"bounded capture incomplete");}
        return 0;
    }catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=ns(e.what());if(std::filesystem::is_directory(out))write(out/"report.json",report);std::cerr<<e.what()<<'\n';return 1;}
 }
}
