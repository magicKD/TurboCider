#include "compiled_complement.hpp"
#include "../../../native/backends/ane_ffn.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

namespace {
using namespace tc;
NSString*ns(const std::string&s){return[NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor&a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
void write(const std::filesystem::path&p,NSDictionary*r){std::ofstream f(p);f<<json(r)<<'\n';require(bool(f),"hybrid complement report write failed");}
NSDictionary*quality(const Tensor&a,const Tensor&b){auto x=mx::astype(a,mx::float32),y=mx::astype(b,mx::float32);
    auto rel=mx::sqrt(mx::sum(mx::square(x-y))/mx::maximum(mx::sum(mx::square(x)),Tensor(1e-30f)));auto maximum=mx::max(mx::abs(x-y));mx::eval(rel,maximum);
    return @{@"relative_l2":@(rel.item<float>()),@"max_abs":@(maximum.item<float>()),@"elements":@(a.size())};}
}
extern "C" __attribute__((visibility("default")))
int tc_compiled_complement_hybrid_probe(const char*model,const char*capture,const char*manifest,const char*output,int repeats,void(*scope)(int)){@autoreleasepool{
    const std::filesystem::path out(output?output:"");
    NSMutableDictionary*r=[@{@"schema":@"tc-compiled-complement-hybrid-v1",@"status":@"starting",@"complete_request":@NO,@"image_export":@NO,
        @"complete_dit_block":@NO,@"cpp_sha256":@TC_COMPILED_COMP_CPP_SHA,@"header_sha256":@TC_COMPILED_COMP_HEADER_SHA,@"compiled_native_id":@TC_COMPILED_COMP_NATIVE_ID,
        @"repeats":@(repeats),@"warmups":@2,@"extra_owned_weight_bytes":@0,@"m":@1024,@"h":@4096,@"f":@12288,@"channels":@6144}mutableCopy];
    try{require(model&&capture&&manifest&&output&&(repeats==16||repeats==32)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh bounded hybrid complement fixture required");
        char*raw=tc_runtime_build_identity();require(raw,"native identity missing");const std::string id(raw);tc_string_free(raw);r[@"loaded_runtime_id"]=ns(id);
        require(id==TC_COMPILED_COMP_NATIVE_ID,"hybrid complement/native C++ ABI mismatch");configure_streams();
        auto captured=mx::load_safetensors(capture);std::array<Tensor,2>inputs{captured.first.at("input_a"),captured.first.at("input_b")};
        auto source=mx::load_safetensors(std::string(model)+"/diffusion_models/qwen_image_2.1_bf16.safetensors");
        std::array<std::unique_ptr<Weights>,2>weights;std::array<std::vector<Tensor>,2>ws;std::array<std::vector<ane::FfnWeight>,2>sources;
        NSMutableDictionary*source_hashes=[NSMutableDictionary dictionary];
        for(int layer=0;layer<2;++layer){const auto prefix="transformer_blocks."+std::to_string(layer)+".img_mlp.";
            ws[layer]={source.first.at(prefix+"gate_up.weight"),source.first.at(prefix+"out.weight")};mx::eval(ws[layer]);
            weights[layer]=std::make_unique<Weights>();weights[layer]->bind_arrays({"gate_up.weight","out.weight"},ws[layer]);
            auto halves=mx::split(ws[layer][0],2,0);mx::eval(halves);sources[layer]={{halves[0]},{halves[1]},{ws[layer][1]}};
            for(int j=0;j<2;++j)source_hashes[ns(prefix+(j?"out":"gate_up"))]=ns(hash(ws[layer][j]));}
        mx::eval(inputs[0],inputs[1]);const std::array<std::string,2>input_hash{hash(inputs[0]),hash(inputs[1])};
        auto traces=std::make_shared<uint64_t>(0);auto compiled=mx::compile([traces](const std::vector<Tensor>&a){++*traces;return compiled_complement_lab::body(a);});
        auto full=mx::compile([](const std::vector<Tensor>&a){auto halves=mx::split(mx::matmul(a[0],mx::transpose(a[1])),2,-1);
            return std::vector<Tensor>{mx::matmul(silu(halves[0])*halves[1],mx::transpose(a[2]))};});
        if(scope)scope(0);std::atomic<bool>cancelled{false};ane::HybridFfn runtime(manifest,4096,12288,2ull<<30,cancelled);
        require(runtime.available()&&runtime.channel_split()&&runtime.fp32_channel_join(),runtime.reason());ane::HybridFfn::SourceScope source_scope(&runtime);
        auto run=[&](int arm,int variant,int layer){const auto&x=inputs[variant];std::vector<Tensor>a{x,ws[layer][0],ws[layer][1]};compiled_complement_lab::admit(a);
            auto plan=runtime.plan_block(layer,1024);require(plan.split(),"fixed hybrid candidate declined share");runtime.stage_weights(layer,1024,sources[layer]);
            auto value=runtime.run(layer,x,[&](const Tensor&v){return full({v,ws[layer][0],ws[layer][1]})[0];},cancelled,nullptr,
                [&](const Tensor&v,int first,int count){require(first==0&&count==6144,"hybrid complement range changed");
                    if(arm==1){auto o=compiled({v,ws[layer][0],ws[layer][1]});return std::make_pair(o[0],o[1]);}
                    auto g=weights[layer]->project_slice(v,"gate_up",0,6144,0,4096,false);
                    auto u=weights[layer]->project_slice(v,"gate_up",12288,18432,0,4096,false);auto hidden=silu(g)*u;
                    auto partial=weights[layer]->project_base_slice_fp32(hidden,"out",0,4096,0,6144);return std::make_pair(partial,hidden);});mx::eval(value);return value;};
        std::unordered_map<std::string,Tensor>payload{{"input_a",inputs[0]},{"input_b",inputs[1]}};
        std::array<std::string,3>expected;NSMutableDictionary*checks=[NSMutableDictionary dictionary];const std::array<const char*,3>tags{"a","b","w"};
        for(int c=0;c<3;++c){const int v=c==1,l=c==2;std::vector<Tensor>results;for(int arm=0;arm<2;++arm){if(scope)scope(1+c*2+arm);results.push_back(run(arm,v,l));
                payload.emplace(std::string(arm?"compiled_":"raw_")+tags[c],results.back());}
            auto gpu=full({inputs[v],ws[l][0],ws[l][1]})[0];mx::eval(gpu);payload.emplace(std::string("gpu_")+tags[c],gpu);expected[c]=hash(results[0]);
            checks[@(tags[c])]=@{@"incremental":quality(results[0],results[1]),@"gpu_quality":quality(gpu,results[1]),@"bits_exact":@(hash(results[0])==hash(results[1]))};}
        r[@"checks"]=checks;r[@"trace_count_after_sources"]=@(*traces);require(expected[0]!=expected[1]&&expected[0]!=expected[2],"changed hybrid X/W returned stale output");
        NSMutableArray*samples=[NSMutableArray array];r[@"samples"]=samples;std::array<std::optional<Tensor>,2>last;
        for(int rep=-2;rep<repeats;++rep)for(int j=0;j<2;++j){const int arm=(j+rep+2)%2,v=bool(rep&1),layer=(rep+2)%4==2;
            if(scope)scope(100+(rep+2)*2+j);auto before=runtime.metrics();const auto started=Clock::now();auto value=run(arm,v,layer);
            const double wall=std::chrono::duration<double>(Clock::now()-started).count();last[arm]=value;auto after=runtime.metrics();
            [samples addObject:@{@"arm":arm?@"compiled":@"raw",@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(v)),@"layer":@(layer),@"seconds":@(wall),
                @"completion_deltas":@{@"prediction_wait_seconds":@(after.model_prediction_seconds-before.model_prediction_seconds),
                    @"output_wait_seconds":@(after.output_handling_seconds-before.output_handling_seconds),
                    @"caller_wait_seconds":@(after.runtime_weight_async_ane_wait_seconds-before.runtime_weight_async_ane_wait_seconds)}}];}
        for(int arm=0;arm<2;++arm){require(last[arm]&&hash(*last[arm])==expected[1],"last hybrid output not owned/restored");
            if(scope)scope(10+arm);auto restored=run(arm,0,0);require(hash(restored)==expected[0],"hybrid W A/B/A restoration failed");}
        if(scope)scope(-3);source_scope.finish();auto metrics=runtime.metrics();
        require(!metrics.runtime_failed&&metrics.runtime_weight_fallback_blocks==0&&metrics.runtime_weight_overflow_retries==0&&
            metrics.runtime_weight_headroom==1&&metrics.runtime_weight_program_instances==1&&metrics.runtime_weight_blocked_rebuilds==0&&
            metrics.runtime_calls==uint64_t(2*repeats+12)&&metrics.runtime_weight_hybrid_blocks==metrics.runtime_calls,"hybrid silently retried/rebuilt/fell back or missed calls");
        for(int layer=0;layer<2;++layer)for(int j=0;j<2;++j){const auto prefix="transformer_blocks."+std::to_string(layer)+".img_mlp.";
            require(hash(ws[layer][j])==std::string([source_hashes[ns(prefix+(j?"out":"gate_up"))]UTF8String]),"original hybrid W mutated");}
        require(hash(inputs[0])==input_hash[0]&&hash(inputs[1])==input_hash[1],"actual hybrid input mutated");
        mx::save_safetensors((out/"arrays.safetensors").string(),payload);r[@"trace_count_final"]=@(*traces);r[@"source_hashes"]=source_hashes;
        r[@"metrics"]=to_dictionary(metrics);r[@"source_unchanged"]=@YES;r[@"status"]=@"complete";
    }catch(const std::exception&e){r[@"status"]=@"failed";r[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(std::filesystem::is_directory(out))write(out/"report.json",r);return[r[@"status"]isEqual:@"complete"]?0:1;
}}
