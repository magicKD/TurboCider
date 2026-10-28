#include "../../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "../../../native/backends/ane_ffn.hpp"
#include "../../../native/backends/private/ane_w8_executor.hpp"
#include "../../../native/backends/ane_w8_functions.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <array>
#include <fstream>
#include <iostream>
#include <map>

#ifndef TC_REAL_FUNCTIONS_CPP_SHA
#define TC_REAL_FUNCTIONS_CPP_SHA "unbound"
#endif
#ifndef TC_REAL_FUNCTIONS_NATIVE_ID
#define TC_REAL_FUNCTIONS_NATIVE_ID "unbound"
#endif

namespace {
using namespace tc;
constexpr int H=4096,F=12288,C=6144;
constexpr std::array<int,4> sizes{1024,1536,2048,2560};
constexpr std::array<const char*,4> arms{"gpu","single1024","single2048","family"};
NSString*ns(const std::string&s){return[NSString stringWithUTF8String:s.c_str()];}
void write(const std::filesystem::path&p,NSDictionary*r){std::ofstream f(p);f<<json(r)<<'\n';require(bool(f),"real function report write failed");}
std::string hash(const Tensor&a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
ane::DeviceMatrixView view(const Tensor&a,int rows){
    require(a.is_available()&&a.flags().row_contiguous&&a.shape()==mx::Shape{1,rows,H},"ready contiguous real FFN matrix required");
    require(a.dtype()==mx::bfloat16||a.dtype()==mx::float32,"original BF16 input or FP32 partial required");
    return {const_cast<void*>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),rows,H,size_t(H)*a.itemsize(),
            a.dtype()==mx::float32?ane::DType::FP32:ane::DType::BF16,std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
Tensor output(int rows){auto b=mx::allocator::malloc(size_t(rows)*H*4);require(b.ptr()!=nullptr,"FFN output allocation failed");return Tensor(b,{1,rows,H},mx::float32);}
ane::DeviceWeightView weight_view(const Tensor&a){
    require(a.is_available()&&a.ndim()==2&&a.flags().row_contiguous&&a.dtype()==mx::bfloat16,"original ready dense BF16 W required");
    return {const_cast<void*>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),size_t(a.shape(1))*2,a.shape(0),a.shape(1),
        ane::DeviceWeightEncoding::Dense,ane::DType::BF16,32,{},{},std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
NSDictionary*quality(const Tensor&a,const Tensor&b){
    require(a.shape()==b.shape(),"whole FFN geometry changed");auto x=mx::astype(a,mx::float32),y=mx::astype(b,mx::float32);
    auto rel=mx::sqrt(mx::sum(mx::square(x-y))/mx::maximum(mx::sum(mx::square(x)),Tensor(1e-30f)));
    auto maximum=mx::max(mx::abs(x-y));auto finite=mx::all(mx::isfinite(x))&&mx::all(mx::isfinite(y));mx::eval(rel,maximum,finite);
    require(finite.item<bool>(),"nonfinite full FFN consumer");return @{@"relative_l2":@(rel.item<float>()),@"max_abs":@(maximum.item<float>()),@"elements":@(a.size())};
}
struct Result{Tensor value;std::optional<Tensor> partial,head,hidden;ane::RunResult native;};
}

extern "C" __attribute__((visibility("default")))
int tc_w8_functions_real_probe(const char*model,const char*capture,const char*directory,int repeats,void(*scope)(int)) {@autoreleasepool{
    const std::filesystem::path out(directory?directory:"");
    NSMutableDictionary*r=[@{@"schema":@"tc-real-w8-function-ffn-v1",@"status":@"starting",@"cpp_sha256":@TC_REAL_FUNCTIONS_CPP_SHA,
        @"compiled_native_id":@TC_REAL_FUNCTIONS_NATIVE_ID,@"whole_ffn":@YES,@"whole_dit_block":@NO,@"image_export":@NO,
        @"h":@(H),@"f":@(F),@"ane_channels":@(C),@"repeats":@(repeats),@"warmups":@2,
        @"row_scope":@"1024 input_a is real base40 decode; input_b is a controlled perturbation; longer rows are concatenated frozen fixtures, not natural prefill"}mutableCopy];
    try {
        require(model&&capture&&directory&&(repeats==8||repeats==16||repeats==32)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh bounded real FFN fixture required");
        char*raw=tc_runtime_build_identity();require(raw,"native identity missing");const std::string id(raw);tc_string_free(raw);r[@"loaded_runtime_id"]=ns(id);
        require(id==TC_REAL_FUNCTIONS_NATIVE_ID,"real FFN helper/native ABI mismatch");
        setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1",1);setenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION","sigmoid-complement",1);
        setenv("TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD","0",1);setenv("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE","1",1);
        setenv("TURBOCIDER_PRIVATE_ANE_SCALE_CACHE","1",1);setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","1",1);
        setenv("TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES","0",1);configure_streams();
        const char*reverse_raw=std::getenv("TURBOCIDER_PROFILE_REVERSE");
        require(!reverse_raw||std::string(reverse_raw)=="0"||std::string(reverse_raw)=="1","explicit reverse schedule requires0 or1");
        const bool reverse=reverse_raw&&std::string(reverse_raw)=="1";r[@"reverse"]=@(reverse);
        auto captured=mx::load_safetensors(capture);std::array<Tensor,2>x{captured.first.at("input_a"),captured.first.at("input_b")};mx::eval(x[0],x[1]);
        require(x[0].shape()==mx::Shape{1,1024,H}&&x[1].shape()==x[0].shape()&&x[0].dtype()==mx::bfloat16&&x[1].dtype()==mx::bfloat16,"original frozen BF16 inputs required");
        const std::array<std::string,2>input_hash{hash(x[0]),hash(x[1])};
        auto model_arrays=mx::load_safetensors(std::string(model)+"/diffusion_models/qwen_image_2.1_bf16.safetensors");
        std::array<std::unique_ptr<Weights>,2>weights;std::array<std::vector<Tensor>,2>ws;
        std::array<std::vector<ane::DeviceWeightRegion>,2>regions;
        std::array<qwen21::runtime_ffn::Function,2>full,head;
        NSMutableDictionary*source_hashes=[NSMutableDictionary dictionary];
        for(int layer=0;layer<2;++layer){const auto p="transformer_blocks."+std::to_string(layer)+".img_mlp.";
            ws[layer]={model_arrays.first.at(p+"gate_up.weight"),model_arrays.first.at(p+"out.weight")};mx::eval(ws[layer]);
            require(ws[layer][0].shape()==mx::Shape{2*F,H}&&ws[layer][1].shape()==mx::Shape{H,F}&&ws[layer][0].dtype()==mx::bfloat16&&ws[layer][1].dtype()==mx::bfloat16,"original BF16 physical weights required");
            weights[layer]=std::make_unique<Weights>();weights[layer]->bind_arrays({"gate_up.weight","out.weight"},ws[layer]);
            require(!weights[layer]->has_runtime_loras(),"base-only FFN consumer required");
            auto halves=mx::split(ws[layer][0],2,0);mx::eval(halves);
            auto g=weight_view(halves[0]),u=weight_view(halves[1]),d=weight_view(ws[layer][1]);
            regions[layer]={{g,{F-C,C,0,H,128,20260930,false,ane::W8Basis::SylvesterDH}},
                           {u,{F-C,C,0,H,128,20260930,false,ane::W8Basis::SylvesterDH}},
                           {d,{0,H,F-C,C,512,20260930,false,ane::W8Basis::SylvesterDH}}};
            full[layer]=qwen21::runtime_ffn::full(*weights[layer],"");head[layer]=qwen21::runtime_ffn::channels(*weights[layer],"",0,F-C,H,F,true);
            for(int j=0;j<2;++j)source_hashes[ns(p+(j?"out":"gate_up"))]=ns(hash(ws[layer][j]));
        }
        std::map<std::pair<int,int>,Tensor>feeds;
        for(int n:{512,1024,1536,2048,2560})for(int v=0;v<2;++v){
            std::vector<Tensor>parts;for(int first=0;first<n;first+=1024){const int count=std::min(1024,n-first);parts.push_back(slice_axis(x[(v+first/1024)%2],1,0,count));}
            auto input=mx::contiguous(parts.size()==1?parts[0]:mx::concatenate(parts,1));mx::eval(input);feeds.emplace(std::make_pair(n,v),input);
        }
        if(scope)scope(0);
        const ane::GraphShape s1{ane::Kind::SwiGLU,1024,H,C,4096,512,false},s2{ane::Kind::SwiGLU,2048,H,C,4096,512,false};
        std::array<std::unique_ptr<ane::PrivateW8Graph>,3>graphs;
        graphs[0]=std::make_unique<ane::PrivateW8Graph>(s1,4ull<<30,out/"single1024-cache");
        graphs[1]=std::make_unique<ane::PrivateW8Graph>(s2,4ull<<30,out/"single2048-cache");
        graphs[2]=std::make_unique<ane::PrivateW8Graph>(s2,4ull<<30,out/"family-cache",ane::W8Basis::SylvesterDH,std::vector<int>{512,1024,1536,2048});
        for(auto&graph:graphs){std::string error;require(graph->self_test(error),error);}
        r[@"source_hashes"]=source_hashes;r[@"input_a_sha256"]=ns(input_hash[0]);r[@"input_b_sha256"]=ns(input_hash[1]);
        NSMutableArray*slots=[NSMutableArray new];for(const auto&g:graphs)[slots addObject:@{@"bytes":@(g->slot_bytes()),@"estimate":@(g->estimated_bytes()),@"recipe":ns(g->weight_recipe())}];r[@"graph_slots"]=slots;
        auto run=[&](int arm,int n,int variant,int layer)->Result{
            const auto&input=feeds.at({n,variant});
            if(!arm){auto v=full[layer]({input})[0];mx::eval(v);return {v,{},{},{},{}};}
            auto&graph=*graphs[arm-1];const int alignment=arm==3?512:graph.shape().rows;
            const int padded=(n+alignment-1)/alignment*alignment;
            auto packed=mx::contiguous(padded==n?input:mx::concatenate({input,mx::zeros({1,padded-n,H},input.dtype())},1));mx::eval(packed);
            auto partial=output(padded);graph.stage_device_weight_regions(regions[layer]);const auto staged=graph.wait_stage();require(staged.ok,staged.error);
            graph.launch_device(view(packed,padded),view(partial,padded));
            auto h=head[layer]({input});mx::async_eval(h);const auto done=graph.finish();mx::eval(h);
            require(done.ok&&done.overflow_retries==0&&done.headroom_scale==1,done.error);
            uint64_t expected=0;
            if(arm==3){for(int left=padded;left>0;){const auto rows=ane::w8_function_chunk(left,graph.fixed_row_functions());left-=rows;++expected;}}
            else expected=padded/graph.shape().rows;
            require(done.calls==expected,"actual complete FFN prediction count differs");
            auto tail=slice_axis(partial,1,0,n);auto merged=mx::astype(h[0]+tail,input.dtype());mx::eval(merged);
            return {merged,tail,h[0],h[1],done};
        };
        // Warm every logical procedure, source layer and GPU shape before any
        // measurement. These constructed long-row feeds are not new images.
        int warm_scope=1;
        for(int n:{512,1024,1536,2048,2560})for(int layer=0;layer<2;++layer)for(int arm=0;arm<4;++arm){const int phase=warm_scope++;if(scope)scope(phase);(void)run(arm,n,layer,layer);}
        std::unordered_map<std::string,Tensor>payload;NSMutableArray*checks=[NSMutableArray new];
        std::map<std::pair<int,int>,std::string>original_hashes;
        int check_scope=50;
        for(int n:sizes)for(int state=0;state<(n==1024?3:1);++state){const int variant=state==1,layer=state==2;
            const auto tag=std::to_string(n)+"_"+std::to_string(state);payload.emplace("input_"+tag,feeds.at({n,variant}));
            std::vector<Result>values;for(int arm=0;arm<4;++arm){const int phase=check_scope++;if(scope)scope(phase);values.push_back(run(arm,n,variant,layer));payload.emplace(std::string(arms[arm])+"_"+tag,values.back().value);
                if(n==1024&&arm){payload.emplace(std::string(arms[arm])+"_tail_"+tag,*values.back().partial);if(arm==1){payload.emplace("head_"+tag,*values.back().head);payload.emplace("hidden_"+tag,*values.back().hidden);}}
            }
            NSMutableArray*q=[NSMutableArray new];for(int arm=0;arm<4;++arm){auto value=quality(values[0].value,values[arm].value);require([value[@"relative_l2"]doubleValue]<.03,"complete FFN approximation exceeds declared3% budget");[q addObject:value];if(state==0)original_hashes[{n,arm}]=hash(values[arm].value);}
            auto incremental=quality(values[2].value,values[3].value);require([incremental[@"relative_l2"]doubleValue]<.005,"family changed FFN arithmetic beyond declared0.5% budget");
            [checks addObject:@{@"rows":@(n),@"state":@(state),@"qualities":q,@"family_vs_single2048":incremental}];
        }
        r[@"checks"]=checks;write(out/"report.json",r);
        NSMutableArray*samples=[NSMutableArray new];r[@"samples"]=samples;
        int sample_scope=100;
        for(int n:sizes)for(int rep=-2;rep<repeats;++rep)for(int j=0;j<4;++j){const int rotated=(j+rep+4)%4,arm=reverse?3-rotated:rotated,variant=bool(rep&1),layer=(rep+2)%4==2,phase=sample_scope++;
            if(scope)scope(phase);const auto started=Clock::now();auto v=run(arm,n,variant,layer);const double wall=std::chrono::duration<double>(Clock::now()-started).count();
            [samples addObject:@{@"rows":@(n),@"arm":@(arms[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(variant),@"layer":@(layer),@"seconds":@(wall),@"scope":@(phase),@"calls":@(v.native.calls)}];}
        if(scope)scope(-3);
        for(int n:sizes)for(int arm=0;arm<4;++arm){auto value=run(arm,n,0,0).value;require(hash(value)==original_hashes.at({n,arm}),"source/shape restoration changed full output bytes");}
        for(auto&graph:graphs){const auto p=graph->program_policy_stats();require(p.fixed&&p.instances==1&&!p.blocked_rebuilds&&!p.disabled,"one retained fixed program per hybrid arm required");}
        for(int layer=0;layer<2;++layer)for(int j=0;j<2;++j){const auto p="transformer_blocks."+std::to_string(layer)+".img_mlp.";require(hash(ws[layer][j])==std::string([source_hashes[ns(p+(j?"out":"gate_up"))]UTF8String]),"original source weights changed");}
        require(hash(x[0])==input_hash[0]&&hash(x[1])==input_hash[1],"original frozen inputs changed");
        mx::save_safetensors((out/"arrays.safetensors").string(),payload);
        r[@"source_unchanged"]=@YES;r[@"restored_bits_exact"]=@YES;r[@"status"]=@"complete";
    }catch(const std::exception&e){r[@"status"]=@"failed";r[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(std::filesystem::is_directory(out))write(out/"report.json",r);return[r[@"status"]isEqual:@"complete"]?0:1;
}}
