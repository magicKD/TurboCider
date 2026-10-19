#include "compiled_complement.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

namespace {
using namespace tc;
NSString*ns(const std::string&s){return[NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor&a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
void write(const std::filesystem::path&p,NSDictionary*r){std::ofstream f(p);f<<json(r)<<'\n';require(bool(f),"compiled complement report write failed");}
NSDictionary*quality(const Tensor&a,const Tensor&b){auto x=mx::astype(a,mx::float32),y=mx::astype(b,mx::float32);
    auto rel=mx::sqrt(mx::sum(mx::square(x-y))/mx::maximum(mx::sum(mx::square(x)),Tensor(1e-30f)));auto maximum=mx::max(mx::abs(x-y));mx::eval(rel,maximum);
    return @{@"relative_l2":@(rel.item<float>()),@"max_abs":@(maximum.item<float>()),@"elements":@(a.size())};}
}
extern "C" __attribute__((visibility("default")))
int tc_compiled_complement_probe(const char*model,const char*capture,const char*output,int repeats,void(*scope)(int)){@autoreleasepool{
    const std::filesystem::path out(output?output:"");
    NSMutableDictionary*r=[@{@"schema":@"tc-compiled-complement-v1",@"status":@"starting",@"complete_request":@NO,@"image_export":@NO,
        @"cpp_sha256":@TC_COMPILED_COMP_CPP_SHA,@"header_sha256":@TC_COMPILED_COMP_HEADER_SHA,@"compiled_native_id":@TC_COMPILED_COMP_NATIVE_ID,
        @"repeats":@(repeats),@"warmups":@2,@"ane_predictions":@0,@"extra_owned_weight_bytes":@0,@"m":@1024,@"h":@4096,@"f":@12288,@"channels":@6144}mutableCopy];
    try{require(model&&capture&&output&&(repeats==16||repeats==32)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh bounded compiled complement fixture required");
        char*raw=tc_runtime_build_identity();require(raw,"native identity missing");const std::string id(raw);tc_string_free(raw);r[@"loaded_runtime_id"]=ns(id);
        require(id==TC_COMPILED_COMP_NATIVE_ID,"compiled complement/native C++ ABI mismatch");configure_streams();
        auto captured=mx::load_safetensors(capture);std::array<Tensor,2>inputs{captured.first.at("input_a"),captured.first.at("input_b")};
        auto source=mx::load_safetensors(std::string(model)+"/diffusion_models/qwen_image_2.1_bf16.safetensors");
        std::array<std::unique_ptr<Weights>,2>weights;std::array<std::vector<Tensor>,2>ws;NSMutableDictionary*source_hashes=[NSMutableDictionary dictionary];
        for(int layer=0;layer<2;++layer){const auto prefix="transformer_blocks."+std::to_string(layer)+".img_mlp.";
            ws[layer]={source.first.at(prefix+"gate_up.weight"),source.first.at(prefix+"out.weight")};mx::eval(ws[layer]);
            weights[layer]=std::make_unique<Weights>();weights[layer]->bind_arrays({"gate_up.weight","out.weight"},ws[layer]);
            for(int j=0;j<2;++j)source_hashes[ns(prefix+(j?"out":"gate_up"))]=ns(hash(ws[layer][j]));}
        mx::eval(inputs[0],inputs[1]);const std::array<std::string,2>input_hash{hash(inputs[0]),hash(inputs[1])};
        auto traces=std::make_shared<uint64_t>(0);
        auto compiled=mx::compile([traces](const std::vector<Tensor>&a){++*traces;return compiled_complement_lab::body(a);});
        auto run=[&](int arm,int variant,int layer){std::vector<Tensor>a{inputs[variant],ws[layer][0],ws[layer][1]};compiled_complement_lab::admit(a);
            if(arm==1)return compiled(a);
            auto g=weights[layer]->project_slice(inputs[variant],"gate_up",0,6144,0,4096,false);
            auto u=weights[layer]->project_slice(inputs[variant],"gate_up",12288,18432,0,4096,false);auto hidden=silu(g)*u;
            auto partial=weights[layer]->project_base_slice_fp32(hidden,"out",0,4096,0,6144);return std::vector<Tensor>{partial,hidden};};
        std::unordered_map<std::string,Tensor>payload{{"input_a",inputs[0]},{"input_b",inputs[1]}};
        std::array<std::array<std::string,2>,3>expected;NSMutableDictionary*checks=[NSMutableDictionary dictionary];
        const std::array<const char*,3>tags{"a","b","w"};
        for(int c=0;c<3;++c){const int v=c==1,l=c==2;std::array<std::vector<Tensor>,2>results;
            for(int arm=0;arm<2;++arm){if(scope)scope(1+c*2+arm);results[arm]=run(arm,v,l);mx::eval(results[arm]);
                for(int j=0;j<2;++j){require(mx::all(mx::isfinite(results[arm][j])).item<bool>(),"nonfinite complement output");
                    payload.emplace(std::string(arm?"compiled_":"raw_")+(j?"hidden_":"partial_")+tags[c],results[arm][j]);}}
            auto direct=compiled_complement_lab::body({inputs[v],ws[l][0],ws[l][1]});mx::eval(direct);
            for(int j=0;j<2;++j){require(hash(direct[j])==hash(results[0][j]),"explicit body changed actual native callback arithmetic");
                expected[c][j]=hash(results[0][j]);}
            checks[@(tags[c])]=@{@"partial":quality(results[0][0],results[1][0]),@"hidden":quality(results[0][1],results[1][1]),
                @"partial_bits_exact":@(hash(results[0][0])==hash(results[1][0])),@"hidden_bits_exact":@(hash(results[0][1])==hash(results[1][1]))};}
        r[@"checks"]=checks;r[@"trace_count_after_sources"]=@(*traces);
        require(expected[0][0]!=expected[1][0]&&expected[0][0]!=expected[2][0],"changed X/W returned stale output");
        uint64_t rejected=0;for(int bad=0;bad<7;++bad){std::vector<Tensor>a{inputs[0],ws[0][0],ws[0][1]};
            if(bad==0)a.pop_back();else if(bad<4)a[bad-1]=mx::astype(a[bad-1],mx::float16);
            else if(bad==4)a[0]=slice_axis(a[0],1,0,512);else if(bad==5)a[1]=slice_axis(a[1],0,0,12288);else a[2]=slice_axis(a[2],1,0,6144);
            bool denied=false;try{compiled_complement_lab::admit(a);}catch(const std::invalid_argument&){denied=true;}require(denied,"invalid compiled source admitted");++rejected;}
        NSMutableArray*samples=[NSMutableArray array];r[@"samples"]=samples;
        std::array<std::vector<Tensor>,2>last;
        for(int rep=-2;rep<repeats;++rep)for(int j=0;j<2;++j){const int arm=(j+rep+2)%2,v=bool(rep&1),layer=(rep+2)%4==2;
            if(scope)scope(100+(rep+2)*2+j);const auto started=Clock::now();auto values=run(arm,v,layer);
            require(!values[0].is_available()&&!values[1].is_available(),"complement reused completed outputs");mx::eval(values);
            const double wall=std::chrono::duration<double>(Clock::now()-started).count();last[arm]=values;
            [samples addObject:@{@"arm":arm?@"compiled":@"raw",@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(v)),@"layer":@(layer),@"seconds":@(wall)}];}
        if(scope)scope(-3);
        for(int arm=0;arm<2;++arm){for(int j=0;j<2;++j)require(hash(last[arm][j])==expected[1][j],"last complement output not owned/restored");
            auto restored=run(arm,0,0);mx::eval(restored);for(int j=0;j<2;++j)require(hash(restored[j])==expected[0][j],"W A/B/A restoration failed");}
        for(int layer=0;layer<2;++layer)for(int j=0;j<2;++j){const auto prefix="transformer_blocks."+std::to_string(layer)+".img_mlp.";
            require(hash(ws[layer][j])==std::string([source_hashes[ns(prefix+(j?"out":"gate_up"))]UTF8String]),"original W mutated");}
        require(hash(inputs[0])==input_hash[0]&&hash(inputs[1])==input_hash[1],"actual input mutated");
        mx::save_safetensors((out/"arrays.safetensors").string(),payload);r[@"trace_count_final"]=@(*traces);r[@"rejected"]=@(rejected);r[@"source_hashes"]=source_hashes;
        r[@"source_unchanged"]=@YES;r[@"explicit_body_native_bits_exact"]=@YES;r[@"status"]=@"complete";
    }catch(const std::exception&e){r[@"status"]=@"failed";r[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(std::filesystem::is_directory(out))write(out/"report.json",r);return[r[@"status"]isEqual:@"complete"]?0:1;
}}
