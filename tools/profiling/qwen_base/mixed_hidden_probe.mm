#include "mixed_hidden.hpp"
#include "../../../native/backends/dense_gpu_projection.hpp"
#include "../../../native/backends/mlx.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

namespace {
using namespace tc;
const std::array<const char*,4>names{"original-bf16","rounded-bf16","mixed-ready","mixed-with-cast"};
NSString*ns(const std::string&s){return[NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor&a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
void write(const std::filesystem::path&p,NSDictionary*r){std::ofstream f(p);f<<json(r)<<'\n';require(bool(f),"mixed down report write failed");}
NSDictionary*quality(const Tensor&a,const Tensor&b){auto x=mx::astype(a,mx::float32),y=mx::astype(b,mx::float32);mx::eval(x,y);
    const auto energy=mx::sum(mx::square(x)),err=mx::sum(mx::square(x-y));auto rel=mx::sqrt(err/mx::maximum(energy,Tensor(1e-30f)));auto maximum=mx::max(mx::abs(x-y));mx::eval(rel,maximum);
    return @{@"relative_l2":@(rel.item<float>()),@"max_abs":@(maximum.item<float>()),@"elements":@(a.size())};}
}
extern "C" __attribute__((visibility("default")))
int tc_mixed_hidden_probe(const char*model,const char*capture,const char*output,int repeats,void(*scope)(int)){@autoreleasepool{
    const bool real=capture&&std::string_view(capture)!="-";const std::filesystem::path out(output?output:"");
    NSMutableDictionary*r=[@{@"schema":@"tc-mixed-hidden-down-v1",@"status":@"starting",@"real_hidden":@(real),@"complete_request":@NO,@"image_export":@NO,
        @"cpp_sha256":@TC_MIXED_HIDDEN_CPP_SHA,@"header_sha256":@TC_MIXED_HIDDEN_HEADER_SHA,@"compiled_native_id":@TC_MIXED_HIDDEN_NATIVE_ID,
        @"repeats":@(repeats),@"warmups":@2,@"ane_predictions":@0,@"extra_owned_weight_bytes":@0}mutableCopy];
    try{require(model&&capture&&output&&(repeats==16||repeats==32)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh bounded mixed down fixture required");
        char*raw=tc_runtime_build_identity();require(raw,"native identity missing");const std::string id(raw);tc_string_free(raw);r[@"loaded_runtime_id"]=ns(id);require(id==TC_MIXED_HIDDEN_NATIVE_ID,"mixed hidden/native C++ ABI mismatch");configure_streams();
        int m=33,k=128,n=257,pitch=320,rb=5,cb=17;std::vector<Tensor>inputs;Tensor weights(0.f);std::string source_w_hash;
        if(real){m=1024;k=6144;n=4096;pitch=12288;rb=cb=0;auto h=mx::load_safetensors(capture);inputs={h.first.at("hidden_original_a"),h.first.at("hidden_original_b")};
            auto w=mx::load_safetensors(std::string(model)+"/diffusion_models/qwen_image_2.1_bf16.safetensors");weights=w.first.at("transformer_blocks.0.img_mlp.out.weight");}
        else{auto ids=mx::arange((m+2)*k,mx::int32);auto x=mx::reshape(mx::astype((ids%29)-14,mx::bfloat16)/Tensor(32.f,mx::bfloat16),{1,m+2,k});
            inputs={slice_axis(x,1,1,m+1)};inputs.push_back(inputs[0]*Tensor(.75f,mx::bfloat16)+Tensor(.015625f,mx::bfloat16));
            auto wi=mx::arange((n+rb+4)*pitch,mx::int32);auto w=mx::reshape(mx::astype(((wi%257)*37+13)%31-15,mx::bfloat16)/Tensor(128.f,mx::bfloat16),{n+rb+4,pitch});weights=slice_axis(w,0,2,n+rb+2);}
        mx::eval(weights);for(auto&x:inputs){mx::eval(x);require(x.shape()==mx::Shape{1,m,k}&&x.dtype()==mx::bfloat16,"actual complete BF16 hidden required");}
        source_w_hash=hash(weights);std::array<std::string,2>input_hash{hash(inputs[0]),hash(inputs[1])};
        std::vector<Tensor>halves,rounds;for(const auto&x:inputs){halves.push_back(mx::astype(x,mx::float16));rounds.push_back(mx::astype(halves.back(),mx::bfloat16));}mx::eval(halves);mx::eval(rounds);
        auto run=[&](int arm,int v){if(arm==0||arm==1)return dense_gpu::projection_range(arm?rounds[v]:inputs[v],weights,rb,rb+n,cb,cb+k,32,true);
            return mixed_hidden_lab::down(arm==2?halves[v]:mx::astype(inputs[v],mx::float16),weights,rb,rb+n,cb,cb+k);};
        std::unordered_map<std::string,Tensor>payload;std::array<std::array<std::string,2>,4>expected;NSMutableDictionary*checks=[NSMutableDictionary dictionary];
        for(int v=0;v<2;++v){const auto tag=std::string(v?"b":"a");payload.emplace("input_"+tag,inputs[v]);payload.emplace("half_"+tag,halves[v]);payload.emplace("round_"+tag,rounds[v]);
            for(int arm=0;arm<4;++arm){if(scope)scope(1+v*4+arm);auto value=run(arm,v);mx::eval(value);require(mx::all(mx::isfinite(value)).item<bool>(),"mixed down nonfinite output");
                expected[arm][v]=hash(value);payload.emplace(std::string(names[arm])+"_"+tag,value);}
            auto control=mixed_hidden_lab::down(inputs[v],weights,rb,rb+n,cb,cb+k);mx::eval(control);require(hash(control)==expected[0][v],"same-type source body changed original F32 partial");
            checks[ns(tag)]=@{@"cast_quality":quality(inputs[v],rounds[v]),@"incremental_partial_quality":quality(payload.at("original-bf16_"+tag),payload.at("mixed-ready_"+tag)),
                @"same_value_partial_quality":quality(payload.at("rounded-bf16_"+tag),payload.at("mixed-ready_"+tag))};}
        require(expected[0][0]!=expected[0][1],"changed hidden returned stale down partial");
        uint64_t rejected=0;for(int bad=0;bad<6;++bad){bool denied=false;try{(void)mixed_hidden_lab::down(bad==0?mx::astype(inputs[0],mx::float32):inputs[0],
                bad==1?mx::astype(weights,mx::float16):weights,bad==2?-1:rb,bad==3?weights.shape(0)+1:rb+n,bad==4?-1:cb,bad==5?cb+k-1:cb+k);}
            catch(const std::invalid_argument&){denied=true;}require(denied,"invalid mixed range/dtype accepted");++rejected;}
        if(!real){
            // Expected format-domain loss, never relabel tiny BF16 as zero.
            NSMutableDictionary*domains=[NSMutableDictionary dictionary];r[@"format_domains"]=domains;
            for(int pattern=0;pattern<4;++pattern){const char*tag=pattern==0?"underflow":pattern==1?"subnormal":pattern==2?"zero":"wide";
                const float factor=pattern==0?0x1p-24f:pattern==1?0x1p-16f:pattern==2?0.f:262144.f;
                auto x=inputs[0]*Tensor(factor,mx::bfloat16);auto h=mx::astype(x,mx::float16);mx::eval(x,h);
                const bool finite=mx::all(mx::isfinite(h)).item<bool>();
                if(pattern==3){require(!finite,"FP16 overflow negative did not reject");r[@"wide_format_rejected"]=@YES;domains[@"wide"]=@{@"finite":@NO,@"rejected_before_down":@YES};}
                else{auto original=dense_gpu::projection_range(x,weights,rb,rb+n,cb,cb+k,32,true);auto mixed=mixed_hidden_lab::down(h,weights,rb,rb+n,cb,cb+k);mx::eval(original,mixed);
                    auto rounded=mx::astype(h,mx::bfloat16);auto reference=dense_gpu::projection_range(rounded,weights,rb,rb+n,cb,cb+k,32,true);mx::eval(reference);
                    payload.emplace(std::string(tag)+"_input",x);payload.emplace(std::string(tag)+"_half",h);payload.emplace(std::string(tag)+"_original",original);
                    payload.emplace(std::string(tag)+"_mixed",mixed);payload.emplace(std::string(tag)+"_rounded",reference);
                    domains[ns(tag)]=@{@"finite":@(finite),@"vs_original":quality(original,mixed),@"vs_rounded":quality(reference,mixed)};}}
            auto changed=weights*Tensor(.875f,mx::bfloat16)+Tensor(.0009765625f,mx::bfloat16);mx::eval(changed);auto c=mixed_hidden_lab::down(halves[0],changed,rb,rb+n,cb,cb+k);mx::eval(c);
            auto restored=run(2,0);mx::eval(restored);require(hash(c)!=expected[2][0]&&hash(restored)==expected[2][0],"mixed W A/B/A or independent output failed");}
        NSMutableArray*samples=[NSMutableArray array];r[@"samples"]=samples;std::array<std::optional<Tensor>,4>last;
        for(int rep=-2;rep<repeats;++rep)for(int j=0;j<4;++j){const int arm=(j+rep+4)%4,v=bool(rep&1);if(scope)scope(100+(rep+2)*4+j);auto start=Clock::now();auto value=run(arm,v);
            require(!value.is_available(),"mixed down reused materialized output");mx::eval(value);const auto wall=std::chrono::duration<double>(Clock::now()-start).count();last[arm]=value;
            [samples addObject:@{@"arm":@(names[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(v)),@"seconds":@(wall)}];}
        if(scope)scope(-3);for(int arm=0;arm<4;++arm)require(last[arm]&&hash(*last[arm])==expected[arm][1],"last mixed down output not restored");
        require(hash(weights)==source_w_hash&&hash(inputs[0])==input_hash[0]&&hash(inputs[1])==input_hash[1],"mixed down modified source weights/hidden");
        mx::save_safetensors((out/"arrays.safetensors").string(),payload);r[@"m"]=@(m);r[@"k"]=@(k);r[@"n"]=@(n);r[@"physical_weight_pitch"]=@(pitch);r[@"row_begin"]=@(rb);r[@"column_begin"]=@(cb);
        r[@"source_weight_sha256"]=ns(source_w_hash);r[@"source_unchanged"]=@YES;r[@"rejected"]=@(rejected);r[@"checks"]=checks;r[@"status"]=@"complete";
    }catch(const std::exception&e){r[@"status"]=@"failed";r[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(std::filesystem::is_directory(out))write(out/"report.json",r);return[r[@"status"]isEqual:@"complete"]?0:1;
}}
