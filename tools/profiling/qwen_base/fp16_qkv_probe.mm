#include "fp16_qkv.hpp"
#include "../../../native/models/qwen21/runtime_private_qkv.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

namespace {
using namespace tc;namespace lab=tc::fp16_qkv_lab;
NSString*ns(const std::string&s){return[NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor&a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
void write(const std::filesystem::path&p,NSDictionary*r){std::ofstream f(p);f<<json(r)<<'\n';require(bool(f),"FP16 QKV report write failed");}
}
extern "C" __attribute__((visibility("default")))
int tc_fp16_qkv_probe_v4(const char*model,const char*capture,const char*output,int repeats,int single_only,int compare_prepack,void(*scope)(int)){@autoreleasepool{
    const std::vector<const char*>names=compare_prepack?std::vector<const char*>{"gpu","native-w8","fp16-single","fp16-prepack"}:
        single_only?std::vector<const char*>{"gpu","native-w8","fp16-single"}:std::vector<const char*>{"gpu","native-w8","fp16-tiled","fp16-single"};const int arms=int(names.size());
    const std::filesystem::path out(output?output:"");NSMutableDictionary*r=[@{@"schema":@"tc-private-fp16-qkv-screen-v4",@"status":@"starting",@"complete_request":@NO,
        @"single_only":@(bool(single_only)),@"compare_prepack":@(bool(compare_prepack)),
        @"image_export":@NO,@"frozen_real_input":@YES,@"repeats":@(repeats),@"warmups":@2,@"extra_owned_weight_bytes":@0,
        @"cpp_sha256":@TC_FP16_QKV_CPP_SHA,@"header_sha256":@TC_FP16_QKV_HEADER_SHA,@"compiled_native_id":@TC_FP16_QKV_NATIVE_ID,
        @"scope":@"frozen actual base40 first-decode normalized input and original W; GPU vs current native W8 vs fixed512 FP16 full QKV; no new image or physical-ISA claim"}mutableCopy];
    try{require(model&&capture&&output&&(single_only==0||single_only==1)&&(compare_prepack==0||compare_prepack==1)&&(!compare_prepack||single_only)&&
            (repeats==16||repeats==32)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"bounded fresh FP16 QKV screen required");
        char*raw=tc_runtime_build_identity();require(raw,"native identity missing");const std::string id(raw);tc_string_free(raw);r[@"loaded_runtime_id"]=ns(id);require(id==TC_FP16_QKV_NATIVE_ID,"FP16 QKV helper/native ABI mismatch");configure_streams();
        auto all=mx::load_safetensors(std::string(model)+"/diffusion_models/qwen_image_2.1_bf16.safetensors");std::vector<std::array<Tensor,3>>weights;NSMutableDictionary*source=[NSMutableDictionary dictionary];
        for(int l=0;l<2;++l){const auto p="transformer_blocks."+std::to_string(l)+".attn.to_";weights.push_back({all.first.at(p+"q.weight"),all.first.at(p+"k.weight"),all.first.at(p+"v.weight")});
            for(int i=0;i<3;++i){mx::eval(weights[l][i]);source[ns(p+std::string(i==0?"q":i==1?"k":"v")+".weight")]=ns(hash(weights[l][i]));}}
        auto captured=mx::load_safetensors(capture);auto xa=captured.first.at("normalized");require(xa.shape()==mx::Shape{1,1024,4096}&&xa.dtype()==mx::bfloat16,"complete actual normalized input required");
        auto xb=xa*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16);mx::eval(xa,xb);const auto original_hash=hash(xa);
        auto gpu=mx::compile([](const std::vector<Tensor>&a){return std::vector<Tensor>{mx::concatenate({mx::matmul(a[0],mx::transpose(a[1])),mx::matmul(a[0],mx::transpose(a[2])),mx::matmul(a[0],mx::transpose(a[3]))},-1)};});
        if(scope)scope(0);std::atomic<bool>cancelled{false};qwen21::RuntimePrivateQkv w8(512ull<<20,cancelled);require(w8.available(),w8.metrics().failure_reason);w8.begin_request("original-bf16-frozen-real-input",false);
        lab::Graph f16{bool(single_only)};r[@"fp16_program_key"]=f16.program?ns(f16.program->cache_key()):@"";r[@"fp16_single_program_key"]=ns(f16.single->cache_key());r[@"w8_program_key"]=ns(w8.metrics().program_key);r[@"fp16_slot_bytes"]=@(f16.slot_bytes());r[@"w8_slot_bytes"]=@(w8.metrics().slot_bytes);
        r[@"fp16_setup_seconds"]=@(f16.setup_seconds);std::unordered_map<std::string,Tensor>payload{{"input_a",xa},{"input_b",xb}};
        std::array<std::array<std::string,3>,4>anchors;
        auto run=[&](int arm,int variant){const int l=variant==2?1:0;const auto&x=variant==1?xb:xa;auto head=[&](const Tensor&v){return gpu({v,weights[l][0],weights[l][1],weights[l][2]})[0];};
            if(!arm){auto y=head(x);mx::eval(y);return y;}if(arm==1){require(w8.plan(1024),"native W8 declined fixed target rows");w8.stage(l,1024,weights[l]);return w8.run(l,x,head,cancelled);}
            f16.stage(weights[l]);return f16.run(x,head,single_only||arm==3,compare_prepack&&arm==3).value;};
        for(int phase=0;phase<4;++phase)for(int arm=0;arm<arms;++arm){const int variant=phase==3?0:phase;if(scope)scope(1+phase*arms+arm);auto value=run(arm,variant);
            if(phase<3){anchors[arm][variant]=hash(value);payload.emplace(std::string(names[arm])+"_"+std::string(variant==0?"a":variant==1?"b":"w"),value);}
            else require(hash(value)==anchors[arm][0],"changed input/W/restore changed original raw QKV");}
        for(int arm=0;arm<arms;++arm)require(anchors[arm][0]!=anchors[arm][1]&&anchors[arm][0]!=anchors[arm][2],"changed actual input/W returned stale projection");
        NSMutableArray*samples=[NSMutableArray array];r[@"samples"]=samples;std::array<std::optional<Tensor>,4>last;
        for(int rep=-2;rep<repeats;++rep)for(int j=0;j<arms;++j){const int arm=(j+rep+arms)%arms,variant=bool(rep&1);if(scope)scope(100+(rep+2)*arms+j);const auto t=Clock::now();
            auto value=run(arm,variant);const auto wall=lab::Graph::seconds(t);last[arm]=value;
            [samples addObject:@{@"arm":@(names[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(variant)),@"seconds":@(wall)}];}
        if(scope)scope(-3);for(int arm=0;arm<arms;++arm)require(last[arm]&&hash(*last[arm])==anchors[arm][1],"last completed raw projection changed");
        const auto&m=w8.metrics();require(m.predictions==uint64_t(repeats+6)&&m.completed_blocks==m.predictions&&m.validation_calls==1&&m.validation_passed&&!m.failed&&m.fallback_blocks==0&&
            f16.predictions==uint64_t(arms-2)*m.predictions&&f16.stages==f16.predictions&&f16.tiled_predictions==(single_only?0:m.predictions)&&
            f16.single_predictions==uint64_t(compare_prepack?2:1)*m.predictions,"actual native W8/FP16 successful counts differ");
        require(hash(xa)==original_hash,"frozen actual input changed");for(int l=0;l<2;++l)for(int i=0;i<3;++i){const auto name="transformer_blocks."+std::to_string(l)+".attn.to_"+std::string(i==0?"q":i==1?"k":"v")+".weight";
            require([source[ns(name)]isEqual:ns(hash(weights[l][i]))],"original BF16 QKV weights changed");}
        mx::save_safetensors((out/"arrays.safetensors").string(),payload);r[@"source_arrays_sha256"]=source;r[@"input_sha256"]=ns(original_hash);
        r[@"w8_predictions"]=@(m.predictions);r[@"fp16_predictions"]=@(f16.predictions);r[@"w8_validation_calls"]=@(m.validation_calls);r[@"w8_validation_relative_l2"]=@(m.validation_relative_l2);
        r[@"fp16_tiled_predictions"]=@(f16.tiled_predictions);r[@"fp16_single_predictions"]=@(f16.single_predictions);
        r[@"source_unchanged"]=@YES;r[@"restored_outputs_verified"]=@YES;r[@"status"]=@"complete";
    }catch(const std::exception&e){r[@"status"]=@"failed";r[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(std::filesystem::is_directory(out))write(out/"report.json",r);return[r[@"status"]isEqual:@"complete"]?0:1;
}}
