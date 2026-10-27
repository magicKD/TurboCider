#include "qkv_stage_batch.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

#ifndef TC_QKV_STAGE_NATIVE_ID
#error exact native C++ ABI binding required
#endif
namespace {
using namespace tc;namespace lab=tc::qkv_stage_lab;
const std::array<const char*,4> names{"native-three-cb","batch-three-codes","fused-three-dispatch","fused-one-dispatch"};
NSString*ns(const std::string&s){return[NSString stringWithUTF8String:s.c_str()];}
std::string hash(const void*p,size_t n){return memory_sha256_hex({static_cast<const char*>(p),n});}
std::string hash(const Tensor&a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);return hash(b.data<uint8_t>(),b.nbytes());}
void save(const std::filesystem::path&p,const void*data,size_t n){std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),n);require(bool(f),"stage array write failed");}
void write(const std::filesystem::path&p,NSDictionary*r){std::ofstream f(p);f<<json(r)<<'\n';require(bool(f),"stage report write failed");}
}
extern "C" __attribute__((visibility("default")))
int tc_qkv_stage_probe(const char*model,const char*output,int repeats,void(*scope)(int)){@autoreleasepool{
    const std::filesystem::path out(output?output:"");NSMutableDictionary*r=[@{@"schema":@"tc-qkv-w-stage-v1",@"status":@"starting",
        @"compiled_native_id":@TC_QKV_STAGE_NATIVE_ID,@"cpp_sha256":@TC_QKV_STAGE_CPP_SHA,@"header_sha256":@TC_QKV_STAGE_HEADER_SHA,
        @"complete_request":@NO,@"image_export":@NO,@"ane_predictions":@0,@"real_source_layers":@[@0,@1],
        @"repeats":@(repeats),@"warmups":@2,@"extra_owned_weight_bytes":@0,@"scale_cache_capacity":@2,
        @"checksum_policy":@"preflight-and-post-loop-only",@"scope":@"real original BF16 QKV W producers only; compact scales preprimed, no ANE/FFN/full40 image speed or physical overlap qualification"}mutableCopy];
    try{require(model&&output&&(repeats==32||repeats==64)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh bounded W stage fixture required");
        char*id=tc_runtime_build_identity();require(id,"runtime identity missing");const std::string identity(id);tc_string_free(id);
        r[@"loaded_runtime_id"]=ns(identity);require(identity==TC_QKV_STAGE_NATIVE_ID,"stage helper/native ABI mismatch");configure_streams();
        auto all=mx::load_safetensors(std::string(model)+"/diffusion_models/qwen_image_2.1_bf16.safetensors");
        std::vector<std::array<Tensor,3>> weights;NSMutableDictionary*source=[NSMutableDictionary dictionary];
        for(int l=0;l<2;++l){const auto p="transformer_blocks."+std::to_string(l)+".attn.to_";
            weights.push_back({all.first.at(p+"q.weight"),all.first.at(p+"k.weight"),all.first.at(p+"v.weight")});
            for(int j=0;j<3;++j){mx::eval(weights[l][j]);const auto name=p+std::string(j==0?"q":j==1?"k":"v")+".weight";source[ns(name)]=ns(hash(weights[l][j]));}}
        if(scope)scope(0);lab::Stager stage;stage.prime(weights[0]);stage.prime(weights[1]);
        r[@"setup_seconds"]=@(stage.setup_seconds);r[@"prime_seconds"]=@(stage.prime_seconds);r[@"metadata_bytes"]=@(stage.metadata_bytes());
        const size_t code_bytes=size_t(lab::N)*lab::H;std::array<std::vector<int8_t>,2>anchors;
        std::array<std::vector<uint16_t>,2>scale_anchors;NSMutableArray*arrays=[NSMutableArray array];
        auto logical_scales=[&]{std::vector<uint16_t>s(lab::N);for(int row=0;row<lab::N;++row)std::memcpy(&s[row],static_cast<const char*>(stage.scales.data())+size_t(row)*stage.scales.pitch(),2);return s;};
        auto run=[&](int arm,int l){if(arm==0)stage.native(weights[l]);else{stage.submit(weights[l],arm);require(stage.finish()==0,"candidate flags differ");}};
        for(int phase=0;phase<3;++phase)for(int arm=0;arm<4;++arm){const int l=phase==1?1:0;if(scope)scope(1+phase*4+arm);run(arm,l);
            const auto*s=static_cast<const int8_t*>(stage.codes.data());auto scales=logical_scales();
            if(arm==0&&phase<2){anchors[l].assign(s,s+code_bytes);scale_anchors[l]=scales;}
            else require(anchors[l].size()==code_bytes&&std::memcmp(s,anchors[l].data(),code_bytes)==0&&scales==scale_anchors[l],"batch/fused/source A-B-A full codes/scales differ");
            if(phase<2){const auto prefix="layer"+std::to_string(l)+"-"+names[arm];save(out/(prefix+".i8"),s,code_bytes);save(out/(prefix+".f16"),scales.data(),scales.size()*2);
                [arrays addObject:@{@"layer":@(l),@"arm":@(names[arm]),@"codes_file":ns(prefix+".i8"),@"scales_file":ns(prefix+".f16"),
                    @"codes_sha256":ns(hash(s,code_bytes)),@"scales_sha256":ns(hash(scales.data(),scales.size()*2))}];}}
        require(anchors[0]!=anchors[1]&&scale_anchors[0]!=scale_anchors[1],"changed real W returned stale codes/scales");
        // Reject missing source identity, unready/layout/dtype and concurrent
        // overwrite before any commit. A submitted job is always drained.
        if(scope)scope(50);
        int rejected=0;const auto calls_before=stage.calls;
        auto reject=[&](auto f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}require(threw,"invalid batch fixture accepted");++rejected;};
        reject([&]{stage.submit(weights[0],0);});reject([&]{stage.submit(weights[0],4);});
        auto unbound=weights[0];unbound[0]=mx::astype(unbound[0],mx::float32);mx::eval(unbound[0]);reject([&]{stage.submit(unbound,3);});
        auto new_id=weights[0];new_id[0]=mx::contiguous(new_id[0]+Tensor(0.0009765625f,mx::bfloat16));mx::eval(new_id[0]);reject([&]{stage.submit(new_id,3);});
        require(stage.calls==calls_before,"rejected source consumed a submission");
        stage.submit(weights[0],3);reject([&]{stage.submit(weights[1],3);});stage.finish();
        require(std::memcmp(stage.codes.data(),anchors[0].data(),code_bytes)==0,"claimed single-bank output changed after rejected overwrite");
        NSMutableArray*samples=[NSMutableArray array];r[@"samples"]=samples;
        for(int rep=-2;rep<repeats;++rep)for(int j=0;j<4;++j){const int arm=(j+rep+4)%4,l=bool(rep&1);if(scope)scope(100+(rep+2)*4+j);
            const auto before_cb=stage.command_buffers,before_dispatch=stage.compute_dispatches;const auto start=Clock::now();run(arm,l);const auto elapsed=lab::Stager::seconds(start);
            [samples addObject:@{@"arm":@(names[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"layer":@(l),@"seconds":@(elapsed),
                @"command_buffers":@(stage.command_buffers-before_cb),@"compute_dispatches":@(stage.compute_dispatches-before_dispatch)}];}
        if(scope)scope(-3);
        require(std::memcmp(stage.codes.data(),anchors[1].data(),code_bytes)==0&&logical_scales()==scale_anchors[1],"last completed layer1 bank changed full bytes");
        for(int l=0;l<2;++l)for(int j=0;j<3;++j){const auto name="transformer_blocks."+std::to_string(l)+".attn.to_"+std::string(j==0?"q":j==1?"k":"v")+".weight";
            require([source[ns(name)]isEqual:ns(hash(weights[l][j]))],"original BF16 payload changed");}
        require(stage.calls==uint64_t(4*repeats+23)&&stage.command_buffers==uint64_t(6*repeats+37)&&stage.compute_dispatches==uint64_t(16*repeats+99)&&
            stage.native_scale_misses==6&&stage.native_scale_hits==uint64_t(3*(repeats+5)),"actual W call/CB/dispatch/cache accounting differs");
        r[@"arrays"]=arrays;r[@"source_arrays_sha256"]=source;r[@"calls"]=@(stage.calls);r[@"command_buffers"]=@(stage.command_buffers);r[@"compute_dispatches"]=@(stage.compute_dispatches);
        r[@"native_calls"]=@(stage.native_calls);r[@"custom_calls"]=@(stage.custom_calls);r[@"native_scale_misses"]=@(stage.native_scale_misses);r[@"native_scale_hits"]=@(stage.native_scale_hits);
        r[@"rejected"]=@(rejected);r[@"source_unchanged"]=@YES;r[@"all_stage_bytes_exact"]=@YES;r[@"status"]=@"complete";
    }catch(const std::exception&e){r[@"status"]=@"failed";r[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(std::filesystem::is_directory(out))write(out/"report.json",r);return[r[@"status"]isEqual:@"complete"]?0:1;
}}
