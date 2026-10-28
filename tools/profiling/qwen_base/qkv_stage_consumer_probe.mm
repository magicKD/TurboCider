#include "qkv_stage_batch.hpp"
#include "qkv_ane.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

namespace {
using namespace tc;namespace lab=tc::qkv_stage_lab;namespace qlab=tc::qwen21::base_qkv_lab;namespace pa=ane::private_api;
const std::array<const char*,4>names{"native-three-cb","batch-three-codes","fused-three-dispatch","fused-one-dispatch"};
NSString*ns(const std::string&s){return[NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor&a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
void write(const std::filesystem::path&p,NSDictionary*r){std::ofstream f(p);f<<json(r)<<'\n';require(bool(f),"stage consumer report write failed");}
struct Graph {
    lab::Stager stage;
    pa::Surface x{stage.device,4096,512,pa::Element::I8},tx{stage.device,1,512,pa::Element::FP16},y{stage.device,12288,512,pa::Element::FP16};
    std::unique_ptr<pa::Program>program;uint64_t timeline=0,predictions=0;double setup_seconds;
    Graph(){const auto t=Clock::now();program=std::make_unique<pa::Program>(stage.device,pa::w8_matmul_program({ane::Kind::Matmul,512,4096,12288,4096,512,false}),
        std::span<const uint8_t>{},pa::default_cache_directory());setup_seconds=lab::Stager::seconds(t);
        auto names=program->inputs();std::sort(names.begin(),names.end());require(names==std::vector<std::string>{"w","x"}&&program->outputs()==std::vector<std::string>{"y"},"fixed QKV ABI differs");}
    template<class GPU>std::pair<Tensor,std::array<double,3>>run(const Tensor&input,const std::array<Tensor,3>&weights,int arm,GPU&&gpu){
        auto t=Clock::now();if(arm==0)stage.native(weights);else{stage.submit(weights,arm);stage.finish();}const auto weight=lab::Stager::seconds(t);
        t=Clock::now();auto v=qlab::device_view(input,512,4096,512);
        ane::DeviceWeightView source{v.buffer,v.buffer_bytes,v.offset_bytes,v.row_stride_bytes,512,4096,ane::DeviceWeightEncoding::Dense,ane::DType::BF16,32,{},{},v.owner,v.allocation_identity};
        auto a=stage.device.stage_w8(source,{0,512,0,4096,128,20260930,true},x,tx);const auto ac=a.finish();require(ac.ok&&!ac.timed_out&&a.validation_flags()==0,"QKV A8 producer failed");
        const auto activation=lab::Stager::seconds(t);auto allocation=mx::allocator::malloc(size_t(512)*12288*2);require(allocation.ptr(),"owned QKV tail allocation failed");
        Tensor tail(allocation,{1,512,12288},mx::bfloat16);const auto ready=++timeline,done=++timeline;
        auto restore=stage.device.prepare_transfer({},{{y,qlab::device_view(tail,512,12288),0,ane::DType::BF16,1.f,stage.scales,tx}},ready,done);
        std::pair<std::string,pa::Surface>ins[]{{"x",x},{"w",stage.codes}},outs[]{{"y",y}};
        auto ticket=program->enqueue(ins,outs,ready,done,restore.failure_callback());std::optional<Tensor>head;
        try{restore.submit();head=gpu(slice_axis(input,1,0,512));require(!head->is_available(),"GPU head reused an old materialized result");mx::async_eval(*head);}
        catch(...){try{(void)ticket.finish();(void)restore.finish();}catch(...){}throw;}
        t=Clock::now();const auto evaluated=ticket.finish(),restored=restore.finish();
        require(evaluated.ok&&evaluated.physical_complete&&!evaluated.retired&&!evaluated.timed_out&&restored.ok&&!restored.timed_out&&restore.validation_flags()==0,"QKV predict/restore physically incomplete or invalid");
        const auto join=lab::Stager::seconds(t);auto result=mx::concatenate({*head,tail},1);mx::eval(result);++predictions;
        return{result,{weight,activation,join}};
    }
};
}
extern "C" __attribute__((visibility("default")))
int tc_qkv_stage_consumer_probe(const char*model,const char*output,int repeats,void(*scope)(int)){@autoreleasepool{
    const std::filesystem::path out(output?output:"");NSMutableDictionary*r=[@{@"schema":@"tc-qkv-stage-consumer-v1",@"status":@"starting",
        @"cpp_sha256":@TC_STAGE_CONSUMER_CPP_SHA,@"header_sha256":@TC_STAGE_CONSUMER_HEADER_SHA,@"compiled_native_id":@TC_STAGE_CONSUMER_NATIVE_ID,
        @"complete_request":@NO,@"image_export":@NO,@"synthetic_inputs":@YES,@"repeats":@(repeats),@"warmups":@2,
        @"extra_owned_weight_bytes":@0,@"scope":@"real QKV weights, synthetic target1024 input, four W producers one fixed ANE graph; completed full projection, not image or FFN/lookahead qualification"}mutableCopy];
    try{require(model&&output&&(repeats==32||repeats==64)&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"bounded fresh projection fixture required");
        char*raw=tc_runtime_build_identity();require(raw,"native identity missing");const std::string id(raw);tc_string_free(raw);r[@"loaded_runtime_id"]=ns(id);
        require(id==TC_STAGE_CONSUMER_NATIVE_ID,"stage consumer/native ABI mismatch");configure_streams();auto all=mx::load_safetensors(std::string(model)+"/diffusion_models/qwen_image_2.1_bf16.safetensors");
        std::vector<std::array<Tensor,3>>weights;NSMutableDictionary*hashes=[NSMutableDictionary dictionary];
        for(int l=0;l<2;++l){const auto p="transformer_blocks."+std::to_string(l)+".attn.to_";weights.push_back({all.first.at(p+"q.weight"),all.first.at(p+"k.weight"),all.first.at(p+"v.weight")});
            for(int j=0;j<3;++j){mx::eval(weights[l][j]);hashes[ns(p+std::string(j==0?"q":j==1?"k":"v")+".weight")]=ns(hash(weights[l][j]));}}
        auto xa=mx::reshape(mx::astype((mx::arange(1024*4096,mx::int32)%127-63),mx::bfloat16)/Tensor(128.f,mx::bfloat16),{1,1024,4096});
        auto xb=xa*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16);mx::eval(xa,xb);
        auto gpu=mx::compile([](const std::vector<Tensor>&a){return std::vector<Tensor>{mx::concatenate({mx::matmul(a[0],mx::transpose(a[1])),mx::matmul(a[0],mx::transpose(a[2])),mx::matmul(a[0],mx::transpose(a[3]))},-1)};});
        if(scope)scope(0);Graph g;g.stage.prime(weights[0]);g.stage.prime(weights[1]);const auto key=g.program->cache_key();
        r[@"program_key"]=ns(key);r[@"program_setup_seconds"]=@(g.setup_seconds);r[@"stager_setup_seconds"]=@(g.stage.setup_seconds);r[@"metadata_prime_seconds"]=@(g.stage.prime_seconds);
        std::unordered_map<std::string,Tensor>payload{{"input_a",xa},{"input_b",xb}};std::array<std::string,3>anchors;
        auto run=[&](int arm,int variant){const int l=variant==2?1:0;const auto&x=variant==1?xb:xa;
            return g.run(x,weights[l],arm,[&](const Tensor&v){return gpu({v,weights[l][0],weights[l][1],weights[l][2]})[0];});};
        for(int phase=0;phase<4;++phase){const int variant=phase==3?0:phase,l=variant==2?1:0;const auto vname=std::string(variant==0?"a":variant==1?"b":"w");
            if(phase<3){auto reference=gpu({variant==1?xb:xa,weights[l][0],weights[l][1],weights[l][2]})[0];mx::eval(reference);payload.emplace("gpu_"+vname,reference);}
            for(int arm=0;arm<4;++arm){if(scope)scope(1+phase*4+arm);auto value=run(arm,variant).first;
                if(!arm&&phase<3)anchors[variant]=hash(value);else require(hash(value)==anchors[variant],"new W stager changed complete QKV projection bits");
                if(phase<3)payload.emplace(std::string(names[arm])+"_"+vname,value);}}
        require(anchors[0]!=anchors[1]&&anchors[0]!=anchors[2],"changed real W/input returned stale raw QKV");
        NSMutableArray*samples=[NSMutableArray array];r[@"samples"]=samples;std::array<std::optional<Tensor>,4>last;
        for(int rep=-2;rep<repeats;++rep)for(int j=0;j<4;++j){const int arm=(j+rep+4)%4,variant=bool(rep&1);if(scope)scope(100+(rep+2)*4+j);
            const auto t=Clock::now();auto value=run(arm,variant);const auto wall=lab::Stager::seconds(t);last[arm]=value.first;
            [samples addObject:@{@"arm":@(names[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(variant)),@"seconds":@(wall),
                @"weight_stage_seconds":@(value.second[0]),@"activation_seconds":@(value.second[1]),@"predict_restore_join_seconds":@(value.second[2])}];}
        if(scope)scope(-3);
        for(int arm=0;arm<4;++arm)require(last[arm]&&hash(*last[arm])==anchors[1],"last completed consumer did not restore alternate raw QKV");
        require(g.predictions==uint64_t(4*repeats+24)&&g.stage.calls==g.predictions+2&&g.stage.command_buffers==uint64_t(6*repeats+42)&&g.stage.compute_dispatches==uint64_t(16*repeats+114),"actual projection/staging counts differ");
        for(int l=0;l<2;++l)for(int j=0;j<3;++j){const auto name="transformer_blocks."+std::to_string(l)+".attn.to_"+std::string(j==0?"q":j==1?"k":"v")+".weight";require([hashes[ns(name)]isEqual:ns(hash(weights[l][j]))],"original QKV W changed");}
        mx::save_safetensors((out/"arrays.safetensors").string(),payload);r[@"source_arrays_sha256"]=hashes;r[@"predictions"]=@(g.predictions);r[@"stage_calls"]=@(g.stage.calls);
        r[@"stage_command_buffers"]=@(g.stage.command_buffers);r[@"stage_dispatches"]=@(g.stage.compute_dispatches);r[@"source_unchanged"]=@YES;r[@"incremental_bits_exact"]=@YES;r[@"status"]=@"complete";
    }catch(const std::exception&e){r[@"status"]=@"failed";r[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(scope)scope(-3);if(std::filesystem::is_directory(out))write(out/"report.json",r);return[r[@"status"]isEqual:@"complete"]?0:1;
}}
