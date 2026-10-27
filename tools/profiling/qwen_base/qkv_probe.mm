#include "qkv_ane.hpp"
#include "../../../native/models/qwen21/pipeline.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <map>

#ifndef TC_BASE_QKV_CPP_SHA
#define TC_BASE_QKV_CPP_SHA "unbound"
#define TC_BASE_QKV_HEADER_SHA "unbound"
#endif

namespace tc::qwen21::base_qkv_lab {
using Fn=DiagnosticGpuBlock;
using Scope=void(*)(int);
NSString *ns(const std::string &s){return [NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor &x){auto bytes=mx::view(mx::contiguous(x),mx::uint8);mx::eval(bytes);
    return memory_sha256_hex({reinterpret_cast<const char*>(bytes.data<uint8_t>()),bytes.nbytes()});}
void write(const std::filesystem::path &path,NSDictionary *value){std::ofstream stream(path);stream<<json(value)<<'\n';require(bool(stream),"QKV report write failed");}
NSDictionary *difference(const Tensor &a,const Tensor &b){require(a.shape()==b.shape(),"comparison shape differs");
    auto x=mx::contiguous(mx::astype(a,mx::float32)),y=mx::contiguous(mx::astype(b,mx::float32));mx::eval(x,y);
    double energy=0,error=0,other=0,dot=0,max_abs=0;uint64_t changed=0;
    for(size_t i=0;i<x.size();++i){double v=x.data<float>()[i],w=y.data<float>()[i],d=v-w;
        require(std::isfinite(v)&&std::isfinite(w),"nonfinite QKV/consumer");energy+=v*v;other+=w*w;error+=d*d;dot+=v*w;max_abs=std::max(max_abs,std::abs(d));changed+=v!=w;}
    return @{@"relative_l2":@(std::sqrt(error/std::max(energy,1e-30))),@"max_abs":@(max_abs),@"changed":@(changed),
        @"cosine":@(dot/std::sqrt(std::max(energy*other,1e-60))),@"elements":@(x.size())};
}
Tensor rope(const Tensor &x,const Tensor &c,const Tensor &s){auto shape=x.shape(),pairs=shape;pairs.back()/=2;pairs.push_back(2);
    auto p=mx::reshape(mx::astype(x,mx::float32),pairs),a=mx::squeeze(slice_axis(p,-1,0,1),-1),b=mx::squeeze(slice_axis(p,-1,1,2),-1);
    auto co=mx::reshape(c,{1,1,shape[2],64}),si=mx::reshape(s,{1,1,shape[2],64});
    return mx::astype(mx::reshape(mx::stack({a*co-b*si,a*si+b*co},-1),shape),x.dtype());
}
void measure(const DiagnosticBlockObservation &o,int chunk,bool tiled,int repeats,const std::filesystem::path &out,
             NSMutableDictionary *report,Scope scope){
    require(o.layer==0&&o.steady&&o.arguments.size()==9&&o.arguments[0].shape()==mx::Shape{1,rows,hidden}&&o.outputs.size()==1,
            "actual complete base block0 decode required");
    const auto &w=o.weights;require(!w.has_runtime_loras(),"base QKV must not carry adapters");
    std::array<Tensor,3> weights{w.at("transformer_blocks.0.attn.to_q.weight"),w.at("transformer_blocks.0.attn.to_k.weight"),w.at("transformer_blocks.0.attn.to_v.weight")};
    auto second=weights;
    for(int i=0;i<3;++i){const auto leaf=i==0?"to_q":i==1?"to_k":"to_v";const auto p=std::string("transformer_blocks.0.attn.")+leaf;
        require(weights[i].shape()==mx::Shape{hidden,hidden}&&weights[i].dtype()==mx::bfloat16&& !w.quantized(p)&&!w.convrot(p)&&!w.nvfp4(p)&&!w.has(p+".bias"),"original dense unbiased BF16 QKV required");
        second[i]=w.at(std::string("transformer_blocks.1.attn.")+leaf+".weight");}
    std::map<std::string,std::string> before;
    for(int i=0;i<3;++i){before["block0-"+std::to_string(i)]=hash(weights[i]);before["block1-"+std::to_string(i)]=hash(second[i]);}
    std::vector<std::string> input_hashes;for(const auto &a:o.arguments)input_hashes.push_back(hash(a));
    const auto original_hash=hash(o.outputs[0]);
    auto normalize=mx::compile(Fn([](const auto &a){return std::vector<Tensor>{mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1])};}));
    auto project=mx::compile(Fn([](const auto &a){return std::vector<Tensor>{mx::concatenate({mx::matmul(a[0],mx::transpose(a[1])),
        mx::matmul(a[0],mx::transpose(a[2])),mx::matmul(a[0],mx::transpose(a[3]))},-1)};}));
    auto finish=mx::compile(Fn([&w](const auto &a){const std::string p="transformer_blocks.0.";
        auto q=heads(slice_axis(a[0],-1,0,hidden),32,128),k=heads(slice_axis(a[0],-1,hidden,2*hidden),32,128),v=heads(slice_axis(a[0],-1,2*hidden,3*hidden),32,128);
        q=rope(mx::fast::rms_norm(q,w.at(p+"attn.norm_q.weight"),1e-6f),a[5],a[6]);
        k=rope(mx::fast::rms_norm(k,w.at(p+"attn.norm_k.weight"),1e-6f),a[5],a[6]);
        auto att=attend(q,mx::concatenate({a[7],k},2),mx::concatenate({a[8],v},2));
        auto h=a[1]+mx::tanh(a[2])*linear(att,w,p+"attn.to_out.0");
        auto x=mx::fast::layer_norm(h,{},{},1e-6f)*(Tensor(1.f,h.dtype())+a[3]);
        auto gu=linear(x,w,p+"img_mlp.gate_up"),f=silu(slice_axis(gu,-1,0,12288))*slice_axis(gu,-1,12288,24576);
        return std::vector<Tensor>{h+mx::tanh(a[4])*linear(f,w,p+"img_mlp.out")};}));
    auto consume=[&](const Tensor &raw,const std::vector<Tensor> &a){return finish({raw,a[0],a[2],a[3],a[4],a[5],a[6],a[7],a[8]})[0];};
    auto original=o.arguments,alternate=o.arguments;
    alternate[0]=mx::astype(alternate[0]*Tensor(.984375f,mx::bfloat16)+Tensor(.00390625f,mx::bfloat16),mx::bfloat16);mx::eval(alternate[0]);
    auto x=normalize({original[0],original[1]})[0],xb=normalize({alternate[0],alternate[1]})[0];mx::eval(x,xb);
    auto raw=project({x,weights[0],weights[1],weights[2]})[0],rawb=project({xb,weights[0],weights[1],weights[2]})[0];mx::eval(raw,rawb);
    auto reference=o.gpu_block(original)[0],refb=o.gpu_block(alternate)[0],staged=consume(raw,original);mx::eval(reference,refb,staged);
    require(hash(reference)==original_hash,"ordinary complete GPU block replay changed");
    auto staged_quality=difference(reference,staged);require([staged_quality[@"relative_l2"] doubleValue]<.001,"staged GPU consumer is not a close control");
    if(scope)scope(0);
    FixedMatmul graph(chunk,tiled,out/"private-cache");
    report[@"setup_seconds"]=@(graph.setup_seconds);report[@"compiled_now"]=@(graph.compiled_now);report[@"slot_bytes"]=@(graph.slot_bytes);
    report[@"cache_key"]=ns(graph.cache_key);report[@"mil_sha256"]=ns(memory_sha256_hex(graph.mil));
    {std::ofstream mil(out/"program.mil");mil<<graph.mil;require(bool(mil),"MIL save failed");}
    auto head=[&](const Tensor &input){return project({input,weights[0],weights[1],weights[2]})[0];};
    if(scope)scope(1);
    auto first=graph.run(x,weights,head,true).value;mx::eval(first);const auto first_hash=hash(first);
    auto different=graph.run(xb,weights,head,false).value;mx::eval(different);
    require(hash(different)!=first_hash&&hash(first)==first_hash,"fresh activation or independent output ownership failed");
    auto restored=graph.run(x,weights,head,false).value;mx::eval(restored);require(hash(restored)==first_hash,"activation A/B/A failed");
    auto other_head=[&](const Tensor &input){return project({input,second[0],second[1],second[2]})[0];};
    auto refw=project({x,second[0],second[1],second[2]})[0],other=graph.run(x,second,other_head,true).value;mx::eval(refw,other);
    auto restored_w=graph.run(x,weights,head,true).value;mx::eval(restored_w);require(hash(restored_w)==first_hash&&hash(first)==first_hash,"same-graph weight A/B/A or old output changed");
    auto block=consume(first,original);mx::eval(block);
    auto rawq=difference(raw,first),blockq=difference(reference,block),altq=difference(rawb,different),weightq=difference(refw,other);
    report[@"raw_quality"]=rawq;report[@"complete_block_quality"]=blockq;report[@"staged_gpu_quality"]=staged_quality;
    report[@"alternate_activation_quality"]=altq;report[@"alternate_weight_quality"]=weightq;
    require([rawq[@"relative_l2"] doubleValue]<.02&&[altq[@"relative_l2"] doubleValue]<.02&&[weightq[@"relative_l2"] doubleValue]<.02&&
        [blockq[@"relative_l2"] doubleValue]<.03&&[blockq[@"cosine"] doubleValue]>.999,"QKV component numeric screen failed");
    report[@"fresh_inputs_weights_outputs_verified"]=@YES;
    const std::array<const char*,4> names{"gpu-original","gpu-staged","private-prestaged","private-restaged"};
    NSMutableArray *samples=[NSMutableArray array];report[@"samples"]=samples;
    std::array<Tensor,4> last{reference,staged,block,block};
    for(int rep=-2;rep<repeats;++rep)for(int j=0;j<4;++j){const int arm=(j+rep+4)%4;
        if(scope)scope(2+(rep+2)*4+j);
        const auto &a=(rep&1)?alternate:original;const auto started=Clock::now();
        Projection p{Tensor(0.f)};Tensor result(0.f);
        if(arm==0)result=o.gpu_block(a)[0];
        else {auto input=normalize({a[0],a[1]})[0];mx::eval(input);
            if(arm==1)result=consume(project({input,weights[0],weights[1],weights[2]})[0],a);
            else {p=graph.run(input,weights,head,arm==3);result=consume(p.value,a);}}
        require(!result.is_available(),"complete consumer unexpectedly reused a materialized output");mx::eval(result);
        const auto wall=FixedMatmul::seconds(started);last[arm]=result;
        [samples addObject:@{@"arm":@(names[arm]),@"rep":@(rep),@"warmup":@(rep<0),@"input_variant":@(bool(rep&1)),@"seconds":@(wall),
            @"weight_stage_seconds":@(p.stage_seconds),@"activation_seconds":@(p.activation_seconds),@"submit_seconds":@(p.submit_seconds),@"join_seconds":@(p.join_seconds)}];
    }
    if(scope)scope(-3);
    const uint64_t expected=5+uint64_t(repeats+2)*2;
    require(graph.predictions==expected&&graph.stages==3+uint64_t(repeats+2),"actual successful predictions/stages differ");
    report[@"predictions"]=@(graph.predictions);report[@"weight_stages"]=@(graph.stages);
    for(int i=0;i<3;++i)require(hash(weights[i])==before.at("block0-"+std::to_string(i))&&hash(second[i])==before.at("block1-"+std::to_string(i)),"original QKV weights changed");
    for(size_t i=0;i<original.size();++i)require(hash(original[i])==input_hashes[i],"borrowed ordinary block input changed");
    require(hash(o.outputs[0])==original_hash,"borrowed block output changed");
    NSMutableDictionary *hashes=[NSMutableDictionary dictionary];for(const auto &[key,value]:before)hashes[ns(key)]=ns(value);
    report[@"actual_qkv_weights_sha256"]=hashes;report[@"prefix_tokens"]=@(o.prefix_tokens);report[@"borrowed_values_unchanged"]=@YES;
    mx::save_safetensors((out/"arrays.safetensors").string(),{{"normalized",x},{"raw_reference",raw},{"raw_private",first},
        {"block_reference",reference},{"block_staged",staged},{"block_private",block},
        {"alternate_raw_reference",rawb},{"alternate_raw_private",different},{"other_weight_reference",refw},{"other_weight_private",other},
        {"last_gpu_original",last[0]},{"last_gpu_staged",last[1]},{"last_private_prestaged",last[2]},{"last_private_restaged",last[3]}});
    report[@"status"]=@"complete";write(out/"report.json",report);
}
} // namespace tc::qwen21::base_qkv_lab

extern "C" __attribute__((visibility("default")))
int tc_qwen_base_qkv_probe(const char *model,const char *reference,const char *output,int chunk,int tiled,int repeats,void(*scope)(int)) {
 @autoreleasepool {
    using namespace tc;using namespace tc::qwen21::base_qkv_lab;
    const std::filesystem::path out(output);
    NSMutableDictionary *report=[@{@"schema":@"tc-qwen-base-private-qkv-v1",@"status":@"starting",@"complete_request":@NO,
        @"chunk":@(chunk),@"tiled":@(bool(tiled)),@"repeats":@(repeats),@"warmups":@2,@"image_export":@NO,
        @"cpp_sha256":@TC_BASE_QKV_CPP_SHA,@"header_sha256":@TC_BASE_QKV_HEADER_SHA,
        @"scope":@"real base layer0 decode, full GPU-block consumers; FFN GPU, no adapters; not whole request or hardware overlap/INT8 proof"} mutableCopy];
    try {
        require(model&&reference&&output&&(chunk==256||chunk==512)&&repeats>=4&&repeats<=20&&
            std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"bounded geometry and fresh output required");
        char *id=tc_runtime_build_identity();require(id,"runtime identity missing");report[@"loaded_runtime_id"]=ns(id);tc_string_free(id);
        configure_streams();qwen21::Session session(model);
        Request r;r.model="qwen-image-2.1";r.operation=std::string_view(reference)=="-"?"image.generate":"image.edit";
        r.width=r.height=512;r.steps=40;r.frames=1;r.audio=false;r.seed=73;r.execution="gpu";r.allow_approximation=false;r.residency="resident";
        r.qwen21_dit_cache="off";r.qwen21_dit_cache_explicit=true;r.qwen21_reference_size=1024;r.output=(out/"not-exported.png").string();
        r.prompt=r.operation=="image.generate"?"A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.":
            "Change the teapot to bright red. Preserve its shape, handle, spout, table and background.";
        if(r.operation=="image.edit")r.inputs.push_back({.kind="image",.role="reference",.path=reference});
        report[@"actual_source_request"]=@{@"model":ns(r.model),@"operation":ns(r.operation),@"width":@(r.width),@"height":@(r.height),
            @"steps":@(r.steps),@"frames":@(r.frames),@"audio":@(r.audio),@"seed":@(r.seed),@"execution":ns(r.execution),
            @"residency":ns(r.residency),@"lora_count":@(r.loras.size()),@"reference_count":@(r.inputs.size()),
            @"reference_size":@(r.qwen21_reference_size),@"allow_approximation":@(r.allow_approximation)};
        write(out/"report.json",report);
        session.set_diagnostic_block_observer([&](const qwen21::DiagnosticBlockObservation &o){
            if(o.layer||!o.steady)return;measure(o,chunk,tiled,repeats,out,report,scope);throw qwen21::DiagnosticBlockReplayComplete();});
        std::atomic<bool> cancelled{false};
        try{(void)session.generate(r,[](const auto&,int,int){},cancelled);throw std::runtime_error("bounded QKV capture did not terminate");}
        catch(const qwen21::DiagnosticBlockReplayComplete&){require([report[@"status"] isEqual:@"complete"],"capture failed before completion");}
        return 0;
    } catch(const std::exception &e) {report[@"status"]=@"failed";report[@"error"]=ns(e.what());
        if(std::filesystem::is_directory(out))write(out/"report.json",report);std::cerr<<e.what()<<'\n';return 1;}
 }
}
