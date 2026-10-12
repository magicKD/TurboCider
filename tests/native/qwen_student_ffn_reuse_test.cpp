#include "../../native/models/qwen21/transformer.hpp"
#include <iostream>

using namespace tc;
int main(int argc,char **argv){try {
    require(argc==2,"temporary adapter path required");configure_streams();
    qwen21::TransformerConfig c;c.layers=3;c.heads=2;c.head_dim=32;c.channels=16;c.context_dim=32;c.rope_axes={4,12,16};
    std::vector<std::string> names;std::vector<Tensor> arrays;int seed=700;
    auto matrix=[&](const std::string &p,int n,int k){names.push_back(p+".weight");arrays.push_back(mx::random::normal({n,k},mx::float32,mx::random::key(seed++))/std::sqrt(float(k)));};
    auto vector=[&](const std::string &p,int n,float v){names.push_back(p+".weight");arrays.push_back(mx::full({n},v));};
    matrix("time_text_embed.timestep_embedder.linear_1",64,256);matrix("time_text_embed.timestep_embedder.linear_2",64,64);
    matrix("modulation.1",256,64);matrix("img_in",64,16);vector("txt_in.text_norm",32,0);
    matrix("txt_in.in_layer",64,32);matrix("txt_in.out_layer",64,64);matrix("norm_out.linear",64,64);matrix("proj_out",16,64);
    std::unordered_map<std::string,Tensor> adapter;
    for(int i=0;i<c.layers;++i) {
        const auto p="transformer_blocks."+std::to_string(i);
        for(const char *a:{"to_q","to_k","to_v","to_out.0"})matrix(p+".attn."+a,64,64);
        vector(p+".attn.norm_q",32,1);vector(p+".attn.norm_k",32,1);matrix(p+".img_mlp.gate_up",192,64);matrix(p+".img_mlp.out",64,96);
        for(const char *a:{"gate_layer","proj","out"}) {
            const auto s=p+".img_mlp."+a;const bool out=std::string(a)=="out";
            adapter.emplace(s+".lora_A.weight",mx::random::normal({8,out?96:64},mx::float32,mx::random::key(seed++))*.01f);
            adapter.emplace(s+".lora_B.weight",mx::random::normal({out?64:96,8},mx::float32,mx::random::key(seed++))*.01f);
        }
    }
    mx::save_safetensors(argv[1],adapter);Weights w;w.bind_arrays(names,arrays);w.materialize();std::atomic<bool> cancelled{false};
    require(w.apply_loras({{argv[1],.75f,"transformer"}},"transformer",[](const std::string&,int,int){},cancelled,true)==9,"fixture LoRA binding incomplete");
    auto x=mx::random::normal({1,4,16},mx::float32,mx::random::key(seed++)),text=mx::random::normal({1,4,32},mx::float32,mx::random::key(seed++));
    auto error=[](const Tensor &a,const Tensor &b){return mx::sqrt(mx::sum(mx::square(a-b))/mx::maximum(mx::sum(mx::square(b)),Tensor(1e-20f))).item<float>();};
    for(bool split:{false,true})for(bool half:{false,true}) {
        qwen21::Transformer original(w,c),candidate(w,c);int calls=0;bool fail=false;
        if(split)candidate.set_decode_mlp([&](int layer,const Tensor &input){
            require(!fail,"callback must not run while complete cached FFN is consumed");++calls;
            const auto p="transformer_blocks."+std::to_string(layer)+".img_mlp.";
            auto halves=mx::split(w.project(input,p+"gate_up"),2,-1);auto y=w.project(silu(halves[0])*halves[1],p+"out");mx::eval(y);return y;
        });
        mx::eval(original.forward(x,text,.8f,2,2));mx::eval(candidate.forward(x,text,.8f,2,2));
        candidate.set_ffn_cache_mode(qwen21::Transformer::FFNCacheMode::Capture);
        auto expected=original.forward(x,text,.2f,2,2),captured=candidate.forward(x,text,.2f,2,2);mx::eval({expected,captured});
        require(error(captured,expected)<1e-5f && candidate.last_ffn_captured_blocks()==3 && candidate.last_ffn_reused_blocks()==0 &&
            candidate.ffn_cache_logical_bytes()==3*4*64*4,"full LoRA capture changed arithmetic/counts/logical bound");
        fail=!half;const int before=calls;candidate.set_ffn_cache_mode(half?qwen21::Transformer::FFNCacheMode::ReuseLast16:qwen21::Transformer::FFNCacheMode::Reuse);
        auto reused=candidate.forward(x,text,.2f,2,2);mx::eval(reused);
        require(error(reused,expected)<1e-5f && calls==before+(split&&half?1:0) && candidate.last_ffn_reused_blocks()==(half?2:3) && candidate.last_ffn_captured_blocks()==0,
            "cache did not reuse completed full FFN including adapters without invoking callback");
        candidate.clear_step_cache();require(candidate.ffn_cache_logical_bytes()==0,"step cache owner was retained");
        candidate.set_ffn_cache_mode(qwen21::Transformer::FFNCacheMode::Reuse);bool rejected=false;
        try{candidate.forward(x,text,.1f,2,2);}catch(const std::exception&){rejected=true;}
        require(rejected,"missing complete preceding cache was consumed");
        candidate.clear_step_cache();fail=false;mx::eval(candidate.forward(x,text,.1f,2,2));
        candidate.set_ffn_cache_mode(qwen21::Transformer::FFNCacheMode::Capture);mx::eval(candidate.forward(x,text,.2f,2,2));
        auto changed=text+Tensor(.01f);candidate.set_ffn_cache_mode(qwen21::Transformer::FFNCacheMode::Reuse);
        auto fresh=candidate.forward(x,changed,.1f,2,2);mx::eval(fresh);
        require(candidate.last_ffn_reused_blocks()==0 && candidate.ffn_cache_logical_bytes()==0,"changed conditioning consumed stale FFN cache");
        candidate.reset();require(candidate.ffn_cache_logical_bytes()==0,"reset retained activation bank");
    }
    std::cout<<"PASS student full-LoRA FFN cache: actual compiled GPU/split capture, complete reuse skips callback, request counts/logical bytes, missing cache rejection, recovery and conditioning reset\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
