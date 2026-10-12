#include "../../native/models/qwen21/transformer.hpp"
#include "../../native/models/qwen21/diagnostic_options.hpp"
#include <iostream>
#include <algorithm>

using namespace tc;
int main(int argc,char **argv) { try {
    require(argc==2,"temporary adapter path required");configure_streams();
    qwen21::TransformerConfig c;c.layers=3;c.heads=2;c.head_dim=32;c.channels=16;c.context_dim=32;c.rope_axes={4,12,16};
    std::vector<std::string> names;std::vector<Tensor> arrays;int seed=1900;
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
    auto x=mx::random::normal({1,4,16},mx::float32,mx::random::key(seed++)),text=mx::random::normal({1,2,32},mx::float32,mx::random::key(seed++));
    qwen21::ReferenceLatents ref{mx::random::normal({1,4,16},mx::float32,mx::random::key(seed++)),{2,2,1}};
    auto error=[](const Tensor &a,const Tensor &b){return mx::sqrt(mx::sum(mx::square(a-b))/mx::maximum(mx::sum(mx::square(b)),Tensor(1e-20f))).item<float>();};
    int cases=0;
    for(bool lora:{false,true}) {
        if(lora)require(w.apply_loras({{argv[1],.75f,"transformer"}},"transformer",[](const std::string&,int,int){},cancelled,true)==9,"fixture LoRA binding incomplete");
      for(bool with_reference:{false,true}) {
        const auto refs=with_reference?std::vector<qwen21::ReferenceLatents>{ref}:std::vector<qwen21::ReferenceLatents>{};
        const int prefill_rows=with_reference?10:6;
        qwen21::Transformer candidate(w,c);
        for(auto policy:{qwen21::RuntimeFfnPhase::Prefill,qwen21::RuntimeFfnPhase::Decode,qwen21::RuntimeFfnPhase::All,qwen21::RuntimeFfnPhase::Prefill}) {
            candidate.reset();qwen21::Transformer original(w,c);int prefill_calls=0,decode_calls=0,stages=0,plans=0;
            auto callback=[&](int layer,const Tensor &input) {
                require(input.shape(1)==prefill_rows || input.shape(1)==4,"phase input rows lost reference/text geometry");
                if(input.shape(1)==prefill_rows)++prefill_calls;else ++decode_calls;
                const auto p="transformer_blocks."+std::to_string(layer)+".img_mlp.";
                auto halves=mx::split(w.project(input,p+"gate_up"),2,-1);
                auto y=w.project(silu(halves[0])*halves[1],p+"out");mx::eval(y);return y;
            };
            candidate.set_stage_mlp([&](int,int){++stages;});
            candidate.set_plan_mlp([&](int,int){++plans;return qwen21::Transformer::MLPPlan::SplitUntimed;});
            candidate.set_observe_mlp([](int,int,double){throw std::runtime_error("untimed phase should not create a measurement fence");});
            candidate.set_prefill_mlp(qwen21::runtime_ffn_phase_runs(policy,false)?qwen21::Transformer::DecodeMLP(callback):qwen21::Transformer::DecodeMLP{});
            candidate.set_decode_mlp(qwen21::runtime_ffn_phase_runs(policy,true)?qwen21::Transformer::DecodeMLP(callback):qwen21::Transformer::DecodeMLP{});
            for(int step=0;step<3;++step) {
                const float sigma=.8f-step*.2f;
                auto expected=original.forward(x,text,sigma,2,2,true,nullptr,refs),actual=candidate.forward(x,text,sigma,2,2,true,nullptr,refs);mx::eval({expected,actual});
                require(error(actual,expected)<1e-5f,"phase switch changed complete base/LoRA FFN arithmetic");++cases;
                require(candidate.prefix_matches(text,2,2,refs),"completed prefix bank was not reusable");
            }
            const int expected_prefill=qwen21::runtime_ffn_phase_runs(policy,false)?3:0;
            const int expected_decode=qwen21::runtime_ffn_phase_runs(policy,true)?6:0;
            require(prefill_calls==expected_prefill && decode_calls==expected_decode && stages==expected_prefill+expected_decode && plans==stages,
                "disabled phase still staged/probed a split instead of using ordinary GPU blocks");
            // A changed condition at a later step must select PREFILL again.
            auto changed=text+Tensor(.01f);require(!candidate.prefix_matches(changed,2,2,refs),"changed condition claimed a prefix hit");
            mx::eval(candidate.forward(x,changed,.1f,2,2,true,nullptr,refs));
            require(prefill_calls==2*expected_prefill && decode_calls==expected_decode,"phase routing used step index instead of actual prefix reuse");
        }
        if(!with_reference)continue;
        int layer_cases=0;
        for(const auto &forced:std::vector<std::vector<int>>{{0},{2},{0,2}}) {
            qwen21::Transformer candidate(w,c),original(w,c);int plans=0,stages=0,calls=0;
            auto selected=[&](int i){return std::find(forced.begin(),forced.end(),i)!=forced.end();};
            candidate.set_plan_mlp([&](int i,int) {
                ++plans;return selected(i)?qwen21::Transformer::MLPPlan::Gpu:qwen21::Transformer::MLPPlan::SplitUntimed;
            });
            candidate.set_stage_mlp([&](int i,int){require(!selected(i),"forced GPU layer staged ANE weights");++stages;});
            candidate.set_observe_mlp([](int,int,double){throw std::runtime_error("fixed prefill layer policy inserted a timing fence");});
            candidate.set_prefill_mlp([&](int i,const Tensor &input) {
                require(!selected(i) && input.shape(1)==10,"forced/disabled layer entered split callback");++calls;
                const auto p="transformer_blocks."+std::to_string(i)+".img_mlp.";
                auto halves=mx::split(w.project(input,p+"gate_up"),2,-1);
                auto output=w.project(silu(halves[0])*halves[1],p+"out");mx::eval(output);return output;
            });
            for(int step=0;step<3;++step) {
                auto expected=original.forward(x,text,.8f-step*.2f,2,2,true,nullptr,{ref});
                auto actual=candidate.forward(x,text,.8f-step*.2f,2,2,true,nullptr,{ref});mx::eval({expected,actual});
                require(error(actual,expected)<1e-5f,"selected full-GPU layers changed complete base/LoRA FFN");++layer_cases;
            }
            require(plans==3 && calls==3-int(forced.size()) && stages==calls,
                "forced layer/decode still planned, staged or ran split work");
            auto changed=text+Tensor(.01f);mx::eval(candidate.forward(x,changed,.1f,2,2,true,nullptr,{ref}));
            require(plans==6 && calls==2*(3-int(forced.size())) && stages==calls,
                "changed condition lost prefill GPU layer routing");
        }
        require(layer_cases==9,"missing complete per-layer phase cases");
      }
    }
    std::cout<<"PASS Qwen FFN phase cases="<<cases<<": generation/edit, compiled base/real-LoRA, prefill/decode/all switches, full-GPU no-stage/no-probe, prefix hits and changed-condition prefill\n";
    std::cout<<"PASS Qwen prefill GPU layer cases=18: base/real-LoRA, first/last/mixed full-GPU blocks, no split staging/probe, unchanged decode and changed-condition prefill\n";
} catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;} }
