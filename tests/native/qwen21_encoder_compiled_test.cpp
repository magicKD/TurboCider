#include "../../native/models/qwen21/text_encoder.hpp"
#include <iostream>
#include <cmath>

using namespace tc;
int main() {try {
    configure_streams();std::atomic<bool> cancelled{false};
    int cases=0;
    for(auto dtype:{mx::bfloat16,mx::float16,mx::float32}) {
        Weights weights;std::vector<std::string> keys;std::vector<Tensor> arrays;int seed=130;
        auto bind=[&](const std::string &key,mx::Shape shape,bool norm=false) {
            keys.push_back(key);arrays.push_back(norm?mx::ones(shape,dtype):
                mx::astype(mx::random::normal(shape,mx::float32,mx::random::key(seed++))*.02f,dtype));
        };
        bind("model.embed_tokens.weight",{8,128});bind("model.norm.weight",{128},true);
        for(int i=0;i<4;++i) {
            const auto p="model.layers."+std::to_string(i);
            for(const char *name:{"input_layernorm","post_attention_layernorm"})bind(p+"."+name+".weight",{128},true);
            for(const char *name:{"q_proj","k_proj","v_proj","o_proj"}) {
                const int out=(std::string(name)=="k_proj" || std::string(name)=="v_proj")?64:128;
                bind(p+".self_attn."+name+".weight",{out,128});bind(p+".self_attn."+name+".bias",{out});
            }
            for(const char *name:{"q_norm","k_norm"})bind(p+".self_attn."+name+".weight",{32},true);
            bind(p+".mlp.gate_proj.weight",{256,128});bind(p+".mlp.up_proj.weight",{256,128});
            bind(p+".mlp.down_proj.weight",{128,256});bind(p+".mlp.down_proj.bias",{128});
        }
        weights.bind_arrays(keys,arrays);weights.materialize();
        qwen21::TextConfig config;config.layers=4;config.heads=4;config.kv_heads=2;config.head_dim=32;config.mrope_sections={8,4,4};
        auto master=mx::copy(weights.at("model.layers.0.mlp.up_proj.weight"));mx::eval(master);
        for(int rows:{1,17,33,67})for(bool final_norm:{false,true})for(bool visual:{false,true}) {
            config.final_norm=final_norm;config.compiled_gpu_blocks=false;
            qwen21::TextEncoder eager(weights,config);
            config.compiled_gpu_blocks=true;qwen21::TextEncoder compiled(weights,config);
            auto input=mx::astype(mx::random::normal({1,rows,128},mx::float32,mx::random::key(123))*.2f,dtype);
            auto index=mx::arange(rows,mx::int32);
            auto positions=mx::stack({index,index/2,index/3});
            const int valid=std::max(1,rows-2);
            std::vector<Tensor> deltas;
            if(visual)for(int i=0;i<3;++i)deltas.push_back(mx::full(input.shape(),.01f*(i+1),dtype));
            auto expected=eager.encode_embeddings(input,positions,valid,{},cancelled,deltas);
            auto actual=compiled.encode_embeddings(input,positions,valid,{},cancelled,deltas);mx::eval({expected,actual});
            const float error=mx::sqrt(mx::sum(mx::square(mx::astype(actual,mx::float32)-mx::astype(expected,mx::float32)))/
                mx::maximum(mx::sum(mx::square(mx::astype(expected,mx::float32))),Tensor(1e-20f))).item<float>();
            require(actual.dtype()==dtype && actual.shape()==expected.shape() && std::isfinite(error) && error<.05f,
                "compiled encoder causal/padding/mRoPE/GQA/bias/DeepStack exceeded5% fixture budget");
            ++cases;
        }
        require(mx::all(master==weights.at("model.layers.0.mlp.up_proj.weight")).item<bool>(),"compiled encoder changed original source");
        // SAME encoder object, dynamic source keys/arrays on every call. A new
        // source binding (including bias values) must not consume stale W.
        qwen21::TextEncoder compiled(weights,config);
        auto input=mx::full({1,17,128},.25f,dtype);auto positions=mx::broadcast_to(mx::reshape(mx::arange(17,mx::int32),{1,17}),{3,17});
        auto before=compiled.encode_embeddings(input,positions,17,{},cancelled);mx::eval(before);
        auto changed=arrays;
        for(size_t i=0;i<keys.size();++i)if(keys[i].ends_with(".mlp.down_proj.bias"))changed[i]=changed[i]+Tensor(.2f,dtype);
        weights.bind_arrays(keys,changed);
        auto after=compiled.encode_embeddings(input,positions,17,{},cancelled);mx::eval(after);
        auto reference_config=config;reference_config.compiled_gpu_blocks=false;qwen21::TextEncoder reference(weights,reference_config);
        auto expected=reference.encode_embeddings(input,positions,17,{},cancelled);mx::eval(expected);
        const float error=mx::sqrt(mx::sum(mx::square(mx::astype(after,mx::float32)-mx::astype(expected,mx::float32)))/
            mx::sum(mx::square(mx::astype(expected,mx::float32)))).item<float>();
        require(error<.05f && !mx::all(before==after).item<bool>(),"compiled encoder source rebind was stale");
        cancelled.store(true);bool caught=false;
        try{compiled.encode_embeddings(input,positions,17,{},cancelled);}catch(const Cancelled&){caught=true;}
        cancelled.store(false);require(caught,"compiled encoder swallowed cancellation");
        caught=false;
        try{compiled.encode_embeddings(mx::full(input.shape(),NAN,dtype),positions,17,{},cancelled);}
        catch(const std::invalid_argument&){caught=true;}
        require(caught,"compiled encoder published nonfinite hidden");
    }
    std::cout<<"PASS compiled encoder cases="<<cases<<": original dynamic sources, BF16/FP16/F32, causal/padded keys, GQA/mRoPE, bias, early DeepStack/final norm, rebind/cancellation/nonfinite and immutable master\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;} }
