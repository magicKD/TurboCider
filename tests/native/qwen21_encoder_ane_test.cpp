#include "models/qwen21/text_encoder.hpp"
#include "backends/ane_ffn.hpp"
#include <iostream>

using namespace tc;
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    try {
        configure_streams();std::atomic<bool> cancelled{false};Weights weights;
        std::vector<std::string> keys;std::vector<Tensor> arrays;
        int seed=100;
        auto bind=[&](const std::string &name,mx::Shape shape,bool norm=false) {
            keys.push_back(name);arrays.push_back(norm?mx::ones(shape,mx::bfloat16):
                mx::astype(mx::random::normal(shape,mx::float32,mx::random::key(seed++))*.025f,mx::bfloat16));
        };
        bind("model.embed_tokens.weight",{8,128});bind("model.norm.weight",{128},true);
        for(int i=0;i<2;++i) {
            const auto p="model.layers."+std::to_string(i);
            for(const char *name:{"input_layernorm","post_attention_layernorm"})bind(p+"."+name+".weight",{128},true);
            for(const char *name:{"q_proj","k_proj","v_proj","o_proj"})bind(p+".self_attn."+name+".weight",{128,128});
            for(const char *name:{"q_norm","k_norm"})bind(p+".self_attn."+name+".weight",{128},true);
            bind(p+".mlp.gate_proj.weight",{512,128});bind(p+".mlp.up_proj.weight",{512,128});bind(p+".mlp.down_proj.weight",{128,512});
        }
        weights.bind_arrays(keys,arrays);weights.materialize();
        qwen21::TextConfig config;config.layers=2;config.heads=config.kv_heads=1;config.final_norm=false;
        qwen21::TextEncoder gpu(weights,config);
        const auto identity=ane::HybridFfn::executor_configuration_identity();
        const std::string chunks=std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS");
        setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","2",1);
        require(identity!=ane::HybridFfn::executor_configuration_identity(),"encoder chunks missing from execution identity");
        setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS",chunks.c_str(),1);
        setenv("TURBOCIDER_RUNTIME_ANE_PROFILE","1",1);
        require(identity!=ane::HybridFfn::executor_configuration_identity(),"encoder profiling missing from execution identity");
        unsetenv("TURBOCIDER_RUNTIME_ANE_PROFILE");
        for(bool compiled:{false,true})for(int rows:{1,33,34,67})for(bool visual:{false,true}) {
            auto input=mx::astype(mx::random::normal({1,rows,128},mx::float32,mx::random::key(34))*.2f,mx::bfloat16);
            auto positions=mx::broadcast_to(mx::reshape(mx::arange(rows,mx::int32),{1,rows}),{3,rows});
            std::vector<Tensor> deltas=visual?std::vector<Tensor>{mx::full({1,rows,128},.03125f,mx::bfloat16)}:std::vector<Tensor>{};
            const int valid=std::max(1,rows-1);
            auto expected=gpu.encode_embeddings(input,positions,valid,{},cancelled,deltas);mx::eval(expected);
            ane::HybridFfn runtime(argv[1],128,512,128ull<<20,cancelled);runtime.begin_request();
            auto candidate_config=config;candidate_config.compiled_gpu_blocks=compiled;
            qwen21::TextEncoder hybrid(weights,candidate_config,&runtime);
            auto actual=hybrid.encode_embeddings(input,positions,valid,{},cancelled,deltas);mx::eval(actual);runtime.drain();
            const float error=mx::sqrt(mx::sum(mx::square(mx::astype(actual,mx::float32)-mx::astype(expected,mx::float32)))/
                mx::sum(mx::square(mx::astype(expected,mx::float32)))).item<float>();
            std::cerr<<"encoder rows="<<rows<<" error="<<error<<" calls="<<runtime.metrics().runtime_calls
                <<" failed="<<runtime.metrics().runtime_failed<<" reason="<<runtime.reason()<<'\n';
            const bool split=rows>33;
            require(error<.02f && !runtime.metrics().runtime_failed &&
                (split ? runtime.metrics().runtime_calls==2 : runtime.metrics().runtime_calls==0 && (compiled || error==0)),
                "Qwen encoder shared runtime/causal/deepstack fixture failed");
            std::cout<<"PASS encoder rows="<<rows<<" visual="<<visual<<" relL2="<<error<<" calls="<<runtime.metrics().runtime_calls<<'\n';
        }
        ane::HybridFfn retained(argv[1],128,512,128ull<<20,cancelled);
        for(int visit=0;visit<8;++visit) {
            retained.begin_request();
            try {
                auto ephemeral=mx::astype(mx::random::normal({512,128},mx::float32,mx::random::key(visit+700))*.01f,mx::bfloat16);
                auto down=mx::contiguous(mx::transpose(ephemeral));mx::eval({ephemeral,down});
                ane::HybridFfn::SourceScope sources(&retained);
                retained.stage(0,34,{ephemeral,ephemeral,down});
                if(visit%2)throw std::runtime_error("attention failed before FFN launch");
                sources.finish();
            } catch(const std::runtime_error &error) {
                require(std::string(error.what())=="attention failed before FFN launch","unexpected source-scope error");
            }
            require(retained.metrics().runtime_calls==0 && !retained.metrics().runtime_failed,
                "source-scope drain performed an unexpected model launch");
        }
        retained.begin_request();
        qwen21::TextEncoder restored(weights,config,&retained);
        auto input=mx::full({1,34,128},.125f,mx::bfloat16);
        auto positions=mx::broadcast_to(mx::reshape(mx::arange(34,mx::int32),{1,34}),{3,34});
        auto output=restored.encode_embeddings(input,positions,34,{},cancelled);mx::eval(output);retained.drain();
        require(retained.metrics().runtime_calls==2 && !retained.metrics().runtime_failed && mx::all(mx::isfinite(output)).item<bool>(),
            "retained runtime did not recover after staged-source unwind");
        std::cout<<"PASS retained encoder source scope: staged-only drain, attention failure and recovery\n";
        // A finite BF16 post-norm exceeds the ANE FP16 bridge's range, while
        // the original BF16 full FFN remains finite. Exercise the REAL compiled
        // full callback on conversion failure, not a constant sentinel result.
        auto large=arrays;
        for(size_t i=0;i<keys.size();++i)if(keys[i].find("post_attention_layernorm.weight")!=std::string::npos)
            large[i]=mx::full(large[i].shape(),1000000.f,mx::bfloat16);
        weights.bind_arrays(keys,large);weights.materialize();
        auto failure_input=mx::full({1,67,128},.125f,mx::bfloat16);
        auto failure_positions=mx::broadcast_to(mx::reshape(mx::arange(67,mx::int32),{1,67}),{3,67});
        auto expected=gpu.encode_embeddings(failure_input,failure_positions,67,{},cancelled);mx::eval(expected);
        ane::HybridFfn failed(argv[1],128,512,128ull<<20,cancelled);failed.begin_request();
        auto compiled_config=config;compiled_config.compiled_gpu_blocks=true;
        qwen21::TextEncoder recovered(weights,compiled_config,&failed);
        auto actual=recovered.encode_embeddings(failure_input,failure_positions,67,{},cancelled);mx::eval(actual);failed.drain();
        const float error=mx::sqrt(mx::sum(mx::square(mx::astype(actual,mx::float32)-mx::astype(expected,mx::float32)))/
            mx::sum(mx::square(mx::astype(expected,mx::float32)))).item<float>();
        require(failed.metrics().runtime_failed && failed.metrics().runtime_weight_fallback_blocks>0 &&
            mx::all(mx::isfinite(actual)).item<bool>() && std::isfinite(error) && error<.05f,
            "compiled encoder did not recover the failed ANE span with the original complete GPU FFN");
        std::cout<<"PASS compiled encoder GPU fallback: finite original BF16 source, actual FP16 bridge failure, complete compiled FFN and finite5% recovery\n";
    } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
