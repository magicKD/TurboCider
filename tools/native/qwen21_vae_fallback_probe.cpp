// Tiny synthetic overflow/recovery contract, never a performance/model sample.
#include "../../native/models/qwen21/vae.hpp"
#include <iostream>
#include <limits>

using namespace tc;
int main() {try {
    configure_streams();Weights weights;
    std::vector<std::string> names;std::vector<Tensor> values;
    auto conv=[&](const std::string &name,int out,int in,int k,float scale) {
        names.push_back(name+".weight");values.push_back(mx::full({out,in,k,k},scale,mx::bfloat16));
        names.push_back(name+".bias");values.push_back(mx::zeros({out},mx::bfloat16));
    };
    auto norm=[&](const std::string &name) {names.push_back(name+".gamma");values.push_back(mx::ones({4,1,1,1},mx::bfloat16));};
    auto residual=[&](const std::string &name) {
        norm(name+".norm1");norm(name+".norm2");conv(name+".conv1",4,4,3,.01f);conv(name+".conv2",4,4,3,.01f);
    };
    conv("post_quant_conv",64,64,1,1000.f); // finite coefficients; FP16 output must overflow
    conv("decoder.conv_in",4,64,3,1e-6f); // original FP32 path stays finite
    residual("decoder.mid_block.resnets.0");residual("decoder.mid_block.resnets.1");
    norm("decoder.mid_block.attentions.0.norm");
    conv("decoder.mid_block.attentions.0.to_qkv",12,4,1,.01f);
    conv("decoder.mid_block.attentions.0.proj",4,4,1,.01f);
    for(int block=0;block<5;++block) {
        const auto p="decoder.up_blocks."+std::to_string(block);
        for(int layer=0;layer<3;++layer)residual(p+".resnets."+std::to_string(layer));
        if(block<4)conv(p+".upsampler.resample.1",4,4,3,.01f);
    }
    norm("decoder.norm_out");conv("decoder.conv_out",4,4,3,.01f);
    weights.bind_arrays(names,values);weights.materialize();
    auto input=mx::full({1,64,1,1},100.f,mx::float32);mx::eval(input);
    std::atomic<bool> cancelled{false};int fallbacks=0;
    for(bool fused:{false,true}) {
        qwen21::VAE original(weights,qwen21::VAE::DecodeConvolution::Original,fused);
        qwen21::VAE half(weights,qwen21::VAE::DecodeConvolution::FP16WithFP32State,fused);
        auto reference=original.decode_checked(input,{},cancelled);
        std::vector<std::string> phases;
        const Event events=[&](const std::string &phase,int n,int total) {
            require(total==5 && n==int(phases.size()%5)+1,"retry progress order changed");phases.push_back(phase);
        };
        auto actual=half.decode_checked(input,events,cancelled);
        require(!reference.fp16_nonfinite_fallback && actual.fp16_nonfinite_fallback && phases.size()==10,
                "nonfinite FP16 attempt did not run complete FP32 fallback");
        for(size_t i=0;i<phases.size();++i)require(phases[i]==(i<5?"qwen21_vae_decode":"qwen21_vae_decode_fp32_fallback"),
                "fallback progress was silently relabelled");
        require(actual.pixels.shape()==mx::Shape{1,4,16,16} && actual.pixels.dtype()==mx::float32 &&
                mx::all(mx::equal(reference.pixels,actual.pixels)).item<bool>(),"fallback did not reproduce complete original-precision output");
        ++fallbacks;
        cancelled=true;bool stopped=false;
        try{(void)half.decode_checked(input,{},cancelled);}catch(const Cancelled&){stopped=true;}
        require(stopped,"cancelled decoder was retried");cancelled=false;
        int calls=0;stopped=false;
        const Event cancel_before_retry=[&](const std::string &phase,int n,int) {
            require(phase=="qwen21_vae_decode","cancelled request entered fallback");++calls;if(n==5)cancelled=true;
        };
        try{(void)half.decode_checked(input,cancel_before_retry,cancelled);}catch(const Cancelled&){stopped=true;}
        require(stopped && calls==5,"cancellation between failed attempt and retry was ignored");cancelled=false;
        auto restored=original.decode_checked(input,{},cancelled);
        require(mx::all(mx::equal(reference.pixels,restored.pixels)).item<bool>(),"retry/cancellation mutated original VAE source");
        bool rejected=false;
        try{(void)half.decode_checked(mx::ones({1,63,1,1}),{},cancelled);}catch(const std::exception&){rejected=true;}
        require(rejected,"malformed latent was hidden by a retry");
        rejected=false;
        try{(void)original.decode_checked(mx::full({1,64,1,1},std::numeric_limits<float>::quiet_NaN(),mx::float32),{},cancelled);}
        catch(const std::exception&){rejected=true;}
        require(rejected,"nonfinite original decoder was falsely marked successful");
    }
    std::cout<<"{\"full_fp32_fallbacks\":"<<fallbacks<<",\"rgba_channels\":4,\"cancel_before_decode\":true,"
        "\"cancel_before_retry\":true,\"invalid_input_rejected\":true,\"original_source_restored\":true}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
