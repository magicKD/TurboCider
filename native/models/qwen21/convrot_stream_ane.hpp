#pragma once

#include "convrot_weights.hpp"
#include "runtime_ffn_graphs.hpp"
#include "transformer.hpp"
#include <string_view>

namespace tc::qwen21 {

// One fixed private diagnostic graph per streamed request. The stream bank
// outlives this bridge and cannot refill its current slot before retire().
class ConvrotStreamAne final {
    static constexpr int rows_=1024, hidden_=4096, width_=12288;
    static constexpr int ane_channels_=4096, gpu_channels_=width_-ane_channels_;
    std::atomic<bool>& cancelled_;
    ane::HybridFfn runtime_;
    runtime_ffn::Function full_, channels_;
    const Weights* current_=nullptr;
    int layer_=-1, split_layer_=-1;

    static const std::filesystem::path& validate(const std::filesystem::path& manifest) {
#if !TURBOCIDER_ENABLE_PRIVATE_ANE
        throw std::invalid_argument("streamed ConvRot Runtime ANE requires a private diagnostic build");
#else
        const auto setting=[](const char* name,std::string_view value,bool optional=false) {
            const char* raw=std::getenv(name);
            require(raw ? std::string_view(raw)==value : optional,
                    std::string("streamed ConvRot Runtime ANE requires ")+name+"="+std::string(value));
        };
        setting("TURBOCIDER_QWEN21_CONVROT_STREAM_ANE","1");
        setting("TURBOCIDER_ANE_BACKEND","private");
        setting("TURBOCIDER_ALLOW_PRIVATE_ANE","1");
        setting("TURBOCIDER_PRIVATE_ANE_DATA_PATH","convrot_w8a8");
        setting("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1");
        setting("TURBOCIDER_RUNTIME_ANE_CHUNKS","1");
        setting("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1");
        setting("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1");
        setting("TURBOCIDER_PRIVATE_ANE_GPU_IO","1");
        setting("TURBOCIDER_PRIVATE_ANE_PREFETCH","0");
        setting("TURBOCIDER_PRIVATE_ANE_SCALE_CACHE","0");
        setting("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE","1");
        setting("TURBOCIDER_PRIVATE_ANE_CHANNELS","4096",true);
        for (const char* name : {"TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD", "TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE",
             "TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES", "TURBOCIDER_PRIVATE_ANE_FP16_BF16_VALUES",
             "TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE", "TURBOCIDER_RUNTIME_ANE_PROFILE",
             "TURBOCIDER_RUNTIME_ANE_CHANNEL_GPU_FIRST", "TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN",
             "TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES"}) setting(name,"0",true);
        const auto descriptor=ane::runtime_template_descriptor(manifest);
        const auto& shape=descriptor.shape;
        require(!descriptor.has_public_artifact() && shape.kind==ane::Kind::SwiGLU && shape.rows==rows_ &&
                shape.hidden==hidden_ && shape.width==width_ && shape.tile_k==2048 && shape.tile_n==1024 &&
                !shape.lora_inputs,"streamed ConvRot Runtime ANE requires the fixed base-only 1024×4096×12288 descriptor");
        return manifest;
#endif
    }

    static std::string prefix(int layer) {
        return "transformer_blocks."+std::to_string(layer)+".img_mlp.";
    }
    static const std::vector<std::string>& source_keys() {
        static const std::vector<std::string> keys{
            "gate_up.weight", "gate_up.weight_scale", "gate_up.comfy_quant",
            "out.weight", "out.weight_scale", "out.comfy_quant"};
        return keys;
    }
    static runtime_ffn::Function compile_gpu(bool full) {
        return mx::compile([full](const std::vector<Tensor>& a) {
            // All checkpoint tensors are dynamic arguments to the graph.
            // Capturing a reusable slot would read another layer on a hit.
            Weights weights;
            weights.bind_arrays(source_keys(),a,1);
            weights.set_convrot_w8a8(true);
            const auto& input=a[0];
            if (full) {
                auto halves=mx::split(weights.project(input,"gate_up"),2,-1);
                return std::vector<Tensor>{weights.project(silu(halves[0])*halves[1],"out")};
            }
            auto gate=weights.project_slice(input,"gate_up",0,gpu_channels_,0,hidden_,false);
            auto up=weights.project_slice(input,"gate_up",width_,width_+gpu_channels_,0,hidden_,false);
            auto intermediate=silu(gate)*up;
            auto base=weights.project_base_slice_fp32(intermediate,"out",0,hidden_,0,gpu_channels_);
            return std::vector<Tensor>{base,intermediate};
        });
    }
    Tensor run(int layer,const Tensor& input) {
        require(current_ && layer_==layer && split_layer_==layer,
                "streamed ConvRot Runtime ANE must use the current planned layer");
        std::vector<Tensor> args{input};
        const auto stem=prefix(layer);
        for (const auto& key : source_keys()) args.push_back(current_->at(stem+key));
        return runtime_.run(layer,input,[&](const Tensor&) { return full_(args)[0]; },cancelled_,nullptr,
            [&](const Tensor&,int first,int count) {
                require(first==0 && count==gpu_channels_,"streamed ConvRot Runtime ANE GPU partition changed");
                auto values=channels_(args);
                return std::pair{values[0],values[1]};
            });
    }
public:
    ConvrotStreamAne(const std::filesystem::path& manifest,std::atomic<bool>& cancelled)
        : cancelled_(cancelled),
          runtime_(validate(manifest),hidden_,width_,size_t(640)<<20,cancelled,false,
                   std::nullopt,{},nullptr,std::nullopt,ane_channels_,std::nullopt,0,0,
                   ane::HybridFfn::SourceMemory::StreamedConvrot),
          full_(compile_gpu(true)), channels_(compile_gpu(false)) {
        runtime_.begin_request();
    }

    void attach(Transformer& dit) {
        // Leave the first pass and its variable-length prefix on the usual GPU graph.
        dit.set_decode_mlp([this](int layer,const Tensor& input) { return run(layer,input); });
        dit.set_plan_mlp([this](int layer,int rows) {
            require(!current_ && split_layer_==-1 && rows==rows_,"invalid streamed ConvRot Runtime ANE plan");
            const auto plan=runtime_.plan_block(layer,rows);
            if (plan.mode==ane::RowScheduler::Mode::HybridUntimed) {
                split_layer_=layer;
                return Transformer::MLPPlan::SplitUntimed;
            }
            require(plan.mode==ane::RowScheduler::Mode::Gpu,"streamed ConvRot Runtime ANE requires an untimed plan");
            return Transformer::MLPPlan::Gpu;
        });
        dit.set_observe_mlp([this](int layer,int rows,double seconds) { runtime_.observe_block(layer,rows,seconds); });
    }

    void acquire(int layer,const Weights& weights) {
        require(!current_ && (split_layer_==-1 || split_layer_==layer) && weights.convrot_w8a8(),
                "streamed ConvRot Runtime ANE requires one current W8A8 layer");
        current_=&weights;
        layer_=layer;
        if (split_layer_==layer) {
            auto sources=convrot_ffn_sources(weights,prefix(layer));
            for (const auto& source : sources)
                require(source.values.dtype()==mx::int8,"streamed ConvRot Runtime ANE requires the original signed slot");
            runtime_.stage_weights(layer,rows_,std::move(sources));
        }
    }
    void retire(int layer) {
        require(current_ && layer_==layer,"streamed ConvRot Runtime ANE retirement must match the current layer");
        if (split_layer_==layer) runtime_.drain();
        current_=nullptr;
        layer_=split_layer_=-1;
    }
    void finish() {
        require(!current_ && split_layer_==-1,"streamed ConvRot Runtime ANE finished with a borrowed layer");
        runtime_.drain();
    }
    HybridMetrics metrics() const { return runtime_.metrics(); }
    bool available() const { return runtime_.available(); }
    const std::string& reason() const { return runtime_.reason(); }
};

} // namespace tc::qwen21
