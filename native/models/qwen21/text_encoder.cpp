#include "text_encoder.hpp"
#include "../../backends/ane_ffn.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <tuple>

namespace tc::qwen21 {
namespace {
// Qwen3-VL explicitly accumulates and scales in FP32, unlike DiT's BF16
// nn.RMSNorm. Do not share the DiT normalization implementation here.
Tensor vl_norm(const Tensor &x, const Tensor &weight, float eps) {
    auto f = mx::astype(x, mx::float32);
    return mx::astype(mx::astype(weight, mx::float32) *
                      (f * mx::rsqrt(mx::mean(f * f, -1, true) + eps)), x.dtype());
}
Tensor rotate_half(const Tensor &x, const Tensor &cosine, const Tensor &sine) {
    auto halves = mx::split(x, 2, -1);
    return x * cosine + mx::concatenate({-halves[1], halves[0]}, -1) * sine;
}
using BlockFunction=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
std::vector<Tensor> attention_block(const Tensor &hidden,const Tensor &cosine,const Tensor &sine,
    const Tensor &mask,const Weights &weights,const TextConfig &config,const std::string &p) {
    auto input=vl_norm(hidden,weights.at(p+".input_layernorm.weight"),config.epsilon);
    auto q=heads(linear(input,weights,p+".self_attn.q_proj"),config.heads,config.head_dim);
    auto k=heads(linear(input,weights,p+".self_attn.k_proj"),config.kv_heads,config.head_dim);
    auto v=heads(linear(input,weights,p+".self_attn.v_proj"),config.kv_heads,config.head_dim);
    q=rotate_half(vl_norm(q,weights.at(p+".self_attn.q_norm.weight"),config.epsilon),cosine,sine);
    k=rotate_half(vl_norm(k,weights.at(p+".self_attn.k_norm.weight"),config.epsilon),cosine,sine);
    if(!config.compiled_gpu_blocks) {
        k=mx::repeat(k,config.heads/config.kv_heads,1);
        v=mx::repeat(v,config.heads/config.kv_heads,1);
    }
    // Explicit approximate profile only. SDPA consumes the already rounded
    // original Q/K/V dtype and native GQA; its softmax remains F32. The mask
    // contains only 0/-Inf, both exact in the input dtype. Keep legacy F32
    // attention/repeated K/V unchanged outside the opt-in profile.
    auto attention_mask=config.compiled_gpu_blocks ? mx::astype(mask,q.dtype()) : mask;
    auto residual=hidden+linear(attend(q,k,v,!config.compiled_gpu_blocks,attention_mask),weights,p+".self_attn.o_proj");
    return {residual,vl_norm(residual,weights.at(p+".post_attention_layernorm.weight"),config.epsilon)};
}
Tensor full_ffn(const Tensor &x,const Weights &weights,const std::string &mlp) {
    return linear(silu(linear(x,weights,mlp+".gate_proj"))*linear(x,weights,mlp+".up_proj"),weights,mlp+".down_proj");
}
BlockFunction compiled_block(const std::vector<std::string> &keys,const TextConfig &config,
    const std::string &p,bool split) {
    return mx::compile([keys,config,p,split](const std::vector<Tensor> &a) {
        // Dynamic original source arrays, not checkpoint constants captured at
        // the first trace. Rebinding the owner's Weights cannot read old W.
        Weights local;local.bind_arrays(keys,a,4);
        auto result=attention_block(a[0],a[1],a[2],a[3],local,config,p);
        if(split)return result;
        return std::vector<Tensor>{result[0]+full_ffn(result[1],local,p+".mlp")};
    });
}
BlockFunction compiled_ffn(const std::vector<std::string> &keys,const std::string &mlp,
    int first=0,int count=0,int hidden=0,bool fp32=false) {
    return mx::compile([keys,mlp,first,count,hidden,fp32](const std::vector<Tensor> &a) {
        Weights local;local.bind_arrays(keys,a,1);
        if(!count)return std::vector<Tensor>{full_ffn(a[0],local,mlp)};
        auto gate=local.project_base_slice(a[0],mlp+".gate_proj",first,first+count,0,hidden,false);
        auto up=local.project_base_slice(a[0],mlp+".up_proj",first,first+count,0,hidden,false);
        auto intermediate=silu(gate)*up;
        auto down=fp32 ? local.project_base_slice_fp32(intermediate,mlp+".down_proj",0,hidden,first,first+count) :
            local.project_base_slice(intermediate,mlp+".down_proj",0,hidden,first,first+count,false);
        return std::vector<Tensor>{down,intermediate};
    });
}
}

TextEncoder::TextEncoder(const Weights &weights, TextConfig config,ane::HybridFfn *runtime)
    : weights_(weights), config_(config), language_prefix_(
          weights.has("model.language_model.embed_tokens.weight") ? "model.language_model." : "model."),runtime_(runtime) {
    require(config.layers > 0 && config.heads > 0 && config.kv_heads > 0 &&
            config.heads % config.kv_heads == 0 && config.head_dim > 0 &&
            config.head_dim % 2 == 0 && config.theta > 0,
            "invalid Qwen21 text encoder geometry");
    require(!config.compiled_gpu_blocks || config.layers<=128,"compiled Qwen encoder layer count exceeds request-local graph bound");
    require(config.mrope_sections[0] + config.mrope_sections[1] + config.mrope_sections[2] == config.head_dim / 2,
            "invalid Qwen21 text mRoPE sections");
    for (int axis = 1; axis <= 2; ++axis)
        require(config.mrope_sections[axis] >= 0 &&
                (config.mrope_sections[axis] == 0 || axis + 3 * (config.mrope_sections[axis] - 1) < config.head_dim / 2),
                "Qwen21 interleaved mRoPE exceeds head dimension");
}

std::string TextEncoder::system_prefix() {
    return "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
}

std::string TextEncoder::prompt_template(const std::string &prompt) {
    const auto content = prompt.find_first_not_of(" \t\r\n") == std::string::npos ? " " : prompt;
    return system_prefix() + "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n";
}

Tensor TextEncoder::encode(const Tokens &tokens, const Event &event, std::atomic<bool> &cancelled) const {
    require(!tokens.ids.empty(), "empty Qwen21 token sequence");
    const auto &embedding = weights_.at(language_prefix_ + "embed_tokens.weight");
    for (int id : tokens.ids) require(id >= 0 && id < embedding.shape(0), "Qwen21 token ID out of vocabulary");
    int count = int(tokens.ids.size());
    auto ids = Tensor(tokens.ids.data(), {1, count}, mx::int32);
    auto positions = mx::broadcast_to(mx::reshape(mx::arange(count, mx::int32), {1, count}), {3, count});
    return encode_embeddings(mx::take(embedding, ids, 0), positions, tokens.valid, event, cancelled);
}

Tensor TextEncoder::encode_embeddings(const Tensor &embeddings, const Tensor &positions,
                                      int valid_tokens, const Event &event,
                                      std::atomic<bool> &cancelled,
                                      const std::vector<Tensor> &deepstack_deltas) const {
    require(embeddings.ndim() == 3 && embeddings.shape(0) == 1 && embeddings.shape(1) > 0,
            "Qwen21 text embeddings must be [1,sequence,hidden]");
    int count = embeddings.shape(1);
    require(valid_tokens > 0 && valid_tokens <= count && positions.shape() == mx::Shape{3, count},
            "Qwen21 positions/valid-token count mismatch");
    require(deepstack_deltas.size() <= size_t(config_.layers), "too many visual deepstack residuals");
    for (const auto &delta : deepstack_deltas)
        require(delta.shape() == embeddings.shape() && delta.dtype() == embeddings.dtype(),
                "Qwen21 deepstack residual shape/dtype mismatch");
    auto frequency = Tensor(1.f) / mx::power(Tensor(config_.theta),
        mx::arange(0, config_.head_dim, 2, mx::float32) / float(config_.head_dim));
    std::vector<Tensor> angles;
    for (int axis = 0; axis < 3; ++axis)
        angles.push_back(mx::reshape(mx::astype(slice_axis(positions, 0, axis, axis + 1), mx::float32), {count, 1}) * frequency);
    std::vector<Tensor> columns;
    for (int column = 0; column < config_.head_dim / 2; ++column) {
        int axis = 0;
        if (column % 3 == 1 && column < config_.mrope_sections[1] * 3) axis = 1;
        if (column % 3 == 2 && column < config_.mrope_sections[2] * 3) axis = 2;
        columns.push_back(slice_axis(angles[axis], 1, column, column + 1));
    }
    auto angle = mx::concatenate(columns, -1);
    angle = mx::concatenate({angle, angle}, -1);
    auto cosine = mx::reshape(mx::astype(mx::cos(angle), embeddings.dtype()), {1, 1, count, config_.head_dim});
    auto sine = mx::reshape(mx::astype(mx::sin(angle), embeddings.dtype()), {1, 1, count, config_.head_dim});
    auto idx = mx::arange(count, mx::int32);
    auto key = mx::reshape(idx, {1, count}), query = mx::reshape(idx, {count, 1});
    auto mask = mx::reshape(mx::where(mx::logical_or(key > query, key >= Tensor(valid_tokens)),
                                      Tensor(-INFINITY), Tensor(0.f)), {1, 1, count, count});
    auto hidden = embeddings;
    require(!config_.compiled_gpu_blocks || !weights_.has_runtime_loras(),
            "compiled Qwen encoder requires original base language weights, not encoder LoRA");
    const auto source_keys=config_.compiled_gpu_blocks ? weights_.sorted_keys() : std::vector<std::string>{};
    uint64_t full_blocks=0,attention_segments=0,channel_segments=0,full_ffn_segments=0;
    // One encode owns the cache. Canonical layer names let structurally equal
    // blocks share a traced function while EVERY original W remains an input.
    // No retained/global graph cache, checkpoint constants or source owners.
    std::map<std::pair<std::vector<std::string>,bool>,BlockFunction> block_graphs;
    std::map<std::tuple<std::vector<std::string>,int,int,int,bool>,BlockFunction> ffn_graphs;
    for (int i = 0; i < config_.layers; ++i) {
        checkpoint(cancelled);
        auto p = language_prefix_ + "layers." + std::to_string(i);
        auto plan=runtime_ ? runtime_->plan_block(i,count) : ane::RowScheduler::Plan{ane::RowScheduler::Mode::Gpu,0};
        if(plan.measured())mx::eval(hidden);
        const auto block_start=std::chrono::steady_clock::now();
        auto source=[&](int layer) {
            const auto stem=language_prefix_+"layers."+std::to_string(layer)+".mlp.";
            return std::vector<ane::FfnWeight>{{weights_.at(stem+"gate_proj.weight"),std::nullopt,std::nullopt},
                {weights_.at(stem+"up_proj.weight"),std::nullopt,std::nullopt},
                {weights_.at(stem+"down_proj.weight"),std::nullopt,std::nullopt}};
        };
        if(plan.split())runtime_->stage_weights(i,count,source(i)); // before attention, same shared stager/banks
        Tensor input=hidden;
        std::vector<std::string> block_keys;
        std::vector<Tensor> block_sources;
        if(config_.compiled_gpu_blocks) {
            for(const auto &key:source_keys)if(key.starts_with(p+".")) {
                block_keys.push_back("block"+key.substr(p.size()));block_sources.push_back(weights_.at(key));
            }
            const auto graph_key=std::make_pair(block_keys,plan.split());
            auto found=block_graphs.find(graph_key);
            if(found==block_graphs.end())found=block_graphs.emplace(graph_key,compiled_block(block_keys,config_,"block",plan.split())).first;
            std::vector<Tensor> arguments{hidden,cosine,sine,mask};
            arguments.insert(arguments.end(),block_sources.begin(),block_sources.end());
            auto result=found->second(arguments);hidden=result[0];
            if(plan.split()) {input=result[1];++attention_segments;} else ++full_blocks;
        } else {
            auto result=attention_block(hidden,cosine,sine,mask,weights_,config_,p);
            hidden=result[0];input=result[1];
        }
        const auto mlp=p+".mlp";
        std::optional<BlockFunction> compiled_full,compiled_partial;
        auto with_sources=[&](const Tensor &x) {
            std::vector<Tensor> args{x};args.insert(args.end(),block_sources.begin(),block_sources.end());return args;
        };
        auto shared_ffn=[&](int first,int count,int width,bool fp32) {
            const auto key=std::make_tuple(block_keys,first,count,width,fp32);
            auto found=ffn_graphs.find(key);
            if(found==ffn_graphs.end())found=ffn_graphs.emplace(key,compiled_ffn(block_keys,"block.mlp",first,count,width,fp32)).first;
            return found->second;
        };
        if(config_.compiled_gpu_blocks && plan.split())compiled_full=shared_ffn(0,0,0,false);
        auto gpu=[&](const Tensor &x) {
            if(compiled_full) {++full_ffn_segments;return (*compiled_full)(with_sources(x))[0];}
            return full_ffn(x,weights_,mlp);
        };
        if(plan.split()) {
            const int width=weights_.at(mlp+".gate_proj.weight").shape(0),h=input.shape(2);
            auto channel_gpu=[&](const Tensor &x,int first,int channels) {
                if(config_.compiled_gpu_blocks) {
                    if(!compiled_partial)compiled_partial=shared_ffn(first,channels,h,runtime_->fp32_channel_join());
                    ++channel_segments;
                    auto result=(*compiled_partial)(with_sources(x));return std::make_pair(result[0],result[1]);
                }
                auto gate=weights_.project_base_slice(x,mlp+".gate_proj",first,first+channels,0,h,false);
                auto up=weights_.project_base_slice(x,mlp+".up_proj",first,first+channels,0,h,false);
                auto intermediate=silu(gate)*up;
                auto down=runtime_->fp32_channel_join() ? weights_.project_base_slice_fp32(intermediate,mlp+".down_proj",0,h,first,first+channels) :
                    weights_.project_base_slice(intermediate,mlp+".down_proj",0,h,first,first+channels,false);
                return std::make_pair(down,intermediate);
            };
            require(width>0,"Qwen21 encoder FFN width missing");
            ane::HybridFfn::NextWeights next=[&](int layer) {return layer<config_.layers ? source(layer) : std::vector<ane::FfnWeight>{};};
            hidden=hidden+runtime_->run(i,input,gpu,cancelled,nullptr,channel_gpu,next);
        } else if(!config_.compiled_gpu_blocks)hidden=hidden+gpu(input);
        // Visual levels enter consecutive early language layers, not layers
        // 8/16/24. The prompt assembler zeros these deltas outside image spans.
        if (size_t(i) < deepstack_deltas.size()) hidden = hidden + deepstack_deltas[i];
        if(plan.measured()) {
            mx::eval(hidden);checkpoint(cancelled);
            runtime_->observe_block(i,count,std::chrono::duration<double>(std::chrono::steady_clock::now()-block_start).count());
        }
        if ((i + 1) % 4 == 0 || i + 1 == config_.layers) mx::eval(hidden);
        if (event) event("qwen21_text_encode", i + 1, config_.layers);
    }
    if(config_.compiled_gpu_blocks) {
        require(mx::all(mx::isfinite(hidden)).item<bool>(),"compiled Qwen encoder returned nonfinite output");
        std::cerr<<"{\"qwen_encoder_compiled_gpu\":{\"rows\":"<<count<<",\"layers\":"<<config_.layers
            <<",\"full_blocks\":"<<full_blocks<<",\"attention_segments\":"<<attention_segments
            <<",\"channel_ffn_segments\":"<<channel_segments<<",\"full_ffn_segments\":"<<full_ffn_segments
            <<",\"block_graphs\":"<<block_graphs.size()<<",\"ffn_graphs\":"<<ffn_graphs.size()
            <<",\"native_gqa_attention\":true"
            <<",\"scope\":\"evaluated language hidden; dynamic original array arguments; graph invocations, not physical kernels\"}}\n";
    }
    return config_.final_norm ? vl_norm(hidden, weights_.at(language_prefix_ + "norm.weight"), config_.epsilon) : hidden;
}
} // namespace tc::qwen21
