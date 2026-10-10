#include "mlx_native_denoiser.hpp"
#include "../../backends/metal_graph/mlx_bridge.hpp"
#include "../../backends/metal_graph/native_lora.hpp"
#include "../../runtime/streaming/source_lease.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace tc::qwen21::native_graph {
namespace {
using namespace metal_graph;
MatrixView matrix(Backend& backend,const Tensor& source) {
    if (source.dtype()!=mx::bfloat16)
        throw std::invalid_argument("native base denoiser requires original dense BF16 weights");
    auto result=import_mlx_matrix(backend,source);
    return {result.storage,result.rows,result.columns,result.row_stride_bytes};
}
MatrixView vector(Backend& backend,const Tensor& source,uint32_t columns) {
    if (source.ndim()==2) {
        auto result=matrix(backend,source);
        if (result.rows!=1 || result.columns!=columns)
            throw std::invalid_argument("native base denoiser norm vector shape mismatch");
        return result;
    }
    if (!source.is_available() || !source.buffer().ptr() || source.offset()<0 ||
        source.dtype()!=mx::bfloat16 || source.ndim()!=1 || source.shape(0)!=int(columns) ||
        source.strides()[0]!=1 || size_t(source.offset())>source.buffer_size() ||
        size_t(columns)*2>source.buffer_size()-size_t(source.offset()))
        throw std::invalid_argument("native base denoiser needs a produced contiguous BF16 norm vector");
    auto owner=std::make_shared<Tensor>(source);
    auto buffer=backend.import_buffer(const_cast<void*>(source.buffer().ptr()),std::move(owner));
    return {buffer.view(size_t(source.offset()),size_t(columns)*2,ScalarType::BF16),1,columns,size_t(columns)*2};
}
Projection projection(Backend& backend,const Weights& weights,const std::string& prefix) {
    if (weights.quantized(prefix) || weights.convrot(prefix) || weights.nvfp4(prefix))
        throw std::invalid_argument("native base denoiser does not silently dequantize source projections");
    Projection result{matrix(backend,weights.at(prefix+".weight")),{}};
    if (weights.has(prefix+".bias")) {
        result.bias=vector(backend,weights.at(prefix+".bias"),result.weight.rows);
    }
    return result;
}
void no_ffn_bias(const Weights& weights,const std::string& prefix) {
    for (const auto& suffix:{"gate_up","gate_layer","proj","out"})
        if (weights.has(prefix+suffix+".bias"))
            throw std::invalid_argument("native base FFN recipe cannot omit checkpoint bias");
}
DenoiserWeights import_weights(Backend& backend,const Weights& weights,DenoiserGeometry g) {
    validate_geometry(g);
    DenoiserWeights result;
    result.image_input=projection(backend,weights,"img_in");
    result.time_input=projection(backend,weights,"time_text_embed.timestep_embedder.linear_1");
    result.time_output=projection(backend,weights,"time_text_embed.timestep_embedder.linear_2");
    result.text_input=projection(backend,weights,"txt_in.in_layer");
    result.text_output=projection(backend,weights,"txt_in.out_layer");
    result.modulation=projection(backend,weights,"modulation.1");
    result.output_scale=projection(backend,weights,"norm_out.linear");
    result.output=projection(backend,weights,"proj_out");
    result.text_norm=vector(backend,weights.at("txt_in.text_norm.weight"),g.block.hidden);
    result.blocks.reserve(g.layers);
    for (uint32_t i=0;i<g.layers;++i) {
        const std::string p="transformer_blocks."+std::to_string(i);
        const std::string f=p+".img_mlp.";
        no_ffn_bias(weights,f);
        BlockWeights block;
        block.query=projection(backend,weights,p+".attn.to_q");
        block.key=projection(backend,weights,p+".attn.to_k");
        block.value=projection(backend,weights,p+".attn.to_v");
        block.attention_output=projection(backend,weights,p+".attn.to_out.0");
        block.query_norm=vector(backend,weights.at(p+".attn.norm_q.weight"),g.block.head_dim);
        block.key_norm=vector(backend,weights.at(p+".attn.norm_k.weight"),g.block.head_dim);
        if (weights.has(f+"gate_up.weight")) {
            auto fused=projection(backend,weights,f+"gate_up").weight;
            if (fused.rows!=2*g.block.ffn_width || fused.columns!=g.block.hidden)
                throw std::invalid_argument("native Qwen fused gate/up shape mismatch");
            block.gate=matrix_rows(fused,0,g.block.ffn_width);
            block.up=matrix_rows(fused,g.block.ffn_width,g.block.ffn_width);
        } else {
            block.gate=projection(backend,weights,f+"gate_layer").weight;
            block.up=projection(backend,weights,f+"proj").weight;
        }
        block.down=projection(backend,weights,f+"out").weight;
        result.blocks.push_back(std::move(block));
    }
    return result;
}
} // namespace

DenoiserWeights import_base_denoiser_weights(Backend& backend,const Weights& weights,DenoiserGeometry g) {
    if (weights.has_runtime_loras())
        throw std::invalid_argument("native base denoiser cannot omit runtime adapter projections");
    return import_weights(backend,weights,g);
}

DenoiserWeights import_runtime_lora_denoiser_weights(Backend& backend,const Weights& weights,DenoiserGeometry g) {
    const auto snapshot=weights.runtime_lora_snapshot();
    if (snapshot.bindings.empty() || snapshot.fp16_rank_math || !snapshot.native_metadata_compatible)
        throw std::invalid_argument("native runtime LoRA needs nonempty FP32-rank compatible bindings: "+
                                    snapshot.native_rejection_reason);
    struct Source { const streaming::SourceLease* lease; std::string logical_id; };
    std::map<size_t,Source> sources;
    for (const auto& binding:snapshot.bindings) {
        if (binding.ordered_terms.empty()) throw std::invalid_argument("native LoRA binding has no terms");
        for (const auto& term:binding.ordered_terms) {
            if (!term.lease_backed || !term.source_lease || term.source_logical_id.empty() ||
                term.source_role!="transformer")
                throw std::invalid_argument("native LoRA requires held-descriptor transformer provenance");
            // A copied descriptor/hash string does not grant verified source
            // trust. This method rejects metadata-only captured leases.
            const auto [source,inserted]=sources.emplace(term.source_adapter_ordinal,
                Source{term.source_lease.get(),term.source_logical_id});
            if (source->second.lease!=term.source_lease.get() || source->second.logical_id!=term.source_logical_id)
                throw std::invalid_argument("native LoRA ordinal names different source generations");
            if (inserted) {
                (void)term.source_lease->artifact_digest();
                (void)term.source_lease->file(term.source_logical_id);
                term.source_lease->revalidate_after_drain();
            }
            if (term.down_dtype!=term.down.dtype() || term.up_dtype!=term.up.dtype() ||
                (term.down_dtype!=mx::bfloat16 && term.down_dtype!=mx::float32) ||
                (term.up_dtype!=mx::bfloat16 && term.up_dtype!=mx::float32) ||
                term.down.ndim()!=2 || term.up.ndim()!=2 || term.down.shape(0)<=0 ||
                term.down.shape(0)>128 || term.up.shape(1)!=term.down.shape(0) ||
                term.output_start<0 || term.output_end<=term.output_start ||
                term.up.shape(0)!=term.output_end-term.output_start || !std::isfinite(term.scale))
                throw std::invalid_argument("native LoRA factor dtype/rank/output geometry is unsupported");
        }
    }
    auto result=import_weights(backend,weights,g);
    struct Target {
        uint32_t first,count,input;
        std::vector<OrderedLoRATerm>* terms;
    };
    std::map<std::string,std::vector<Target>> targets;
    auto projection_target=[&](const std::string& prefix,Projection& projection) {
        targets[prefix].push_back({0,projection.weight.rows,projection.weight.columns,&projection.lora});
    };
    projection_target("img_in",result.image_input);
    projection_target("time_text_embed.timestep_embedder.linear_1",result.time_input);
    projection_target("time_text_embed.timestep_embedder.linear_2",result.time_output);
    projection_target("txt_in.in_layer",result.text_input);
    projection_target("txt_in.out_layer",result.text_output);
    projection_target("modulation.1",result.modulation);
    projection_target("norm_out.linear",result.output_scale);
    projection_target("proj_out",result.output);
    for (uint32_t i=0;i<g.layers;++i) {
        const auto p="transformer_blocks."+std::to_string(i);
        const auto f=p+".img_mlp.";
        auto& block=result.blocks[i];
        projection_target(p+".attn.to_q",block.query);
        projection_target(p+".attn.to_k",block.key);
        projection_target(p+".attn.to_v",block.value);
        projection_target(p+".attn.to_out.0",block.attention_output);
        const bool fused=weights.has(f+"gate_up.weight");
        targets[f+(fused?"gate_up":"gate_layer")].push_back(
            {0,g.block.ffn_width,g.block.hidden,&block.gate_lora});
        targets[f+(fused?"gate_up":"proj")].push_back(
            {fused?g.block.ffn_width:0,g.block.ffn_width,g.block.hidden,&block.up_lora});
        targets[f+"out"].push_back({0,g.block.hidden,g.block.ffn_width,&block.down_lora});
    }
    for (const auto& binding:snapshot.bindings) {
        const auto found=targets.find(binding.prefix);
        if (found==targets.end()) throw std::invalid_argument("native LoRA target is not consumed: "+binding.prefix);
        for (const auto& term:binding.ordered_terms) {
            // Import the actual produced source once; native subviews preserve
            // its Data owner and pitch. No mx::slice/eval/cast/dense bank copy.
            const auto a_source=import_mlx_matrix(backend,term.down);
            const auto b_source=import_mlx_matrix(backend,term.up);
            const MatrixView a{a_source.storage,a_source.rows,a_source.columns,a_source.row_stride_bytes};
            const MatrixView b{b_source.storage,b_source.rows,b_source.columns,b_source.row_stride_bytes};
            uint64_t consumed=0;
            for (const auto& target:found->second) {
                if (a.columns!=target.input)
                    throw std::invalid_argument("native LoRA input does not match model projection: "+binding.prefix);
                const auto first=std::max(uint32_t(term.output_start),target.first);
                const auto end=std::min(uint32_t(term.output_end),target.first+target.count);
                if (first>=end) continue;
                const auto count=end-first;
                target.terms->push_back({a,matrix_rows(b,first-uint32_t(term.output_start),count),
                    term.scale,first-target.first,count,term.source_adapter_ordinal});
                consumed+=count;
            }
            if (consumed!=uint32_t(term.output_end-term.output_start))
                throw std::invalid_argument("native LoRA output range is not completely consumed: "+binding.prefix);
        }
    }
    return result;
}
} // namespace tc::qwen21::native_graph
