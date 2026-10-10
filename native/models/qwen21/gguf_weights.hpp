#pragma once
#include "../../runtime/streaming/gguf_packed_bank.hpp"
#include "../../backends/ane_memory.hpp"
#include <map>

namespace tc::qwen21 {

inline std::filesystem::path component_source(const std::filesystem::path &root,
        const char *bf16,const char *gguf) {
    if(std::filesystem::is_regular_file(root/bf16))return root/bf16;
    require(std::filesystem::is_regular_file(root/gguf),std::string("missing Qwen Image 2.1 component: ")+bf16+" or "+gguf);
    return root/gguf;
}

inline std::string gguf_text_key(const std::string &key) {
    if(key.starts_with("token_embd."))return "model.embed_tokens."+key.substr(11);
    if(key.starts_with("output_norm."))return "model.norm."+key.substr(12);
    require(key.starts_with("blk."),"unsupported Qwen3-VL GGUF text tensor: "+key);
    const auto dot=key.find('.',4);
    require(dot!=std::string::npos && dot>4,"invalid Qwen3-VL GGUF layer name");
    for(size_t i=4;i<dot;++i)require(key[i]>='0' && key[i]<='9',"invalid Qwen3-VL GGUF layer ordinal");
    const auto field_end=key.find('.',dot+1);
    require(field_end!=std::string::npos,"missing Qwen3-VL GGUF parameter suffix");
    static const std::map<std::string,std::string> fields{
        {"attn_q","self_attn.q_proj"},{"attn_k","self_attn.k_proj"},{"attn_v","self_attn.v_proj"},
        {"attn_output","self_attn.o_proj"},{"attn_q_norm","self_attn.q_norm"},{"attn_k_norm","self_attn.k_norm"},
        {"attn_norm","input_layernorm"},{"ffn_norm","post_attention_layernorm"},
        {"ffn_gate","mlp.gate_proj"},{"ffn_up","mlp.up_proj"},{"ffn_down","mlp.down_proj"}};
    const auto found=fields.find(key.substr(dot+1,field_end-dot-1));
    require(found!=fields.end(),"unsupported Qwen3-VL GGUF projection: "+key);
    return "model.layers."+key.substr(4,dot-4)+"."+found->second+key.substr(field_end);
}

struct GgufComponent {
    // Bank references the ledger; declaration order preserves that lifetime.
    std::unique_ptr<MemoryLedger> ledger;
    std::unique_ptr<streaming::GgufPackedBank> bank;
};

inline std::unique_ptr<GgufComponent> load_gguf_component(const std::filesystem::path &path,
        Weights &weights,bool text,const Event &event,std::atomic<bool> &cancelled) {
    streaming::SourceFileIdentity source;source.logical_id="qwen21-component";source.path=path;
    auto lease=streaming::SourceLease::capture_verified({source},&cancelled);
    auto result=std::make_unique<GgufComponent>();
    result->ledger=std::make_unique<MemoryLedger>(uint64_t(16)<<30);
    streaming::GgufKImportOptions options;options.enabled=true;options.floating_dtype=mx::float16;
    if(text)options.include_tensor=[](std::string_view key){return !key.starts_with("output.");};
    result->bank=std::make_unique<streaming::GgufPackedBank>(lease,"qwen21-component",*result->ledger,
        uint64_t(1)<<20,true,0,6,options);
    const auto observed=ane::observe_runtime_memory(mx::get_active_memory());
    uint64_t growth=result->bank->metrics().planned_packed_capacity_bytes+(uint64_t(1)<<20);
    if(text) {
        const auto &embedding=result->bank->directory().tensor("token_embd.weight");
        growth=gguf::checked_add(growth,gguf::checked_mul(embedding.elements,2));
    }
    const auto decision=ane::admit_memory(observed,{uint64_t(4)<<30,observed.physical_bytes},0,growth);
    require(decision.allowed(),"Qwen21 GGUF source admission declined: "+ane::memory_denial_reason(decision.denial,observed));
    result->bank->load(weights,&cancelled,event);
    if(text) {
        weights.remap_keys(gguf_text_key);
        // Only this table is expanded for the current prompt assembler;
        // attention/FFN matrices stay packed. Not a full dense checkpoint.
        weights.dequantize({"model.embed_tokens"});
    } else {
        weights.remap_keys([](const std::string &key) {
            constexpr std::string_view p="model.diffusion_model.";
            require(key.starts_with(p),"unsupported Qwen21 GGUF denoiser tensor namespace");return key.substr(p.size());
        });
    }
    weights.materialize();result->bank->check_unchanged();return result;
}

} // namespace tc::qwen21
