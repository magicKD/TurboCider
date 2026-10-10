#pragma once
#include "../../runtime/streaming/gguf_packed_bank.hpp"
#include "../../backends/ane_memory.hpp"
#include "../../backends/ane_ffn.hpp"
#include "source_registration.hpp"
#include "diagnostic_options.hpp"
#include <map>
#include <iostream>

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

// Preserve all three immutable affine planes when presenting a fused FFN to
// the runtime stager. Splitting only codes would reinterpret UINT32 as dense
// weights and silently discard the decoded K subgroup coefficients.
inline std::vector<ane::FfnWeight> gguf_ffn_sources(const Weights &weights,const std::string &prefix) {
    auto matrix=[&](const std::string &name,int rows,int columns) {
        require(weights.quantized(name) && !weights.convrot(name) && !weights.nvfp4(name),
                "Qwen GGUF FFN requires explicit affine packed source");
        const auto &values=weights.at(name+".weight"),&scales=weights.at(name+".scales"),&biases=weights.at(name+".biases");
        require(values.ndim()==2 && scales.shape()==mx::Shape{rows,columns/32} && biases.shape()==scales.shape() &&
            values.shape(0)==rows && values.dtype()==mx::uint32 && scales.dtype()==mx::float16 && biases.dtype()==mx::float16,
            "Qwen GGUF FFN packed geometry/typed coefficient mismatch");
        const int bits=values.shape(1)*32/columns;
        require((bits==4 || bits==8) && values.shape(1)*32==columns*bits,
                "Qwen GGUF FFN requires aligned original Q4/Q8 affine planes");
        return ane::FfnWeight{values,scales,biases,32,bits};
    };
    auto fused=matrix(prefix+"gate_up",24576,4096);
    auto values=mx::split(fused.values,2,0),scales=mx::split(*fused.scales,2,0),biases=mx::split(*fused.offsets,2,0);
    return {{values[0],scales[0],biases[0],32,fused.bits},
            {values[1],scales[1],biases[1],32,fused.bits},matrix(prefix+"out",4096,12288)};
}

inline std::unique_ptr<GgufComponent> load_gguf_component(const std::filesystem::path &path,
        Weights &weights,bool text,const Event &event,std::atomic<bool> &cancelled) {
    const auto import_start=Clock::now();
    streaming::SourceFileIdentity source;source.logical_id="qwen21-component";source.path=path;
    auto result=std::make_unique<GgufComponent>();
    result->ledger=std::make_unique<MemoryLedger>(uint64_t(16)<<30);
    streaming::GgufKImportOptions options;options.enabled=true;options.floating_dtype=mx::float16;
    if(const char *workers=std::getenv("TURBOCIDER_QWEN21_GGUF_DECODE_WORKERS")) {
        const auto value=std::string_view(workers);
        require(value.size()==1 && value[0]>='1' && value[0]<='8',"Qwen GGUF decode workers must be 1..8");
        options.decode_workers=uint32_t(value[0]-'0');
    }
    if(text)options.include_tensor=[](std::string_view key){return !key.starts_with("output.");};
    const char *flag=std::getenv("TURBOCIDER_QWEN21_GGUF_VERIFY_IMPORT_OVERLAP");
    require(!flag || std::string_view(flag)=="0" || std::string_view(flag)=="1",
            "Qwen GGUF verify/import overlap requires 0 or 1");
    const bool overlap=flag && std::string_view(flag)=="1";
    const bool registered=source_registration_enabled();
    const int read_mib=gguf_read_buffer_mib(std::getenv("TURBOCIDER_QWEN21_GGUF_READ_BUFFER_MIB"));
    require(read_mib>0,"Qwen GGUF read buffer requires1,4 or8 MiB");
    const uint64_t read_bytes=uint64_t(read_mib)<<20;
    auto admission=[&](const streaming::GgufPackedBankMetrics &metrics) {
        const auto observed=ane::observe_runtime_memory(mx::get_active_memory());
        const uint64_t growth=metrics.planned_packed_capacity_bytes+read_bytes+(overlap ? uint64_t(1)<<20 : 0);
        const auto decision=ane::admit_memory(observed,{uint64_t(4)<<30,observed.physical_bytes},0,growth);
        require(decision.allowed(),"Qwen21 GGUF source admission declined: "+ane::memory_denial_reason(decision.denial,observed));
    };
    if(overlap) {
        streaming::PendingSourceVerification pending({source},&cancelled,registered);
        result->bank=streaming::GgufPackedBank::load_verifying(pending,"qwen21-component",*result->ledger,
            weights,read_bytes,options,admission,&cancelled,event);
    } else {
        auto lease=registered ? streaming::SourceLease::capture_registered({source},&cancelled) :
                                streaming::SourceLease::capture_verified({source},&cancelled);
        result->bank=std::make_unique<streaming::GgufPackedBank>(lease,"qwen21-component",*result->ledger,
            read_bytes,true,0,6,options);
        admission(result->bank->metrics());result->bank->load(weights,&cancelled,event);
    }
    if(text) {
        weights.remap_keys(gguf_text_key);
        // Embedding stays packed too. Prompt assembly gathers only consumed
        // codes/scales/biases rows before GPU decode, not the full vocabulary.
    } else {
        weights.remap_keys([](const std::string &key) {
            constexpr std::string_view p="model.diffusion_model.";
            require(key.starts_with(p),"unsupported Qwen21 GGUF denoiser tensor namespace");return key.substr(p.size());
        });
    }
    weights.materialize();result->bank->check_unchanged();
    {
        const auto m=result->bank->metrics();
        std::cerr<<"{\"qwen_gguf_verified_import\":{\"component\":\""<<(text ? "encoder" : "dit")
            <<"\",\"source_sha256\":\""<<m.source_sha256<<"\",\"verification_bytes\":"<<m.verification_bytes
            <<",\"verification_cache_hits\":"<<m.verification_cache_hits
            <<",\"registration_enabled\":"<<(registered ? "true" : "false")
            <<",\"registration_cache_hits\":"<<m.registration_cache_hits
            <<",\"registration_records_written\":"<<m.registration_records_written
            <<",\"read_buffer_capacity_bytes\":"<<m.read_buffer_capacity_bytes
            <<",\"first_registration_seconds\":"<<(registered && m.verification_bytes ? m.verification_seconds : 0)
            <<",\"cpu_verify_import_schedule\":"<<(overlap ? "true" : "false")
            <<",\"verified_before_publication\":true,\"physical_overlap_proved\":false"
            <<",\"verification_seconds\":"<<m.verification_seconds<<",\"import_wall_seconds\":"
            <<std::chrono::duration<double>(Clock::now()-import_start).count()<<"}}\n";
    }
    return result;
}

} // namespace tc::qwen21
