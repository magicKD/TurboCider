#include "backends/mlx.hpp"
#include "runtime/streaming/source_lease.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>

namespace {
template<class F> void rejects(F&& function) {
    bool failed=false;
    try { function(); } catch(const std::exception&) { failed=true; }
    assert(failed);
}
tc::Tensor matrix(int rows,int columns,float value) {
    std::vector<float> data(size_t(rows)*columns,value);
    return tc::Tensor(data.data(),{rows,columns},tc::mx::float32);
}
void base(tc::Weights& weights) {
    weights.bind_arrays({"transformer_blocks.0.img_mlp.gate_up.weight",
                         "transformer_blocks.0.img_mlp.out.weight"},
                        {matrix(8,2,0.25f),matrix(2,4,0.5f)});
}
void adapter(const std::filesystem::path& path,float value,bool malformed=false,
             const std::unordered_map<std::string,std::string>& metadata={}) {
    std::unordered_map<std::string,tc::Tensor> values;
    for(const auto* target:{"gate_layer","proj"}) {
        const std::string stem=std::string("transformer.transformer_blocks.0.img_mlp.")+target;
        values.emplace(stem+".lora_A.default.weight",matrix(2,2,value));
        values.emplace(stem+".lora_B.default.weight",matrix(4,2,value+1));
    }
    const std::string down="transformer_blocks.0.img_mlp.out";
    values.emplace(down+".lora_A.weight",matrix(2,4,value+2));
    if(!malformed) values.emplace(down+".lora_B.weight",matrix(2,2,value+3));
    values.emplace(down+".alpha",tc::Tensor(4.f,tc::mx::float32));
    tc::mx::save_safetensors(path.string(),std::move(values),metadata);
}
}

// Generated tiny file/CPU loader fixture. No model, Metal graph or GPU work.
int main(int argc,char** argv) {
    assert(argc==2);
    tc::mx::set_default_device(tc::mx::Device(tc::mx::Device::cpu));
    const std::filesystem::path root(argv[1]);
    std::filesystem::create_directories(root);
    const auto path=root/"adapter.safetensors";
    adapter(path,1);
    std::atomic<bool> cancelled=false;
    const tc::Event event=[](const std::string&,int,int){};
    tc::streaming::SourceFileIdentity file;
    file.logical_id="adapter-1"; file.path=path;
    auto lease=tc::streaming::SourceLease::capture_verified({file},&cancelled);
    std::vector<tc::LoRAAsset> adapters{{"missing-unused",1.f,"text"},{path.string(),0.75f,"transformer"}};
    tc::Weights weights; base(weights);
    const auto applied=weights.apply_loras_lease(lease,{"unused","adapter-1"},adapters,
        "transformer",event,cancelled);
    assert(applied==3);
    auto snapshot=weights.runtime_lora_snapshot();
    assert(snapshot.native_metadata_compatible && !snapshot.fp16_rank_math && snapshot.bindings.size()==2);
    const auto& terms=snapshot.bindings[0].ordered_terms;
    assert(snapshot.bindings[0].prefix=="transformer_blocks.0.img_mlp.gate_up" && terms.size()==2);
    assert(terms[0].output_start==0 && terms[0].output_end==4);
    assert(terms[1].output_start==4 && terms[1].output_end==8);
    for(const auto& binding:snapshot.bindings) for(const auto& term:binding.ordered_terms) {
        assert(term.source_adapter_ordinal==1 && term.lease_backed && term.source_lease==lease);
        assert(term.source_logical_id=="adapter-1" && term.source_role=="transformer");
        assert(term.source_strength==0.75f);
        assert(term.down_dtype==tc::mx::float32 && term.up_dtype==tc::mx::float32);
        assert(term.scale==(binding.prefix.ends_with(".out")?1.5f:0.75f));
    }
    std::filesystem::rename(path,root/"held-original.safetensors");
    adapter(path,9);
    weights.materialize();
    assert(terms[0].down.data<float>()[0]==1.f);
    rejects([&]{lease->revalidate_after_drain();});
    weights.clear();
    // Snapshot owns both factors and the exact held descriptor generation.
    assert(!weights.has_runtime_loras() && terms[0].down.data<float>()[0]==1.f);

    tc::Weights pathname; base(pathname);
    pathname.apply_loras({{path.string(),1.f,"transformer"}},"transformer",event,cancelled,true,true);
    const auto untrusted=pathname.runtime_lora_snapshot();
    assert(!untrusted.bindings[0].ordered_terms[0].lease_backed);
    assert(!untrusted.bindings[0].ordered_terms[0].source_lease);

    adapter(path,1,true);
    rejects([&]{pathname.apply_loras({{path.string(),1.f,"transformer"}},"transformer",event,cancelled,true,true);});
    assert(!pathname.has_runtime_loras() && pathname.sorted_keys().empty());
    base(pathname);
    rejects([&]{pathname.apply_loras_lease(lease,{},adapters,"transformer",event,cancelled);});
    assert(pathname.sorted_keys().empty());

    adapter(path,1,false,{{"lora_adapter_metadata",R"({"use_rslora":true})"}});
    base(pathname);
    pathname.apply_loras({{path.string(),1.f,"transformer"}},"transformer",event,cancelled,true,true);
    assert(!pathname.runtime_lora_snapshot().native_metadata_compatible);
    // Legacy binding still uses its existing tensor-alpha semantics.
    assert(pathname.runtime_lora_snapshot().bindings[0].ordered_terms[0].scale==1.f);
    std::cout << "PASS MLX LoRA lease: shared resolver, fused ranges, source ordinals, path replacement, snapshot ownership and atomic cleanup\n";
}
