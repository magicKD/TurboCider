// Reuse the existing deterministic tiny weights, padded source guards, exact
// MLX Transformer configuration and error diagnostics without changing its
// historical source/binary. The old entry point is never invoked here.
#define main turbocider_base_denoiser_component_entry_unused
#include "native_qwen_denoiser_component.cpp"
#undef main

#include "../../native/models/qwen21/native_memory_plan.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"

#include <fstream>

namespace {
constexpr ng::DenoiserGeometry lora_geometry{{hidden,head_count,dimensions,ffn},layers,64};
constexpr uint32_t target_rows=4,conditioning_rows=3;
constexpr double lora_rrms_limit=0.01;
constexpr const char* lora_source_revision="tiny-qwen-all-targets-bf16-with-bias-v1";
struct TargetShape { std::string prefix; int input,output,rank; };
std::vector<TargetShape> target_shapes() {
    std::vector<TargetShape> values{
        {"img_in",64,hidden,16},{"time_text_embed.timestep_embedder.linear_1",256,hidden,128},
        {"time_text_embed.timestep_embedder.linear_2",hidden,hidden,16},
        {"txt_in.in_layer",hidden,hidden,128},{"txt_in.out_layer",hidden,hidden,16},
        {"modulation.1",hidden,4*hidden,128},{"norm_out.linear",hidden,hidden,16},
        {"proj_out",hidden,64,128}};
    for(int i=0;i<layers;++i) {
        const auto p="transformer_blocks."+std::to_string(i);
        for(const auto* suffix:{"to_q","to_k","to_v","to_out.0"})
            values.push_back({p+".attn."+suffix,hidden,hidden,values.size()%2?16:128});
        values.push_back({p+".img_mlp.gate_layer",hidden,ffn,128});
        values.push_back({p+".img_mlp.proj",hidden,ffn,16});
        values.push_back({p+".img_mlp.out",ffn,hidden,128});
    }
    return values;
}
struct FactorGuard { Tensor owner; mg::MatrixView view; uint64_t digest; };
struct LoRAFixture {
    mg::Backend source_backend;
    Fixture generated;
    Weights weights;
    std::vector<std::string> base_names;
    std::vector<Tensor> base_arrays;
    std::optional<Tensor> latent,text;
    std::vector<FactorGuard> factors;
    std::shared_ptr<const streaming::SourceLease> adapter_source;
    std::string adapter_revision;
    std::filesystem::path adapter_path;
    explicit LoRAFixture(const std::filesystem::path& owned):generated(source_backend) {
        make_weights(weights,generated);
        base_names=weights.sorted_keys();
        for(const auto& key:base_names) base_arrays.push_back(weights.at(key));
        uint32_t salt=14001;
        for(const auto& target:target_shapes()) {
            if(target.prefix.find(".img_mlp.")!=std::string::npos) continue;
            // Cover bias-last on every global and attention projection. FFN
            // bias stays absent, matching the native model's explicit ABI.
            base_names.push_back(target.prefix+".bias");
            base_arrays.push_back(mx::reshape(generated.padded(target.prefix+"-bias",1,target.output,
                salt++,0.05f),{target.output}));
        }
        mx::eval(base_arrays); weights.bind_arrays(base_names,base_arrays);
        latent=mx::expand_dims(generated.padded("lora-latent",target_rows,channels,7103,0.5f),0);
        text=mx::expand_dims(generated.padded("lora-text",conditioning_rows,hidden,7207,0.7f),0);
        mx::eval(*latent,*text);
        std::unordered_map<std::string,Tensor> tensors;
        for(const auto& target:target_shapes()) {
            // Distinct salts and ranks ensure gate/up do NOT share A. Mixed
            // source ranks exercise both real local rank classes in one run.
            auto a=generated.padded(target.prefix+"-A",target.rank,target.input,salt++,
                0.2f*std::sqrt(3.f/target.input));
            auto b=generated.padded(target.prefix+"-B",target.output,target.rank,salt++,
                0.35f*std::sqrt(3.f/target.rank));
            const auto stem="transformer."+target.prefix;
            tensors.emplace(stem+".lora_A.default.weight",a);
            tensors.emplace(stem+".lora_B.default.weight",b);
            if(target.prefix.ends_with(".img_mlp.out"))
                tensors.emplace(stem+".alpha",Tensor(float(target.rank*2),mx::float32));
        }
        adapter_path=owned/"generated-adapter.safetensors";
        mx::save_safetensors(adapter_path.string(),std::move(tensors),{{"format","pt"}});
        check(std::filesystem::file_size(adapter_path)<4*1024*1024,"generated adapter exceeded bounded tiny size");
        streaming::SourceFileIdentity file; file.logical_id="generated-adapter"; file.path=adapter_path;
        adapter_source=streaming::SourceLease::capture_verified({file});
        adapter_revision=std::string(adapter_source->artifact_digest())+":strength-0.75:ordered-fp32-v1";
        mx::synchronize(); verify();
    }
    void bind_adapter() {
        std::atomic<bool> cancelled=false;
        const Event event=[](const std::string&,int,int){};
        const auto count=weights.apply_loras_lease(adapter_source,{"generated-adapter"},
            {{adapter_path.string(),0.75f,"transformer"}},"transformer",event,cancelled,true,true);
        check(count==target_shapes().size(),"loader did not bind every generated logical target");
        weights.materialize(); mx::synchronize(); adapter_source->revalidate_after_drain();
        const auto snapshot=weights.runtime_lora_snapshot();
        check(snapshot.native_metadata_compatible && !snapshot.fp16_rank_math,"unexpected native adapter metadata rejection");
        size_t terms=0;
        for(const auto& binding:snapshot.bindings) for(const auto& term:binding.ordered_terms) {
            check(term.lease_backed && term.source_adapter_ordinal==0,"adapter source lineage was lost");
            for(const auto& factor:{term.down,term.up}) {
                auto view=matrix(source_backend,factor);
                factors.push_back({factor,view,hash_bytes(view)});
            }
            ++terms;
        }
        check(terms==22,"fixture must cover 8 globals and 2*7 block targets");
        verify();
    }
    void reset_base() {
        weights.bind_arrays(base_names,base_arrays);
        check(!weights.has_runtime_loras() && weights.runtime_lora_snapshot().bindings.empty(),
              "base rebind retained adapter state");
        verify();
    }
    void verify() const {
        generated.verify();
        for(const auto& factor:factors)
            check(hash_bytes(factor.view)==factor.digest,"loaded adapter source was modified");
        if(adapter_source) adapter_source->revalidate_after_drain();
    }
};
size_t terms(const ng::DenoiserWeights& weights) {
    size_t count=0;
    for(const auto* projection:{&weights.image_input,&weights.time_input,&weights.time_output,
         &weights.text_input,&weights.text_output,&weights.modulation,&weights.output_scale,&weights.output}) {
        check(projection->lora.size()==1,"global target was omitted or duplicated"); count+=projection->lora.size();
    }
    for(const auto& block:weights.blocks) {
        for(const auto* projection:{&block.query,&block.key,&block.value,&block.attention_output}) {
            check(projection->lora.size()==1,"attention target was omitted or duplicated"); count+=projection->lora.size();
        }
        for(const auto* values:{&block.gate_lora,&block.up_lora,&block.down_lora}) {
            check(values->size()==1,"FFN target was omitted or duplicated"); count+=values->size();
        }
        check(block.gate_lora[0].a.rows==128 && block.up_lora[0].a.rows==16 &&
              block.gate_lora[0].a.storage.buffer().allocation_identity()!=block.up_lora[0].a.storage.buffer().allocation_identity(),
              "gate and up A incorrectly shared");
        check(block.down_lora[0].a.columns==ffn && block.down_lora[0].effective_scale==1.5f,
              "down LoRA input width or tensor-alpha scale changed");
    }
    return count;
}
struct LoRATrial {
    std::shared_ptr<LoRAFixture> source;
    std::unique_ptr<ng::NativePhaseMemory> phase;
    std::unique_ptr<mg::Backend> backend;
    std::optional<ng::NativeDenoiser> model;
    std::unique_ptr<ng::DenoiserRun> run;
    void release_native() { run.reset(); model.reset(); backend.reset(); }
};
struct LoRAOutcome { std::vector<uint16_t> latents; size_t scratch=0; uint32_t backings=0; };
void complete(const mg::Completion& status) {
    check(status.ok && status.physically_complete && !status.timed_out &&
          status.submitted_buffers==status.completed_buffers,"native physical completion contract failed");
}
LoRAOutcome run_lora_route(const std::shared_ptr<LoRAFixture>& source,bool adapted,uint64_t generation) {
    source->verify();
    const std::vector<float> schedule{1.f,0.7f,0.f};
    // This is the actual Transformer and actual BF16 Euler, not a replacement
    // model or a LoRA CPU approximation. Prefix K/V must cover both layers.
    Tensor oracle=*source->latent;
    {
        qwen21::Transformer reference(source->weights,mlx_config());
        for(size_t step=0;step+1<schedule.size();++step) {
            auto noise=reference.forward(oracle,*source->text,schedule[step],2,2,true);
            oracle=oracle+noise*Tensor(schedule[step+1]-schedule[step],oracle.dtype());
            mx::eval(oracle);
            check(oracle.dtype()==mx::bfloat16 && mx::all(mx::isfinite(oracle)).item<bool>(),"reference Euler is not finite BF16");
            check(reference.cached_layers()==layers,"reference prefix missed a layer");
        }
    }
    mx::synchronize(); mx::clear_cache();
    auto trial=std::make_unique<LoRATrial>(); trial->source=source;
    LoRAOutcome outcome;
    try {
        ng::NativePhaseShape shape{lora_geometry,target_rows,conditioning_rows,{}};
        shape.lora_max_rank=adapted?128:0; shape.lora_ffn=adapted;
        trial->phase=std::make_unique<ng::NativePhaseMemory>(shape,lora_source_revision,
            observe_process_memory(),ng::NativePhaseLimits{},0x514c4f524154494eULL,generation);
        outcome.backings=trial->phase->plan().backing_count;
        mg::BackendOptions options; options.allocation=trial->phase->hooks();
        trial->backend=std::make_unique<mg::Backend>(std::move(options));
        auto imported=adapted?ng::import_runtime_lora_denoiser_weights(*trial->backend,source->weights,lora_geometry):
            ng::import_base_denoiser_weights(*trial->backend,source->weights,lora_geometry);
        if(adapted) {
            check(terms(imported)==22,"adapted importer lost a target");
            const auto& query=imported.blocks[0].query.lora;
            const auto input_alias=matrix(*trial->backend,*source->text);
            // This deliberately INVALID recording never reads these bytes:
            // input/output overlap must reject before a graph node or PSO.
            // The rank view has sufficient physical extent but aliases A too.
            const auto& a=query[0].a;
            const mg::MatrixView bad_low{a.storage.buffer().view(a.storage.offset_bytes(),
                conditioning_rows*128*4,mg::ScalarType::FP32),conditioning_rows,128,128*4};
            mg::CommandGraph rejected;
            mg::NativeLoRA unprepared;
            rejects<std::invalid_argument>([&] {
                unprepared.encode_ordered_corrections(rejected,input_alias,query,
                    input_alias,input_alias,bad_low);
            },"aliased LoRA input/output was accepted");
            check(rejected.empty(),"rejected alias partially appended a graph");
        }
        ng::NativeDenoiserExecutionOptions execution;
        execution.allocation=trial->phase->hooks();
        execution.schedule_variant=ng::DenoiserScheduleVariant::GeneratedTiny;
        execution.adapter_revision=adapted?source->adapter_revision:std::string{};
        trial->model.emplace(ng::NativeDenoiser::prepare(*trial->backend,std::move(imported),lora_source_revision,lora_geometry,execution));
        ng::DenoiserInput input;
        input.latents=matrix(*trial->backend,*source->latent); input.text=matrix(*trial->backend,*source->text);
        input.height=2; input.width=2; input.sigmas=schedule;
        input.identity={lora_source_revision,"same-tiny-conditioning",execution.adapter_revision,generation};
        auto wrong=input; wrong.identity.adapter_revision="mismatched-adapter";
        rejects<std::invalid_argument>([&]{(void)trial->model->begin(wrong);},"adapter revision mismatch accepted");
        wrong=input; wrong.identity.source_revision="mismatched-base";
        rejects<std::invalid_argument>([&]{(void)trial->model->begin(wrong);},"base revision mismatch accepted");
        trial->run=trial->model->begin(input);
        outcome.scratch=trial->run->memory().lora_scratch_bytes;
        check(adapted?outcome.scratch>0:outcome.scratch==0,"LoRA scratch allocation does not match actual terms");
        complete(trial->run->physical_status()); complete(trial->run->drain(std::chrono::milliseconds(0)));
        rejects<std::logic_error>([&]{(void)trial->run->final_latents();},"early final output accepted");
        rejects<std::invalid_argument>([&]{(void)trial->run->drain(std::chrono::milliseconds(-1));},"negative drain accepted");
        for(size_t step=0;step<2;++step) {
            const auto status=trial->run->step(); complete(status);
            check(status.submitted_buffers>0 && trial->run->completed_steps()==step+1,"step receipt count mismatch");
            complete(trial->run->physical_status()); complete(trial->run->drain(std::chrono::milliseconds(0)));
            source->verify();
        }
        check(trial->run->state()==ng::DenoiserState::Complete,"native did not reach Complete");
        auto output=trial->run->final_latents();
        const auto error=compare(output,oracle);
        std::cout << std::setprecision(10) << "{\"case\":\"" << (adapted?"lora-all-targets":"base")
            << "\",\"generation\":" << generation << ",\"targets\":" << (adapted?22:0)
            << ",\"ranks\":[16,128],\"layers\":2,\"steps\":2,\"rrms\":" << error.rrms
            << ",\"max_abs\":" << error.max_abs << ",\"rrms_limit\":" << lora_rrms_limit
            << ",\"lora_scratch_bytes\":" << outcome.scratch << ",\"qualification\":\"generated-component-only\"";
        print_error_diagnostics(error); std::cout << "}\n";
        check(error.rrms<=lora_rrms_limit,"native runtime LoRA failed fixed 1% RRMS; investigate before widening");
        for(uint32_t row=0;row<output.rows;++row) {
            const auto* values=reinterpret_cast<const uint16_t*>(static_cast<const std::byte*>(output.storage.contents())+row*output.row_stride_bytes);
            outcome.latents.insert(outcome.latents.end(),values,values+output.columns);
        }
        rejects<std::logic_error>([&]{(void)trial->run->step();},"step after Complete accepted");
        // Release all imported wrappers and native-owned buffers before phase
        // finish. These external source owners remain alive in source baseline.
        input={}; wrong={}; output={};
        trial->release_native();
        trial->phase->finish_after_release(true);
        const auto report=trial->phase->context().take_report();
        const auto& memory=report.metrics;
        check(memory.execution_state=="succeeded" && !memory.tainted && !memory.worker_quarantined &&
            memory.ledger_storage_bytes==0 && memory.ledger_reserved_bytes==0 && memory.ledger_pending_release_bytes==0 &&
            memory.ledger_cached_bytes==0 && memory.ledger_unknown_bytes==0 && memory.ledger_storage_count==0,
            "native LoRA phase did not release its complete ledger");
        std::cout << "native_lora_phase=PASS generation=" << generation << " zero_ledger=true source_padding_unchanged=true\n";
        return outcome;
    } catch (...) {
        bool drained=!trial->run;
        try { if(trial->run) drained=trial->run->drain(std::chrono::seconds(30)).physically_complete; }
        catch (...) { drained=false; }
        if(drained) {
            trial->release_native();
            if(trial->phase) {
                const auto disposition=trial->phase->fail_after_drain(true,"generated LoRA fixture failed");
                if(disposition!=MemoryFailureDisposition::Clean && disposition!=MemoryFailureDisposition::None)
                    (void)trial.release();
            }
        } else (void)trial.release(); // Retain source/lease/phase under pending native work.
        throw;
    }
}
} // namespace

int main(int argc,char** argv) try {
    if(argc==1) {
        std::cout << "usage: native-qwen-lora-denoiser-component --device /absolute/fresh-owned-output\n"
                     "Generated tiny base -> all-target LoRA -> base only; no real models or performance qualification.\n";
        return 0;
    }
    check(argc==3 && std::string_view(argv[1])=="--device","explicit --device and fresh owned output required");
    reject_transformer_overrides();
    const std::filesystem::path owned(argv[2]);
    check(owned.is_absolute() && owned!=owned.root_path() && !std::filesystem::exists(owned) &&
          !std::filesystem::is_symlink(std::filesystem::symlink_status(owned)),"fixture output must be a fresh absolute directory");
    check(std::filesystem::create_directory(owned),"cannot create fresh fixture output (parent must exist)");
    std::ofstream(owned/"OWNED-generated-lora-fixture.txt") << "Generated tiny adapter only; no model downloads or full checkpoints.\n";
    auto source=std::make_shared<LoRAFixture>(owned);
    const auto before=run_lora_route(source,false,1);
    source->bind_adapter();
    const auto adapted=run_lora_route(source,true,2);
    check(before.latents!=adapted.latents,"adapter had no observable final latent effect");
    source->reset_base();
    const auto after=run_lora_route(source,false,3);
    check(before.latents==after.latents && before.scratch==0 && after.scratch==0 && before.backings==after.backings,
          "base -> LoRA -> base changed base numerics or allocated unused LoRA storage");
    source->verify();
    std::cout << "native_lora_full_dit=PASS base_lora_base=true all_targets=22 mixed_ranks=16,128 real_models=0 e2e_images=0 performance_qualified=false\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr << "native_lora_full_dit=FAIL " << error.what() << '\n';
    return 1;
}
