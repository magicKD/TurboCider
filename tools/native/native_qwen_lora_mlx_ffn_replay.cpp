// One captured input, warm AB / record AB / record BA: three MLX and three native full FFNs.
// No extra oracle execution; this tool does not reuse evaluated MLX output arrays.
// Default CPU contract never creates a device or opens a model artifact.
#include "native_qwen_lora_ffn_replay.hpp"

#ifndef TC_MLX_FFN_REPLAY_SOURCE_SHA
#define TC_MLX_FFN_REPLAY_SOURCE_SHA "unrecorded-host-only"
#endif
#ifndef TC_MLX_FFN_REPLAY_EXPECTED_BUILD_ID
#define TC_MLX_FFN_REPLAY_EXPECTED_BUILD_ID "unbound-host-only"
#endif
using namespace tc;
namespace {
using namespace tc::qwen_lora_replay;

constexpr const char* candidate_id="native-lora-mlx-layer0-component";
enum class Arm { Mlx, Native };
constexpr std::array<std::array<Arm,2>,3> arm_order{{{Arm::Mlx,Arm::Native},{Arm::Mlx,Arm::Native},{Arm::Native,Arm::Mlx}}};
enum class Stage { Gate, Up, GateCorrection, UpCorrection, SiluUp, Down, DownCorrection };
constexpr std::array native_stages{Stage::Gate,Stage::Up,Stage::GateCorrection,Stage::UpCorrection,Stage::SiluUp,Stage::Down,Stage::DownCorrection};
constexpr size_t dispatches(Stage s) {return s==Stage::GateCorrection || s==Stage::UpCorrection || s==Stage::DownCorrection?2:1;}
constexpr size_t native_dispatches() {size_t n=0;for(auto s:native_stages)n+=dispatches(s);return n;}
struct FreshOutputs {
    std::vector<std::weak_ptr<const void>> retired;
    void check(const std::shared_ptr<const void>& owner,const std::vector<std::shared_ptr<const void>>& sources) {
        require(bool(owner),"MLX output lacks actual Data owner");
        for(const auto& old:retired)require(owner.owner_before(old) || old.owner_before(owner),"MLX reused previously evaluated output Data");
        for(const auto& source:sources)require(owner.owner_before(source) || source.owner_before(owner),"MLX output aliases a retained immutable source");
        retired.emplace_back(owner);
    }
    void require_retired() const {for(const auto& old:retired)require(old.expired(),"previous MLX output payload still retained");}
};
struct Trial {
    std::shared_ptr<const streaming::SourceLease> base_lease,adapter_lease;
    Weights weights;std::vector<Tensor> tensors,mlx_outputs;
    std::vector<std::shared_ptr<const void>> source_owners;
    std::shared_ptr<MlxLeaseFdReader> input_reader;
    std::unique_ptr<MemoryExecutionContext> memory;
    std::unique_ptr<mg::MemoryBridge> bridge;
    std::unique_ptr<mg::Backend> backend;
    std::vector<mg::MatrixView> sources,scratch;
    std::array<mg::OrderedLoRATerm,3> terms;
    std::optional<mg::NativeLinear> gate_up,down;
    std::optional<mg::NativeLoRA> corrections;
    std::optional<mg::Ticket> pending;
    std::optional<mg::MatrixView> mlx_view;
    qwen21::runtime_ffn::Function mlx_function;
    FreshOutputs freshness;
    bool uncertain=false;
    const char* pending_domain="none";
    void retire_mlx() {mlx_view.reset();mlx_outputs.clear();freshness.require_retired();}
    void release_native() {
        mlx_view.reset();mlx_outputs.clear();pending.reset();terms={};sources.clear();scratch.clear();
        corrections.reset();gate_up.reset();down.reset();backend.reset();
    }
};
void run(const std::filesystem::path& checkpoint,const std::filesystem::path& adapter,
         const std::filesystem::path& capture,const std::filesystem::path& output) {
    const auto build=build_identity(TC_MLX_FFN_REPLAY_EXPECTED_BUILD_ID);
    for(const auto& p:{checkpoint,adapter,capture,output})require(p.is_absolute() && p==p.lexically_normal(),"explicit normalized absolute paths required");
    require(capture.filename()==std::string(capture_id)+"-replay" && !std::filesystem::exists(output) && std::filesystem::is_directory(output.parent_path()),"wrong capture or output already exists");
    const auto* pin=qwen21::viggle_v021_adapter(adapter.filename().string());require(pin && pin->rank=="r128","only pinned Viggle r128 accepted");
    auto trial=std::make_unique<Trial>();NSMutableArray* records=[NSMutableArray array];
    try {
        const auto source_start=Time::now();std::atomic<bool> cancelled{false};
        const auto input_path=capture/"layer-0-step-5.safetensors";
        require(!std::filesystem::is_symlink(input_path),"capture symlink is not accepted");
        trial->base_lease=streaming::SourceLease::capture({{.logical_id="checkpoint",.path=checkpoint},
            {.logical_id="complete",.path=capture/"complete.json"},{.logical_id="input",.path=input_path}});
        const auto identity=held_fingerprint(*trial->base_lease);
        const auto hash_start=Time::now();
        trial->adapter_lease=streaming::SourceLease::capture_verified({{.logical_id="adapter",.path=adapter,.content_digest=std::string(pin->sha256)}},&cancelled);
        const double adapter_verify_seconds=seconds(hash_start);
        require(trial->adapter_lease->has_verified_content() && trial->adapter_lease->file("adapter").content_digest==pin->sha256,"held adapter content pin mismatch");
        const auto& af=trial->adapter_lease->file("adapter");
        const auto adapter_identity=af.canonical_target_path.string()+":"+std::to_string(af.bytes)+":"+
            std::to_string(af.mtime_ns)+":"+std::to_string(std::bit_cast<uint32_t>(1.f))+":"+af.content_digest;
        auto complete_fd=trial->base_lease->duplicate_fd("complete");
        NSDictionary* complete=parse_json(read_small(complete_fd.get(),trial->base_lease->file("complete").bytes).c_str());
        NSDictionary* binding=complete[@"binding"];
        require([complete[@"complete"] isEqual:@YES] && integer(complete,@"schema_version")==1 &&
            string_value(complete,@"artifact_type")=="bounded_ffn_input_calibration" &&
            string_value(complete,@"capture_recipe")=="ffn-input-fp16-regions-outliers-v1" && [binding isKindOfClass:NSDictionary.class],"invalid capture completion");
        require(string_value(binding,@"request_id")==capture_id && string_value(binding,@"model_id")=="qwen-image-2.1" &&
            string_value(binding,@"identity_kind")=="canonical-stat-identity" && string_value(binding,@"model_fingerprint")==identity &&
            string_value(binding,@"actual_execution")=="forced-gpu-runtime-boundary" && integer(binding,@"width")==512 &&
            integer(binding,@"height")==512 && integer(binding,@"total_steps")==6 && integer(binding,@"seed")==42 &&
            integer(binding,@"reference_count")==0 && integer(binding,@"reference_size")==0,"capture provenance mismatch");
        NSArray* loras=binding[@"loras"];require([loras isKindOfClass:NSArray.class] && loras.count==1,"capture adapter count mismatch");
        NSDictionary* lora=loras[0];require([lora isKindOfClass:NSDictionary.class] && integer(lora,@"strength")==1 &&
            string_value(lora,@"id")==adapter.filename().string() && string_value(lora,@"fingerprint")==fingerprint(adapter_identity+":0"),"capture adapter identity mismatch");
        require(trial->base_lease->file("input").bytes<9ULL<<20,"input exceeds replay bound");
        auto fd=trial->base_lease->duplicate_fd("input");trial->input_reader=std::make_shared<MlxLeaseFdReader>(MlxOwnedFd(fd.release()),"input");
        auto [arrays,metadata]=mx::load_safetensors(trial->input_reader);
        require(metadata.at("capture_recipe")=="qwen21-real-ffn-bf16-v1" && metadata.at("trajectory")=="forced-gpu" &&
            metadata.at("model_identity")==identity && metadata.at("adapter_identity")==adapter_identity &&
            metadata.at("prompt")==prompt && metadata.at("seed")=="42" && metadata.at("total_steps")=="6" &&
            metadata.at("layer")=="0" && metadata.at("step")=="5" && arrays.size()==1 &&
            arrays.at("input").shape()==mx::Shape{1,int(rows),int(hidden)} && arrays.at("input").dtype()==mx::bfloat16,"full BF16 LoRA capture mismatch");
        const auto file_digest=[&](const char* id) {auto f=trial->base_lease->duplicate_fd(id);return memory_sha256_fd(f.get(),trial->base_lease->file(id).bytes);};
        const auto complete_sha=file_digest("complete"),input_sha=file_digest("input");
        configure_streams();auto event=[](const std::string&,int,int){};
        trial->weights.load_lease(trial->base_lease,{"checkpoint"},event,cancelled);trial->weights.set_runtime_lora_fp16(false);
        require(trial->weights.apply_loras_lease(trial->adapter_lease,{"adapter"},{{adapter.string(),1.f,"transformer"}},"transformer",event,cancelled,true,true)==227,"expected exactly 227 resolved adapter projections");
        const auto snapshot=trial->weights.runtime_lora_snapshot();
        require(snapshot.native_metadata_compatible && !snapshot.fp16_rank_math && snapshot.bindings.size()==195,"unexpected adapter snapshot");
        const auto fused=trial->weights.at(std::string(prefix)+"gate_up.weight");const auto down=trial->weights.at(std::string(prefix)+"out.weight");
        require(fused.shape()==mx::Shape{int(2*width),int(hidden)} && down.shape()==mx::Shape{int(hidden),int(width)} &&
            fused.dtype()==mx::bfloat16 && down.dtype()==mx::bfloat16,"dense layer0 shape mismatch");
        trial->tensors={slice_axis(fused,0,0,width),slice_axis(fused,0,width,2*width),down,arrays.at("input")};
        std::array<std::optional<Weights::ResolvedLoRATerm>,3> selected;
        for(const auto& b:snapshot.bindings)for(const auto& t:b.ordered_terms) {
            int which=-1;if(b.prefix==std::string(prefix)+"gate_up")which=t.output_start==0?0:1;
            else if(b.prefix==std::string(prefix)+"out")which=2;
            if(which<0)continue;
            require(!selected[size_t(which)] && t.source_lease==trial->adapter_lease && t.lease_backed &&
                t.source_logical_id=="adapter" && t.source_role=="transformer" && t.source_adapter_ordinal==0 &&
                t.source_strength==1.f && t.scale==1.f && t.down_dtype==mx::bfloat16 && t.up_dtype==mx::bfloat16 &&
                t.output_start==(which==1?int(width):0) && t.output_end==int(which==2?hidden:(which==1?2*width:width)) &&
                t.down.shape()==mx::Shape{int(rank),int(which==2?width:hidden)} &&
                t.up.shape()==mx::Shape{int(which==2?hidden:width),int(rank)},"unexpected resolved layer0 factor/range");
            selected[size_t(which)]=t;
        }
        for(const auto& t:selected) {require(bool(t),"missing layer0 LoRA term");trial->tensors.push_back(t->down);trial->tensors.push_back(t->up);}
        trial->pending_domain="mlx-source-production";trial->uncertain=true;mx::eval(trial->tensors);mx::synchronize();trial->uncertain=false;trial->pending_domain="none";
        mx::clear_cache(); // Exactly once, before all warmup/recorded arms.
        const auto source_active=mx::get_active_memory();
        require(source_active<=source_bytes+(32ULL<<20),"unexpected full-model/factor materialization");
        trial->base_lease->revalidate_after_drain();trial->adapter_lease->revalidate_after_drain();
        for(const auto& t:trial->tensors)trial->source_owners.push_back(t.data_shared_ptr());
        const auto source_prepare_seconds=seconds(source_start);
        auto start=Time::now();trial->mlx_function=qwen21::runtime_ffn::full(trial->weights,prefix);
        const auto mlx_closure_prepare=seconds(start); // First invocation/evaluation still occurs in warm A.
        const auto baseline=observe_process_memory();
        trial->memory=make_context(baseline,
            std::string(trial->base_lease->digest())+std::string(trial->adapter_lease->artifact_digest()),candidate_id);
        trial->bridge=std::make_unique<mg::MemoryBridge>(*trial->memory,0x514d4c5846464eULL,trial->base_lease->generation(),site_bindings());
        mg::BackendOptions options;options.allocation=trial->bridge->hooks();
        trial->backend=std::make_unique<mg::Backend>(std::move(options));auto& backend=*trial->backend;
        for(const auto& t:trial->tensors)trial->sources.push_back(imported(backend,t));
        for(const auto& s:specs)trial->scratch.push_back(allocate(backend,s));
        for(size_t i=0;i<3;++i)trial->terms[i]={trial->sources[4+i*2],trial->sources[5+i*2],1.f,0,i==2?hidden:width,0};
        auto& v=trial->scratch;
        for(size_t i=0;i<3;++i)require(eligible(i==2?v[2]:trial->sources[3],trial->terms[i],i==2?v[3]:v[i],v[4]),"real r128 term cannot select SGMMA");
        start=Time::now();trial->gate_up=mg::NativeLinear::prepare(backend,{hidden,width});trial->down=mg::NativeLinear::prepare(backend,{width,hidden});
        const auto linear_prepare=seconds(start);
        start=Time::now();trial->corrections=mg::NativeLoRA::prepare(backend,{width,width,rank},mg::native_lora::KernelVariant::SgMmaR128);
        const auto lora_prepare=seconds(start);const std::string lora_recipe(trial->corrections->actual_precision_recipe());
        require(lora_recipe==mg::native_lora::sg_mma_precision_recipe,"comparison requires the frozen SGMMA v2 correction recipe");
        std::vector<std::string> source_hashes;for(const auto& s:trial->sources)source_hashes.push_back(matrix_hash(s));
        const auto verify=[&](bool content) {
            trial->base_lease->revalidate_after_drain();trial->adapter_lease->revalidate_after_drain();
            require(held_fingerprint(*trial->base_lease)==identity,"held model generation changed");
            if(content) {
                require(file_digest("complete")==complete_sha && file_digest("input")==input_sha,"held capture source changed");
                for(size_t i=0;i<trial->sources.size();++i)require(matrix_hash(trial->sources[i])==source_hashes[i],"immutable source tensor changed");
            }
            for(const auto& s:v)padding(s);
        };
        NSMutableArray* pairs=[NSMutableArray array];size_t mlx_calls=0,native_calls=0;bool numeric_pass=true;
        for(size_t round=0;round<arm_order.size() && numeric_pass;++round) {
            NSMutableArray* pair=[NSMutableArray array];
            for(size_t order=0;order<2;++order) {
                const bool mlx=arm_order[round][order]==Arm::Mlx;
                NSMutableDictionary* record=[@{@"arm":mlx?@"mlx":@"native-split-sgmma-v2",@"round":@(round),@"order":@(order),@"warmup":@(round==0)} mutableCopy];
                mg::MatrixView output_view;
                if(mlx) {
                    trial->freshness.require_retired();require(trial->mlx_outputs.empty() && !trial->mlx_view,"old MLX output not retired");
                    start=Time::now();trial->pending_domain="mlx-full-ffn-eval-synchronize";trial->uncertain=true;
                    trial->mlx_outputs=trial->mlx_function({trial->tensors[3]});
                    const auto invocation_seconds=seconds(start);
                    mx::eval(trial->mlx_outputs);mx::synchronize();const auto wall=seconds(start);trial->uncertain=false;trial->pending_domain="none";
                    require(trial->mlx_outputs.size()==1,"MLX full FFN output count differs");const auto& value=trial->mlx_outputs.front();
                    require(value.is_available() && value.dtype()==mx::bfloat16 && value.shape()==mx::Shape{1,int(rows),int(hidden)} &&
                        value.strides()[1]==hidden && value.strides()[2]==1,"MLX full FFN output is not produced compact BF16");
                    trial->freshness.check(value.data_shared_ptr(),trial->source_owners);
                    trial->mlx_view=imported(backend,value);output_view=*trial->mlx_view;++mlx_calls;
                    record[@"wall_seconds"]=@(wall);record[@"invoke_seconds"]=@(invocation_seconds);
                    record[@"eval_synchronize_seconds"]=@(wall-invocation_seconds);
                    record[@"first_compile_execution_in_warmup"]=@(mlx_calls==1);
                    record[@"fresh_output_data_owner"]=@YES;record[@"completion_source"]=@"actual mx::eval + mx::synchronize";
                    record[@"native_ticket_available"]=@NO;record[@"gpu_span_available"]=@NO;
                    record[@"gpu_span_seconds"]=[NSNull null];
                } else {
                    const auto& op=*trial->corrections;start=Time::now();mg::CommandGraph graph;
                    for(const auto stage:native_stages) {
                        const auto before=graph.steps().size();
                        switch(stage) {
                        case Stage::Gate:trial->gate_up->encode(graph,{trial->sources[3],trial->sources[0],v[0],{}});break;
                        case Stage::Up:trial->gate_up->encode(graph,{trial->sources[3],trial->sources[1],v[1],{}});break;
                        case Stage::GateCorrection:op.encode_ordered_corrections(graph,trial->sources[3],std::span(&trial->terms[0],1),v[0],v[0],v[4]);break;
                        case Stage::UpCorrection:op.encode_ordered_corrections(graph,trial->sources[3],std::span(&trial->terms[1],1),v[1],v[1],v[4]);break;
                        case Stage::SiluUp:op.encode_silu_up(graph,v[0],v[1],v[2]);break;
                        case Stage::Down:trial->down->encode(graph,{v[2],trial->sources[2],v[3],{}});break;
                        case Stage::DownCorrection:op.encode_ordered_corrections(graph,v[2],std::span(&trial->terms[2],1),v[3],v[3],v[4]);break;
                        }
                        require(graph.steps().size()-before==dispatches(stage),"native full FFN stage dispatch count mismatch");
                    }
                    require(graph.steps().size()==native_dispatches(),"native full FFN must encode ten dispatches");
                    trial->pending_domain="native-full-ffn-ticket";trial->uncertain=true;trial->pending.emplace(backend.submit(std::move(graph)));
                    const auto done=trial->pending->wait(std::chrono::seconds(30));const auto wall=seconds(start);
                    if(done.physically_complete) {trial->uncertain=false;trial->pending_domain="none";}
                    require(done.ok && done.physically_complete && !done.timed_out && done.submitted_buffers &&
                        done.submitted_buffers==done.completed_buffers,"native FFN physical completion failed");
                    ++native_calls;output_view=v[3];trial->pending.reset();
                    record[@"wall_seconds"]=@(wall);record[@"dispatches"]=@(native_dispatches());record[@"actual_sgmma_low_rank_dispatches"]=@6;
                    record[@"completion_source"]=@"native Ticket physical completion";record[@"native_ticket_available"]=@YES;
                    record[@"physically_complete"]=@YES;record[@"submitted_buffers"]=@(done.submitted_buffers);record[@"completed_buffers"]=@(done.completed_buffers);
                    record[@"gpu_span_available"]=@(done.gpu_seconds>0);record[@"gpu_span_seconds"]=@(done.gpu_seconds);
                }
                start=Time::now();
                if(order==0)for(size_t r=0;r<rows;++r)std::memcpy(static_cast<char*>(v[5].storage.contents())+r*v[5].row_stride_bytes,
                    static_cast<const char*>(output_view.storage.contents())+r*output_view.row_stride_bytes,hidden*2);
                record[@"output_snapshot_seconds"]=@(order==0?seconds(start):0.);record[@"output_snapshot_bytes"]=@(order==0?size_t(rows)*hidden*2:0);
                output_view={}; // Release the temporary import before testing MLX payload retirement.
                if(mlx && order==0)trial->retire_mlx();
                [pair addObject:record];[records addObject:record];
            }
            start=Time::now();const bool ab=arm_order[round][0]==Arm::Mlx;
            const auto error=compare(cpu(ab?v[3]:v[5]),cpu(ab?v[5]:*trial->mlx_view));numeric_pass=error.rrms<=rrms_limit;
            trial->retire_mlx();verify(false);trial->memory->checkpoint("mlx_native_ffn_pair_complete");
            NSDictionary* a=pair[ab?0:1];NSDictionary* b=pair[ab?1:0];const double aw=[a[@"wall_seconds"] doubleValue],bw=[b[@"wall_seconds"] doubleValue];
            require(aw>0 && bw>0,"invalid complete FFN wall timing");
            [pairs addObject:@{@"round":@(round),@"warmup":@(round==0),@"order":ab?@"AB":@"BA",@"mlx_full_ffn_wall_seconds":@(aw),
                @"native_full_ffn_wall_seconds":@(bw),@"native_over_mlx_wall":@(bw/aw),@"native_time_saved_fraction":@(1.-bw/aw),
                @"native_vs_mlx_error":dictionary(error),@"validation_after_both_arms_seconds":@(seconds(start)),@"mlx_output_retired":@YES}];
        }
        require(!numeric_pass || (mlx_calls==3 && native_calls==3),"unexpected complete FFN arm count");verify(true);
        const auto observation=observe_process_memory();const auto ledger=trial->memory->admission().snapshot();
        const auto device=backend.device_name(),manifest=trial->memory->compiled_plan().manifest_digest;
        auto memory=std::move(trial->memory);trial->release_native();trial->bridge->close();trial->bridge.reset();
        const auto released=memory->admission().snapshot();require(!released.storage_count && !released.reservation_count &&
            !released.pending_release_count && !released.unknown_bytes,"native phase ledger did not return to zero");
        memory->begin_draining();memory->finish_success();const auto report=memory->take_report();
        require(!report.metrics.tainted && !report.metrics.worker_quarantined,"memory phase failed or quarantined");
        NSMutableArray* hashes=[NSMutableArray array];for(const auto& h:source_hashes)[hashes addObject:ns(h)];
        write_exclusive(output,@{@"schema":@"native-qwen-lora-mlx-ffn-replay-v1",@"status":numeric_pass?@"screening-pass":@"screening-rejected",
            @"scope":@"one historical layer0/step5 complete FFN; no DiT, E2E, ANE or general quality qualification",
            @"library_build_identity":ns(build),@"tool_source_sha256":@(TC_MLX_FFN_REPLAY_SOURCE_SHA),@"device":ns(device),
            @"capture_id":@(capture_id),@"layer":@0,@"step":@5,@"rows":@(rows),@"hidden":@(hidden),@"width":@(width),@"rank":@(rank),@"strength":@1,
            @"mlx_full_ffn_calls":@(mlx_calls),@"native_full_ffn_calls":@(native_calls),@"additional_oracle_calls":@0,@"rrms_limit":@(rrms_limit),
            @"native_recipe":ns(lora_recipe),@"base_recipe":ns(mg::NativeLinear::precision_recipe()),@"native_dispatches":@(native_dispatches()),
            @"native_schedule":@"split gate-base/up-base, ordered gate/up corrections, SiLU/up, down-base, ordered down correction",
            @"mlx_recipe":@"actual runtime_ffn::full / same held Weights::project / FP32 rank branch / ordered BF16 projection adds",
            @"model_stat_fingerprint":ns(identity),@"capture_complete_sha256":ns(complete_sha),@"input_file_sha256":ns(input_sha),
            @"adapter_sha256":ns(af.content_digest),@"adapter_artifact_digest":ns(trial->adapter_lease->artifact_digest()),
            @"adapter_verification_bytes_read":@(trial->adapter_lease->verification_bytes_read()),@"base_source_proof":@"same held fd/canonical-stat; no whole-checkpoint hash",
            @"adapter_source_proof":@"verified-held-fd",@"produced_source_pre_post_sha256":hashes,
            @"produced_source_digest_order":@[@"gate",@"up",@"down",@"input",@"gate.A",@"gate.B",@"up.A",@"up.B",@"down.A",@"down.B"],
            @"prepare_seconds":@{@"source_including_adapter_verification":@(source_prepare_seconds),@"adapter_verification_included":@(adapter_verify_seconds),
                @"mlx_compile_closure_creation_only":@(mlx_closure_prepare),@"native_linear_pso":@(linear_prepare),@"native_sgmma_pso":@(lora_prepare)},
            @"memory":@{@"source_payload_bytes":@(source_bytes),@"mlx_active_after_source_production":@(source_active),@"native_scratch_backings":@6,
                @"native_owned_bytes_before_release":@(ledger.storage_bytes),@"same_phase_peak_committed_bytes":@(ledger.peak_committed_bytes),
                @"process_baseline_bytes":@(baseline.process_footprint_bytes),@"process_observed_after_bytes":@(observation.process_footprint_bytes),
                @"framework_allowance_bytes":@(512ULL<<20),@"framework_guard_certified":@NO,@"whole_request_certified":@NO,
                @"mlx_output_native_claims":@0,@"max_live_mlx_outputs":@1,@"mlx_output_bytes":@(size_t(rows)*hidden*2),@"manifest_digest":ns(manifest),
                @"ledger_storage_count_after_release":@(released.storage_count),@"ledger_storage_bytes_after_release":@(released.storage_bytes),
                @"ledger_reservation_count_after_release":@(released.reservation_count),@"ledger_pending_release_count_after_release":@(released.pending_release_count),
                @"ledger_unknown_bytes_after_release":@(released.unknown_bytes)},
            @"source_immutable":@YES,@"actual_mlx_data_owner_verified":@YES,@"padding_intact":@YES,@"all_arms_completed_at_recorded_boundary":@YES,
            @"same_native_ledger_released":@YES,@"all_mlx_outputs_fresh_and_retired":@YES,@"distinct_mlx_output_data_owners":@(trial->freshness.retired.size()),
            @"timing_scope":@"MLX graph invocation+eval+synchronize, including its runtime output/intermediate allocation; native graph encode+submit+Ticket physical completion using preallocated scratch. Explicit source/native allocation, cold PSO, hash, snapshot and CPU comparison are outside arm wall",
            @"memory_scope":@"Produced immutable MLX sources retained in baseline. Fresh MLX outputs, intermediates and allocator cache belong to explicit framework allowance, not native allocation claims",
            @"pair_scope":@"Only first output 8-MiB snapshot between arms; validation after both. MLX first invocation/evaluation is warmup, not a separate oracle call",
            @"gpu_span_scope":@"Native Ticket only; absent for MLX. Never fabricate a MLX Ticket or compare/add unequal GPU span domains",
            @"cache_scope":@"One pre-warm clear_cache; no cache clear between arms. OS/driver/compiler caches shared and not reset",
            @"sample_limit":@"warm AB, record AB, record BA: three MLX and three native calls maximum; no additional oracle",@"pairs":pairs,@"records":records});
        std::cout<<output<<'\n';require(numeric_pass,"RRMS screening rejected; receipt preserved");
    } catch(const std::exception& e) {
        const bool complete=trial && !trial->uncertain;
        const std::string incomplete_domain=trial?trial->pending_domain:"none";
        if(trial && complete) {
            trial->release_native();if(trial->bridge)trial->bridge->close();
            if(trial->memory) {trial->memory->finalize_failure(e.what());trial->memory->complete_failure_cleanup(true);}
        }
        // Quarantine before receipt I/O: a disk/encoding error must not unwind
        // an uncertain owner whose imported sources may still be consumed.
        if(!complete)(void)trial.release();
        if(!std::filesystem::exists(output))write_exclusive(output,@{@"schema":@"native-qwen-lora-mlx-ffn-replay-v1",@"status":@"failed",
            @"error":ns(e.what()),@"library_build_identity":ns(build),@"all_started_work_completed_at_recorded_boundary":@(complete),
            @"incomplete_domain":ns(incomplete_domain),
            @"owner_quarantined":@(!complete),@"process_exit_is_not_physical_proof":@YES,@"records":records});
        throw;
    }
}
void cpu_contract() {
    static_assert(source_bytes==308ULL<<20 && native_dispatches()==10);
    size_t mlx_count=0,native_count=0,rejected=0;
    for(const auto& pair:arm_order)for(auto arm:pair)(arm==Arm::Mlx?mlx_count:native_count)++;
    require(mlx_count==3 && native_count==3 && native_stages[0]==Stage::Gate && native_stages[1]==Stage::Up &&
        native_stages[2]==Stage::GateCorrection && native_stages[3]==Stage::UpCorrection,"bounded arm/stage schedule changed");
    const auto rejects=[&](const auto& f){try {f();}catch(const std::exception&){++rejected;}};
    std::array<uint16_t,4> a{0x3f80,0xbf80,0,0x8000},b=a;
    const auto view=[](const auto& x){return CpuView{x.data(),2,2,4,x.size()*2};};
    require(compare(view(a),view(a)).rrms==0,"exact finite comparison failed");b[3]=0;
    require(compare(view(a),view(b)).differences==1 && compare(view(a),view(b)).rrms==0,"signed zero diagnostic changed");
    b[0]=0x4000;require(compare(view(b),view(a)).rrms>rrms_limit,"numeric rejection failed");
    rejects([&]{auto bad=view(a);bad.extent=1;(void)compare(bad,view(a));});
    b[0]=0x7fc0;rejects([&]{(void)compare(view(b),view(a));});
    FreshOutputs fresh;std::shared_ptr<const void> source=std::make_shared<int>(1),owner=std::make_shared<int>(2);
    fresh.check(owner,{source});rejects([&]{fresh.check(owner,{source});});rejects([&]{fresh.require_retired();});
    rejects([&]{FreshOutputs other;other.check(source,{source});});owner.reset();fresh.require_retired();
    owner=std::make_shared<int>(3);fresh.check(owner,{source});owner.reset();fresh.require_retired();
    using namespace mg::native_lora;
    const Layout x{rows,hidden,hidden*2,size_t(rows)*hidden*2,0,mg::ScalarType::BF16,1};
    const Layout out{rows,width,width*2,size_t(rows)*width*2,0,mg::ScalarType::BF16,2};
    const Layout low{rows,rank,rank*4,size_t(rows)*rank*4,0,mg::ScalarType::FP32,3};
    const TermLayout term{{rank,hidden,hidden*2,size_t(rank)*hidden*2,0,mg::ScalarType::BF16,4},
        {width,rank,rank*2,size_t(width)*rank*2,0,mg::ScalarType::BF16,5},1.f,0,width};
    const auto checked=validate_bindings({width,width,rank},x,std::span(&term,1),out,out,low);
    require(sg_mma_eligible(checked.terms.front()),"real rank128 gate projection must select SGMMA");
    rejects([&]{auto bad=low;bad.storage=x.storage;(void)validate_bindings({width,width,rank},x,std::span(&term,1),out,out,bad);});
    const ProcessMemoryObservation baseline{true,400ULL<<20,16ULL<<30,12ULL<<30,"cpu-fixture"};
    auto c=make_context(baseline,"cpu-only",candidate_id,true);mg::MemoryBridge bridge(*c,55,1,site_bindings());auto hooks=bridge.hooks();
    std::array<int,6> handles{};std::vector<std::shared_ptr<void>> claims;
    for(size_t i=0;i<specs.size();++i) {auto claim=hooks.reserve(upper(specs[i]),specs[i].label);
        hooks.allocated(claim,&handles[i],backing(specs[i]));claims.push_back(std::move(claim));}
    require(c->admission().snapshot().storage_count==6,"six native backings must have six claims");
    rejects([&]{(void)hooks.reserve(1,specs[0].label);});claims.clear();bridge.close();
    require(c->admission().snapshot().storage_count==0,"CPU ledger cleanup failed");c->begin_draining();c->finish_success();
    require(rejected==7,"CPU negative checks did not all reject");
    std::cout<<"lora_mlx_ffn_replay_cpu=PASS mlx_calls=3 native_calls=3 additional_oracle_calls=0 native_dispatches=10 fresh_data_owner=PASS rejected=7 device_execution=0 model_loads=0\n";
}

}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            if(argc==1 || (argc==2 && std::string_view(argv[1])=="--cpu-contract")) {cpu_contract();return 0;}
            require(argc==6 && std::string_view(argv[1])=="--device","usage: replay [--cpu-contract | --device checkpoint adapter capture-directory new-receipt.json]");
            run(argv[2],argv[3],argv[4],argv[5]);return 0;
        } catch(const std::exception& e) {std::cerr<<"native MLX/native LoRA FFN replay failed: "<<e.what()<<'\n';return 1;}
    }
}
