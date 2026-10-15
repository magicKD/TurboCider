#include "native_memory_plan.hpp"
#include "native_denoiser_contract.hpp"
#include "../../backends/metal_graph/native_ane_stage_contract.hpp"

#include <algorithm>
#include <limits>

namespace tc::qwen21::native_graph {
namespace {
void check(bool value,const char* message) { if (!value) throw std::invalid_argument(message); }
uint64_t add(uint64_t a,uint64_t b) {
    check(a<=UINT64_MAX-b,"native phase memory byte overflow"); return a+b;
}
uint64_t multiply(uint64_t a,uint64_t b) {
    check(!b || a<=UINT64_MAX/b,"native phase memory byte overflow"); return a*b;
}
uint64_t matrix(uint64_t rows,uint64_t columns,uint64_t item=2) {
    check(rows && columns,"native phase empty matrix"); return multiply(multiply(rows,columns),item);
}
uint64_t capacity(uint64_t payload) {
    constexpr uint64_t alignment=16384;
    return add(payload,alignment-1)&~(alignment-1);
}
uint64_t surface_payload(uint64_t rows,uint64_t columns,uint64_t item) {
    // Surface's actual producer ABI has a 64-byte physical row pitch, including
    // narrow scale vectors. The common 16-KiB backing upper is applied below;
    // Surface fails closed if the driver exceeds the admitted capacity.
    const auto pitch=add(multiply(columns,item),63)&~uint64_t(63);
    return multiply(rows,pitch);
}
} // namespace

NativePhaseMemoryPlan plan_native_phase(const NativePhaseShape& shape,std::string source,
    const ProcessMemoryObservation& baseline,NativePhaseLimits limits) {
    // Reuse actual workspace geometry/byte formula, not a drifting estimator.
    validate_geometry(shape.geometry.block);
    const auto& g=shape.geometry;
    check(g.layers && g.layers<=32 && g.maximum_rows && g.maximum_rows<=65536,
          "native phase invalid layer/row bound");
    check(shape.target_rows && shape.text_rows && shape.reference_rows.size()<=2,
          "native phase requires target/text and at most two references");
    uint64_t prefix=shape.text_rows;
    for (auto rows:shape.reference_rows) {
        check(rows>0,"native phase reference rows must be positive"); prefix=add(prefix,rows);
    }
    const auto full=add(prefix,shape.target_rows);
    check(full<=g.maximum_rows,"native phase combined rows exceed native recipe capacity");
    if (shape.hybrid_ffn) {
        check(shape.reference_rows.empty() &&
              ((detail::production_hybrid_geometry(g) && shape.target_rows==1024) ||
               (detail::tiny_hybrid_geometry(g) && shape.target_rows==32)),
              "native hybrid memory plan requires the same base/tiny denoiser geometry without references");
        metal_graph::native_ane_stage_contract::validate_geometry(
            {shape.target_rows,g.block.hidden,g.block.ffn_width,g.block.ffn_width/2});
    }
    check(!shape.weight_scale_cache_entries ||
          (shape.hybrid_ffn && shape.weight_scale_cache_entries==g.layers),
          "native compact scale cache requires one entry per hybrid layer");
    check(shape.lora_max_rank<=128 && (!shape.lora_ffn || shape.lora_max_rank),
          "native LoRA phase requires an actual bounded rank for FFN scratch");
    check(!shape.hybrid_ffn || !shape.lora_max_rank,
          "native hybrid LoRA needs a separately implemented program and memory ABI");
    check(!source.empty() && baseline.available && baseline.process_footprint_bytes &&
          !baseline.source.empty() && baseline.physical_memory_bytes,
          "native phase requires produced-source identity and observed process baseline");
    check(limits.framework_upper_bytes>0 && limits.system_reserve_bytes>0 &&
          baseline.physical_memory_bytes>limits.system_reserve_bytes,
          "native phase needs explicit framework/system guards and available physical memory");
    const auto physical_budget=baseline.physical_memory_bytes-limits.system_reserve_bytes;
    const auto budget=limits.request_budget_bytes ?
        std::min(physical_budget,limits.request_budget_bytes):physical_budget;
    NativePhaseMemoryPlan result;
    auto& manifest=result.manifest;
    manifest.candidate_id=shape.hybrid_ffn ? "qwen21-native-hybrid-dit-phase-v1" : "qwen21-native-dit-phase-v1";
    // Digest a caller-validated source lease revision; this is NOT a newly
    // verified checkpoint hash or a production capability/evidence record.
    manifest.checkpoint_digest=memory_sha256_hex("native-phase-source-lease-v1:"+source);
    manifest.backend_revision=shape.hybrid_ffn ? "native-qwen-hybrid-dit-phase-v1" : "native-qwen-dit-phase-v1";
    if (shape.weight_scale_cache_entries) {
        manifest.candidate_id="qwen21-native-hybrid-scale-cache-dit-phase-v1";
        manifest.backend_revision="native-qwen-hybrid-scale-cache-dit-phase-v1";
    }
    if (shape.lora_max_rank) {
        manifest.candidate_id="qwen21-native-runtime-lora-dit-phase-v1";
        manifest.backend_revision="native-qwen-runtime-lora-dit-phase-v1";
    }
    manifest.runtime_revision=memory_runtime_revision();
    auto site_allocation=[&](std::string label,std::string site,MemoryClass kind,
                            uint64_t payload,uint32_t instances=1,uint64_t allocator_headroom=0) {
        const uint64_t upper=capacity(add(payload,allocator_headroom));
        result.allocations.push_back({label,site,kind,payload,upper,instances});
        result.bindings.push_back({label,site,kind});
        result.payload_bytes=add(result.payload_bytes,multiply(payload,instances));
        result.allocation_upper_bytes=add(result.allocation_upper_bytes,multiply(upper,instances));
        result.backing_count+=instances;
        manifest.sites.push_back({site,"qwen21.native-dit","native-phase",kind,
            AllocationLifetime::Request,UpperProvenance::ExactShapeFormula});
        for (uint32_t i=0;i<instances;++i)
            manifest.instances.push_back({site,i,upper,0,1,0,"native-phase.physical-drain-and-owner-release"});
    };
    auto allocation=[&](std::string suffix,MemoryClass kind,uint64_t payload,uint32_t instances=1) {
        site_allocation("qwen-native-"+suffix,"qwen.native."+suffix,kind,payload,instances);
    };
    constexpr auto activation=MemoryClass::Activation,conditioning=MemoryClass::Conditioning;
    const auto h=g.block.hidden;
    allocation("hidden",activation,matrix(full,h),2);
    allocation("latents",activation,matrix(shape.target_rows,64),2);
    allocation("prefix-modulation",activation,matrix(full,h),4);
    allocation("time-constants",activation,matrix(2,256));
    for (const auto* name:{"time-first","time-silu","time-embedding","embedding-silu"})
        allocation(name,activation,matrix(2,h));
    allocation("modulation",activation,matrix(2,uint64_t(4)*h));
    allocation("output-scale",activation,matrix(1,h));
    allocation("output-normalized",activation,matrix(shape.target_rows,h));
    allocation("noise",activation,matrix(shape.target_rows,64));
    for (const auto* name:{"text-normalized","text-first","text-gelu","text-context"})
        allocation(name,activation,matrix(shape.text_rows,h));
    allocation("rope-cosine",conditioning,matrix(full,64,4));
    allocation("rope-sine",conditioning,matrix(full,64,4));
    const auto arena=std::max(
        NativeBlock::workspace_bytes(g.block,uint32_t(full),uint32_t(prefix),BlockPhase::PrefixBuild),
        NativeBlock::workspace_bytes(g.block,shape.target_rows,uint32_t(prefix),BlockPhase::SteadyDecode));
    allocation("shared-block-arena",activation,arena);
    allocation("prefix-key",conditioning,matrix(prefix,h),g.layers);
    allocation("prefix-value",conditioning,matrix(prefix,h),g.layers);
    allocation("invalid",activation,sizeof(uint32_t));
    if (shape.lora_max_rank) {
        allocation("lora-low",activation,matrix(std::max(full,uint64_t(2)),shape.lora_max_rank,4));
        if (shape.lora_ffn) {
            allocation("lora-gate",activation,matrix(full,g.block.ffn_width));
            allocation("lora-up",activation,matrix(full,g.block.ffn_width));
        }
    }
    if (shape.hybrid_ffn) {
        const auto channels=g.block.ffn_width/2;
        allocation("hybrid-gpu-partial",activation,matrix(shape.target_rows,h,4));
        allocation("hybrid-ane-partial",activation,matrix(shape.target_rows,h,4));
        auto surface=[&](std::string name,MemoryClass kind,uint64_t rows,uint64_t columns,uint64_t item) {
            site_allocation("tc-native-hybrid-"+name,"qwen.native.hybrid.surface."+name,
                kind,surface_payload(rows,columns,item));
        };
        // One exact W1 bank. Scale vectors also occupy 64 bytes per row; using
        // rows*2 would substantially undercount sg/su/sd physical storage.
        surface("wg",MemoryClass::RefillSlot,channels,h,1);
        surface("sg",MemoryClass::RefillSlot,channels,1,2);
        surface("wu",MemoryClass::RefillSlot,channels,h,1);
        surface("su",MemoryClass::RefillSlot,channels,1,2);
        surface("wd",MemoryClass::RefillSlot,h,channels,1);
        surface("sd",MemoryClass::RefillSlot,h,1,2);
        surface("x",activation,h,shape.target_rows,1);
        surface("tx",activation,1,shape.target_rows,2);
        surface("y",activation,uint64_t(h)+1,shape.target_rows,2);
        site_allocation("tc-native-ane-stage-signs","qwen.native.hybrid.signs",
            conditioning,512*sizeof(float));
        site_allocation("tc-native-ane-stage-zero","qwen.native.hybrid.zero",
            conditioning,sizeof(uint32_t));
        site_allocation("tc-native-hybrid-finite-status","qwen.native.hybrid.finite-status",
            activation,sizeof(uint32_t));
        for (const auto* name:{"producer-status","restore-status"})
            site_allocation(std::string("tc-native-ane-stage-")+name,
                std::string("qwen.native.hybrid.")+name,activation,sizeof(uint32_t));
        if (shape.weight_scale_cache_entries) {
            // Dense FP16 row scales, not padded IOSurface rows. Only payload is
            // allocated; one extra page is explicit conservative capacity
            // headroom for driver allocation accounting, not a measured bound.
            const auto payload=multiply(add(multiply(channels,2),h),2);
            site_allocation("tc-native-hybrid-weight-scale-cache",
                "qwen.native.hybrid.weight-scale-cache",conditioning,payload,
                shape.weight_scale_cache_entries,16384);
        }
        // Each status site permits one live allocation, not layers*steps owned
        // pages. The controller releases a physically drained generation before
        // admitting the next; uncertain/late consumers keep the claim charged.
    }
    const auto increment=add(result.allocation_upper_bytes,limits.framework_upper_bytes);
    check(baseline.process_footprint_bytes<=budget && increment<=budget-baseline.process_footprint_bytes,
          "native phase baseline, native capacity and framework guard exceed process budget");
    check(limits.system_reserve_bytes<=baseline.system_available_bytes &&
          increment<=baseline.system_available_bytes-limits.system_reserve_bytes,
          "native phase available system memory cannot cover allocations plus guards");
    result.compiled=compile_memory_plan(manifest,baseline.process_footprint_bytes,
        limits.framework_upper_bytes,budget,true);
    check(result.compiled.complete && result.compiled.fits_budget,
          "native phase allocation plan is not executable within budget");
    auto& policy=result.policy;
    policy.enabled=true; policy.user_limit_bytes=limits.request_budget_bytes;
    policy.effective_budget_bytes=budget; policy.system_reserve_bytes=limits.system_reserve_bytes;
    policy.framework_upper_bytes=limits.framework_upper_bytes;
    policy.planned_upper_bytes=result.compiled.planned_peak_bytes;
    policy.denoiser_budget_bytes=result.allocation_upper_bytes;
    policy.estimate_fits=true; policy.route_available=true; policy.execution_supported=true;
    policy.capability_level=MemoryCapabilityLevel::AllocationGuarded;
    policy.certification_state=MemoryCertificationState::ExperimentalGuarded;
    policy.release_stable=false; policy.manifest_digest=result.compiled.manifest_digest;
    policy.adapter_candidate=manifest.candidate_id; policy.candidate_backend="native_metal_phase";
    policy.candidate_dtype=shape.hybrid_ffn ? "bf16+w8a8" : "bf16";
    if (shape.lora_max_rank) policy.candidate_dtype="bf16+runtime_lora_fp32_rank";
    policy.estimate_provenance=shape.hybrid_ffn ?
        "native_exact_shapes+ane_surface_64b_pitch+16k_capacity_upper+uncertified_compiler_driver_guard" :
        "native_exact_shapes+16k_capacity_upper+uncertified_framework_guard";
    if (shape.weight_scale_cache_entries)
        policy.estimate_provenance+="+compact_fp16_scales+explicit_16k_per_entry_capacity_headroom";
    if (shape.lora_max_rank)
        policy.estimate_provenance+="+shared_fp32_lora_low+optional_bf16_gate_up";
    policy.admission_state="native_phase_guarded";
    policy.enforcement_scope="native_dit_begin_to_physical_drain_only;produced_sources_in_baseline";
    policy.reason=shape.hybrid_ffn ?
        "Native and single-bank ANE surfaces guarded; compiler/driver allowance uncertified; source baseline retained; full request not certified" :
        "Native allocations guarded; source baseline retained; encoder/VAE/full request not certified";
    policy.digest=result.compiled.plan_digest;
    return result;
}

NativePhaseMemory::NativePhaseMemory(const NativePhaseShape& shape,std::string source,
    const ProcessMemoryObservation& baseline,NativePhaseLimits limits,uint64_t domain,uint64_t generation,
    MemoryExecutionContext* parent,ProcessMemoryObserver observer) {
    check(!parent,"native phase cannot nest in an unqualified full-request memory context");
    plan_=plan_native_phase(shape,std::move(source),baseline,limits);
    context_=std::make_unique<MemoryExecutionContext>(plan_.policy,plan_.compiled,baseline,std::move(observer));
    bridge_=std::make_unique<metal_graph::MemoryBridge>(*context_,domain,generation,plan_.bindings);
    context_->begin_running();
}
NativePhaseMemory::~NativePhaseMemory() noexcept { close(); }
metal_graph::AllocationHooks NativePhaseMemory::hooks() const { return bridge_->hooks(); }
void NativePhaseMemory::close() noexcept { if (bridge_) bridge_->close(); }
void NativePhaseMemory::finish_after_release(bool physically_drained) {
    check(physically_drained,"native phase finish requires proven physical drain");
    close();
    context_->checkpoint("native_phase_owners_released");
    context_->enter_epoch(1,"native_phase_physical_drain_and_owner_release");
    context_->begin_draining();
    context_->finish_success();
}
MemoryFailureDisposition NativePhaseMemory::fail_after_drain(bool physically_drained,std::string reason) noexcept {
    close();
    const auto disposition=context_->finalize_failure(std::move(reason));
    if (disposition==MemoryFailureDisposition::NeedsGpuDrain)
        return context_->complete_failure_cleanup(physically_drained,
            physically_drained ? std::string{}:"native phase physical completion is unproven");
    return disposition;
}

} // namespace tc::qwen21::native_graph
