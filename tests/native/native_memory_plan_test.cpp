// CPU metadata/ledger checks. No Backend construction, shader compilation,
// device access, checkpoint import or allocation of the reported tensor sizes.
#include "../../native/models/qwen21/native_memory_plan.hpp"

#include <algorithm>
#include <iostream>

using namespace tc;
using namespace tc::qwen21::native_graph;
namespace {
constexpr uint64_t GiB=uint64_t(1)<<30;
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
template<class Function> void rejects(Function&& f) {
    bool rejected=false; try { f(); } catch (const std::exception&) { rejected=true; }
    check(rejected,"invalid native phase accepted");
}
ProcessMemoryObservation baseline() { return {true,8*GiB,64*GiB,48*GiB,"synthetic-produced-source-baseline"}; }
NativePhaseShape production() { return {{},1024,34,{}}; }
NativePhaseShape hybrid_production() { auto result=production(); result.hybrid_ffn=true; return result; }
NativePhaseShape tiny() { return {{{128,1,128,256},2,8192},6,3,{2,4}}; }
const NativePhaseAllocation& labeled(const NativePhaseMemoryPlan& plan,std::string_view label) {
    auto found=std::find_if(plan.allocations.begin(),plan.allocations.end(),
        [&](const auto& a) { return a.label==label; });
    check(found!=plan.allocations.end(),"native allocation label omitted"); return *found;
}
const NativePhaseAllocation& allocation(const NativePhaseMemoryPlan& plan,std::string suffix) {
    return labeled(plan,"qwen-native-"+suffix);
}

void formulas() {
    const auto shape=production();
    const auto plan=plan_native_phase(shape,"retained-source-v1",baseline());
    check(plan.allocations.size()==22 && plan.manifest.sites.size()==22 &&
          plan.bindings.size()==22 && plan.backing_count==89 && plan.manifest.instances.size()==89,
          "native phase must describe exactly 22 labels / 89 backings for 32 layers");
    const uint64_t full=1058,h=4096;
    check(allocation(plan,"hidden").payload_bytes==full*h*2 && allocation(plan,"hidden").instances==2,
          "hidden payload must use full rows and two banks");
    check(allocation(plan,"latents").payload_bytes==1024*64*2 && allocation(plan,"latents").instances==2,
          "latents must use target rows only");
    check(allocation(plan,"prefix-modulation").payload_bytes==full*h*2 &&
          allocation(plan,"prefix-modulation").instances==4,"modulation expanded banks");
    check(allocation(plan,"time-constants").payload_bytes==1024 &&
          allocation(plan,"time-constants").capacity_upper_bytes==16384,"time constant page capacity");
    check(allocation(plan,"time-first").payload_bytes==2*h*2,"time projection has two rows");
    check(allocation(plan,"modulation").payload_bytes==2*4*h*2,"four modulation channels");
    check(allocation(plan,"text-context").payload_bytes==34*h*2,"text payload excludes references");
    check(allocation(plan,"rope-cosine").payload_bytes==full*64*4,"RoPE must remain FP32");
    check(allocation(plan,"prefix-key").payload_bytes==34*h*2 &&
          allocation(plan,"prefix-key").instances==32,"per-layer K/V capacity");
    check(allocation(plan,"invalid").payload_bytes==4 &&
          allocation(plan,"invalid").capacity_upper_bytes==16384,"finite flag needs a physical page upper");
    uint64_t payload=0,upper=0;
    for (const auto& a:plan.allocations) {
        check(a.capacity_upper_bytes%16384==0 && a.capacity_upper_bytes>=a.payload_bytes &&
              a.capacity_upper_bytes-a.payload_bytes<16384,"capacity is not exact page upper");
        payload+=a.payload_bytes*a.instances; upper+=a.capacity_upper_bytes*a.instances;
    }
    check(payload==plan.payload_bytes && upper==plan.allocation_upper_bytes && upper>payload,
          "unique backing aggregates incorrect");
    check(plan.compiled.planned_peak_bytes==baseline().process_footprint_bytes+upper+GiB,
          "baseline/native/framework peak should be counted exactly once");
    check(plan.policy.effective_budget_bytes==60*GiB && !plan.policy.release_stable &&
          plan.policy.certification_state==MemoryCertificationState::ExperimentalGuarded &&
          plan.policy.evidence_digest.empty(),"phase guard must not claim production certification");
    for (const auto& instance:plan.manifest.instances)
        check(instance.live_begin==0 && instance.live_end==1 && !instance.alias_group,
              "run-owned storage cannot expire at an earlier dispatch");
    for (const auto& refs:{std::vector<uint32_t>{},std::vector<uint32_t>{1024},std::vector<uint32_t>{1024,512}}) {
        auto editing=shape; editing.reference_rows=refs;
        auto edited=plan_native_phase(editing,"retained-source-v1",baseline());
        uint32_t prefix=shape.text_rows; for (auto rows:refs) prefix+=rows;
        const auto arena=std::max(
            NativeBlock::workspace_bytes(shape.geometry.block,shape.target_rows+prefix,prefix,BlockPhase::PrefixBuild),
            NativeBlock::workspace_bytes(shape.geometry.block,shape.target_rows,prefix,BlockPhase::SteadyDecode));
        check(allocation(edited,"shared-block-arena").payload_bytes==arena,
              "arena must be max of actual prefill/decode workspace functions");
        check(allocation(edited,"prefix-value").payload_bytes==uint64_t(prefix)*h*2,
              "reference rows must extend every prefix bank");
    }
    const auto small=plan_native_phase(tiny(),"tiny",baseline());
    check(small.backing_count==29,"small layer count must change cache instance count");
}

void hybrid_formulas_and_guards() {
    const auto shape=hybrid_production();
    const auto gpu=plan_native_phase(production(),"source",baseline());
    const auto plan=plan_native_phase(shape,"source",baseline());
    check(plan.allocations.size()==38 && plan.bindings.size()==38 && plan.manifest.sites.size()==38 &&
          plan.backing_count==105 && plan.manifest.instances.size()==105,
          "hybrid must add exactly 16 labels/backings, not layers*steps status pages");
    struct Expected { const char* label; uint64_t payload; MemoryClass kind; };
    constexpr uint64_t H=4096,C=6144,M=1024;
    const Expected expected[]={
        {"qwen-native-hybrid-gpu-partial",M*H*4,MemoryClass::Activation},
        {"qwen-native-hybrid-ane-partial",M*H*4,MemoryClass::Activation},
        {"tc-native-hybrid-wg",C*H,MemoryClass::RefillSlot},
        {"tc-native-hybrid-sg",C*64,MemoryClass::RefillSlot},
        {"tc-native-hybrid-wu",C*H,MemoryClass::RefillSlot},
        {"tc-native-hybrid-su",C*64,MemoryClass::RefillSlot},
        {"tc-native-hybrid-wd",H*C,MemoryClass::RefillSlot},
        {"tc-native-hybrid-sd",H*64,MemoryClass::RefillSlot},
        {"tc-native-hybrid-x",H*M,MemoryClass::Activation},
        {"tc-native-hybrid-tx",M*2,MemoryClass::Activation},
        {"tc-native-hybrid-y",(H+1)*M*2,MemoryClass::Activation},
        {"tc-native-ane-stage-signs",512*4,MemoryClass::Conditioning},
        {"tc-native-ane-stage-zero",4,MemoryClass::Conditioning},
        {"tc-native-hybrid-finite-status",4,MemoryClass::Activation},
        {"tc-native-ane-stage-producer-status",4,MemoryClass::Activation},
        {"tc-native-ane-stage-restore-status",4,MemoryClass::Activation},
    };
    uint64_t payload=0,upper=0;
    for (const auto& e:expected) {
        const auto& a=labeled(plan,e.label);
        const auto capacity=(e.payload+16383)&~uint64_t(16383);
        check(a.payload_bytes==e.payload && a.capacity_upper_bytes==capacity &&
              a.instances==1 && a.memory_class==e.kind,"hybrid allocation formula/layout differs from controller");
        payload+=e.payload;upper+=capacity;
    }
    check(plan.payload_bytes==gpu.payload_bytes+payload &&
          plan.allocation_upper_bytes==gpu.allocation_upper_bytes+upper,
          "hybrid must add one bank, shared partials and bounded live statuses only");
    // Every old GPU allocation and its site identity must remain unchanged.
    for (const auto& before:gpu.allocations) {
        const auto& after=labeled(plan,before.label);
        check(before.site_id==after.site_id && before.payload_bytes==after.payload_bytes &&
              before.capacity_upper_bytes==after.capacity_upper_bytes && before.instances==after.instances &&
              before.memory_class==after.memory_class,"hybrid changed an existing GPU site");
    }
    check(plan.compiled.planned_peak_bytes==baseline().process_footprint_bytes+plan.allocation_upper_bytes+GiB &&
          plan.policy.certification_state==MemoryCertificationState::ExperimentalGuarded &&
          plan.policy.capability_level==MemoryCapabilityLevel::AllocationGuarded &&
          !plan.policy.release_stable && plan.policy.evidence_digest.empty(),
          "program/compiler allowance must remain a guard, not full-request certification");
    check(plan.compiled.plan_digest!=gpu.compiled.plan_digest && plan.manifest.candidate_id!=gpu.manifest.candidate_id &&
          plan.policy.candidate_dtype=="bf16+w8a8","hybrid and GPU plans need distinct identities");
    NativePhaseLimits exact;exact.request_budget_bytes=plan.compiled.planned_peak_bytes;
    check(plan_native_phase(shape,"source",baseline(),exact).compiled.fits_budget,"hybrid exact budget must fit");
    --exact.request_budget_bytes;
    rejects([&] {plan_native_phase(shape,"source",baseline(),exact);});
    rejects([&] {auto s=shape;s.target_rows=1023;plan_native_phase(s,"source",baseline());});
    rejects([&] {auto s=shape;s.reference_rows={1024};plan_native_phase(s,"source",baseline());});
    rejects([&] {auto s=shape;s.geometry.layers=31;plan_native_phase(s,"source",baseline());});
    rejects([&] {auto s=shape;s.geometry.block.ffn_width=10240;plan_native_phase(s,"source",baseline());});
    rejects([&] {auto l=NativePhaseLimits{};l.framework_upper_bytes=0;plan_native_phase(shape,"source",baseline(),l);});
}

void compact_scale_cache_plan_and_lifetime() {
    auto shape=hybrid_production();
    const auto plain=plan_native_phase(shape,"source",baseline());
    shape.weight_scale_cache_entries=32;
    const auto cached=plan_native_phase(shape,"source",baseline());
    const auto& entry=labeled(cached,"tc-native-hybrid-weight-scale-cache");
    // Independent hand-calculated byte oracle: 6144 gate + 6144 up + 4096
    // down FP16 scalars. No 64-byte IOSurface pitch in compact storage.
    check(entry.payload_bytes==32768 && entry.capacity_upper_bytes==49152 && entry.instances==32 &&
          entry.memory_class==MemoryClass::Conditioning,"compact row-scale entry sizing/headroom changed");
    check(cached.payload_bytes-plain.payload_bytes==1048576 &&
          cached.allocation_upper_bytes-plain.allocation_upper_bytes==1572864 &&
          cached.backing_count==137 && cached.manifest.instances.size()==137 &&
          cached.allocations.size()==39 && cached.bindings.size()==39,
          "cache must add only one compact backing per layer, with explicit capacity headroom");
    for (const auto& a:plain.allocations) {
        const auto& after=labeled(cached,a.label);
        check(after.site_id==a.site_id && after.payload_bytes==a.payload_bytes &&
              after.capacity_upper_bytes==a.capacity_upper_bytes && after.instances==a.instances,
              "enabling scale cache changed an existing native allocation");
    }
    check(cached.compiled.plan_digest!=plain.compiled.plan_digest &&
          cached.manifest.candidate_id!=plain.manifest.candidate_id &&
          cached.policy.estimate_provenance.find("explicit_16k_per_entry_capacity_headroom")!=std::string::npos &&
          !cached.policy.release_stable && cached.policy.evidence_digest.empty(),
          "cache plan must carry explicit identity/guard, not borrowed qualification");
    NativePhaseLimits limit; limit.request_budget_bytes=cached.compiled.planned_peak_bytes;
    check(plan_native_phase(shape,"source",baseline(),limit).compiled.fits_budget,"exact cached budget rejected");
    --limit.request_budget_bytes;
    rejects([&] { plan_native_phase(shape,"source",baseline(),limit); });
    for (uint32_t count:{1U,31U,33U,UINT32_MAX})
        rejects([&] { auto bad=shape;bad.weight_scale_cache_entries=count;
            plan_native_phase(bad,"source",baseline()); });
    rejects([&] { auto bad=shape;bad.hybrid_ffn=false;plan_native_phase(bad,"source",baseline()); });
    NativePhaseShape small{{{128,1,128,1024},2,64},32,7,{},true,2};
    const auto small_plan=plan_native_phase(small,"source",baseline());
    const auto& small_entry=labeled(small_plan,"tc-native-hybrid-weight-scale-cache");
    check(small_entry.payload_bytes==2304 && small_entry.capacity_upper_bytes==32768 &&
          small_entry.instances==2,"tiny cache sizing must preserve compact payload vs capacity distinction");

    // Use real phase hooks with synthetic buffer identities, no GPU allocation.
    const auto b=baseline();
    NativePhaseMemory phase(small,"source",b,{},75,1,nullptr,[b] { return b; });
    auto hooks=phase.hooks();
    std::array<int,2> handles{};
    std::vector<std::shared_ptr<void>> owners;
    for (auto& handle:handles) {
        auto claim=hooks.reserve(small_entry.capacity_upper_bytes,small_entry.label);
        hooks.allocated(claim,&handle,small_entry.payload_bytes);
        owners.push_back(std::move(claim));
    }
    check(phase.context().admission().snapshot().storage_count==2 &&
          phase.context().admission().snapshot().storage_bytes==4608,
          "only actual compact capacities, not planner headroom, may be charged");
    rejects([&] { hooks.reserve(small_entry.capacity_upper_bytes,small_entry.label); });
    auto retained=owners.front(); owners.clear();
    check(phase.context().admission().snapshot().storage_count==1,
          "late cache owner must preserve its allocation claim");
    rejects([&] { phase.finish_after_release(true); });
    retained.reset(); phase.finish_after_release(true);
    check(phase.context().state()==MemoryExecutionState::Succeeded &&
          phase.context().admission().snapshot().storage_count==0,
          "cache phase must release every retained compact backing before successful finish");
}

void runtime_lora_plan_and_lifetime() {
    auto shape=production();
    const auto base=plan_native_phase(shape,"base",baseline());
    shape.lora_max_rank=128;
    const auto global=plan_native_phase(shape,"base+adapter",baseline());
    const auto& low=allocation(global,"lora-low");
    check(low.payload_bytes==uint64_t(1058)*128*4 && low.instances==1 &&
          global.backing_count==base.backing_count+1 && global.allocations.size()==23,
          "global/attention LoRA must add one shared FP32 low only");
    check(std::none_of(global.allocations.begin(),global.allocations.end(),
          [](const auto& a) {return a.label=="qwen-native-lora-gate" || a.label=="qwen-native-lora-up";}),
          "non-FFN adapters must not allocate gate/up scratch");
    shape.lora_ffn=true;
    const auto ffn=plan_native_phase(shape,"base+adapter",baseline());
    check(allocation(ffn,"lora-gate").payload_bytes==uint64_t(1058)*12288*2 &&
          allocation(ffn,"lora-up").payload_bytes==uint64_t(1058)*12288*2 &&
          ffn.backing_count==92 && ffn.allocations.size()==25,
          "FFN LoRA must add gate/up once per request, not once per layer");
    for (const auto& before:base.allocations) {
        const auto& after=labeled(ffn,before.label);
        check(before.site_id==after.site_id && before.payload_bytes==after.payload_bytes &&
              before.capacity_upper_bytes==after.capacity_upper_bytes && before.instances==after.instances,
              "LoRA changed a base allocation/arena");
    }
    check(ffn.compiled.plan_digest!=base.compiled.plan_digest &&
          ffn.manifest.candidate_id!=base.manifest.candidate_id &&
          ffn.policy.candidate_dtype=="bf16+runtime_lora_fp32_rank" &&
          !ffn.policy.release_stable && ffn.policy.evidence_digest.empty(),
          "LoRA phase identity must not borrow Base certification");
    NativePhaseLimits limit;limit.request_budget_bytes=ffn.compiled.planned_peak_bytes;
    check(plan_native_phase(shape,"base+adapter",baseline(),limit).compiled.fits_budget,
          "LoRA exact budget rejected");
    --limit.request_budget_bytes;
    rejects([&] {plan_native_phase(shape,"base+adapter",baseline(),limit);});
    rejects([&] {auto s=shape;s.lora_max_rank=0;plan_native_phase(s,"source",baseline());});
    rejects([&] {auto s=shape;s.lora_max_rank=129;plan_native_phase(s,"source",baseline());});
    rejects([&] {auto s=shape;s.hybrid_ffn=true;plan_native_phase(s,"source",baseline());});
    auto small=tiny();small.lora_max_rank=16;small.lora_ffn=true;
    const auto b=baseline();
    NativePhaseMemory phase(small,"base+adapter",b,{},76,1,nullptr,[b] {return b;});
    auto hooks=phase.hooks();std::array<int,3> handles{};
    std::vector<std::shared_ptr<void>> claims;
    size_t index=0;
    for (const auto* suffix:{"lora-low","lora-gate","lora-up"}) {
        const auto& a=allocation(phase.plan(),suffix);
        auto token=hooks.reserve(a.capacity_upper_bytes,a.label);
        hooks.allocated(token,&handles[index++],a.payload_bytes);
        claims.push_back(std::move(token));
    }
    check(phase.context().admission().snapshot().storage_count==3,
          "LoRA scratch must use the same phase ledger");
    auto late=claims.front();claims.clear();
    rejects([&] {phase.finish_after_release(true);});
    check(phase.context().admission().snapshot().storage_count==1,
          "retained rank scratch owner must remain charged");
    late.reset();phase.finish_after_release(true);
    check(phase.context().state()==MemoryExecutionState::Succeeded &&
          phase.context().admission().snapshot().storage_count==0,
          "LoRA physical owner release must leave a zero ledger");
}

void rejected_plans() {
    auto shape=production(); const auto observation=baseline();
    rejects([&] { auto s=shape;s.text_rows=0;plan_native_phase(s,"source",observation); });
    rejects([&] { auto s=shape;s.target_rows=0;plan_native_phase(s,"source",observation); });
    rejects([&] { auto s=shape;s.reference_rows={1,1,1};plan_native_phase(s,"source",observation); });
    rejects([&] { auto s=shape;s.reference_rows={0};plan_native_phase(s,"source",observation); });
    rejects([&] { auto s=shape;s.reference_rows={UINT32_MAX};plan_native_phase(s,"source",observation); });
    rejects([&] { auto s=shape;s.geometry.layers=0;plan_native_phase(s,"source",observation); });
    rejects([&] { auto s=shape;s.geometry.maximum_rows=1057;plan_native_phase(s,"source",observation); });
    rejects([&] { auto b=observation;b.available=false;plan_native_phase(shape,"source",b); });
    rejects([&] { plan_native_phase(shape,"",observation); });
    rejects([&] { auto b=observation;b.physical_memory_bytes=4*GiB;plan_native_phase(shape,"source",b); });
    rejects([&] { auto b=observation;b.system_available_bytes=4*GiB;plan_native_phase(shape,"source",b); });
    rejects([&] { NativePhaseLimits l;l.framework_upper_bytes=0;plan_native_phase(shape,"source",observation,l); });
    auto plan=plan_native_phase(shape,"source",observation);
    NativePhaseLimits exact; exact.request_budget_bytes=plan.compiled.planned_peak_bytes;
    check(plan_native_phase(shape,"source",observation,exact).compiled.fits_budget,"exact budget must fit");
    --exact.request_budget_bytes;
    rejects([&] { plan_native_phase(shape,"source",observation,exact); });
    NativePhaseLimits huge; huge.framework_upper_bytes=UINT64_MAX;
    rejects([&] { plan_native_phase(shape,"source",observation,huge); });
}

void actual_context_and_finish() {
    const auto b=baseline();
    NativePhaseMemory phase(tiny(),"source",b,{},71,1,nullptr,[b] { return b; });
    auto hooks=phase.hooks();
    rejects([&] { NativePhaseMemory nested(tiny(),"source",b,{},71,2,&phase.context(),[b] { return b; }); });
    std::vector<std::shared_ptr<void>> tokens;
    std::vector<uint64_t> handles(phase.plan().backing_count);
    size_t next=0;
    for (const auto& a:phase.plan().allocations) for (uint32_t i=0;i<a.instances;++i) {
        auto token=hooks.reserve(a.capacity_upper_bytes,a.label);
        check(bool(token),"compiled native site did not admit its planned instance");
        hooks.allocated(token,&handles[next++],a.capacity_upper_bytes);
        tokens.push_back(std::move(token));
    }
    const auto before=phase.context().admission().snapshot();
    check(before.storage_count==phase.plan().backing_count &&
          before.storage_bytes==phase.plan().allocation_upper_bytes,"phase hooks did not charge full native domain");
    rejects([&] { phase.finish_after_release(false); });
    rejects([&] { phase.finish_after_release(true); }); // Cannot hide still-retained buffers at epoch 1.
    check(phase.context().current_epoch()==0,"failed drain advanced past live native allocations");
    tokens.clear();
    phase.finish_after_release(true);
    const auto after=phase.context().admission().snapshot();
    check(after.storage_count==0 && after.reservation_count==0 && after.site_allocation_count==0 &&
          phase.context().state()==MemoryExecutionState::Succeeded,"phase finish requires existing ledger zero");
    rejects([&] { hooks.reserve(16384,"qwen-native-invalid"); });
    const auto report=phase.context().take_report();
    check(report.metrics.ledger_storage_count==0 && report.metrics.execution_state=="succeeded",
          "phase terminal report did not reflect drain");
}
void failed_drain() {
    const auto b=baseline();
    NativePhaseMemory phase(tiny(),"source",b,{},72,1,nullptr,[b] { return b; });
    auto hooks=phase.hooks(); int native=0;
    auto token=hooks.reserve(16384,"qwen-native-invalid"); hooks.allocated(token,&native,16384);
    check(phase.fail_after_drain(false,"synthetic unproven completion")==MemoryFailureDisposition::QuarantineWorker,
          "unproven physical completion must quarantine live storage");
    check(phase.context().admission().snapshot().storage_count==1,
          "logical failure released an in-flight budget claim");
    rejects([&] { hooks.reserve(16384,"qwen-native-invalid"); });
    token.reset();
    check(phase.context().state()==MemoryExecutionState::Quarantined,
          "late owner release revived a failed phase");
}

void hybrid_live_status_and_import_ownership() {
    const auto b=baseline();
    const auto plan=plan_native_phase(hybrid_production(),"source",b);
    MemoryExecutionContext context(plan.policy,plan.compiled,b,[b] { return b; });
    metal_graph::MemoryBridge bridge(context,73,1,plan.bindings);
    context.begin_running();
    auto hooks=bridge.hooks();
    auto surface_backing=std::make_shared<int>(0); // Synthetic identity, no device allocation.
    const auto& surface=labeled(plan,"tc-native-hybrid-wg");
    auto surface_claim=hooks.reserve(surface.capacity_upper_bytes,surface.label);
    hooks.allocated(surface_claim,surface_backing.get(),surface.capacity_upper_bytes);
    // Surface owns the one committed claim. Multiple imported views retain that
    // same owner; no second reserve/commit is performed by an import.
    auto first_import=bridge.retain_import(surface_claim,surface_backing.get(),surface.capacity_upper_bytes,surface_backing);
    auto second_import=bridge.retain_import(surface_claim,surface_backing.get(),surface.capacity_upper_bytes,surface_backing);
    surface_claim.reset();
    surface_backing.reset();
    check(context.admission().snapshot().storage_count==1 &&
          context.admission().snapshot().storage_bytes==surface.capacity_upper_bytes,
          "duplicate imported owners must retain exactly one backing charge");
    const auto& producer=labeled(plan,"tc-native-ane-stage-producer-status");
    const auto& restore=labeled(plan,"tc-native-ane-stage-restore-status");
    int producer_backing=0,restore_backing=0;
    // Simulate sequential W1 jobs without allocating tensors or launching Metal.
    // Reusing an address after release is legal; overlapping claims are not.
    for (unsigned generation=0;generation<64;++generation) {
        auto p=hooks.reserve(producer.capacity_upper_bytes,producer.label);
        hooks.allocated(p,&producer_backing,producer.capacity_upper_bytes);
        auto r=hooks.reserve(restore.capacity_upper_bytes,restore.label);
        hooks.allocated(r,&restore_backing,restore.capacity_upper_bytes);
        auto late_consumer_owner=p;
        p.reset();r.reset();
        rejects([&] {hooks.reserve(producer.capacity_upper_bytes,producer.label);});
        check(context.admission().snapshot().storage_count==2,
              "late producer/consumer owner must remain charged with surface");
        late_consumer_owner.reset();
        check(context.admission().snapshot().storage_count==1,
              "physically released status must free its single live site for next generation");
    }
    first_import.reset();
    check(context.admission().snapshot().storage_count==1,"one remaining view must retain surface claim");
    second_import.reset();
    bridge.close();
    context.checkpoint("hybrid_owners_physically_released");
    context.enter_epoch(1,"hybrid_phase_drain_and_release");
    context.begin_draining();
    context.finish_success();
    const auto report=context.take_report();
    check(report.metrics.execution_state=="succeeded" && report.metrics.ledger_storage_count==0 &&
          report.metrics.ledger_reserved_bytes==0 && report.metrics.ledger_pending_release_bytes==0,
          "hybrid phase must finish with no residual storage/reservation/pending claim");
}

void hybrid_full_allocation_and_failed_cleanup() {
    const auto b=baseline();
    NativePhaseMemory phase(hybrid_production(),"source",b,{},74,1,nullptr,[b] { return b; });
    auto hooks=phase.hooks();
    std::vector<std::shared_ptr<void>> owners;
    std::vector<uint64_t> handles(phase.plan().backing_count);
    size_t next=0;
    for (const auto& a:phase.plan().allocations) for (uint32_t i=0;i<a.instances;++i) {
        auto token=hooks.reserve(a.capacity_upper_bytes,a.label);
        hooks.allocated(token,&handles[next++],a.capacity_upper_bytes);
        owners.push_back(std::move(token));
    }
    check(phase.context().admission().snapshot().storage_bytes==phase.plan().allocation_upper_bytes &&
          phase.context().admission().snapshot().storage_count==105,"hybrid complete manifest/hooks coverage");
    check(phase.fail_after_drain(false,"hybrid consumer completion unknown")==MemoryFailureDisposition::QuarantineWorker,
          "unknown hybrid consumer must quarantine the existing phase");
    check(phase.context().admission().snapshot().storage_count==105,"retirement must retain every live backing");
    owners.clear();
    check(phase.context().admission().snapshot().storage_count==0 &&
          phase.context().state()==MemoryExecutionState::Quarantined,"late release must not revive retired hybrid state");
    rejects([&] {hooks.reserve(16384,"tc-native-ane-stage-producer-status");});
}
} // namespace
int main() {
    formulas(); hybrid_formulas_and_guards(); compact_scale_cache_plan_and_lifetime(); runtime_lora_plan_and_lifetime(); rejected_plans(); actual_context_and_finish(); failed_drain();
    hybrid_live_status_and_import_ownership(); hybrid_full_allocation_and_failed_cleanup();
    std::cout << "native_phase_memory=PASS sites=22 production_backings=89 actual_context_hooks=checked "
                 "hybrid_sites=38 hybrid_backings=105 cache_sites=39 cache_backings=137 compact_cache=checked "
                 "bounded_status_reuse=checked duplicate_owners=checked "
                 "runtime_lora_sites=25 runtime_lora_backings=92 empty_adapter_zero_scratch=checked "
                 "budget_and_zero_ledger_finish=checked device_created=0\n";
}
