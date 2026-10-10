// CPU-only: invokes allocation callbacks with opaque handle identities, never
// constructs Backend, an MTLDevice, an MLX array or an IOSurface.
#include "../../native/backends/metal_graph/memory_bridge.hpp"
#include "../../native/runtime/memory_execution.hpp"

#include <iostream>
#include <thread>

using namespace tc;
using namespace tc::metal_graph;
namespace {
void check(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function) {
    bool rejected=false;
    try { function(); } catch (const std::exception&) { rejected=true; }
    check(rejected,"invalid native memory bridge operation accepted");
}
std::unique_ptr<MemoryExecutionContext> context(bool running=true) {
    MemoryManifest manifest;
    manifest.candidate_id="cpu-native-bridge";
    manifest.checkpoint_digest=std::string(64,'a');
    manifest.backend_revision="cpu-native-bridge-v1";
    manifest.runtime_revision=memory_runtime_revision();
    manifest.sites={
        {"native.scratch","synthetic","run",MemoryClass::Activation,
         AllocationLifetime::Request,UpperProvenance::ExactShapeFormula},
        {"native.weights","synthetic","run",MemoryClass::Weights,
         AllocationLifetime::Request,UpperProvenance::ExactShapeFormula},
    };
    manifest.instances={
        {"native.scratch",0,64,0,2,0,"drain"},
        {"native.scratch",1,64,0,2,0,"drain"},
        {"native.weights",0,96,0,2,0,"drain"},
    };
    auto compiled=compile_memory_plan(manifest,10,5,1024);
    EffectiveMemoryPolicy policy;
    policy.enabled=true; policy.effective_budget_bytes=1024;
    policy.planned_upper_bytes=compiled.planned_peak_bytes;
    policy.framework_upper_bytes=5;
    const ProcessMemoryObservation baseline{true,10,4096,4096,"cpu-fixture"};
    auto result=std::make_unique<MemoryExecutionContext>(policy,std::move(compiled),baseline,
        [baseline] { return baseline; });
    if (running) result->begin_running();
    return result;
}
std::vector<AllocationSiteBinding> bindings() {
    return {{"scratch","native.scratch",MemoryClass::Activation},
            {"weights","native.weights",MemoryClass::Weights}};
}
void finish(MemoryExecutionContext& value) {
    value.begin_draining(); value.finish_success();
}

void admission_contract() {
    auto execution=context(false);
    MemoryBridge bridge(*execution,41,7,bindings());
    auto hooks=bridge.hooks();
    rejects([&] { hooks.reserve(32,"scratch"); });
    execution->begin_running();
    rejects([&] { hooks.reserve(0,"scratch"); });
    rejects([&] { hooks.reserve(32,"unknown-label"); });
    rejects([&] { hooks.reserve(65,"scratch"); });
    check(execution->admission().snapshot().reservation_count==0,"invalid admission reserved bytes");
    auto first=hooks.reserve(64,"scratch"), second=hooks.reserve(32,"scratch");
    check(first && second,"valid native reservations missing");
    rejects([&] { hooks.reserve(1,"scratch"); }); // No generic fallback on exhausted site.
    check(execution->admission().snapshot().reserved_bytes==96,"reservation upper not charged");
    first.reset(); second.reset();
    check(execution->admission().snapshot().reservation_count==0,"abandoned allocation did not cancel");
    int native_identity=0;
    auto token=hooks.reserve(64,"scratch");
    rejects([&] { hooks.allocated(token,nullptr,32); });
    rejects([&] { hooks.allocated(token,&native_identity,0); });
    rejects([&] { hooks.allocated(token,&native_identity,65); });
    rejects([&] { bridge.allocation_identity(token); });
    check(execution->admission().snapshot().reserved_bytes==64,"invalid commit consumed reservation");
    hooks.allocated(token,&native_identity,48);
    const auto id=bridge.allocation_identity(token);
    check(id.allocator_domain==41 && id.session_generation==7 &&
          id.handle==uint64_t(reinterpret_cast<uintptr_t>(&native_identity)) && id.capacity==48,
          "committed StorageId lost physical identity");
    const auto snapshot=execution->admission().snapshot();
    check(snapshot.reserved_bytes==0 && snapshot.storage_bytes==48 && snapshot.storage_count==1,
          "actual capacity not committed to existing ledger");
    rejects([&] { hooks.allocated(token,&native_identity,48); });
    token.reset();
    check(execution->admission().snapshot().storage_count==0,"final token did not release claim");
    bridge.close(); finish(*execution);
}

void provenance_and_import_contract() {
    auto execution=context();
    MemoryBridge bridge(*execution,51,9,bindings());
    MemoryBridge other(*execution,51,10,bindings());
    auto hooks=bridge.hooks();
    int native=0, foreign=0;
    auto claim=hooks.reserve(64,"scratch");
    auto owner=std::make_shared<int>(1);
    rejects([&] { bridge.retain_import(claim,&native,32,owner); });
    rejects([&] { hooks.allocated(owner,&native,32); }); // No unsafe opaque cast.
    hooks.allocated(claim,&native,32);
    rejects([&] { other.allocation_identity(claim); });
    rejects([&] { other.hooks().allocated(claim,&native,32); });
    rejects([&] { bridge.retain_import(claim,&foreign,32,owner); });
    rejects([&] { bridge.retain_import(claim,&native,31,owner); });
    rejects([&] { bridge.retain_import(claim,&native,32,{}); });
    rejects([&] { bridge.retain_import(claim,&native,32,claim); });
    rejects([&] { bridge.retain_import({},&native,32,owner); });
    auto impostor=std::shared_ptr<void>(claim.get(),[](void*) {});
    rejects([&] { bridge.allocation_identity(impostor); });
    auto alias=std::shared_ptr<void>(claim,&foreign);
    rejects([&] { bridge.allocation_identity(alias); });
    alias.reset(); impostor.reset();
    bool backing_released=false;
    owner=std::shared_ptr<int>(new int(2),[&](int* p) {
        backing_released=true;
        check(execution->admission().snapshot().storage_bytes==32,
              "import claim released before allocator backing");
        delete p;
    });
    auto imported=bridge.retain_import(claim,&native,32,owner);
    auto second_view=bridge.retain_import(claim,&native,32,owner);
    check(execution->admission().snapshot().storage_count==1 &&
          execution->admission().snapshot().storage_bytes==32,"import view duplicated physical charge");
    claim.reset(); owner.reset(); imported.reset();
    check(!backing_released && execution->admission().snapshot().storage_count==1,
          "escaped import lost backing or accounting owner");
    second_view.reset();
    check(backing_released && execution->admission().snapshot().storage_count==0,
          "final imported owner did not release original claim");
    bridge.close(); other.close(); finish(*execution);
}

void thread_and_closed_binding_contract() {
    auto execution=context();
    AllocationHooks saved;
    std::shared_ptr<void> reserved, committed;
    int native=0;
    // A copy of the ledger observes the SAME shared ledger state after Context
    // destruction; no second budget/ledger is constructed by the adapter.
    auto ledger=execution->admission().ledger();
    {
        MemoryBridge bridge(*execution,61,11,bindings());
        saved=bridge.hooks();
        reserved=saved.reserve(32,"scratch");
        committed=saved.reserve(64,"scratch");
        bool reserve_rejected=false,commit_rejected=false;
        std::thread wrong_thread([&] {
            try { saved.reserve(16,"weights"); } catch (...) { reserve_rejected=true; }
            try { saved.allocated(committed,&native,48); } catch (...) { commit_rejected=true; }
        });
        wrong_thread.join();
        check(reserve_rejected && commit_rejected,"off-owner callback touched Context");
        saved.allocated(committed,&native,48);
        bridge.close();
        rejects([&] { saved.reserve(16,"weights"); });
        rejects([&] { saved.allocated(reserved,&native,16); });
        rejects([&] { bridge.allocation_identity(committed); });
    }
    check(ledger.snapshot().storage_bytes==48 && ledger.snapshot().reserved_bytes==32,
          "bridge close prematurely dropped token ownership");
    execution.reset();
    rejects([&] { saved.reserve(16,"weights"); }); // Never dereferences dead Context.
    std::thread completion([a=std::move(reserved),b=std::move(committed)]() mutable {
        a.reset(); b.reset(); // Ledger-only cancellation/release after Context died.
    });
    completion.join();
    check(ledger.snapshot().storage_count==0 && ledger.snapshot().reservation_count==0,
          "late callback ownership failed after Context destruction");
}

void configuration_and_budget_contract() {
    auto execution=context();
    rejects([&] { MemoryBridge b(*execution,0,1,bindings()); });
    rejects([&] { MemoryBridge b(*execution,1,0,bindings()); });
    rejects([&] { MemoryBridge b(*execution,1,1,{}); });
    rejects([&] { MemoryBridge b(*execution,1,1,{{"scratch","missing",MemoryClass::Activation}}); });
    rejects([&] { MemoryBridge b(*execution,1,1,{{"scratch","native.scratch",MemoryClass::Weights}}); });
    rejects([&] { MemoryBridge b(*execution,1,1,{{"scratch","native.scratch",MemoryClass::UnknownExternal}}); });
    rejects([&] { auto sites=bindings(); sites.push_back(sites.front()); MemoryBridge b(*execution,1,1,sites); });
    MemoryBridge bridge(*execution,71,1,bindings());
    auto hooks=bridge.hooks();
    const auto budget=execution->admission().snapshot().budget_bytes;
    execution->admission().ledger().set_process_baseline(budget-8);
    check(!hooks.reserve(16,"scratch"),"budget denial was bypassed");
    check(execution->admission().snapshot().reservation_count==0,"denied request retained reservation");
    execution->admission().ledger().set_process_baseline(10);
    bridge.close(); finish(*execution);
}
} // namespace

int main() {
    admission_contract(); provenance_and_import_contract();
    thread_and_closed_binding_contract(); configuration_and_budget_contract();
    std::cout << "native_memory_bridge=PASS reserve_site actual_capacity provenance "
                 "import_aliases late_release closed_context device_created=0\n";
}
