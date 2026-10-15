// CPU-only admission/schema tests of the actual model contract. No Backend,
// MLX device, ANE client, checkpoint file or generated model is instantiated.
#include "../../native/models/qwen21/native_denoiser_contract.hpp"

#include <iostream>

namespace ng=tc::qwen21::native_graph;
namespace contract=ng::detail;
namespace {
size_t accepted=0,rejected=0;
void check(bool ok,const char* message) {
    if (!ok) throw std::runtime_error(message);
    ++accepted;
}
template<class F> void rejects(F&& function) {
    try { function(); } catch (const std::invalid_argument&) { ++rejected; return; }
    throw std::runtime_error("unsupported native LoRA recipe was accepted");
}
void recipes() {
    using V=ng::DenoiserScheduleVariant;
    const ng::DenoiserGeometry production{};
    const ng::DenoiserGeometry tiny{{128,1,128,256},2,64};
    ng::NativeDenoiserExecutionOptions o;
    check(contract::validate_lora_recipe(production,o,false)==V::Base20,"base production did not resolve to 20 steps");
    check(contract::validate_lora_recipe(tiny,o,false)==V::GeneratedTiny,"existing generated base fixture changed");
    rejects([&] { contract::validate_lora_recipe(production,o,true); });
    o.adapter_revision="verified-adapter-order-strength-and-source-v1";
    rejects([&] { contract::validate_lora_recipe(production,o,false); });
    rejects([&] { contract::validate_lora_recipe(production,o,true); });
    o.schedule_variant=V::Base20;
    check(contract::validate_lora_recipe(production,o,true)==V::Base20,"ordinary LoRA lost its 20-step recipe");
    rejects([&] { contract::validate_lora_recipe(tiny,o,true); });
    o.schedule_variant=V::ViggleLoRA6;
    check(contract::validate_lora_recipe(production,o,true)==V::ViggleLoRA6,"explicit distilled LoRA rejected");
    { auto matrix=o; matrix.lora_kernel=tc::metal_graph::native_lora::KernelVariant::SgMmaR128;
      check(contract::validate_lora_recipe(production,matrix,true)==V::ViggleLoRA6,
            "explicit matrix LoRA lost its exact six-step variant");
      matrix.adapter_revision.clear(); matrix.schedule_variant=V::Base20;
      rejects([&] { contract::validate_lora_recipe(production,matrix,false); }); }
    { auto invalid=o;
      invalid.lora_kernel=static_cast<tc::metal_graph::native_lora::KernelVariant>(255);
      rejects([&] { contract::validate_lora_recipe(production,invalid,true); }); }
    rejects([&] { contract::validate_lora_recipe(tiny,o,true); });
    { auto bad=o; bad.adapter_revision.clear();
      rejects([&] { contract::validate_lora_recipe(production,bad,false); }); }
    o.hybrid_ffn=true;
    rejects([&] { contract::validate_lora_recipe(production,o,true); });
    o.hybrid_ffn=false; o.schedule_variant=V::GeneratedTiny;
    check(contract::validate_lora_recipe(tiny,o,true)==V::GeneratedTiny,"explicit generated adapter fixture rejected");
    rejects([&] { contract::validate_lora_recipe(production,o,true); });
    o.schedule_variant=static_cast<V>(255);
    rejects([&] { contract::validate_lora_recipe(tiny,o,true); });
    for (const auto steps:{size_t(1),size_t(6),size_t(19),size_t(21),size_t(40)})
        rejects([&] { contract::validate_schedule_variant(V::Base20,steps); });
    for (const auto steps:{size_t(1),size_t(5),size_t(7),size_t(20)})
        rejects([&] { contract::validate_schedule_variant(V::ViggleLoRA6,steps); });
    contract::validate_schedule_variant(V::Base20,20); ++accepted;
    contract::validate_schedule_variant(V::ViggleLoRA6,6); ++accepted;
    for (const auto steps:{size_t(1),size_t(2),size_t(256)}) {
        contract::validate_schedule_variant(V::GeneratedTiny,steps); ++accepted;
    }
    for (const auto variant:{V::Base20,V::ViggleLoRA6,V::GeneratedTiny,V::AutomaticBase,static_cast<V>(255)}) {
        rejects([&] { contract::validate_schedule_variant(variant,0); });
        rejects([&] { contract::validate_schedule_variant(variant,257); });
    }
    rejects([&] { contract::validate_schedule_variant(V::AutomaticBase,20); });
    contract::validate_adapter_identity("",""); ++accepted;
    contract::validate_adapter_identity("immutable-order-a-then-b","immutable-order-a-then-b"); ++accepted;
    rejects([&] { contract::validate_adapter_identity("immutable-order-a-then-b","immutable-order-b-then-a"); });
    rejects([&] { contract::validate_adapter_identity("adapter",""); });
    rejects([&] { contract::validate_adapter_identity("","adapter"); });
}
} // namespace
int main() {
    try {
        recipes();
        std::cout<<"native_qwen_lora_denoiser_cpu_contract=PASS accepted="<<accepted
                 <<" rejected="<<rejected<<" devices=0 models=0\n";
        return 0;
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
