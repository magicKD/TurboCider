// CPU-only tests of the same recipe/deadline/completion reducers used by the
// native denoiser. No Backend, private ANE client, model or GPU is instantiated.
#include "../../native/models/qwen21/native_denoiser_contract.hpp"

#include <iostream>

namespace ng=tc::qwen21::native_graph;
namespace contract=ng::detail;
namespace {
void check(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function) {
    bool rejected=false;
    try { function(); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"invalid native hybrid denoiser contract accepted");
}
void options() {
    const ng::DenoiserGeometry production{};
    const ng::DenoiserGeometry tiny{{128,1,128,1024},2,64};
    const ng::DenoiserGeometry old_gpu{{128,1,128,256},2,64};
    ng::NativeDenoiserExecutionOptions opt;
    contract::validate_execution_options(old_gpu,opt,false);
    contract::validate_execution_options(production,opt,false);
    rejects([&] { auto bad=opt; bad.register_weight_staging=true;
        contract::validate_execution_options(production,bad,true); });
    rejects([&] { auto bad=opt; bad.cache_weight_scales=true;
        contract::validate_execution_options(production,bad,true); });
    opt.hybrid_ffn=true;
    opt.ane_program_cache="/private/tmp/not-created-native-hybrid-contract";
    contract::validate_execution_options(production,opt,true);
    contract::validate_execution_options(tiny,opt,true);
    { auto variant=opt; variant.register_weight_staging=true;
      contract::validate_execution_options(production,variant,true); }
    for (bool specialized:{false,true}) {
        auto cached=opt; cached.cache_weight_scales=true; cached.register_weight_staging=specialized;
        contract::validate_execution_options(production,cached,true);
        contract::validate_execution_options(tiny,cached,true);
        rejects([&] { contract::validate_execution_options(production,cached,false); });
    }
    rejects([&] { contract::validate_execution_options(production,opt,false); });
    rejects([&] { contract::validate_execution_options(old_gpu,opt,true); });
    rejects([&] { auto bad=tiny; bad.layers=1; contract::validate_execution_options(bad,opt,true); });
    rejects([&] { auto bad=tiny; bad.block.heads=2; contract::validate_execution_options(bad,opt,true); });
    rejects([&] { auto bad=opt; bad.ane_program_cache.clear(); contract::validate_execution_options(tiny,bad,true); });
    rejects([&] { auto bad=opt; bad.ane_program_cache="relative"; contract::validate_execution_options(tiny,bad,true); });
    rejects([&] {
        auto bad=opt;
        bad.allocation.reserve=[](size_t,std::string_view) { return std::shared_ptr<void>{}; };
        contract::validate_execution_options(tiny,bad,true);
    });
    contract::validate_hybrid_request(production,1024,0,20);
    contract::validate_hybrid_request(tiny,32,0,2);
    rejects([&] { contract::validate_hybrid_request(production,1024,1,20); });
    rejects([&] { contract::validate_hybrid_request(production,1024,0,19); });
    rejects([&] { contract::validate_hybrid_request(production,1088,0,20); });
    rejects([&] { contract::validate_hybrid_request(tiny,32,0,3); });
    rejects([&] { contract::validate_hybrid_request(tiny,64,0,2); });
    rejects([&] { contract::validate_hybrid_request(old_gpu,32,0,2); });
}
void deadlines() {
    using namespace std::chrono;
    const auto now=steady_clock::time_point(seconds(42));
    check(contract::step_deadline(milliseconds(0),now)==now,"zero drain deadline must observe");
    check(contract::step_deadline(milliseconds(30),now)==now+milliseconds(30),"deadline moved");
    check(contract::step_remaining(now+microseconds(1),now)==milliseconds(1),"positive fraction truncated to zero");
    check(contract::step_remaining(now+microseconds(1001),now)==milliseconds(2),"remaining wait must ceil");
    check(contract::step_remaining(now,now)==milliseconds(0),"expired deadline extended");
    check(contract::step_deadline(milliseconds::max(),now)==steady_clock::time_point::max(),"deadline overflow");
    rejects([&] { contract::step_deadline(milliseconds(-1),now); });
}
void completion_domains() {
    const tc::metal_graph::Completion gpu{true,true,false,3,3,0.012,{}};
    auto result=contract::join_physical_domain(gpu,false,false,false,false,"inactive");
    check(result.ok && result.physically_complete && result.gpu_seconds==gpu.gpu_seconds && result.error.empty(),
          "unsubmitted hybrid changed GPU-only completion");
    result=contract::join_physical_domain(gpu,true,true,true,false,{});
    check(result.ok && result.physically_complete && result.gpu_seconds==0 && result.submitted_buffers==3,
          "hybrid completion invented an overlapping GPU time or command count");
    result=contract::join_physical_domain(gpu,true,false,false,true,"ANE still pending");
    check(!result.ok && !result.physically_complete && result.timed_out && result.error=="ANE still pending",
          "completed GPU hid pending/timed-out ANE");
    result=contract::join_physical_domain(gpu,true,false,true,false,"retired arithmetic");
    check(!result.ok && result.physically_complete,"late drain revived retired arithmetic");
    auto failed=gpu; failed.ok=false; failed.physically_complete=false; failed.error="GPU pending";
    result=contract::join_physical_domain(failed,true,true,true,false,{});
    check(!result.ok && !result.physically_complete && result.error=="GPU pending",
          "successful ANE hid pending postprocessing GPU");
    result=contract::join_physical_domain(failed,true,false,false,true,"consumer missing");
    check(result.error=="GPU pending; consumer missing" && !result.physically_complete,
          "failed domain diagnostic was discarded");
}
} // namespace
int main() {
    try {
        options(); deadlines(); completion_domains();
        std::cout<<"native_qwen_hybrid_denoiser_cpu_contract=PASS devices=0 models=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
