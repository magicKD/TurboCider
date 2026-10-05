#include "../../native/backends/ane_smoothquant_mlx.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include <iostream>

namespace mx = tc::mx;
namespace sq = tc::ane::smoothquant;
static void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class Function> static void rejected(Function function) {
    bool failed = false;
    try { function(); } catch (const std::exception &) { failed = true; }
    check(failed,"invalid S1 ownership contract accepted");
}
static void fallback_metrics(const tc::ane::HybridFfn &runtime, const std::string &digest) {
    const auto metrics = runtime.metrics();
    check(metrics.runtime_weight_s1_requested == !digest.empty() && metrics.runtime_weight_s1_digest == digest &&
          metrics.runtime_weight_s1_bytes == 0 && metrics.runtime_weight_s1_bank_margin_bytes == 0 &&
          metrics.runtime_weight_s1_stage_submissions == 0 && metrics.runtime_weight_s1_hybrid_blocks == 0 &&
          metrics.runtime_weight_source_recipe.find("smoothquant") == std::string::npos &&
          !runtime.available() && !runtime.retains_resources(),"fallback retained or claimed S1 application");
    // The actual HybridMetrics exporter is sufficient; RunResult export would
    // collect device information and is deliberately not exercised here.
    NSDictionary *exported = tc::to_dictionary(metrics)[@"runtime_weight"];
    check([exported isKindOfClass:NSDictionary.class],"runtime receipt missing");
    check(![exported[@"a8_single_pass_requested"] boolValue] &&
          ![exported[@"a8_single_pass_pipeline_compiled"] boolValue] &&
          [exported[@"a8_single_pass_submissions_session_total"] unsignedLongLongValue] == 0 &&
          [exported[@"a8_single_pass_ineligible_submissions_session_total"] unsignedLongLongValue] == 0 &&
          [exported[@"a8_single_pass_counter_scope"] containsString:@"not_completion_or_speed_evidence"],
          "disabled executor attested A8 candidate execution or speed");
    for (NSString *key in @[@"s1_requested",@"s1_vector_bytes",@"s1_bank_margin_bytes",
                           @"s1_stage_submissions_profile_total",@"s1_hybrid_blocks_profile_total",
                           @"a8_single_pass_requested",@"a8_single_pass_pipeline_compiled",
                           @"a8_single_pass_submissions_session_total",
                           @"a8_single_pass_ineligible_submissions_session_total"])
        check([exported[key] isKindOfClass:NSNumber.class],"S1 numeric exporter field missing");
    check([exported[@"s1_requested"] boolValue] == !digest.empty() &&
          [exported[@"s1_profile_digest"] isEqualToString:@(digest.c_str())] &&
          [exported[@"s1_vector_bytes"] unsignedLongLongValue] == 0 &&
          [exported[@"s1_bank_margin_bytes"] unsignedLongLongValue] == 0 &&
          [exported[@"s1_stage_submissions_profile_total"] unsignedLongLongValue] == 0 &&
          [exported[@"s1_hybrid_blocks_profile_total"] unsignedLongLongValue] == 0 &&
          exported[@"executor_backend"] == NSNull.null &&
          [exported[@"s1_counter_scope"] containsString:@"does_not_attest_quality_or_physical_INT8"],
          "exporter misreported requested versus applied S1");
}
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 2,"expected disposable CPU fixture directory");
            mx::set_default_device(mx::Device(mx::Device::cpu));
            ::setenv("TURBOCIDER_ANE_BACKEND","off",1);
            ::setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","1",1);
            ::setenv("TURBOCIDER_RUNTIME_ANE_PROFILE","0",1);
            ::unsetenv("TURBOCIDER_PRIVATE_ANE_S1_PROFILE");
            ::unsetenv("TURBOCIDER_ANE_CALIBRATION_DIR");
            const auto missing = std::filesystem::path(argv[1]) / "never-created-model-or-manifest.json";
            std::atomic<bool> cancelled{false};
            tc::ane::HybridFfn runtime(missing,8,512,64*1024*1024,cancelled);
            check(!std::filesystem::exists(missing) && runtime.reason() == "ANE disabled by backend policy",
                  "disabled backend accessed a manifest/model");
            fallback_metrics(runtime,{});
            int produced = 0;
            const auto producer = [&]() -> std::vector<tc::Tensor> {
                ++produced; throw std::runtime_error("fallback must not allocate S1 scales");
            };
            const std::string digest(64,'a');
            runtime.set_smoothquant(digest,producer);
            runtime.set_smoothquant(digest,producer);
            check(produced == 0,"unavailable graph invoked S1 producer");
            fallback_metrics(runtime,digest);
            for (const auto &bad : {std::string(63,'a'),std::string(65,'a'),std::string(64,'A'),std::string(64,'z')}) {
                rejected([&] { runtime.set_smoothquant(bad,producer); });
                fallback_metrics(runtime,digest);
            }
            rejected([&] { runtime.set_smoothquant(digest); });
            rejected([&] { runtime.set_smoothquant({},producer); });
            check(produced == 0,"invalid contract invoked S1 producer");
            fallback_metrics(runtime,digest);

            // Exercise a real fallback dispatch, with a tiny CPU identity
            // callback, so its counter cannot be mistaken for an S1 block.
            const std::vector<float> values(16,1.f);
            const tc::Tensor input(values.data(),{1,2,8},mx::float32);
            runtime.stage(0,2,{});
            int cpu_calls = 0;
            const auto output = runtime.run(0,input,[&](const tc::Tensor &x) { ++cpu_calls; return x; },cancelled);
            mx::eval(output);
            check(cpu_calls == 1 && output.shape() == input.shape() &&
                  runtime.metrics().runtime_weight_gpu_blocks == 1,"disabled FFN did not use baseline callback");
            fallback_metrics(runtime,digest);

            runtime.set_smoothquant();
            fallback_metrics(runtime,{});
            runtime.set_smoothquant(digest,producer);
            tc::Request request; // Unset profile binding clears without checkpoint access.
            sq::bind_request(runtime,request,missing,"unchanged-adapter-identity",8);
            fallback_metrics(runtime,{});
            ::setenv("TURBOCIDER_PRIVATE_ANE_S1_PROFILE",missing.c_str(),1);
            request.model = "z-image-turbo"; request.hybrid_mlp_mode = "runtime";
            request.execution = "gpu_ane"; request.allow_approximation = true;
            request.width = request.height = 512;
            rejected([&] { sq::bind_request(runtime,request,missing,"unchanged-adapter-identity",8); });
            fallback_metrics(runtime,{});
            ::unsetenv("TURBOCIDER_PRIVATE_ANE_S1_PROFILE");
            check(produced == 0 && !std::filesystem::exists(missing),"binding accessed model or produced scales");
            std::cout << "PASS CPU S1 FFN ownership/fallback/exporter no model/no ANE\n";
        } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
