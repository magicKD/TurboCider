// Header-only request domain checks in both public/private builds. JSON type
// and placement checks live in test_qwen_native_route.py against the rebuilt ABI.
#include "../../native/models/qwen21/native_route.hpp"
#include <iostream>

namespace {
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F&& f) {
    bool rejected=false;
    try { f(); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"native scale-cache option escaped its request domain");
}
tc::Request gpu() {
    tc::Request r;
    r.model="qwen-image-2.1"; r.operation="image.generate";
    r.width=512; r.height=512; r.steps=20; r.frames=1;
    r.execution="gpu"; r.residency="component_staged"; r.allow_approximation=true;
    r.qwen21_denoiser_backend="native-metal-graph";
    return r;
}
void domain() {
    const auto base=gpu();
    check(!base.qwen21_native_scale_cache && !base.qwen21_native_scale_cache_specified,
          "scale cache must remain off and absent by default");
    tc::qwen21::validate_native_denoiser_request(base);
    auto w1=base; w1.execution="gpu_ane"; w1.qwen21_w8a8=true;
    w1.qwen21_native_ane_cache="/private/tmp/native-scale-cache-contract-unused";
    for (bool enabled:{false,true}) for (bool specialized:{false,true}) {
        auto variant=w1; variant.qwen21_native_scale_cache=enabled;
        variant.qwen21_native_scale_cache_specified=true;
        variant.qwen21_native_w8_staging=specialized?"register":"generic";
        variant.qwen21_native_w8_staging_specified=true;
#if defined(TURBOCIDER_ENABLE_PRIVATE_ANE) && TURBOCIDER_ENABLE_PRIVATE_ANE
        tc::qwen21::validate_native_denoiser_request(variant);
#else
        rejects([&] { tc::qwen21::validate_native_denoiser_request(variant); });
#endif
        rejects([&] { auto bad=variant;bad.loras.push_back({"adapter.safetensors",1.f,"transformer"});
            tc::qwen21::validate_native_denoiser_request(bad); });
        rejects([&] { auto bad=variant;bad.operation="image.edit";
            tc::qwen21::validate_native_denoiser_request(bad); });
    }
    for (bool specified:{false,true}) for (bool enabled:{false,true}) {
        if (!specified && !enabled) continue;
        for (bool mlx:{false,true}) {
            auto invalid=base;
            invalid.qwen21_native_scale_cache=enabled;
            invalid.qwen21_native_scale_cache_specified=specified;
            if (mlx) invalid.qwen21_denoiser_backend="mlx";
            rejects([&] { tc::qwen21::validate_native_denoiser_request(invalid); });
        }
    }
    auto absent=base; absent.qwen21_denoiser_backend="mlx";
    tc::qwen21::validate_native_denoiser_request(absent);
}
} // namespace
int main() {
    try {
        domain();
        std::cout<<"native_qwen_scale_cache_route_contract=PASS devices=0 models=0\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
