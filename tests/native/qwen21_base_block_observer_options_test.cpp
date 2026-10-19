#include "../../native/models/qwen21/diagnostic_options.hpp"
#include <cassert>
int main() {
    using tc::qwen21::base_block_observer_request;
    tc::Request r;r.model="qwen-image-2.1";r.operation="image.generate";r.width=r.height=512;
    r.steps=40;r.frames=1;r.audio=false;r.execution="gpu";r.residency="resident";
    assert(base_block_observer_request(r));
    auto no=[&](auto change){auto v=r;change(v);assert(!base_block_observer_request(v));};
    no([](auto&v){v.model="other";});no([](auto&v){v.steps=20;});no([](auto&v){v.steps=6;});
    no([](auto&v){v.width=1024;});no([](auto&v){v.audio=true;});no([](auto&v){v.frames=2;});
    no([](auto&v){v.allow_approximation=true;});no([](auto&v){v.execution="gpu_ane";});
    no([](auto&v){v.residency="component_staged";});no([](auto&v){v.hybrid_mlp_mode="runtime";});
    no([](auto&v){v.qwen21_denoiser_backend="native_metal";});
    no([](auto&v){v.loras.push_back({});});no([](auto&v){v.ane_manifest="other";});
    no([](auto&v){v.qwen21_w8a8=true;});no([](auto&v){v.qwen21_gpu_w8a16=true;});
    no([](auto&v){v.encoder_ane_manifest="other";});no([](auto&v){v.memory_budget_bytes=1;});
    no([](auto&v){v.memory_constrained.enabled=true;});no([](auto&v){v.prompt_enhance=true;});
    no([](auto&v){v.dump="trace";});no([](auto&v){v.noise_path="noise";});
    no([](auto&v){v.inputs.push_back({});});no([](auto&v){v.qwen21_dit_cache="fast";});
    r.operation="image.edit";r.inputs.push_back({});assert(base_block_observer_request(r));
    r.inputs.push_back({});assert(base_block_observer_request(r));
    r.inputs.push_back({});assert(!base_block_observer_request(r));
    r.inputs.resize(1);r.qwen21_reference_size=512;assert(!base_block_observer_request(r));
}
