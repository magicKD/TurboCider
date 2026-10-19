#include "../../native/models/qwen21/runtime_private_qkv_options.hpp"
#include <cassert>
#include <iostream>

int main(){using namespace tc;using namespace tc::qwen21;
    Request r;r.model="qwen-image-2.1";r.model_variant="qwen-image-2.1";r.operation="image.edit";
    r.execution="gpu_ane";r.hybrid_mlp_mode="runtime";r.allow_approximation=true;r.residency="resident";
    r.width=r.height=512;r.steps=40;r.frames=1;r.audio=false;r.ane_manifest="shape.json";r.qwen21_reference_size=1024;
    r.inputs.emplace_back();r.inputs.back().kind="image";r.inputs.back().role="reference";r.inputs.back().path="reference.png";
    unsetenv("TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0");
    unsetenv("TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT");
    assert(configured_runtime_edit_qkv_step0(r)==RuntimeTextQkvStep0::Ane);
    setenv("TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0","gpu",1);
    auto denied=[](auto&&f){try{f();}catch(const std::exception&){return true;}return false;};
    assert(denied([&]{configured_runtime_edit_qkv_step0(r);}));
    setenv("TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT","1",1);
    assert(configured_runtime_edit_qkv_step0(r)==RuntimeTextQkvStep0::Gpu);
    for(auto change:{0,1,2,3,4}){auto v=r;
        if(change==0)v.steps=8;if(change==1)v.width=768;if(change==2)v.loras.push_back({"a",1.f,"transformer"});
        if(change==3)v.model_variant="qwen-image-2.1-turbo";if(change==4)v.qwen21_reference_size=512;
        assert(denied([&]{configured_runtime_edit_qkv_step0(v);}));}
    {auto v=r;v.operation="image.generate";v.inputs.clear();assert(configured_runtime_edit_qkv_step0(v)==RuntimeTextQkvStep0::Ane);}
    for(int step=0;step<40;++step)for(int layer=0;layer<32;++layer){
        assert(runtime_text_qkv_permitted(RuntimeTextQkvStep0::Gpu,true,step)==(step!=0));
        assert(runtime_text_qkv_permitted(RuntimeTextQkvStep0::Gpu,false,step));
        const auto next=qkv_lookahead_target(layer,step,1024,RuntimeTextQkvStep0::Gpu,true);
        if(step==39&&layer==31)assert(!next.permitted);
        else assert(next.permitted==(layer==31?step+1:step)>0);
    }
    const auto boundary=qkv_lookahead_target(31,0,1024,RuntimeTextQkvStep0::Gpu,true);
    assert(boundary.layer==0&&boundary.step==1&&boundary.permitted);
    assert(runtime_edit_qkv_recipe(RuntimeTextQkvStep0::Ane)=="base-private-w8-prefix-v1:");
    assert(runtime_edit_qkv_recipe(RuntimeTextQkvStep0::Gpu)!=runtime_edit_qkv_recipe(RuntimeTextQkvStep0::Ane));
    setenv("TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0","auto",1);
    assert(denied([&]{configured_runtime_edit_qkv_step0(r);}));
    unsetenv("TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0");unsetenv("TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT");
    std::cout<<"PASS edit GPU-step0 domain, unchanged default recipe, all QKV decisions and step0->1 prefetch boundary\n";
}
