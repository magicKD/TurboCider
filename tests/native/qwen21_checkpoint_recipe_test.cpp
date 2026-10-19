#include "../../native/models/qwen21/checkpoint_recipe.hpp"
#include <fstream>
#include <iostream>

namespace {
void check(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejected(F f){bool denied=false;try{f();}catch(const std::invalid_argument&){denied=true;}check(denied,"invalid checkpoint/sampler recipe accepted");}
}
int main(int argc,char**argv){try{
    using namespace tc;using namespace tc::qwen21;check(argc==2,"fresh temporary directory required");
    const std::filesystem::path root(argv[1]);std::filesystem::create_directory(root/"diffusion_models");
    rejected([&]{transformer_checkpoint_source(root);});
    const auto turbo=root/"diffusion_models/qwen_image_2.1_turbo_bf16.safetensors";
    {std::ofstream f(turbo);f<<"metadata-only source selection fixture";}
    check(transformer_checkpoint_source(root)==turbo&&turbo_checkpoint(turbo),"original Turbo filename not selected");
    const auto base=root/"diffusion_models/qwen_image_2.1_bf16.safetensors";
    {std::ofstream f(base);f<<"metadata-only base preference fixture";}
    check(transformer_checkpoint_source(root)==base&&!turbo_checkpoint(base),"existing base preference changed");
    Request r;r.model="qwen-image-2.1";r.operation="image.generate";r.steps=8;r.frames=1;r.audio=false;
    validate_checkpoint_recipe(r,turbo);
    for(auto dims:{std::pair{512,512},std::pair{768,512},std::pair{1024,1024}}){r.width=dims.first;r.height=dims.second;validate_turbo_request(r);}
    r.model_variant="qwen-image-2.1-turbo";validate_checkpoint_recipe(r,turbo);rejected([&]{validate_checkpoint_recipe(r,base);});
    r.model_variant="qwen-image-2.1";rejected([&]{validate_checkpoint_recipe(r,turbo);});r.model_variant="auto";
    for(int steps:{0,1,6,7,9,20,40}){r.steps=steps;rejected([&]{validate_turbo_request(r);});}r.steps=8;
    r.execution="gpu_ane";rejected([&]{validate_turbo_request(r);});r.hybrid_mlp_mode="runtime";r.allow_approximation=true;validate_turbo_request(r);
    r.qwen21_w8a8=true;rejected([&]{validate_turbo_request(r);});r.qwen21_w8a8=false;
    r.qwen21_dit_cache="conservative";rejected([&]{validate_turbo_request(r);});r.qwen21_dit_cache="off";
    r.loras.emplace_back();rejected([&]{validate_turbo_request(r);});r.loras.clear();
    r.operation="image.edit";rejected([&]{validate_turbo_request(r);});r.inputs.resize(2);validate_turbo_request(r);
    r.inputs.resize(3);rejected([&]{validate_turbo_request(r);});r.inputs.clear();r.operation="image.generate";
    const auto convrot=root/"diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors";
    check(turbo_checkpoint(convrot)&&convrot_checkpoint(convrot),"Turbo ConvRot recipe not distinct");
    rejected([&]{validate_checkpoint_recipe(r,convrot);});r.execution="gpu";validate_checkpoint_recipe(r,convrot);
    check(configured_convrot_packed(),"ConvRot default must preserve compact original codes/F32 coefficients");
    r.execution="gpu_ane";r.ane_manifest="fixed.json";
    setenv("TURBOCIDER_ANE_BACKEND","private",1);setenv("TURBOCIDER_ALLOW_PRIVATE_ANE","1",1);
    setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","convrot_w8a8",1);setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1",1);
    setenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1",1);validate_checkpoint_recipe(r,convrot);
    setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","w8a8",1);rejected([&]{validate_checkpoint_recipe(r,convrot);});
    setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","convrot_w8a8",1);setenv("TURBOCIDER_QWEN21_CONVROT_PACKED","0",1);
    rejected([&]{validate_checkpoint_recipe(r,convrot);});unsetenv("TURBOCIDER_QWEN21_CONVROT_PACKED");
    setenv("TURBOCIDER_QWEN21_PRIVATE_QKV","1",1);rejected([&]{validate_checkpoint_recipe(r,convrot);});unsetenv("TURBOCIDER_QWEN21_PRIVATE_QKV");
    const float nodes[]={1.f,.978453f,.95418f,.926626f,.89508f,.845148f,.704534f,.414568f,0.f};
    for(size_t i=0;i<9;++i){check(turbo_sigma_nodes[i]==nodes[i],"authored final sigma changed");if(i<8)check(nodes[i]>nodes[i+1],"Turbo nodes not descending");}
    std::cout<<"PASS checkpoint selection, separate Turbo fixed8 recipe and strict format/route guards\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
