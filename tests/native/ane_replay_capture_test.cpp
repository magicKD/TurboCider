#include "../../native/backends/ane_calibration_mlx.hpp"
#include <fstream>
#include <iostream>
namespace cal=tc::ane::calibration;
namespace mx=tc::mx;
int main(int argc,char **argv) {
 try {
    tc::require(argc==2,"expected temporary directory");
    mx::set_default_device(mx::Device(mx::Device::cpu));
    const auto root=std::filesystem::path(argv[1]);
    const auto checkpoint=root/"fake-checkpoint";std::ofstream(checkpoint)<<"stat fixture";
    const auto captures=root/"captures";
    ::setenv("TURBOCIDER_ANE_CALIBRATION_DIR",captures.c_str(),1);
    ::setenv("TURBOCIDER_ANE_CALIBRATION_REPLAY","1",1);
    ::setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","0",1);
    ::setenv("TURBOCIDER_RUNTIME_ANE_PROFILE","0",1);
    tc::Request request;request.model="qwen-image-2.1";request.hybrid_mlp_mode="runtime";
    request.width=request.height=512;request.steps=6;request.qwen21_dit_cache="off";
    request.prompt="CPU fixture";
    auto reject=[&] {bool failed=false;try {cal::ScopedCapture bad(request,checkpoint,"");}catch(...){failed=true;}
        tc::require(failed && !cal::ScopedCapture::current(),"invalid replay request accepted");};
    ::setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","1",1);reject();
    ::setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","0",1);
    request.model="z-image-turbo";reject();request.model="qwen-image-2.1";
    request.steps=2;reject();request.steps=6;
    auto input=mx::ones({1,1024,4096},mx::bfloat16)*tc::Tensor(1.125f,mx::bfloat16);
    tc::Weights weights;weights.bind_arrays({"fixture.weight"},{mx::ones({2,4096},mx::bfloat16)});
    {
        cal::ScopedCapture owner(request,checkpoint,"");
        owner.step(0);tc::require(!owner.wants(0),"prefill unexpectedly captured");
        for(int step:{1,3,5}) {
            owner.step(step);tc::require(!owner.wants(7),"unselected layer captured");
            for(int layer:{0,15,31}) {
                tc::require(owner.wants(layer),"missing point");
                owner.observe(layer,input,weights,{"fixture"},{layer,step,"denoise",{{"target",0,1024}}});
                tc::require(!owner.wants(layer),"duplicate point allowed");
            }
        }
        owner.finish();owner.finish();
    }
    size_t count=0,bytes=0;bool complete=false;
    for(const auto &dir:std::filesystem::directory_iterator(captures)) {
        if(!dir.path().filename().string().ends_with("-replay"))continue;
        complete=std::filesystem::is_regular_file(dir.path()/"complete.json");
        for(const auto &file:std::filesystem::directory_iterator(dir.path())) {
            bytes+=file.file_size();if(file.path().extension()!=".safetensors")continue;
            auto [tensors,metadata]=mx::load_safetensors(file.path().string());
            auto saved=tensors.at("input");
            tc::require(saved.dtype()==mx::bfloat16 && saved.shape()==input.shape() &&
                mx::all(saved==input).item<bool>() && metadata.at("trajectory")=="forced-gpu",
                "original BF16 replay input/provenance changed");++count;
        }
    }
    tc::require(complete && count==9 && bytes<73*1024*1024,"replay completion/budget mismatch");
    std::cout<<"PASS replay original BF16/selection/forced GPU/budget/completion\n";
 }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
