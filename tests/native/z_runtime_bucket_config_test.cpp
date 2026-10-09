#include "../../native/models/z_image/runtime_bucket_config.hpp"
#include <iostream>

using namespace tc;
int main(){try {
    require(z_image::matched_runtime_bucket(1024,32)==1152 && z_image::matched_runtime_bucket(1024,64)==1152 &&
        z_image::matched_runtime_bucket(1024,128)==1152 && z_image::matched_runtime_bucket(1024,160)==1280 &&
        z_image::matched_runtime_bucket(1024,7168)==8192,"caption bucket grid/cliff policy mismatch");
    require(z_image::matched_runtime_bucket(1024,32,1056)==1056 && z_image::matched_runtime_bucket(1024,64,1056)==1152 &&
        z_image::matched_runtime_bucket(1024,32,1152)==1152,"sufficient template was unnecessarily enlarged");
    for(auto shape:std::vector<std::pair<int,int>>{{0,32},{1024,0},{1024,33},{1024,7200}}) {
        bool rejected=false;try{z_image::matched_runtime_bucket(shape.first,shape.second);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"invalid/overflow caption geometry accepted");
    }
    Request r;r.model="z-image-turbo";r.operation="image.generate";r.width=r.height=512;r.residency="resident";
    r.execution="gpu_ane";r.hybrid_mlp_mode="runtime";r.allow_approximation=true;
    setenv("TURBOCIDER_Z_RUNTIME_MATCH_ROWS","1",1);setenv("TURBOCIDER_ANE_BACKEND","private",1);
    setenv("TURBOCIDER_ALLOW_PRIVATE_ANE","1",1);setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","4096",1);
    setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","w8a8",1);setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","1",1);
    setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);
    require(z_image::configured_runtime_match_rows(r),"valid explicit BF16 policy declined");
    r.model="z-image-turbo-gguf";require(z_image::configured_runtime_match_rows(r),"valid GGUF policy declined");
    for(const auto &value:std::vector<std::pair<const char*,const char*>>{{"TURBOCIDER_ANE_BACKEND","public"},
        {"TURBOCIDER_ANE_BACKEND","auto"},{"TURBOCIDER_PRIVATE_ANE_CHANNELS","0"},{"TURBOCIDER_PRIVATE_ANE_CHANNELS","auto"},
        {"TURBOCIDER_PRIVATE_ANE_DATA_PATH","fp16"},{"TURBOCIDER_RUNTIME_ANE_CHUNKS","auto"},
        {"TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","0"},{"TURBOCIDER_Z_RUNTIME_MATCH_ROWS","2"}}) {
        const auto old=std::string(std::getenv(value.first));setenv(value.first,value.second,1);bool rejected=false;
        try{z_image::configured_runtime_match_rows(r);}catch(const std::exception&){rejected=true;}
        setenv(value.first,old.c_str(),1);require(rejected,"unsupported runtime bucket policy accepted");
    }
    r.execution="gpu";require(!z_image::configured_runtime_match_rows(r),"GPU control acquired row override");
    std::cout<<"PASS request-matched Z bucket: bounded128 grid, caption cliff, BF16/GGUF fixed Private guards and GPU inert control\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
