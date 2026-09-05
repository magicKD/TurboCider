#include "../ltx_components.hpp"
#include <iostream>
int main(int argc,char **argv){@autoreleasepool {try {
    if(argc!=3)return 2;using namespace tc;std::filesystem::path source(argv[1]),output(argv[2]);
    mx::set_default_device(mx::Device(mx::Device::gpu));Weights weights;weights.load_file(source/"weights.safetensors");
    auto tensors=mx::load_safetensors((source/"input.safetensors").string()).first;std::atomic<bool> cancel{false};
    auto states=ltx_gemma_hidden(read_json(source/"config.json"),weights,tensors.at("ids"),tensors.at("mask"),[](auto&,int,int){},cancel);
    std::filesystem::create_directories(output);for(size_t i=0;i<states.size();++i)mx::save_safetensors((output/("layer_"+std::to_string(i)+".safetensors")).string(),{{"tensor",states[i]}});
    std::cout<<"{\"layers\":"<<states.size()<<"}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}}
