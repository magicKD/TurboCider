#include "../../native/models/qwen21/vision.hpp"
#include <iostream>
int main(int argc,char **argv) {
    try {
        tc::require(argc==4,"usage: qwen21-vision-probe weights inputs output");
        tc::configure_streams();
        tc::Weights weights;
        weights.load_file(argv[1],"model.visual.");
        weights.remap_keys([](const std::string &key){return "model.visual."+key;});
        auto [inputs,meta]=tc::mx::load_safetensors(argv[2]);
        tc::qwen21::VisionConfig c;
        c.patch=std::stoi(meta.at("patch")); c.hidden=std::stoi(meta.at("hidden"));
        c.heads=std::stoi(meta.at("heads")); c.layers=std::stoi(meta.at("layers"));
        c.position_side=std::stoi(meta.at("position_side"));
        c.deepstack_layers={std::stoi(meta.at("deep0")),std::stoi(meta.at("deep1")),std::stoi(meta.at("deep2"))};
        tc::qwen21::VisionEncoder encoder(weights,c);
        std::atomic<bool> cancel{false};
        auto features=encoder.encode(inputs.at("patches"),std::stoi(meta.at("height")),std::stoi(meta.at("width")),{},cancel);
        std::unordered_map<std::string,tc::Tensor> outputs{{"merged",features.merged}};
        for(size_t i=0;i<features.deepstack.size();++i) outputs.emplace("deep"+std::to_string(i),features.deepstack[i]);
        tc::mx::save_safetensors(argv[3],outputs);
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
