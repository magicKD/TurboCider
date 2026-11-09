#include "../../native/models/wan22/block.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 8 || argc == 9, "usage: wan22-block-probe WEIGHTS INPUT OUTPUT HIDDEN HEADS INTERMEDIATE FIRST_FRAME_TOKENS [TRACE]");
        auto input = tc::mx::load_safetensors(argv[2]).first;
        tc::wan22::BlockConfig config;
        config.hidden = std::stoi(argv[4]); config.heads = std::stoi(argv[5]);
        config.intermediate = std::stoi(argv[6]);
        tc::wan22::Block block;
        block.load(std::filesystem::absolute(argv[1]), 0, config);
        std::unordered_map<std::string, tc::Tensor> trace_values;
        tc::wan22::Block::Trace trace;
        if (argc == 9) trace = [&](const std::string &name, const tc::Tensor &value) {
            tc::mx::eval(value);
            trace_values.emplace(name, value);
        };
        auto output = block.forward(input.at("hidden"), input.at("context"), input.at("modulation"),
                                    input.at("cosine"), input.at("sine"), std::stoi(argv[7]), trace);
        tc::mx::eval(output);
        tc::mx::save_safetensors(argv[3], {{"output", output}});
        if (argc == 9) tc::mx::save_safetensors(argv[8], trace_values);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
