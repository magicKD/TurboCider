#include "../../native/models/ltx098/block.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 6, "usage: ltx098-block-probe CHECKPOINT INPUT.safetensors OUTPUT.safetensors BLOCK HEADS");
        auto input = tc::mx::load_safetensors(argv[2]).first;
        const auto &hidden = input.at("hidden");
        tc::ltx098::BlockConfig config;
        config.hidden = hidden.shape(-1);
        config.heads = std::stoi(argv[5]);
        tc::ltx098::Block block;
        block.load(std::filesystem::absolute(argv[1]), std::stoi(argv[4]), config);
        std::optional<tc::ltx_mlx::RopeFrequencies> rope;
        if (input.count("cosine") && input.count("sine")) rope.emplace(tc::ltx_mlx::RopeFrequencies{input.at("cosine"), input.at("sine")});
        std::optional<tc::Tensor> mask;
        if (input.count("text_mask")) mask.emplace(input.at("text_mask"));
        auto result = block.forward(hidden, input.at("context"), input.at("modulation"), rope, mask);
        tc::mx::eval(result);
        tc::mx::save_safetensors(argv[3], {{"output", result}});
        std::cout << "LTX 0.9.8 block output saved; weight bytes=" << block.bytes() << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
