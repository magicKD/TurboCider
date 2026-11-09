#include "../../native/models/ltx098/transformer.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 6 || argc == 7, "usage: ltx098-transformer-probe CHECKPOINT INPUT.safetensors OUTPUT.safetensors BLOCKS EVAL_EVERY [SIGMA]");
        auto input = tc::mx::load_safetensors(argv[2]).first;
        tc::ltx098::Transformer model;
        model.load(std::filesystem::absolute(argv[1]), std::stoi(argv[4]), std::stoi(argv[5]));
        auto text = model.prepare_text(input.at("text"));
        auto mask = input.at("text_mask");
        const float sigma = argc == 7 ? std::stof(argv[6]) : .7f;
        auto output = model.forward(input.at("latents"), text, sigma, 24.f, true, 1.f, mask);
        tc::mx::eval(output);
        tc::mx::save_safetensors(argv[3], {{"output", output}});
        std::cout << "LTX 0.9.8 transformer output saved\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
