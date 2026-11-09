#include "../../native/models/ltx098/vae.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4, "usage: ltx098-vae-encoder-probe CHECKPOINT INPUT.safetensors OUTPUT.safetensors");
        auto input = tc::mx::load_safetensors(argv[2]).first;
        tc::ltx098::VAEEncoder encoder;
        encoder.load(std::filesystem::absolute(argv[1]));
        auto result = encoder.encode(input.at("pixels"));
        tc::mx::eval(result);
        tc::mx::save_safetensors(argv[3], {{"latents", result}});
        std::cout << "LTX 0.9.8 encoder output saved; weight bytes=" << encoder.bytes() << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
