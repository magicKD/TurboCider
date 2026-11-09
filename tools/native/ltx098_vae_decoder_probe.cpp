#include "../../native/models/ltx098/vae.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4, "usage: ltx098-vae-decoder-probe CHECKPOINT INPUT.safetensors OUTPUT.safetensors");
        tc::configure_streams();
        auto input = tc::mx::load_safetensors(argv[2]).first;
        tc::ltx098::VAEDecoder decoder;
        decoder.load(std::filesystem::absolute(argv[1]));
        auto result = decoder.decode(input.at("latents"), input.at("noise"));
        tc::mx::eval(result);
        tc::mx::save_safetensors(argv[3], {{"pixels", result}});
        std::cout << "LTX 0.9.8 decoder output saved; weight bytes=" << decoder.bytes() << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
