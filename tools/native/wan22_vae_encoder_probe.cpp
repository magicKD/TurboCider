#include "../../native/models/wan22/vae.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4, "usage: wan22-vae-encoder-probe WEIGHTS INPUT OUTPUT");
        auto input = tc::mx::load_safetensors(argv[2]).first;
        tc::wan22::VAEEncoder encoder;
        encoder.load(std::filesystem::absolute(argv[1]));
        auto output = encoder.encode_first_frame(input.at("pixels"));
        tc::mx::eval(output);
        tc::mx::save_safetensors(argv[3], {{"output", output}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
