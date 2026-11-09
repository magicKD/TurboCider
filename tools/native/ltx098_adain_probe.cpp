#include "../../native/models/ltx098/adain.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3, "usage: ltx098-adain-probe INPUT OUTPUT");
        auto values = tc::mx::load_safetensors(argv[1]).first;
        auto output = tc::ltx098::adain_latents(values.at("upsampled"), values.at("reference"));
        tc::mx::eval(output);
        tc::mx::save_safetensors(argv[2], {{"value", output}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
