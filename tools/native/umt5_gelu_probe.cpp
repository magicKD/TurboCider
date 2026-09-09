#include "../../native/components/text/umt5.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3, "usage: umt5-gelu-probe INPUT OUTPUT.safetensors");
        auto input = tc::mx::load_safetensors(argv[1]).first.at("first_gate_input");
        auto output = tc::components::UMT5Encoder::activation(input);
        tc::mx::eval(output);
        tc::mx::save_safetensors(argv[2], {{"output", output}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
