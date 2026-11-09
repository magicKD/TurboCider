#include "../../native/models/ltx098/activation.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3 && !std::filesystem::exists(argv[2]),
                    "usage: ltx098-addcmul-probe INPUT.safetensors NEW_OUTPUT.safetensors");
        auto inputs = tc::mx::load_safetensors(argv[1]).first;
        auto result = tc::ltx098::addcmul(inputs.at("a"), inputs.at("b"), inputs.at("c"));
        tc::mx::eval(result);
        tc::mx::save_safetensors(argv[2], {{"output", result}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
