#include "../../native/models/ltx098/activation.hpp"

#include <cmath>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3 && !std::filesystem::exists(argv[2]),
                    "usage: ltx098-activation-probe INPUT.safetensors NEW_OUTPUT.safetensors");
        auto input = tc::mx::load_safetensors(argv[1]).first.at("input");
        tc::require(input.dtype() == tc::mx::bfloat16, "BF16 input required");
        const auto type = input.dtype();
        const tc::Tensor one(1.f, type), half(.5f, type), cubic(.044715f, type);
        const tc::Tensor factor(float(std::sqrt(2.0 / M_PI)), type);
        auto old = half * input * (one + tc::mx::tanh(factor * (input + cubic * tc::mx::power(input, tc::Tensor(3.f, type)))));
        auto candidate = tc::ltx098::gelu_tanh(input);
        tc::mx::eval(old, candidate);
        tc::mx::save_safetensors(argv[2], {{"old", old}, {"candidate", candidate}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
