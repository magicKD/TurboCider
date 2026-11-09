#include "../../native/backends/mlx.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3 && !std::filesystem::exists(argv[2]), "usage: ltx098-silu-probe INPUT NEW_OUTPUT");
        auto x = tc::mx::load_safetensors(argv[1]).first.at("x");
        tc::require(x.dtype() == tc::mx::bfloat16, "expected BF16 input");
        auto old = tc::silu(x);
        static auto graph = tc::mx::compile([](const std::vector<tc::Tensor> &args) {
            auto f = tc::mx::astype(args[0], tc::mx::float32);
            return std::vector<tc::Tensor>{tc::mx::astype(f * tc::mx::sigmoid(f), args[0].dtype())};
        }, true);
        auto candidate = graph({x})[0];
        tc::mx::eval(old, candidate);
        tc::mx::save_safetensors(argv[2], {{"current", old}, {"fp32_intermediate", candidate}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
