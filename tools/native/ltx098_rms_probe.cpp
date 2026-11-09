#include "../../native/backends/mlx.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4 && !std::filesystem::exists(argv[2]),
                    "usage: ltx098-rms-probe INPUT NEW_OUTPUT EPSILON");
        auto values = tc::mx::load_safetensors(argv[1]).first;
        const auto &x = values.at("x"), &weight = values.at("weight");
        tc::require(x.dtype() == tc::mx::bfloat16 && weight.dtype() == tc::mx::bfloat16 &&
                    x.shape(-1) == weight.shape(0), "BF16 RMS input/weight mismatch");
        const float epsilon = std::stof(argv[3]);
        auto f = tc::mx::astype(x, tc::mx::float32);
        auto wf = tc::mx::astype(weight, tc::mx::float32);
        auto unweighted_native = tc::mx::fast::rms_norm(x, std::nullopt, epsilon);
        auto unweighted_float = tc::mx::astype(tc::mx::fast::rms_norm(f, std::nullopt, epsilon), x.dtype());
        auto weighted_native = tc::mx::fast::rms_norm(x, weight, epsilon);
        auto weighted_float = tc::mx::astype(tc::mx::fast::rms_norm(f, wf, epsilon), x.dtype());
        auto weighted_separate = unweighted_float * weight;
        tc::mx::eval(unweighted_native, unweighted_float, weighted_native, weighted_float, weighted_separate);
        tc::mx::save_safetensors(argv[2], {{"unweighted_native", unweighted_native}, {"unweighted_float", unweighted_float},
            {"weighted_native", weighted_native}, {"weighted_float", weighted_float}, {"weighted_separate", weighted_separate}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
