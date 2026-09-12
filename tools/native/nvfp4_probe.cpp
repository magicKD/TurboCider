#include "../../native/backends/mlx.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3, "usage: nvfp4-probe CHECKPOINT OUTPUT");
        tc::configure_streams();
        tc::Weights w;
        w.load_file(argv[1]);
        auto count = w.pack_comfy_nvfp4();
        tc::require(count > 0, "no NVFP4 layers");
        const std::string prefix = "layers.0.attention.qkv";
        auto weight = tc::mx::dequantize(w.at(prefix + ".weight"), w.at(prefix + ".nvfp4_scales"),
            std::nullopt, 16, 4, "nvfp4", std::nullopt, tc::mx::float32) * w.at(prefix + ".weight_scale_2");
        auto x = tc::mx::reshape(tc::mx::sin(tc::mx::arange(32 * 3840, tc::mx::float32)), {32, 3840});
        x = tc::mx::astype(x, tc::mx::bfloat16);
        auto y = w.project(x, prefix);
        auto ref = tc::mx::matmul(tc::mx::astype(x, tc::mx::float32), tc::mx::transpose(weight));
        tc::mx::eval(weight, y, ref);
        auto delta = tc::mx::astype(y, tc::mx::float32) - ref;
        const auto relative = tc::mx::sqrt(tc::mx::sum(delta * delta) / tc::mx::sum(ref * ref));
        tc::mx::eval(relative);
        tc::require(relative.item<float>() < 0.015f, "NVFP4 projection differs from decoded-weight oracle");
        tc::mx::save_safetensors(argv[2], {{"weight", weight}});
        std::cout << "{\"packed_layers\":" << count << ",\"projection_relative_l2\":" << relative.item<float>() << "}" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
