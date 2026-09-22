#include "../../native/models/qwen21/vae.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4, "usage: qwen21-vae-probe weights inputs output");
        tc::configure_streams();
        tc::Weights weights;
        weights.load_file(argv[1]);
        tc::qwen21::VAE vae(weights);
        weights.clear();
        auto [inputs, metadata] = tc::mx::load_safetensors(argv[2]);
        std::atomic<bool> cancelled{false};
        std::unordered_map<std::string, tc::Tensor> outputs;
        if (inputs.count("latents")) {
            auto output = vae.decode(inputs.at("latents"), {}, cancelled);
            tc::mx::eval(output);
            tc::require(output.shape(1) == 4, "VAE dropped alpha channel");
            outputs.emplace("decoded", output);
        }
        if (inputs.count("image")) outputs.emplace("encoded", vae.encode(inputs.at("image"), {}, cancelled,
            metadata.count("trace") ? &outputs : nullptr));
        tc::mx::save_safetensors(argv[3], outputs);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
