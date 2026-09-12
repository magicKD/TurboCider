#include "../../native/models/h3_mlx/video_vae.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3,
                    "usage: h3-mlx-video-vae-probe VAE_DIR OUTPUT");
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::h3_mlx::VideoVAE vae;
        vae.load(std::filesystem::absolute(argv[1]), event, cancelled);
        auto normalized = tc::mx::zeros({1, 24, 7, 2, 2}, tc::mx::float32);
        auto latents = vae.denormalize_latents(normalized);
        auto pixels = vae.decode(latents, 22, 32, 32, false, event, cancelled);
        tc::mx::save_safetensors(std::filesystem::absolute(argv[2]).string(),
                                 {{"latents", latents}, {"pixels", pixels}});
        std::cout << "{\"frames\":" << pixels.shape(2)
                  << ",\"height\":" << pixels.shape(3)
                  << ",\"width\":" << pixels.shape(4) << "}\n";
        vae.unload();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
