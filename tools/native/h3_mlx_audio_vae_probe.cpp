#include "../../native/models/h3_mlx/audio_vae.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3,
                    "usage: h3-mlx-audio-vae-probe AUDIO_VAE_DIR OUTPUT");
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::h3_mlx::AudioVAE vae;
        vae.load(std::filesystem::absolute(argv[1]), event, cancelled);
        auto normalized = tc::mx::zeros({2, 32, 4}, tc::mx::float32);
        auto latents = vae.denormalize_latents(normalized);
        auto waveform = vae.decode(latents, event, cancelled);
        tc::mx::save_safetensors(std::filesystem::absolute(argv[2]).string(),
                                 {{"latents", latents},
                                  {"waveform", waveform}});
        std::cout << "{\"samples\":" << waveform.shape(-1)
                  << ",\"channels\":" << waveform.shape(0) << "}\n";
        vae.unload();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
