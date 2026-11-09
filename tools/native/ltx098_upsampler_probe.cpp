#include "../../native/models/ltx098/vae.hpp"
#include "../../native/models/ltx_runtime/ltx_mlx_upsampler.h"

#include <array>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5, "usage: ltx098-upsampler-probe MODEL_ROOT INPUT_LATENT OUTPUT_LATENT legacy|ltx098");
        const std::filesystem::path root(argv[1]), output(argv[3]);
        tc::require(!std::filesystem::exists(output), "output must not exist");
        auto state = tc::mx::load_safetensors(argv[2]).first.at("value");
        tc::require(state.ndim() == 5 && state.shape(0) == 1 && state.shape(4) == 128, "invalid input latent");
        const int f = state.shape(1), h = state.shape(2), w = state.shape(3);
        tc::ltx098::VAEEncoder vae;
        vae.load(root / "ltxv-2b-0.9.8-distilled.safetensors");
        auto unnormalized = tc::mx::contiguous(vae.denormalize_latents(tc::mx::astype(state, tc::mx::bfloat16)));
        tc::mx::eval(unnormalized);
        const size_t count = size_t(f) * h * w * 128;
        std::vector<uint16_t> result(count * 4);
        std::array<char, 2048> error{};
        const std::string mode(argv[4]);
        tc::require(mode == "legacy" || mode == "ltx098", "unknown upsampler mode");
        std::unique_ptr<ltx_mlx_upsampler, decltype(&ltx_mlx_upsampler_free)> upsampler(
            mode == "legacy" ? ltx_mlx_upsampler_create((root / "ltxv-spatial-upscaler-0.9.8.safetensors").c_str(), error.data(), error.size())
                             : ltx_mlx_upsampler_create_ltx098((root / "ltxv-spatial-upscaler-0.9.8.safetensors").c_str(), error.data(), error.size()),
            &ltx_mlx_upsampler_free);
        tc::require(bool(upsampler), error.data());
        tc::require(ltx_mlx_upsampler_run_tokens_bf16(upsampler.get(), result.data(), result.size(),
                    unnormalized.data<uint16_t>(), count, 1, f, h, w, error.data(), error.size()), error.data());
        auto high = tc::Tensor(reinterpret_cast<const tc::mx::bfloat16_t *>(result.data()), {1, f, h * 2, w * 2, 128});
        high = tc::mx::astype(vae.normalize_latents(high), tc::mx::float32);
        tc::mx::eval(high);
        tc::mx::save_safetensors(output.string(), {{"value", high}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
