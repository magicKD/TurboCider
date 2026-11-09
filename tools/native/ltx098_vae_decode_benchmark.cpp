// Full 0.9.8 VAE decode only. Two separately linked binaries use either the
// production decoder or a probe-only padding overlay; inputs are identical.
#include "../../native/models/ltx098/vae.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 6 && !std::filesystem::exists(argv[3]),
                    "usage: ltx098-vae-decode-benchmark CHECKPOINT TRACE_DIR NEW_OUTPUT WARMUPS REPEATS");
        const auto trace = std::filesystem::absolute(argv[2]);
        const int warmups = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
        tc::require(warmups >= 1 && warmups <= 5 && repeats >= 2 && repeats <= 10, "invalid decode sample counts");
        tc::configure_streams();
        auto latent = tc::mx::astype(tc::mx::load_safetensors((trace / "step_7.safetensors").string()).first.at("value"), tc::mx::bfloat16);
        auto noise = tc::mx::load_safetensors((trace / "decode_noise.safetensors").string()).first.at("value");
        tc::require(latent.shape() == noise.shape() && noise.dtype() == tc::mx::bfloat16, "trace latent/noise mismatch");
        tc::mx::eval(latent, noise);
        tc::ltx098::VAEDecoder decoder;
        decoder.load(argv[1]);
        std::vector<double> samples;
        std::optional<tc::Tensor> result;
        for (int index = 0; index < warmups + repeats; ++index) {
            const auto start = std::chrono::steady_clock::now();
            result = decoder.decode(latent, noise);
            tc::mx::eval(*result);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (index >= warmups) samples.push_back(elapsed);
        }
        tc::require(result && result->dtype() == tc::mx::bfloat16 && tc::mx::all(tc::mx::isfinite(*result)).item<bool>(),
                    "decoder output is missing or nonfinite");
        tc::mx::save_safetensors(argv[3], {{"pixels", *result}});
        std::cout << std::setprecision(12) << "{\"scope\":\"full resident VAE decode only\",\"frames\":" << result->shape(1)
                  << ",\"height\":" << result->shape(2) << ",\"width\":" << result->shape(3)
                  << ",\"peak_mlx_bytes\":" << tc::mx::get_peak_memory() << ",\"seconds\":[";
        for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
        std::cout << "]}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
