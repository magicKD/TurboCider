#include "../../native/models/wan22/latent.hpp"
#include "../../native/models/wan22/tae_super.hpp"
#include "../../native/models/wan22/vae.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 7, "usage: wan22-vae-benchmark-probe full|super WEIGHTS RAW_INPUT OUTPUT WARMUPS RUNS");
        const std::string mode(argv[1]);
        tc::require(mode == "full" || mode == "super", "invalid decoder mode");
        const int warmups = std::stoi(argv[5]), runs = std::stoi(argv[6]);
        tc::require(warmups >= 0 && warmups <= 3 && runs >= 1 && runs <= 10 &&
                        !std::filesystem::exists(argv[4]), "invalid benchmark repetition or output");
        auto raw = tc::mx::load_safetensors(argv[3]).first.at("latents");
        tc::mx::eval(raw);
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        const auto load_started = tc::Clock::now();
        std::unique_ptr<tc::wan22::VAEDecoder> full;
        std::unique_ptr<tc::wan22::TAESuperDecoder> super;
        if (mode == "full") {
            full = std::make_unique<tc::wan22::VAEDecoder>();
            full->load(std::filesystem::absolute(argv[2]));
        } else super = std::make_unique<tc::wan22::TAESuperDecoder>(std::filesystem::absolute(argv[2]));
        const double load_seconds = std::chrono::duration<double>(tc::Clock::now() - load_started).count();
        auto normalized = tc::wan22::normalize_vae_latents(raw);
        tc::mx::eval(normalized);
        std::optional<tc::Tensor> output;
        std::vector<double> times;
        std::vector<size_t> peaks;
        for (int index = -warmups; index < runs; ++index) {
            tc::mx::reset_peak_memory();
            const auto started = tc::Clock::now();
            auto value = mode == "full" ? full->decode(raw, event, &cancelled)
                                         : super->decode(normalized, event, &cancelled);
            tc::mx::eval(value);
            const double elapsed = std::chrono::duration<double>(tc::Clock::now() - started).count();
            if (index >= 0) {
                times.push_back(elapsed);
                peaks.push_back(tc::mx::get_peak_memory());
            }
            output = std::move(value);
        }
        tc::require(output && tc::mx::all(tc::mx::isfinite(*output)).item<bool>(), "decoder produced nonfinite pixels");
        // Full VAE emits [-1,1]-style raw pixels; Super emits [0,1]. Store
        // both in the full decoder's range for an explicit visual comparison.
        auto pixels = mode == "super" ? tc::mx::astype(*output, tc::mx::float32) * 2.f - 1.f : *output;
        tc::mx::eval(pixels);
        tc::mx::save_safetensors(argv[4], {{"output", pixels}});
        std::cout << "{\"mode\":\"" << mode << "\",\"load_seconds\":" << load_seconds << ",\"decode_seconds\":[";
        for (size_t i = 0; i < times.size(); ++i) std::cout << (i ? "," : "") << times[i];
        std::cout << "],\"mlx_peak_bytes\":[";
        for (size_t i = 0; i < peaks.size(); ++i) std::cout << (i ? "," : "") << peaks[i];
        std::cout << "]}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
