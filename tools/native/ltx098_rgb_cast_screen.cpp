// Probe-only screen for fusing the existing BF16-pixels -> RGB8 conversion.
// It checks every finite BF16 bit pattern and a captured full-video tensor.
#include "ltx098_rgb_cast_candidate.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 3, "usage: ltx098-rgb-cast-screen PIXELS.safetensors REPEATS");
        const int repeats = std::stoi(argv[2]);
        tc::require(repeats >= 2 && repeats <= 20, "invalid repeat count");
        tc::configure_streams();
        std::vector<uint16_t> bits;
        for (uint32_t value = 0; value <= 0xffff; ++value)
            if ((value & 0x7f80) != 0x7f80) bits.push_back(uint16_t(value));
        auto all_values = tc::mx::view(tc::Tensor(bits.data(), {int(bits.size())}, tc::mx::uint16), tc::mx::bfloat16);
        auto pixels = tc::mx::load_safetensors(argv[1]).first.at("value");
        tc::require(pixels.ndim() == 5 && pixels.shape(0) == 1 && pixels.shape(4) == 3 &&
                    pixels.dtype() == tc::mx::bfloat16 && tc::mx::all(tc::mx::isfinite(pixels)).item<bool>(),
                    "expected finite BF16 [1,frames,height,width,3] pixels");
        tc::mx::eval(all_values, pixels);
        for (const auto &input : {all_values, pixels}) {
            auto old = tc::ltx098::rgb_cast_probe::reference(input), fused = tc::ltx098::rgb_cast_probe::fused(input);
            auto mismatch = tc::mx::sum(tc::mx::astype(old != fused, tc::mx::int32));
            tc::mx::eval(mismatch);
            tc::require(mismatch.item<int>() == 0, "fused RGB conversion changed output bytes: " +
                        std::to_string(mismatch.item<int>()));
        }
        std::vector<std::vector<double>> samples(4);
        for (int group = 0; group < 4; ++group) {
            const bool fused = group == 1 || group == 2;
            for (int iteration = 0; iteration < repeats + 1; ++iteration) {
                const auto start = std::chrono::steady_clock::now();
                auto result = fused ? tc::ltx098::rgb_cast_probe::fused(pixels) : tc::ltx098::rgb_cast_probe::reference(pixels);
                tc::mx::eval(result);
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                if (iteration) samples[group].push_back(elapsed);
            }
        }
        std::cout << std::setprecision(12) << "{\"scope\":\"GPU RGB conversion component only\",\"finite_bf16_values\":"
                  << bits.size() << ",\"exact\":true,\"frames\":" << pixels.shape(1)
                  << ",\"height\":" << pixels.shape(2) << ",\"width\":" << pixels.shape(3) << ",\"runs\":[";
        for (int group = 0; group < 4; ++group) {
            std::cout << (group ? "," : "") << "{\"fused\":" << (group == 1 || group == 2 ? "true" : "false") << ",\"seconds\":[";
            for (size_t i = 0; i < samples[group].size(); ++i) std::cout << (i ? "," : "") << samples[group][i];
            std::cout << "]}";
        }
        std::cout << "]}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
