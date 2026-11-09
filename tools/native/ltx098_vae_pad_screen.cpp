// Component-only, exactness-first screen for an LTX 0.9.8 decoder pad kernel.
#include "../../native/models/ltx098/vae_padding.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 7, "usage: ltx098-vae-pad-screen FRAMES HEIGHT WIDTH CHANNELS REPEATS contiguous|transpose|slice|flip|bits");
        const int f = std::stoi(argv[1]), h = std::stoi(argv[2]), w = std::stoi(argv[3]);
        const int c = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
        const std::string layout(argv[6]);
        tc::require(f >= 1 && f <= 121 && h >= 2 && h <= 128 && w >= 2 && w <= 128 &&
                    c >= 1 && c <= 2048 && repeats >= 2 && repeats <= 20 &&
                    (layout == "contiguous" || layout == "transpose" || layout == "slice" || layout == "flip" || layout == "bits"),
                    "invalid screen dimensions/layout");
        tc::mx::set_default_device(tc::mx::Device(tc::mx::Device::gpu));
        auto input = tc::mx::random::normal({1, f, layout == "transpose" ? w : h + (layout == "slice" ? 2 : 0),
                                              layout == "transpose" ? h : w + (layout == "slice" ? 2 : 0), c},
                                             tc::mx::bfloat16, tc::mx::random::key(901));
        std::vector<uint16_t> bit_fixture;
        if (layout == "bits") {
            const uint16_t patterns[] = {0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80,
                                         0x7fc1, 0xffc1, 0x0001, 0x8001, 0x7f7f};
            bit_fixture.resize(size_t(f) * h * w * c);
            for (size_t i = 0; i < bit_fixture.size(); ++i) bit_fixture[i] = patterns[i % std::size(patterns)];
            input = tc::mx::view(tc::Tensor(bit_fixture.data(), {1, f, h, w, c}, tc::mx::uint16), tc::mx::bfloat16);
        }
        if (layout == "transpose") input = tc::mx::transpose(input, {0, 1, 3, 2, 4});
        if (layout == "slice") input = tc::mx::slice(input, {0, 0, 1, 1, 0}, {1, f, h + 1, w + 1, c});
        if (layout == "flip") input = tc::mx::flip(input, 3);
        tc::mx::eval(input);
        bool negative_stride = false;
        for (auto stride : input.strides()) negative_stride = negative_stride || stride < 0;
        {
            auto expected = tc::ltx098::vae_padding::reference(input);
            auto actual = tc::ltx098::vae_padding::apply(input);
            tc::require(expected.shape() == actual.shape() && expected.dtype() == actual.dtype() &&
                        tc::mx::array_equal(tc::mx::view(expected, tc::mx::uint8),
                                            tc::mx::view(actual, tc::mx::uint8)).item<bool>(),
                        "pad candidate is not bit-exact");
        }
        std::vector<std::vector<double>> samples(4);
        for (int group = 0; group < 4; ++group) {
            const bool candidate = group == 1 || group == 2;
            for (int iteration = 0; iteration < repeats + 1; ++iteration) {
                const auto start = std::chrono::steady_clock::now();
                auto result = candidate ? tc::ltx098::vae_padding::apply(input) : tc::ltx098::vae_padding::reference(input);
                tc::mx::eval(result);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                if (iteration) samples[group].push_back(seconds);
            }
        }
        std::cout << std::setprecision(12) << "{\"scope\":\"GPU pad component only\",\"frames\":" << f
                  << ",\"height\":" << h << ",\"width\":" << w << ",\"channels\":" << c
                  << ",\"layout\":\"" << layout << "\",\"exact\":true,\"fused\":"
                  << (tc::ltx098::vae_padding::eligible(input) ? "true" : "false") << ",\"negative_stride\":"
                  << (negative_stride ? "true" : "false") << ",\"runs\":[";
        for (int group = 0; group < 4; ++group) {
            std::cout << (group ? "," : "") << "{\"candidate\":" << (group == 1 || group == 2 ? "true" : "false")
                      << ",\"seconds\":[";
            for (size_t i = 0; i < samples[group].size(); ++i)
                std::cout << (i ? "," : "") << samples[group][i];
            std::cout << "]}";
        }
        std::cout << "]}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
