// Diagnostic-only GPU screen of current vs compiled BF16 interleaved RoPE.
#include "../../native/models/ltx098/activation.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace mx = mlx::core;
using Clock = std::chrono::steady_clock;

static tc::Tensor current(const tc::Tensor &value, const tc::Tensor &cosine, const tc::Tensor &sine) {
    auto pairs = mx::reshape(value, {value.shape(0), value.shape(1), value.shape(2), value.shape(3) / 2, 2});
    auto parts = mx::split(pairs, 2, -1);
    auto even = mx::squeeze(parts[0], -1), odd = mx::squeeze(parts[1], -1);
    auto first_even = even * cosine, first_odd = even * sine;
    auto rotated = mx::stack({tc::ltx098::addcmul(first_even, -sine, odd),
                              tc::ltx098::addcmul(first_odd, cosine, odd)}, -1);
    return mx::reshape(rotated, value.shape());
}

static tc::Tensor candidate(const tc::Tensor &value, const tc::Tensor &cosine, const tc::Tensor &sine) {
    static auto compiled = mx::compile([](const std::vector<tc::Tensor> &args) {
        const auto &x = args[0], &c = args[1], &s = args[2];
        auto pairs = mx::reshape(x, {x.shape(0), x.shape(1), x.shape(2), x.shape(3) / 2, 2});
        auto parts = mx::split(pairs, 2, -1);
        auto even = mx::squeeze(parts[0], -1), odd = mx::squeeze(parts[1], -1);
        auto ef = mx::astype(even, mx::float32), of = mx::astype(odd, mx::float32);
        auto cf = mx::astype(c, mx::float32), sf = mx::astype(s, mx::float32);
        auto base_even = mx::astype(ef * cf, x.dtype()), base_odd = mx::astype(ef * sf, x.dtype());
        auto out_even = mx::astype(mx::astype(base_even, mx::float32) - of * sf, x.dtype());
        auto out_odd = mx::astype(mx::astype(base_odd, mx::float32) + of * cf, x.dtype());
        return std::vector<tc::Tensor>{mx::reshape(mx::stack({out_even, out_odd}, -1), x.shape())};
    }, true);
    return compiled({value, cosine, sine})[0];
}

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4 && !std::filesystem::exists(argv[1]), "usage: ltx098-rope-screen NEW_OUTPUT TOKENS REPEATS");
        const int tokens = std::stoi(argv[2]), repeats = std::stoi(argv[3]);
        tc::require(tokens > 0 && tokens <= 4096 && repeats >= 2 && repeats <= 50, "invalid screen geometry");
        auto q = mx::astype(mx::random::normal({1, 32, tokens, 64}, mx::float32, mx::random::key(81)), mx::bfloat16);
        auto k = mx::astype(mx::random::normal({1, 32, tokens, 64}, mx::float32, mx::random::key(82)), mx::bfloat16);
        auto angle = mx::random::normal({1, 32, tokens, 32}, mx::float32, mx::random::key(83));
        auto cosine = mx::astype(mx::cos(angle), mx::bfloat16), sine = mx::astype(mx::sin(angle), mx::bfloat16);
        mx::eval(q, k, cosine, sine);
        std::vector<std::vector<double>> samples(4);
        std::vector<tc::Tensor> reference, result;
        for (int group = 0; group < 4; ++group) {
            const bool fused = group == 1 || group == 2;
            for (int iteration = 0; iteration < repeats + 2; ++iteration) {
                const auto start = Clock::now();
                std::vector<tc::Tensor> values = fused ? std::vector<tc::Tensor>{candidate(q, cosine, sine), candidate(k, cosine, sine)}
                                                       : std::vector<tc::Tensor>{current(q, cosine, sine), current(k, cosine, sine)};
                mx::eval(values);
                const auto elapsed = std::chrono::duration<double>(Clock::now() - start).count();
                if (iteration >= 2) samples[group].push_back(elapsed);
                if (group == 0) reference = values;
                if (group == 2) result = values;
            }
        }
        mx::save_safetensors(argv[1], {{"reference_q", reference[0]}, {"reference_k", reference[1]},
                                      {"candidate_q", result[0]}, {"candidate_k", result[1]}});
        std::cout << std::setprecision(12) << "{\"tokens\":" << tokens << ",\"runs\":[";
        for (int group = 0; group < 4; ++group) {
            std::cout << (group ? "," : "") << "{\"compiled\":" << ((group == 1 || group == 2) ? "true" : "false") << ",\"seconds\":[";
            for (size_t i = 0; i < samples[group].size(); ++i) std::cout << (i ? "," : "") << samples[group][i];
            std::cout << "]}";
        }
        std::cout << "]}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
