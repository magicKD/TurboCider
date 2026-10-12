// Bounded GPU-only screen: one 512px/r32 FFN head, two implementations,
// one warmup and three samples each. No checkpoint, ANE or full-image claim.
#include "../../native/models/z_image/metal/projection.hpp"
#include <mlx/random.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
using Graph = std::function<std::vector<mx::array>(const std::vector<mx::array> &)>;

namespace {
constexpr int rows = 1056, hidden = 3840, width = 10240, channels = 6144, rank = 32;

mx::array fixture(const mx::Shape &shape, unsigned seed, float scale) {
    return mx::astype(mx::random::normal(shape, mx::float32, mx::random::key(seed)) * scale,
                      mx::bfloat16);
}

float relative_l2(const mx::array &actual, const mx::array &expected) {
    auto a = mx::astype(actual, mx::float32), b = mx::astype(expected, mx::float32);
    return mx::sqrt(mx::sum(mx::square(a-b)) /
        mx::maximum(mx::sum(mx::square(b)), mx::array(1e-20f))).item<float>();
}
}

int main(int argc, char **) {
    if (argc != 1) return 2;
    try {
        mx::set_default_device(mx::Device(mx::Device::gpu));
        std::vector<mx::array> args{
            fixture({1,rows,hidden}, 1, .1f),
            fixture({width,hidden}, 2, .01f),
            fixture({width,hidden}, 3, .01f),
            fixture({hidden,width}, 4, .01f),
            fixture({rank,hidden}, 5, .02f),
            fixture({rank,hidden}, 6, .02f),
            fixture({width,rank}, 7, .02f),
            fixture({width,rank}, 8, .02f)};
        mx::eval(args);
        // The real channel route has completed x @ A at correction readiness.
        // Prepare those same rank-sized dependencies before either timed head;
        // B projections, ordered FP32 adds/BF16 casts and down are timed.
        auto input = mx::astype(args[0], mx::float32);
        args[4] = mx::matmul(input, mx::transpose(mx::astype(args[4], mx::float32)));
        args[5] = mx::matmul(input, mx::transpose(mx::astype(args[5], mx::float32)));
        mx::eval(args[4], args[5]);
        std::array<Graph,2> graphs;
        for (int mode=0; mode<2; ++mode) {
            graphs[mode] = mx::compile([mode](const std::vector<mx::array> &a) {
                auto base = [&](const mx::array &x, const mx::array &weight, bool down) {
                    if (mode)
                        return tc::z_metal::projection_range(x, weight, 0, down ? hidden : channels,
                                                             0, down ? channels : hidden);
                    auto selected = down ? mx::slice(weight, {0,0}, {hidden,channels}) :
                                           mx::slice(weight, {0,0}, {channels,hidden});
                    return mx::matmul(x, mx::transpose(selected));
                };
                auto corrected = [&](const mx::array &checkpoint, const mx::array &low,
                                     const mx::array &up) {
                    const auto selected = mx::astype(mx::slice(up, {0,0}, {channels,rank}), mx::float32);
                    const auto delta = mx::matmul(low, mx::transpose(selected));
                    return mx::astype(mx::astype(checkpoint, mx::float32) + delta, mx::bfloat16);
                };
                auto gate = corrected(base(a[0], a[1], false), a[4], a[6]);
                auto up = corrected(base(a[0], a[2], false), a[5], a[7]);
                auto activation = (gate * mx::sigmoid(gate)) * up;
                return std::vector<mx::array>{base(activation, a[3], true), activation,
                                              mx::array(mode, mx::int32)};
            });
        }
        std::array<std::vector<mx::array>,2> warmup;
        for (int mode=0; mode<2; ++mode) {
            warmup[mode] = graphs[mode](args);
            mx::eval(warmup[mode]);
            if (warmup[mode][2].item<int>() != mode)
                throw std::runtime_error("compiled component mode capture/cache alias");
            if (!mx::all(mx::isfinite(warmup[mode][0])).item<bool>())
                throw std::runtime_error("nonfinite FFN component output");
        }
        const float down_l2 = relative_l2(warmup[1][0], warmup[0][0]);
        const float activation_l2 = relative_l2(warmup[1][1], warmup[0][1]);
        if (!std::isfinite(down_l2) || !std::isfinite(activation_l2) ||
            down_l2 > .02f || activation_l2 > .02f)
            throw std::runtime_error("MPP LoRA component differs beyond the screening tolerance");
        std::array<std::vector<double>,2> samples;
        for (int sample=0; sample<3; ++sample) for (int order=0; order<2; ++order) {
            const int mode = (sample + order) % 2;
            const auto start = std::chrono::steady_clock::now();
            mx::eval(graphs[mode](args));
            samples[mode].push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
        }
        for (int mode=0; mode<2; ++mode) {
            auto sorted = samples[mode]; std::sort(sorted.begin(), sorted.end());
            std::cout << "{\"scope\":\"synthetic r32 GPU FFN channel head; prepared A projections; no ANE or model speedup\","
                      << "\"mode\":\"" << (mode ? "mpp" : "native") << "\",\"rows\":" << rows
                      << ",\"gpu_channels\":" << channels << ",\"rank\":" << rank
                      << ",\"warmup\":1,\"median_seconds\":" << sorted[1]
                      << ",\"down_relative_l2\":" << down_l2
                      << ",\"hidden_relative_l2\":" << activation_l2 << ",\"samples\":[";
            for (size_t i=0; i<samples[mode].size(); ++i)
                std::cout << (i ? "," : "") << samples[mode][i];
            std::cout << "]}\n";
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
