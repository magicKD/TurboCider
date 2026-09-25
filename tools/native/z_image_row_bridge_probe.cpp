// Single-block, warm, real-capture FFN fork/join with the native Core ML bridge.
// The fixed row and channel artifacts are experimental; this does not alter
// Z-Image model routing or qualify all-layer image quality.
#include "../../native/backends/coreml_partitions.hpp"
#include "z_image_gpu_benchmark_lock.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mx = tc::mx;
using Tensor = tc::Tensor;
using Clock = std::chrono::steady_clock;

static double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return (samples[(samples.size()-1)/2] + samples[samples.size()/2]) * .5;
}

static Tensor ff(const Tensor &x, const Tensor &gate_weight,
                 const Tensor &up_weight, const Tensor &down_weight) {
    auto gate = mx::matmul(x, mx::transpose(gate_weight));
    auto up = mx::matmul(x, mx::transpose(up_weight));
    return mx::matmul((mx::sigmoid(gate) * gate) * up, mx::transpose(down_weight));
}

static double l2(const Tensor &a, const Tensor &b) {
    auto diff = mx::astype(a, mx::float32) - mx::astype(b, mx::float32);
    auto target = mx::astype(a, mx::float32);
    auto result = mx::sqrt(mx::sum(diff * diff) / mx::sum(target * target));
    return result.item<float>();
}

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5,
                    "usage: z-image-row-bridge-probe BUNDLE.safetensors ROW.mlmodelc CHANNEL.mlmodelc ROUNDS");
        const int rounds = std::stoi(argv[4]);
        tc::require(rounds >= 2 && rounds <= 64, "expected 2..64 symmetric rounds");
        ZImageGpuBenchmarkLock benchmark_lock;
        mx::set_default_device(mx::Device::gpu);
        auto arrays = mx::load_safetensors(argv[1]).first;
        auto input = arrays.at("input"), gate = arrays.at("gate");
        auto up = arrays.at("up"), down = arrays.at("down");
        tc::require(input.shape() == mx::Shape{1, 1536, 3840} &&
                    input.dtype() == mx::float16 &&
                    gate.shape() == mx::Shape{10240, 3840} &&
                    up.shape() == gate.shape() &&
                    down.shape() == mx::Shape{3840, 10240} &&
                    gate.dtype() == mx::bfloat16 && up.dtype() == mx::bfloat16 &&
                    down.dtype() == mx::bfloat16, "incorrect block-18 probe geometry");
        input = mx::astype(input, mx::bfloat16);
        mx::eval({input, gate, up, down});
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::CoreMLPartitions row({std::filesystem::absolute(argv[2])}, 544, 3840,
                                 event, cancelled);
        tc::CoreMLPartitions channel({std::filesystem::absolute(argv[3])}, 1024, 3840,
                                     event, cancelled);

        auto reference = [&]() { return ff(input, gate, up, down); };
        auto split = [&](bool rows) {
            const int ane_rows = rows ? 544 : 1024;
            // Account for the real GPU BF16 -> contiguous FP16 Core ML input
            // pack. The producer is already ready, as at the FFN fork.
            auto packed = mx::contiguous(mx::astype(
                mx::slice(input, {0, 0, 0}, {1, ane_rows, 3840}), mx::float16));
            mx::eval(packed);
            Tensor gpu = [&]() {
                if (rows)
                    return ff(mx::slice(input, {0, ane_rows, 0}, {1, 1536, 3840}),
                              gate, up, down);
                auto image = mx::slice(input, {0, 0, 0}, {1, 1024, 3840});
                auto caption = mx::slice(input, {0, 1024, 0}, {1, 1536, 3840});
                auto suffix_gate = mx::slice(gate, {5120, 0}, {10240, 3840});
                auto suffix_up = mx::slice(up, {5120, 0}, {10240, 3840});
                auto suffix_down = mx::slice(down, {0, 5120}, {3840, 10240});
                return mx::concatenate({ff(image, suffix_gate, suffix_up, suffix_down),
                                        ff(caption, gate, up, down)}, 1);
            }();
            mx::async_eval(gpu);
            auto ane = rows ? row.predict(0, packed) : channel.predict(0, packed);
            auto partial = mx::astype(ane, mx::bfloat16) * Tensor(32.f, mx::bfloat16);
            if (rows)
                return mx::concatenate({partial, gpu}, 1);
            auto image_gpu = mx::slice(gpu, {0, 0, 0}, {1, 1024, 3840});
            auto caption_gpu = mx::slice(gpu, {0, 1024, 0}, {1, 1536, 3840});
            return mx::concatenate({image_gpu + partial, caption_gpu}, 1);
        };
        auto timed = [](auto &&compute) {
            const auto begin = Clock::now();
            auto out = compute();
            mx::eval(out);
            return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        };
        for (int i = 0; i < 3; ++i) {
            timed(reference); timed([&] { return split(true); });
            timed([&] { return split(false); });
        }
        auto full = reference(); mx::eval(full);
        auto row_output = split(true); mx::eval(row_output);
        auto channel_output = split(false); mx::eval(channel_output);
        const auto row_image_error = l2(mx::slice(full, {0, 0, 0}, {1, 1024, 3840}),
            mx::slice(row_output, {0, 0, 0}, {1, 1024, 3840}));
        const auto channel_image_error = l2(mx::slice(full, {0, 0, 0}, {1, 1024, 3840}),
            mx::slice(channel_output, {0, 0, 0}, {1, 1024, 3840}));
        const auto row_caption_error = l2(mx::slice(full, {0, 1024, 0}, {1, 1536, 3840}),
            mx::slice(row_output, {0, 1024, 0}, {1, 1536, 3840}));
        const auto channel_caption_error = l2(mx::slice(full, {0, 1024, 0}, {1, 1536, 3840}),
            mx::slice(channel_output, {0, 1024, 0}, {1, 1536, 3840}));
        tc::require(std::isfinite(row_image_error) && std::isfinite(channel_image_error) &&
                    row_caption_error == 0 && channel_caption_error == 0,
                    "nonfinite or changed BF16 GPU caption result");
        std::vector<double> full_ms, row_ms, channel_ms;
        for (int i = 0; i < rounds; ++i) {
            if (i % 2 == 0) {
                full_ms.push_back(timed(reference));
                row_ms.push_back(timed([&] { return split(true); }));
                channel_ms.push_back(timed([&] { return split(false); }));
                channel_ms.push_back(timed([&] { return split(false); }));
                row_ms.push_back(timed([&] { return split(true); }));
                full_ms.push_back(timed(reference));
            } else {
                full_ms.push_back(timed(reference));
                channel_ms.push_back(timed([&] { return split(false); }));
                row_ms.push_back(timed([&] { return split(true); }));
                row_ms.push_back(timed([&] { return split(true); }));
                channel_ms.push_back(timed([&] { return split(false); }));
                full_ms.push_back(timed(reference));
            }
        }
        std::cout << "{\"scope\":\"single-block captured BF16 FFN, native GPU pack/Core ML backing/MLX join; no attention or full image\""
                  << ",\"rounds\":" << rounds << ",\"gpu_full_ms\":" << median(full_ms)
                  << ",\"row_ms\":" << median(row_ms)
                  << ",\"channel_ms\":" << median(channel_ms)
                  << ",\"row_image_l2\":" << row_image_error
                  << ",\"channel_image_l2\":" << channel_image_error
                  << ",\"row_caption_l2\":" << row_caption_error
                  << ",\"channel_caption_l2\":" << channel_caption_error
                  << ",\"row_calls\":" << row.calls()
                  << ",\"channel_calls\":" << channel.calls()
                  << ",\"row_copied_bytes\":" << row.copied_bytes()
                  << ",\"channel_copied_bytes\":" << channel.copied_bytes() << "}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
