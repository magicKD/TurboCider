// Small native correctness/alternating benchmark for the production helper.
// No checkpoint download, persistent dense cache, or ANE performance claim.
#include "models/z_image/ffn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace tc;

namespace {
Weights fixture(int hidden, int width, bool bias, bool dense_up) {
    std::vector<std::string> keys;
    std::vector<Tensor> arrays;
    auto add = [&](const std::string &key, Tensor value) {
        keys.push_back(key);
        arrays.push_back(std::move(value));
    };
    for (int projection = 0; projection < 3; ++projection) {
        const std::string name = projection == 0 ? "ffn.w1" : projection == 1 ? "ffn.w3" : "ffn.w2";
        const int rows = projection == 2 ? hidden : width;
        const int cols = projection == 2 ? width : hidden;
        // Include the entire signed range, with small nonuniform row scales.
        auto index = mx::arange(rows * cols, mx::int32);
        auto codes = mx::reshape(mx::astype(mx::remainder(index * 13 + projection * 7,
            Tensor(256, mx::int32)) - Tensor(128, mx::int32), mx::int8), {rows, cols});
        auto scale = mx::reshape((mx::remainder(mx::arange(rows, mx::float32), Tensor(5.f)) +
            Tensor(1.f)) * Tensor(0.00002f), {rows, 1});
        if (dense_up && projection == 1) {
            add(name + ".weight", mx::astype(mx::astype(codes, mx::float32) * scale, mx::bfloat16));
        } else {
            add(name + ".weight", codes);
            add(name + ".weight_scale", scale);
            add(name + ".comfy_quant", Tensor(1, mx::uint8));
        }
        if (bias) add(name + ".bias", mx::full({rows}, 0.001f, mx::float32));
    }
    mx::eval(arrays);
    Weights weights;
    weights.bind_arrays(keys, arrays);
    return weights;
}

Tensor input(int rows, int hidden, mx::Dtype dtype, bool batched) {
    auto value = mx::sin(mx::arange(rows * hidden, mx::float32) * Tensor(0.017f));
    return mx::astype(mx::reshape(value, batched ? mx::Shape{1, rows, hidden}
                                                : mx::Shape{rows, hidden}), dtype);
}

void parity() {
    int cases = 0;
    for (bool metal : {false, true}) {
        for (int group : {0, 32, 64, 128}) {
            for (auto dtype : {mx::bfloat16, mx::float16, mx::float32}) {
                for (int rows : {1, 33}) {
                    for (bool mixed : {false, true}) {
                        auto weights = fixture(256, 512, true, mixed);
                        weights.set_metal_convrot(metal);
                        if (group) weights.pack_convrot_q8(group, mx::bfloat16);
                        auto x = input(rows, 256, dtype, rows == 33);
                        auto control = z_image::feed_forward(x, weights, "ffn", false);
                        auto candidate = z_image::feed_forward(x, weights, "ffn", true);
                        mx::eval({control, candidate});
                        require(control.dtype() == candidate.dtype() && control.shape() == candidate.shape(),
                                "ConvRot FFN dtype/shape changed");
                        require(mx::all(mx::isfinite(control)).item<bool>() &&
                                mx::all(mx::isfinite(candidate)).item<bool>(), "nonfinite ConvRot FFN");
                        require(mx::all(control == candidate).item<bool>(), "shared ConvRot FFN is not exact");
                        ++cases;
                    }
                }
            }
        }
    }
    std::cout << "{\"kind\":\"parity\",\"cases\":" << cases
              << ",\"passed\":true,\"exact\":true}\n";
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) * .5;
}

void benchmark(int rows, int hidden, int width, int iterations) {
    require(rows > 0 && rows <= 8192 && hidden > 0 && hidden <= 10240 &&
            width > 0 && width <= 16384 && hidden % 256 == 0 && width % 256 == 0 &&
            iterations >= 4 && iterations <= 100, "invalid benchmark geometry/iterations");
    auto weights = fixture(hidden, width, false, false);
    weights.pack_convrot_q8(32, mx::bfloat16);
    auto x = input(rows, hidden, mx::bfloat16, true);
    mx::eval(x);
    auto run = [&](bool shared) {
        const auto start = std::chrono::steady_clock::now();
        auto y = z_image::feed_forward(x, weights, "ffn", shared);
        mx::eval(y);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    for (int i = 0; i < 3; ++i) { run(false); run(true); }
    std::vector<double> control, candidate;
    for (int i = 0; i < iterations; ++i) {
        if (i % 2) { candidate.push_back(run(true)); control.push_back(run(false)); }
        else { control.push_back(run(false)); candidate.push_back(run(true)); }
    }
    auto a = z_image::feed_forward(x, weights, "ffn", false);
    auto b = z_image::feed_forward(x, weights, "ffn", true);
    require(mx::all(mx::isfinite(a)).item<bool>() && mx::all(a == b).item<bool>(),
            "benchmark outputs differ or are nonfinite");
    auto emit = [](const std::vector<double> &samples) {
        std::cout << '[';
        for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
        std::cout << ']';
    };
    std::cout << "{\"kind\":\"synthetic_ffn\",\"rows\":" << rows << ",\"hidden\":" << hidden
              << ",\"width\":" << width << ",\"warmups_per_arm\":3,\"samples_per_arm\":" << iterations
              << ",\"control_median_ms\":" << median(control)
              << ",\"shared_median_ms\":" << median(candidate)
              << ",\"speedup\":" << median(control) / median(candidate)
              << ",\"exact\":true,\"control_ms\":";
    emit(control);
    std::cout << ",\"shared_ms\":";
    emit(candidate);
    std::cout << "}\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 1) parity();
        else {
            require(argc == 5, "usage: convrot-ffn-probe [rows hidden width iterations]");
            benchmark(std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]));
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
