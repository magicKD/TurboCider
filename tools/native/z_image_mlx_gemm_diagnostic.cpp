// Read-only, ABI-local MLX GEMM diagnostic. Compile this source separately
// against each retained MLX include/lib pair; do not mix headers and dylibs.
#include "z_image_gpu_benchmark_lock.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;

int main(int argc, char **argv) {
    try {
        ZImageGpuBenchmarkLock benchmark_lock;
        if (argc != 5)
            throw std::invalid_argument("usage: probe ROWS OUTPUT INPUT RUNS");
        const int rows = std::stoi(argv[1]);
        const int output = std::stoi(argv[2]);
        const int input = std::stoi(argv[3]);
        const int runs = std::stoi(argv[4]);
        if (rows < 1 || output < 1 || input < 1 || runs < 1)
            throw std::invalid_argument("all arguments must be positive");

        mx::set_default_device(mx::Device::gpu);
        mx::random::seed(42);
        auto x = mx::random::normal({rows, input}, mx::bfloat16);
        auto weight = mx::random::normal({output, input}, mx::bfloat16);
        mx::eval(x, weight);

        std::vector<double> samples;
        for (int i = 0; i < runs + 3; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto value = mx::matmul(x, mx::transpose(weight));
            mx::eval(value);
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (i >= 3) samples.push_back(milliseconds);
        }
        std::sort(samples.begin(), samples.end());
        const double median = (samples[(runs - 1) / 2] + samples[runs / 2]) / 2.0;
        std::cout << "{\"rows\":" << rows << ",\"output\":" << output
                  << ",\"input\":" << input << ",\"dtype\":\"bf16\","
                  << "\"median_ms\":" << median << ",\"effective_tflops\":"
                  << (2.0 * rows * output * input / (median * 1.0e9)) << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
