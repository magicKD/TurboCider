// Read-only runtime diagnostic: no model or production-path changes.
// Run serially against different MLX dylibs to separate runtime from model code.
#include <mlx/mlx.h>
#include "z_image_gpu_benchmark_lock.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace mx = mlx::core;

template <class F>
void measure(const char* name, const char* dtype, F fn) {
    std::vector<double> times;
    for (int i = 0; i < 5; ++i) {
        const auto start = std::chrono::steady_clock::now();
        auto y = fn();
        mx::eval(y);
        mx::synchronize();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (i >= 2) times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    std::cout << "{\"operation\":\"" << name << "\",\"dtype\":\"" << dtype
              << "\",\"median_ms\":" << times[1] << "}" << std::endl;
}

int main() {
    ZImageGpuBenchmarkLock lock;
    mx::set_default_device(mx::Device::gpu);
    mx::random::seed(42);
    for (const auto dtype : {mx::float32, mx::float16, mx::bfloat16}) {
        const char* label = dtype == mx::float32 ? "fp32" :
                            dtype == mx::float16 ? "fp16" : "bf16";
        auto x = mx::random::normal({1,1056,3840}, dtype);
        auto w = mx::random::normal({11520,3840}, dtype);
        auto image = mx::random::normal({1,256,256,128}, dtype);
        auto conv = mx::random::normal({128,3,3,128}, dtype);
        mx::eval(x,w,image,conv);
        measure("qkv_1056x3840x11520", label,
            [&] { return mx::matmul(x,mx::transpose(w)); });
        measure("conv_256x256_128x128_k3", label,
            [&] { return mx::conv2d(image,conv,{1,1},{1,1}); });
        measure("add_256x256x128", label,
            [&] { return image + mx::array(1.0f,dtype); });
    }
}
