// Compare automatic SDPA with an explicit attention graph. No production flags.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include "z_image_gpu_benchmark_lock.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace mx = mlx::core;
int main() {
    try {
        ZImageGpuBenchmarkLock lock;
        mx::set_default_device(mx::Device::gpu);
        mx::random::seed(42);
        auto q = mx::random::normal({1,30,1056,128},mx::bfloat16);
        auto k = mx::random::normal(q.shape(),mx::bfloat16);
        auto v = mx::random::normal(q.shape(),mx::bfloat16);
        mx::eval(q,k,v);
        auto automatic = [&] {return mx::fast::scaled_dot_product_attention(q,k,v,1.f/std::sqrt(128.f));};
        auto reference = [&] {
            auto scores = mx::matmul(mx::astype(q,mx::float32),
                mx::swapaxes(mx::astype(k,mx::float32),-1,-2)) / std::sqrt(128.f);
            return mx::astype(mx::matmul(mx::softmax(scores,-1),
                mx::astype(v,mx::float32)),mx::bfloat16);
        };
        auto a = automatic(), b = reference(); mx::eval(a,b);
        std::cout << "max_abs=" << mx::max(mx::abs(mx::astype(a,mx::float32)-mx::astype(b,mx::float32))).item<float>() << '\n';
        std::vector<double> automatic_ms, reference_ms;
        auto timed = [](auto fn) {
            auto start=std::chrono::steady_clock::now(); auto y=fn(); mx::eval(y);
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        };
        for (int i=0;i<13;++i) {
            double at,bt;
            if (i%2) {at=timed(automatic);bt=timed(reference);}
            else {bt=timed(reference);at=timed(automatic);}
            if (i>=3) {automatic_ms.push_back(at);reference_ms.push_back(bt);}
        }
        std::sort(automatic_ms.begin(),automatic_ms.end());
        std::sort(reference_ms.begin(),reference_ms.end());
        std::cout << "automatic_sdpa_ms=" << (automatic_ms[4]+automatic_ms[5])/2
                  << " explicit_fp32_reference_ms=" << (reference_ms[4]+reference_ms[5])/2 << '\n';
    } catch (const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
