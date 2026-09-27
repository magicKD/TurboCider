// Shape-matched Qwen21 GPU QKV screening; not an end-to-end speedup claim.
#include "z_image_gpu_benchmark_lock.hpp"
#include "../../native/models/qwen21/metal/qk_norm_rope.hpp"
#include "../../native/models/qwen21/metal/qkv_projection.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
using Arrays = std::vector<mx::array>;

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
}

int main(int argc, char **argv) try {
    if (argc != 3 && argc != 4)
        throw std::invalid_argument("usage: qwen21-qkv-probe ROWS RUNS [--mpp]");
    const std::string_view option = argc == 4 ? argv[3] : "";
    const bool mpp = option == "--mpp" || option == "--mpp16" ||
                     option == "--mpp32" || option == "--mpp64";
    if (argc == 4 && !mpp) throw std::invalid_argument("unknown QKV option");
    const int bm = option == "--mpp16" ? 16 : option == "--mpp32" ? 32 :
                   option == "--mpp64" ? 64 : 48;
    const int rows = std::stoi(argv[1]), runs = std::stoi(argv[2]);
    if ((rows != 1024 && rows != 2091 && rows != 3157 && rows != 4226 &&
         rows != 4096 && rows != 13442) || runs < 3 || runs > 30)
        throw std::invalid_argument("unsupported Qwen21 QKV rows/runs");
    ZImageGpuBenchmarkLock lock;
    mx::set_default_device(mx::Device(mx::Device::gpu));
    mx::random::seed(42);
    auto x = mx::random::normal({1, rows, 4096}, mx::bfloat16);
    auto q = mx::random::normal({4096, 4096}, mx::bfloat16);
    auto k = mx::random::normal({4096, 4096}, mx::bfloat16);
    auto v = mx::random::normal({4096, 4096}, mx::bfloat16);
    mx::eval(x, q, k, v);
    auto w = mx::concatenate({q, k, v}, 0);
    mx::eval(w);
    if (mpp) {
        auto qw = mx::random::normal({128}, mx::bfloat16);
        auto kw = mx::random::normal({128}, mx::bfloat16);
        auto cos = mx::random::normal({rows, 64}, mx::float32);
        auto sin = mx::random::normal({rows, 64}, mx::float32);
        mx::eval(qw, kw, cos, sin);
        auto reference = mx::compile([](const Arrays &a) {
            auto qv = mx::matmul(a[0], mx::transpose(a[1]));
            auto kv = mx::matmul(a[0], mx::transpose(a[2]));
            auto vv = mx::matmul(a[0], mx::transpose(a[3]));
            auto pair = tc::qwen21::metal::prepare_qk(
                qv, kv, a[4], a[5], a[6], a[7], 1e-6f);
            return Arrays{pair[0], pair[1],
                          mx::transpose(mx::reshape(vv, {1, vv.shape(1), 32, 128}),
                                        {0, 2, 1, 3})};
        });
        auto candidate = mx::compile([bm](const Arrays &a) {
            return tc::qwen21::metal::project_prepare_qkv(
                a[0], a[1], a[2], a[3], a[4], a[5], bm);
        });
        const Arrays a{x, q, k, v, qw, kw, cos, sin};
        const Arrays b{x, w, qw, kw, cos, sin};
        auto expected = reference(a), actual = candidate(b);
        mx::eval(expected); mx::eval(actual);
        double error = 0;
        for (int i = 0; i < 3; ++i) {
            auto x32 = mx::astype(expected[i], mx::float32);
            auto y32 = mx::astype(actual[i], mx::float32);
            auto relative = mx::sqrt(mx::sum(mx::square(x32 - y32)) /
                                           mx::sum(mx::square(x32))).item<float>();
            if (!std::isfinite(relative)) throw std::runtime_error("nonfinite MPP QKV error");
            error = std::max(error, double(relative));
        }
        auto time = [](auto &graph, const Arrays &args) {
            const auto start = std::chrono::steady_clock::now();
            mx::eval(graph(args));
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        };
        std::vector<double> old_ms, mpp_ms;
        for (int i = 0; i < runs + 4; ++i) {
            double t0, t1;
            if (i % 2) { t1 = time(candidate, b); t0 = time(reference, a); }
            else { t0 = time(reference, a); t1 = time(candidate, b); }
            if (i >= 4) { old_ms.push_back(t0); mpp_ms.push_back(t1); }
        }
        std::cout << "{\"rows\":" << rows << ",\"bm\":" << bm
                  << ",\"qkv_qk_gpu_ms\":" << median(old_ms)
                  << ",\"mpp_fused_ms\":" << median(mpp_ms)
                  << ",\"relative_l2_max\":" << error << "}" << std::endl;
        return 0;
    }
    auto separate = mx::compile([](const Arrays &a) {
        return Arrays{mx::matmul(a[0], mx::transpose(a[1])),
                      mx::matmul(a[0], mx::transpose(a[2])),
                      mx::matmul(a[0], mx::transpose(a[3]))};
    });
    auto fused = mx::compile([](const Arrays &a) {
        return mx::split(mx::matmul(a[0], mx::transpose(a[1])), 3, -1);
    });
    const Arrays a{x, q, k, v}, b{x, w};
    auto reference = separate(a), candidate = fused(b);
    mx::eval(reference); mx::eval(candidate);
    double error = 0;
    for (int i = 0; i < 3; ++i) {
        auto expected = mx::astype(reference[i], mx::float32);
        auto actual = mx::astype(candidate[i], mx::float32);
        const auto relative = mx::sqrt(mx::sum(mx::square(actual - expected)) /
                                       mx::sum(mx::square(expected))).item<float>();
        if (!std::isfinite(relative)) throw std::runtime_error("nonfinite QKV error");
        error = std::max(error, double(relative));
    }
    auto time = [](auto &graph, const Arrays &args) {
        const auto start = std::chrono::steady_clock::now();
        mx::eval(graph(args));
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    };
    std::vector<double> baseline_ms, fused_ms;
    for (int i = 0; i < runs + 4; ++i) {
        double t0, t1;
        if (i % 2) { t1 = time(fused, b); t0 = time(separate, a); }
        else { t0 = time(separate, a); t1 = time(fused, b); }
        if (i >= 4) { baseline_ms.push_back(t0); fused_ms.push_back(t1); }
    }
    std::cout << "{\"rows\":" << rows << ",\"separate_ms\":" << median(baseline_ms)
              << ",\"fused_ms\":" << median(fused_ms)
              << ",\"relative_l2_max\":" << error << "}" << std::endl;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
}
