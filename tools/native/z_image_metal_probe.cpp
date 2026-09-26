// Exercises the actual production candidate and the existing compiled path.
#include "../../native/models/z_image/metal_kernels.hpp"
#include "../../native/models/z_image/metal/norm_mod_virtual.hpp"
#include "z_image_gpu_benchmark_lock.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>

namespace mx = mlx::core;
using Arrays = std::vector<mx::array>;

Arrays reference(const Arrays &a) {
    const int n = a[0].shape(1);
    auto parts = mx::split(a[0], 3, -1);
    auto heads = [n](const mx::array &v) {
        return mx::transpose(mx::reshape(v, {1, n, 30, 128}), {0, 2, 1, 3});
    };
    auto norm = [](const mx::array &v, const mx::array &w) {
        return mx::astype(mx::fast::rms_norm(mx::astype(v, mx::float32),
                           mx::astype(w, mx::float32), 1e-5f), v.dtype());
    };
    auto q = norm(heads(parts[0]), a[1]), k = norm(heads(parts[1]), a[2]);
    static auto rope = mx::fast::metal_kernel(
        "tc_z_probe_reference_rope", {"q", "k", "freqs", "pairs", "span"},
        {"q_out", "k_out"}, R"metal(
        uint pair = thread_position_in_grid.x;
        if (pair < uint(pairs)) {
            uint base = pair * 2, fbase = (pair % uint(span)) * 2;
            float c = float(freqs[fbase]), s = float(freqs[fbase + 1]);
            float qa = float(q[base]), qb = float(q[base+1]);
            float ka = float(k[base]), kb = float(k[base+1]);
            q_out[base] = T(qa*c - qb*s); q_out[base+1] = T(qa*s + qb*c);
            k_out[base] = T(ka*c - kb*s); k_out[base+1] = T(ka*s + kb*c);
        }
        )metal");
    auto result = rope({q, k, a[3], mx::array(int(q.size()/2)), mx::array(n*64)},
                      {q.shape(), k.shape()}, {q.dtype(), k.dtype()},
                      {int(q.size()/2), 1, 1}, {256, 1, 1}, {{"T", q.dtype()}}, {}, false, {});
    result.push_back(heads(parts[2]));
    return result;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return (v[(v.size()-1)/2] + v[v.size()/2]) * 0.5;
}

int main(int argc, char **argv) {
    try {
        ZImageGpuBenchmarkLock benchmark_lock;
        const int n = argc > 1 ? std::stoi(argv[1]) : 1056;
        const int repetitions = argc > 2 ? std::stoi(argv[2]) : 20;
        const bool precompute_gate = argc > 3 && std::string(argv[3]) == "precompute";
        const bool vector_norm = argc > 4 && std::string(argv[4]) == "vector";
        const int virtual_threads = argc > 5 ? std::stoi(argv[5]) : 0;
        if (virtual_threads && !vector_norm)
            throw std::invalid_argument("virtual threads require vector comparison mode");
        if (n <= 0 || repetitions <= 0) throw std::invalid_argument("positive rows/runs required");
        mx::set_default_device(mx::Device::gpu);
        mx::random::seed(42);
        auto base = mx::compile(reference);
        auto fused = mx::compile([](const Arrays &a) {
            return tc::z_metal::prepare_qkv(a[0], a[1], a[2], a[3]);
        });
        auto timed = [&](auto &fn, const Arrays &a) {
            const auto start = std::chrono::steady_clock::now();
            auto out = fn(a); mx::eval(out);
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now()-start).count();
        };
        bool passed = true;
        std::cout << "{\"rows\":" << n << ",\"cases\":[";
        bool first = true;
        for (auto dtype : {mx::bfloat16, mx::float16, mx::float32}) {
            auto qkv = mx::random::normal({1, n, 11520}, dtype);
            auto qw = mx::astype(mx::random::uniform(.5f, 1.5f, {128}), dtype);
            auto kw = mx::astype(mx::random::uniform(.5f, 1.5f, {128}), dtype);
            auto angle = mx::random::uniform(-3.14f, 3.14f, {n, 64});
            auto freqs = mx::stack({mx::cos(angle), mx::sin(angle)}, -1);
            Arrays a{qkv, qw, kw, freqs}; mx::eval(a);
            auto expected = base(a), actual = fused(a); mx::eval(expected); mx::eval(actual);
            double max_error = 0, rel_error = 0;
            for (int i = 0; i < 3; ++i) {
                auto e = mx::astype(expected[i], mx::float32);
                auto v = mx::astype(actual[i], mx::float32);
                double max = mx::max(mx::abs(e-v)).item<float>();
                double rel = mx::sqrt(mx::sum(mx::square(e-v)) /
                                      mx::sum(mx::square(e))).item<float>();
                if (!std::isfinite(max) || !std::isfinite(rel)) passed = false;
                max_error = std::max(max_error, max); rel_error = std::max(rel_error, rel);
            }
            passed &= rel_error < (dtype == mx::float32 ? 2e-6 : 0.001);
            std::vector<double> b, f;
            for (int i = 0; i < 4; ++i) { timed(base,a); timed(fused,a); }
            for (int i = 0; i < repetitions; ++i) {
                if (i%2) { f.push_back(timed(fused,a)); b.push_back(timed(base,a)); }
                else { b.push_back(timed(base,a)); f.push_back(timed(fused,a)); }
            }
            if (!first) std::cout << ','; first = false;
            std::cout << "{\"dtype\":\"" << (dtype==mx::bfloat16?"bf16":dtype==mx::float16?"fp16":"fp32")
                      << "\",\"max_abs\":" << max_error << ",\"relative_l2\":" << rel_error
                      << ",\"baseline_ms\":" << median(b) << ",\"fused_ms\":" << median(f)
                      << ",\"speedup\":" << median(b)/median(f) << '}';
        }
        std::cout << "],\"passed\":" << (passed ? "true" : "false") << "}\n";
        for (auto dtype : {mx::bfloat16, mx::float16, mx::float32}) {
            auto x = mx::random::normal({1,n,3840}, dtype);
            auto w = mx::random::uniform(.5f, 1.5f, {3840}, dtype);
            auto mod = mx::random::normal({1,1,3840}, dtype);
            auto res = mx::random::normal({1,n,3840}, dtype);
            Arrays a{x,w,mod,res}; mx::eval(a);
            for (bool gated : {false,true}) {
                // Match production: FP32 uses precise inline tanh.
                const bool use_precomputed_gate = precompute_gate && dtype != mx::float32;
                auto ref = mx::compile([gated, use_precomputed_gate, vector_norm, virtual_threads](const Arrays &v) {
                    if (vector_norm)
                        return Arrays{tc::z_metal::norm_mod(v[0],v[1],v[2],v[3],gated,use_precomputed_gate,virtual_threads != 0)};
                    auto norm = mx::astype(mx::fast::rms_norm(mx::astype(v[0], mx::float32),
                        mx::astype(v[1], mx::float32), 1e-5f), v[0].dtype());
                    return Arrays{gated ? v[3] + mx::tanh(v[2]) * norm
                                        : norm * (mx::array(1.f, v[0].dtype()) + v[2])};
                });
                auto fast = mx::compile([gated, use_precomputed_gate, vector_norm, virtual_threads](const Arrays &v) {
                    if (virtual_threads)
                        return Arrays{tc::z_metal::norm_mod_virtual(v[0],v[1],v[2],v[3],gated,use_precomputed_gate,virtual_threads)};
                    return Arrays{tc::z_metal::norm_mod(v[0],v[1],v[2],v[3],gated,use_precomputed_gate,vector_norm)};
                });
                auto e = mx::astype(ref(a)[0], mx::float32);
                auto v = mx::astype(fast(a)[0], mx::float32);
                double max = mx::max(mx::abs(e-v)).item<float>();
                double rel = mx::sqrt(mx::sum(mx::square(e-v)) / mx::sum(mx::square(e))).item<float>();
                bool ok = std::isfinite(max) && std::isfinite(rel) && rel < (dtype == mx::float32 ? 2e-6 : 0.001);
                if (vector_norm) ok = ok && max == 0;
                passed &= ok;
                std::vector<double> b, f;
                for (int i=0; i<4; ++i) {timed(ref,a);timed(fast,a);}
                for (int i=0; i<repetitions; ++i) {
                    if (i%2) {f.push_back(timed(fast,a));b.push_back(timed(ref,a));}
                    else {b.push_back(timed(ref,a));f.push_back(timed(fast,a));}
                }
                std::cout << "{\"norm_mod\":true,\"rows\":" << n
                          << ",\"vector_norm\":" << (vector_norm ? "true" : "false")
                          << ",\"virtual_threads\":" << virtual_threads
                          << ",\"dtype\":\"" << (dtype==mx::bfloat16?"bf16":dtype==mx::float16?"fp16":"fp32")
                          << "\",\"gated\":" << (gated ? "true" : "false")
                          << ",\"precompute_gate\":" << (use_precomputed_gate ? "true" : "false")
                          << ",\"max_abs\":" << max << ",\"relative_l2\":" << rel
                          << ",\"baseline_ms\":" << median(b) << ",\"fused_ms\":" << median(f)
                          << ",\"passed\":" << (ok ? "true" : "false") << "}\n";
            }
        }
        for (auto dtype : {mx::bfloat16, mx::float16, mx::float32}) {
            Arrays a{mx::random::normal({1,n,3840},dtype),
                     mx::random::uniform(.5f,1.5f,{3840},dtype),
                     mx::random::normal({1,1,3840},dtype),
                     mx::random::normal({1,n,3840},dtype)};
            mx::eval(a);
            auto ref = mx::compile([](const Arrays &a) {
                auto norm = [](const mx::array &x, const mx::array &w) {
                    return mx::astype(mx::fast::rms_norm(mx::astype(x,mx::float32),
                        mx::astype(w,mx::float32),1e-5f),x.dtype());
                };
                auto value = a[3] + mx::tanh(a[2])*norm(a[0],a[1]);
                auto feed = norm(value,a[1])*(mx::array(1.f,a[2].dtype())+a[2]);
                return Arrays{value,feed};
            });
            auto fast = mx::compile([](const Arrays &a) {
                return tc::z_metal::gate_norm(a[0],a[3],a[1],a[2],a[1],a[2]);
            });
            auto expected = ref(a), actual = fast(a);
            double max_error=0, rel_error=0;
            for (int i=0;i<2;++i) {
                auto e=mx::astype(expected[i],mx::float32), v=mx::astype(actual[i],mx::float32);
                double abs=mx::max(mx::abs(e-v)).item<float>();
                double rel=mx::sqrt(mx::sum(mx::square(e-v))/mx::sum(mx::square(e))).item<float>();
                if (!std::isfinite(abs) || !std::isfinite(rel)) passed=false;
                max_error=std::max(max_error,abs); rel_error=std::max(rel_error,rel);
            }
            bool ok = rel_error < (dtype == mx::float32 ? 2e-6 : .001);
            passed &= ok;
            std::vector<double> b,f;
            for(int i=0;i<4;++i) {timed(ref,a);timed(fast,a);}
            for(int i=0;i<repetitions;++i) {
                if(i%2) {f.push_back(timed(fast,a));b.push_back(timed(ref,a));}
                else {b.push_back(timed(ref,a));f.push_back(timed(fast,a));}
            }
            std::cout << "{\"gate_norm\":true,\"rows\":" << n << ",\"dtype\":\""
                      << (dtype==mx::bfloat16?"bf16":dtype==mx::float16?"fp16":"fp32")
                      << "\",\"max_abs\":" << max_error << ",\"relative_l2\":" << rel_error
                      << ",\"baseline_ms\":" << median(b) << ",\"fused_ms\":" << median(f)
                      << ",\"passed\":" << (ok?"true":"false") << "}\n";
        }
        return passed ? 0 : 1;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}
