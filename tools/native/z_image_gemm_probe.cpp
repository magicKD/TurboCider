// Shape-matched operator attribution. This is not an end-to-end speed claim.
#include "../../native/models/z_image/metal_kernels.hpp"
#include "z_image_gpu_benchmark_lock.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace mx = mlx::core;
using Arrays = std::vector<mx::array>;

mx::array mpp_gemm(const mx::array &x, const mx::array &w, int bm, int bn,
                   const mx::array &aux, bool swiglu, int simdgroups) {
    static auto kernel = mx::fast::metal_kernel(
        "tc_z_mpp_gemm_probe", {"x", "w", "aux"}, {"out"}, R"metal(
        using namespace mpp::tensor_ops;
        uint row = threadgroup_position_in_grid.y * BM;
        uint col = threadgroup_position_in_grid.x * BN;
        // TensorOps traits reject const element types. These views remain
        // read-only operands: no instruction stores through either pointer.
        auto a = tensor(const_cast<device T*>(x), dextents<int, 2>{K, M}, array<int, 2>{1, K});
        auto b = tensor(const_cast<device T*>(w), dextents<int, 2>{K, N}, array<int, 2>{1, K});
        auto aa = a.slice(0, row);
        auto bb = b.slice(0, col);
        matmul2d<matmul2d_descriptor(BM, BN, K, false, true, false),
                 execution_simdgroups<SG>> op;
        auto acc = op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
        op.run(aa, bb, acc);
        for (uint i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            auto index = acc.get_multidimensional_index(i);
            if (row + index[1] < M && col + index[0] < N) {
                uint offset = (row+index[1])*N+col+index[0];
                T value = T(acc[i]);
                if (SWIGLU) {
                    auto y = 1 / (1 + metal::exp(metal::abs(value)));
                    T sigmoid = (value < 0) ? y : 1 - y;
                    value = T(T(value * sigmoid) * aux[offset]);
                }
                out[offset] = value;
            }
        }
        )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const int m=x.shape(0), n=w.shape(0), k=x.shape(1);
    return kernel({x,w,aux}, {{m,n}}, {x.dtype()},
                  {((n+bn-1)/bn)*simdgroups*32,(m+bm-1)/bm,1}, {simdgroups*32,1,1},
                  {{"T",x.dtype()},{"M",m},{"N",n},{"K",k},{"BM",bm},{"BN",bn},
                   {"SWIGLU",swiglu},{"SG",simdgroups}}, {}, false, {})[0];
}

int main(int argc, char **argv) {
    try {
        ZImageGpuBenchmarkLock benchmark_lock;
        int n = argc > 1 ? std::stoi(argv[1]) : 1056;
        int runs = argc > 2 ? std::stoi(argv[2]) : 10;
        const std::string mode = argc > 3 ? argv[3] : "";
        bool mpp = mode.rfind("mpp", 0) == 0;
        const int quant_bits = mode == "q8" ? 8 : mode == "q4" ? 4 : 0;
        const int simdgroups = argc > 4 ? std::stoi(argv[4]) : 4;
        const std::string target = argc > 5 ? argv[5] : "all";
        if (simdgroups != 4 && simdgroups != 8)
            throw std::invalid_argument("simdgroups must be 4 or 8");
        if (target != "all" && target != "qkv" && target != "attention_out" &&
            target != "ffn_in" && target != "ffn_packed" && target != "ffn_down")
            throw std::invalid_argument("target must be all, qkv, attention_out, ffn_in, ffn_packed or ffn_down");
        const bool control_swiglu = mode == "mpp_swiglu_control";
        const bool wide_swiglu = mode == "mpp_swiglu_wide" || control_swiglu;
        bool swiglu = mode == "mpp_swiglu" || wide_swiglu;
        const auto dtype = argc > 3 && std::string(argv[3]) == "mpp_fp16"
            ? mx::float16 : mx::bfloat16;
        if (n < 1 || runs < 1) throw std::invalid_argument("positive rows/runs required");
        auto cpu = mx::new_thread_unsafe_stream(mx::Device(mx::Device::cpu));
        auto gpu = mx::new_thread_unsafe_stream(mx::Device(mx::Device::gpu));
        mx::set_default_stream(cpu); mx::set_default_stream(gpu);
        mx::set_default_device(mx::Device::gpu);
        mx::random::seed(42);
        bool all_finite = true;
        for (auto shape : {mx::Shape{11520,3840}, mx::Shape{3840,3840},
                           mx::Shape{10240,3840}, mx::Shape{20480,3840},
                           mx::Shape{3840,10240}}) {
            const std::string shape_target =
                shape == mx::Shape{11520,3840} ? "qkv" :
                shape == mx::Shape{3840,3840} ? "attention_out" :
                shape == mx::Shape{10240,3840} ? "ffn_in" :
                shape == mx::Shape{20480,3840} ? "ffn_packed" : "ffn_down";
            if (target != "all" && target != shape_target) continue;
            if (swiglu && shape != mx::Shape{10240,3840}) continue;
            auto x = mx::random::normal({n,shape[1]}, dtype);
            auto w = mx::random::normal(shape, dtype);
            auto aux = mx::random::normal({n,shape[0]},dtype);
            Arrays a{x,w,aux}; mx::eval(a);
            auto fn = mx::compile([swiglu,wide_swiglu](const Arrays &a) {
                if (wide_swiglu) {
                    auto out = tc::z_metal::swiglu_gemm(
                        mx::reshape(a[0],{1,a[0].shape(0),3840}),a[1],
                        mx::reshape(a[2],{1,a[0].shape(0),10240}));
                    return Arrays{mx::reshape(out,{a[0].shape(0),10240})};
                }
                auto gate = mx::matmul(a[0],mx::transpose(a[1]));
                return Arrays{swiglu ? (gate * mx::sigmoid(gate)) * a[2] : gate};
            });
            std::vector<double> times;
            for (int i = 0; i < runs + 3; ++i) {
                auto start = std::chrono::steady_clock::now();
                mx::eval(fn(a));
                if (i >= 3) times.push_back(std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-start).count());
            }
            std::sort(times.begin(),times.end());
            double ms = (times[(runs-1)/2]+times[runs/2])/2;
            std::cout << "{\"rows\":" << n << ",\"output\":" << shape[0]
                      << ",\"input\":" << shape[1] << ",\"dtype\":\""
                      << (dtype == mx::float16 ? "fp16" : "bf16") << "\",\"median_ms\":" << ms
                      << ",\"effective_tflops\":" << (2.*n*shape[0]*shape[1]/(ms*1.e9)) << "}\n";
            std::cout.flush();
            if (quant_bits) {
                auto qw = mx::quantize(w,64,quant_bits,"affine");
                mx::eval(qw);
                Arrays qa{x,qw[0],qw[1],qw[2]};
                auto qfn = mx::compile([quant_bits](const Arrays &v) {
                    return Arrays{mx::quantized_matmul(v[0],v[1],v[2],v[3],true,64,quant_bits,"affine")};
                });
                auto expected = mx::astype(fn(a)[0],mx::float32);
                auto actual = mx::astype(qfn(qa)[0],mx::float32);
                const float rel = mx::sqrt(mx::sum(mx::square(actual-expected))/
                                          mx::sum(mx::square(expected))).item<float>();
                std::vector<double> baseline, quantized;
                auto timed = [&](auto &graph, const Arrays &inputs) {
                    auto start = std::chrono::steady_clock::now(); mx::eval(graph(inputs));
                    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                };
                for (int i=0;i<runs+3;++i) {
                    double b,f;
                    if (i%2) {f=timed(qfn,qa);b=timed(fn,a);} else {b=timed(fn,a);f=timed(qfn,qa);}
                    if (i>=3) {baseline.push_back(b);quantized.push_back(f);}
                }
                std::sort(baseline.begin(),baseline.end());
                std::sort(quantized.begin(),quantized.end());
                std::cout << "{\"bits\":" << quant_bits << ",\"group_size\":64,\"rows\":" << n
                          << ",\"output\":" << shape[0] << ",\"input\":" << shape[1]
                          << ",\"matched_baseline_ms\":" << (baseline[(runs-1)/2]+baseline[runs/2])/2
                          << ",\"quantized_ms\":" << (quantized[(runs-1)/2]+quantized[runs/2])/2
                          << ",\"relative_l2\":" << rel << "}\n";
                std::cout.flush();
            }
            if (mpp) {
                auto expected = mx::astype(fn(a)[0], mx::float32); mx::eval(expected);
                std::vector<std::pair<int,int>> tiles;
                if (wide_swiglu) {
                    // New region plus the shape-matched generic32x256 source.
                    // The generic source is not identical to production.
                    tiles = {{32,256},{96,128},{96,256},{96,512},
                             {128,128},{128,256},{128,512}};
                    if (control_swiglu) tiles = {{32,256}};
                } else {
                    for (int bm : {16,24,32,48,64})
                        for (int bn : {64,128,256}) tiles.emplace_back(bm,bn);
                }
                for (auto [bm,bn] : tiles) {
                    auto fast = mx::compile([bm,bn,swiglu,simdgroups](const Arrays &a) {return Arrays{mpp_gemm(a[0],a[1],bm,bn,a[2],swiglu,simdgroups)};});
                    auto actual = mx::astype(fast(a)[0], mx::float32);
                    float rel = mx::sqrt(mx::sum(mx::square(actual-expected))/mx::sum(mx::square(expected))).item<float>();
                    float max = mx::max(mx::abs(actual-expected)).item<float>();
                    if (!std::isfinite(rel) || !std::isfinite(max)) {
                        all_finite = false;
                        std::cout << "{\"mpp\":true,\"rows\":" << n
                                  << ",\"bm\":" << bm << ",\"bn\":" << bn
                                  << ",\"simdgroups\":" << simdgroups
                                  << ",\"valid\":false,\"reason\":\"nonfinite output\"}\n";
                        std::cout.flush();
                        continue;
                    }
                    std::vector<double> candidate;
                    std::vector<double> controls;
                    auto timed = [&](auto &graph) {
                        auto start=std::chrono::steady_clock::now(); mx::eval(graph(a));
                        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    };
                    for (int i=0;i<runs+3;++i) {
                        double b, f;
                        if (i%2) {f=timed(fast);b=timed(fn);} else {b=timed(fn);f=timed(fast);}
                        if (i>=3) {candidate.push_back(f);controls.push_back(b);}
                    }
                    std::sort(candidate.begin(),candidate.end());
                    std::sort(controls.begin(),controls.end());
                    std::cout << "{\"mpp\":true,\"rows\":" << n << ",\"output\":" << shape[0]
                              << ",\"input\":" << shape[1] << ",\"bm\":" << bm << ",\"bn\":" << bn
                              << ",\"simdgroups\":" << simdgroups
                              << ",\"control\":\"" << (wide_swiglu ? "production_swiglu_32x256" : "mlx") << "\""
                              << ",\"median_ms\":" << (candidate[(runs-1)/2]+candidate[runs/2])/2
                              << ",\"matched_baseline_ms\":" << (controls[(runs-1)/2]+controls[runs/2])/2
                              << ",\"swiglu\":" << (swiglu?"true":"false")
                              << ",\"relative_l2\":" << rel << ",\"max_abs\":" << max << "}\n";
                    std::cout.flush();
                }
            }
        }
        return all_finite ? 0 : 1;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
