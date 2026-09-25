// Qwen21 FFN shape-matched GPU screening; never a model-level speedup claim.
#include "z_image_gpu_benchmark_lock.hpp"
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
using Arrays = std::vector<mx::array>;

mx::array mpp_projection(const mx::array &x, const mx::array &weight,
                         int bm, int bn) {
    static auto kernel = mx::fast::metal_kernel(
        "tc_qwen21_mpp_ffn_probe", {"x", "weight"}, {"output"}, R"metal(
        using namespace mpp::tensor_ops;
        uint row = threadgroup_position_in_grid.y * BM;
        uint column = threadgroup_position_in_grid.x * BN;
        auto a = tensor(const_cast<device T*>(x), dextents<int,2>{K,M}, array<int,2>{1,K});
        auto b = tensor(const_cast<device T*>(weight), dextents<int,2>{K,N}, array<int,2>{1,K});
        auto aa = a.slice(0,row), bb = b.slice(0,column);
        matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),
                 execution_simdgroups<4>> op;
        auto acc = op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
        op.run(aa,bb,acc);
        for (uint i=0; i<acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            auto coord = acc.get_multidimensional_index(i);
            if (row+coord[1]<M && column+coord[0]<N)
                output[(row+coord[1])*N+column+coord[0]] = T(acc[i]);
        }
        )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const int m=x.shape(0), n=weight.shape(0), k=x.shape(1);
    return kernel({x,weight}, {{m,n}}, {x.dtype()},
                  {((n+bn-1)/bn)*128,(m+bm-1)/bm,1}, {128,1,1},
                  {{"T",x.dtype()},{"M",m},{"N",n},{"K",k},{"BM",bm},{"BN",bn}},
                  {}, false, {})[0];
}

mx::array mpp_swiglu(const mx::array &x, const mx::array &packed,
                     int bm, int bn) {
    static auto kernel = mx::fast::metal_kernel(
        "tc_qwen21_mpp_packed_swiglu_probe", {"x", "packed"}, {"output"}, R"metal(
        using namespace mpp::tensor_ops;
        uint row = threadgroup_position_in_grid.y * BM;
        uint column = threadgroup_position_in_grid.x * BN;
        auto a = tensor(const_cast<device T*>(x), dextents<int,2>{4096,M}, array<int,2>{1,4096});
        auto b = tensor(const_cast<device T*>(packed), dextents<int,2>{4096,24576},
                        array<int,2>{1,4096});
        auto aa = a.slice(0,row), gate = b.slice(0,column);
        auto up = b.slice(0,column+12288);
        matmul2d<matmul2d_descriptor(BM,BN,4096,false,true,false),
                 execution_simdgroups<4>> op;
        auto g = op.template get_destination_cooperative_tensor<decltype(aa),decltype(gate),float>();
        auto u = op.template get_destination_cooperative_tensor<decltype(aa),decltype(up),float>();
        op.run(aa,gate,g); op.run(aa,up,u);
        for (uint i=0; i<g.get_capacity(); ++i) {
            if (!g.is_valid_element(i)) continue;
            auto coord = g.get_multidimensional_index(i);
            if (row+coord[1]<M && column+coord[0]<12288) {
                T gg=T(g[i]), uu=T(u[i]);
                float sigmoid = 1.0f/(1.0f+metal::exp(-float(gg)));
                output[(row+coord[1])*12288+column+coord[0]] = T(T(float(gg)*sigmoid)*uu);
            }
        }
        )metal", "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const int m=x.shape(0);
    return kernel({x,packed}, {{m,12288}}, {x.dtype()},
                  {(12288/bn)*128,(m+bm-1)/bm,1}, {128,1,1},
                  {{"T",x.dtype()},{"M",m},{"BM",bm},{"BN",bn}},
                  {}, false, {})[0];
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(),samples.end());
    const size_t n=samples.size();
    return (samples[(n-1)/2]+samples[n/2])/2;
}

int main(int argc, char **argv) {
    try {
        if (argc != 4 && argc != 5)
            throw std::invalid_argument("usage: qwen21-gemm-probe gate_up|down|gate_swiglu ROWS RUNS [--fp16]");
        const std::string operation(argv[1]);
        if (operation != "gate_up" && operation != "down" && operation != "gate_swiglu")
            throw std::invalid_argument("operation must be gate_up, down or gate_swiglu");
        const int rows=std::stoi(argv[2]), runs=std::stoi(argv[3]);
        const bool fp16_only = argc == 5 && std::string(argv[4]) == "--fp16";
        if (argc == 5 && !fp16_only)
            throw std::invalid_argument("unknown Qwen21 GEMM probe option");
        if ((rows != 1024 && rows != 4096) || runs < 2 || runs > 50)
            throw std::invalid_argument("rows must be 1024/4096 and runs must be 2...50");
        ZImageGpuBenchmarkLock benchmark_lock;
        const int input=operation=="down" ? 12288 : 4096;
        const int output=operation=="down" ? 4096 : 24576;
        mx::set_default_device(mx::Device(mx::Device::gpu));
        mx::random::seed(42);
        auto x=mx::random::normal({rows,input},mx::bfloat16);
        auto w=mx::random::normal({output,input},mx::bfloat16);
        mx::eval(x,w);
        Arrays args{x,w};
        auto baseline=mx::compile([operation](const Arrays &a) {
            auto projected = mx::matmul(a[0],mx::transpose(a[1]));
            if (operation == "gate_swiglu") {
                auto parts = mx::split(projected,2,-1);
                return Arrays{(parts[0]*mx::sigmoid(parts[0]))*parts[1]};
            }
            return Arrays{projected};
        });
        auto expected=mx::astype(baseline(args)[0],mx::float32);
        mx::eval(expected);
        if (fp16_only) {
            auto x16=mx::astype(x,mx::float16), w16=mx::astype(w,mx::float16);
            mx::eval(x16,w16);
            Arrays fp16_args{x16,w16};
            auto fp16=mx::compile([operation](const Arrays &a) {
                auto projected=mx::matmul(a[0],mx::transpose(a[1]));
                if (operation == "gate_swiglu") {
                    auto parts=mx::split(projected,2,-1);
                    return Arrays{(parts[0]*mx::sigmoid(parts[0]))*parts[1]};
                }
                return Arrays{projected};
            });
            auto actual=mx::astype(fp16(fp16_args)[0],mx::float32);
            auto relative_l2=mx::sqrt(mx::sum(mx::square(actual-expected)) /
                                      mx::sum(mx::square(expected))).item<float>();
            auto max_abs=mx::max(mx::abs(actual-expected)).item<float>();
            if (!std::isfinite(relative_l2) || !std::isfinite(max_abs))
                throw std::runtime_error("nonfinite Qwen21 FP16 projection error");
            auto timed=[&](auto &graph,const Arrays &input_args) {
                auto start=std::chrono::steady_clock::now();
                mx::eval(graph(input_args));
                return std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-start).count();
            };
            std::vector<double> bf16_times, fp16_times;
            for (int i=0;i<runs+3;++i) {
                double bf16_ms,fp16_ms;
                if (i%2) { fp16_ms=timed(fp16,fp16_args); bf16_ms=timed(baseline,args); }
                else { bf16_ms=timed(baseline,args); fp16_ms=timed(fp16,fp16_args); }
                if (i>=3) { bf16_times.push_back(bf16_ms); fp16_times.push_back(fp16_ms); }
            }
            std::cout << "{\"operation\":\"" << operation << "\",\"rows\":" << rows
                      << ",\"precision\":\"fp16\",\"bf16_ms\":" << median(bf16_times)
                      << ",\"fp16_ms\":" << median(fp16_times)
                      << ",\"relative_l2\":" << relative_l2
                      << ",\"max_abs\":" << max_abs << "}" << std::endl;
            return 0;
        }
        for (const auto [bm,bn] : std::vector<std::pair<int,int>>{{16,128},{32,128},{48,128},{32,256},{64,128}}) {
            auto candidate=mx::compile([bm,bn,operation](const Arrays &a) {
                return Arrays{operation == "gate_swiglu"
                    ? mpp_swiglu(a[0],a[1],bm,bn)
                    : mpp_projection(a[0],a[1],bm,bn)};
            });
            auto actual=mx::astype(candidate(args)[0],mx::float32);
            auto relative_l2=mx::sqrt(mx::sum(mx::square(actual-expected)) /
                                      mx::sum(mx::square(expected))).item<float>();
            auto max_abs=mx::max(mx::abs(actual-expected)).item<float>();
            if (!std::isfinite(relative_l2) || !std::isfinite(max_abs))
                throw std::runtime_error("nonfinite Qwen21 MPP projection error");
            auto timed=[&](auto &graph) {
                auto start=std::chrono::steady_clock::now();
                mx::eval(graph(args));
                return std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-start).count();
            };
            std::vector<double> mlxtimes, mpptimes;
            for (int i=0;i<runs+3;++i) {
                double a,b;
                if (i%2) { b=timed(candidate); a=timed(baseline); }
                else { a=timed(baseline); b=timed(candidate); }
                if (i>=3) { mlxtimes.push_back(a); mpptimes.push_back(b); }
            }
            std::cout << "{\"operation\":\"" << operation << "\",\"rows\":" << rows
                      << ",\"input\":" << input << ",\"output\":" << output
                      << ",\"bm\":" << bm << ",\"bn\":" << bn
                      << ",\"mlx_ms\":" << median(mlxtimes)
                      << ",\"mpp_ms\":" << median(mpptimes)
                      << ",\"relative_l2\":" << relative_l2
                      << ",\"max_abs\":" << max_abs << "}" << std::endl;
        }
        if (operation == "gate_swiglu") return 0;
        for (int group_size : {32,64,128}) {
            auto quantized=mx::quantize(mx::astype(w,mx::float16),group_size,8,"affine");
            if (quantized.size()!=3)
                throw std::runtime_error("Qwen21 W8A16 expected weight, scales and biases");
            mx::eval(quantized);
            Arrays qargs{x,quantized[0],quantized[1],quantized[2]};
            auto qgemm=mx::compile([group_size](const Arrays &a) {
                return Arrays{mx::quantized_matmul(a[0],a[1],a[2],a[3],true,
                                                   group_size,8,"affine")};
            });
            auto actual=mx::astype(qgemm(qargs)[0],mx::float32);
            const auto relative_l2=mx::sqrt(mx::sum(mx::square(actual-expected))/
                                            mx::sum(mx::square(expected))).item<float>();
            const auto max_abs=mx::max(mx::abs(actual-expected)).item<float>();
            if (!std::isfinite(relative_l2) || !std::isfinite(max_abs))
                throw std::runtime_error("nonfinite Qwen21 W8A16 projection error");
            auto timed=[&](auto &graph,const Arrays &input_args) {
                auto start=std::chrono::steady_clock::now();
                mx::eval(graph(input_args));
                return std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-start).count();
            };
            std::vector<double> mlxtimes, qtimes;
            for (int i=0;i<runs+3;++i) {
                double a,b;
                if (i%2) { b=timed(qgemm,qargs); a=timed(baseline,args); }
                else { a=timed(baseline,args); b=timed(qgemm,qargs); }
                if (i>=3) { mlxtimes.push_back(a); qtimes.push_back(b); }
            }
            std::cout << "{\"operation\":\"" << operation << "\",\"rows\":" << rows
                      << ",\"precision\":\"w8a16\",\"group_size\":" << group_size
                      << ",\"mlx_ms\":" << median(mlxtimes)
                      << ",\"quant_ms\":" << median(qtimes)
                      << ",\"relative_l2\":" << relative_l2
                      << ",\"max_abs\":" << max_abs << "}" << std::endl;
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
