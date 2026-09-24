#include "../../native/models/z_image/metal_kernels.hpp"
#include "../../native/models/z_image/metal/gate_norm_virtual.hpp"
#include "z_image_gpu_benchmark_lock.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace mx=mlx::core;
using Arrays=std::vector<mx::array>;
double median(std::vector<double> x) {
    std::sort(x.begin(),x.end());
    return (x[(x.size()-1)/2]+x[x.size()/2])/2;
}
int main(int argc,char **argv) {
    try {
        ZImageGpuBenchmarkLock lock;
        int rows=argc>1?std::stoi(argv[1]):1056;
        int threads=argc>2?std::stoi(argv[2]):256;
        int runs=argc>3?std::stoi(argv[3]):20;
        if(rows<1 || runs<1) throw std::invalid_argument("positive rows/runs required");
        mx::set_default_device(mx::Device::gpu);
        mx::random::seed(123);
        bool passed=true;
        for(auto dtype:{mx::bfloat16,mx::float16,mx::float32}) {
            Arrays a{mx::random::normal({1,rows,3840},dtype),
                     mx::random::normal({1,rows,3840},dtype),
                     mx::random::uniform(.5f,1.5f,{3840},dtype),
                     mx::random::normal({1,1,3840},dtype),
                     mx::random::uniform(.5f,1.5f,{3840},dtype),
                     mx::random::normal({1,1,3840},dtype)};
            mx::eval(a);
            // Match current M4 Max defaults, not the old960 scalar fusion.
            const bool virtual_control=dtype==mx::bfloat16 && (rows==4096 || rows==4128);
            auto control=mx::compile([virtual_control](const Arrays &a) {
                auto norm=[virtual_control](const mx::array &x,const mx::array &w,
                    const mx::array &mod,const mx::array &res,bool gated) {
                    bool precomputed=x.dtype()!=mx::float32;
                    if(virtual_control)
                        return tc::z_metal::norm_mod_virtual(x,w,mod,res,gated,precomputed,256);
                    return tc::z_metal::norm_mod(x,w,mod,res,gated,precomputed,true);
                };
                auto value=norm(a[0],a[2],a[3],a[1],true);
                return Arrays{value,norm(value,a[4],a[5],value,false)};
            });
            auto candidate=mx::compile([threads](const Arrays &a) {
                return tc::z_metal::gate_norm_virtual(a[0],a[1],a[2],a[3],a[4],a[5],threads);
            });
            auto expected=control(a),actual=candidate(a);
            double error=0;
            for(int i=0;i<2;++i) {
                float e=mx::max(mx::abs(mx::astype(expected[i],mx::float32)-
                                        mx::astype(actual[i],mx::float32))).item<float>();
                if(!std::isfinite(e)) throw std::runtime_error("nonfinite candidate");
                error=std::max(error,double(e));
            }
            passed &= error==0;
            auto timed=[&](auto &fn) {
                auto start=std::chrono::steady_clock::now();mx::eval(fn(a));
                return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            };
            std::vector<double> b,f;
            for(int i=0;i<runs+4;++i) {
                double bt,ft;
                if(i%2) {ft=timed(candidate);bt=timed(control);}
                else {bt=timed(control);ft=timed(candidate);}
                if(i>=4) {b.push_back(bt);f.push_back(ft);}
            }
            std::cout<<"{\"rows\":"<<rows<<",\"threads\":"<<threads
                <<",\"dtype\":\""<<(dtype==mx::bfloat16?"bf16":dtype==mx::float16?"fp16":"fp32")
                <<"\",\"virtual_control\":"<<(virtual_control?"true":"false")
                <<",\"max_abs\":"<<error<<",\"control_ms\":"<<median(b)
                <<",\"candidate_ms\":"<<median(f)<<"}"<<std::endl;
        }
        return passed?0:1;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 2;}
}
