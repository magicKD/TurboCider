// Isolated synthetic GPU channel-head sweep. No ANE/checkpoint/model-speed
// claims: measure the same physical weight views, output and hidden ownership.
#include "../../native/models/z_image/metal/projection.hpp"
#include "../../native/models/z_image/metal/swiglu_gemm.hpp"
#include <mlx/random.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace mx = mlx::core;
using Graph = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;
int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) return 2;
    try {
        const int rows = std::stoi(argv[1]), fg = argc == 3 ? std::stoi(argv[2]) : 6144;
        if ((rows != 1056 && rows != 4128) || fg <= 0 || fg >= 10240 || fg % 128) return 2;
        auto bf = [](mx::array value) { return mx::astype(value, mx::bfloat16); };
        std::vector<mx::array> args{
            bf(mx::random::normal({1,rows,3840},mx::float32,mx::random::key(1))*.1f),
            bf(mx::random::normal({10240,3840},mx::float32,mx::random::key(2))*.01f),
            bf(mx::random::normal({10240,3840},mx::float32,mx::random::key(3))*.01f),
            bf(mx::random::normal({3840,10240},mx::float32,mx::random::key(4))*.01f)};
        mx::eval(args);
        const std::vector<std::string> labels{"native", "native_fused_gate", "mpp32", "mpp_down16", "mpp_up_down16",
                                              "mpp_dual128", "mpp_dual256"};
        std::vector<Graph> graphs;
        for (int mode=0;mode<int(labels.size());++mode) graphs.push_back(mx::compile([mode,fg](const std::vector<mx::array>&a) {
            auto select=[&](const mx::array&w,int axis) {
                return axis==0 ? mx::slice(w,{0,0},{fg,w.shape(1)}) : mx::slice(w,{0,0},{w.shape(0),fg});
            };
            auto hidden=[&] {
                if (mode >= 5) return tc::z_metal::swiglu_dual_gemm_range(a[0],a[1],a[2],0,fg,mode==6?256:128);
                auto up=mode<2 ? mx::matmul(a[0],mx::transpose(select(a[2],0))) :
                    tc::z_metal::projection_range(a[0],a[2],0,fg,0,3840,mode==4?16:32);
                if (mode) return tc::z_metal::swiglu_gemm_range(a[0],a[1],up,0,fg);
                auto gate=mx::matmul(a[0],mx::transpose(select(a[1],0)));
                return (gate*mx::sigmoid(gate))*up;
            }();
            auto down=mode<2 ? mx::matmul(hidden,mx::transpose(select(a[3],1))) :
                tc::z_metal::projection_range(hidden,a[3],0,3840,0,fg,(mode==3||mode==4)?16:32);
            // A trace tag checks each captured mode really reached its own
            // compiled function; equal arithmetic must not hide cache aliasing.
            return std::vector<mx::array>{down,hidden,mx::array(mode,mx::int32)};
        }));
        for (const auto &graph:graphs) for (int i=0;i<3;++i) mx::eval(graph(args));
        auto reference=graphs[0](args);mx::eval(reference);
        for (int mode=0;mode<int(labels.size());++mode) {
            auto got=graphs[mode](args);mx::eval(got);
            if(got[2].item<int>()!=mode)throw std::runtime_error("compiled mode capture/cache alias");
            auto a=mx::astype(reference[0],mx::float32),b=mx::astype(got[0],mx::float32);
            const double l2=mx::sqrt(mx::sum(mx::square(a-b))/mx::sum(mx::square(a))).item<float>();
            if (!std::isfinite(l2) || l2>.02) throw std::runtime_error("channel GPU source oracle exceeds L2: "+labels[mode]);
            std::vector<double> samples;
            for (int i=0;i<10;++i) {
                const auto start=std::chrono::steady_clock::now();mx::eval(graphs[mode](args));
                samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
            }
            auto sorted=samples;std::sort(sorted.begin(),sorted.end());
            std::cout<<"{\"scope\":\"synthetic GPU-only channel-head component, NOT model speedup\",\"mode\":\""<<labels[mode]
                <<"\",\"rows\":"<<rows<<",\"gpu_channels\":"<<fg<<",\"median_seconds\":"<<(sorted[4]+sorted[5])*.5
                <<",\"source_relative_l2\":"<<l2<<",\"samples\":[";
            for(size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];std::cout<<"]}\n";
        }
    } catch (const std::exception&e) {std::cerr<<e.what()<<"\n";return 1;}
}
