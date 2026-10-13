#include "../../native/models/seedvr2/metal/conv_tiled.hpp"
#include "../../native/models/seedvr2/metal/group_norm.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <mlx/random.h>
#include <cmath>
#include <fstream>
#include <iostream>

namespace mx = mlx::core;
using tc::Tensor;

struct Case {
    int batch, time, height, width, channels, out_channels, kt, ks, halo, st, ss, pad;
    bool tail, repeat, fp32;
};

int main(int argc, char **argv) {
    try {
        tc::require(argc == 2, "usage: seedvr2-conv-probe FRESH_REPORT.json");
        const std::filesystem::path report_path = argv[1];
        tc::require(!std::filesystem::exists(report_path), "fresh convolution report required");
        if (!report_path.parent_path().empty()) std::filesystem::create_directories(report_path.parent_path());
        mx::set_default_device(mx::Device::gpu);
        tc::seedvr2::AllocatorCacheScope allocator(512ull << 20);
        const std::vector<Case> cases{
            {1,1, 9,11, 3,32, 3,3,2,1,1,1, false,true,false},
            {2,4,11,13,16,32, 3,3,1,2,2,0, true,false,false},
            {1,3,16,17,128,128,3,3,2,1,1,1,false,false,false},
            {2,3,11, 7,32,64, 1,1,0,1,1,0, false,false,false},
            {1,5,11,13,16,32, 1,3,0,1,2,0, true,false,false},
            {1,5, 9,11,32,32, 3,3,2,2,2,0, true,true,false},
            {1,1,17,19,128,128,1,3,0,1,1,1,false,false,true},
            {1,4, 9,11, 3, 5, 3,3,1,2,2,0, true,false,false},
            {1,1,17,19, 3,32, 1,3,0,1,1,1,false,false,true},
            {1,1,17,19,512,32,1,3,0,1,1,1,false,false,false},
        };
        std::ofstream report(report_path);
        tc::require(bool(report), "cannot write convolution report");
        report << "{\"schema\":\"seedvr2-tiled-convolution-parity-v1\",\"cases\":[";
        bool passed = true;
        for (size_t index = 0; index < cases.size(); ++index) {
            const auto &c = cases[index];
            auto input = mx::random::normal({c.batch,c.time,c.height,c.width,c.channels}, mx::float16, mx::random::key(42+index));
            auto head = c.repeat ? mx::slice(input, {0,0,0,0,0}, {c.batch,1,c.height,c.width,c.channels}) :
                (c.halo ? mx::random::normal({c.batch,c.halo,c.height,c.width,c.channels}, mx::float16, mx::random::key(142+index)) : input);
            auto weight = mx::astype(mx::random::normal({c.out_channels,c.kt,c.ks,c.ks,c.channels}, mx::float32,
                mx::random::key(242+index)) * (.2f / std::sqrt(float(c.kt*c.ks*c.ks*c.channels))), c.fp32 ? mx::float32 : mx::float16);
            auto full = c.halo ? mx::concatenate({c.repeat ? mx::repeat(head,c.halo,1) : head,input},1) : input;
            if (c.tail) full = mx::pad(full, std::vector<std::pair<int,int>>{{0,0},{0,0},{0,1},{0,1},{0,0}});
            auto reference = mx::astype(mx::conv3d(full, weight, {c.st,c.ss,c.ss}, {0,c.pad,c.pad}), mx::float16);
            auto actual = tc::seedvr2::metal::conv_tiled(input, head, c.halo, c.repeat, weight, c.st, c.ss, c.pad, c.tail, 128);
            tc::require(actual.shape() == reference.shape() && actual.dtype() == reference.dtype(), "tiled convolution ABI mismatch");
            mx::eval({reference,actual});
            auto x = mx::astype(reference,mx::float32), y = mx::astype(actual,mx::float32);
            const bool finite = mx::all(mx::isfinite(y)).item<bool>() && mx::all(mx::isfinite(x)).item<bool>();
            const double nx = mx::sum(x*x).item<float>(), ny = mx::sum(y*y).item<float>();
            const double l2 = std::sqrt(mx::sum((x-y)*(x-y)).item<float>() / nx);
            const double cosine = mx::sum(x*y).item<float>() / std::sqrt(nx*ny);
            const double maximum = mx::max(mx::abs(x-y)).item<float>();
            const bool ok = finite && l2 <= .003 && cosine >= .99999;
            passed &= ok;
            if (index) report << ',';
            report << "{\"index\":" << index << ",\"finite\":" << (finite ? "true" : "false")
                   << ",\"relative_l2\":" << l2 << ",\"cosine\":" << cosine
                   << ",\"max_abs\":" << maximum << ",\"passed\":" << (ok ? "true" : "false") << '}';
            std::cout << "case=" << index << " relative_l2=" << l2 << " cosine=" << cosine << " max_abs=" << maximum << '\n';
        }
        report << "],\"norm_cases\":[";
        for (int mode = 0; mode < 5; ++mode) {
            const int c = mode == 4 ? 512 : 128;
            auto input = mx::random::normal({1,2,19,23,c}, mx::float32, mx::random::key(942+mode));
            if (mode == 1) input = mx::ones(input.shape(), mx::float32);
            if (mode == 2) input = input * .001f + 1.f;
            if (mode == 3) input = input * 1000.f;
            input = mx::astype(input, mx::float16);
            if (mode == 4) input = mx::transpose(input, {0,1,3,2,4});
            auto weight = mx::astype(mx::random::normal({c}, mx::float32, mx::random::key(1042+mode)) * .1f + 1.f, mx::float16);
            auto bias = mx::astype(mx::random::normal({c}, mx::float32, mx::random::key(1142+mode)) * .1f, mx::float16);
            for (bool activate : {false,true}) {
                const int bt = input.shape(0)*input.shape(1), hw = input.shape(2)*input.shape(3);
                auto grouped = mx::reshape(input,{bt,hw,32,c/32});
                grouped = mx::reshape(mx::transpose(grouped,{0,2,1,3}), {bt,32,hw*(c/32)});
                auto normalized = mx::fast::layer_norm(mx::astype(grouped,mx::float32),{},{},1e-6f);
                normalized = mx::reshape(mx::transpose(mx::reshape(normalized,{bt,32,hw,c/32}),{0,2,1,3}), input.shape());
                auto reference = mx::astype(normalized*mx::astype(weight,mx::float32)+mx::astype(bias,mx::float32),mx::float16);
                if (activate) reference = reference*mx::sigmoid(reference);
                auto actual = tc::seedvr2::metal::group_norm(input,weight,bias,activate);
                mx::eval({reference,actual});
                auto x = mx::astype(reference,mx::float32), y = mx::astype(actual,mx::float32);
                const double nx = mx::sum(x*x).item<float>(), ny = mx::sum(y*y).item<float>();
                const double l2 = std::sqrt(mx::sum((x-y)*(x-y)).item<float>() / nx);
                const double cosine = mx::sum(x*y).item<float>() / std::sqrt(nx*ny);
                const bool finite = mx::all(mx::isfinite(y)).item<bool>();
                const bool ok = finite && l2 <= .003 && cosine >= .99999;
                passed &= ok;
                if (mode || activate) report << ',';
                report << "{\"mode\":" << mode << ",\"activate\":" << (activate ? "true" : "false")
                       << ",\"finite\":" << (finite ? "true" : "false")
                       << ",\"relative_l2\":" << l2 << ",\"cosine\":" << cosine
                       << ",\"passed\":" << (ok ? "true" : "false") << '}';
                std::cout << "norm=" << mode << " activate=" << activate << " relative_l2=" << l2 << " cosine=" << cosine << '\n';
                if (!ok) mx::save_safetensors((report_path.parent_path()/(report_path.stem().string()+"-norm-failed-"+std::to_string(mode)+"-"+std::to_string(activate)+".safetensors")).string(),
                    {{"input",input},{"weight",weight},{"bias",bias},{"reference",reference},{"actual",actual}});
            }
        }
        report << "],\"passed\":" << (passed ? "true" : "false") << "}\n";
        report.close();
        tc::require(passed, "SeedVR2 convolution/norm numerical parity failed");
        std::cout << "PASS causal/replicated/batched/strided/rectangular/incomplete-tile/FP32-filter convolutions and norm cases\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
