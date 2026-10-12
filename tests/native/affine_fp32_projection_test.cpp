#include "../../native/backends/affine_gpu_fp32.hpp"
#include <cmath>
#include <iostream>

namespace mx=mlx::core;
int main() {
    try {
        int cases=0;
        for(auto dtype:{mx::float16,mx::bfloat16})for(int bits:{4,8})for(int group:{32,64,128}) {
            auto physical=mx::astype(mx::random::normal({96,1024},mx::float32,mx::random::key(group+bits))*.1f,dtype);
            auto packed=mx::quantize(physical,group,bits);
            auto decoded=mx::dequantize(packed[0],packed[1],packed[2],group,bits,"affine",std::nullopt,dtype);
            for(int rows:{33,67}) {
                auto x=mx::astype(mx::random::normal({1,rows,512},mx::float32,mx::random::key(rows))*.2f,dtype);
                constexpr int rb=7,re=72,cb=256,ce=768;
                auto got=tc::affine_gpu::projection_fp32(x,packed[0],packed[1],packed[2],bits,rb,re,cb,ce);
                auto selected=mx::slice(decoded,{rb,cb},{re,ce});
                auto expected=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(selected,mx::float32)));
                mx::eval({got,expected});
                const float absolute=mx::max(mx::abs(got-expected)).item<float>();
                const float relative=mx::sqrt(mx::sum(mx::square(got-expected))/mx::sum(mx::square(expected))).item<float>();
                if(!mx::all(mx::isfinite(got)).item<bool>() || relative>3e-6f || absolute>2e-5f)
                    throw std::runtime_error("decoded-dtype F32 partial mismatch: bits="+std::to_string(bits)+" group="+std::to_string(group)+" rel="+std::to_string(relative)+" abs="+std::to_string(absolute));
                ++cases;
                for(int bad_begin:{-1,1,1024}) {
                    bool rejected=false;
                    try {tc::affine_gpu::projection_fp32(x,packed[0],packed[1],packed[2],bits,rb,re,bad_begin,ce);}
                    catch(const std::invalid_argument&) {rejected=true;}
                    if(!rejected)throw std::runtime_error("invalid affine F32 range accepted");
                }
            }
        }
        std::cout<<"PASS "<<cases<<" affine F32 partial cases: original FP16/BF16 decode, Q4/Q8, physical row/column offsets, group32/64/128 and tails\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
