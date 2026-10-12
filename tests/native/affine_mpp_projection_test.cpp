#include "../../native/backends/affine_gpu_mpp.hpp"
#include "../../native/backends/affine_gpu_fp32.hpp"
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

namespace mx=mlx::core;
void check(bool ok,const char *reason){if(!ok)throw std::runtime_error(reason);}
int main() {try {
    int cases=0;
    const std::vector<std::tuple<int,int,int>> recipes{{32,32,16},{32,64,16},{64,32,16},{32,32,32},{32,32,64}};
    for(auto dtype:{mx::float16,mx::bfloat16})for(int bits:{4,8})for(int group:{32,64,128}) {
        auto physical=mx::astype(mx::random::normal({96,1024},mx::float32,mx::random::key(group+bits))*.1f,dtype);
        auto packed=mx::quantize(physical,group,bits);
        auto decoded=mx::dequantize(packed[0],packed[1],packed[2],group,bits,"affine",std::nullopt,dtype);
        constexpr int rb=7,re=72,cb=256,ce=768;
        for(int rows:{33,67,128}) {
            auto source=mx::astype(mx::random::normal(rows==67?mx::Shape{512,rows}:mx::Shape{rows,512},
                mx::float32,mx::random::key(rows))*.2f,dtype);
            auto x=mx::expand_dims(rows==67?mx::transpose(source):source,0);
            auto selected=mx::slice(decoded,{rb,cb},{re,ce});
            auto expected=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(selected,mx::float32)));
            auto control=tc::affine_gpu::projection_fp32(x,packed[0],packed[1],packed[2],bits,rb,re,cb,ce);
            mx::eval({expected,control});
            for(const auto &[bk,sn,sm]:recipes) {
                auto got=tc::affine_gpu::projection_mpp_fp32(x,packed[0],packed[1],packed[2],bits,rb,re,cb,ce,bk,sn,sm);
                mx::eval(got);
                const float absolute=mx::max(mx::abs(got-expected)).item<float>();
                const float relative=mx::sqrt(mx::sum(mx::square(got-expected))/mx::sum(mx::square(expected))).item<float>();
                if(!mx::all(mx::isfinite(got)).item<bool>() || relative>3e-6f || absolute>2e-5f)
                    throw std::runtime_error("MPP typed affine F32 partial mismatch: dtype="+std::string(dtype==mx::bfloat16?"bf16":"f16")+
                        " bits="+std::to_string(bits)+" group="+std::to_string(group)+" BK="+std::to_string(bk)+
                        " SN="+std::to_string(sn)+" SM="+std::to_string(sm)+" rel="+std::to_string(relative)+" abs="+std::to_string(absolute));
                check(got.shape()==expected.shape() && got.dtype()==mx::float32,"MPP output geometry/dtype mismatch");
                ++cases;
            }
            if(group==32 && rows==33) {
                // K480 has a half-filled BK64 final tile. Its source is still
                // the original physical 1024-column matrix, not a compact W.
                auto tail_x=mx::slice(x,{0,0,0},{1,rows,480});
                auto tail_w=mx::slice(decoded,{rb,cb},{re,cb+480});
                auto wanted=mx::matmul(mx::astype(tail_x,mx::float32),mx::transpose(mx::astype(tail_w,mx::float32)));
                for(const auto &[bk,sn,sm]:recipes) {
                    auto got=tc::affine_gpu::projection_mpp_fp32(tail_x,packed[0],packed[1],packed[2],bits,rb,re,cb,cb+480,bk,sn,sm);
                    const float relative=mx::sqrt(mx::sum(mx::square(got-wanted))/mx::sum(mx::square(wanted))).item<float>();
                    check(mx::all(mx::isfinite(got)).item<bool>() && relative<=3e-6f &&
                        mx::max(mx::abs(got-wanted)).item<float>()<=2e-5f,"MPP K tail failed original typed F32 gate");
                    ++cases;
                }
            }
            for(int bad_begin:{-1,1,1024}) {
                bool rejected=false;
                try{tc::affine_gpu::projection_mpp_fp32(x,packed[0],packed[1],packed[2],bits,rb,re,bad_begin,ce);}
                catch(const std::invalid_argument&){rejected=true;}
                check(rejected,"invalid MPP affine range accepted");
            }
            for(int invalid:{0,16,48,128}) {
                bool rejected=false;
                try{tc::affine_gpu::projection_mpp_fp32(x,packed[0],packed[1],packed[2],bits,rb,re,cb,ce,invalid);}
                catch(const std::invalid_argument&){rejected=true;}
                check(rejected,"invalid MPP K tile accepted");
            }
            bool rejected=false;
            try{tc::affine_gpu::projection_mpp_fp32(x,packed[0],packed[1],packed[2],bits,rb,re,cb,ce,64,64);}
            catch(const std::invalid_argument&){rejected=true;}
            check(rejected,"numerically rejected MPP tile admitted");
            for(int sm:{0,8,48,128}) {
                rejected=false;
                try{tc::affine_gpu::projection_mpp_fp32(x,packed[0],packed[1],packed[2],bits,rb,re,cb,ce,32,32,sm);}
                catch(const std::invalid_argument&){rejected=true;}
                check(rejected,"invalid MPP row tile admitted");
            }
        }
    }
    std::cout<<"PASS MPP register-decode affine F32 cases="<<cases<<": FP16/BF16 Q4/Q8, groups32/64/128, five recipes, physical offsets, tails/strided input and invalid geometry\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
