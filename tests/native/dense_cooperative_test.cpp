#include "../../native/backends/dense_gpu_cooperative.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iostream>
#include <cmath>

namespace mx=mlx::core;
int main(){try {
    struct Tile{int sm,bk,sn;};const std::vector<Tile> tiles{{16,32,32},{16,32,64},{16,64,32},{32,32,32},{64,32,32}};
    int cases=0;float maximum=0;
    for(auto dtype:{mx::float16,mx::bfloat16})for(int m:{1,33,67,128})for(int k:{1,95,96,449,480})for(auto t:tiles)for(bool narrow:{false,true})for(bool registers:{false,true}) {
        constexpr int n=83,rb=7,cb=32;
        auto raw=mx::reshape(mx::astype(mx::sin(mx::arange(m*(k+64),mx::float32)*.011f)*.12f,dtype),{1,m,k+64});
        auto x=mx::slice(raw,{0,0,0},{1,m,k});
        auto w=mx::astype(mx::reshape(mx::cos(mx::arange((n+19)*(k+64),mx::float32)*.013f)*.025f,{n+19,k+64}),dtype);
        auto selected=mx::slice(w,{rb,cb},{rb+n,cb+k});
        auto expected=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(selected,mx::float32)));
        auto actual=tc::dense_gpu::projection_cooperative_range(x,w,rb,rb+n,cb,cb+k,!narrow,t.bk,t.sn,t.sm,registers);
        mx::eval({expected,actual});
        auto af=mx::astype(actual,mx::float32),ef=narrow?mx::astype(mx::astype(expected,dtype),mx::float32):expected;
        const float e=mx::sqrt(mx::sum(mx::square(af-ef))/mx::maximum(mx::sum(mx::square(ef)),mx::array(1e-20f))).item<float>();
        const float a=mx::max(mx::abs(af-ef)).item<float>();
        if(!std::isfinite(e) || e>(narrow?.003f:1e-5f) || a>(narrow?.0005f:2e-5f))throw std::runtime_error(
            "cooperative dense F32 oracle/typed boundary failed M="+std::to_string(m)+" K="+std::to_string(k)+
            " SM/BK/SN="+std::to_string(t.sm)+"/"+std::to_string(t.bk)+"/"+std::to_string(t.sn)+
            " dtype="+(dtype==mx::bfloat16?std::string("BF16"):std::string("FP16"))+" narrow="+std::to_string(narrow)+
            " rel="+std::to_string(e)+" abs="+std::to_string(a));
        maximum=std::max(maximum,e);++cases;
    }
    int rejected=0;
    auto reject=[&](auto &&fn) {
        try {fn();} catch(const std::invalid_argument &) {++rejected;return;}
        throw std::runtime_error("invalid cooperative dense contract accepted at index="+std::to_string(rejected));
    };
    auto x=mx::zeros({1,3,8},mx::bfloat16);
    auto w=mx::reshape(mx::astype(mx::arange(64,mx::float32),mx::bfloat16),{8,8});
    mx::eval({x,w});
    auto transposed=mx::transpose(w);mx::eval(transposed);
    auto project=[&](const mx::array &a,const mx::array &b,int rb=0,int re=8,int cb=0,int ce=8,
                     int bk=32,int sn=32,int sm=64) {
        return tc::dense_gpu::projection_cooperative_range(a,b,rb,re,cb,ce,true,bk,sn,sm);
    };
    reject([&]{project(mx::reshape(x,{3,8}),w);});
    reject([&]{project(mx::zeros({2,3,8},mx::bfloat16),w);});
    reject([&]{project(mx::zeros({1,0,8},mx::bfloat16),w);});
    reject([&]{project(x,mx::reshape(w,{1,8,8}));});
    reject([&]{project(mx::astype(x,mx::float32),mx::astype(w,mx::float32));});
    reject([&]{project(x,mx::astype(w,mx::float16));});
    reject([&]{project(x,transposed);});
    reject([&]{project(x,w,-1);});
    reject([&]{project(x,w,0,9);});
    reject([&]{project(x,w,0,0);});
    reject([&]{project(x,w,0,8,-1);});
    reject([&]{project(x,w,0,8,0,9);});
    reject([&]{project(x,w,0,8,0,0);});
    reject([&]{project(x,w,0,8,0,7);});
    reject([&]{project(x,w,0,8,0,8,16);});
    reject([&]{project(x,w,0,8,0,8,32,16);});
    reject([&]{project(x,w,0,8,0,8,64,64,16);});
    reject([&]{project(x,w,0,8,0,8,32,32,8);});
    reject([&]{project(x,w,0,8,0,8,64,32,32);});
    reject([&]{project(x,w,0,8,0,8,32,64,64);});
    std::cout<<"PASS cooperative dense cases="<<cases<<" maximum_relative_l2="<<maximum
             <<": original FP16/BF16 physical source, F32 accumulation, offsets/strided input/arbitrary K tails and5 tiles\n"
             <<"PASS cooperative dense rejected_contracts="<<rejected<<'\n';
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
