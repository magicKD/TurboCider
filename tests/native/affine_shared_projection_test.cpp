#include "../../native/backends/affine_gpu_shared.hpp"
#include <iostream>
#include <tuple>
#include <optional>
#include <vector>
namespace mx=mlx::core;
int main(){try {
    int cases=0,rejected=0;float worst=0;
    const std::vector<std::tuple<int,int,int>> recipes{{32,64,32},{64,64,64},{128,64,64},{64,128,64},{64,64,128},{64,128,128}};
    for(auto dtype:{mx::float16,mx::bfloat16})for(int bits:{4,8})for(int group:{32,64,128}) {
        auto physical=mx::astype(mx::random::normal({96,1024},mx::float32,mx::random::key(group+bits))*.1f,dtype);
        auto p=mx::quantize(physical,group,bits);
        auto decoded=mx::dequantize(p[0],p[1],p[2],group,bits,"affine",std::nullopt,dtype);
        for(int m:{33,67,128})for(int k:group==32?std::vector<int>{480,512}:std::vector<int>{512}) {
            constexpr int rb=7,re=72,cb=256;
            auto source=mx::astype(mx::random::normal(m==67?mx::Shape{k,m}:mx::Shape{m,k},mx::float32,mx::random::key(m+k))*.2f,dtype);
            auto x=mx::expand_dims(m==67?mx::transpose(source):source,0);
            auto w=mx::slice(decoded,{rb,cb},{re,cb+k});
            auto expected=mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(w,mx::float32)));mx::eval(expected);
            for(const auto &[bm,bn,bk]:recipes) {
              std::optional<mx::array> fp32;
              for(auto out:{mx::float32,dtype}) {
                auto got=tc::affine_gpu::projection_shared(x,p[0],p[1],p[2],bits,rb,re,cb,cb+k,bm,bn,bk,out);
                auto wanted=mx::astype(expected,out);mx::eval({got,wanted});
                auto gf=mx::astype(got,mx::float32),wf=mx::astype(wanted,mx::float32);
                const float rel=mx::sqrt(mx::sum(mx::square(gf-wf))/mx::sum(mx::square(wf))).item<float>();
                const float abs=mx::max(mx::abs(gf-wf)).item<float>();worst=std::max(worst,rel);
                bool narrow_ok=true;
                if(out!=mx::float32) {
                    auto cast=mx::astype(*fp32,out);mx::eval(cast);
                    const float epsilon=dtype==mx::bfloat16?1.f/128:1.f/1024;
                    narrow_ok=mx::all(mx::view(cast,mx::uint8)==mx::view(got,mx::uint8)).item<bool>() &&
                        mx::all(mx::abs(gf-wf)<=mx::array(2e-5f)+epsilon*mx::abs(wf)).item<bool>();
                }
                if(!mx::all(mx::isfinite(got)).item<bool>() || rel>(out==mx::float32?3e-6f:.003f) ||
                    (out==mx::float32?abs>2e-5f:!narrow_ok))
                    throw std::runtime_error("shared affine mismatch dtype="+std::string(dtype==mx::bfloat16?"bf16":"f16")+
                        " bits="+std::to_string(bits)+" group="+std::to_string(group)+" M="+std::to_string(m)+" K="+std::to_string(k)+
                        " tile="+std::to_string(bm)+"/"+std::to_string(bn)+"/"+std::to_string(bk)+
                        " output="+(out==mx::float32?std::string("f32"):std::string("narrow"))+" relative="+std::to_string(rel)+" abs="+std::to_string(abs));
                if(got.shape()!=wanted.shape() || got.dtype()!=out)throw std::runtime_error("shared affine shape/dtype changed");
                if(out==mx::float32)fp32=got;
                ++cases;
              }
            }
        }
        auto x=mx::zeros({1,33,512},dtype);
        for(int bad=0;bad<9;++bad) {
            bool denied=false;
            try{tc::affine_gpu::projection_shared(bad==0?mx::astype(x,mx::float32):x,p[0],p[1],p[2],bad==1?2:bits,
                bad==2?-1:7,72,bad==3?1:256,768,bad==4?16:64,bad==5?32:64,bad==6?16:64,
                bad==7?(dtype==mx::bfloat16?mx::float16:mx::bfloat16):bad==8?mx::int32:mx::float32);}
            catch(const std::invalid_argument&){denied=true;}
            if(!denied)throw std::runtime_error("invalid shared affine contract admitted");++rejected;
        }
    }
    std::cout<<"PASS shared affine cases="<<cases<<" rejected="<<rejected<<" maxrel="<<worst
        <<": original FP16/BF16 Q4/Q8, physical pitch/offsets, M/N/K tails, strided input, six TG tiles and narrow/F32 outputs\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
