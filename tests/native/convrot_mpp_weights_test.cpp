#include "../../native/backends/mlx.hpp"
#include <iostream>

using namespace tc;
bool same(const Tensor &a,const Tensor &b) {
    return a.shape()==b.shape() && a.dtype()==b.dtype() && mx::all(mx::view(a,mx::uint8)==mx::view(b,mx::uint8)).item<bool>();
}
int main(){try {
    configure_streams();int cases=0;
    for(auto dtype:{mx::bfloat16,mx::float16})for(bool metal:{false,true})for(int rows:{33,67}) {
        auto codes=mx::reshape(mx::astype(mx::remainder(mx::arange(96*1024,mx::int32)*13,Tensor(256,mx::int32))-
            Tensor(128,mx::int32),mx::int8),{96,1024});
        auto scale=mx::reshape((mx::remainder(mx::arange(96,mx::float32),Tensor(7.f))+1.f)*.0002f,{96,1});
        Weights weights;weights.bind_arrays({"p.weight","p.weight_scale","p.comfy_quant"},{codes,scale,Tensor(1,mx::uint8)});
        weights.pack_convrot_q8(32,dtype);weights.set_metal_convrot(metal);weights.materialize();
        auto x=mx::astype(mx::reshape(mx::sin(mx::arange(rows*512,mx::float32)*.017f),{1,rows,512}),dtype);mx::eval(x);
        require(!weights.affine_fp32_mpp(),"MPP Weights profile changed default");
        auto original=weights.project_base_slice_fp32(x,"p",7,72,256,768);mx::eval(original);
        weights.set_affine_fp32_mpp(true);
        auto candidate=weights.project_base_slice_fp32(x,"p",7,72,256,768);
        auto range=weights.project_range_fp32(x,"p",7,72,256,768);mx::eval({candidate,range});
        const float error=mx::sqrt(mx::sum(mx::square(candidate-original))/mx::sum(mx::square(original))).item<float>();
        require(std::isfinite(error) && error<3e-6f && same(candidate,range),"MPP Weights F32 entry points changed source/geometry");
        weights.set_affine_fp32_mpp(false);
        auto restored=weights.project_base_slice_fp32(x,"p",7,72,256,768);mx::eval(restored);
        require(same(restored,original),"MPP original/candidate/original switch changed source");
        ++cases;
    }
    std::cout<<"PASS MPP Weights cases="<<cases<<": default off, original/candidate/original, two F32 APIs, same ConvRot rotation/basis and physical offsets\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
