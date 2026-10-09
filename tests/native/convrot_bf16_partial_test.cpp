#include "../../native/backends/mlx.hpp"
#include <iostream>

using namespace tc;
bool same(const Tensor &a,const Tensor &b) {
    return a.shape()==b.shape() && a.dtype()==b.dtype() && mx::all(mx::view(a,mx::uint8)==mx::view(b,mx::uint8)).item<bool>();
}
int main(int argc,char **argv){try {
    require(argc==2,"temporary adapter path required");configure_streams();int cases=0;
    for(int group:{32,64,128})for(bool metal:{false,true})for(int rows:{33,67,128}) {
        auto codes=mx::reshape(mx::astype(mx::remainder(mx::arange(96*1024,mx::int32)*13,Tensor(256,mx::int32))-
            Tensor(128,mx::int32),mx::int8),{96,1024});
        auto scale=mx::reshape((mx::remainder(mx::arange(96,mx::float32),Tensor(7.f))+1.f)*.0002f,{96,1});
        Weights weights;weights.bind_arrays({"p.weight","p.weight_scale","p.comfy_quant"},{codes,scale,Tensor(1,mx::uint8)});
        weights.pack_convrot_q8(group,mx::bfloat16);weights.set_metal_convrot(metal);weights.materialize();
        const auto master=weights.at("p.weight").id();
        auto x=mx::astype(mx::reshape(mx::sin(mx::arange(rows*512,mx::float32)*.017f),{1,rows,512}),mx::bfloat16);mx::eval(x);
        require(!weights.affine_bf16_fp32_partial(),"narrow partial changed default");
        auto original=weights.project_base_slice_fp32(x,"p",7,72,256,768);mx::eval(original);
        auto oracle=mx::astype(weights.project_base_slice(x,"p",7,72,256,768,false),mx::float32);mx::eval(oracle);
        weights.set_affine_bf16_fp32_partial(true);
        auto candidate=weights.project_base_slice_fp32(x,"p",7,72,256,768);
        auto range=weights.project_range_fp32(x,"p",7,72,256,768);mx::eval({candidate,range});
        require(same(candidate,oracle) && same(candidate,range) && candidate.dtype()==mx::float32,
                "narrow partial did not preserve original typed QMM boundary/F32 join geometry");
        const float error=mx::sqrt(mx::sum(mx::square(candidate-original))/mx::maximum(mx::sum(mx::square(original)),Tensor(1e-20f))).item<float>();
        require(mx::all(mx::isfinite(candidate)).item<bool>() && std::isfinite(error) && error<.01f,
                "narrow partial exceeded independent1% component boundary budget");
        bool rejected=false;try{weights.set_affine_fp32_mpp(true);}catch(const std::exception&){rejected=true;}
        require(rejected,"mixed partial recipes accepted");
        rejected=false;try{weights.project_base_slice_fp32(mx::astype(x,mx::float16),"p",7,72,256,768);}catch(const std::exception&){rejected=true;}
        require(rejected,"FP16 input silently narrowed under BF16 recipe");
        weights.set_affine_bf16_fp32_partial(false);
        auto restored=weights.project_base_slice_fp32(x,"p",7,72,256,768);mx::eval(restored);
        require(same(restored,original) && weights.at("p.weight").id()==master,"recipe switch modified original source/master");
        auto a=mx::reshape(mx::sin(mx::arange(8*1024,mx::float32)*.013f)*.04f,{8,1024});
        auto b=mx::reshape(mx::cos(mx::arange(96*8,mx::float32)*.017f)*.035f,{96,8});
        mx::save_safetensors(argv[1],{{"p.lora_A.weight",a},{"p.lora_B.weight",b},{"p.alpha",Tensor(4.f,mx::bfloat16)}});
        std::atomic<bool> cancelled{false};
        for(float strength:{.75f,-.25f})weights.apply_loras({{argv[1],strength,"transformer"}},"transformer",[](const std::string&,int,int){},cancelled,true);
        weights.set_affine_bf16_fp32_partial(true);
        auto base_only=weights.project_base_slice_fp32(x,"p",7,72,256,768);mx::eval(base_only);
        require(same(base_only,oracle),"GPU base partial consumed/rounded a per-shard down-LoRA");++cases;
    }
    std::cout<<"PASS BF16 partial cases="<<cases<<": original typed packed QMM/F32 widen, offsets/groups/rotation, explicit extra rounding budget, source immutable, +/- alpha, base-only despite adapters, flags/dtype guards\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
