#include "../../native/backends/mlx.hpp"
#include <iostream>
using namespace tc;
float rel(const Tensor &a,const Tensor &b) {
    auto af=mx::astype(a,mx::float32),bf=mx::astype(b,mx::float32);
    return mx::sqrt(mx::sum(mx::square(af-bf))/mx::maximum(mx::sum(mx::square(bf)),Tensor(1e-20f))).item<float>();
}
int main(int argc,char **argv){try {
    require(argc==2,"temporary adapter path required");configure_streams();constexpr int h=128,f=1024;
    const std::string p="transformer_blocks.0.img_mlp.out";int cases=0;float maximum=0;
    for(auto storage:{mx::bfloat16,mx::float32})for(int rank:{64,256}) {
        auto a=mx::astype(mx::reshape(mx::sin(mx::arange(rank*f,mx::float32)*.013f)*.025f,{rank,f}),storage);
        auto b=mx::astype(mx::reshape(mx::cos(mx::arange(h*rank,mx::float32)*.017f)*.03f,{h,rank}),storage);
        mx::save_safetensors(argv[1],{{p+".lora_A.weight",a},{p+".lora_B.weight",b},{p+".alpha",Tensor(4.f,mx::bfloat16)}});
        Weights w;w.bind_arrays({p+".weight"},{mx::astype(mx::reshape(mx::cos(mx::arange(h*f,mx::float32)*.003f)*.04f,{h,f}),mx::bfloat16)});w.materialize();
        const auto source_id=w.at(p+".weight").id();std::atomic<bool> cancelled{false};
        for(float strength:{.75f,-.25f})w.apply_loras({{argv[1],strength,"transformer"}},"transformer",[](const std::string&,int,int){},cancelled,true);
        for(int m:{1,128,145})for(auto range:{std::pair{0,f},std::pair{128,896}}) {
            const auto [cs,ce]=range;auto x=mx::astype(mx::reshape(mx::sin(mx::arange(m*(ce-cs),mx::float32)*.007f)*.2f,{1,m,ce-cs}),mx::bfloat16);
            w.set_runtime_lora_bf16_fp32_ranks(false);w.set_runtime_lora_b_epilogue(false);
            auto original=w.project_slice(x,p,7,65,cs,ce,false),delta=w.lora_delta_slice(x,p,7,65,cs,ce,mx::float32);mx::eval({original,delta});
            auto oracle=mx::matmul(mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(slice_axis(a,1,cs,ce),mx::float32))),
                mx::transpose(mx::astype(slice_axis(b,0,7,65),mx::float32)))*Tensor(.5f*4.f/rank,mx::float32);mx::eval(oracle);
            require(rel(delta,oracle)<1e-5f,"independent original stacked alpha oracle failed");
            w.set_runtime_lora_bf16_fp32_ranks(true);w.set_runtime_lora_b_epilogue(true);
            auto ranks=w.lora_input_ranks(x,p,cs,ce);mx::eval(ranks);
            for(const auto &r:ranks)require(r.dtype()==mx::float32,"joint profile narrowed rank storage");
            auto actual=w.project_slice(x,p,7,65,cs,ce,false),fresh=w.lora_delta_slice(x,p,7,65,cs,ce,mx::float32);
            auto shared=w.project_slice_with_ranks(x,p,ranks,7,65,cs,ce,false);mx::eval({actual,fresh,shared});
            const float e=rel(fresh,oracle);maximum=std::max(maximum,e);
            require(std::isfinite(e) && e<=.05f && rel(actual,original)<.05f && rel(shared,actual)<1e-5f,"joint delta/projection/source-range budget exceeded");
            require(w.at(p+".weight").id()==source_id,"joint profile changed the master source");++cases;
        }
    }
    std::cout<<"PASS joint BF16 A/B cases="<<cases<<" maximum_delta="<<maximum<<": F32 ranks, stacked +/- BF16 alpha, independent original oracle, physical column slices, shared/full dtype fallbacks and immutable master\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
