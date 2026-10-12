#include "../../native/backends/mlx.hpp"
#include <iostream>
#include <limits>

using namespace tc;
// Performance-first experimental delta screen, not an equivalence promise or
// whole-model acceptance. Keep the raw maximum and require finite diagnostics.
constexpr float approximate_delta_budget=.05f;
bool acceptable_delta_error(float value) {
    return std::isfinite(value) && value>=0 && value<=approximate_delta_budget;
}
float rel(const Tensor &a,const Tensor &b) {
    auto af=mx::astype(a,mx::float32),bf=mx::astype(b,mx::float32);
    return mx::sqrt(mx::sum(mx::square(af-bf))/mx::maximum(mx::sum(mx::square(bf)),Tensor(1e-20f))).item<float>();
}
int main(int argc,char **argv){try {
    require(argc==2,"temporary adapter path required");configure_streams();int cases=0;float maximum_delta_error=0;
    require(acceptable_delta_error(.03f) && acceptable_delta_error(.05f) &&
        !acceptable_delta_error(.051f) && !acceptable_delta_error(-.01f) &&
        !acceptable_delta_error(std::numeric_limits<float>::infinity()) &&
        !acceptable_delta_error(std::numeric_limits<float>::quiet_NaN()),
        "approximate delta budget must not admit invalid/nonfinite diagnostics");
    constexpr int h=128,f=1024;const std::string p="transformer_blocks.0.img_mlp.out";
    for(auto dtype:{mx::float32,mx::float16,mx::bfloat16})for(auto storage:{mx::float32,mx::bfloat16})for(int rank:{8,64,256}) {
        auto a=mx::astype(mx::reshape(mx::sin(mx::arange(rank*f,mx::float32)*.013f)*.025f,{rank,f}),storage);
        auto b=mx::astype(mx::reshape(mx::cos(mx::arange(h*rank,mx::float32)*.017f)*.03f,{h,rank}),storage);
        mx::save_safetensors(argv[1],{{p+".lora_A.weight",a},{p+".lora_B.weight",b},{p+".alpha",Tensor(4.f,mx::bfloat16)}});
        Weights w;w.bind_arrays({p+".weight"},{mx::astype(mx::reshape(mx::cos(mx::arange(h*f,mx::float32)*.003f)*.04f,{h,f}),dtype)});w.materialize();
        std::atomic<bool> cancelled{false};for(float strength:{.75f,-.25f})w.apply_loras({{argv[1],strength,"transformer"}},"transformer",[](const std::string&,int,int){},cancelled,true);
        for(int rows:{1,33,131})for(const auto &range:{std::pair{0,f},std::pair{128,896}}) {
            const auto [cs,ce]=range;auto x=mx::astype(mx::reshape(mx::sin(mx::arange(rows*(ce-cs),mx::float32)*.007f)*.2f,{1,rows,ce-cs}),dtype);
            w.set_runtime_lora_b_epilogue(false);
            auto ranks=w.lora_input_ranks(x,p,cs,ce);
            auto original=w.project_slice(x,p,7,65,cs,ce,false),delta=w.lora_delta_slice(x,p,7,65,cs,ce,mx::float32);
            mx::eval(ranks);mx::eval({original,delta});
            w.set_runtime_lora_b_epilogue(true);
            auto prepared=w.lora_input_ranks(x,p,cs,ce);
            auto actual=w.project_slice(x,p,7,65,cs,ce,false),fresh=w.lora_delta_slice(x,p,7,65,cs,ce,mx::float32);
            auto shared=w.project_slice_with_ranks(x,p,prepared,7,65,cs,ce,false);
            mx::eval(prepared);mx::eval({actual,fresh,shared});
            for(size_t i=0;i<prepared.size();++i)require(prepared[i].dtype()==mx::float32 && rel(prepared[i],ranks[i])<1e-5f,"B epilogue changed original F32 A ranks");
            const auto projection_error=rel(actual,original),delta_error=rel(fresh,delta),shared_error=rel(shared,actual);
            // Approximate B operand screen; original F32 A ranks, stacked
            // delta aggregation and source/type/range checks remain intact.
            maximum_delta_error=std::max(maximum_delta_error,delta_error);
            require(projection_error<.05f && acceptable_delta_error(delta_error) && shared_error<1e-5f,
                "fused B boundary dtype="+std::string(dtype==mx::bfloat16?"bf16":dtype==mx::float16?"f16":"f32")+
                " storage="+std::string(storage==mx::bfloat16?"bf16":"f32")+" rank="+std::to_string(rank)+
                " rows="+std::to_string(rows)+" cs="+std::to_string(cs)+" projection="+std::to_string(projection_error)+
                " delta="+std::to_string(delta_error)+" shared="+std::to_string(shared_error));
            if(cs==0) {
                // A single intersecting adapter takes the direct output cast;
                // this differs from the stacked F32 aggregation above.
                Weights single;single.bind_arrays({p+".weight"},{w.at(p+".weight")});single.materialize();
                single.apply_loras({{argv[1],.75f,"transformer"}},"transformer",[](const std::string&,int,int){},cancelled,true);
                auto one=single.lora_delta_slice(x,p,7,65,cs,ce);mx::eval(one);
                single.set_runtime_lora_b_epilogue(true);auto one_new=single.lora_delta_slice(x,p,7,65,cs,ce);mx::eval(one_new);
                require(one_new.dtype()==dtype && acceptable_delta_error(rel(one_new,one)),"single B direct destination rounding exceeded approximate budget");
                w.set_runtime_lora_b_epilogue(false);auto full=w.project(x,p);mx::eval(full);
                w.set_runtime_lora_b_epilogue(true);auto full_new=w.project(x,p);mx::eval(full_new);
                require(rel(full_new,full)<.05f,"approximate B full projection exceeded5% component screen");
                // A rank borrowed from another leading shape never escapes
                // the original rank geometry checks.
                auto bad=prepared;bad[0]=mx::zeros({1,rows,rank},mx::float16);bool rejected=false;
                try{w.project_slice_with_ranks(x,p,bad,7,65,cs,ce,false);}catch(const std::exception&){rejected=true;}
                require(rejected,"FP16 rank array admitted by original F32 A-rank route");
            }
            ++cases;
        }
        w.set_runtime_lora_fp16(true);bool rejected=false;
        try{w.project_slice(mx::zeros({1,1,f},dtype),p,0,h,0,f,false);}catch(const std::exception&){rejected=true;}
        require(rejected,"incompatible FP16 ranks admitted by fused B route");
    }
    std::cout<<"PASS fused B runtime cases="<<cases<<" maximum_delta="<<maximum_delta_error<<": original F32 A ranks, stacked +/- alpha, sliced/shared/full projections and dtype/geometry/precision guards; approximate delta budget5%, not whole-model acceptance\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
