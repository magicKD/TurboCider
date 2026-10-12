#include "../../native/backends/ane_ffn.hpp"
#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include <iostream>

using namespace tc;
void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}
float rel(const Tensor &a,const Tensor &b) {
    auto af=mx::astype(a,mx::float32),bf=mx::astype(b,mx::float32);
    return mx::sqrt(mx::sum(mx::square(af-bf))/mx::maximum(mx::sum(mx::square(bf)),Tensor(1e-20f))).item<float>();
}
int main(int argc,char **argv) {try {
    check(argc==3,"manifest and temporary adapter path required");configure_streams();
    constexpr int h=128,f=1024,fg=512;
    const std::string p="transformer_blocks.0.img_mlp.";
    auto wg=mx::astype(mx::reshape(mx::sin(mx::arange(2*f*h,mx::float32)*.005f)*.04f,{2*f,h}),mx::bfloat16);
    auto wd=mx::astype(mx::reshape(mx::cos(mx::arange(h*f,mx::float32)*.003f)*.04f,{h,f}),mx::bfloat16);
    Weights weights;weights.bind_arrays({p+"gate_up.weight",p+"out.weight"},{wg,wd});weights.materialize();
    auto a=mx::reshape(mx::sin(mx::arange(8*f,mx::float32)*.017f)*.04f,{8,f});
    auto b=mx::reshape(mx::cos(mx::arange(h*8,mx::float32)*.013f)*.05f,{h,8});
    mx::save_safetensors(argv[2],{{p+"out.lora_A.weight",a},{p+"out.lora_B.weight",b}});
    std::atomic<bool> cancelled{false};weights.apply_loras({{argv[2],.75f,"transformer"}},"transformer",
        [](const std::string&,int,int){},cancelled,true);
    auto gu=mx::split(wg,2,0);mx::eval(gu);
    auto full=qwen21::runtime_ffn::full(weights,p);
    // The synthetic full graph factory uses production 4096 defaults only in
    // down helpers; gate/up projection dimensions are inferred from weights.
    auto head=qwen21::runtime_ffn::channels(weights,p,0,fg,h,f,true);
    auto old_down=qwen21::runtime_ffn::down_add(weights,p,h,f);
    auto prepare=qwen21::runtime_ffn::down_gpu_ranks(weights,p,fg,f);
    auto finish=qwen21::runtime_ffn::down_add_split_ranks(weights,p,fg,h,f);
    auto run=[&](ane::HybridFfn &op,int rows,const Tensor &input,int fault,bool split,int &fallbacks) {
        ane::HybridFfn::Adapter adapter{
            [&](const Tensor &x){return std::make_pair(mx::zeros({1,x.shape(1),f},x.dtype()),mx::zeros({1,x.shape(1),f},x.dtype()));},
            [&](const Tensor &hidden,const Tensor &base){return old_down({hidden,base})[0];},
            [&](const Tensor &x,int,int count){return std::make_pair(mx::zeros({1,x.shape(1),count},x.dtype()),mx::zeros({1,x.shape(1),count},x.dtype()));}};
        if(split)adapter.channel_down_ranks=ane::HybridFfn::Adapter::ChannelDownRanks{8*3*sizeof(float),
            [&](const Tensor &hidden) {
                if(fault==1)throw std::runtime_error("prepare-original-error");
                if(fault==3)return std::vector<Tensor>{mx::zeros({1,hidden.shape(1),8},mx::float16)};
                return prepare({hidden});
            },
            [&](const Tensor &tail,const std::vector<Tensor> &ranks,const Tensor &base) {
                if(fault==2)throw std::runtime_error("finish-original-error");
                std::vector<Tensor> args{tail,base};args.insert(args.end(),ranks.begin(),ranks.end());return finish(args)[0];
            }};
        op.begin_request();op.plan_block(0,rows);op.stage(0,rows,{gu[0],gu[1],wd});
        return op.run(0,input,[&](const Tensor &x){++fallbacks;return fault==4?mx::full(x.shape(),7.f,x.dtype()):full({x})[0];},
            cancelled,&adapter,[&](const Tensor &x,int,int){auto out=head({x});return std::make_pair(out[0],out[1]);});
    };
    for(int rows:{17,67}) {
        auto input=mx::astype(mx::reshape(mx::sin(mx::arange(rows*h,mx::float32)*.01f)*.2f,{1,rows,h}),mx::bfloat16);
        ane::HybridFfn op(argv[1],h,f,512u<<20,cancelled,true);
        int fallbacks=0;auto old=run(op,rows,input,0,false,fallbacks);auto fresh=run(op,rows,input,0,true,fallbacks);
        mx::eval({old,fresh});
        check(rel(fresh,old)<.002f && !fallbacks && op.metrics().runtime_weight_down_rank_blocks==1 &&
              op.metrics().runtime_weight_down_rank_arrays==1,"actual split down-rank output/receipt mismatch");
        for(int fault:{1,2,3}) {
            const auto before=op.metrics().runtime_weight_down_rank_blocks;bool rejected=false;
            try{run(op,rows,input,fault,true,fallbacks);}catch(const std::exception &error) {
                rejected=true;if(fault<3)check(std::string(error.what())==(fault==1?"prepare-original-error":"finish-original-error"),"cleanup replaced callback error");
            }
            check(rejected && op.metrics().runtime_weight_down_rank_blocks==before,"failed down-rank callback counted successful");
            auto recovered=run(op,rows,input,0,true,fallbacks);mx::eval(recovered);
            check(rel(recovered,old)<.002f && op.available(),"down-rank callback failure poisoned next source/reader");
        }
    }
    {
        constexpr int rows=67;std::vector<float> data(rows*h,.1f);data[50*h]=INFINITY;
        auto input=mx::astype(Tensor(data.data(),{1,rows,h},mx::float32),mx::bfloat16);
        ane::HybridFfn op(argv[1],h,f,512u<<20,cancelled,true);int fallbacks=0;
        auto result=run(op,rows,input,4,true,fallbacks);mx::eval(result);
        check(fallbacks==1 && mx::all(result==Tensor(7.f,mx::bfloat16)).item<bool>() &&
              op.metrics().runtime_weight_down_rank_blocks==0,"late ANE failure published split partials instead of complete GPU result");
    }
    std::cout<<"PASS down-rank channel: actual Private, rows/tails, original/split, counted completion, prepare/finish/geometry failure recovery and late whole-GPU fallback\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
