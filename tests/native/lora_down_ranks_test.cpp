#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include <iostream>

using namespace tc;
float relative(const Tensor &a,const Tensor &b) {
    auto af=mx::astype(a,mx::float32),bf=mx::astype(b,mx::float32);
    return mx::sqrt(mx::sum(mx::square(af-bf))/mx::maximum(mx::sum(mx::square(bf)),Tensor(1e-20f))).item<float>();
}
int main(int argc,char **argv) {
    try {
        require(argc==2,"temporary down adapter path required");configure_streams();
        constexpr int h=128,f=1024,r=8;
        const std::string p="transformer_blocks.0.img_mlp.";
        int cases=0;
        for(auto dtype:{mx::float32,mx::float16,mx::bfloat16}) {
            Weights weights;
            weights.bind_arrays({p+"out.weight"},{mx::zeros({h,f},dtype)});
            auto a=mx::reshape(mx::sin(mx::arange(r*f,mx::float32)*.013f)*.05f,{r,f});
            auto b=mx::reshape(mx::cos(mx::arange(h*r,mx::float32)*.017f)*.03f,{h,r});
            mx::save_safetensors(argv[1],{{p+"out.lora_A.weight",a},{p+"out.lora_B.weight",b},{p+"out.alpha",Tensor(4.f,mx::float16)}});
            std::atomic<bool> cancelled{false};
            for(float strength:{.75f,-.25f})weights.apply_loras({{argv[1],strength,"transformer"}},"transformer",
                [](const std::string &,int,int){},cancelled,true);
            require(weights.lora_rank_count(p+"out")==2 && weights.lora_rank_width(p+"out")==16,"stacked down rank geometry");
            for(int rows:{1,17,67})for(int split:{256,640}) {
                auto input=mx::astype(mx::reshape(mx::cos(mx::arange(rows*f,mx::float32)*.003f)*.2f,{1,rows,f}),dtype);
                auto base=mx::astype(mx::reshape(mx::sin(mx::arange(rows*h,mx::float32)*.019f)*.1f,{1,rows,h}),dtype);
                auto head=slice_axis(input,-1,0,split),tail=slice_axis(input,-1,split,f);
                auto gr=weights.lora_input_ranks(head,p+"out",0,split),ar=weights.lora_input_ranks(tail,p+"out",split,f);
                std::vector<Tensor> joined;for(size_t i=0;i<gr.size();++i)joined.push_back(gr[i]+ar[i]);
                auto old=weights.lora_delta_slice(input,p+"out",0,h,0,f,mx::float32);
                auto fresh=weights.lora_delta_from_ranks(input.shape(),dtype,p+"out",joined,0,h,0,f,mx::float32);
                mx::eval({old,fresh});
                require(relative(fresh,old)<1e-5f,"split FP32 input rank reduction exceeded fixture tolerance");
                auto expected=qwen21::runtime_ffn::down_add(weights,p,h,f)({input,base})[0];
                auto prepare=qwen21::runtime_ffn::down_gpu_ranks(weights,p,split,f);
                auto gpu=prepare({head});std::vector<Tensor> args{tail,base};args.insert(args.end(),gpu.begin(),gpu.end());
                auto actual=qwen21::runtime_ffn::down_add_split_ranks(weights,p,split,h,f)(args)[0];
                mx::eval({expected,actual});
                require(relative(actual,expected)<.002f,"split down B/add changed output boundary excessively");
                auto bad=joined;bad.pop_back();bool rejected=false;
                try{weights.lora_delta_from_ranks(input.shape(),dtype,p+"out",bad,0,h,0,f);}catch(const std::exception&){rejected=true;}
                require(rejected,"metadata-only rank count mismatch admitted");
                rejected=false;auto wrong=input.shape();wrong.back()=f-1;
                try{weights.lora_delta_from_ranks(wrong,dtype,p+"out",joined,0,h,0,f);}catch(const std::exception&){rejected=true;}
                require(rejected,"metadata-only input range mismatch admitted");
                ++cases;
            }
        }
        std::cout<<"PASS down rank shards cases="<<cases<<" stacked +/- adapters, alpha, FP32/FP16/BF16, ONE B/delta boundary and metadata guards\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
