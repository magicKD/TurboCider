#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include <iostream>

using namespace tc;
float rel(const Tensor &a,const Tensor &b) {
    return mx::sqrt(mx::sum(mx::square(mx::astype(a,mx::float32)-mx::astype(b,mx::float32)))/
        mx::maximum(mx::sum(mx::square(mx::astype(b,mx::float32))),Tensor(1e-20f))).item<float>();
}
int main(int argc,char **argv) {
    try {
        require(argc==2,"temporary adapter fixture path required");configure_streams();
        for(auto dtype:{mx::float32,mx::float16,mx::bfloat16})for(bool rank_half:{false,true}) {
            constexpr int h=128,f=512;
            Weights weights;const std::string p="transformer_blocks.0.img_mlp.";
            auto gate=mx::astype(mx::reshape(mx::sin(mx::arange(2*f*h,mx::float32)*.007f)*.04f,{2*f,h}),dtype);
            auto down=mx::astype(mx::reshape(mx::cos(mx::arange(h*f,mx::float32)*.003f)*.04f,{h,f}),dtype);
            weights.bind_arrays({p+"gate_up.weight",p+"out.weight"},{gate,down});weights.materialize();
            auto a=mx::reshape(mx::sin(mx::arange(8*h,mx::float32)*.013f)*.05f,{8,h});
            auto b=mx::reshape(mx::cos(mx::arange(f*8,mx::float32)*.017f)*.04f,{f,8});
            mx::save_safetensors(argv[1],{{p+"gate_layer.lora_A.weight",a},
                {p+"gate_layer.lora_B.weight",b},
                {p+"gate_layer.alpha",Tensor(4.f,mx::float16)},
                {p+"proj.lora_A.weight",a*.75f},
                {p+"proj.lora_B.weight",b*.5f},
                {p+"proj.alpha",Tensor(12.f,mx::bfloat16)}});
            std::atomic<bool> cancelled{false};
            for(float strength:{.75f,-.25f})weights.apply_loras({{argv[1],strength,"transformer"}},"transformer",
                [](const std::string &,int,int){},cancelled,true);
            weights.set_runtime_lora_fp16(rank_half);
            auto input=mx::astype(mx::reshape(mx::cos(mx::arange(34*h,mx::float32)*.01f)*.25f,{1,34,h}),dtype);
            for(auto columns:{std::pair{0,64},std::pair{32,96},std::pair{0,128}}) {
                const auto [cs,ce]=columns;
                auto x=slice_axis(input,-1,cs,ce);auto ranks=weights.lora_input_ranks(x,p+"gate_up",cs,ce);
                require(ranks.size()==4,"stacked gate/up rank count mismatch");
                for(auto range:{std::pair{0,128},std::pair{384,640},std::pair{768,1024}}) {
                    auto expected=weights.project_slice(x,p+"gate_up",range.first,range.second,cs,ce,false);
                    auto actual=weights.project_slice_with_ranks(x,p+"gate_up",ranks,range.first,range.second,cs,ce,false);
                    auto delta=weights.lora_delta_slice(x,p+"gate_up",range.first,range.second,cs,ce);
                    auto shared=weights.lora_delta_slice_with_ranks(x,p+"gate_up",ranks,range.first,range.second,cs,ce);
                    mx::eval({expected,actual,delta,shared});
                    require(rel(actual,expected)<1e-6f && rel(shared,delta)<1e-6f,"shared slice changed LoRA arithmetic");
                }
                bool rejected=false;ranks.pop_back();
                try{weights.lora_delta_slice_with_ranks(x,p+"gate_up",ranks,0,128,cs,ce);}catch(const std::exception&){rejected=true;}
                require(rejected,"shared adapter rank count mismatch admitted");
                for(bool bad_dtype:{false,true}) {
                    auto invalid=weights.lora_input_ranks(x,p+"gate_up",cs,ce);
                    for(auto &rank:invalid)rank=bad_dtype ? mx::astype(rank,rank_half?mx::float32:mx::float16) :
                        slice_axis(rank,-1,0,4);
                    rejected=false;
                    try{weights.project_slice_with_ranks(x,p+"gate_up",invalid,384,640,cs,ce);}
                    catch(const std::exception&){rejected=true;}
                    require(rejected,"shared rank shape/dtype mismatch admitted");
                }
            }
            using namespace qwen21::runtime_ffn;
            auto original=channels(weights,p,0,256,h,f,false);
            auto prepared=channels(weights,p,0,256,h,f,false,true);
            auto rank_graph=input_ranks(weights,p,h);auto ranks=rank_graph({input});
            std::vector<Tensor> args{input};args.insert(args.end(),ranks.begin(),ranks.end());
            auto expected=original({input}),actual=prepared(args);
            auto delta=corrections(weights,p,256,256,h,f)({input});
            auto shared=corrections_shared_ranks(weights,p,256,256,h,f)(args);
            mx::eval({expected[0],expected[1],actual[0],actual[1],delta[0],delta[1],shared[0],shared[1]});
            require(rel(actual[0],expected[0])<.002f && rel(actual[1],expected[1])<.002f &&
                rel(shared[0],delta[0])<.002f && rel(shared[1],delta[1])<.002f,"shared compiled head/corrections mismatch");
            std::cout<<"PASS shared LoRA ranks dtype="<<(dtype==mx::float32?"f32":dtype==mx::float16?"f16":"bf16")
                <<" half="<<rank_half<<" adapters="<<ranks.size()<<'\n';
        }
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
