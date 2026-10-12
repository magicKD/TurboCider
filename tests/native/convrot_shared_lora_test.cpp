#include "../../native/models/z_image/ffn.hpp"
#include <iostream>

using namespace tc;
bool same(const Tensor &a,const Tensor &b) {
    return a.dtype()==b.dtype() && a.shape()==b.shape() && mx::all(mx::view(a,mx::uint8)==mx::view(b,mx::uint8)).item<bool>();
}
int main(int argc,char **argv){try {
    require(argc==2,"temporary adapter path required");configure_streams();int cases=0;
    const std::string f="layers.0.feed_forward",q="layers.0.attention.qkv";
    for(auto dtype:{mx::float32,mx::float16,mx::bfloat16})for(bool metal:{false,true})
      for(int packing:{0,32,64})for(bool half_ranks:{false,true}) {
        std::vector<std::string> keys;std::vector<Tensor> arrays;
        auto add=[&](const std::string &key,Tensor value){keys.push_back(key);arrays.push_back(std::move(value));};
        for(const auto &p:{f+".w1",f+".w3",f+".w2",q}) {
            const int rows=p==q?768:p==f+".w2"?256:512,cols=p==f+".w2"?512:256;
            auto codes=mx::reshape(mx::astype(mx::remainder(mx::arange(rows*cols,mx::int32)*13,
                Tensor(256,mx::int32))-Tensor(128,mx::int32),mx::int8),{rows,cols});
            add(p+".weight",codes);add(p+".weight_scale",mx::full({rows,1},.0001f,mx::float32));
            add(p+".comfy_quant",Tensor(1,mx::uint8));add(p+".bias",mx::full({rows},.001f,mx::float32));
        }
        Weights weights;weights.bind_arrays(keys,arrays);weights.set_metal_convrot(metal);
        if(packing)weights.pack_convrot_q8(packing,dtype);
        weights.materialize();
        auto base=weights.at(f+".w1.weight"),base_copy=mx::copy(base);mx::eval(base_copy);
        std::unordered_map<std::string,Tensor> adapters;
        for(const auto &p:{f+".w1",f+".w3",f+".w2",std::string("layers.0.attention.to_q"),std::string("layers.0.attention.to_k")}) {
            const int rows=p==f+".w1" || p==f+".w3"?512:256,cols=p==f+".w2"?512:256;
            adapters.emplace(p+".lora_A.weight",mx::reshape(mx::sin(mx::arange(8*cols,mx::float32)*.013f)*.025f,{8,cols}));
            adapters.emplace(p+".lora_B.weight",mx::reshape(mx::cos(mx::arange(rows*8,mx::float32)*.017f)*.03f,{rows,8}));
            adapters.emplace(p+".alpha",Tensor(4.f,mx::bfloat16));
        }
        mx::save_safetensors(argv[1],adapters);std::atomic<bool> cancelled{false};
        for(float strength:{.75f,-.25f})weights.apply_loras({{argv[1],strength,"transformer"}},"transformer",
            [](const std::string&,int,int){},cancelled,true);
        weights.set_runtime_lora_fp16(half_ranks);
        require(weights.at(f+".w1.weight").id()==base.id() && same(base,base_copy),"runtime LoRA modified ConvRot master");
        for(int rows:{1,67}) {
            auto x=mx::astype(mx::reshape(mx::sin(mx::arange(rows*256,mx::float32)*.007f)*.3f,{1,rows,256}),dtype);
            auto separate=std::vector<Tensor>{weights.project(x,f+".w1"),weights.project(x,f+".w3"),weights.project(x,q)};
            auto shared=weights.project_many(x,{f+".w1",f+".w3",q});
            mx::eval(separate);mx::eval(shared);
            for(size_t i=0;i<shared.size();++i)require(same(shared[i],separate[i]),"shared ConvRot changed stacked LoRA/bias/QKV rounding");
            // Independent adapter arithmetic prevents a common helper defect
            // from making both single/shared paths agree while omitting LoRA.
            for(const auto &p:{f+".w1",f+".w3",q}) {
                auto wanted=weights.project_base_slice(x,p,0,weights.at(p+".weight").shape(0),0,256,false);
                const auto rd=half_ranks?mx::float16:mx::float32;
                for(float strength:{.75f,-.25f}) {
                    const std::vector<std::pair<std::string,int>> parts=p==q ?
                        std::vector<std::pair<std::string,int>>{{"layers.0.attention.to_k",256},{"layers.0.attention.to_q",0}} :
                        std::vector<std::pair<std::string,int>>{{p,0}};
                    for(const auto &[target,begin]:parts) {
                        auto a=mx::astype(adapters.at(target+".lora_A.weight"),rd),b=mx::astype(adapters.at(target+".lora_B.weight"),rd);
                        auto delta=mx::astype(mx::matmul(mx::matmul(mx::astype(x,rd),mx::transpose(a)),mx::transpose(b)),mx::float32)*Tensor(strength*.5f,mx::float32);
                        if(p!=q)wanted=mx::astype(mx::astype(wanted,mx::float32)+delta,dtype);
                        else {
                            auto middle=mx::astype(mx::astype(slice_axis(wanted,-1,begin,begin+256),mx::float32)+delta,dtype);
                            std::vector<Tensor> fragments;
                            if(begin)fragments.push_back(slice_axis(wanted,-1,0,begin));
                            fragments.push_back(middle);
                            if(begin+256<768)fragments.push_back(slice_axis(wanted,-1,begin+256,768));
                            wanted=mx::concatenate(fragments,-1);
                        }
                    }
                }
                wanted=wanted+mx::astype(weights.at(p+".bias"),dtype);
                require(same(weights.project(x,p),wanted),"common projection helper omitted/changed independently computed LoRA");
            }
            auto old=z_image::feed_forward(x,weights,f,false),fresh=z_image::feed_forward(x,weights,f,true);mx::eval({old,fresh});
            require(same(old,fresh),"shared ConvRot full LoRA FFN changed results");
            ++cases;
        }
    }
    std::cout<<"PASS shared ConvRot LoRA cases="<<cases<<": three dtypes, raw/packed, dense/Metal rotation, rank precision, stacked +/- alpha, bias, partial fused QKV, complete FFN, immutable masters\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
