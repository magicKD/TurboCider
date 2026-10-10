#include "models/qwen21/metal/encoder_prefill.hpp"
#include "models/qwen21/encoder_prefill.hpp"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>

using namespace tc;
namespace {
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception &){failed=true;}require(failed,"invalid encoder prefill input accepted");}
float relative(const Tensor &a,const Tensor &b){auto x=mx::astype(a,mx::float32),y=mx::astype(b,mx::float32);return mx::sqrt(mx::sum(mx::square(x-y))/mx::maximum(mx::sum(mx::square(y)),Tensor(1e-20f))).item<float>();}
Tensor norm(const Tensor &x,const Tensor &w){auto f=mx::astype(x,mx::float32);return mx::astype(mx::astype(w,mx::float32)*(f*mx::rsqrt(mx::mean(f*f,-1,true)+1e-6f)),x.dtype());}
Tensor rope(const Tensor &x,const Tensor &c,const Tensor &s){auto halves=mx::split(x,2,-1);return x*c+mx::concatenate({-halves[1],halves[0]},-1)*s;}
}
int main(int argc,char **argv){try{
    require(argc==2,"owned processor fixture required");configure_streams();mx::set_cache_limit(0);std::atomic<bool> cancel{false};
    int cases=0;
    for(auto dtype:{mx::float16,mx::bfloat16})for(int m:{1,38,145,513}) {
        auto x=mx::astype(mx::reshape(mx::sin(mx::arange(m*4096,mx::float32)*.013f),{1,m,4096}),dtype);
        auto w=mx::astype(mx::cos(mx::arange(4096,mx::float32)*.03f)*.1f+1.f,dtype);
        auto expected=norm(x,w),actual=qwen21::metal::encoder_rms(x,w,1e-6f);mx::eval({actual,expected});
        require(relative(actual,expected)<.003 && mx::all(mx::isfinite(actual)).item<bool>(),"fused encoder RMS changed typed/F32 contract");++cases;
        auto q=x,k=mx::astype(mx::reshape(mx::cos(mx::arange(m*1024,mx::float32)*.019f),{1,m,1024}),dtype);
        auto qw=mx::astype(mx::cos(mx::arange(128,mx::float32)*.07f)*.05f+1.f,dtype),kw=qw;
        auto a=mx::reshape(mx::arange(m*64,mx::float32)*.017f,{1,1,m,64});a=mx::concatenate({a,a},-1);
        auto c=mx::astype(mx::cos(a),dtype),s=mx::astype(mx::sin(a),dtype);
        auto pair=qwen21::metal::encoder_qk(q,k,qw,kw,c,s,1e-6f);
        auto eq=rope(norm(heads(q,32,128),qw),c,s),ek=rope(norm(heads(k,8,128),kw),c,s);mx::eval({pair[0],pair[1],eq,ek});
        require(pair[0].shape()==eq.shape() && pair[1].shape()==ek.shape() &&
            relative(pair[0],eq)<.006 && relative(pair[1],ek)<.006,"fused GQA Q/K norm/NeoX RoPE differs");cases+=2;
        auto packed=mx::concatenate({q,k,k},-1);auto slices=mx::split(packed,mx::Shape{4096,5120},-1);
        auto strided=qwen21::metal::encoder_qk(slices[0],slices[1],qw,kw,c,s,1e-6f);mx::eval(strided);
        require(relative(strided[0],eq)<.006 && relative(strided[1],ek)<.006,"fused encoder Q/K misread fused projection row pitch");cases+=2;
        rejects([&]{qwen21::metal::encoder_qk(q,q,qw,kw,c,s,1e-6f);});
        rejects([&]{qwen21::metal::encoder_rms(x,w,1e-5f);});
    }
    Weights original;std::vector<std::string> keys;std::vector<Tensor> arrays;
    for(int layer=0;layer<36;++layer)for(const auto &suffix:{"self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"}) {
        const int rows=std::string(suffix).ends_with("q_proj") ? 32 : std::string(suffix).starts_with("mlp") ? 64 : 16;
        auto dense=mx::astype(mx::reshape(mx::sin(mx::arange(rows*64,mx::float32)*.017f+float(layer)*.03f),{rows,64}),mx::float16);
        auto planes=mx::quantize(dense,32,(layer<18 && std::string(suffix)=="self_attn.v_proj") ? 8 : 4);const auto p="model.layers."+std::to_string(layer)+"."+suffix;
        for(size_t i=0;i<3;++i){keys.push_back(p+(i==0?".weight":i==1?".scales":".biases"));arrays.push_back(planes[i]);}
    }
    original.bind_arrays(keys,arrays);original.materialize();Weights fused=original;
    auto pack=qwen21::pack_encoder_prefill(fused,cancel);require(pack.qkv==18 && pack.qk==18 && pack.gate_up==36,"mixed-bit fused packed source coverage incomplete");
    for(int layer:{0,17,35}) {
        const auto p="model.layers."+std::to_string(layer)+".";
        for(const auto &definition:{std::make_pair(layer<18 ? "self_attn.qk_proj" : "self_attn.qkv_proj",layer<18 ? std::vector<std::string>{"self_attn.q_proj","self_attn.k_proj"} : std::vector<std::string>{"self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"}),
            std::make_pair("mlp.gate_up",std::vector<std::string>{"mlp.gate_proj","mlp.up_proj"})}) {
            for(const auto &plane:{".weight",".scales",".biases"}) {
                std::vector<Tensor> pieces;for(const auto &name:definition.second)pieces.push_back(original.at(p+name+plane));
                auto expected=mx::concatenate(pieces,0);mx::eval(expected);const auto &actual=fused.at(p+definition.first+plane);
                require(mx::all(actual==expected).item<bool>(),"fusion changed packed codes/scales/biases row order");++cases;
            }
            auto x=mx::ones({1,38,64},mx::float16);std::vector<Tensor> pieces;for(const auto &name:definition.second)pieces.push_back(linear(x,original,p+name));
            auto expected=mx::concatenate(pieces,-1),actual=linear(x,fused,p+definition.first);mx::eval({expected,actual});
            require(relative(actual,expected)<.003,"packed fused projection differs from original source");++cases;
        }
    }
    auto biased_keys=keys;auto biased_arrays=arrays;
    biased_keys.push_back("model.layers.0.self_attn.q_proj.bias");biased_arrays.push_back(mx::ones({32},mx::float16));
    Weights biased;biased.bind_arrays(biased_keys,biased_arrays);biased.materialize();
    const auto biased_pack=qwen21::pack_encoder_prefill(biased,cancel);
    require(biased_pack.qkv==18 && biased_pack.qk==17 && biased.has("model.layers.0.self_attn.q_proj.bias") &&
        !biased.has("model.layers.0.self_attn.qkv_proj.weight") && !biased.has("model.layers.0.self_attn.qk_proj.weight"),
        "prefill fusion silently discarded an incompatible additive bias");
    cancel.store(true);rejects([&]{qwen21::pack_encoder_prefill(original,cancel);});cancel.store(false);
    const auto folder=std::filesystem::canonical(argv[1]);
    require(folder.filename().string().starts_with("tc-qwen-prefill-"),"refusing to mutate non-owned processor fixture");
    qwen21::VerifiedEncoderTokenizer tokenizer(folder,cancel);require(tokenizer.tokenizer().raw("abcabc").ids==std::vector<int>{5,5},"held-fd tokenizer differs");
    tokenizer.check_unchanged();const int fd=::open((folder/"tokenizer.json").c_str(),O_WRONLY|O_CLOEXEC);require(fd>=0,"owned tokenizer open failed");
    const int status=::ftruncate(fd,0);::close(fd);require(status==0,"owned tokenizer truncate failed");rejects([&]{tokenizer.check_unchanged();});
    std::cout<<"PASS encoder GPU prefill: "<<cases<<" actual Metal typed RMS/NeoX/GQA and packed field/projection cases; geometry/epsilon/tokenizer-generation negatives\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
