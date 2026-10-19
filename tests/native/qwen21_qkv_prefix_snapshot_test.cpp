#include "../../native/models/qwen21/transformer.hpp"
#include "../../tools/native/qwen21_qkv_numeric_prepare.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>

namespace {
using namespace tc;
#if defined(TC_QKV_PREFIX_COUNTEREXAMPLE)
constexpr int hidden=256, rows=128, height=8, width=16;
#else
constexpr int hidden=4096, rows=1024, height=32, width=32;
#endif
#if defined(TC_LORA_SWIGLU_PREFIX_FIXTURE)
constexpr int ffn=12288;constexpr bool swiglu=true;
#else
constexpr int ffn=256;constexpr bool swiglu=false;
#endif
#if defined(TC_LORA_SWIGLU_ORDINARY_QKV)
constexpr bool direct_qkv=false;
#else
constexpr bool direct_qkv=true;
#endif
constexpr int depth=2, channels=16, context=32, heads_count=hidden/128;
size_t mpp_mismatches=0,native_base_mismatches=0;
Tensor matrix(int out,int in,int seed) {
    return mx::astype(mx::random::normal({out,in},mx::float32,mx::random::key(seed))/std::sqrt(float(in)),mx::bfloat16);
}
qwen21::TransformerConfig config() {
    qwen21::TransformerConfig c;c.layers=depth;c.heads=heads_count;c.head_dim=128;c.channels=channels;c.context_dim=context;return c;
}
void bind(Weights &w,const std::filesystem::path &adapter) {
    std::vector<std::string> names;std::vector<Tensor> arrays;int seed=7100;
    auto mat=[&](const std::string &p,int out,int in){names.push_back(p+".weight");arrays.push_back(matrix(out,in,seed++));};
    auto vec=[&](const std::string &p,int width,float value){names.push_back(p+".weight");arrays.push_back(mx::full({width},value,mx::bfloat16));};
    mat("time_text_embed.timestep_embedder.linear_1",hidden,256);mat("time_text_embed.timestep_embedder.linear_2",hidden,hidden);
    mat("modulation.1",4*hidden,hidden);mat("img_in",hidden,channels);vec("txt_in.text_norm",context,0.f);
    mat("txt_in.in_layer",hidden,context);mat("txt_in.out_layer",hidden,hidden);
    mat("norm_out.linear",hidden,hidden);mat("proj_out",channels,hidden);
    std::unordered_map<std::string,Tensor> lora;
    for(int i=0;i<depth;++i){const auto p="transformer_blocks."+std::to_string(i);
        for(const char *name:{"to_q","to_k","to_v","to_out.0"})mat(p+".attn."+name,hidden,hidden);
        vec(p+".attn.norm_q",128,1.f);vec(p+".attn.norm_k",128,1.f);
        mat(p+".img_mlp.gate_up",2*ffn,hidden);mat(p+".img_mlp.out",hidden,ffn);
        for(const char *name:{"to_q","to_k","to_v"}){
            const auto stem=p+".attn."+name;
            lora.emplace(stem+".lora_A.weight",matrix(128,hidden,seed++)*Tensor(.1f,mx::bfloat16));
            lora.emplace(stem+".lora_B.weight",matrix(hidden,128,seed++)*Tensor(.1f,mx::bfloat16));
        }
        if(swiglu)for(const char *name:{"gate_layer","proj","out"}){
            const auto stem=p+".img_mlp."+name;const bool down=std::string_view(name)=="out";
            lora.emplace(stem+".lora_A.weight",matrix(128,down?ffn:hidden,seed++)*Tensor(.1f,mx::bfloat16));
            lora.emplace(stem+".lora_B.weight",matrix(down?hidden:ffn,128,seed++)*Tensor(.1f,mx::bfloat16));
        }
    }
    mx::save_safetensors(adapter.string(),lora);w.bind_arrays(names,arrays);
    std::atomic<bool> cancelled{false};
    require(w.apply_loras({{adapter.string(),1.f,"transformer"}},"transformer",[](const auto&,int,int){},cancelled,true)==(swiglu?6:3)*depth,"tiny complete adapter binding differs");
    w.set_runtime_lora_bf16_fp32_ranks(true);w.set_runtime_lora_b_epilogue(true);w.materialize();
}
std::vector<float> values(const Tensor &x) {
    auto f=mx::contiguous(mx::astype(x,mx::float32));mx::eval(f);require(mx::all(mx::isfinite(f)).item<bool>(),"nonfinite tiny QKV output");
    return {f.data<float>(),f.data<float>()+f.size()};
}
bool same_bits(const Tensor &a,const Tensor &b) {
    require(a.shape()==b.shape()&&a.dtype()==b.dtype(),"tensor geometry/dtype differs");
    auto x=mx::view(mx::contiguous(a),mx::uint8),y=mx::view(mx::contiguous(b),mx::uint8);mx::eval(x,y);
    return std::equal(x.data<uint8_t>(),x.data<uint8_t>()+x.size(),y.data<uint8_t>());
}
Tensor rotate(const Tensor &x,const Tensor &c,const Tensor &s) {
    auto paired=mx::reshape(mx::astype(x,mx::float32),{1,heads_count,x.shape(2),64,2});
    auto a=mx::squeeze(slice_axis(paired,-1,0,1),-1),b=mx::squeeze(slice_axis(paired,-1,1,2),-1);
    auto co=mx::reshape(c,{1,1,x.shape(2),64}),si=mx::reshape(s,{1,1,x.shape(2),64});
    return mx::astype(mx::reshape(mx::stack({a*co-b*si,a*si+b*co},-1),x.shape()),x.dtype());
}
[[maybe_unused]] void diagnose(const qwen21::DiagnosticBlockObservation &o) {
    if(o.layer!=0)return;
    const auto p="transformer_blocks.0.attn.";const auto &w=o.weights;
    auto x=mx::fast::layer_norm(o.arguments[0],{},{},1e-6f)*(Tensor(1.f,mx::bfloat16)+o.arguments[1]);mx::eval(x);
    std::array<Tensor,3> ranks{Tensor(0.f),Tensor(0.f),Tensor(0.f)},up=ranks;
    std::array<float,3> scales{};std::vector<Tensor> bases;
    auto snapshot=w.runtime_lora_snapshot();
    for(int i=0;i<3;++i){const auto name=std::string(p)+(i==0?"to_q":i==1?"to_k":"to_v");
        for(const auto &b:snapshot.bindings)if(b.prefix==name){up[i]=b.ordered_terms[0].up;scales[i]=b.ordered_terms[0].scale;}
        ranks[i]=w.lora_input_ranks(x,name,0,hidden)[0];bases.push_back(w.at(name+".weight"));
    }
    auto packed=mx::concatenate(bases,0);mx::eval(packed);
    auto reference=mx::compile([&w,p](const std::vector<Tensor> &a){
        std::vector<Tensor> base,corrected,normed,roped;
        for(int i=0;i<3;++i){const auto name=std::string(p)+(i==0?"to_q":i==1?"to_k":"to_v");
            auto b=heads(mx::matmul(a[0],mx::transpose(w.at(name+".weight"))),heads_count,128);
            auto y=heads(linear(a[0],w,name),heads_count,128);
            auto n=i<2?mx::fast::rms_norm(y,w.at(std::string(p)+(i==0?"norm_q.weight":"norm_k.weight")),1e-6f):y;
            base.push_back(b);corrected.push_back(y);normed.push_back(n);roped.push_back(i<2?rotate(n,a[1],a[2]):n);
        }
        base.insert(base.end(),corrected.begin(),corrected.end());base.insert(base.end(),normed.begin(),normed.end());base.insert(base.end(),roped.begin(),roped.end());return base;
    });
    auto expected=reference({x,o.arguments[5],o.arguments[6]});mx::eval(expected);
    std::array<Tensor,3> native_base{mx::matmul(x,mx::transpose(bases[0])),mx::matmul(x,mx::transpose(bases[1])),mx::matmul(x,mx::transpose(bases[2]))};
    mx::eval(native_base[0],native_base[1],native_base[2]);
    for(bool supplied:{false,true})for(int stage=0;stage<4;++stage){auto actual=qwen21::qkv_numeric_lab::prepare_stage(x,packed,ranks,up,scales,
        w.at(std::string(p)+"norm_q.weight"),w.at(std::string(p)+"norm_k.weight"),o.arguments[5],o.arguments[6],stage,32,supplied?&native_base:nullptr,1);
        for(int i=0;i<3;++i){auto a=values(expected[stage*3+i]),b=values(actual[i]);size_t mismatch=0;
            for(size_t j=0;j<a.size();++j)mismatch+=a[j]!=b[j];
            if(supplied)native_base_mismatches+=mismatch;else mpp_mismatches+=mismatch;
            std::cerr<<"native_base="<<supplied<<" stage="<<stage<<" projection="<<i<<" mismatches="<<mismatch<<'\n';}
    }
}
void run(const std::filesystem::path &directory) {
    configure_streams();Weights weights;bind(weights,directory/"adapter.safetensors");
    auto text=mx::reshape(matrix(4,context,7300),{1,4,context});
    auto noise=mx::reshape(matrix(rows,channels,7301),{1,rows,channels});
    std::vector<qwen21::ReferenceLatents> refs{{mx::reshape(matrix(16,channels,7302),{1,16,channels}),{4,4,1}}};
    mx::eval(text,noise,refs[0].latents);
    std::optional<qwen21::Transformer::PrefixSnapshot> snapshot;
    const auto cap=std::numeric_limits<uint64_t>::max();
    {
        qwen21::Transformer source(weights,config(),nullptr,nullptr,direct_qkv,swiglu);
        qwen21::Transformer original(weights,config());
        require(same_bits(source.forward(noise,text,1.f,height,width,true,nullptr,refs),original.forward(noise,text,1.f,height,width,true,nullptr,refs)),"fusion changed ordinary prefill");
        require(!source.export_prefix_snapshot(cap),"direct prefix export became implicit");
        require(!source.export_prefix_snapshot(cap,true,direct_qkv,swiglu),"fused/DB snapshot combination admitted");
        snapshot=source.export_prefix_snapshot(cap,false,direct_qkv,swiglu);require(snapshot.has_value(),"explicit fused K/V export failed");
        require(!source.export_prefix_snapshot(snapshot->bytes-1,false,direct_qkv,swiglu),"fused snapshot exceeded cap");
        source.set_decode_mlp([](int,const Tensor &x){return x;});
        require(!source.export_prefix_snapshot(cap,false,direct_qkv,swiglu),"live callback bypassed fused snapshot guard");
    }
    // A snapshot must not keep the source Transformer's compiled blocks alive.
    // New receivers exercise target-only step zero, multiple decode steps and
    // changed noise, with independent original-prefix arithmetic as oracle.
    for(int seed=0;seed<2;++seed){
        qwen21::Transformer direct(weights,config(),nullptr,nullptr,direct_qkv,swiglu),original(weights,config());
        require(!direct.import_prefix_snapshot(*snapshot,text,height,width,refs),"direct prefix import became implicit");
        auto bad=*snapshot;bad.bytes+=1;
        require(!direct.import_prefix_snapshot(bad,text,height,width,refs,direct_qkv,swiglu)&&direct.cached_layers()==0,"failed import mutated receiver");
        require(direct.import_prefix_snapshot(*snapshot,text,height,width,refs,direct_qkv,swiglu),"explicit fused K/V import failed");
        require(original.import_prefix_snapshot(*snapshot,text,height,width,refs),"ordinary prefix control import failed");
#if defined(TC_QKV_PREFIX_COUNTEREXAMPLE)
        original.set_diagnostic_block_observer(diagnose);
#endif
        auto a=mx::reshape(matrix(rows,channels,7400+seed),{1,rows,channels});auto b=a;
        for(int step=0;step<3;++step){
            const float sigma=1.f-.25f*step;
            auto expected=original.forward(b,text,sigma,height,width,true,nullptr,refs);
            auto actual=direct.forward(a,text,sigma,height,width,true,nullptr,refs);
#if defined(TC_QKV_PREFIX_COUNTEREXAMPLE)
            require(values(expected)!=values(actual)&&mpp_mismatches>0&&native_base_mismatches==0,"small-GEMM counterexample or native-base isolation changed");
            std::cout<<"{\"small_shape_not_equivalent\":true,\"native_base_counterfactual_exact\":true}\n";return;
#endif
            require(same_bits(expected,actual),"same-prefix original/fused final tensor bits differ");
            a=a+actual*Tensor(-.25f,mx::bfloat16);b=b+expected*Tensor(-.25f,mx::bfloat16);
            require(same_bits(a,b),"updated tiny image state bits differ");
        }
        require(direct.corrected_qkv_decode_blocks()==(direct_qkv?3*depth:0),"prefix-hit QKV first-step decode coverage differs");
        require(direct.lora_swiglu_decode_blocks()==(swiglu?3*depth:0),"prefix-hit SwiGLU first-step decode coverage differs");
    }
    std::cout<<"{\"same_prefix_tensor_exact\":true,\"first_step_decode_counted\":true,\"source_destroyed\":true,\"callback_and_budget_rejected\":true}\n";
}
}
int main(int argc,char **argv){try{tc::require(argc==2,"fresh fixture directory required");run(argv[1]);return 0;}
catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
