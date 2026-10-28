#include "qwen21_qkv_lora_prepare.hpp"
#include "../../native/models/qwen21/metal/lora_qkv_prepare.hpp"
#include "../../native/models/qwen21/pipeline.hpp"
#include "../../native/models/qwen21/viggle_adapter.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <limits>
#include <tuple>

using namespace tc;
#ifndef TC_QKV_PROBE_CPP_SHA
#define TC_QKV_PROBE_CPP_SHA "unbound"
#define TC_QKV_PROBE_HEADER_SHA "unbound"
#endif
namespace lab {
using Fn=qwen21::DiagnosticGpuBlock;
constexpr std::array<const char*,9> names{"gpu-original","gpu-replay","mlx-packed-base","fused-b32","fused-b16","fused-b64",
    "joint-base-a-mlx-b32","joint-base-a-mpp-b32","fused-b48"};
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
NSString *ns(const std::string &s){return [NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor &a){auto bytes=mx::view(mx::contiguous(a),mx::uint8);mx::eval(bytes);
    return memory_sha256_hex({reinterpret_cast<const char*>(bytes.data<uint8_t>()),bytes.nbytes()});}
NSDictionary *difference(const Tensor &reference,const Tensor &value){
    require(reference.shape()==value.shape(),"full output shape differs");
    auto a=mx::contiguous(mx::astype(reference,mx::float32)),b=mx::contiguous(mx::astype(value,mx::float32));mx::eval(a,b);
    double error=0,energy=0,other=0,dot=0,maximum=0;size_t changed=0;
    for(size_t i=0;i<a.size();++i){const double x=a.data<float>()[i],y=b.data<float>()[i],d=x-y;
        require(std::isfinite(x)&&std::isfinite(y),"nonfinite complete output");
        error+=d*d;energy+=x*x;other+=y*y;dot+=x*y;maximum=std::max(maximum,std::abs(d));changed+=x!=y;}
    const double cosine=energy==0&&other==0?1.:dot/std::sqrt(std::max(energy*other,1e-60));
    return @{@"relative_l2":@(std::sqrt(error/std::max(energy,1e-30))),@"cosine":@(cosine),@"max_abs":@(maximum),
        @"changed":@(changed),@"elements":@(a.size())};
}
Tensor rotate(const Tensor &x,const Tensor &cosine,const Tensor &sine){
    auto shape=x.shape(),pairs=shape;pairs.back()/=2;pairs.push_back(2);
    auto p=mx::reshape(mx::astype(x,mx::float32),pairs);
    auto a=mx::squeeze(slice_axis(p,-1,0,1),-1),b=mx::squeeze(slice_axis(p,-1,1,2),-1);
    auto c=mx::reshape(cosine,{1,1,shape[2],shape[3]/2}),s=mx::reshape(sine,{1,1,shape[2],shape[3]/2});
    return mx::astype(mx::reshape(mx::stack({a*c-b*s,a*s+b*c},-1),shape),x.dtype());
}
std::vector<Tensor> prepare_reference(const Tensor &x,const Tensor &w,const std::array<Tensor,3> &ranks,
    const std::array<Tensor,3> &up,const std::array<float,3> &scales,const Tensor &qw,const Tensor &kw,
    const Tensor &cosine,const Tensor &sine){
    const int h=x.shape(2);std::vector<Tensor> result;
    for(int i=0;i<3;++i){auto base=mx::matmul(x,mx::transpose(slice_axis(w,0,i*h,(i+1)*h)));
        auto y=dense_gpu::lora_b_epilogue(ranks[i],up[i],0,h,scales[i],mx::bfloat16,base);
        auto out=heads(y,h/128,128);
        if(i<2)out=rotate(mx::fast::rms_norm(out,i==0?qw:kw,1e-6f),cosine,sine);
        result.push_back(out);
    }return result;
}
void kernel_test(){
    // Actual-library witness, not an assumption that a source tag matches the
    // installed binary: weighted BF16 RMSNorm equals typed unweighted*gamma.
    std::vector<float> xv(33*128),gv(128);
    for(size_t i=0;i<xv.size();++i)xv[i]=float((i*37)%128+64)/128.f;
    for(size_t i=0;i<gv.size();++i)gv[i]=float((i*13)%128+64)/64.f;
    auto witness_x=mx::astype(Tensor(xv.data(),{33,128},mx::float32),mx::bfloat16);
    auto witness_g=mx::astype(Tensor(gv.data(),{128},mx::float32),mx::bfloat16);
    auto native=mx::fast::rms_norm(witness_x,witness_g,1e-6f);
    auto normalized=mx::fast::rms_norm(witness_x,mx::ones({128},mx::bfloat16),1e-6f);
    mx::eval(normalized);
    auto typed=normalized*witness_g;
    auto one_round=mx::astype(mx::fast::rms_norm(mx::astype(witness_x,mx::float32),mx::astype(witness_g,mx::float32),1e-6f),mx::bfloat16);
    mx::eval(native,typed,one_round);
    require([difference(native,typed)[@"changed"] unsignedLongLongValue]==0 &&
            [difference(native,one_round)[@"changed"] unsignedLongLongValue]>0,"actual native RMSNorm typed-rounding witness failed");
    size_t cases=0,rejected=0,migrated_cases=0,migrated_rejected=0,direct_cases=0,direct_rejected=0;double worst=0;
    for(int fixture=0;fixture<5;++fixture){
        const int m=fixture==0?1:fixture==1?33:fixture==2?129:fixture==3?65:33;
        const int h=(fixture==1||fixture==3)?256:128;
        const int r=fixture==2?256:fixture==1||fixture==3?128:64;
        const std::array<float,3> scales{.75f,-.5f,1.25f};
        auto make=[&]{
            auto x=mx::astype(mx::reshape(mx::sin(mx::arange((m+2)*h,mx::float32)*.007f)*.125f,{1,m+2,h}),mx::bfloat16);
            x=slice_axis(x,1,1,m+1);
            if(fixture==3)x=mx::expand_dims(mx::transpose(mx::astype(mx::reshape(
                mx::cos(mx::arange(m*h,mx::float32)*.013f)*.08f,{h,m}),mx::bfloat16)),0);
            auto w=slice_axis(mx::astype(mx::reshape(mx::sin(mx::arange((3*h+2)*h,mx::float32)*.001f)*.02f,{3*h+2,h}),mx::bfloat16),0,1,3*h+1);
            std::array<Tensor,3> ranks{Tensor(0.f),Tensor(0.f),Tensor(0.f)},up=ranks;
            for(int i=0;i<3;++i){
                ranks[i]=mx::reshape(mx::sin(mx::arange((m+1)*(r+3),mx::float32)*(.011f+i*.003f))*.1f,{1,m+1,r+3});
                ranks[i]=slice_axis(slice_axis(ranks[i],1,1,m+1),2,1,r+1);
                up[i]=slice_axis(mx::astype(mx::reshape(mx::cos(mx::arange((h+4)*r,mx::float32)*(.003f+i*.001f))*.03f,{h+4,r}),mx::bfloat16),0,2,h+2);
                if(fixture==4)ranks[i]=mx::zeros_like(ranks[i]);
            }
            if(fixture==4)x=mx::zeros_like(x);
            auto gamma=mx::astype(mx::sin(mx::arange(140,mx::float32)*.013f)*.2f+1.f,mx::bfloat16);
            auto qw=slice_axis(gamma,0,1,129),kw=slice_axis(gamma,0,2,130);
            auto angle=mx::reshape(mx::arange(m*64,mx::float32)*.0013f,{m,64});
            return std::tuple{x,w,ranks,up,qw,kw,mx::cos(angle),mx::sin(angle)};
        };
        auto [x,w,ranks,up,qw,kw,c,s]=make();auto reference=prepare_reference(x,w,ranks,up,scales,qw,kw,c,s);mx::eval(reference);
        for(int bm:{16,32,48,64}){
            auto [nx,nw,nr,nu,nqw,nkw,nc,nsine]=make();
            require(!nx.is_available(),"lazy source fixture unexpectedly evaluated");
            auto values=qwen21::qkv_lora_lab::prepare(nx,nw,nr,nu,scales,nqw,nkw,nc,nsine,bm);mx::eval(values);
            auto migrated=qwen21::metal::prepare_corrected_qkv(nx,nw,nr,nu,scales,nqw,nkw,nc,nsine,bm);mx::eval(migrated);
            for(int i=0;i<3;++i)require(hash(reference[i])==hash(migrated[i]),"native corrected-QKV does not match the original typed arithmetic bits");
            ++migrated_cases;
            std::array<Tensor,3> direct_base{Tensor(0.f),Tensor(0.f),Tensor(0.f)};
            for(int i=0;i<3;++i){auto original=slice_axis(nw,0,i*h,(i+1)*h);auto wide=mx::astype(original,mx::float32);mx::eval(wide);
                direct_base[i]=mx::astype(wide,mx::bfloat16);mx::eval(direct_base[i]);
                require(hash(direct_base[i])==hash(original)&&direct_base[i].data_shared_ptr()!=nw.data_shared_ptr(),"direct source must own original exact values independently");
                for(int j=0;j<i;++j)require(direct_base[i].data_shared_ptr()!=direct_base[j].data_shared_ptr(),"three independent source owners required");}
            auto direct=qwen21::metal::prepare_corrected_qkv_direct(nx,direct_base,nr,nu,scales,nqw,nkw,nc,nsine,bm);mx::eval(direct);
            for(int i=0;i<3;++i)require(hash(reference[i])==hash(direct[i]),"direct-buffer QKV changed original arithmetic bits");++direct_cases;
            require(values.size()==3,"complete QKV required");
            for(int i=0;i<3;++i){auto q=difference(reference[i],values[i]);
                const double error=[q[@"relative_l2"] doubleValue];worst=std::max(worst,error);
                require(error<=.005 && [q[@"cosine"] doubleValue]>=.999,"corrected QKV numeric contract failed");}
            ++cases;
            std::vector<Tensor> joined;
            for(int i=0;i<3;++i)joined.push_back(mx::matmul(nx,mx::transpose(slice_axis(nw,0,i*h,(i+1)*h))));
            for(const auto &rank:nr)joined.push_back(mx::astype(rank,mx::bfloat16));
            auto first=mx::concatenate(joined,2);
            auto finished=qwen21::qkv_lora_lab::finish_projected(first,nu,scales,nqw,nkw,nc,nsine,bm);mx::eval(finished);
            for(int i=0;i<3;++i){auto q=difference(reference[i],finished[i]);const double error=[q[@"relative_l2"] doubleValue];
                worst=std::max(worst,error);require(error<=.005 && [q[@"cosine"] doubleValue]>=.999,"joint projection finish numeric contract failed");}
            ++cases;
        }
        for(int invalid=0;invalid<10;++invalid){
            auto nx=x,nw=w,nqw=qw,nc=c;auto nr=ranks,nu=up;auto sc=scales;int bm=32;
            if(invalid==0)nx=mx::astype(nx,mx::float32);
            if(invalid==1)nw=slice_axis(nw,0,0,3*h-1);
            if(invalid==2)nqw=mx::astype(nqw,mx::float32);
            if(invalid==3)nc=slice_axis(nc,1,0,63);
            if(invalid==4)nr[1]=slice_axis(nr[1],2,0,r-1);
            if(invalid==5)nu[1]=mx::astype(nu[1],mx::float32);
            if(invalid==6)sc[0]=std::numeric_limits<float>::infinity();
            if(invalid==7)bm=24;
            if(invalid==8)nu[2]=slice_axis(nu[2],0,0,h-1);
            if(invalid==9)nr[0]=mx::astype(nr[0],mx::bfloat16);
            bool denied=false;try{(void)qwen21::qkv_lora_lab::prepare(nx,nw,nr,nu,sc,nqw,kw,nc,s,bm);}catch(const std::exception&){denied=true;}
            require(denied,"malformed corrected QKV contract accepted");++rejected;
            denied=false;try{(void)qwen21::metal::prepare_corrected_qkv(nx,nw,nr,nu,sc,nqw,kw,nc,s,bm);}catch(const std::exception&){denied=true;}
            require(denied,"native corrected QKV malformed contract accepted");++migrated_rejected;
            std::array<Tensor,3> base{slice_axis(w,0,0,h),slice_axis(w,0,h,2*h),slice_axis(w,0,2*h,3*h)};
            if(invalid==1)base[0]=slice_axis(base[0],0,0,h-1);
            denied=false;try{(void)qwen21::metal::prepare_corrected_qkv_direct(nx,base,nr,nu,sc,nqw,kw,nc,s,bm);}catch(const std::exception&){denied=true;}
            require(denied,"malformed direct-buffer contract accepted");++direct_rejected;
        }
        for(int bad=0;bad<3;++bad){std::array<Tensor,3> base{slice_axis(w,0,0,h),slice_axis(w,0,h,2*h),slice_axis(w,0,2*h,3*h)};
            if(bad==0)base[1]=mx::transpose(base[1]);if(bad==1)base[2]=mx::astype(base[2],mx::float16);
            if(bad==2)base[0]=slice_axis(base[0],1,0,h-1);
            for(auto &value:base)mx::eval(value); // lazy flags are not the final physical layout
            bool denied=false;try{(void)qwen21::metal::prepare_corrected_qkv_direct(x,base,ranks,up,scales,qw,kw,c,s,32);}catch(const std::exception&){denied=true;}
            require(denied,"invalid original direct source accepted");++direct_rejected;}
        for(int invalid=0;invalid<8;++invalid){
            auto projected=mx::zeros({1,m,3*h+3*r},mx::bfloat16);auto nu=up;auto nc=c;auto sc=scales;int bm=32;
            if(invalid==0)projected=mx::astype(projected,mx::float32);
            if(invalid==1)projected=slice_axis(projected,2,0,3*h+3*r-1);
            if(invalid==2)nu[1]=mx::astype(nu[1],mx::float32);
            if(invalid==3)nu[2]=slice_axis(nu[2],0,0,h-1);
            if(invalid==4)nc=slice_axis(c,1,0,63);
            if(invalid==5)sc[1]=std::numeric_limits<float>::quiet_NaN();
            if(invalid==6)bm=24;
            if(invalid==7)nu[0]=slice_axis(nu[0],1,0,r-1);
            bool denied=false;try{(void)qwen21::qkv_lora_lab::finish_projected(projected,nu,sc,qw,kw,nc,s,bm);}catch(const std::exception&){denied=true;}
            require(denied,"malformed joint projection finish accepted");++rejected;
        }
    }
    std::cout<<"kernel_cases="<<cases<<" rejected="<<rejected<<" worst_relative_l2="<<worst<<" native_reference_exact_cases="<<migrated_cases
        <<" migrated_rejected="<<migrated_rejected<<" direct_exact_cases="<<direct_cases<<" direct_rejected="<<direct_rejected
        <<" actual_rms_typed_rounding_witness=true\n";
}
struct Binding{
    Tensor packed{0.f},joint{0.f};std::array<Tensor,3> up{Tensor(0.f),Tensor(0.f),Tensor(0.f)};
    std::array<float,3> scales{1.f,1.f,1.f};
    const std::array<std::string,3> prefixes{"transformer_blocks.0.attn.to_q","transformer_blocks.0.attn.to_k","transformer_blocks.0.attn.to_v"};
    explicit Binding(const Weights &w){
        require(w.runtime_lora_bf16_fp32_ranks()&&w.runtime_lora_b_epilogue(),"original joint arithmetic required");
        const auto snapshot=w.runtime_lora_snapshot();require(!snapshot.fp16_rank_math,"F32 ranks required");
        std::vector<Tensor> bases,downs;
        for(int i=0;i<3;++i){const auto &p=prefixes[i];const auto &value=w.at(p+".weight");
            require(value.shape()==mx::Shape{4096,4096} && value.dtype()==mx::bfloat16 && !w.has(p+".bias") &&
                !w.quantized(p) && !w.convrot(p) && !w.nvfp4(p),"original unbiased BF16 QKV required");
            auto found=std::find_if(snapshot.bindings.begin(),snapshot.bindings.end(),[&](const auto &b){return b.prefix==p;});
            require(found!=snapshot.bindings.end() && found->ordered_terms.size()==1,"one complete QKV adapter term required");
            const auto &term=found->ordered_terms[0];
            require(term.down.shape()==mx::Shape{128,4096} && term.up.shape()==mx::Shape{4096,128} &&
                term.down.dtype()==mx::bfloat16 && term.up.dtype()==mx::bfloat16 && term.output_start==0 && term.output_end==4096 &&
                term.source_adapter_ordinal==0,"full pinned r128 QKV correction required");
            up[i]=term.up;scales[i]=term.scale;bases.push_back(value);downs.push_back(term.down);
        }
        packed=mx::contiguous(mx::concatenate(bases,0));mx::eval(packed);
        for(int i=0;i<3;++i)require(hash(slice_axis(packed,0,i*4096,(i+1)*4096))==hash(bases[i]),"base source pack differs");
        std::vector<Tensor> all{packed};all.insert(all.end(),downs.begin(),downs.end());joint=mx::contiguous(mx::concatenate(all,0));mx::eval(joint);
        require(hash(slice_axis(joint,0,0,12288))==hash(packed),"joint base rows differ");
        for(int i=0;i<3;++i)require(hash(slice_axis(joint,0,12288+i*128,12288+(i+1)*128))==hash(downs[i]),"joint independent A rows differ");
    }
    Tensor joint_project(const Tensor &x,bool mpp) const{return mpp?
        dense_gpu::projection_range(x,joint,0,12672,0,4096,32,false,128):mx::matmul(x,mx::transpose(joint));}
    std::vector<Tensor> project(const Weights &w,const Tensor &x,const Tensor &c,const Tensor &s,int mode) const{
        if(mode==6||mode==7)return qwen21::qkv_lora_lab::finish_projected(joint_project(x,mode==7),up,scales,
            w.at("transformer_blocks.0.attn.norm_q.weight"),w.at("transformer_blocks.0.attn.norm_k.weight"),c,s,32);
        if(mode>=3){std::array<Tensor,3> ranks{Tensor(0.f),Tensor(0.f),Tensor(0.f)};
            for(int i=0;i<3;++i){auto r=w.lora_input_ranks(x,prefixes[i],0,4096);require(r.size()==1,"complete online A rank required");ranks[i]=r[0];}
            return qwen21::qkv_lora_lab::prepare(x,packed,ranks,up,scales,w.at("transformer_blocks.0.attn.norm_q.weight"),
                w.at("transformer_blocks.0.attn.norm_k.weight"),c,s,mode==3?32:mode==4?16:mode==5?64:48);
        }
        std::vector<Tensor> qkv;Tensor base(0.f);
        if(mode==2)base=mx::matmul(x,mx::transpose(packed));
        for(int i=0;i<3;++i){auto y=mode==2?w.apply_runtime_lora_slice(mx::contiguous(slice_axis(base,2,i*4096,(i+1)*4096)),x,
                prefixes[i],0,4096,0,4096):linear(x,w,prefixes[i]);
            auto v=heads(y,32,128);
            if(i<2)v=rotate(mx::fast::rms_norm(v,w.at(i==0?"transformer_blocks.0.attn.norm_q.weight":"transformer_blocks.0.attn.norm_k.weight"),1e-6f),c,s);
            qkv.push_back(v);
        }return qkv;
    }
};
Fn block(const Weights &w,const Binding &binding,int mode){return mx::compile(Fn([&w,&binding,mode](const std::vector<Tensor> &a){
    const std::string p="transformer_blocks.0.";
    auto x=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);
    auto qkv=binding.project(w,x,a[5],a[6],mode);
    auto attended=attend(qkv[0],mx::concatenate({a[7],qkv[1]},2),mx::concatenate({a[8],qkv[2]},2));
    auto hidden=a[0]+mx::tanh(a[2])*linear(attended,w,p+"attn.to_out.0");
    x=mx::fast::layer_norm(hidden,{},{},1e-6f)*(Tensor(1.f,hidden.dtype())+a[3]);
    auto gu=linear(x,w,p+"img_mlp.gate_up");const int half=gu.shape(-1)/2;
    auto ff=silu(slice_axis(gu,-1,0,half))*slice_axis(gu,-1,half,gu.shape(-1));
    return std::vector<Tensor>{hidden+mx::tanh(a[4])*linear(ff,w,p+"img_mlp.out")};
}));}
void gate(NSDictionary *q,bool exact){require([q[@"relative_l2"] doubleValue]<=.001 && [q[@"cosine"] doubleValue]>=.999,
    "whole-block numerical gate failed");if(exact)require([q[@"changed"] unsignedLongLongValue]==0,"ordinary GPU replay is not exact");}
void measure(const qwen21::DiagnosticBlockObservation &o,const std::filesystem::path &out,NSMutableDictionary *report){
    require(o.layer==0&&o.steady&&o.arguments.size()==9&&o.outputs.size()==1&&o.arguments[0].shape()==mx::Shape{1,1024,4096},"real ordinary decode block required");
    const auto &w=o.weights;size_t terms=0;for(const auto &binding:w.runtime_lora_snapshot().bindings)terms+=binding.ordered_terms.size();
    require(terms==227,"full pinned student required");report[@"runtime_lora_applied_terms"]=@(terms);report[@"prefix_tokens"]=@(o.prefix_tokens);
    std::map<std::string,std::string> hashes;
    for(const auto &key:w.sorted_keys())if(key.starts_with("transformer_blocks.0."))hashes.emplace(key,hash(w.at(key)));
    for(const auto &binding:w.runtime_lora_snapshot().bindings)if(binding.prefix.starts_with("transformer_blocks.0."))
        for(size_t i=0;i<binding.ordered_terms.size();++i){const auto &term=binding.ordered_terms[i];const auto key=binding.prefix+".term"+std::to_string(i);
            hashes.emplace(key+".A",hash(term.down));hashes.emplace(key+".B",hash(term.up));}
    std::vector<std::string> argument_hashes;for(const auto &a:o.arguments)argument_hashes.push_back(hash(a));
    const auto original_hash=hash(o.outputs[0]);const auto setup=Clock::now();Binding binding(w);
    report[@"pack_and_verify_seconds"]=@(elapsed(setup));report[@"packed_base_bytes"]=@(binding.packed.nbytes());
    report[@"packed_joint_base_a_bytes"]=@(binding.joint.nbytes());report[@"qkv_recipe_version"]=@"joint-base-a-v1";
    report[@"qkv_scales"]=@[@(binding.scales[0]),@(binding.scales[1]),@(binding.scales[2])];
    auto input=mx::fast::layer_norm(o.arguments[0],{},{},1e-6f)*(Tensor(1.f,o.arguments[0].dtype())+o.arguments[1]);mx::eval(input);
    auto mlx_joint=binding.joint_project(input,false),mpp_joint=binding.joint_project(input,true);mx::eval(mlx_joint,mpp_joint);
    NSMutableArray *rank_quality=[NSMutableArray array];report[@"rank_boundary_quality"]=rank_quality;
    for(int i=0;i<3;++i){auto reference=mx::astype(w.lora_input_ranks(input,binding.prefixes[i],0,4096)[0],mx::bfloat16);mx::eval(reference);
        auto mlq=difference(reference,slice_axis(mlx_joint,2,12288+i*128,12288+(i+1)*128));
        auto mpq=difference(reference,slice_axis(mpp_joint,2,12288+i*128,12288+(i+1)*128));
        require([mlq[@"relative_l2"] doubleValue]<=.005 && [mpq[@"relative_l2"] doubleValue]<=.005,"typed sole-B rank boundary differs");
        [rank_quality addObject:@{@"projection":@(i),@"mlx":mlq,@"mpp":mpq}];}
    std::vector<Fn> functions{o.gpu_block};for(int mode=1;mode<int(names.size());++mode)functions.push_back(block(w,binding,mode));
    NSMutableArray *initial_quality=[NSMutableArray array];report[@"initial_quality"]=initial_quality;
    for(int mode=1;mode<int(names.size());++mode){const auto first_start=Clock::now();auto y=functions[mode](o.arguments)[0];mx::eval(y);
        const double first_wall=elapsed(first_start);auto q=difference(o.outputs[0],y);
        [initial_quality addObject:@{@"arm":@(names[mode]),@"quality":q,@"first_replay_seconds":@(first_wall)}];
        mx::save_safetensors((out/(std::string(names[mode])+"-initial.safetensors")).string(),{{"block",y}});
        gate(q,mode==1);}
    auto changed=o.arguments;changed[0]=mx::astype(changed[0]*Tensor(.996f,mx::bfloat16)+Tensor(.001f,mx::bfloat16),mx::bfloat16);mx::eval(changed[0]);
    auto fresh=functions[0](changed)[0];mx::eval(fresh);for(int mode=1;mode<int(names.size());++mode){auto y=functions[mode](changed)[0];mx::eval(y);gate(difference(fresh,y),mode==1);}
    report[@"fresh_arguments_verified"]=@YES;
    mx::save_safetensors((out/"inputs-reference.safetensors").string(),{{"input",o.arguments[0]},{"block_reference",o.outputs[0]}});
    NSMutableArray *samples=[NSMutableArray array];report[@"samples"]=samples;
    std::vector<Tensor> last;for(size_t mode=0;mode<names.size();++mode)last.push_back(o.outputs[0]);
    report[@"measured_samples_per_arm"]=@18;
    for(int iteration=-3;iteration<18;++iteration)for(int visit=0;visit<int(names.size());++visit){
        const int mode=(visit+iteration+int(names.size()))%int(names.size());const auto start=Clock::now();auto y=functions[mode](o.arguments)[0];
        require(!y.is_available(),"new block call returned an already available cached output");mx::eval(y);mx::synchronize();
        const double wall=elapsed(start);auto q=difference(o.outputs[0],y);gate(q,mode<2);last[mode]=y;
        [samples addObject:@{@"iteration":@(iteration),@"arm":@(names[mode]),@"seconds":@(wall),@"quality":q}];
    }
    std::unordered_map<std::string,Tensor> results;for(size_t mode=0;mode<names.size();++mode)results.emplace(names[mode],last[mode]);
    mx::save_safetensors((out/"complete-block-outputs.safetensors").string(),results);
    for(size_t i=0;i<o.arguments.size();++i)require(hash(o.arguments[i])==argument_hashes[i],"borrowed block argument changed");
    require(hash(o.outputs[0])==original_hash,"borrowed original block output changed");
    for(const auto &key:w.sorted_keys())if(key.starts_with("transformer_blocks.0."))require(hash(w.at(key))==hashes.at(key),"original layer weights changed");
    for(const auto &b:w.runtime_lora_snapshot().bindings)if(b.prefix.starts_with("transformer_blocks.0."))
        for(size_t i=0;i<b.ordered_terms.size();++i){const auto &t=b.ordered_terms[i];const auto key=b.prefix+".term"+std::to_string(i);
            require(hash(t.down)==hashes.at(key+".A")&&hash(t.up)==hashes.at(key+".B"),"original adapter factors changed");}
    NSMutableDictionary *source=[NSMutableDictionary dictionary];for(const auto &[key,value]:hashes)source[ns(key)]=ns(value);
    report[@"actual_layer_source_sha256"]=source;report[@"input_sha256"]=ns(argument_hashes[0]);report[@"ordinary_block_sha256"]=ns(original_hash);
    report[@"borrowed_values_unchanged"]=@YES;report[@"all_outputs_lazy_before_eval"]=@YES;report[@"status"]=@"measured";
}
} // namespace lab

int main(int argc,const char **argv){@autoreleasepool {
    if(argc==2&&std::string_view(argv[1])=="--describe"){
        NSMutableArray *arms=[NSMutableArray array];for(const auto *name:lab::names)[arms addObject:@(name)];
        char *runtime=tc_runtime_build_identity();if(!runtime)return 1;
        auto info=@{@"recipe_version":@"joint-base-a-v1",@"arms":arms,@"primary_recipe":@"joint-base-a-mpp-b32",
            @"measured_samples_per_arm":@18,@"cpp_sha256":@TC_QKV_PROBE_CPP_SHA,@"header_sha256":@TC_QKV_PROBE_HEADER_SHA,
            @"loaded_runtime_id":lab::ns(runtime)};
        tc_string_free(runtime);std::cout<<json(info)<<'\n';return 0;
    }
    if(argc==2&&std::string_view(argv[1])=="--kernel-contract"){
        try{configure_streams();lab::kernel_test();return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
    if(argc!=5){std::cerr<<"usage: probe MODEL PINNED_R128 FRESH_OUTPUT REFERENCE_OR_DASH\n";return 2;}
    const auto out=std::filesystem::absolute(argv[3]);bool created=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-qwen-corrected-qkv-block-v1",@"status":@"starting",@"qualification_passed":@NO,
        @"image_export":@NO,@"scope":@"ordinary r128/joint step1 layer0 full-block replay; complete online A/B, attention/FFN/residual; not full request or image quality"} mutableCopy];
    try{
        require(!std::filesystem::exists(out)&&std::filesystem::create_directories(out),"fresh owned output required");created=true;
        require(std::filesystem::space(out).available>(2ull<<30),"2GiB free reserve required");
        const auto adapter=std::filesystem::canonical(argv[2]);const auto *pin=qwen21::viggle_v021_adapter(adapter.filename().string());
        require(pin&&pin->rank=="r128"&&sha256_file(adapter)==pin->sha256,"original pinned r128 required");
        setenv("TURBOCIDER_QWEN21_LORA_BF16_AB","1",1);configure_streams();std::atomic<bool> cancelled{false};
        qwen21::Session session(argv[1]);int observations=0;bool finished=false;
        session.set_diagnostic_block_observer([&](const auto &o){require(o.layer==0&&observations<2&&o.steady==(observations==1),"bounded observation order differs");
            ++observations;if(!o.steady)return;lab::measure(o,out,report);throw qwen21::DiagnosticBlockReplayComplete();});
        const bool edit=std::string_view(argv[4])!="-";Request r;r.model="qwen-image-2.1";r.operation=edit?"image.edit":"image.generate";
        r.prompt=edit?"Change the teapot to bright red. Preserve its shape, handle, spout, table and background.":
                     "A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.";
        r.width=r.height=512;r.steps=6;r.seed=73;r.audio=false;r.execution="gpu";r.residency="resident";r.allow_approximation=true;
        r.lora_strategy="inference_time";r.loras.push_back({adapter.string(),1.f,"transformer"});r.qwen21_dit_cache="off";r.qwen21_dit_cache_explicit=true;
        r.output=(out/"must-not-export.png").string();if(edit){InputAsset ref;ref.kind="image";ref.role="reference";ref.path=std::filesystem::canonical(argv[4]).string();
            r.inputs.push_back(ref);r.qwen21_reference_size=512;report[@"reference_sha256"]=lab::ns(sha256_file(ref.path));}
        report[@"adapter_sha256"]=lab::ns(std::string(pin->sha256));report[@"operation"]=lab::ns(r.operation);report[@"seed"]=@(r.seed);
        const Event event=[](const std::string &p,int n,int total){std::cerr<<p<<' '<<n<<'/'<<total<<'\n';};
        try{(void)session.generate(r,event,cancelled);}catch(const qwen21::DiagnosticBlockReplayComplete&){finished=true;}
        session.set_diagnostic_block_observer({});session.unload();mx::synchronize();
        require(finished&&observations==2&&!std::filesystem::exists(r.output),"probe did not stop before image decoding");
        char *identity=tc_runtime_build_identity();require(identity,"loaded runtime identity missing");report[@"loaded_runtime_id"]=lab::ns(identity);tc_string_free(identity);
        report[@"completed_full_dit_steps"]=@1;report[@"stopped_step"]=@1;report[@"stopped_layer"]=@0;
    }catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=lab::ns(e.what());}
    if(created){std::ofstream file(out/"native.json");file<<json(report)<<'\n';}std::cout<<json(report)<<'\n';return [report[@"status"] isEqual:@"measured"]?0:1;
}}
