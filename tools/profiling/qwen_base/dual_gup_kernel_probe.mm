#include "dual_gup.hpp"
#include "../../../native/backends/mlx.hpp"
#include "../../../native/backends/dense_gpu_projection.hpp"
#include "../../../native/platform/apple/bridge.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

#ifndef TC_DUAL_GUP_CPP_SHA
#define TC_DUAL_GUP_CPP_SHA "unbound"
#define TC_DUAL_GUP_HEADER_SHA "unbound"
#define TC_DUAL_GUP_NATIVE_ID "unbound"
#endif
namespace {
using namespace tc;
NSString*ns(const std::string&s){return [NSString stringWithUTF8String:s.c_str()];}
std::string hash(const Tensor&a){auto b=mx::view(mx::contiguous(a),mx::uint8);mx::eval(b);return memory_sha256_hex({reinterpret_cast<const char*>(b.data<uint8_t>()),b.nbytes()});}
NSDictionary*quality(const Tensor&a,const Tensor&b){
    require(a.shape()==b.shape()&&a.dtype()==mx::bfloat16&&b.dtype()==a.dtype(),"complete typed kernel output required");
    auto x=mx::contiguous(mx::astype(a,mx::float32)),y=mx::contiguous(mx::astype(b,mx::float32));mx::eval(x,y);
    double error=0,energy=0,maximum=0;uint64_t changed=0;
    for(size_t i=0;i<x.size();++i){double u=x.data<float>()[i],v=y.data<float>()[i],d=u-v;require(std::isfinite(u)&&std::isfinite(v),"nonfinite dual GEMM output");error+=d*d;energy+=u*u;maximum=std::max(maximum,std::abs(d));changed+=u!=v;}
    return @{@"relative_l2":@(std::sqrt(error/std::max(energy,1e-30))),@"max_abs":@(maximum),@"changed":@(changed),@"elements":@(a.size()),@"bits_exact":@(hash(a)==hash(b))};
}
std::vector<Tensor> reference(const Tensor&x,const Tensor&w,int first,int count,bool mpp){
    const int f=w.shape(0)/2,k=x.shape(2);
    auto g=mpp?dense_gpu::projection_range(x,w,first,first+count,0,k,32,false,64):mx::matmul(x,mx::transpose(slice_axis(w,0,first,first+count)));
    auto u=mpp?dense_gpu::projection_range(x,w,f+first,f+first+count,0,k,32,false,64):mx::matmul(x,mx::transpose(slice_axis(w,0,f+first,f+first+count)));
    auto s=mx::sigmoid(g),a=silu(g);return {g,u,s,a,a*u};
}
}
extern "C" __attribute__((visibility("default")))
int tc_dual_gup_kernel_probe(const char*output){@autoreleasepool{
    const std::filesystem::path out(output?output:"");bool fresh=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-dual-gup-kernel-v1",@"status":@"starting",@"synthetic_only":@YES,
        @"complete_request":@NO,@"image_export":@NO,@"cpp_sha256":@TC_DUAL_GUP_CPP_SHA,@"header_sha256":@TC_DUAL_GUP_HEADER_SHA,
        @"compiled_native_id":@TC_DUAL_GUP_NATIVE_ID,@"extra_packed_weight_bytes":@0} mutableCopy];
    try{
        require(output&&std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"fresh dual kernel output required");fresh=true;
        char *raw=tc_runtime_build_identity();require(raw,"native ID missing");std::string id(raw);tc_string_free(raw);report[@"loaded_runtime_id"]=ns(id);
        require(id==TC_DUAL_GUP_NATIVE_ID,"dual kernel helper/native ABI mismatch");configure_streams();
        NSMutableArray *cases=[NSMutableArray array];report[@"cases"]=cases;uint64_t rejected=0;
        for(int fixture=0;fixture<2;++fixture){
            const int m=fixture?1024:33,k=fixture?4096:128,f=fixture?12288:320,first=fixture?0:17,count=fixture?6144:257;
            auto ix=mx::arange((m+2)*k,mx::int32),iw=mx::arange((2*f+4)*k,mx::int32);
            auto x=mx::astype(mx::reshape(mx::astype(((ix%127)*17+7)%29-14,mx::float32)/Tensor(32.f),{1,m+2,k}),mx::bfloat16);
            auto w=mx::astype(mx::reshape(mx::astype(((iw%257)*37+13)%31-15,mx::float32)/Tensor(128.f),{2*f+4,k}),mx::bfloat16);
            x=slice_axis(x,1,1,m+1);w=slice_axis(w,0,2,2*f+2);mx::eval(x,w);const auto x_hash=hash(x),w_hash=hash(w);
            auto baseline=reference(x,w,first,count,false),staged=reference(x,w,first,count,true);mx::eval(baseline);mx::eval(staged);
            std::unordered_map<std::string,Tensor> arrays{{"input",x}};
            for(int stage=0;stage<5;++stage){arrays.emplace("reference-"+std::to_string(stage),baseline[stage]);arrays.emplace("mpp-"+std::to_string(stage),staged[stage]);}
            const auto file="fixture-"+std::to_string(fixture)+".safetensors";
            for(bool shared:{false,true})for(int stage=0;stage<5;++stage){
                auto actual=dual_gup_lab::prepare(x,w,first,count,shared,32,64,stage);mx::eval(actual);
                const auto q=quality(baseline[stage],actual),qm=quality(staged[stage],actual);
                arrays.emplace(std::string(shared?"shared-":"register-")+std::to_string(stage),actual);
                [cases addObject:@{@"fixture":@(fixture),@"m":@(m),@"k":@(k),@"f":@(f),@"first":@(first),@"count":@(count),
                    @"shared":@(shared),@"bm":@32,@"bn":@64,@"stage":@(stage),@"quality_vs_mlx":q,@"quality_vs_mpp":qm,
                    @"input_sha256":ns(x_hash),@"weight_sha256":ns(w_hash)}];
            }
            for(int bad=0;bad<10;++bad){bool denied=false;try{(void)dual_gup_lab::prepare(
                bad==0?mx::astype(x,mx::float32):bad==1?mx::transpose(x,{0,2,1}):x,
                bad==2?mx::astype(w,mx::float16):bad==3?mx::transpose(w):w,bad==4?-1:first,bad==5?f+1:count,false,
                bad==6?64:32,bad==7?32:64,bad==8?-1:bad==9?5:4);
            }catch(const std::invalid_argument&){denied=true;}require(denied,"illegal paired-GEMM source/tile/stage accepted");++rejected;}
            auto alternate=mx::astype(x*Tensor(.75f,mx::bfloat16)+Tensor(.015625f,mx::bfloat16),mx::bfloat16);mx::eval(alternate);
            auto other_weight=mx::astype(w*Tensor(.875f,mx::bfloat16)+Tensor(.0009765625f,mx::bfloat16),mx::bfloat16);mx::eval(other_weight);
            auto other_reference=reference(x,other_weight,first,count,false)[4];mx::eval(other_reference);
            arrays.emplace("changed-input",alternate);arrays.emplace("weight-changed-reference",other_reference);
            for(bool shared:{false,true}){
                auto old=dual_gup_lab::prepare(x,w,first,count,shared),changed=dual_gup_lab::prepare(alternate,w,first,count,shared);mx::eval(old,changed);
                const auto old_hash=hash(old);require(hash(changed)!=old_hash,"changed dual input returned stale output");
                auto restored=dual_gup_lab::prepare(x,w,first,count,shared);mx::eval(restored);
                require(hash(old)==old_hash&&hash(restored)==old_hash,"old dual output overwritten or A/B/A not restored");
                auto changed_weight=dual_gup_lab::prepare(x,other_weight,first,count,shared);mx::eval(changed_weight);
                require(hash(changed_weight)!=old_hash,"changed dual weight returned stale output");
                auto restored_weight=dual_gup_lab::prepare(x,w,first,count,shared);mx::eval(restored_weight);
                require(hash(restored_weight)==old_hash&&hash(old)==old_hash,"weight A/B/A or old output ownership failed");
                require([quality(other_reference,changed_weight)[@"bits_exact"] boolValue],"changed weight synthetic hidden differs");
                arrays.emplace(shared?"shared-changed":"register-changed",changed);arrays.emplace(shared?"shared-restored":"register-restored",restored);
                arrays.emplace(shared?"shared-weight-changed":"register-weight-changed",changed_weight);
            }
            require(hash(x)==x_hash&&hash(w)==w_hash,"paired dense kernel modified borrowed source");
            mx::save_safetensors((out/file).string(),arrays);
        }
        report[@"rejected"]=@(rejected);report[@"source_unchanged"]=@YES;report[@"owned_output_restoration_exact"]=@YES;report[@"status"]=@"complete";
    }catch(const std::exception&e){report[@"status"]=@"failed";report[@"error"]=ns(e.what());std::cerr<<e.what()<<'\n';}
    if(fresh){std::ofstream file(out/"report.json");file<<json(report)<<'\n';}
    return [report[@"status"] isEqual:@"complete"]?0:1;
}}
