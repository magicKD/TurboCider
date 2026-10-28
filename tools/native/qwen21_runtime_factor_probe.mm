#include "qwen21_runtime_factor_mil.hpp"
#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/mlx.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <map>

#ifndef TC_RUNTIME_FACTOR_CPP_SHA
#define TC_RUNTIME_FACTOR_CPP_SHA "unbound"
#define TC_RUNTIME_FACTOR_HEADER_SHA "unbound"
#endif
using namespace tc;
namespace rf {
namespace pa=ane::private_api;
using Half=mx::float16_t;
NSString *ns(const std::string &s){return [NSString stringWithUTF8String:s.c_str()];}
void put(pa::Surface &s,int row,int col,float value){
    require(row>=0&&row<int(s.rows())&&col>=0&&col<int(s.columns()),"surface write extent differs");
    reinterpret_cast<Half*>(static_cast<uint8_t*>(s.data())+size_t(row)*s.pitch())[col]=Half(value);
}
float get(const pa::Surface &s,int row,int col){return float(reinterpret_cast<const Half*>(static_cast<const uint8_t*>(s.data())+size_t(row)*s.pitch())[col]);}
std::vector<Half> read(const pa::Surface &s){
    std::vector<Half> result(size_t(s.rows())*s.columns());
    for(size_t row=0;row<s.rows();++row)std::memcpy(result.data()+row*s.columns(),static_cast<const uint8_t*>(s.data())+row*s.pitch(),size_t(s.columns())*sizeof(Half));
    return result;
}
std::string hash(const std::vector<Half>&a){return memory_sha256_hex({reinterpret_cast<const char*>(a.data()),a.size()*sizeof(Half)});}
NSDictionary *quality(const std::vector<Half>&actual,const std::vector<float>&expected,int begin,int rows,int m){
    double error=0,energy=0,other=0,dot=0,maximum=0;
    for(int row=begin;row<begin+rows;++row)for(int token=0;token<m;++token){const size_t i=size_t(row)*m+token;
        const double x=expected[i],y=float(actual[i]),d=x-y;require(std::isfinite(x)&&std::isfinite(y),"nonfinite runtime factor output");
        error+=d*d;energy+=x*x;other+=y*y;dot+=x*y;maximum=std::max(maximum,std::abs(d));}
    const double l2=std::sqrt(error/std::max(energy,1e-30)),cosine=dot/std::sqrt(std::max(energy*other,1e-60));
    require(l2<=.03&&cosine>=.999,"synthetic runtime factor numerical gate failed");
    return @{@"elements":@(size_t(rows)*m),@"relative_l2":@(l2),@"cosine":@(cosine),@"max_abs":@(maximum)};
}
void run(const std::filesystem::path &out,bool production,NSMutableDictionary *report){
    const int m=production?1024:32,h=production?4096:64,f=production?2048:128,r=production?128:32;
    const ane::GraphShape shape{ane::Kind::SwiGLU,m,h,f,h,h,true};const auto mil=qwen21::runtime_factor_lab::program(shape,r,.75f,-.5f);
    report[@"geometry"]=@{@"rows":@(m),@"hidden":@(h),@"channels":@(f),@"rank":@(r)};
    report[@"mil_sha256"]=ns(memory_sha256_hex(mil));report[@"model_constants_bytes"]=@0;
    {std::ofstream source(out/"runtime-factors.mil");source<<mil;}
    pa::Device device;pa::Program program(device,mil,{},out/"cache");
    auto names=program.inputs();std::sort(names.begin(),names.end());
    require(names==std::vector<std::string>{"ag","au","bg","bu","wd","wg","wu","x"}&&program.outputs()==std::vector<std::string>{"y"},"actual eight-input runtime interface differs");
    report[@"compiled_now"]=@(program.compiled_now());report[@"program_cache_key"]=ns(program.cache_key());
    std::vector<std::pair<std::string,pa::Surface>> inputs;
    for(const auto &[name,rows,cols]:std::vector<std::tuple<std::string,int,int>>{{"x",h,m},{"wg",f,h},{"wu",f,h},{"wd",h,f},{"ag",r,h},{"au",r,h},{"bg",f,r},{"bu",f,r}}){
        inputs.emplace_back(name,pa::Surface(device,rows,cols,pa::Element::FP16));std::memset(inputs.back().second.data(),0,inputs.back().second.bytes());}
    const auto surface=[&](const std::string &name)->pa::Surface&{for(auto &[key,value]:inputs)if(key==name)return value;throw std::runtime_error("missing surface");};
    for(int token=0;token<m;++token)put(surface("x"),(token*37)%h,token,token%2?-.125f:.125f);
    for(int c=0;c<f;++c){put(surface("wg"),c,c%h,.25f);put(surface("wu"),c,c%h,.5f);
        put(surface("bg"),c,c%r,c%2?-.0625f:.0625f);put(surface("bu"),c,c%r,.125f);}
    for(int c=0;c<h;++c)put(surface("wd"),c,c%f,c%2?-.25f:.25f);
    for(int rank=0;rank<r;++rank)for(int col=rank;col<h;col+=r){put(surface("ag"),rank,col,.25f);put(surface("au"),rank,col,.5f);}
    std::vector<std::pair<std::string,pa::Surface>> outputs;outputs.emplace_back("y",pa::Surface(device,h+f,m,pa::Element::FP16));
    uint64_t timeline=0;std::vector<Half> first;NSMutableArray *cases=[NSMutableArray array];report[@"cases"]=cases;
    for(int iteration=0;iteration<3;++iteration){
        const float factor=iteration==1?2.f:1.f;
        for(int c=0;c<f;++c)put(surface("bg"),c,c%r,factor*(c%2?-.0625f:.0625f));
        std::memset(outputs[0].second.data(),0xff,outputs[0].second.bytes());const auto ready=++timeline,done=++timeline;
        device.signal(ready);auto ticket=program.enqueue(inputs,outputs,ready,done);const auto completion=ticket.finish();
        require(completion.ok&&completion.physical_complete&&!completion.retired,"runtime factor request did not complete physically");
        auto actual=read(outputs[0].second);std::vector<float> expected(size_t(h+f)*m),hidden(size_t(f)*m);
        // Independent sparse FP64 formula. All ranks and source columns occur
        // across tokens; use actual physical FP16 inputs, never model outputs.
        for(int token=0;token<m;++token){const int col=(token*37)%h;const double x=get(surface("x"),col,token);
            for(int c=0;c<f;++c){const int rank=c%r;
                const double lowg=double(get(surface("ag"),rank,col))*x,lowu=double(get(surface("au"),rank,col))*x;
                const double g=double(get(surface("wg"),c,col))*x+.75*double(get(surface("bg"),c,rank))*lowg;
                const double u=double(get(surface("wu"),c,col))*x-.5*double(get(surface("bu"),c,rank))*lowu;
                const double value=g/(1+std::exp(-g))*u;hidden[size_t(c)*m+token]=float(value);expected[size_t(h+c)*m+token]=float(value);}
            for(int c=0;c<h;++c)expected[size_t(c)*m+token]=float(double(get(surface("wd"),c,c%f))*hidden[size_t(c%f)*m+token]);}
        auto y=quality(actual,expected,0,h,m),hh=quality(actual,expected,h,f,m);const auto bits=hash(actual);
        if(iteration==0)first=actual;
        if(iteration==1)require(hash(first)!=bits,"new runtime B factors reused previous output");
        if(iteration==2)require(hash(first)==bits,"A-B-A runtime factor restoration failed");
        mx::save_safetensors((out/("case-"+std::to_string(iteration)+".safetensors")).string(),{
            {"actual",Tensor(actual.data(),{h+f,m},mx::float16)},{"expected_fp32",Tensor(expected.data(),{h+f,m},mx::float32)}});
        [cases addObject:@{@"case":@(iteration),@"full_base_partial":y,@"full_corrected_hidden":hh,@"output_sha256":ns(bits),@"factor_multiplier":@(factor),@"physical_complete":@YES}];
    }
    report[@"successful_requests"]=@3;report[@"same_program_rebound_factors"]=@YES;report[@"a_b_a_exact_restore"]=@YES;
    report[@"status"]=@"complete";
}
}
int main(int argc,const char **argv){@autoreleasepool{
    if(argc==2&&std::string_view(argv[1])=="--describe"){
        char *id=tc_runtime_build_identity();if(!id)return 1;
        std::cout<<json(@{@"recipe":@"runtime-base-and-factors-fp16-v1",@"cpp_sha256":@TC_RUNTIME_FACTOR_CPP_SHA,@"header_sha256":@TC_RUNTIME_FACTOR_HEADER_SHA,@"loaded_runtime_id":rf::ns(id)})<<'\n';tc_string_free(id);return 0;}
    if(argc!=3){std::cerr<<"usage: runtime-factor-probe tiny|qwen FRESH_OUTPUT\n";return 2;}
    const std::filesystem::path out=std::filesystem::absolute(argv[2]);bool created=false;
    NSMutableDictionary *report=[@{@"schema":@"tc-qwen-runtime-factor-capability-v1",@"status":@"starting",@"qualification_passed":@NO,
        @"scope":@"synthetic sparse same-program runtime base/factor rebinding; not checkpoint, full block/image latency or physical overlap"} mutableCopy];
    try{require(std::string_view(argv[1])=="tiny"||std::string_view(argv[1])=="qwen","known shape required");
        require(!std::filesystem::exists(out)&&std::filesystem::create_directories(out),"fresh owned output required");created=true;
        require(std::filesystem::space(out).available>(3ull<<30),"3GiB disk reserve required");
        rf::run(out,std::string_view(argv[1])=="qwen",report);
    }catch(const std::exception &error){report[@"status"]=@"failed";report[@"error"]=rf::ns(error.what());}
    if(created){std::ofstream file(out/"native.json");file<<json(report)<<'\n';}std::cout<<json(report)<<'\n';return [report[@"status"] isEqual:@"complete"]?0:1;
}}
