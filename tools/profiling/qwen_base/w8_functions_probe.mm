#include "../../../native/backends/private/ane_w8_executor.hpp"
#include "../../../native/backends/ane_gpu.hpp"
#include "../../../native/backends/ane_w8_functions.hpp"
#include "turbocider/turbocider.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <fstream>

#ifndef TC_W8_FUNCTION_CPP_SHA
#define TC_W8_FUNCTION_CPP_SHA "unbound"
#endif
#ifndef TC_W8_FUNCTION_NATIVE_ID
#define TC_W8_FUNCTION_NATIVE_ID "unbound"
#endif
namespace {
using namespace tc::ane;
constexpr int H=128,F=512,N=128;
void require(bool ok,const std::string &reason){if(!ok)throw CapabilityError(reason);}
struct Buffer {
    id<MTLBuffer> value;
    Buffer(void *device,size_t bytes){value=[(__bridge id<MTLDevice>)device newBufferWithLength:bytes options:MTLResourceStorageModeShared];require(value!=nil,"probe GPU allocation failed");}
    float *data(){return static_cast<float*>(value.contents);}
};
DeviceMatrixView matrix(const std::shared_ptr<Buffer> &b,int rows,int cols){
    return {(__bridge void*)b->value,b->value.length,0,rows,cols,size_t(cols)*4,DType::FP32,b,b};
}
DeviceWeightView weight(const std::shared_ptr<Buffer> &b,int rows,int cols){
    return {(__bridge void*)b->value,b->value.length,0,size_t(cols)*4,rows,cols,DeviceWeightEncoding::Dense,DType::FP32,32,{},{},b,b};
}
void raw(const std::filesystem::path&p,const std::shared_ptr<Buffer>&b){std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(b->value.contents),b->value.length);require(bool(f),"raw export failed");}
void json(const std::filesystem::path&p,NSDictionary*v){NSError*e=nil;NSData*d=[NSJSONSerialization dataWithJSONObject:v options:NSJSONWritingPrettyPrinted error:&e];require(d!=nil,"JSON encoding failed");std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(d.bytes),d.length);require(bool(f),"JSON write failed");}
NSDictionary *policy(const ProgramPolicyStats&p){return @{@"fixed":@(p.fixed),@"instances":@(p.instances),@"blocked_rebuilds":@(p.blocked_rebuilds),@"disabled":@(p.disabled)};}
}

extern "C" __attribute__((visibility("default")))
int tc_w8_functions_probe(const char *directory,void(*scope)(int)) {@autoreleasepool{
    const std::filesystem::path out(directory?directory:"");
    NSMutableDictionary*r=[@{@"schema":@"tc-w8-functions-consumer-v1",@"status":@"starting",@"cpp_sha256":@TC_W8_FUNCTION_CPP_SHA,
        @"compiled_native_id":@TC_W8_FUNCTION_NATIVE_ID,@"synthetic_only":@YES,@"image_export":@NO,@"performance_qualified":@NO,@"h":@(H),@"f":@(F),@"max_rows":@64}mutableCopy];
    try {
        require(directory && std::filesystem::is_directory(out) && std::filesystem::is_empty(out),"fresh bounded probe output required");
        char*id=tc_runtime_build_identity();require(id,"native identity missing");
        const std::string loaded(id);r[@"loaded_runtime_id"]=@(id);tc_string_free(id);
        require(loaded==TC_W8_FUNCTION_NATIVE_ID,"W8 helper/native C++ ABI mismatch; no Surface may be constructed");
        setenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM","1",1);
        if(!std::getenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"))setenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION","exp",1);
        r[@"activation"]=@(std::getenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"));
        setenv("TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD","0",1);setenv("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE","1",1);
        setenv("TURBOCIDER_PRIVATE_ANE_SCALE_CACHE","0",1);setenv("TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES","0",1);
        gpu::Device device;
        gpu::Surface root(device,8,64,gpu::Element::FP16);auto v=root.leading_columns(32);
        const auto a=root.metal_buffer_lease(),b=v.metal_buffer_lease();
        require(a.native_buffer==b.native_buffer && a.owner==b.owner && a.allocated_bytes==b.allocated_bytes &&
            a.row_stride_bytes==b.row_stride_bytes && b.cols==32 && !root.is_view() && v.is_view(),"column view reallocated or lost pitch/owner");
        int rejected=0;for(int count:{0,65})try{(void)root.leading_columns(count);}catch(const CapabilityError&){++rejected;}
        try{(void)v.leading_columns(64);}catch(const CapabilityError&){++rejected;}
        require(rejected==3,"invalid column view admitted");r[@"column_view_negative_controls"]=@(rejected);
        const GraphShape shape{Kind::SwiGLU,64,H,F,128,512,false};
        if(scope)scope(0);
        PrivateW8Graph legacy(shape,768u<<20,out/"legacy-cache");
        PrivateW8Graph family(shape,768u<<20,out/"family-cache",W8Basis::SylvesterDH,{32,64});
        std::string error;require(legacy.self_test(error),error);require(family.self_test(error),error);
        require(legacy.slot_bytes()==family.slot_bytes(),"function family duplicated physical slots");
        r[@"legacy_slot_bytes"]=@(legacy.slot_bytes());r[@"family_slot_bytes"]=@(family.slot_bytes());
        r[@"same_slot_bytes"]=@YES;r[@"family_recipe"]=@(family.weight_recipe().c_str());
        auto input=std::make_shared<Buffer>(device.native_device(),size_t(N)*H*4);
        std::array<std::shared_ptr<Buffer>,2> outputs{std::make_shared<Buffer>(device.native_device(),size_t(N)*H*4),std::make_shared<Buffer>(device.native_device(),size_t(N)*H*4)};
        std::array<std::array<std::shared_ptr<Buffer>,3>,2> weights;
        for(int variant=0;variant<2;++variant) {
            weights[variant]={std::make_shared<Buffer>(device.native_device(),F*H*4),std::make_shared<Buffer>(device.native_device(),F*H*4),std::make_shared<Buffer>(device.native_device(),H*F*4)};
            for(auto &w:weights[variant])std::fill_n(w->data(),w->value.length/4,0.f);
            for(int c=0;c<F;++c){weights[variant][0]->data()[c*H+c%H]=variant?.25f:.125f;weights[variant][1]->data()[c*H+c%H]=variant?.25f:.125f;}
            for(int c=0;c<H;++c)weights[variant][2]->data()[c*F+c]=variant?-.5f:.25f;
            for(int j=0;j<3;++j)raw(out/("weight-"+std::to_string(variant)+"-"+std::to_string(j)+".bin"),weights[variant][j]);
        }
        NSMutableArray*runs=[NSMutableArray new];r[@"runs"]=runs;
        for(int point=0;point<4;++point) {
            const int rows=point==1?64:point==2?96:32,variant=point==2;
            const std::string prefix="run-"+std::to_string(point);
            std::fill_n(input->data(),N*H,0.f);
            for(int row=0;row<rows;++row)for(int c=0;c<H;++c)input->data()[row*H+c]=
                point==1||point==2?float((row+(point==1?3:0))%8-4)*.5f+float(c%8-3)*.125f:
                4.f+float(row%8)*.125f+float(c%8)*.0625f;
            raw(out/(prefix+"-input.bin"),input);
            NSMutableArray*arms=[NSMutableArray new];
            for(int arm=0;arm<2;++arm) {
                auto &graph=arm?family:legacy;const int launch_rows=arm?rows:((rows+63)/64)*64;
                std::fill_n(outputs[arm]->data(),N*H,1234.f);
                if(scope)scope(1+point*2+arm);
                graph.stage_device_weights({weight(weights[variant][0],F,H),weight(weights[variant][1],F,H),weight(weights[variant][2],H,F)});
                require(graph.wait_stage().ok,"weight stage failed");
                graph.launch_device(matrix(input,launch_rows,H),matrix(outputs[arm],launch_rows,H));
                const auto done=graph.finish();require(done.ok && done.overflow_retries==0 && done.headroom_scale==1.f,done.error);
                const int expected_calls=(rows+63)/64;require(done.calls==uint64_t(expected_calls),"actual function/chunk call count differs");
                raw(out/(prefix+(arm?"-family.bin":"-legacy.bin")),outputs[arm]);
                double energy=0,delta=0;bool canary=true;
                for(int row=0;row<N;++row)for(int c=0;c<H;++c) {
                    const float actual=outputs[arm]->data()[row*H+c];require(std::isfinite(actual),"nonfinite full consumer output");
                    if(row>=launch_rows)canary=canary && actual==1234.f;
                    if(row<rows){const double x=input->data()[row*H+c],g=x*(variant?.25:.125),u=g;
                        const double reference=(g/(1.+std::exp(-g)))*u*(variant?-.5:.25);
                        energy+=reference*reference;delta+=(actual-reference)*(actual-reference);}
                }
                const double relative=std::sqrt(delta/std::max(energy,1e-30));
                require(relative<.06 && canary,"complete SwiGLU consumer/reference or destination tail failed");
                [arms addObject:@{@"arm":arm?@"family":@"legacy",@"launch_rows":@(launch_rows),@"calls":@(done.calls),
                    @"relative_l2_cpu_reference":@(relative),@"untouched_destination_tail":@(canary),@"policy":policy(graph.program_policy_stats())}];
            }
            double energy=0,delta=0;for(int i=0;i<rows*H;++i){const double l=outputs[0]->data()[i],f=outputs[1]->data()[i];energy+=l*l;delta+=(f-l)*(f-l);}
            const double relative=std::sqrt(delta/std::max(energy,1e-30));require(relative<.005,"family changed arithmetic beyond declared component budget");
            [runs addObject:@{@"index":@(point),@"rows":@(rows),@"weight_variant":@(variant),@"arms":arms,@"family_vs_legacy_relative_l2":@(relative)}];
            json(out/"report.json",r);
        }
        for(auto *graph:{&legacy,&family}){const auto p=graph->program_policy_stats();require(p.fixed&&p.instances==1&&!p.disabled&&!p.blocked_rebuilds,"fixed program rebuilt or failed");}
        r[@"status"]=@"complete";json(out/"report.json",r);return 0;
    }catch(const std::exception&e){r[@"status"]=@"failed";r[@"error"]=@(e.what());if(std::filesystem::is_directory(out))json(out/"report.json",r);return 1;}
}}
