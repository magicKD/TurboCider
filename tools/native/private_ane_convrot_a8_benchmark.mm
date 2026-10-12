// Same-binary, serial generic/register Comfy A8 staging diagnostic.
// No ANE prediction, physical overlap or model-quality/speed claim.
#include "../../native/backends/private/ane_program.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tc::ane;
using namespace tc::ane::private_api;
namespace {
void check(bool value,const char *message) {if(!value)throw std::runtime_error(message);}
double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());return values[values.size()/2];
}
void samples(const std::vector<double> &values) {
    std::cout<<'[';
    for(size_t i=0;i<values.size();++i)std::cout<<(i?",":"")<<values[i];
    std::cout<<']';
}
}
int main() {
    @autoreleasepool {
      try {
        auto gpu=MTLCreateSystemDefaultDevice();check(gpu!=nil,"Metal unavailable");
        constexpr int warmups=10,repeats=31;
        setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","0",1);Device generic;
        setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","1",1);Device registers;
        std::cout<<std::setprecision(17)<<"{\"schema\":\"tc-convrot-register-a8-staging-v1\","
            "\"scope\":\"GPU BF16 Comfy H256 plus row/group A8 codes/scales; host readiness included; not ANE or E2E qualification\","
            "\"qualification_passed\":false,\"device\":\""<<gpu.name.UTF8String<<"\",\"warmups\":"<<warmups
            <<",\"repeats\":"<<repeats<<",\"cases\":[";
        bool comma=false;
        for(int m:{1056,4224})for(int k:{3840,10240}) {
            id<MTLBuffer> buffer=[gpu newBufferWithLength:size_t(m)*k*2 options:MTLResourceStorageModeShared];
            check(buffer!=nil,"source allocation failed");
            auto owner=std::shared_ptr<void>((__bridge_retained void*)buffer,[](void *p){CFRelease(p);});
            auto *data=static_cast<uint16_t*>(buffer.contents);
            for(size_t i=0;i<size_t(m)*k;++i) {
                // Exact finite BF16 values with signs and nonuniform peaks.
                data[i]=uint16_t(0x3a00u+(i*13+i/256)%1536u)|uint16_t((i%7<3)?0x8000u:0);
            }
            DeviceWeightView source{(__bridge void*)buffer,buffer.length,0,size_t(k)*2,m,k,
                DeviceWeightEncoding::Dense,DType::BF16,32,{},{},owner};
            for(int group:{0,256}) {
                W8StageSpec spec{0,m,0,k,256,0,true,W8Basis::ComfyH256,group};
                Surface q(generic,k,m,Element::I8),s(generic,group?k/256:1,m,Element::FP16),
                    fq(registers,k,m,Element::I8),fs(registers,group?k/256:1,m,Element::FP16);
                std::memset(q.data(),0x5a,q.rows()*q.pitch());std::memset(s.data(),0x5a,s.rows()*s.pitch());
                std::memset(fq.data(),0x5a,fq.rows()*fq.pitch());std::memset(fs.data(),0x5a,fs.rows()*fs.pitch());
                std::vector<double> control,fast;
                auto run=[&](bool optimized) {
                    auto &device=optimized?registers:generic;auto &codes=optimized?fq:q;auto &scales=optimized?fs:s;
                    const auto start=std::chrono::steady_clock::now();
                    auto ticket=device.stage_w8(source,spec,codes,scales);
                    check(ticket.finish().ok,"A8 staging failed");
                    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                };
                for(int i=0;i<warmups;++i) {run(false);run(true);}
                for(int i=0;i<repeats;++i) {
                    double a,b;
                    if(i%2) {b=run(true);a=run(false);}else {a=run(false);b=run(true);}
                    control.push_back(a);fast.push_back(b);
                }
                check(!std::memcmp(q.data(),fq.data(),q.rows()*q.pitch()) &&
                    !std::memcmp(s.data(),fs.data(),s.rows()*s.pitch()),"generic/register final payload or padding mismatch");
                check(generic.value()==0 && registers.value()==0,"staging changed inference event timeline");
                if(comma)std::cout<<',';comma=true;
                std::cout<<"{\"rows\":"<<m<<",\"columns\":"<<k<<",\"activation_group\":"<<group
                    <<",\"output_bytes_per_arm\":"<<q.bytes()+s.bytes()<<",\"payload_and_padding_exact\":true,\"generic_seconds\":";
                samples(control);std::cout<<",\"register_seconds\":";samples(fast);
                std::cout<<",\"generic_median_seconds\":"<<median(control)<<",\"register_median_seconds\":"<<median(fast)
                    <<",\"component_ratio\":"<<median(control)/median(fast)<<"}"<<std::flush;
            }
        }
        std::cout<<"]}\n";
      }catch(const std::exception &error) {std::cerr<<error.what()<<"\n";return 1;}
    }
}
