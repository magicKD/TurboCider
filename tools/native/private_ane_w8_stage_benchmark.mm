// Standalone staging calibration. Run without concurrent inference/GPU jobs.
// This measures weight conversion, NOT full-model acceleration or ANE MACs.
#include "../../native/backends/private/ane_program.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>

using namespace tc::ane;
using namespace tc::ane::private_api;
namespace {
struct Source {
    id<MTLBuffer> buffer;
    DeviceWeightView view;
    Source(id<MTLDevice> gpu, int rows, int cols) {
        buffer = [gpu newBufferWithLength:size_t(rows)*cols*2 options:MTLResourceStorageModeShared];
        if (!buffer) throw std::runtime_error("calibration source allocation failed");
        auto owner=std::shared_ptr<void>((__bridge_retained void*)buffer,[](void*p){CFRelease(p);});
        view={(__bridge void*)buffer,buffer.length,0,size_t(cols)*2,rows,cols,DeviceWeightEncoding::Dense,DType::BF16,32,{},{},owner};
        view.allocation_identity=owner;view.immutable_generation=true;
        auto *values=static_cast<uint16_t*>(buffer.contents);
        for(size_t i=0;i<size_t(rows)*cols;++i) values[i]=uint16_t(0x3b00u+(i*7)%512u);
    }
};
struct Bank {
    Surface g,sg,u,su,d,sd;
    Bank(Device&device,int h,int f):g(device,f,h,Element::I8),sg(device,f,1,Element::FP16),
        u(device,f,h,Element::I8),su(device,f,1,Element::FP16),d(device,h,f,Element::I8),sd(device,h,1,Element::FP16){}
    size_t bytes()const{return g.bytes()+sg.bytes()+u.bytes()+su.bytes()+d.bytes()+sd.bytes();}
};
}
int main(int argc,char**argv) {
    if(argc!=2||(std::string(argv[1])!="z"&&std::string(argv[1])!="qwen"))return 2;
    @autoreleasepool {
      try {
        const bool qwen=std::string(argv[1])=="qwen";
        const int h=qwen?4096:3840,f=qwen?12288:10240;
        auto gpu=MTLCreateSystemDefaultDevice();Device device;
        Source g(gpu,f,h),u(gpu,f,h),d(gpu,h,f);
        std::array<Bank,2>banks{Bank(device,h,f),Bank(device,h,f)};
        std::vector<double>samples;
        for(int i=0;i<9;++i) {
            auto&bank=banks[i%2]; const auto start=std::chrono::steady_clock::now();
            auto gs=device.stage_w8(g.view,{0,f,0,h,128},bank.g,bank.sg);
            auto us=device.stage_w8(u.view,{0,f,0,h,128},bank.u,bank.su);
            auto ds=device.stage_w8(d.view,{0,h,0,f,512},bank.d,bank.sd);
            const auto gr=gs.finish(),ur=us.finish(),dr=ds.finish();
            if(!gr.ok||!ur.ok||!dr.ok)throw std::runtime_error("calibration staging failed");
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(i)samples.push_back(seconds); // shader/pipeline first-use excluded
        }
        auto sorted=samples;std::sort(sorted.begin(),sorted.end());
        const double median=(sorted[3]+sorted[4])*.5;
        std::cout<<"{\"scope\":\"GPU dense BF16 gate/up/down to W8 H128/H512, host readiness included\",\"model_geometry\":\""
            <<argv[1]<<"\",\"hidden\":"<<h<<",\"width\":"<<f<<",\"banks\":2,\"bank_bytes\":"
            <<banks[0].bytes()+banks[1].bytes()<<",\"median_stage_seconds\":"<<median<<",\"samples\":[";
        for(size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];
        auto cache=device.scale_cache_stats();
        auto pipelines=device.stage_pipeline_stats();
        std::cout<<"],\"scale_cache_enabled\":"<<(cache.enabled?"true":"false")<<",\"scale_cache_hits\":"<<cache.hits
                 <<",\"scale_cache_misses\":"<<cache.misses<<",\"scale_cache_bytes\":"<<cache.bytes
                 <<",\"stage_specialized\":"<<(pipelines.specialized?"true":"false")
                 <<",\"stage_pipeline_variants\":"<<pipelines.variants<<"}\n";
      }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
    }
}
