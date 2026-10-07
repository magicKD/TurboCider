// Serial same-binary W/A staging control. Not E2E or physical overlap proof.
#include "../../native/backends/private/ane_program.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace tc::ane;
using namespace tc::ane::private_api;
double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    const int repeats=std::stoi(argv[1]);if(repeats<7 || repeats>101 || repeats%2==0)return 2;
    @autoreleasepool {
        try {
            auto gpu=MTLCreateSystemDefaultDevice();
            setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","0",1);Device generic;
            setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","1",1);Device specialized;
            for(auto geometry:{std::array<int,4>{4096,3840,128,0},{3840,4096,512,0},
                    {5120,4096,128,0},{4096,5120,512,0},
                    {7168,4096,128,0},{4096,7168,512,0},
                    {1056,4096,128,1},{4224,4096,128,1}}) {
                const int rows=geometry[0],cols=geometry[1],block=geometry[2];const bool transpose=geometry[3];
                id<MTLBuffer> input=[gpu newBufferWithLength:size_t(rows)*cols*2 options:MTLResourceStorageModeShared];
                if(!input)throw std::runtime_error("staging source allocation failed");
                auto *data=static_cast<uint16_t*>(input.contents);
                for(size_t i=0;i<size_t(rows)*cols;++i)data[i]=uint16_t(0x3b00u+(i*7)%512u)|uint16_t((i%3==0)?0x8000:0);
                auto owner=std::shared_ptr<void>((__bridge_retained void*)input,[](void *p){CFRelease(p);});
                DeviceWeightView source{(__bridge void*)input,input.length,0,size_t(cols)*2,rows,cols,
                    DeviceWeightEncoding::Dense,DType::BF16,32,{},{},owner};
                Surface codes0(generic,transpose?cols:rows,transpose?rows:cols,Element::I8);
                Surface scales0(generic,transpose?1:rows,transpose?rows:1,Element::FP16);
                Surface codes1(specialized,codes0.rows(),codes0.columns(),Element::I8);
                Surface scales1(specialized,scales0.rows(),scales0.columns(),Element::FP16);
                std::memset(codes0.data(),0x5a,codes0.rows()*codes0.pitch());std::memset(codes1.data(),0x5a,codes1.rows()*codes1.pitch());
                std::memset(scales0.data(),0x5a,scales0.rows()*scales0.pitch());std::memset(scales1.data(),0x5a,scales1.rows()*scales1.pitch());
                W8StageSpec spec{0,rows,0,cols,block,20260930,transpose};
                auto run=[&](Device &device,Surface &codes,Surface &scales) {
                    const auto start=std::chrono::steady_clock::now();auto staged=device.stage_w8(source,spec,codes,scales);
                    auto done=staged.finish();if(!done.ok)throw std::runtime_error("staging failed");
                    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                };
                for(int i=0;i<10;++i){run(generic,codes0,scales0);run(specialized,codes1,scales1);}
                std::vector<double> original,reg;
                for(int i=0;i<repeats;++i) {
                    if(i%2){reg.push_back(run(specialized,codes1,scales1));original.push_back(run(generic,codes0,scales0));}
                    else {original.push_back(run(generic,codes0,scales0));reg.push_back(run(specialized,codes1,scales1));}
                }
                const bool same=!std::memcmp(codes0.data(),codes1.data(),codes0.rows()*codes0.pitch()) &&
                    !std::memcmp(scales0.data(),scales1.data(),scales0.rows()*scales0.pitch());
                if(!same)throw std::runtime_error("generic/register codes/scales/padding differ");
                std::cout<<std::setprecision(17)<<"{\"scope\":\"dense BF16 W/A H128/H512 staging host readiness; no E2E qualification\",\"rows\":"<<rows<<",\"columns\":"<<cols
                    <<",\"block\":"<<block<<",\"transpose\":"<<(transpose?"true":"false")<<",\"codes_scales_padding_exact\":true,\"warmups\":10"
                    <<",\"generic_median\":"<<median(original)<<",\"specialized_register_median\":"<<median(reg)<<",\"speedup\":"<<median(original)/median(reg)<<",\"generic_samples\":[";
                for(size_t i=0;i<original.size();++i)std::cout<<(i?",":"")<<original[i];std::cout<<"],\"register_samples\":[";
                for(size_t i=0;i<reg.size();++i)std::cout<<(i?",":"")<<reg[i];std::cout<<"]}"<<std::endl;
            }
        }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
    }
}
