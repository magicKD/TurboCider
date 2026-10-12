// Serial same-binary W/A staging control. Not E2E or physical overlap proof.
#include "../../native/backends/private/ane_program.hpp"
#include "../../native/core/gguf_decode.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace tc::ane;
using namespace tc::ane::private_api;
double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}
namespace {
std::shared_ptr<void> retain(id<MTLBuffer> buffer) {
    if(!buffer)throw std::runtime_error("staging source allocation failed");
    return {(__bridge_retained void*)buffer,[](void *p){CFRelease(p);}};
}
uint32_t random_word(uint32_t x) {
    x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);
}
DeviceWeightView make_source(id<MTLDevice> gpu,int rows,int cols,DeviceWeightEncoding encoding) {
    const bool affine=encoding==DeviceWeightEncoding::AffineQ4 || encoding==DeviceWeightEncoding::AffineQ8;
    const int bits=encoding==DeviceWeightEncoding::AffineQ4?4:8;
    const int raw_group=encoding==DeviceWeightEncoding::GgufQ4_0 || encoding==DeviceWeightEncoding::GgufQ8_0?32:256;
    const int raw_bytes=encoding==DeviceWeightEncoding::GgufQ4_0?18:encoding==DeviceWeightEncoding::GgufQ4_K?144:
        encoding==DeviceWeightEncoding::GgufQ8_0?34:210;
    const size_t pitch=encoding==DeviceWeightEncoding::Dense?size_t(cols)*2:
        affine?size_t(cols)*bits/8:size_t(cols/raw_group)*raw_bytes;
    id<MTLBuffer> input=[gpu newBufferWithLength:size_t(rows)*pitch options:MTLResourceStorageModeShared];
    auto owner=retain(input);
    DeviceWeightView source{(__bridge void*)input,input.length,0,pitch,rows,cols,
        encoding,encoding==DeviceWeightEncoding::Dense?DType::BF16:DType::FP32,32,{},{},owner};
    if(encoding==DeviceWeightEncoding::Dense) {
        auto *data=static_cast<uint16_t*>(input.contents);
        for(size_t i=0;i<size_t(rows)*cols;++i)data[i]=uint16_t(0x3b00u+(i*7)%512u)|uint16_t((i%3==0)?0x8000:0);
    } else {
        auto *data=static_cast<uint8_t*>(input.contents);
        for(size_t i=0;i<input.length;++i)data[i]=uint8_t(random_word(uint32_t(i)));
        if(affine) {
            const size_t meta_pitch=size_t(cols/32)*2;
            id<MTLBuffer> scale=[gpu newBufferWithLength:size_t(rows)*meta_pitch options:MTLResourceStorageModeShared];
            id<MTLBuffer> bias=[gpu newBufferWithLength:size_t(rows)*meta_pitch options:MTLResourceStorageModeShared];
            auto scale_owner=retain(scale),bias_owner=retain(bias);
            auto *s=static_cast<uint16_t*>(scale.contents),*b=static_cast<uint16_t*>(bias.contents);
            for(size_t i=0;i<size_t(rows)*cols/32;++i) {s[i]=uint16_t(0x3a80u+i%64u);b[i]=0xbb00u;}
            source.scales=DeviceMatrixView{(__bridge void*)scale,scale.length,0,rows,cols/32,meta_pitch,DType::BF16,scale_owner};
            source.offsets=DeviceMatrixView{(__bridge void*)bias,bias.length,0,rows,cols/32,meta_pitch,DType::BF16,bias_owner};
        } else {
            const uint16_t scale=tc::gguf::float_to_fp16_rne(.002f),bias=tc::gguf::float_to_fp16_rne(.001f);
            for(int row=0;row<rows;++row)for(int group=0;group<cols/raw_group;++group) {
                auto *block=data+size_t(row)*pitch+group*raw_bytes;
                std::memcpy(block+(encoding==DeviceWeightEncoding::GgufQ6_K?208:0),&scale,2);
                if(encoding==DeviceWeightEncoding::GgufQ4_K)std::memcpy(block+2,&bias,2);
            }
        }
    }
    // Mutable identity deliberately measures both scale and code passes,
    // not immutable scale-cache hits. No full dense decode allocation.
    return source;
}
}
int main(int argc,char **argv) {
    if(argc<2 || argc>3 || (argc==3 && std::string(argv[2])!="packed"))return 2;
    const int repeats=std::stoi(argv[1]);if(repeats<7 || repeats>101 || repeats%2==0)return 2;
    @autoreleasepool {
        try {
            auto gpu=MTLCreateSystemDefaultDevice();
            setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","0",1);Device generic;
            setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","1",1);Device specialized;
            std::vector<DeviceWeightEncoding> encodings{DeviceWeightEncoding::Dense};
            if(argc==3)encodings={DeviceWeightEncoding::AffineQ4,DeviceWeightEncoding::AffineQ8,
                DeviceWeightEncoding::GgufQ4_0,DeviceWeightEncoding::GgufQ4_K,
                DeviceWeightEncoding::GgufQ8_0,DeviceWeightEncoding::GgufQ6_K};
            for(auto encoding:encodings)for(auto geometry:{std::array<int,4>{4096,3840,128,0},{3840,4096,512,0},
                    {5120,4096,128,0},{4096,5120,512,0},
                    {7168,4096,128,0},{4096,7168,512,0},
                    {1056,4096,128,1},{4224,4096,128,1}}) {
                const int rows=geometry[0],cols=geometry[1],block=geometry[2];const bool transpose=geometry[3];
                if(encoding!=DeviceWeightEncoding::Dense && (transpose || rows==5120 || cols==5120))continue;
                auto source=make_source(gpu,rows,cols,encoding);
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
                std::cout<<std::setprecision(17)<<"{\"scope\":\"dense/affine/raw-GGUF H128/H512 staging host readiness; no E2E qualification\",\"source_encoding\":"<<uint32_t(encoding)
                    <<",\"scale_cache_eligible\":false,\"source_pitch\":"<<source.row_stride_bytes<<",\"rows\":"<<rows<<",\"columns\":"<<cols
                    <<",\"block\":"<<block<<",\"transpose\":"<<(transpose?"true":"false")<<",\"codes_scales_padding_exact\":true,\"warmups\":10"
                    <<",\"generic_median\":"<<median(original)<<",\"specialized_register_median\":"<<median(reg)<<",\"speedup\":"<<median(original)/median(reg)<<",\"generic_samples\":[";
                for(size_t i=0;i<original.size();++i)std::cout<<(i?",":"")<<original[i];std::cout<<"],\"register_samples\":[";
                for(size_t i=0;i<reg.size();++i)std::cout<<(i?",":"")<<reg[i];std::cout<<"]}"<<std::endl;
            }
        }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
    }
}
