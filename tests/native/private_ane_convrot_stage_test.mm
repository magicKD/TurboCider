#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/ane_w8a8_math.hpp"
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>
#include <limits>

#pragma clang fp contract(off)
using namespace tc::ane;
using namespace tc::ane::private_api;
namespace {
void check(bool ok,const char *message) { if(!ok)throw std::runtime_error(message); }
struct Buffer {
    id<MTLBuffer> value;
    std::shared_ptr<void> owner;
    Buffer(id<MTLDevice> gpu,size_t bytes) {
        value=[gpu newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        check(value!=nil,"allocation failed");
        owner={(__bridge_retained void*)value,[](void*p){CFRelease(p);}};
        std::memset(value.contents,0,bytes);
    }
    DeviceMatrixView matrix(int rows,int cols,DType dtype,size_t pitch,size_t offset=0) {
        return {(__bridge void*)value,value.length,offset,rows,cols,pitch,dtype,owner};
    }
};
size_t item_size(DType dtype) {return dtype==DType::FP32?4:2;}
float store(void *p,float x,DType dtype) {
    if(dtype==DType::FP32) {std::memcpy(p,&x,4);return x;}
    const uint16_t bits=!std::isfinite(x)?(dtype==DType::BF16?0x7f80:0x7c00):
        dtype==DType::BF16?tc::gguf::float_to_bf16_rne(x):tc::gguf::float_to_fp16_rne(x);
    std::memcpy(p,&bits,2);
    return dtype==DType::BF16?std::bit_cast<float>(uint32_t(bits)<<16):tc::gguf::fp16_to_float(bits);
}
void padding(const Surface &s,size_t row_bytes) {
    for(uint32_t r=0;r<s.rows();++r)for(size_t c=row_bytes;c<s.pitch();++c)
        check(static_cast<const uint8_t*>(s.data())[r*s.pitch()+c]==0x5a,"target padding overwritten");
}
void fill(Surface &s) {std::memset(s.data(),0x5a,s.rows()*s.pitch());}
}
int main() {
    @autoreleasepool {
      try {
        auto gpu=MTLCreateSystemDefaultDevice();
        setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","0",1);Device generic;
        setenv("TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE","1",1);Device specialized;
        constexpr int rows=5,cols=1024;
        int cases=0;
        for(bool packed:{false,true})for(int group:{32,64,128,256})
            for(DType md:{DType::FP16,DType::BF16,DType::FP32})for(DType bd:{DType::FP16,DType::BF16,DType::FP32}) {
            if(!packed && (group!=32 || md!=DType::FP32 || bd!=DType::FP32))continue;
            const int groups=packed?cols/group:1;
            const size_t pitch=cols+17,offset=257,mpitch=groups*item_size(md)+16,bpitch=groups*item_size(bd)+16;
            Buffer source(gpu,offset+rows*pitch),meta(gpu,4+rows*mpitch),bias(gpu,4+rows*bpitch);
            const float scales[rows]={.002f,0,-.001953125f,0x1p-25f,.003f};
            float stored_scales[rows]{};
            for(int r=0;r<rows;++r) {
                auto *codes=static_cast<uint8_t*>(source.value.contents)+offset+r*pitch;
                for(int c=0;c<cols;++c) {const int code=(r*7+c)%256-128;codes[c]=packed?uint8_t(code+128):uint8_t(int8_t(code));}
                for(int g=0;g<groups;++g) {
                    const float s=store(static_cast<char*>(meta.value.contents)+4+r*mpitch+g*item_size(md),scales[r],md);
                    stored_scales[r]=s;
                    store(static_cast<char*>(bias.value.contents)+4+r*bpitch+g*item_size(bd),-128.f*s,bd);
                }
            }
            DeviceWeightView view{(__bridge void*)source.value,source.value.length,offset,pitch,rows,cols,
                packed?DeviceWeightEncoding::ConvrotQ8Packed:DeviceWeightEncoding::ConvrotQ8Signed,DType::FP32,group,
                meta.matrix(rows,groups,md,mpitch,4),std::nullopt,source.owner};
            if(packed)view.offsets=bias.matrix(rows,groups,bd,bpitch,4);
            const W8StageSpec spec{1,3,256,512,256,0,false,W8Basis::ComfyH256};
            Surface q(generic,3,512,Element::I8),s(generic,3,1,Element::FP16),
                fast_q(specialized,3,512,Element::I8),fast_s(specialized,3,1,Element::FP16);
            fill(q);fill(s);fill(fast_q);fill(fast_s);
            auto job=generic.stage_w8(view,spec,q,s),fast=specialized.stage_w8(view,spec,fast_q,fast_s);
            check(job.finish().ok && fast.finish().ok,"direct W staging failed");
            check(!std::memcmp(q.data(),fast_q.data(),q.rows()*q.pitch()) &&
                  !std::memcmp(s.data(),fast_s.data(),s.rows()*s.pitch()),"direct generic/specialized mismatch");
            for(int r=0;r<3;++r) {
                const auto expected=tc::gguf::float_to_fp16_rne(stored_scales[r+1]*128.f);
                check(*reinterpret_cast<const uint16_t*>(static_cast<const char*>(s.data())+r*s.pitch())==expected,"signed/zero row scale changed");
                for(int c=0;c<512;++c)check(static_cast<const int8_t*>(q.data())[r*q.pitch()+c]==(int((r+1)*7+c+256)%256-128),"original signed code changed");
            }
            padding(q,512);padding(s,2);
            check(generic.value()==0 && specialized.value()==0,"direct staging advanced inference timeline");
            auto wrong=spec;wrong.basis=W8Basis::SylvesterDH;wrong.rotation_block=128;
            try {generic.stage_w8(view,wrong,q,s);throw std::runtime_error("wrong basis accepted");}catch(const CapabilityError&){}
            if(packed) {
                // Last physical group is OUTSIDE the selected column slice.
                auto *at=static_cast<char*>(bias.value.contents)+4+2*bpitch+(groups-1)*item_size(bd);
                store(at,1,bd);
                auto bad=generic.stage_w8(view,spec,q,s);
                check(!bad.finish().ok && (bad.validation_flags()&16),"unselected malformed offset accepted");
                store(at,-128.f*scales[2],bd);
                auto *sa=static_cast<char*>(meta.value.contents)+4+2*mpitch+(groups-1)*item_size(md);
                store(sa,scales[2]*2,md);
                auto bad_scale=generic.stage_w8(view,spec,q,s);
                check(!bad_scale.finish().ok && (bad_scale.validation_flags()&16),"unselected nonuniform scale accepted");
                store(sa,scales[2],md);
            }
            auto *sa=static_cast<char*>(meta.value.contents)+4+2*mpitch;
            store(sa,std::numeric_limits<float>::infinity(),md);
            auto bad=generic.stage_w8(view,spec,q,s);check(!bad.finish().ok,"nonfinite direct scale accepted");
            store(sa,scales[2],md);
            auto clean=generic.stage_w8(view,spec,q,s);check(clean.finish().ok,"direct fresh refill failed");
            ++cases;
        }
        check(cases==37,"direct case count wrong");
        std::cout<<"PASS 37 direct raw/packed Q8 cases: original -128, signed/zero/tiny scales, physical strides/slices, full-row metadata and recovery\n";
        for(DType dtype:{DType::FP16,DType::BF16,DType::FP32})for(int k:{1024,3840,10240}) {
            constexpr int physical_rows=35,m=33;
            const size_t item=item_size(dtype),pitch=k*item+17,offset=257;
            Buffer source(gpu,offset+physical_rows*pitch);
            std::vector<float> values(size_t(physical_rows)*k);
            for(int r=0;r<physical_rows;++r)for(int c=0;c<k;++c) {
                float x=r==1?0:r==2?0x1p-20f:float((r*13+c*17)%1024-512)/1024.f;
                values[r*k+c]=store(static_cast<char*>(source.value.contents)+offset+r*pitch+c*item,x,dtype);
            }
            DeviceWeightView view{(__bridge void*)source.value,source.value.length,offset,pitch,physical_rows,k,
                DeviceWeightEncoding::Dense,dtype,32,{},{},source.owner};
            const int first=k==1024?256:0,count=k==1024?512:k;
            const W8StageSpec spec{1,m,first,count,256,0,true,W8Basis::ComfyH256};
            Surface q(generic,count,m,Element::I8),s(generic,1,m,Element::FP16),
                fast_q(specialized,count,m,Element::I8),fast_s(specialized,1,m,Element::FP16);
            fill(q);fill(s);fill(fast_q);fill(fast_s);
            auto job=generic.stage_w8(view,spec,q,s),fast=specialized.stage_w8(view,spec,fast_q,fast_s);
            check(job.finish().ok && fast.finish().ok,"Comfy A8 staging failed");
            check(!std::memcmp(q.data(),fast_q.data(),q.rows()*q.pitch()) &&
                !std::memcmp(s.data(),fast_s.data(),s.rows()*s.pitch()),"Comfy A8 specialization mismatch");
            for(int r=0;r<m;++r) {
                std::vector<float> rotated(values.begin()+(r+1)*k+first,values.begin()+(r+1)*k+first+count);
                for(int c=0;c<count;c+=256)rotate_comfy_block({rotated.data()+c,256},dtype);
                float peak=0;for(float x:rotated)peak=std::max(peak,std::abs(x));
                const auto expected=normalized_scale(peak);
                check(static_cast<const uint16_t*>(s.data())[r]==expected,"Comfy A8 scale/source rounding mismatch");
                for(int c=0;c<count;++c)check(static_cast<const int8_t*>(q.data())[c*q.pitch()+r]==quantize_rotated(rotated[c],expected),"Comfy A8 code/source rounding mismatch");
            }
            padding(q,m);padding(s,m*2);
        }
        std::cout<<"PASS 9 Comfy H256 A8 typed/strided cases: K512/3840/10240, exact CPU radix-4/source rounding/RNE and padding\n";
        // Direct scale overflow/underflow is a failed stage, never zeroed W
        // reported as successful. No quantizer floor may alter source scales.
        Buffer codes(gpu,256),meta(gpu,4);
        DeviceWeightView view{(__bridge void*)codes.value,codes.value.length,0,256,1,256,
            DeviceWeightEncoding::ConvrotQ8Signed,DType::FP32,32,meta.matrix(1,1,DType::FP32,4),{},codes.owner};
        Surface q(generic,1,256,Element::I8),s(generic,1,1,Element::FP16);
        for(float value:{1e6f,1e-30f}) {
            store(meta.value.contents,value,DType::FP32);
            auto bad=generic.stage_w8(view,{0,1,0,256,256,0,false,W8Basis::ComfyH256},q,s);
            check(!bad.finish().ok && (bad.validation_flags()&4),"direct scale range failure hidden");
        }
        std::cout<<"PASS direct scale finite/range rejection; no weight requantization or quantizer-floor substitution\n";
        // Signed scale policy is local to direct down restore. It never
        // relaxes the strictly positive activation/token scale contract.
        Surface norm(generic,3,1,Element::FP16),ws(generic,3,1,Element::FP16),xs(generic,1,1,Element::FP16);
        Buffer result(gpu,6);
        const float signed_scales[3]={-2,0,2};
        for(int r=0;r<3;++r) {
            *reinterpret_cast<uint16_t*>(static_cast<char*>(norm.data())+r*norm.pitch())=0x3c00;
            *reinterpret_cast<uint16_t*>(static_cast<char*>(ws.data())+r*ws.pitch())=tc::gguf::float_to_fp16_rne(signed_scales[r]);
        }
        *static_cast<uint16_t*>(xs.data())=0x3c00;
        Download download{norm,result.matrix(1,3,DType::BF16,6),0,DType::BF16,1,ws,xs};
        auto positive=generic.prepare_gpu_transfer({}, {download});positive.submit();
        check(positive.finish().ok && (positive.validation_flags()&2),"default scale policy was weakened");
        download.row_scale_policy=RowScalePolicy::SignedFinite;
        auto signed_restore=generic.prepare_gpu_transfer({}, {download});signed_restore.submit();
        check(signed_restore.finish().ok && !signed_restore.validation_flags(),"signed finite restore failed");
        for(int r=0;r<3;++r)check(static_cast<const uint16_t*>(result.value.contents)[r]==tc::gguf::float_to_bf16_rne(signed_scales[r]),"signed/zero restore changed result");
        for(uint16_t invalid:{uint16_t(0),uint16_t(0xbc00),uint16_t(0x7c00)}) {
            *static_cast<uint16_t*>(xs.data())=invalid;
            auto bad=generic.prepare_gpu_transfer({}, {download});bad.submit();
            check(bad.finish().ok && (bad.validation_flags()&2),"signed policy accepted invalid token scale");
        }
        std::cout<<"PASS recipe-local signed/zero row-scale restore; default positive and token-scale guards unchanged\n";
      } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
    }
}
