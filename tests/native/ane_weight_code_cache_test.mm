#include "../../native/backends/ane_gpu.hpp"
#include "../../native/backends/ane_w8a8_math.hpp"
#import <Metal/Metal.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

using namespace tc::ane;
using namespace tc::ane::gpu;
void check(bool value,const char *message) { if(!value)throw std::runtime_error(message); }
struct Buffer {
    id<MTLBuffer> value;
    std::shared_ptr<void> owner;
    Buffer(id<MTLDevice> device,size_t bytes) {
        value=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        check(value!=nil,"fixture allocation failed");
        owner={(__bridge_retained void*)value,[](void *p){CFRelease(p);}};
        std::memset(value.contents,0,bytes);
    }
};
void put_half(void *where,float value) {
    const auto h=tc::gguf::float_to_fp16_rne(value);std::memcpy(where,&h,2);
}
void wait_live(Device &device,uint64_t expected) {
    const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(device.weight_code_cache_stats().live_capacity_bytes!=expected && std::chrono::steady_clock::now()<end)
        std::this_thread::yield();
    check(device.weight_code_cache_stats().live_capacity_bytes==expected,"completed ticket retained stale capacity");
}
int main() {
    @autoreleasepool {try {
        id<MTLDevice> metal=MTLCreateSystemDefaultDevice();
        int cases=0;
        for(auto encoding:{DeviceWeightEncoding::Dense,DeviceWeightEncoding::AffineQ4,DeviceWeightEncoding::AffineQ8,
            DeviceWeightEncoding::GgufQ4_0,DeviceWeightEncoding::GgufQ4_K,DeviceWeightEncoding::GgufQ8_0,
            DeviceWeightEncoding::GgufQ6_K,DeviceWeightEncoding::ConvrotQ8Signed,DeviceWeightEncoding::ConvrotQ8Packed}) {
          for(auto dtype:{DType::FP16,DType::BF16,DType::FP32}) {
            if(encoding!=DeviceWeightEncoding::Dense && dtype!=DType::FP32)continue;
            constexpr int rows=5,cols=1024;
            const bool dense=encoding==DeviceWeightEncoding::Dense;
            const bool raw=encoding==DeviceWeightEncoding::GgufQ4_0 || encoding==DeviceWeightEncoding::GgufQ4_K ||
                encoding==DeviceWeightEncoding::GgufQ8_0 || encoding==DeviceWeightEncoding::GgufQ6_K;
            const bool comfy=encoding==DeviceWeightEncoding::ConvrotQ8Signed || encoding==DeviceWeightEncoding::ConvrotQ8Packed;
            const bool affine=encoding==DeviceWeightEncoding::AffineQ4 || encoding==DeviceWeightEncoding::AffineQ8 || encoding==DeviceWeightEncoding::ConvrotQ8Packed;
            const int bits=encoding==DeviceWeightEncoding::AffineQ4?4:8;
            const int ggml=encoding==DeviceWeightEncoding::GgufQ4_0?2:encoding==DeviceWeightEncoding::GgufQ4_K?12:
                encoding==DeviceWeightEncoding::GgufQ8_0?8:14;
            const int group=ggml==2 || ggml==8?32:256,block_bytes=ggml==2?18:ggml==12?144:ggml==8?34:210;
            const int item=dtype==DType::FP32?4:2;
            const size_t row_bytes=dense?cols*item:raw?cols/group*block_bytes:cols*bits/8;
            const size_t pitch=row_bytes+17,offset=257;
            Buffer source(metal,offset+rows*pitch+256),meta(metal,rows*256),bias(metal,rows*128);
            auto generation=std::make_shared<int>(1),logical=std::make_shared<int>(1);
            DeviceWeightView view{(__bridge void*)source.value,source.value.length,offset,pitch,rows,cols,encoding,dtype,32,
                {},{},source.owner,generation,true};
            if(raw)view.logical_content_identity=logical;
            if(affine) {
                view.scales=DeviceMatrixView{(__bridge void*)meta.value,meta.value.length,0,rows,cols/32,256,DType::FP32,meta.owner,meta.owner};
                view.offsets=DeviceMatrixView{(__bridge void*)bias.value,bias.value.length,0,rows,cols/32,128,DType::BF16,bias.owner,bias.owner};
            } else if(comfy) {
                view.scales=DeviceMatrixView{(__bridge void*)meta.value,meta.value.length,0,rows,1,256,DType::FP32,meta.owner,meta.owner};
            }
            for(int r=0;r<rows;++r) {
                auto *row=static_cast<uint8_t*>(source.value.contents)+offset+r*pitch;
                if(dense)for(int c=0;c<cols;++c) {
                    const float f=float((r*31+c*7)%127-63)*.003f;
                    if(dtype==DType::FP32)std::memcpy(row+c*4,&f,4);
                    else {const auto h=dtype==DType::FP16?tc::gguf::float_to_fp16_rne(f):tc::gguf::float_to_bf16_rne(f);std::memcpy(row+c*2,&h,2);}
                } else if(raw) {
                    for(size_t c=0;c<row_bytes;++c)row[c]=uint8_t(r*13+c*19);
                    for(int g=0;g<cols/group;++g) {
                        put_half(row+g*block_bytes+(ggml==14?208:0),.002f);
                        if(ggml==12)put_half(row+g*block_bytes+2,.001f);
                    }
                } else for(size_t c=0;c<row_bytes;++c)row[c]=uint8_t(c*7+r*11);
                if(view.scales) {
                    auto *sc=static_cast<uint8_t*>(meta.value.contents)+r*256;
                    for(int g=0;g<view.scales->cols;++g) {
                        const float scale=.003f;std::memcpy(sc+g*4,&scale,4);
                        if(view.offsets) {
                            const float b=comfy?-128.f*scale:.004f;
                            const auto h=tc::gguf::float_to_bf16_rne(b);
                            // Direct packed Comfy requires EXACT -128 * stored
                            // scale, so use F32 offset metadata for that case.
                            if(comfy)std::memcpy(static_cast<uint8_t*>(bias.value.contents)+r*128+g*4,&b,4);
                            else std::memcpy(static_cast<uint8_t*>(bias.value.contents)+r*128+g*2,&h,2);
                        }
                    }
                }
            }
            if(encoding==DeviceWeightEncoding::ConvrotQ8Packed) {
                view.offsets->dtype=DType::FP32;view.offsets->row_stride_bytes=128;
            }
            Device control(false,true,0),cached(false,true,1<<20);
            W8StageSpec spec{1,3,comfy?256:128,comfy?512:384,comfy?256:128,comfy?0u:20260930u,false,
                comfy?W8Basis::ComfyH256:W8Basis::SylvesterDH};
            Surface reference(control,5,spec.columns,Element::I8),rs(control,5,1,Element::FP16),
                    candidate(cached,5,spec.columns,Element::I8),cs(cached,5,1,Element::FP16);
            auto prepare=[&](Device &device,Surface codes,Surface scales,const DeviceWeightView &input) {
                std::memset(codes.data(),0x5a,codes.rows()*codes.pitch());
                std::memset(scales.data(),0x5a,scales.rows()*scales.pitch());
                auto job=device.stage_w8(input,spec,codes.slice_rows(1,3),scales.slice_rows(1,3));
                check(job.finish().ok,"valid code cache fixture failed");
            };
            prepare(control,reference,rs,view);prepare(cached,candidate,cs,view);prepare(cached,candidate,cs,view);
            check(!std::memcmp(reference.data(),candidate.data(),5*reference.pitch()) &&
                  !std::memcmp(rs.data(),cs.data(),5*rs.pitch()),"cached codes/scales or untouched target padding changed");
            auto stats=cached.weight_code_cache_stats();
            check(stats.fills==1 && stats.hits==1 && stats.entries==1 && stats.ready_entries==1 &&
                  stats.live_capacity_bytes<=stats.budget_bytes && stats.peak_capacity_bytes<=stats.budget_bytes,"completed code cache evidence missing");
            // A/B/A under distinct live content generations must not overwrite
            // A's immutable cached buffers. Raw logical tags take precedence.
            auto other=view;auto other_generation=std::make_shared<int>(2);
            if(raw)other.logical_content_identity=other_generation;else other.allocation_identity=other_generation;
            prepare(cached,candidate,cs,other);prepare(cached,candidate,cs,view);
            check(cached.weight_code_cache_stats().fills==2 && cached.weight_code_cache_stats().hits==2,"A/B/A generation separation failed");
            Buffer different(metal,source.value.length);
            std::memcpy(different.value.contents,source.value.contents,source.value.length);
            for(int r=0;r<rows;++r) {
                auto *row=static_cast<uint8_t*>(different.value.contents)+offset+r*pitch;
                if(dense)for(int c=0;c<cols;++c)row[c*item+item-1]^=0x80; // finite sign change, independent B payload
                else if(raw)for(int g=0;g<cols/group;++g)row[g*block_bytes+(ggml==12?16:ggml==14?0:2)]^=1;
                else for(size_t c=0;c<row_bytes;++c)row[c]^=comfy?0x80:1;
            }
            auto changed=view;auto changed_generation=std::make_shared<int>(3);
            changed.buffer=(__bridge void*)different.value;changed.owner=different.owner;
            changed.allocation_identity=changed_generation;
            if(raw)changed.logical_content_identity=changed_generation;
            prepare(control,reference,rs,changed);prepare(cached,candidate,cs,changed);
            check(!std::memcmp(reference.data(),candidate.data(),5*reference.pitch()) &&
                  !std::memcmp(rs.data(),cs.data(),5*rs.pitch()),"independent B cached payload changed");
            prepare(control,reference,rs,view);prepare(cached,candidate,cs,view);
            check(!std::memcmp(reference.data(),candidate.data(),5*reference.pitch()) &&
                  !std::memcmp(rs.data(),cs.data(),5*rs.pitch()),"B fill overwrote immutable A cache");
            if(raw) {
                Buffer refill(metal,source.value.length);
                std::memcpy(refill.value.contents,source.value.contents,source.value.length);
                auto moved=view;moved.buffer=(__bridge void*)refill.value;moved.owner=refill.owner;moved.allocation_identity=refill.owner;
                const auto before_hits=cached.weight_code_cache_stats().hits;prepare(cached,candidate,cs,moved);
                check(cached.weight_code_cache_stats().hits==before_hits+1,"verified raw logical content did not survive a physical refill");
            }
            auto mutable_view=view;mutable_view.immutable_generation=false;mutable_view.logical_content_identity.reset();
            const auto hits=cached.weight_code_cache_stats().hits;prepare(cached,candidate,cs,mutable_view);
            check(cached.weight_code_cache_stats().hits==hits && cached.weight_code_cache_stats().ineligible>0,"mutable source cached");
            cached.clear_weight_code_cache();wait_live(cached,0);++cases;
          }
        }
        // Clear while a completed producer ticket is retained: no false
        // capacity release and no allocator/address-only reuse on refill.
        Device leased(false,true,32768);Buffer input(metal,5*512*4);
        auto tag=std::make_shared<int>(1);
        DeviceWeightView view{(__bridge void*)input.value,input.value.length,0,512*4,5,512,
            DeviceWeightEncoding::Dense,DType::FP32,32,{},{},input.owner,tag,true};
        W8StageSpec spec{0,5,0,512,128};Surface codes(leased,5,512,Element::I8),scales(leased,5,1,Element::FP16);
        {
            auto first=leased.stage_w8(view,spec,codes,scales);check(first.finish().ok,"first leased fill failed");
            const auto held=leased.weight_code_cache_stats().live_capacity_bytes;
            check(held>0 && held<=32768,"first producer did not hold a bounded cache allocation");
            leased.clear_weight_code_cache();check(leased.weight_code_cache_stats().live_capacity_bytes==held,"escaped producer lost capacity lease");
            auto second=leased.stage_w8(view,spec,codes,scales);check(second.finish().ok,"declined cache must still run original staging");
            check(leased.weight_code_cache_stats().entries==0 && leased.weight_code_cache_stats().declines==1,"escaped lease allowed oversize cache refill");
        }
        wait_live(leased,0);
        // Validation failure must never publish a ready entry. New generation
        // and healthy data can subsequently refill the same physical buffer.
        auto failed_tag=std::make_shared<int>(3);view.allocation_identity=failed_tag;
        auto *values=static_cast<float*>(input.value.contents);values[0]=INFINITY;
        {
            auto failed=leased.stage_w8(view,spec,codes,scales);check(!failed.finish().ok,"nonfinite source admitted");
            check(leased.weight_code_cache_stats().ready_entries==0 && leased.weight_code_cache_stats().failed_fills==1,"failed fill published");
        }
        values[0]=0;auto next_tag=std::make_shared<int>(2);view.allocation_identity=next_tag;
        {auto healthy=leased.stage_w8(view,spec,codes,scales);check(healthy.finish().ok,"healthy cache recovery failed");}
        check(leased.weight_code_cache_stats().ready_entries==1,"healthy generation not published after failure");
        leased.clear_weight_code_cache();wait_live(leased,0);
        // An expired identity is not kept alive by cached converted buffers.
        auto expire=std::make_shared<int>(4);view.allocation_identity=expire;
        {auto job=leased.stage_w8(view,spec,codes,scales);check(job.finish().ok,"expiry fixture fill failed");}
        expire.reset();auto fresh=std::make_shared<int>(5);view.allocation_identity=fresh;
        const auto evictions=leased.weight_code_cache_stats().evictions;
        {auto job=leased.stage_w8(view,spec,codes,scales);check(job.finish().ok,"expired generation ordinary staging failed");}
        check(leased.weight_code_cache_stats().evictions>evictions,"expired source generation retained by code cache");
        leased.clear_weight_code_cache();wait_live(leased,0);
        std::cout<<"PASS weight code cache GPU cases="<<cases<<"; dense/affine/raw/ConvRot exact, generation, refill, padding, leased budget, finite/failure recovery\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
}
