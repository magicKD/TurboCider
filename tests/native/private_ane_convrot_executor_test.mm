#include "../../native/backends/private/ane_w8_executor.hpp"
#include "../../native/backends/ane_w8a8_math.hpp"
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>

using namespace tc::ane;
namespace {
void check(bool ok,const std::string &message) {if(!ok)throw std::runtime_error(message);}
struct Storage {
    id<MTLBuffer> buffer;
    DeviceMatrixView view;
    Storage(id<MTLDevice> gpu,int rows,int cols,DType dtype=DType::BF16) {
        const size_t item=dtype==DType::FP32?4:2,pitch=size_t(cols)*item+16;
        buffer=[gpu newBufferWithLength:256+size_t(rows)*pitch+256 options:MTLResourceStorageModeShared];
        check(buffer!=nil,"test allocation failed");
        auto owner=std::shared_ptr<void>((__bridge_retained void*)buffer,[](void*p){CFRelease(p);});
        view={(__bridge void*)buffer,buffer.length,256,rows,cols,pitch,dtype,owner};
        std::memset(buffer.contents,0x5a,buffer.length);
    }
    void *row(int r) {return static_cast<char*>(buffer.contents)+view.offset_bytes+r*view.row_stride_bytes;}
    float value(int r,int c) {return view.dtype==DType::FP32?static_cast<float*>(row(r))[c]:std::bit_cast<float>(uint32_t(static_cast<uint16_t*>(row(r))[c])<<16);}
    void guard() {
        const auto *p=static_cast<const uint8_t*>(buffer.contents);
        for(size_t i=0;i<view.offset_bytes;++i)check(p[i]==0x5a,"prefix overwritten");
        for(int r=0;r<view.rows;++r)for(size_t c=size_t(view.cols)*(view.dtype==DType::FP32?4:2);c<view.row_stride_bytes;++c)
            check(static_cast<const uint8_t*>(row(r))[c]==0x5a,"row padding overwritten");
        for(size_t i=view.offset_bytes+view.rows*view.row_stride_bytes;i<buffer.length;++i)check(p[i]==0x5a,"suffix overwritten");
    }
};
struct Weight {
    id<MTLBuffer> codes;
    std::shared_ptr<void> owner;
    Storage scales,offsets;
    int rows,cols;
    bool packed;
    size_t pitch;
    Weight(id<MTLDevice> gpu,int r,int c,bool p) : scales(gpu,r,p?c/32:1,DType::FP32),
        offsets(gpu,r,c/32,DType::FP32),rows(r),cols(c),packed(p),pitch(c+16) {
        codes=[gpu newBufferWithLength:256+size_t(r)*pitch+256 options:MTLResourceStorageModeShared];
        check(codes!=nil,"weight allocation failed");
        owner={(__bridge_retained void*)codes,[](void*p){CFRelease(p);}};
        std::memset(codes.contents,0x5a,codes.length);
    }
    void fill(float value,int column_offset=0) {
        for(int r=0;r<rows;++r) {
            auto *at=static_cast<uint8_t*>(codes.contents)+256+r*pitch;
            std::memset(at,packed?128:0,cols);
            const int index=(r%(cols-column_offset))+column_offset,begin=index/256*256;
            for(int c=0;c<256;++c) {
                const int q=64*comfy_h256_sign(index%256,c);at[begin+c]=packed?uint8_t(q+128):uint8_t(int8_t(q));
            }
            for(int g=0;g<scales.view.cols;++g)static_cast<float*>(scales.row(r))[g]=value/1024.f;
            for(int g=0;g<offsets.view.cols;++g)static_cast<float*>(offsets.row(r))[g]=-128.f*value/1024.f;
        }
    }
    DeviceWeightView view() {
        DeviceWeightView w{(__bridge void*)codes,codes.length,256,pitch,rows,cols,
            packed?DeviceWeightEncoding::ConvrotQ8Packed:DeviceWeightEncoding::ConvrotQ8Signed,DType::FP32,32,scales.view,{},owner};
        if(packed)w.offsets=offsets.view;return w;
    }
};
}
int main(int argc,char **argv) {
    if(argc<2 || argc>4)return 2;
    @autoreleasepool {
      try {
        constexpr int bucket=33,rows=bucket*3,h=512,f=512,physical_width=1024;
        const bool grouped=argc>=3 && std::string(argv[2])=="group256";
        const bool bf16=argc>=3 && std::string(argv[2])=="bf16-values";
        if(argc>=3 && !grouped && !bf16)return 2;
        setenv("TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES",bf16?"1":"0",1);
        const std::string scope=argc==4?argv[3]:"both";
        if(grouped)setenv("TURBOCIDER_PRIVATE_ANE_A8_GROUP_SCOPE",scope.c_str(),1);
        else unsetenv("TURBOCIDER_PRIVATE_ANE_A8_GROUP_SCOPE");
        const int input_group=grouped && scope!="hidden"?256:0,hidden_group=grouped && scope!="input"?256:0;
        setenv("TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE",grouped?"256":"0",1);
        auto gpu=MTLCreateSystemDefaultDevice();
        Storage x(gpu,rows,h),y(gpu,rows,h),hidden(gpu,rows,f),dg(gpu,rows,f,DType::FP32),du(gpu,rows,f,DType::FP32);
        for(int r=0;r<rows;++r) {
            for(int c=0;c<h;++c)static_cast<uint16_t*>(x.row(r))[c]=tc::gguf::float_to_bf16_rne(((r*3+c*7)%17-8)/8.f);
            for(int c=0;c<f;++c) {static_cast<float*>(dg.row(r))[c]=.02f*((r+c)%3-1);static_cast<float*>(du.row(r))[c]=.03f*((r*2+c)%3-1);}
        }
        for(bool packed:{false,true})for(bool lookahead:{false,true}) {
            setenv("TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD",lookahead?"1":"0",1);
            setenv("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE","1",1);
            const auto cache=std::filesystem::path(argv[1])/(packed?"packed":"signed")/(lookahead?"lookahead":"serial");
            PrivateW8Graph graph({Kind::SwiGLU,bucket,h,f,256,512,!bf16},256u<<20,cache,W8Basis::ComfyH256);
            const auto recipe=bf16?convrot_bf16_value_recipe:input_group && hidden_group?convrot_group_w8a8_recipe:
                input_group?convrot_input_group_w8a8_recipe:hidden_group?convrot_hidden_group_w8a8_recipe:convrot_w8a8_recipe;
            check(graph.data_path()=="w8a8_convrot" && graph.weight_recipe()==recipe &&
                graph.activation_group_size()==input_group && graph.hidden_activation_group_size()==hidden_group,"recipe/group identity missing");
            check(graph.slot_bytes()<=graph.estimated_bytes(),"memory estimate understated");
            std::string error;check(graph.self_test(error),error);
            Weight g(gpu,physical_width,h,packed),u(gpu,physical_width,h,packed),d(gpu,h,physical_width,packed);
            auto regions=[&] {return std::vector<DeviceWeightRegion>{{g.view(),{512,f,0,h,256,0,false,W8Basis::ComfyH256}},
                {u.view(),{512,f,0,h,256,0,false,W8Basis::ComfyH256}},
                {d.view(),{0,h,512,f,256,0,false,W8Basis::ComfyH256}}};};
            auto stage=[&](float gv,float uv,float dv) {
                g.fill(gv);u.fill(uv);d.fill(dv,512);
                graph.stage_device_weight_regions(regions());const auto result=graph.wait_stage();check(result.ok,result.error);
            };
            auto run=[&](bool adapter,float gv,float uv,float dv) {
                adapter=adapter && !bf16;
                graph.launch_device(x.view,y.view,adapter?std::optional<DeviceAdapterInput>({dg.view,du.view,hidden.view}):std::nullopt);
                const auto result=graph.finish();check(result.ok,result.error);
                check(result.calls==3 && result.activation_prefetches==(lookahead?2u:0u),"chunk/lookahead counts wrong");
                double diff=0,norm=0,hdiff=0,hnorm=0;
                for(int r=0;r<rows;++r)for(int c=0;c<h;++c) {
                    const float gate=x.value(r,c)*gv+(adapter?static_cast<float*>(dg.row(r))[c]:0),
                        up=x.value(r,c)*uv+(adapter?static_cast<float*>(du.row(r))[c]:0),hv=gate/(1+std::exp(-gate))*up,expected=hv*dv;
                    const float actual=y.value(r,c);
                    check(std::isfinite(actual) && std::abs(actual-expected)<.0003f+.08f*std::abs(expected),"pointwise source oracle failed");
                    diff+=double(actual-expected)*(actual-expected);norm+=double(expected)*expected;
                    if(adapter) {const float actual_h=hidden.value(r,c);check(std::isfinite(actual_h),"nonfinite corrected hidden");
                        hdiff+=double(actual_h-hv)*(actual_h-hv);hnorm+=double(hv)*hv;}
                }
                check(dv==0?diff==0:norm>0 && std::sqrt(diff/norm)<.05,"direct source FFN relative L2 failed");
                check(!adapter || (hnorm>0 && std::sqrt(hdiff/hnorm)<.05),"direct corrected hidden relative L2 failed");
                y.guard();if(adapter)hidden.guard();
            };
            for(bool adapter:{false,true,false}) {stage(.125f,-.25f,.25f);run(adapter,.125f,-.25f,.25f);}
            Storage f32(gpu,rows,h,DType::FP32);
            check(graph.supports_fp32_device_output(),"direct F32 output capability absent");
            graph.launch_device(x.view,f32.view);const auto f32_result=graph.finish();check(f32_result.ok,f32_result.error);
            check(f32_result.copied_output_bytes==size_t(rows)*h*4,"F32 byte receipt incorrect");
            for(int r=0;r<rows;++r)for(int c=0;c<h;++c)
                check(tc::gguf::float_to_bf16_rne(f32.value(r,c))==static_cast<uint16_t*>(y.row(r))[c],"F32 restore changed normalized computation or final BF16 rounding");
            f32.guard();
            if(!bf16) {
                run(true,.125f,-.25f,.25f);
                std::vector<uint8_t> saved_hidden(hidden.buffer.length);
                std::memcpy(saved_hidden.data(),hidden.buffer.contents,saved_hidden.size());
                graph.launch_device(x.view,f32.view,DeviceAdapterInput{dg.view,du.view,hidden.view});
                const auto partial=graph.finish();check(partial.ok,partial.error);
                check(partial.copied_output_bytes==size_t(rows)*(h*4+f*2),"ConvRot F32 partial/hidden byte receipt wrong");
                for(int r=0;r<rows;++r)for(int c=0;c<h;++c)
                    check(tc::gguf::float_to_bf16_rne(f32.value(r,c))==static_cast<uint16_t*>(y.row(r))[c],
                        "ConvRot F32 partial changed base-down boundary");
                check(!std::memcmp(saved_hidden.data(),hidden.buffer.contents,saved_hidden.size()),"ConvRot F32 partial changed LoRA hidden/padding");
                Storage bad_hidden(gpu,rows,f,DType::FP32);
                graph.launch_device(x.view,f32.view,DeviceAdapterInput{dg.view,du.view,bad_hidden.view});
                check(!graph.finish().ok,"experimental F32 output silently widened LoRA hidden ABI");
            } else {
                graph.launch_device(x.view,f32.view,DeviceAdapterInput{dg.view,du.view,hidden.view});
                check(!graph.finish().ok,"base-only BF16 software rounding silently accepted LoRA");
            }
            stage(-.25f,.125f,-.5f);run(true,-.25f,.125f,-.5f);
            stage(.125f,-.25f,0);run(true,.125f,-.25f,0);
            stage(.125f,-.25f,.25f);
            graph.launch_device(x.view,y.view);graph.prefetch_device_weight_regions(regions());
            check(graph.finish().ok,"prefetch corrupted current output");
            auto wrong=regions();wrong[2].selection.basis=W8Basis::SylvesterDH;
            check(!graph.activate_prefetched_weights(wrong),"future bank ignored basis identity");
            auto ready=graph.activate_prefetched_weights(regions());check(ready && ready->ok,"direct future bank failed");
            run(false,.125f,-.25f,.25f);
            // A late failed activation leaves scratch partially computed but
            // returns failure; the model-facing wrapper must recompute ALL.
            *static_cast<uint16_t*>(x.row(bucket))=0x7f80;
            graph.launch_device(x.view,y.view);const auto late=graph.finish();
            check(!late.ok && late.calls==1,"late direct activation failure reported success");
            *static_cast<uint16_t*>(x.row(bucket))=tc::gguf::float_to_bf16_rne(((bucket*3)%17-8)/8.f);
            run(false,.125f,-.25f,.25f);
            auto malformed=regions();malformed[1].source.buffer_bytes=1;
            graph.stage_device_weight_regions(malformed);check(!graph.wait_stage().ok,"bad second W producer accepted");
            graph.launch_device(x.view,y.view);check(!graph.finish().ok,"failed W bank reused as ready");
            stage(.125f,-.25f,.25f);run(true,.125f,-.25f,.25f);
            if(bf16) {
                // BF16-rounded 65504 would need 65536, outside the FP16
                // carrier. Do not clamp or silently change the headroom/
                // recipe after writing an earlier chunk into scratch.
                stage(.125f,65504.f,.25f);
                graph.launch_device(x.view,y.view);
                const auto overflow=graph.finish();
                if(overflow.ok || overflow.overflow_retries || overflow.error.find("cannot change headroom")==std::string::npos)
                    std::cerr<<"OVERFLOW diagnostic ok="<<overflow.ok<<" retries="<<overflow.overflow_retries
                        <<" calls="<<overflow.calls<<" error="<<overflow.error<<'\n';
                check(!overflow.ok && overflow.overflow_retries==0 &&
                    overflow.error.find("cannot change headroom")!=std::string::npos,
                    "BF16 value recipe hid a carrier overflow/headroom change");
                stage(.125f,-.25f,.25f);run(false,.125f,-.25f,.25f);
            }
            std::cout<<"PASS direct ConvRot Executor packed="<<packed<<" lookahead="<<lookahead<<" group="<<(grouped?256:0)<<" scope="<<scope
                <<" bf16_values="<<bf16<<": Comfy H256/A8, signed down scales, "
                <<(bf16?"base-only source oracle":"source/LoRA-hidden oracles")
                <<", physical channels, future-basis identity, late failure and refill\n";
        }
      } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
    }
}
