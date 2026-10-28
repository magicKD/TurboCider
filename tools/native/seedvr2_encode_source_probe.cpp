#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
namespace mx=mlx::core;
using tc::Tensor;
namespace {
template<class F>void rejects(F function){bool rejected=false;try{function();}catch(const std::exception &){rejected=true;}tc::require(rejected,"invalid frame source accepted");}
}
int main(int argc,char **argv){
    try {
        tc::require(argc==4,"usage: encode-source-probe MODEL_ROOT INPUT.safetensors FRESH_OUTPUT");
        const std::filesystem::path model(argv[1]),output(argv[3]);tc::require(!std::filesystem::exists(output),"fresh encode-source output required");
        std::filesystem::create_directories(output);mx::set_default_device(mx::Device::gpu);
        auto inputs=mx::load_safetensors(argv[2]).first;
        auto pixels=inputs.at("pixels");tc::require(pixels.dtype()==mx::float32 && pixels.shape(0)==1 && pixels.shape(1)==3,"normalized FP32 fixture required");
        const int available=pixels.shape(2),h=pixels.shape(3),w=pixels.shape(4);
        tc::require(available>=9,"at least nine fixture frames required to test temporal partitions");
        auto flat=mx::contiguous(mx::transpose(pixels,{0,2,3,4,1}));mx::eval(flat);
        const size_t frame_values=size_t(h)*w*3;
        std::vector<float> host(size_t(available)*frame_values);std::memcpy(host.data(),flat.data<float>(),host.size()*4);
        inputs.clear();pixels=Tensor(0);flat=Tensor(0);mx::synchronize();mx::clear_cache();
        std::atomic<bool> cancelled{false};tc::seedvr2::AllocatorCacheScope allocator(tc::seedvr2::allocator_cache_limit());
        tc::seedvr2::VAE vae;vae.load(model/"vae/seedvr2_ema_vae_fp16.safetensors");
        std::ofstream report(output/"report.json");tc::require(bool(report),"cannot write source encode report");
        report<<"{\"schema\":\"seedvr2-vae-encode-source-screen-v1\",\"cases\":[";
        bool first=true;
        for(int count:{1,2,4,5,6,9}){
            const int total=tc::seedvr2::padded_frames(count);
            auto input=mx::expand_dims(mx::transpose(Tensor(host.data(),{count,h,w,3},mx::float32),{3,0,1,2}),0);
            if(total!=count){auto last=mx::slice(input,{0,0,count-1,0,0},{1,3,count,h,w});input=mx::concatenate({input,mx::repeat(last,total-count,2)},2);}
            mx::eval(input);mx::reset_peak_memory();auto begin=tc::Clock::now();
            auto reference=vae.encode(input,{},cancelled);const double buffered=std::chrono::duration<double>(tc::Clock::now()-begin).count();
            const auto buffered_peak=mx::get_peak_memory();reference=mx::contiguous(reference);mx::eval(reference);
            std::vector<uint16_t> expected(reference.size());std::memcpy(expected.data(),reference.data<uint16_t>(),expected.size()*2);
            const auto shape=reference.shape();reference=Tensor(0);input=Tensor(0);mx::synchronize();mx::clear_cache();mx::reset_peak_memory();
            int calls=0;begin=tc::Clock::now();
            auto actual=vae.encode_stream(count,h,w,[&](int index){tc::require(index==calls && index<count,"source index/count changed");++calls;
                return mx::expand_dims(mx::transpose(Tensor(host.data()+size_t(index)*frame_values,{1,h,w,3},mx::float32),{3,0,1,2}),0);}, {},cancelled);
            const double streaming=std::chrono::duration<double>(tc::Clock::now()-begin).count();const auto stream_peak=mx::get_peak_memory();
            actual=mx::contiguous(actual);mx::eval(actual);
            tc::require(calls==count && actual.shape()==shape && actual.dtype()==mx::float16 &&
                std::memcmp(expected.data(),actual.data<uint16_t>(),expected.size()*2)==0,"streaming source encoded latent is not completely bitwise equal");
            if(!first)report<<',';first=false;
            report<<"{\"source_frames\":"<<count<<",\"padded_frames\":"<<total<<",\"calls\":"<<calls
                <<",\"buffered_seconds\":"<<buffered<<",\"streaming_seconds\":"<<streaming
                <<",\"buffered_active_peak_bytes\":"<<buffered_peak<<",\"streaming_active_peak_bytes\":"<<stream_peak<<",\"bitwise_equal\":true}";
            mx::save_safetensors((output/("latent-t"+std::to_string(count)+".safetensors")).string(),{{"latent",actual}});
            actual=Tensor(0);mx::synchronize();mx::clear_cache();std::cout<<"PASS source_frames="<<count<<" calls="<<calls<<'\n';
        }
        rejects([&]{vae.encode_stream(0,h,w,[](int){return Tensor(0);},{},cancelled);});
        rejects([&]{vae.encode_stream(122,h,w,[](int){return Tensor(0);},{},cancelled);});
        rejects([&]{vae.encode_stream(1,h,w,{}, {},cancelled);});
        rejects([&]{vae.encode_stream(1,h,w,[](int){return Tensor(0);},{},cancelled);});
        cancelled=true;bool typed=false;try{vae.encode_stream(1,h,w,[](int){return Tensor(0);},{},cancelled);}catch(const tc::Cancelled &){typed=true;}
        tc::require(typed,"streaming VAE cancellation type changed");cancelled=false;
        report<<"],\"passed\":true,\"typed_cancellation_passed\":true,\"scope\":\"encode-only mathematical/lifetime screen; independent host fixture retained; not App or paired request speed/physical peak gate\"}\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
