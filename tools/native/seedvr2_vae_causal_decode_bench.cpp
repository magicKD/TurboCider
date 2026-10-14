#include "seedvr2_vae_ane_causal_conv.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <mach/mach.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
namespace mx=mlx::core;
using tc::Tensor;
namespace {
struct Sampler{
    std::atomic<bool> stop{false};std::atomic<uint64_t> peak{0};std::thread worker;
    Sampler():worker([&]{while(!stop){task_vm_info_data_t value{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
        if(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&value),&count)==KERN_SUCCESS){
            auto old=peak.load();while(old<value.phys_footprint && !peak.compare_exchange_weak(old,value.phys_footprint)){};}
        std::this_thread::sleep_for(std::chrono::milliseconds(20));}}){}
    ~Sampler(){stop=true;if(worker.joinable())worker.join();}
};
}
int main(int argc,char **argv){
    try{
        tc::require(argc==7 || argc==8,"usage: causal-decode-bench MODEL FIXTURE HOT_MANIFEST FRESH_OUTPUT gpu|ne|cpu|upscale|dual ITERATIONS [UPSCALE_MANIFEST]");
        const std::filesystem::path model(argv[1]),output(argv[4]);const std::string route(argv[5]);const int repeats=std::stoi(argv[6]);
        const bool upscale=route=="upscale" || route=="dual",hot=route=="ne" || route=="cpu" || route=="dual";
        tc::require(!std::filesystem::exists(output) && (route=="gpu" || hot || upscale) && repeats>=2 && repeats<=7 && upscale==(argc==8),
            "fresh output/known route and 2...7 fresh iterations required");std::filesystem::create_directories(output);
        mx::set_default_device(mx::Device::gpu);auto fixture=mx::load_safetensors(argv[2]).first;
        auto latent=fixture.at("latent"),reference=fixture.at(fixture.count("decoded")?"decoded":"pixels");
        tc::require(reference.dtype()==mx::float16 && reference.ndim()==5 && reference.shape(0)==1 && reference.shape(1)==3,
            "original complete FP16 decoder witness required");const int frames=reference.shape(2),h=reference.shape(3),w=reference.shape(4);
        tc::require(frames==1+4*(latent.shape(2)-1),"benchmark must retain the complete fixture");
        auto flat=mx::contiguous(mx::transpose(reference,{0,2,3,4,1}));mx::eval(flat);
        std::vector<uint16_t> saved(flat.size());std::memcpy(saved.data(),flat.data<uint16_t>(),flat.nbytes());
        fixture.clear();flat=Tensor(0);reference=Tensor(0);mx::eval(latent);
        tc::require(latent.dtype()==mx::float16 || latent.dtype()==mx::float32,"original FP16/FP32 latent required");
        // Real restored latents can be FP32 even though the saved pixels are
        // FP16. Preserve their original bytes; never size this as half words.
        std::vector<uint8_t> source(latent.nbytes());std::memcpy(source.data(),latent.data<uint8_t>(),latent.nbytes());
        tc::seedvr2::AllocatorCacheScope allocator(1ull<<30);std::atomic<bool> cancelled{false};
        tc::seedvr2::VAE vae;vae.load(model/"vae/seedvr2_ema_vae_fp16.safetensors");
        tc::SeedVR2VAEConfig config;config.history="frames";config.share_upscale_history=true;vae.configure(config);
        std::shared_ptr<tc::seedvr2::ExperimentalCausalConvStats> stats,upscale_stats;
        if(hot)vae.use_experimental_causal_conv(tc::seedvr2::make_experimental_ane_causal_conv(argv[3],route=="cpu",cancelled,stats));
        if(upscale)vae.use_experimental_upscale_causal_conv(tc::seedvr2::make_experimental_ane_upscale_causal_conv(argv[7],false,cancelled,upscale_stats));
        mx::synchronize();mx::clear_cache();Sampler sampler;std::ofstream out(output/"report.json");tc::require(bool(out),"cannot write causal benchmark report");
        out<<std::setprecision(17)<<"{\"schema\":\"seedvr2-vae-causal-decode-bench-v1\",\"route\":\""<<route<<"\",\"frames\":"<<frames
            <<",\"height\":"<<h<<",\"width\":"<<w<<",\"share_upscale_history\":true,\"iterations\":[";
        for(int iteration=0;iteration<repeats;++iteration){
            double nx=0,ny=0,dot=0,error=0;int emitted=0;uint64_t unequal=0;bool first_gpu=false;
            const uint64_t old_frames=stats?stats->frames.load():0,old_calls=stats?stats->coreml_calls.load():0;
            const uint64_t old_shared=vae.shared_upscale_frames();const double old_boundary=stats?stats->boundary_seconds:0;
            const uint64_t old_up_frames=upscale_stats?upscale_stats->frames.load():0,old_up_calls=upscale_stats?upscale_stats->coreml_calls.load():0;
            const double old_up_boundary=upscale_stats?upscale_stats->boundary_seconds:0;
            mx::reset_peak_memory();const auto start=tc::Clock::now();
            vae.decode_stream(latent,frames,[&](int index,const Tensor &pixels){
                tc::require(index==emitted && pixels.shape()==mx::Shape{1,3,1,h,w},"fresh decoder frame order/geometry changed");
                auto current=mx::contiguous(mx::transpose(pixels,{0,2,3,4,1}));mx::eval(current);
                const auto *actual=current.data<mx::float16_t>(),*expected=reinterpret_cast<const mx::float16_t *>(saved.data()+size_t(index)*h*w*3);
                uint64_t frame_unequal=0;double frame_nx=0,frame_ny=0,frame_dot=0,frame_error=0;
                for(size_t i=0;i<size_t(h)*w*3;++i){double a=float(actual[i]),b=float(expected[i]);
                    tc::require(std::isfinite(a)&&std::isfinite(b),"nonfinite warm decoder pixels");
                    frame_nx+=b*b;frame_ny+=a*a;frame_dot+=a*b;frame_error+=(a-b)*(a-b);frame_unequal+=std::memcmp(actual+i,expected+i,2)!=0;}
                tc::require(std::sqrt(frame_error/std::max(frame_nx,1e-20))<=.05 &&
                    frame_dot/std::max(std::sqrt(frame_nx*frame_ny),1e-20)>=.999,"unchanged per-frame decoder quality gate failed");
                nx+=frame_nx;ny+=frame_ny;dot+=frame_dot;error+=frame_error;
                unequal+=frame_unequal;
                if(index==0){first_gpu=frame_unequal==0 && (!stats || stats->coreml_calls==old_calls) &&
                    (!upscale_stats || upscale_stats->coreml_calls==old_up_calls);tc::require(first_gpu,"initial GPU FP32 frame changed or executed Core ML");}
                ++emitted;
            },{},cancelled);
            const double seconds=std::chrono::duration<double>(tc::Clock::now()-start).count();const uint64_t peak=mx::get_peak_memory();
            const double l2=std::sqrt(error/std::max(nx,1e-20)),cosine=dot/std::max(std::sqrt(nx*ny),1e-20);
            const uint64_t calls=stats?stats->coreml_calls.load()-old_calls:0,completed=stats?stats->frames.load()-old_frames:0;
            const uint64_t up_calls=upscale_stats?upscale_stats->coreml_calls.load()-old_up_calls:0,up_completed=upscale_stats?upscale_stats->frames.load()-old_up_frames:0;
            const int th=stats?stats->tile_rows:64,tw=stats?stats->tile_cols:64;
            const uint64_t expected_calls=hot?uint64_t(frames-1)*((h+th-1)/th)*((w+tw-1)/tw):0;
            const int uh=upscale_stats?upscale_stats->tile_rows:64,uw=upscale_stats?upscale_stats->tile_cols:64;
            const uint64_t expected_up_calls=upscale?uint64_t(frames-1)*((h+uh-1)/uh)*((w+uw-1)/uw):0;
            tc::require(emitted==frames && l2<=.05 && cosine>=.999 && (route!="gpu" || unequal==0) &&
                calls==expected_calls && completed==uint64_t(hot?frames-1:0) &&
                up_calls==expected_up_calls && up_completed==uint64_t(upscale?frames-1:0) &&
                (!stats || stats->coreml_calls==stats->backing_hits) && (!upscale_stats || upscale_stats->coreml_calls==upscale_stats->backing_hits) &&
                vae.shared_upscale_frames()-old_shared==uint64_t(frames>1?frames:0),"fresh complete decoder quality/execution gate failed");
            if(iteration)out<<',';
            out<<"{\"iteration\":"<<iteration<<",\"warm\":"<<(iteration?"true":"false")<<",\"seconds_with_host_quality\":"<<seconds
                <<",\"mlx_active_peak_bytes\":"<<peak<<",\"emitted_frames\":"<<emitted<<",\"compared_values\":"<<uint64_t(frames)*h*w*3
                <<",\"relative_l2\":"<<l2<<",\"cosine\":"<<cosine<<",\"unequal_values\":"<<unequal
                <<",\"coreml_calls\":"<<calls<<",\"causal_frames\":"<<completed<<",\"initial_gpu_frame_checked\":true,\"quality_passed\":true"
                <<",\"causal_boundary_seconds\":"<<(stats?stats->boundary_seconds-old_boundary:0)
                <<",\"upscale_coreml_calls\":"<<up_calls<<",\"upscale_causal_frames\":"<<up_completed
                <<",\"total_coreml_calls\":"<<calls+up_calls<<",\"per_frame_quality_checked\":true"
                <<",\"upscale_causal_boundary_seconds\":"<<(upscale_stats?upscale_stats->boundary_seconds-old_up_boundary:0)<<'}';out.flush();
            std::cout<<"PASS iteration="<<iteration<<" route="<<route<<" seconds="<<seconds<<" active="<<peak<<" calls="<<calls+up_calls<<'\n';
        }
        tc::require(std::memcmp(source.data(),latent.data<uint8_t>(),latent.nbytes())==0,"warm benchmark changed latent input");
        out<<"],\"status\":\"complete\",\"sampled_process_footprint_peak_bytes\":"<<sampler.peak
            <<",\"external_slot_bytes\":"<<(stats?stats->slot_bytes:0)+(upscale_stats?upscale_stats->slot_bytes:0)
            <<",\"upscale_layer\":\""<<(upscale_stats?upscale_stats->layer:"")<<"\",\"upscale_external_slot_bytes\":"<<(upscale_stats?upscale_stats->slot_bytes:0)
            <<",\"physical_ane_occupancy_measured\":false,\"qualified\":false"
            <<",\"scope\":\"fresh complete decoder per iteration, weights/kernels/cache/model stay warm; host pixel gate included; not App/full request/physical RAM qualification\"}\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
