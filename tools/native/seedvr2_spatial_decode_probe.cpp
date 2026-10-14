#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#if defined(TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE) && TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE
#include "seedvr2_vae_ane_causal_conv.hpp"
#endif
#if defined(TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE) && TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE
#include "seedvr2_vae_ane_projection.hpp"
#endif
#include <mach/mach.h>
#include <thread>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
namespace mx=mlx::core;
using tc::Tensor;
namespace {
struct Sampler {
    std::atomic<bool> stop{false};std::atomic<uint64_t> peak{0};std::thread worker;
    Sampler():worker([&]{while(!stop){task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
        if(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS){
            auto old=peak.load();while(old<info.phys_footprint && !peak.compare_exchange_weak(old,info.phys_footprint)){};}
        std::this_thread::sleep_for(std::chrono::milliseconds(20));}}){}
    ~Sampler(){stop=true;if(worker.joinable())worker.join();}
};
}
int main(int argc,char **argv){
    try {
        const bool projection_experiment=argc==10;
        const bool causal_experiment=argc==11 || argc==13;
        const bool upscale_causal_experiment=argc==13;
        const bool hot_causal_experiment=causal_experiment && std::string(argv[9])!="none";
        bool valid_arguments=argc==7 || argc==8 || argc==9;
#if defined(TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE) && TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE
        valid_arguments=valid_arguments || projection_experiment;
#endif
#if defined(TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE) && TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE
        valid_arguments=valid_arguments || causal_experiment;
#endif
        tc::require(valid_arguments,"usage: spatial-decode-probe MODEL FIXTURE OUTPUT baseline|spatial THRESHOLD ROWS [fresh|reuse [share|copy]]; offloads require experimental builds");
        const std::filesystem::path model(argv[1]),output(argv[3]);const std::string route(argv[4]);
        tc::require(!std::filesystem::exists(output) && (route=="baseline" || route=="spatial"),"fresh output and known route required");
        std::filesystem::create_directories(output);mx::set_default_device(mx::Device::gpu);
        auto tensors=mx::load_safetensors(argv[2]).first;auto latent=tensors.at("latent");
        auto reference=tensors.at(tensors.count("decoded")?"decoded":"pixels");
        const int frames=reference.shape(2),h=reference.shape(3),w=reference.shape(4);
        tc::require(reference.dtype()==mx::float16 && reference.shape(0)==1 && reference.shape(1)==3,"saved FP16 full decoder reference required");
        auto flat=mx::contiguous(mx::transpose(reference,{0,2,3,4,1}));mx::eval(flat);
        std::vector<uint16_t> saved(flat.size());std::memcpy(saved.data(),flat.data<uint16_t>(),flat.nbytes());
        tensors.clear();flat=Tensor(0);reference=Tensor(0);mx::synchronize();mx::clear_cache();
        std::atomic<bool> cancelled{false};tc::seedvr2::AllocatorCacheScope allocator(1ull<<30);
        tc::seedvr2::VAE vae;vae.load(model/"vae/seedvr2_ema_vae_fp16.safetensors");
        tc::SeedVR2VAEConfig config;config.history="frames";
        if(argc==9 || causal_experiment){tc::require(std::string(argv[8])=="share" || std::string(argv[8])=="copy","known upscale history recipe required");
            config.share_upscale_history=std::string(argv[8])=="share";tc::require(!config.share_upscale_history || route=="baseline","shared upscale history requires full decoding");}
        vae.configure(config);mx::eval(latent);
#if defined(TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE) && TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE
        std::shared_ptr<tc::seedvr2::ExperimentalCausalConvStats> causal_stats,upscale_causal_stats;
        if(hot_causal_experiment){
            tc::require(route=="baseline" && (std::string(argv[10])=="ne" || std::string(argv[10])=="cpu"),"full causal-conv compute control required");
            vae.use_experimental_causal_conv(tc::seedvr2::make_experimental_ane_causal_conv(
                argv[9],std::string(argv[10])=="cpu",cancelled,causal_stats));
        }
        if(upscale_causal_experiment){
            tc::require(route=="baseline" && (std::string(argv[12])=="ne" || std::string(argv[12])=="cpu"),"upscale causal-conv compute control required");
            vae.use_experimental_upscale_causal_conv(tc::seedvr2::make_experimental_ane_upscale_causal_conv(
                argv[11],std::string(argv[12])=="cpu",cancelled,upscale_causal_stats));
        }
#endif
        tc::seedvr2::SpatialTileOptions options;options.threshold_bytes=std::stoull(argv[5]);options.rows=std::stoi(argv[6]);
        if(argc>=8){tc::require(std::string(argv[7])=="fresh" || std::string(argv[7])=="reuse","known read-window recipe required");options.reuse_read_windows=std::string(argv[7])=="reuse";}
#if defined(TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE) && TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE
        std::shared_ptr<tc::seedvr2::ExperimentalProjectionStats> projection_stats;
        if(projection_experiment){
            tc::require(std::string(argv[9])=="ne" || std::string(argv[9])=="cpu","known projection compute control required");
            vae.use_experimental_upscale_projection(tc::seedvr2::make_experimental_ane_projection(
                argv[8],std::string(argv[9])=="cpu",cancelled,projection_stats));
        }
#endif
        std::shared_ptr<tc::seedvr2::SpatialFileStats> stats;
        std::ofstream pixels_file(output/"decoded.fp16.bin",std::ios::binary);
        tc::require(bool(pixels_file),"cannot create bounded frame export");
        uint64_t compared=0,unequal=0;int emitted=0;double nx=0,ny=0,dot=0,es=0,max_abs=0;
        uint64_t phase_count=0;
#if defined(TURBOCIDER_ENABLE_VAE_PHASE_PROFILE) && TURBOCIDER_ENABLE_VAE_PHASE_PROFILE
        std::ofstream phases(output/"phases.jsonl");tc::require(bool(phases),"cannot create phase diagnostic trace");
        phases<<std::setprecision(17);
        vae.observe_phases([&](const tc::seedvr2::VAEPhaseObservation &row){
            auto shape=[&](const mx::Shape &value){phases<<'[';for(size_t i=0;i<value.size();++i){if(i)phases<<',';phases<<value[i];}phases<<']';};
            phases<<"{\"schema\":\""<<row.schema<<"\",\"index\":"<<phase_count++<<",\"next_output_frame\":"<<emitted
                <<",\"key\":\""<<row.key<<"\",\"initial\":"<<(row.initial?"true":"false")<<",\"inclusive\":"<<(row.inclusive?"true":"false")
                <<",\"input_shape\":";shape(row.input_shape);phases<<",\"output_shape\":";shape(row.output_shape);
            phases<<",\"seconds\":"<<row.seconds<<",\"active_before\":"<<row.active_before<<",\"active_after\":"<<row.active_after
                <<",\"peak_before\":"<<row.peak_before<<",\"peak_after\":"<<row.peak_after<<",\"cache_after\":"<<row.cache_after
                <<",\"history_unique_before\":"<<row.history_unique_before<<",\"history_unique_after\":"<<row.history_unique_after
                <<",\"history_logical_after\":"<<row.history_logical_after<<",\"history_unavailable_after\":"<<row.history_unavailable_after<<"}\n";
            tc::require(bool(phases),"phase diagnostic trace write failed");
        });
#endif
        auto sink=[&](int index,const Tensor &pixels){
            tc::require(index==emitted && pixels.shape()==mx::Shape{1,3,1,h,w} && pixels.dtype()==mx::float16,"spatial frame index/shape changed");
            auto current=mx::contiguous(mx::transpose(pixels,{0,2,3,4,1}));mx::eval(current);
            const auto *actual=current.data<mx::float16_t>();const auto *expected=reinterpret_cast<const mx::float16_t *>(saved.data()+size_t(index)*h*w*3);
            if(index==0 && causal_experiment){
                tc::require(std::memcmp(actual,expected,size_t(h)*w*3*2)==0,"initial GPU frame changed");
#if defined(TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE) && TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE
                for(const auto &observed:{causal_stats,upscale_causal_stats})
                    tc::require(!observed || (observed->frames==0 && observed->coreml_calls==0),"initial frame unexpectedly used Core ML");
#endif
            }
            pixels_file.write(reinterpret_cast<const char *>(actual),size_t(h)*w*3*2);
            tc::require(bool(pixels_file),"bounded spatial frame export failed");
            for(size_t i=0;i<size_t(h)*w*3;++i){const double a=float(actual[i]),b=float(expected[i]);
                tc::require(std::isfinite(a) && std::isfinite(b),"nonfinite spatial/reference pixels");
                nx+=b*b;ny+=a*a;dot+=a*b;es+=(a-b)*(a-b);max_abs=std::max(max_abs,std::abs(a-b));
                unequal+=std::memcmp(actual+i,expected+i,2)!=0;++compared;}
            ++emitted;
        };
        mx::reset_peak_memory();Sampler sampler;auto start=tc::Clock::now();
        if(route=="baseline")vae.decode_stream(latent,frames,sink,{},cancelled);
        else vae.decode_spatial_stream(latent,frames,sink,options,stats,{},cancelled);
        const double seconds=std::chrono::duration<double>(tc::Clock::now()-start).count();const auto active=mx::get_peak_memory();
#if defined(TURBOCIDER_ENABLE_VAE_PHASE_PROFILE) && TURBOCIDER_ENABLE_VAE_PHASE_PROFILE
        phases.close();tc::require(bool(phases),"phase diagnostic trace close failed");
#endif
        const double l2=std::sqrt(es/std::max(nx,1e-20)),cosine=dot/std::max(std::sqrt(nx*ny),1e-20);
        const bool quality=emitted==frames && l2<=.05 && cosine>=.999;
#if defined(TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE) && TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE
        for(const auto &observed:{causal_stats,upscale_causal_stats})if(observed){
            const uint64_t expected=uint64_t(frames-1)*((h+observed->tile_rows-1)/observed->tile_rows)*((w+observed->tile_cols-1)/observed->tile_cols);
            tc::require(observed->frames==uint64_t(frames-1) && observed->coreml_calls==expected && observed->backing_hits==expected,
                "complete per-layer causal frame/tile/backing coverage differs");
        }
#endif
#if defined(TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE) && TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE
        if(projection_stats){
            tc::require(projection_stats->frames==uint64_t(frames) && projection_stats->coreml_calls==projection_stats->backing_hits,
                        "complete VAE projection execution/backing coverage differs");
        }
#endif
        tc::require(!stats || (stats->disk_live_bytes==0 && stats->host_live_bytes==0),"spatial scratch/buffer live accounting did not drain");
        std::ofstream out(output/"report.json");tc::require(bool(out),"cannot write spatial decode report");
        out<<std::setprecision(17)<<"{\"schema\":\"seedvr2-spatial-decode-screen-v1\",\"route\":\""<<route
            <<"\",\"vae_upscale_recipe\":\""<<vae.upscale_recipe()<<"\",\"frames\":"<<emitted<<",\"height\":"<<h<<",\"width\":"<<w
            <<",\"seconds_with_host_quality\":"<<seconds<<",\"mlx_active_peak_bytes\":"<<active<<",\"sampled_process_footprint_peak_bytes\":"<<sampler.peak
            <<",\"bitwise_equal\":"<<(unequal==0?"true":"false")<<",\"unequal_values\":"<<unequal<<",\"compared_values\":"<<compared
            <<",\"relative_l2\":"<<l2<<",\"cosine\":"<<cosine<<",\"max_abs\":"<<max_abs<<",\"quality_passed\":"<<(quality?"true":"false")
            <<",\"file_written_bytes\":"<<(stats?stats->written_bytes:0)<<",\"file_read_bytes\":"<<(stats?stats->read_bytes:0)
            <<",\"file_peak_bytes\":"<<(stats?stats->disk_peak_bytes:0)<<",\"host_window_peak_bytes\":"<<(stats?stats->host_peak_bytes.load():0)
            <<",\"tile_convolutions\":"<<(stats?stats->tile_convolutions:0)<<",\"global_normalizations\":"<<(stats?stats->global_normalizations:0)
            <<",\"read_window_recipe\":\""<<(options.reuse_read_windows?"reuse":"fresh")<<"\",\"host_window_allocations\":"<<(stats?stats->host_allocations.load():0)
            <<",\"host_window_reuses\":"<<(stats?stats->host_reuses.load():0)
            <<",\"projection_backend\":\""<<(projection_experiment?argv[9]:"none")<<"\"";
        out<<",\"share_upscale_history\":"<<(vae.shares_upscale_history()?"true":"false")
            <<",\"shared_upscale_frames\":"<<vae.shared_upscale_frames()<<",\"shared_upscale_logical_bytes\":"<<vae.shared_upscale_logical_bytes();
#if defined(TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE) && TURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE
        if(projection_stats)out<<",\"projection_frames\":"<<projection_stats->frames<<",\"projection_coreml_calls\":"<<projection_stats->coreml_calls
            <<",\"projection_backing_hits\":"<<projection_stats->backing_hits<<",\"projection_slot_bytes\":"<<projection_stats->slot_bytes
            <<",\"projection_boundary_seconds\":"<<projection_stats->boundary_seconds<<",\"projection_load_seconds\":"<<projection_stats->load_seconds;
#endif
        out<<",\"causal_conv_backend\":\""<<(hot_causal_experiment?argv[10]:"none")<<"\"";
#if defined(TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE) && TURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE
        if(causal_stats)out<<",\"causal_conv_frames\":"<<causal_stats->frames<<",\"causal_conv_coreml_calls\":"<<causal_stats->coreml_calls
            <<",\"causal_conv_backing_hits\":"<<causal_stats->backing_hits<<",\"causal_conv_slot_bytes\":"<<causal_stats->slot_bytes
            <<",\"causal_conv_boundary_seconds\":"<<causal_stats->boundary_seconds<<",\"causal_conv_load_seconds\":"<<causal_stats->load_seconds;
        if(upscale_causal_stats)out<<",\"upscale_causal_conv_layer\":\""<<upscale_causal_stats->layer<<"\",\"upscale_causal_conv_output_channels\":"<<upscale_causal_stats->output_channels
            <<",\"upscale_causal_conv_backend\":\""<<argv[12]<<"\",\"upscale_causal_conv_frames\":"<<upscale_causal_stats->frames
            <<",\"upscale_causal_conv_coreml_calls\":"<<upscale_causal_stats->coreml_calls<<",\"upscale_causal_conv_backing_hits\":"<<upscale_causal_stats->backing_hits
            <<",\"upscale_causal_conv_slot_bytes\":"<<upscale_causal_stats->slot_bytes<<",\"upscale_causal_conv_boundary_seconds\":"<<upscale_causal_stats->boundary_seconds
            <<",\"upscale_causal_conv_load_seconds\":"<<upscale_causal_stats->load_seconds;
#endif
        out<<",\"initial_gpu_frame_checked\":"<<(causal_experiment?"true":"false");
        out<<",\"diagnostic_phase_count\":"<<phase_count
            <<",\"qualified\":false,\"physical_ane_occupancy_measured\":false,\"scope\":\"decoder-only saved FP16 witness, host quality scan included; not original-FP32/full-request/paired speed or low-RAM qualification\"}\n";
        out.close();tc::require(quality,"spatial decoder quality gate failed");
        if(route=="baseline" && !projection_experiment && !causal_experiment)tc::require(unequal==0,"baseline changed saved decoder bytes");
        std::cout<<"PASS "<<route<<" frames="<<emitted<<" l2="<<l2<<" cosine="<<cosine<<" peak="<<active<<'\n';
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
