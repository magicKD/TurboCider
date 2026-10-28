#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <mach/mach.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>
namespace mx=mlx::core;
namespace {
struct Sampler {
    std::atomic<bool> done{false};std::atomic<uint64_t> peak{0};std::thread worker;
    Sampler():worker([&]{while(!done){task_vm_info_data_t info{};mach_msg_type_number_t n=TASK_VM_INFO_COUNT;
        if(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&n)==KERN_SUCCESS){
            auto old=peak.load();while(old<info.phys_footprint && !peak.compare_exchange_weak(old,info.phys_footprint)){};}
        std::this_thread::sleep_for(std::chrono::milliseconds(20));}}){}
    ~Sampler(){done=true;if(worker.joinable())worker.join();}
};
}
int main(int argc,char **argv) {
    try {
        tc::require(argc==7 || argc==8,"usage: spatial-encode-probe MODEL PIXELS OUTPUT baseline|spatial THRESHOLD_BYTES ROWS [REPEATS]");
        const std::filesystem::path output(argv[3]);const std::string route(argv[4]);
        tc::require(!std::filesystem::exists(output) && (route=="baseline" || route=="spatial"),"fresh output/known route required");
        mx::set_default_device(mx::Device::gpu);
        auto inputs=mx::load_safetensors(argv[2]).first;auto pixels=inputs.at("pixels");inputs.clear();
        tc::require(pixels.ndim()==5 && pixels.shape(0)==1 && pixels.shape(1)==3 && pixels.shape(2)<=121,
            "complete RGB BCTHW input required");mx::eval(pixels);
        std::filesystem::create_directories(output);std::atomic<bool> cancelled{false};
        tc::seedvr2::AllocatorCacheScope allocator(1ull<<30);tc::seedvr2::VAE vae;
        vae.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors");
        tc::SeedVR2VAEConfig config;config.history="frames";vae.configure(config);
        tc::seedvr2::SpatialTileOptions options;options.threshold_bytes=std::stoull(argv[5]);options.rows=std::stoi(argv[6]);
        const int repeats=argc==8?std::stoi(argv[7]):1;tc::require(repeats>=1 && repeats<=15,"bounded encoder repetitions required");
        struct Observation{int iteration;double seconds;size_t peak;};std::vector<Observation> iterations;
        std::shared_ptr<tc::seedvr2::SpatialFileStats> stats;Sampler sampler;tc::Tensor latent(0);double seconds=0;size_t active=0;
        for(int iteration=0;iteration<repeats;++iteration){
            latent=tc::Tensor(0);mx::synchronize();mx::reset_peak_memory();auto begin=tc::Clock::now();
            latent=route=="baseline"?vae.encode(pixels,{},cancelled):vae.encode_spatial(pixels,options,stats,{},cancelled);
            mx::eval(latent);seconds=std::chrono::duration<double>(tc::Clock::now()-begin).count();active=mx::get_peak_memory();
            tc::require(mx::all(mx::isfinite(latent)).item<bool>(),"nonfinite spatial encoder latent");
            iterations.push_back({iteration,seconds,active});
        }
        mx::save_safetensors((output/"latent.safetensors").string(),{{"latent",latent}});
        std::ofstream report(output/"report.json");report<<std::setprecision(17)
            <<"{\"schema\":\"seedvr2-spatial-encode-observation-v1\",\"route\":\""<<route<<"\",\"source_frames\":"<<pixels.shape(2)
            <<",\"height\":"<<pixels.shape(3)<<",\"width\":"<<pixels.shape(4)<<",\"encode_seconds\":"<<seconds
            <<",\"mlx_active_peak_bytes\":"<<active<<",\"sampled_process_footprint_peak_bytes\":"<<sampler.peak
            <<",\"tile_convolutions\":"<<(stats?stats->tile_convolutions:0)<<",\"global_normalizations\":"<<(stats?stats->global_normalizations:0)
            <<",\"file_written_bytes\":"<<(stats?stats->written_bytes:0)<<",\"file_read_bytes\":"<<(stats?stats->read_bytes:0)
            <<",\"file_peak_bytes\":"<<(stats?stats->disk_peak_bytes:0)<<",\"host_window_peak_bytes\":"<<(stats?stats->host_peak_bytes.load():0)
            <<",\"file_live_bytes_after_encode\":"<<(stats?stats->disk_live_bytes:0)<<",\"host_live_bytes_after_encode\":"<<(stats?stats->host_live_bytes.load():0)
            <<",\"cleanup_failures\":"<<(stats?stats->cleanup_failures:0)<<",\"finite\":true,\"qualified\":false,\"iterations\":[";
        for(const auto &row:iterations){if(row.iteration)report<<',';report<<"{\"iteration\":"<<row.iteration<<",\"encode_seconds\":"<<row.seconds<<",\"mlx_active_peak_bytes\":"<<row.peak<<'}';}
        report<<"],"
            "\"scope\":\"complete encoder, high-resolution singleton head only; global statistics/causal history/temporal downsampling/all attention preserved; single component observation not request/App/RAM qualification\"}\n";
        report.close();tc::require(bool(report),"encoder report write failed");
        std::cout<<"PASS "<<route<<" frames="<<pixels.shape(2)<<" seconds="<<seconds<<" peak="<<active<<'\n';
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
