#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <cstring>
#include <iostream>
namespace mx=mlx::core;
int main(int argc,char **argv) {
    try {
        tc::require(argc==3 || argc==4,"usage: spatial-encode-cancel MODEL PIXELS128t9 [reuse]");mx::set_default_device(mx::Device::gpu);
        auto inputs=mx::load_safetensors(argv[2]).first;auto pixels=inputs.at("pixels");inputs.clear();mx::eval(pixels);
        tc::require(pixels.shape()==mx::Shape{1,3,9,128,128},"complete128/t9 input required");
        std::vector<uint8_t> source(pixels.nbytes());std::memcpy(source.data(),pixels.data<uint8_t>(),source.size());
        tc::seedvr2::AllocatorCacheScope allocator(512ull<<20);tc::seedvr2::VAE vae;
        vae.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors");
        tc::SeedVR2VAEConfig config;config.history="frames";vae.configure(config);
        tc::seedvr2::SpatialTileOptions options;options.threshold_bytes=4ull<<20;options.rows=16;
        if(argc==4){tc::require(std::string(argv[3])=="reuse","known read-window recipe required");options.reuse_read_windows=true;}
        std::shared_ptr<tc::seedvr2::SpatialFileStats> stats;std::atomic<bool> cancelled{false};bool called=false,typed=false;
        try {
            vae.encode_spatial(pixels,options,stats,[&](const std::string &phase,int completed,int) {
                if(phase=="seedvr2_vae_encode" && completed==1) {
                    tc::require(stats && stats->disk_live_bytes>0,"cancel did not reach active file histories");
                    called=true;cancelled=true;
                }
            },cancelled);
        }catch(const tc::Cancelled &){typed=true;}
        tc::require(called && typed && stats && stats->disk_live_bytes==0 && stats->host_live_bytes==0 && stats->cleanup_failures==0,
            "encoder cancellation did not retire owned files/windows with typed status");
        cancelled=false;options.disk_budget_bytes=4ull<<20;bool quota=false;
        try {vae.encode_spatial(pixels,options,stats,{},cancelled);}
        catch(const std::exception &error){quota=std::string(error.what()).find("budget")!=std::string::npos;}
        tc::require(quota && stats && stats->disk_live_bytes==0 && stats->host_live_bytes==0 && stats->cleanup_failures==0,
            "encoder quota failure did not clean up");
        auto recovered=vae.encode(pixels,{},cancelled);mx::eval(recovered);
        tc::require(recovered.shape()==mx::Shape{1,16,3,16,16} && mx::all(mx::isfinite(recovered)).item<bool>() &&
            std::memcmp(source.data(),pixels.data<uint8_t>(),source.size())==0,"same-VAE recovery/source preservation failed");
        std::cout<<"PASS encoder active-history cancellation, quota failure, cleanup and same-VAE recovery\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
