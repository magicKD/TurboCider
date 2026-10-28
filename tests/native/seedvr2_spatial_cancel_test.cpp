#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <iostream>
namespace mx=mlx::core;
int main(int argc,char **argv){
    try {
        tc::require(argc==3 || argc==4,"usage: spatial-cancel-test MODEL FIXTURE [reuse]");mx::set_default_device(mx::Device::gpu);
        auto loaded=mx::load_safetensors(argv[2]).first;auto latent=loaded.at("latent");loaded.clear();
        const int frames=1+4*(latent.shape(2)-1);tc::require(latent.shape(3)==16 && frames>=5,"128-size multiframe fixture required");
        std::atomic<bool> cancelled{false};tc::seedvr2::AllocatorCacheScope allocator(512ull<<20);
        tc::seedvr2::VAE vae;vae.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors");
        tc::SeedVR2VAEConfig config;config.history="frames";vae.configure(config);
        tc::seedvr2::SpatialTileOptions options;options.threshold_bytes=4ull<<20;options.rows=16;
        if(argc==4){tc::require(std::string(argv[3])=="reuse","known window recipe required");options.reuse_read_windows=true;}
        std::shared_ptr<tc::seedvr2::SpatialFileStats> stats;int emitted=0;bool typed=false;
        try {vae.decode_spatial_stream(latent,frames,[&](int index,const tc::Tensor &){tc::require(index==0,"cancel read ahead");++emitted;cancelled=true;},
            options,stats,{},cancelled);}catch(const tc::Cancelled &){typed=true;}
        tc::require(typed && emitted==1 && stats && stats->disk_live_bytes==0 && stats->host_live_bytes==0,
            "spatial cancellation did not drain files/windows with typed status");cancelled=false;emitted=0;
        bool sink_failed=false;
        try {vae.decode_spatial_stream(latent,frames,[&](int,const tc::Tensor &){throw std::runtime_error("owned-test-sink-failure");},
            options,stats,{},cancelled);}catch(const std::runtime_error &error){sink_failed=std::string(error.what())=="owned-test-sink-failure";}
        tc::require(sink_failed && stats->disk_live_bytes==0 && stats->host_live_bytes==0,"failed sink leaked spatial ownership");
        vae.decode_stream(latent,frames,[&](int index,const tc::Tensor &frame){tc::require(index==emitted && mx::all(mx::isfinite(frame)).item<bool>(),
            "same-VAE recovery failed");++emitted;},{},cancelled);
        tc::require(emitted==frames,"same-VAE recovery truncated frames");
        std::cout<<"PASS spatial typed cancellation, sink failure, drained files/windows and same-VAE recovery\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
