#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <cstring>
#include <iostream>
namespace mx=mlx::core;
int main(int argc,char **argv) {
    try {
        tc::require(argc==2,"usage: spatial-encode-stream-test MODEL");mx::set_default_device(mx::Device::gpu);
        tc::seedvr2::AllocatorCacheScope allocator(512ull<<20);tc::seedvr2::VAE vae;
        vae.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors");
        tc::SeedVR2VAEConfig config;config.history="frames";vae.configure(config);
        tc::seedvr2::SpatialTileOptions options;options.threshold_bytes=4ull<<20;options.rows=16;
        std::atomic<bool> cancelled{false};int cases=0;
        for(int frames:{1,2,4,5,6,9}) {
            auto pixels=mx::random::uniform(-.5f,.5f,{1,3,frames,128,128},mx::float32,mx::random::key(23));mx::eval(pixels);
            const int total=tc::seedvr2::padded_frames(frames);auto padded=pixels;
            if(total>frames)padded=mx::concatenate({pixels,mx::repeat(mx::slice(pixels,{0,0,frames-1,0,0},{1,3,frames,128,128}),total-frames,2)},2);
            std::shared_ptr<tc::seedvr2::SpatialFileStats> first,second;
            auto buffered=vae.encode_spatial(padded,options,first,{},cancelled);int consumed=0;
            auto streamed=vae.encode_stream_spatial(frames,128,128,[&](int index) {
                tc::require(index==consumed++ && index<frames,"stream source index/read-ahead changed");
                return mx::full({1,3,1,128,128},mx::slice(pixels,{0,0,index,0,0},{1,3,index+1,128,128}),mx::float32);
            },options,second,{},cancelled);mx::eval({buffered,streamed});
            tc::require(consumed==frames && buffered.shape()==streamed.shape() &&
                std::memcmp(buffered.data<uint16_t>(),streamed.data<uint16_t>(),buffered.nbytes())==0,
                "spatial streaming encoder changed complete latent/padding words");
            tc::require(first && second && first->disk_live_bytes==0 && second->disk_live_bytes==0 &&
                first->host_live_bytes==0 && second->host_live_bytes==0 && first->cleanup_failures==0 && second->cleanup_failures==0,
                "spatial stream/buffer ownership did not drain");++cases;
        }
        std::cout<<"PASS spatial source streaming initial5/then4/padding exact cases="<<cases<<'\n';
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
