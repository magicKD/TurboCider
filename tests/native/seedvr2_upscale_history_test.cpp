#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <cstring>
#include <iostream>
namespace mx=mlx::core;
using tc::Tensor;
int main(int argc,char **argv){
    try{
        tc::require(argc==3,"usage: upscale-history-test MODEL COMPLETE_FP16_DECODER_FIXTURE");mx::set_default_device(mx::Device::gpu);
        {
            auto loaded=mx::load_safetensors(argv[2]).first;auto latent=loaded.at("latent");
            auto reference=loaded.at(loaded.count("decoded")?"decoded":"pixels");loaded.clear();
            tc::require(reference.dtype()==mx::float16 && reference.shape(2)==9 && reference.shape(3)==128,"complete128/t9 FP16 witness required");
            auto expected=mx::contiguous(mx::transpose(reference,{0,2,3,4,1}));mx::eval(expected);
            const int h=reference.shape(3),w=reference.shape(4);const size_t frame_values=size_t(h)*w*3;
            std::vector<uint16_t> saved(expected.size());std::memcpy(saved.data(),expected.data<uint16_t>(),expected.nbytes());
            reference=Tensor(0);expected=Tensor(0);mx::eval(latent);std::vector<uint16_t> source(latent.size());
            std::memcpy(source.data(),latent.data<uint16_t>(),latent.nbytes());
            tc::seedvr2::AllocatorCacheScope allocator(512ull<<20);std::atomic<bool> cancelled{false};
            tc::seedvr2::VAE vae;vae.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors");
            tc::SeedVR2VAEConfig config;config.history="frames";config.share_upscale_history=true;vae.configure(config);
            int emitted=0;auto sink=[&](int index,const Tensor &value){
                tc::require(index==emitted++ && value.shape()==mx::Shape{1,3,1,h,w},"shared upscale frame order/shape changed");
                auto pixels=mx::contiguous(mx::transpose(value,{0,2,3,4,1}));mx::eval(pixels);
                tc::require(std::memcmp(pixels.data<uint16_t>(),saved.data()+index*frame_values,frame_values*2)==0,
                    "shared upscale changed complete decoder words");
            };
            vae.decode_stream(latent,9,sink,{},cancelled);
            tc::require(emitted==9 && vae.shared_upscale_frames()==9 && vae.shared_upscale_logical_bytes()==uint64_t(9)*h*w*256*2,
                        "shared upscale frame/byte coverage differs");
            vae.configure(config);bool typed=false;
            try{vae.decode_stream(latent,9,[&](int index,const Tensor &){tc::require(index==0,"cancel read ahead");cancelled=true;},{},cancelled);}
            catch(const tc::Cancelled &){typed=true;}
            tc::require(typed && vae.shared_upscale_frames()==1,"shared upscale cancellation lost its type or count");cancelled=false;
            vae.configure(config);emitted=0;vae.decode_stream(latent,9,sink,{},cancelled);
            tc::require(emitted==9,"same-VAE recovery lost frames");
            vae.configure(config);bool failed=false;
            try{vae.decode_stream(latent,9,[&](int index,const Tensor &){if(index==1)throw std::runtime_error("owned-upscale-sink-failure");},{},cancelled);}
            catch(const std::runtime_error &error){failed=std::string(error.what())=="owned-upscale-sink-failure";}
            tc::require(failed && vae.shared_upscale_frames()==2,"shared upscale sink failure was not exercised");
            vae.configure(config);emitted=0;vae.decode_stream(latent,9,sink,{},cancelled);
            tc::require(emitted==9 && std::memcmp(source.data(),latent.data<uint16_t>(),latent.nbytes())==0,"recovery modified the source latent");
            config.share_upscale_history=false;vae.configure(config);emitted=0;vae.decode_stream(latent,9,sink,{},cancelled);
            tc::require(emitted==9 && !vae.shares_upscale_history() && vae.shared_upscale_frames()==0,"request switch did not restore copy history");
            config.share_upscale_history=true;vae.configure(config);emitted=0;
            auto image=mx::full({1,16,1,16,16},mx::slice(latent,{0,0,0,0,0},{1,16,1,16,16}),mx::float16);mx::eval(image);
            vae.decode_stream(image,1,[&](int index,const Tensor &pixels){tc::require(index==0 && mx::all(mx::isfinite(pixels)).item<bool>(),"single image failed");++emitted;},{},cancelled);
            tc::require(emitted==1 && vae.shared_upscale_frames()==0,"single image retained unused history");
            vae.clear();image=Tensor(0);latent=Tensor(0);
        }
        mx::synchronize();mx::clear_cache();tc::require(mx::get_active_memory()==0,"upscale history/session teardown leaked allocations");
        std::cout<<"PASS complete9 word equality, frame/byte counters, typed cancellation, sink failure, same-VAE recovery, source preservation, option switches, image/no unused history, zero-allocation teardown\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
