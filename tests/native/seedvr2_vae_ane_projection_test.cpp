#include "../../tools/native/seedvr2_vae_ane_projection.hpp"
#include <mlx/random.h>
#include <cstring>
#include <iostream>
#include <thread>
namespace mx=mlx::core;
using tc::Tensor;
template<class F>void rejects(F call){bool failed=false;try{call();}catch(const std::exception &){failed=true;}tc::require(failed,"invalid projection operation accepted");}
int main(int argc,char **argv){
    try{
        tc::require(argc==3,"usage: vae-ane-projection-test MODEL PROJECTION_MANIFEST");mx::set_default_device(mx::Device::gpu);
        {
        std::atomic<bool> cancelled{false};std::shared_ptr<tc::seedvr2::ExperimentalProjectionStats> stats;
        auto execute=tc::seedvr2::make_experimental_ane_projection(argv[2],false,cancelled,stats);
        tc::seedvr2::TensorBank bank;bank.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors",mx::float16,true);
        const auto &weight=bank.at("decoder.up_blocks.2.upsamplers.0.upscale_conv.weight");
        const auto &bias=bank.at("decoder.up_blocks.2.upsamplers.0.upscale_conv.bias");
        auto source=mx::random::normal({1,1,64,80,256},mx::float16,mx::random::key(178));
        auto input=mx::slice(source,{0,0,0,8,0},{1,1,64,72,256});mx::eval(input);
        Tensor held(0);std::vector<uint16_t> held_words;
        for(int iteration=0;iteration<3;++iteration){
            auto x=iteration==1?mx::negative(input):input;
            auto actual=execute(x,weight,bias);
            // Original coefficients, FP32 projection witness; preserve native
            // separate FP16 output and bias rounding before the phase shuffle.
            auto projected=mx::astype(mx::matmul(mx::astype(mx::reshape(x,{4096,256}),mx::float32),
                mx::transpose(mx::astype(mx::reshape(weight,{1024,256}),mx::float32))),mx::float16)+bias;
            auto expected=mx::contiguous(mx::reshape(mx::transpose(mx::reshape(projected,{1,1,64,64,2,2,256}),
                {0,1,2,4,3,5,6}),{1,1,128,128,256}));mx::eval(expected);
            const auto *a=actual.data<mx::float16_t>(),*b=expected.data<mx::float16_t>();
            double error=0,nx=0,ny=0,dot=0;
            for(size_t i=0;i<actual.size();++i){double av=float(a[i]),bv=float(b[i]);tc::require(std::isfinite(av)&&std::isfinite(bv),"nonfinite signed projection");
                error+=(av-bv)*(av-bv);nx+=bv*bv;ny+=av*av;dot+=av*bv;}
            tc::require(std::sqrt(error/std::max(nx,1e-20))<=.05 && dot/std::max(std::sqrt(nx*ny),1e-20)>=.999,"signed projection quality failed");
            if(iteration==0){held=actual;held_words.resize(actual.size());std::memcpy(held_words.data(),a,actual.nbytes());}
            else tc::require(std::memcmp(held.data<uint16_t>(),held_words.data(),held.nbytes())==0,"later projection overwrote a retained output");
        }
        const auto frames=stats->frames.load(),calls=stats->coreml_calls.load();
        cancelled=true;bool typed=false;try{execute(input,weight,bias);}catch(const tc::Cancelled &){typed=true;}
        cancelled=false;tc::require(typed && stats->frames==frames && stats->coreml_calls==calls,"pre-cancelled projection submitted work or lost cancellation type");
        rejects([&]{execute(mx::astype(input,mx::float32),weight,bias);});
        rejects([&]{execute(mx::zeros({1,1,63,64,256},mx::float16),weight,bias);});
        rejects([&]{execute(input,mx::zeros({1023,1,1,1,256},mx::float16),bias);});
        // Cancel after a genuine prediction, while the two-slot pipeline is
        // active. Exceptional unwinding must drain producer/consumer owners.
        auto larger=mx::ones({1,1,256,256,256},mx::float16);mx::eval(larger);mx::synchronize();
        std::atomic<bool> finished{false};std::thread canceller([&]{
            while(!finished && stats->coreml_calls.load()==calls)std::this_thread::yield();
            if(!finished)cancelled=true;
        });
        typed=false;
        try{execute(larger,weight,bias);}catch(const tc::Cancelled &){typed=true;}
        catch(...){finished=true;canceller.join();throw;}
        finished=true;canceller.join();
        tc::require(typed && stats->coreml_calls>calls && stats->frames==frames,"active projection cancellation was not exercised");cancelled=false;
        auto recovered=execute(input,weight,bias);mx::eval(recovered);
        tc::require(stats->frames==frames+1 && stats->coreml_calls==stats->backing_hits,"same-runtime recovery/backing coverage failed");
        execute={};bank.clear();held=Tensor(0);recovered=Tensor(0);source=Tensor(0);input=Tensor(0);larger=Tensor(0);mx::synchronize();mx::clear_cache();
        // A real VAE sink failure must also leave the same executor usable.
        tc::seedvr2::VAE vae;vae.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors");
        tc::SeedVR2VAEConfig config;config.history="frames";vae.configure(config);
        vae.use_experimental_upscale_projection(tc::seedvr2::make_experimental_ane_projection(argv[2],false,cancelled,stats));
        auto latent=mx::zeros({1,16,3,16,16},mx::float16);mx::eval(latent);int emitted=0;
        rejects([&]{vae.decode_stream(latent,9,[&](int index,const Tensor &){tc::require(index==emitted++,"failure sink index changed");
            if(index==1)throw std::runtime_error("intentional projection sink failure");},{},cancelled);});
        const auto completed=stats->frames.load();tc::require(emitted==2 && completed==2,"VAE sink failure was not exercised after actual projections");
        emitted=0;vae.decode_stream(latent,9,[&](int index,const Tensor &pixels){tc::require(index==emitted++,"recovery sink index changed");
            tc::require(mx::all(mx::isfinite(pixels)).item<bool>(),"nonfinite recovery pixels");},{},cancelled);
        tc::require(emitted==9 && stats->frames==completed+9 && stats->coreml_calls==stats->backing_hits,"same-VAE complete recovery failed");
        vae.clear();latent=Tensor(0);mx::synchronize();mx::clear_cache();
        }
        // Even Tensor(0) owns a four-byte allocation. End the test's complete
        // tensor scope before asserting that the executor released everything.
        mx::synchronize();mx::clear_cache();
        tc::require(mx::get_active_memory()==0,"projection/VAE teardown retained MLX allocations");
        std::cout<<"PASS signed ABA FP32 projection, strided input, retained output, invalid operands, pre/active cancellation, same-runtime/same-VAE recovery, teardown\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
