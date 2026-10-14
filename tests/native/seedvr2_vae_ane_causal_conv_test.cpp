#include "../../tools/native/seedvr2_vae_ane_causal_conv.hpp"
#include <mlx/random.h>
#include <cstring>
#include <iostream>
#include <thread>
namespace mx=mlx::core;
using tc::Tensor;
template<class F>void rejects(F call){bool failed=false;try{call();}catch(const std::exception &){failed=true;}tc::require(failed,"invalid causal operation accepted");}
void quality(const Tensor &actual,const Tensor &reference){
    tc::require(actual.shape()==reference.shape(),"causal output geometry changed");mx::eval(reference);
    const auto *a=actual.data<mx::float16_t>(),*b=reference.data<mx::float16_t>();double error=0,nx=0,ny=0,dot=0;
    for(size_t i=0;i<actual.size();++i){double av=float(a[i]),bv=float(b[i]);tc::require(std::isfinite(av)&&std::isfinite(bv),"nonfinite causal witness");
        error+=(av-bv)*(av-bv);nx+=bv*bv;ny+=av*av;dot+=av*bv;}
    tc::require(std::sqrt(error/std::max(nx,1e-20))<=.05 && dot/std::max(std::sqrt(nx*ny),1e-20)>=.999,"original-FP32 causal gate failed");
}
int main(int argc,char **argv){
    try{
        tc::require(argc==3,"usage: causal-conv-test MODEL FULL_CONV_MANIFEST");mx::set_default_device(mx::Device::gpu);
        {
            std::atomic<bool> cancelled{false};std::shared_ptr<tc::seedvr2::ExperimentalCausalConvStats> stats;
            auto execute=tc::seedvr2::make_experimental_ane_causal_conv(argv[2],false,cancelled,stats,true);
            tc::seedvr2::TensorBank bank;bank.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors",mx::float16,true);
            const auto &weight=bank.at("decoder.up_blocks.3.resnets.0.conv1.weight"),&bias=bank.at("decoder.up_blocks.3.resnets.0.conv1.bias");
            Tensor held(0);std::vector<uint16_t> saved;int cases=0;
            for(const auto &geometry:std::vector<std::array<int,2>>{{17,19},{65,71},{64,64}}){
                const int h=geometry[0],w=geometry[1];std::vector<Tensor> source;
                for(int tap=0;tap<3;++tap){
                    const int extra=h==64?0:4,start=extra/2;
                    auto parent=mx::astype(mx::random::normal({1,2,h,w+extra,256},mx::float32,mx::random::key(431+tap+h))*Tensor(.2f)+Tensor(float(tap-1)*.5f),mx::float16);
                    mx::eval(parent);auto frame=mx::slice(parent,{0,1,0,start,0},{1,2,h,w+start,256});mx::eval(frame);
                    tc::require(frame.dtype()==mx::float16 && frame.offset()!=0 && (extra==0?frame.flags().row_contiguous:!frame.flags().row_contiguous),
                        "offset/strided causal source was not actually exercised");
                    source.push_back(frame);
                }
                mx::eval(source);
                for(int sign:{1,-1,1}){
                    tc::seedvr2::FrameHistory past{sign>0?source[0]:mx::negative(source[0]),sign>0?source[1]:mx::negative(source[1])};
                    auto current=sign>0?source[2]:mx::negative(source[2]);
                    if(cases==1)past[1]=past[0]; // Initial replicated history is a real alias, not a copied fake.
                    auto actual=execute(current,past,weight,bias);
                    auto joined=mx::concatenate({past[0],past[1],current},1);
                    auto reference=mx::astype(mx::conv3d(mx::astype(joined,mx::float32),mx::astype(weight,mx::float32),{1,1,1},{0,1,1}),mx::float16)+bias;
                    quality(actual,reference);
                    if(cases==0){held=actual;saved.resize(actual.size());std::memcpy(saved.data(),actual.data<uint16_t>(),actual.nbytes());}
                    else tc::require(std::memcmp(held.data<uint16_t>(),saved.data(),held.nbytes())==0,"causal slot reuse overwrote retained output");
                    ++cases;
                }
            }
            tc::require(stats->frames==uint64_t(cases) && stats->coreml_calls==stats->backing_hits &&
                stats->pack_checked_words>0 && stats->consume_checked_words>0,"actual independent IO witnesses were not exercised");
            auto frame=mx::ones({1,1,256,256,256},mx::float16);mx::eval(frame);tc::seedvr2::FrameHistory past{frame,frame};
            const auto frames=stats->frames.load(),calls=stats->coreml_calls.load();
            cancelled=true;bool typed=false;try{execute(frame,past,weight,bias);}catch(const tc::Cancelled &){typed=true;}
            cancelled=false;tc::require(typed && stats->coreml_calls==calls,"pre-cancelled causal operation submitted work");
            rejects([&]{execute(mx::astype(frame,mx::float32),past,weight,bias);});
            rejects([&]{execute(frame,{frame},weight,bias);});
            rejects([&]{execute(frame,past,mx::zeros({127,3,3,3,256},mx::float16),bias);});
            std::atomic<bool> finished{false};std::thread canceller([&]{
                while(!finished && stats->coreml_calls.load()==calls)std::this_thread::yield();if(!finished)cancelled=true;
            });typed=false;
            try{execute(frame,past,weight,bias);}catch(const tc::Cancelled &){typed=true;}
            catch(...){finished=true;canceller.join();throw;}
            finished=true;canceller.join();tc::require(typed && stats->coreml_calls>calls && stats->frames==frames,
                "causal cancellation was not exercised during a real prediction");cancelled=false;
            auto recovered=execute(frame,past,weight,bias);mx::eval(recovered);
            tc::require(stats->frames==frames+1 && stats->coreml_calls==stats->backing_hits,"same-causal-runtime recovery failed");
            std::cout<<"causal bitwise pack words="<<stats->pack_checked_words<<" consume words="<<stats->consume_checked_words<<'\n';
            execute={};bank.clear();held=Tensor(0);frame=Tensor(0);recovered=Tensor(0);past.clear();mx::synchronize();mx::clear_cache();
            tc::seedvr2::VAE vae;vae.load(std::filesystem::path(argv[1])/"vae/seedvr2_ema_vae_fp16.safetensors");
            tc::SeedVR2VAEConfig config;config.history="frames";config.share_upscale_history=true;vae.configure(config);
            vae.use_experimental_causal_conv(tc::seedvr2::make_experimental_ane_causal_conv(argv[2],false,cancelled,stats));
            auto latent=mx::zeros({1,16,3,16,16},mx::float16);mx::eval(latent);int emitted=0;bool failed=false;
            try{vae.decode_stream(latent,9,[&](int index,const Tensor &){tc::require(index==emitted++,"causal failure sink order changed");
                if(index==1)throw std::runtime_error("causal-owned-sink-failure");},{},cancelled);}
            catch(const std::runtime_error &error){failed=std::string(error.what())=="causal-owned-sink-failure";}
            tc::require(failed && emitted==2 && stats->frames==1,"first GPU frame/subsequent causal prediction/sink failure were not exercised");
            const auto before=stats->frames.load();emitted=0;
            vae.decode_stream(latent,9,[&](int index,const Tensor &pixels){tc::require(index==emitted++ && mx::all(mx::isfinite(pixels)).item<bool>(),"causal VAE recovery failed");},{},cancelled);
            tc::require(emitted==9 && stats->frames==before+8 && stats->coreml_calls==stats->backing_hits,"complete same-VAE causal recovery failed");
            vae.clear();latent=Tensor(0);
        }
        mx::synchronize();mx::clear_cache();tc::require(mx::get_active_memory()==0,"causal session teardown leaked MLX allocations");
        std::cout<<"PASS causal signed ABA original-FP32, ragged halo/borders, replicated history, retained output, IO bitwise witnesses, invalid operands, pre/active cancellation, same-runtime/VAE recovery, first-GPU folding, teardown\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
