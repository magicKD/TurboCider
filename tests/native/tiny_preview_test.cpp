#include "../../native/components/vae/tiny.hpp"
#include <mlx/random.h>
#include <fstream>
#include <iostream>
#include <limits>

int main(int argc,char **argv) {
    try {
        tc::require(argc==4,"usage: tiny-preview-test MODEL WEIGHTS OUTPUT_BASE");
        tc::Request r;r.model=argv[1];r.output=std::filesystem::absolute(argv[3]).string();r.steps=5;
        r.width=512;r.height=512;
        r.tiny_vae=tc::TinyVAEConfig{std::filesystem::absolute(argv[2]).string(),"original","original",true,2};
        bool qwen=r.model=="qwen-image-2.1";
        auto input=tc::mx::random::normal({1,qwen?64:16,qwen?32:64,qwen?32:64});tc::mx::eval(input);
        std::vector<float> original(input.data<float>(),input.data<float>()+input.size());
        auto velocity=tc::mx::full(input.shape(),.75f,tc::mx::float32);
        auto noisy=input+velocity*.6f;
        auto estimate=tc::components::flow_preview_estimate(noisy,velocity,.6f);
        tc::mx::eval({noisy,velocity,estimate});
        tc::require(tc::mx::allclose(estimate,input,1e-5,1e-6).item<bool>(),
                    "flow preview decoded noisy x_sigma instead of predicted clean x0");
        tc::require(tc::mx::array_equal(tc::components::flow_preview_estimate(input,velocity,0.f),input).item<bool>(),
                    "zero-sigma preview changed the final latent");
        for(float sigma : {-1.f,std::numeric_limits<float>::infinity()}) {
            bool rejected=false;
            try {tc::components::flow_preview_estimate(input,velocity,sigma);}
            catch(const std::invalid_argument &){rejected=true;}
            tc::require(rejected,"preview accepted invalid sigma");
        }
        std::atomic<bool> cancelled{false};
        tc::Event event;
        std::vector<int> steps;
        bool cancel_on_preview=false;
        auto path=r.output+".preview.png";
        event.preview=[&](const std::string &value,int step,int total) {
            tc::require(value==path && total==5,"wrong preview correlation");
            std::ifstream file(value,std::ios::binary);
            unsigned char header[24]{};file.read(reinterpret_cast<char *>(header),24);
            tc::require(file.gcount()==24 && header[0]==137 && header[1]=='P' && header[2]=='N' && header[3]=='G',"preview was not a PNG");
            auto u32=[&](int i){return (uint32_t(header[i])<<24)|(uint32_t(header[i+1])<<16)|(uint32_t(header[i+2])<<8)|header[i+3];};
            tc::require(u32(16)==512 && u32(20)==512,"preview differs from final 512x512 dimensions");
            tc::require(!std::filesystem::exists(r.output),"preview became a final output");
            steps.push_back(step);if(cancel_on_preview)cancelled=true;
        };
        {
            tc::components::TinyPreview preview(r,event,cancelled);
            tc::require(!preview.due(0) && !preview.due(6),"preview accepted a step outside its schedule");
            for(int i=1;i<=5;++i)if(preview.due(i))preview.publish(input,i);
            tc::require(steps==std::vector<int>({2,4,5}),"preview cadence/final-step mismatch");
        }
        tc::require(!std::filesystem::exists(path),"success leaked preview sidecar");
        steps.clear();cancel_on_preview=true;bool rejected=false;
        try {
            tc::components::TinyPreview preview(r,event,cancelled);
            preview.publish(input,2);preview.publish(input,4);
        }catch(const tc::Cancelled &){rejected=true;}
        tc::require(rejected && !std::filesystem::exists(path),"cancel did not reject/clean preview");
        cancelled=false;cancel_on_preview=false;steps.clear();
        {
            tc::components::TinyPreview preview(r,event,cancelled);preview.publish(input,5);
        }
        tc::require(steps==std::vector<int>{5} && !std::filesystem::exists(path),"preview did not recover after cancellation");
        rejected=false;
        try {
            tc::components::TinyPreview preview(r,event,cancelled);
            auto wrong=tc::mx::slice(input,{0,0,0,0},
                {1,input.shape(1),input.shape(2)-1,input.shape(3)});
            preview.publish(wrong,2);
        }catch(const std::invalid_argument &){rejected=true;}
        tc::require(rejected && !std::filesystem::exists(path),"wrong-size preview was published");
        r.tiny_vae->interval=0;rejected=false;
        try {tc::components::TinyPreview invalid(r,event,cancelled);}catch(const std::invalid_argument &){rejected=true;}
        tc::require(rejected,"direct C++ preview accepted zero interval");r.tiny_vae->interval=2;
        {std::ofstream owned(path);owned<<"preserve-existing-file";}
        rejected=false;
        try {tc::components::TinyPreview occupied(r,event,cancelled);}catch(const std::invalid_argument &){rejected=true;}
        {std::ifstream owned(path);std::string value;owned>>value;tc::require(rejected && value=="preserve-existing-file","preview clobbered an unrelated sidecar");}
        std::filesystem::remove(path);
        r.tiny_vae->preview=false;r.tiny_vae->path="/nonexistent/disabled.safetensors";
        {
            tc::components::TinyPreview disabled(r,event,cancelled);
            tc::require(!disabled.due(2),"disabled preview did model work");disabled.publish(input,2);
        }
        tc::require(std::equal(original.begin(),original.end(),input.data<float>()),"preview mutated the sampler latent");
        tc::require(tc::mx::all(velocity==.75f).item<bool>(),"preview mutated the model velocity");
        std::cout<<"clean flow x0, full 512x512 PNG, cadence/final-step, cancel/cleanup/recovery, disabled and latent invariance passed\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
