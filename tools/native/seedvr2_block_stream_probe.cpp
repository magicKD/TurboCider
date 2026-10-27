#include "seedvr2_block_weight_source.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <mach/mach.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

using namespace tc::seedvr2;
using tc::Tensor;
namespace mx=mlx::core;
namespace {
struct Sampler{
    std::atomic<bool> stop{false};std::atomic<uint64_t> peak{0},samples{0};std::thread worker;
    Sampler():worker([&]{while(!stop){task_vm_info_data_t value{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
        if(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&value),&count)==KERN_SUCCESS){
            auto old=peak.load();while(old<value.phys_footprint && !peak.compare_exchange_weak(old,value.phys_footprint)){};++samples;}
        std::this_thread::sleep_for(std::chrono::milliseconds(20));}}){}
    ~Sampler(){stop=true;if(worker.joinable())worker.join();}
};
std::shared_ptr<const tc::streaming::SourceLease> verified(const std::filesystem::path &path,std::atomic<bool> &cancelled){
    tc::streaming::SourceFileIdentity file;file.logical_id="seedvr2_dit";file.path=path;file.content_digest=kDiTCheckpointSHA;
    return tc::streaming::SourceLease::capture_verified({file},&cancelled);
}
void equal_words(const Tensor &a,const Tensor &b){
    tc::require(a.shape()==b.shape() && a.dtype()==b.dtype(),"grouped weight/output dtype or geometry changed");
    mx::eval({a,b});tc::require(std::memcmp(a.data<uint8_t>(),b.data<uint8_t>(),a.nbytes())==0,"grouped weight/output words changed");
}
void layout(const BlockWeightSource &source){
    const auto &records=source.records();
    tc::require(records.at("vid_in.proj.weight").shape==mx::Shape{2560,132} &&
        records.at("positive_conditioning").shape==mx::Shape{58,5120} &&
        records.at("positive_conditioning").dtype==mx::bfloat16 && records.at(kSharedRoPE).dtype==mx::float16 &&
        records.at("blocks.31.mlp.all.proj_in_gate.weight").shape==mx::Shape{6912,2560},"original 3B source layout required");
    for(int i=0;i<32;++i)tc::require(source.group_counts()[i+1]==size_t(i==0?28:i<10?29:15),"original block tensor coverage changed");
}
}
int main(int argc,char **argv){
    try{
        tc::require(argc==4 || argc==7,"usage: block-stream-probe weights MODEL FRESH_OUTPUT | forward MODEL INPUT FRESH_OUTPUT whole|stream ITERATIONS");
        const bool weights=std::string(argv[1])=="weights";
        tc::require(weights?argc==4:(argc==7 && std::string(argv[1])=="forward"),"invalid block stream probe mode");
        const std::filesystem::path model=argv[2],destination=argv[weights?3:4];
        tc::require(!std::filesystem::exists(destination),"fresh block-stream output required");std::filesystem::create_directories(destination);
        mx::set_default_device(mx::Device::gpu);tc::seedvr2::AllocatorCacheScope cache(1ull<<30);std::atomic<bool> cancelled{false};
        const auto checkpoint=model/"diffusion_models/seedvr2_3b_fp16.safetensors";
        auto lease=verified(checkpoint,cancelled);auto source=std::make_shared<BlockWeightSource>(lease,"seedvr2_dit",cancelled,weights);layout(*source);
        std::ofstream out(destination/"report.json");tc::require(bool(out),"cannot write block-stream report");out<<std::setprecision(17);
        if(weights){
            uint64_t compared=0,read=0;size_t tensors=0;
            {
                TensorBank full,grouped;full.load(checkpoint,mx::float16);lease->revalidate_after_drain();grouped.begin_grouped(mx::float16);
                for(int index=-1;index<32;++index){
                    const auto before=source->stats()->payload_bytes.load();auto names=grouped.adopt_group(source->load(index),[source]{source->materialized();});
                    const auto bytes=source->stats()->payload_bytes.load()-before;
                    tc::require(bytes==source->group_bytes()[index+1],"group read bytes include unselected/missing payload");
                    for(const auto &name:names){equal_words(grouped.at(name),full.at(name));compared+=grouped.at(name).nbytes();++tensors;}
                    read+=bytes;
                    if(index>=0){mx::synchronize();grouped.erase_group(names);}
                }
                tc::require(compared==full.bytes() && tensors==source->records().size(),"not every original coefficient was compared");
                const auto global_bytes=grouped.bytes();const auto before=source->stats()->payload_bytes.load();
                cancelled=true;bool typed=false;
                try{grouped.adopt_group(source->load(0),[source]{source->materialized();});}catch(const tc::Cancelled &){typed=true;}
                cancelled=false;tc::require(typed && source->stats()->payload_bytes==before && grouped.bytes()==global_bytes,
                    "pre-cancelled grouped loading performed IO or altered globals");
                auto names=grouped.adopt_group(source->load(31),[source]{source->materialized();});mx::synchronize();grouped.erase_group(names);full.clear();grouped.clear();
            }
            mx::synchronize();mx::clear_cache();tc::require(mx::get_active_memory()==0,"grouped weight test leaked MLX allocations");
            out<<"{\"schema\":\"seedvr2-block-weight-parity-v1\",\"status\":\"complete\",\"qualified\":false,\"tensors\":"<<tensors
                <<",\"compared_compute_bytes\":"<<compared<<",\"original_source_payload_bytes\":"<<read
                <<",\"header_bytes_read\":"<<source->stats()->header_bytes<<",\"all_words_equal\":true,\"zero_active_teardown\":true,\"reads\":[";
            bool first=true;for(const auto &range:source->stats()->ranges){if(!first)out<<',';first=false;
                out<<"{\"group\":"<<range.group<<",\"offset\":"<<range.offset<<",\"bytes\":"<<range.bytes<<'}';}
            out<<"],\"scope\":\"all original coefficients including source-F16 RoPE and BF16 conditioning; not forward/speed/low-memory hardware qualification\"}\n";
        }else{
            const std::string route=argv[5];const int repeats=std::stoi(argv[6]);
            tc::require((route=="whole" || route=="stream") && repeats>=1 && repeats<=5,"known route and one...five iterations required");
            uint64_t total_blocks=0,peak_parameters=0,globals=0;Sampler sampler;
            {
                auto input=mx::load_safetensors(argv[3]).first;auto noise=input.at("noise"),condition=input.at("condition");input.clear();
                tc::require(noise.dtype()==mx::float32 && noise.ndim()==5 && noise.shape(0)==1 && noise.shape(1)==16 &&
                    condition.shape()==noise.shape(),"complete original F32 noise/matching conditioning required");mx::eval({noise,condition});
                DiT dit;const auto loading=tc::Clock::now();
                if(route=="whole"){lease->revalidate_after_drain();dit.load(checkpoint,mx::float16);lease->revalidate_after_drain();}
                else dit.load_block_streamed([source](int group){return source->load(group);},[source]{source->materialized();},[source]{source->revalidate();});
                const double load_seconds=std::chrono::duration<double>(tc::Clock::now()-loading).count();globals=dit.bytes();
                tc::require(dit.stable_fp16_profile(),"block-stream probe changed stable profile");
                mx::synchronize();mx::clear_cache();
                out<<"{\"schema\":\"seedvr2-block-stream-forward-v1\",\"status\":\"complete\",\"qualified\":false,\"route\":\""<<route
                    <<"\",\"frames_latent\":"<<noise.shape(2)<<",\"height_latent\":"<<noise.shape(3)<<",\"width_latent\":"<<noise.shape(4)
                    <<",\"load_seconds_excluding_verification\":"<<load_seconds<<",\"ready_parameter_bytes\":"<<globals
                    <<",\"source_verification_seconds\":"<<lease->verification_seconds()
                    <<",\"source_verification_bytes\":"<<lease->verification_bytes_read()
                    <<",\"source_verification_cache_hits\":"<<lease->verification_cache_hits()<<",\"iterations\":[";
                for(int iteration=0;iteration<repeats;++iteration){
                    const auto before_reads=source->stats()->payload_bytes.load(),before_blocks=dit.streamed_blocks();int events=0;
                    mx::reset_peak_memory();const auto start=tc::Clock::now();
                    auto velocity=dit.forward(noise,condition,1000.f,[&](const std::string &stage,int current,int total){
                        tc::require(stage=="seedvr2_dit" && current==events++ && total==32,"complete block event coverage changed");
                    },cancelled);
                    const double seconds=std::chrono::duration<double>(tc::Clock::now()-start).count();const auto peak=mx::get_peak_memory();
                    tc::require(events==33 && mx::all(mx::isfinite(velocity)).item<bool>(),"incomplete/nonfinite full DiT velocity");
                    auto latent=noise-mx::astype(velocity,mx::float32);mx::eval(latent);
                    mx::save_safetensors((destination/("result-"+std::to_string(iteration)+".safetensors")).string(),{{"velocity",velocity},{"latent",latent}});
                    if(iteration)out<<',';
                    out<<"{\"iteration\":"<<iteration<<",\"warm\":"<<(iteration?"true":"false")<<",\"forward_seconds\":"<<seconds
                        <<",\"mlx_active_peak_bytes\":"<<peak<<",\"blocks_completed\":"<<events-1
                        <<",\"streamed_blocks\":"<<dit.streamed_blocks()-before_blocks
                        <<",\"source_payload_bytes\":"<<source->stats()->payload_bytes.load()-before_reads<<'}';out.flush();
                    tc::require(dit.bytes()==globals && (route=="whole" || dit.streamed_blocks()-before_blocks==32),"dynamic weights remained after forward");
                    std::cout<<"PASS iteration="<<iteration<<" route="<<route<<" seconds="<<seconds<<" peak="<<peak<<std::endl;
                }
                total_blocks=dit.streamed_blocks();peak_parameters=route=="whole"?dit.bytes():dit.peak_parameter_bytes();dit.clear();
            }
            mx::synchronize();mx::clear_cache();tc::require(mx::get_active_memory()==0,"block-stream forward teardown leaked MLX allocations");
            out<<"],\"peak_parameter_bytes\":"<<peak_parameters<<",\"total_streamed_blocks\":"<<total_blocks
                <<",\"sampled_process_footprint_peak_bytes\":"<<sampler.peak<<",\"footprint_samples\":"<<sampler.samples
                <<",\"source_payload_bytes\":"<<source->stats()->payload_bytes<<",\"header_bytes\":"<<source->stats()->header_bytes
                <<",\"zero_active_teardown\":true,\"scope\":\"complete context/all32 DiT blocks; serial original weight reads included in forward; no VAE/full request/App/low-RAM qualification\"}\n";
        }
        lease->revalidate_after_drain();std::cout<<"PASS block-stream probe "<<destination<<std::endl;
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
