#include "../../tools/native/seedvr2_block_weight_source.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>

using namespace tc::seedvr2;
using tc::Tensor;
namespace mx=mlx::core;
template<class F>void rejects(F call){bool failed=false;try{call();}catch(const std::exception &){failed=true;}tc::require(failed,"invalid grouped source/weight accepted");}
std::shared_ptr<const tc::streaming::SourceLease> capture(const std::filesystem::path &path,std::atomic<bool> &cancelled,bool original=false){
    tc::streaming::SourceFileIdentity file;file.logical_id="dit";file.path=path;if(original)file.content_digest=kDiTCheckpointSHA;
    return tc::streaming::SourceLease::capture_verified({file},&cancelled);
}
void fixture(const std::filesystem::path &path){
    WeightGroup values;values.emplace(kSharedRoPE,mx::ones({21},mx::float16));values.emplace("positive_conditioning",mx::ones({2,2},mx::bfloat16));
    for(int i=0;i<32;++i)values.emplace("blocks."+std::to_string(i)+".weight",mx::full({2,2},i,mx::float16));
    mx::save_safetensors(path.string(),values);
}
int main(int argc,char **argv){
    try{
        tc::require(argc==3,"usage: block-stream-test MODEL FRESH_OUTPUT");
        const std::filesystem::path output=argv[2];tc::require(!std::filesystem::exists(output),"fresh lifecycle output required");
        std::filesystem::create_directories(output);mx::set_default_device(mx::Device::gpu);AllocatorCacheScope cache(1ull<<30);
        std::atomic<bool> cancelled{false};uint64_t cancelled_payload=0,completed_blocks=0;
        {
            auto lease=capture(std::filesystem::path(argv[1])/"diffusion_models/seedvr2_3b_fp16.safetensors",cancelled,true);
            auto source=std::make_shared<BlockWeightSource>(lease,"dit",cancelled);
            {
                auto materialized=[source]{source->materialized();};
                TensorBank bank;bank.begin_grouped(mx::float16);bank.adopt_group(source->load(-1),materialized);const auto globals=bank.bytes();
                const auto before_duplicate=source->stats()->payload_bytes.load();
                rejects([&]{bank.adopt_group(source->load(-1),materialized);});
                tc::require(bank.bytes()==globals && source->stats()->payload_bytes==before_duplicate,"duplicate group read/published partial weights");
                const auto before=source->stats()->payload_bytes.load();cancelled=true;bool typed=false;
                try{bank.adopt_group(source->load(0),materialized);}catch(const tc::Cancelled &){typed=true;}
                cancelled=false;tc::require(typed && source->stats()->payload_bytes==before,"pre-cancelled group performed payload IO");
                std::atomic<bool> finished{false};std::exception_ptr unexpected;
                std::thread cancel([&]{while(!finished && source->stats()->payload_bytes==before)std::this_thread::yield();if(!finished)cancelled=true;});
                typed=false;
                try{bank.adopt_group(source->load(0),materialized);}catch(const tc::Cancelled &){typed=true;}catch(...){unexpected=std::current_exception();}
                finished=true;cancel.join();cancelled=false;if(unexpected)std::rethrow_exception(unexpected);
                cancelled_payload=source->stats()->payload_bytes-before;
                tc::require(typed && cancelled_payload>0 && bank.bytes()==globals,"actual mid-read cancellation was not atomic/typed: typed="+
                    std::to_string(typed)+" payload="+std::to_string(cancelled_payload)+" bytes="+std::to_string(bank.bytes())+" globals="+std::to_string(globals));
                auto names=bank.adopt_group(source->load(0),materialized);mx::synchronize();bank.erase_group(names);
                tc::require(bank.bytes()==globals,"same-source group recovery retained dynamic parameters");
                source->stats()->inject_read_failure=true;bool read_error=false;
                try{bank.adopt_group(source->load(0),materialized);}catch(const std::runtime_error &error){
                    read_error=std::string(error.what())=="seedvr2-block-source-injected-read-failure";
                }
                source->stats()->inject_read_failure=false;
                tc::require(read_error && bank.bytes()==globals,"non-cancel Reader failure was swallowed or published partial parameters");
                names=bank.adopt_group(source->load(0),materialized);mx::synchronize();bank.erase_group(names);bank.clear();
            }
            mx::synchronize();mx::clear_cache();
            {
                DiT model;bool wrong_group=false;
                model.load_block_streamed([&](int group){return source->load(wrong_group && group==1?0:group);},[source]{source->materialized();},[source]{source->revalidate();});
                const auto globals=model.bytes();auto noise=mx::ones({1,16,3,16,16},mx::float32),condition=mx::full(noise.shape(),.2f,mx::float16);mx::eval({noise,condition});
                std::vector<uint8_t> original(noise.nbytes());std::memcpy(original.data(),noise.data<uint8_t>(),original.size());
                cancelled=true;bool typed=false;const auto before=source->stats()->payload_bytes.load();
                try{model.forward(noise,condition,1000.f,{},cancelled);}catch(const tc::Cancelled &){typed=true;}
                cancelled=false;tc::require(typed && source->stats()->payload_bytes==before && model.bytes()==globals,"pre-cancelled forward performed IO");
                typed=false;
                try{model.forward(noise,condition,1000.f,[&](const std::string &,int index,int){if(index==1)cancelled=true;},cancelled);}
                catch(const tc::Cancelled &){typed=true;}
                cancelled=false;tc::require(typed && model.streamed_blocks()==1 && model.bytes()==globals,"cancellation after actual block failed to retire weights");
                auto recovered=model.forward(noise,condition,1000.f,{},cancelled);mx::eval(recovered);
                tc::require(model.streamed_blocks()==33 && model.bytes()==globals,"same-model complete32-block recovery failed");
                wrong_group=true;rejects([&]{model.forward(noise,condition,1000.f,{},cancelled);});wrong_group=false;
                tc::require(model.bytes()==globals,"wrong-group rejection retained a prior block");
                auto again=model.forward(noise,condition,1000.f,{},cancelled);mx::eval(again);
                tc::require(again.shape()==recovered.shape() && again.dtype()==recovered.dtype() &&
                    std::memcmp(again.data<uint8_t>(),recovered.data<uint8_t>(),again.nbytes())==0,"same-model recovery changed complete output words");
                tc::require(std::memcmp(noise.data<uint8_t>(),original.data(),original.size())==0,"block streaming mutated source noise");
                tc::require(model.peak_parameter_bytes()<500ull<<20,"unbounded grouped parameter residency");
                completed_blocks=model.streamed_blocks();model.clear();
            }
        }
        {
            const auto mutated=output/"owned-mutated.safetensors";fixture(mutated);auto lease=capture(mutated,cancelled);
            auto source=std::make_shared<BlockWeightSource>(lease,"dit",cancelled);
            tc::streaming::SourceFileIdentity metadata;metadata.logical_id="dit";metadata.path=mutated;
            rejects([&]{BlockWeightSource invalid(tc::streaming::SourceLease::capture({metadata}),"dit",cancelled);});
            {std::ofstream append(mutated,std::ios::binary|std::ios::app);append.put('x');}
            rejects([&]{source->load(0);});
            const auto named=output/"owned-named.safetensors",replacement=output/"owned-replacement.safetensors",archived=output/"owned-original-archived.safetensors";
            fixture(named);fixture(replacement);lease=capture(named,cancelled);source=std::make_shared<BlockWeightSource>(lease,"dit",cancelled);
            std::filesystem::rename(named,archived);std::filesystem::rename(replacement,named);
            rejects([&]{source->load(0);});
            auto fresh=capture(named,cancelled);BlockWeightSource recovered(fresh,"dit",cancelled);
            TensorBank bank;bank.begin_grouped(mx::float16);auto names=bank.adopt_group(recovered.load(0),[&]{recovered.materialized();});bank.clear();
        }
        mx::synchronize();mx::clear_cache();tc::require(mx::get_active_memory()==0,"block-stream lifecycle leaked MLX allocations");
        std::ofstream report(output/"report.json");report<<"{\"schema\":\"seedvr2-block-stream-lifecycle-v1\",\"status\":\"complete\",\"qualified\":false"
            <<",\"mid_read_cancelled_payload_bytes\":"<<cancelled_payload<<",\"completed_blocks\":"<<completed_blocks
            <<",\"pre_cancel_checked\":true,\"mid_read_cancel_typed\":true,\"after_block_cancel_checked\":true,\"same_model_recovery_words_equal\":true"
            <<",\"non_cancel_read_error_latched\":true"
            <<",\"duplicate_group_atomic\":true,\"wrong_group_rejected\":true,\"source_mutation_rejected\":true,\"source_replacement_rejected\":true"
            <<",\"unverified_source_rejected\":true,\"fresh_source_recovery\":true,\"zero_active_teardown\":true"
            <<",\"scope\":\"isolated serial original-weight DiT lifecycle; only private generated fixture files are mutated/replaced; no App/full-request/low-memory qualification\"}\n";
        std::cout<<"PASS grouped source atomic cancellation/recovery, complete32 blocks, duplicate/wrong-group guards, private source mutation/replacement, zero-active teardown\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
