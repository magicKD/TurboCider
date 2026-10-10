#pragma once
#include "encoder_runtime_policy.hpp"
#include "encoder_calibration.hpp"
#include "text_encoder.hpp"
#include "../../backends/ane_ffn.hpp"
#include "../../backends/ane_backend.hpp"
#include "../../backends/dense_gpu_projection.hpp"

namespace tc::qwen21 {
inline void validate_encoder_runtime_process_recipe() {
    const char* chunks=std::getenv("TURBOCIDER_RUNTIME_ANE_CHUNKS");
    require(chunks && std::string_view(chunks)=="1","encoder runtime requires a fixed one-chunk plan");
    const auto profile=encoder_profile_path("TURBOCIDER_QWEN21_ENCODER_S1_PROFILE");
    const auto current=ane::HybridFfn::executor_configuration_identity()+encoder_runtime::process_recipe()+
        (profile.empty()?"":sha256_file(profile));
    static const std::string recipe=current;
    require(recipe==current,"encoder runtime recipe changed between cached requests");
}
struct EncoderRuntimeStats {
    size_t gpu_graph_builds=0,gpu_graph_calls=0;
    double input_ready_seconds=0,gpu_graph_encode_host_seconds=0;
    bool preparation_reused=false;
    int bucket_rows=0;
    uint64_t next_source_submissions=0,next_source_skips=0;
};
// Session-owned, one idle geometry only. No model/source tensors are cached.
// An in-flight request owns the executable exclusively, returning it only
// after private drain and GPU completion. Eviction/unload drops all banks.
class EncoderRuntimeCache {
    friend class EncoderRuntime;
    std::string key_;
    std::unique_ptr<ane::HybridFfn> bridge_;
    std::function<std::vector<Tensor>(const std::vector<Tensor>&)> graph_;
    bool borrowed_=false;
public:
    void clear() {require(!borrowed_,"encoder cache cannot evict an in-flight request");graph_={};bridge_.reset();key_.clear();}
};
// Base-only language prefill bridge. Uses existing device-owner/physical drain
// executor contracts. No visual/attention/VAE offload or per-model export.
class EncoderRuntime {
    const Weights& weights_;
    std::atomic<bool>& cancelled_;
    std::unique_ptr<ane::HybridFfn> bridge_;
    std::string executor_identity_,prefix_;
    int layer_=-1,rows_=0;
    bool split_=false;
    encoder_runtime::GpuRecipe gpu_recipe_=encoder_runtime::gpu_recipe();
    std::function<std::vector<Tensor>(const std::vector<Tensor>&)> dynamic_graph_;
    EncoderRuntimeStats stats_;
    EncoderRuntimeCache* cache_=nullptr;
    std::string cache_key_;
    size_t budget_=0;
    std::unordered_map<std::string,ane::HybridFfn::ChannelGpu> channel_graphs_;
public:
    EncoderRuntime(const Weights& weights,const std::filesystem::path& manifest,
                   size_t budget,std::atomic<bool>& cancelled,int request_rows=0,EncoderRuntimeCache* cache=nullptr,
                   const EncoderScaleProfile* profile=nullptr)
        : weights_(weights),cancelled_(cancelled),budget_(budget) {
        const auto policy=ane::configured_backend();
        require(policy.preferred==ane::BackendPreference::Private && policy.allow_private,
                "encoder runtime requires explicit authorized private backend");
        const auto descriptor=ane::runtime_template_descriptor(manifest);
        const auto& s=descriptor.shape;
        encoder_runtime::validate_bucket(s.rows);
        require(!descriptor.has_public_artifact() && s.kind==ane::Kind::SwiGLU && s.hidden==4096 &&
                    s.width==12288 && !s.lora_inputs,"encoder runtime requires base-only Qwen3-VL shape descriptor");
        require(!ane::configured_request_rows(),"encoder runtime uses its bounded descriptor bucket; no per-prompt programs");
        const char* path=std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH");
        require(path && std::string_view(path)=="w8a8","encoder runtime candidate requires W8A8 Hadamard");
        require(ane::private_channel_count(12288)>0,"encoder runtime requires explicit partial ANE channels");
        const auto exact=[](const char* name,const char* value) {
            const char* raw=std::getenv(name);require(raw && std::string_view(raw)==value,std::string("encoder runtime requires ")+name+"="+value);
        };
        exact("TURBOCIDER_RUNTIME_ANE_CHUNKS","1");
        exact("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1");
        exact("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1");
        require(!std::getenv("TURBOCIDER_RUNTIME_ANE_PROFILE") || std::string_view(std::getenv("TURBOCIDER_RUNTIME_ANE_PROFILE"))=="0",
                "encoder runtime profiling must not change the fixed plan");
        executor_identity_=ane::HybridFfn::executor_configuration_identity();
        stats_.bucket_rows=encoder_runtime::select_bucket(request_rows,s.rows);
        cache_key_=sha256_file(manifest)+executor_identity_+encoder_runtime::process_recipe()+
            ":bucket="+std::to_string(stats_.bucket_rows)+":budget="+std::to_string(budget)+(profile?profile->digest:"");
        if(cache && encoder_runtime::configured_bool("TURBOCIDER_QWEN21_ENCODER_RUNTIME_REUSE")) {
            require(!cache->borrowed_,"encoder cache is already checked out");
            if(cache->key_!=cache_key_)cache->clear();
            bridge_=std::move(cache->bridge_);dynamic_graph_=std::move(cache->graph_);
            stats_.preparation_reused=bool(bridge_);
            cache_=cache;
        }
        if(!bridge_)bridge_=std::make_unique<ane::HybridFfn>(manifest,4096,12288,budget,cancelled,false,
            std::nullopt,"",nullptr,std::nullopt,0,stats_.bucket_rows);
        require(bridge_->available() && bridge_->channel_split() && bridge_->fp32_channel_join(),
                "encoder runtime unavailable: "+bridge_->reason());
        bridge_->begin_request();
        bridge_->reset_phase_metrics(stats_.preparation_reused);
        if(profile)bridge_->set_smoothquant(profile->digest,[&]{return profile->tensors();},36);
        else bridge_->set_smoothquant({});
        if(gpu_recipe_==encoder_runtime::GpuRecipe::MppDynamicBf16 && !dynamic_graph_) {
            const int count=bridge_->gpu_channels();
            // One pure graph, dynamic weight arguments. No 36 per-layer
            // traces or FP32-expanded weight constants kept by compilation.
            dynamic_graph_=mx::compile([count](const std::vector<Tensor>& a) {
                auto gate=mx::matmul(a[0],mx::transpose(slice_axis(a[1],0,0,count)));
                auto up=mx::matmul(a[0],mx::transpose(slice_axis(a[2],0,0,count)));
                auto hidden=silu(gate)*up;
                auto partial=dense_gpu::projection_range(hidden,a[3],0,4096,0,count,32,true);
                return std::vector<Tensor>{partial,hidden};
            });
            ++stats_.gpu_graph_builds;
        }
        if(cache_)cache_->borrowed_=true;
    }
    ~EncoderRuntime() {if(cache_)cache_->borrowed_=false;}
    TextFfnHooks hooks() {
        return {[this](int i,const std::string& p,int rows){prepare(i,p,rows);},
                [this](int i,const Tensor& x,const TextFfnHooks::Gpu& gpu){return execute(i,x,gpu);}};
    }
    void prepare(int i,const std::string& p,int rows) {
        require(executor_identity_==ane::HybridFfn::executor_configuration_identity(),"encoder runtime recipe changed mid-process");
        require(i>=0 && i<36 && encoder_runtime::eligible_rows(rows),"encoder runtime layer/row scope unsupported");
        const bool new_layer=p!=prefix_;
        layer_=i;rows_=rows;
        const auto plan=bridge_->plan_block(i,rows);
        // plan_block drains the previous owner. A prefill visits each layer
        // once; retaining all compiled FP32 down constants would waste RAM.
        if(new_layer)channel_graphs_.clear();
        prefix_=p;
        split_=plan.split();
        if(!split_)return;
        std::vector<Tensor> sources;
        for(const char* name:{"gate_proj","up_proj","down_proj"}) {
            auto w=weights_.at(p+".mlp."+name+".weight");
            const auto shape=std::string_view(name)=="down_proj"?mx::Shape{4096,12288}:mx::Shape{12288,4096};
            require(w.shape()==shape && w.dtype()==mx::bfloat16,"encoder runtime source geometry/precision changed");
            require(!weights_.has(p+".mlp."+name+".bias"),"encoder runtime cannot omit FFN biases");
            sources.push_back(w);
        }
        bridge_->stage(i,rows,std::move(sources));
    }
    Tensor execute(int i,const Tensor& x,const TextFfnHooks::Gpu& gpu) {
        require(i==layer_ && x.shape()==mx::Shape{1,rows_,4096} && x.dtype()==mx::bfloat16,
                "encoder runtime invocation/input mismatch");
        if(!split_)return gpu(x);
        // The ANE staging queue imports this allocation. Retain real owners,
        // and prove readiness instead of treating MLX graph creation as a fence.
        const auto ready_start=Clock::now();
        // MLX 0.32 eval waits on the output's dependency event, including an
        // already async-produced input. No separate whole-stream drain is
        // needed before importing this produced allocation on the ANE queue.
        // run_channels evals packed/padded x before creating its device view.
        // This dependency wait also produces x: avoid a second pre-pack fence.
        if(gpu_recipe_==encoder_runtime::GpuRecipe::LegacyFp32Constants) {mx::eval(x);mx::synchronize();}
        stats_.input_ready_seconds+=std::chrono::duration<double>(Clock::now()-ready_start).count();
        if(dynamic_graph_) {
            ane::HybridFfn::ChannelGpu channel=[this](const Tensor& value,int first,int count) {
                require(first==0 && count==bridge_->gpu_channels(),"encoder dynamic GPU range changed");
                const auto start=Clock::now();
                auto a=dynamic_graph_({value,weights_.at(prefix_+".mlp.gate_proj.weight"),
                    weights_.at(prefix_+".mlp.up_proj.weight"),weights_.at(prefix_+".mlp.down_proj.weight")});
                ++stats_.gpu_graph_calls;
                stats_.gpu_graph_encode_host_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
                return std::pair{a[0],a[1]};
            };
            auto next=[this](int n) {
                std::vector<ane::FfnWeight> result;
                if(n>=36)return result;
                constexpr uint64_t source_bytes=uint64_t(3)*4096*12288*2;
                const auto admission=ane::admit_memory(ane::observe_runtime_memory(mx::get_active_memory()),
                    {uint64_t(4)<<30,budget_},bridge_->metrics().runtime_weight_estimated_bytes,source_bytes);
                if(!admission.allowed()){++stats_.next_source_skips;return result;}
                const auto p=prefix_.substr(0,prefix_.find_last_of('.')+1)+std::to_string(n);
                for(const char* name:{"gate_proj","up_proj","down_proj"}) {
                    const auto key=p+".mlp."+name+".weight";
                    if(!weights_.has(key))return std::vector<ane::FfnWeight>{};
                    const auto w=weights_.at(key);
                    require(w.dtype()==mx::bfloat16 && w.shape()==(std::string_view(name)=="down_proj"?
                        mx::Shape{4096,12288}:mx::Shape{12288,4096}),"encoder future source geometry changed");
                    result.push_back({w,std::nullopt,std::nullopt});
                }
                ++stats_.next_source_submissions;return result;
            };
            return bridge_->run(i,x,gpu,cancelled_,nullptr,channel,next);
        }
        auto it=channel_graphs_.find(prefix_);
        if(it==channel_graphs_.end()) {
            const auto p=prefix_;const auto& weights=weights_;const int count=bridge_->gpu_channels();
            auto fn=mx::compile([&weights,p,count](const std::vector<Tensor>& a) {
                const int first=0;
                auto gate=mx::matmul(a[0],mx::transpose(slice_axis(weights.at(p+".mlp.gate_proj.weight"),0,first,first+count)));
                auto up=mx::matmul(a[0],mx::transpose(slice_axis(weights.at(p+".mlp.up_proj.weight"),0,first,first+count)));
                auto hidden=silu(gate)*up;
                auto down=slice_axis(weights.at(p+".mlp.down_proj.weight"),1,first,first+count);
                auto partial=mx::matmul(mx::astype(hidden,mx::float32),mx::transpose(mx::astype(down,mx::float32)));
                return std::vector<Tensor>{partial,hidden};
            });
            // Geometry is fixed for this bridge; no item() inside a traced graph.
            auto channel=[fn,count](const Tensor& value,int first,int n) mutable {
                require(first==0 && n==count,"encoder runtime GPU range changed");
                auto a=fn({value});return std::pair{a[0],a[1]};
            };
            it=channel_graphs_.emplace(prefix_,std::move(channel)).first;
            ++stats_.gpu_graph_builds;
        }
        ++stats_.gpu_graph_calls;
        return bridge_->run(i,x,gpu,cancelled_,nullptr,it->second);
    }
    HybridMetrics finish() {
        bridge_->drain();mx::synchronize();
        auto metrics=bridge_->metrics();metrics.block_count=36;
        metrics.runtime_weight_source_recipe+="+encoder-gpu:"+encoder_runtime::gpu_recipe_identity();
        channel_graphs_.clear();
        const bool retained=cache_ && bridge_->available();
        if(retained) {
            cache_->key_=cache_key_;cache_->graph_=std::move(dynamic_graph_);cache_->bridge_=std::move(bridge_);
            cache_->borrowed_=false;cache_=nullptr;
        } else {dynamic_graph_={};bridge_.reset();}
        metrics.runtime_weight_session_released=!retained;
        return metrics;
    }
    HybridMetrics metrics() const {return bridge_->metrics();}
    EncoderRuntimeStats statistics() const {return stats_;}
};
} // namespace tc::qwen21
