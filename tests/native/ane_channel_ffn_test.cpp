#include "../../native/backends/ane_ffn.hpp"
#include "../../native/models/z_image/metal/projection.hpp"
#include <cmath>
#include <iostream>

using namespace tc;
void check(bool ok,const std::string&reason){if(!ok)throw std::runtime_error(reason);}
int main(int argc,char**argv) {
    if(argc!=3)return 2;
    try {
        std::atomic<bool> cancelled{false};
        // Same kernel math on original physical pitch vs compact reference.
        // Include nonzero row/column starts, tails, both supported dtypes.
        for(auto dtype:{mx::bfloat16,mx::float16}) for(int tile:{16,32}) {
            auto x=mx::astype(mx::random::normal({1,33,512},mx::float32,mx::random::key(3))*.1f,dtype);
            auto w=mx::astype(mx::random::normal({96,1024},mx::float32,mx::random::key(4))*.1f,dtype);
            auto compact=mx::contiguous(slice_axis(slice_axis(w,0,7,72),1,512,1024));
            auto wide=z_metal::projection_range(x,w,7,72,512,1024,tile);
            auto packed=z_metal::projection_range(x,compact,0,65,0,512,tile);
            check(mx::all(wide==packed).item<bool>(),"physical dense range vs compact MPP not bit-exact");
            bool bad_tile=false;
            try { z_metal::projection_range(x,w,7,72,512,1024,0); }
            catch(const std::invalid_argument&) { bad_tile=true; }
            check(bad_tile,"invalid MPP range tile accepted");
        }
        constexpr int h=128,f=1024;
        const std::string p="transformer_blocks.0.img_mlp.";
        std::vector<float> gu(size_t(2*f)*h,0),out(size_t(h)*f,0);
        for(int c=0;c<f;++c) {gu[size_t(c)*h+c%h]=.125f;gu[size_t(f+c)*h+c%h]=-.25f;}
        for(int c=0;c<h;++c) {out[size_t(c)*f+c]=.25f;out[size_t(c)*f+512+c]=-.125f;}
        auto wg=mx::astype(Tensor(gu.data(),{2*f,h},mx::float32),mx::bfloat16);
        auto wd=mx::astype(Tensor(out.data(),{h,f},mx::float32),mx::bfloat16);
        auto snap_g=mx::copy(wg),snap_d=mx::copy(wd);mx::eval(wg,wd,snap_g,snap_d);
        Weights w;w.bind_arrays({p+"gate_up.weight",p+"out.weight"},{wg,wd});
        const auto adapter_path=std::filesystem::path(argv[2])/"channel-lora.safetensors";
        mx::save_safetensors(adapter_path.string(),{
            {"transformer."+p+"gate_up.lora_A.weight",mx::full({4,h},.02f,mx::bfloat16)},
            {"transformer."+p+"gate_up.lora_B.weight",mx::full({2*f,4},.03f,mx::bfloat16)},
            {"transformer."+p+"out.lora_A.weight",mx::full({4,f},.04f,mx::bfloat16)},
            {"transformer."+p+"out.lora_B.weight",mx::full({h,4},.05f,mx::bfloat16)}});
        setenv("TURBOCIDER_ANE_BACKEND","private",1);setenv("TURBOCIDER_ALLOW_PRIVATE_ANE","1",1);
        setenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH","w8a8",1);setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","512",1);
        setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS","1",1);
        setenv("TURBOCIDER_PRIVATE_ANE_PREFETCH","1",1);
        setenv("TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD","1",1);
        ane::HybridFfn runtime(argv[1],h,f,512u<<20,cancelled,true);
        check(runtime.available()&&runtime.channel_split()&&runtime.ane_channels()==512&&runtime.gpu_channels()==512,runtime.reason());
        check(runtime.metrics().runtime_weight_a8_lookahead_enabled,"channel executor did not bind A8 lookahead");
        check(runtime.selection_label().find("intermediate-channel")!=std::string::npos &&
              runtime.selection_label().find("w8a8_hadamard")!=std::string::npos &&
              runtime.resolve_selection("pending; physical placement unverified; preserved diagnostic").ends_with("; preserved diagnostic"),
              "actual channel/data-path selection label or diagnostic suffix lost");
        {
            setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","513",1);
            bool invalid=false;
            try { ane::HybridFfn rejected(argv[1],h,f,512u<<20,cancelled,true); }
            catch(const std::exception&) { invalid=true; }
            check(invalid,"malformed private channel count reported GPU success");
            setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","512",1);
            ane::HybridFfn unavailable(argv[1],h,f,1,cancelled,true);
            check(!unavailable.available() && unavailable.metrics().runtime_failed &&
                  unavailable.backend_label()=="mlx_cpp_metal" && unavailable.precision_label()=="bf16" &&
                  unavailable.backend_label(true)=="mlx_cpp_metal_gguf" && unavailable.precision_label(true)=="gguf_native_gpu",
                  "unselected/failed private executor masqueraded as Core ML");
        }
        double worst=0,worst_partial_normalized=0;
        for(int rows:{17,67}) {
            auto x=mx::astype(mx::random::normal({1,rows,h},mx::float32,mx::random::key(rows))*.3f,mx::bfloat16);
            std::optional<Tensor> base;
            for(float strength:{0.f,.75f,-.5f,0.f}) {
                // Model adapter switches clear/rebind checkpoint handles.
                // apply_loras({}) alone is intentionally not a detach API.
                w.clear();w.bind_arrays({p+"gate_up.weight",p+"out.weight"},{wg,wd});
                w.apply_loras(strength?std::vector<LoRAAsset>{{adapter_path.string(),strength}}:std::vector<LoRAAsset>{},
                              "transformer",[](const std::string&,int,int){},cancelled,true);
                runtime.begin_request(std::to_string(strength));
                auto full=[&](const Tensor&input) {
                    auto parts=mx::split(w.project(input,p+"gate_up"),2,-1);
                    return w.project(silu(parts[0])*parts[1],p+"out");
                };
                int down_calls=0;
                ane::HybridFfn::Adapter adapter{
                    [&](const Tensor&input){return std::make_pair(w.lora_delta_slice(input,p+"gate_up",0,f,0,h),
                                                                 w.lora_delta_slice(input,p+"gate_up",f,2*f,0,h));},
                    [&](const Tensor&hidden,const Tensor&partial){
                        ++down_calls;check(hidden.shape()==mx::Shape({1,rows,f}),"down-LoRA did not receive joined FULL hidden");
                        auto delta=w.lora_delta_slice(hidden,p+"out",0,h,0,f);
                        return mx::astype(mx::astype(partial,mx::float32)+mx::astype(delta,mx::float32),partial.dtype());
                    }};
                auto gu=mx::split(w.at(p+"gate_up.weight"),2,0);mx::eval(gu);
                runtime.stage(0,rows,{gu[0],gu[1],w.at(p+"out.weight")});
                const auto a8_before=runtime.metrics().runtime_weight_a8_prefetches;
                auto y=runtime.run(0,x,full,cancelled,strength?&adapter:nullptr,
                    [&](const Tensor&input,int first,int count){
                        auto g=w.project_slice(input,p+"gate_up",first,first+count,0,h,false);
                        auto u=w.project_slice(input,p+"gate_up",f+first,f+first+count,0,h,false);
                        auto hidden=silu(g)*u;
                        return std::make_pair(w.project_base_slice(hidden,p+"out",0,h,first,first+count,false),hidden);
                    });
                check(runtime.metrics().runtime_weight_a8_prefetches-a8_before==(rows==17?0u:2u),
                      "channel padded-row lookahead/slot reuse not exercised");
                auto expected=full(x); mx::eval(expected);
                auto source_gu=mx::split(w.project(x,p+"gate_up"),2,-1);
                auto source_hidden=silu(source_gu[0])*source_gu[1]; mx::eval(source_hidden);
                auto source_gpu=w.project_base_slice(slice_axis(source_hidden,-1,0,512),p+"out",0,h,0,512,false);
                auto source_ane=w.project_base_slice(slice_axis(source_hidden,-1,512,f),p+"out",0,h,512,f,false);
                auto squared_error=mx::sum(mx::square(mx::astype(y,mx::float32)-mx::astype(expected,mx::float32)));
                const float relative=mx::sqrt(mx::sum(mx::square(mx::astype(y,mx::float32)-mx::astype(expected,mx::float32)))/
                                              mx::sum(mx::square(mx::astype(expected,mx::float32)))).item<float>();
                // The fixture deliberately cancels +.25*h with -.125*h.
                // Quantization error is bounded against the two source
                // partials' energy, not an arbitrarily tiny cancelled sum.
                // Preserve/report raw full-output L2 for model qualification.
                auto energy=mx::sum(mx::square(mx::astype(source_gpu,mx::float32)))+
                            mx::sum(mx::square(mx::astype(source_ane,mx::float32)));
                if(strength)energy=energy+mx::sum(mx::square(mx::astype(w.lora_delta_slice(source_hidden,p+"out",0,h,0,f),mx::float32)));
                const float partial_normalized=mx::sqrt(squared_error/energy).item<float>();
                check(std::isfinite(relative)&&std::isfinite(partial_normalized)&&partial_normalized<.04f&&!runtime.metrics().runtime_failed,
                      "channel source oracle: rows="+std::to_string(rows)+" strength="+std::to_string(strength)+
                      " l2="+std::to_string(relative)+" partial_normalized="+std::to_string(partial_normalized)+" reason="+runtime.reason());
                worst=std::max(worst,double(relative));worst_partial_normalized=std::max(worst_partial_normalized,double(partial_normalized));
                check(down_calls==(strength?1:0),"down-LoRA rounded separately per channel/chunk");
                if(strength==0) {if(base)check(mx::all(*base==y).item<bool>(),"channel base/A/B/base state leaked");else base=y;}
                check(mx::all(w.at(p+"gate_up.weight")==snap_g).item<bool>()&&mx::all(w.at(p+"out.weight")==snap_d).item<bool>(),"channel staging mutated base weights");
            }
        }
        auto metrics=runtime.metrics();check(metrics.runtime_weight_channel_blocks==8&&metrics.runtime_weight_device_io_calls>=8,"channel execution counters missing");
        // Model-level producer/consumer path: first block prefetches layer 1,
        // then stage(1) must consume that source-matched bank rather than refill.
        runtime.begin_request("two-layer-prefetch");
        auto gu_parts_prefetch=mx::split(wg,2,0);mx::eval(gu_parts_prefetch);
        auto px=mx::full({1,67,h},.25f,mx::bfloat16);
        std::vector<Tensor> sources{gu_parts_prefetch[0],gu_parts_prefetch[1],wd};
        auto full=[&](const Tensor&input){auto gu=mx::split(w.project(input,p+"gate_up"),2,-1);return w.project(silu(gu[0])*gu[1],p+"out");};
        auto partial=[&](const Tensor&input,int first,int count){auto g=w.project_base_slice(input,p+"gate_up",first,first+count,0,h,false);
            auto u=w.project_base_slice(input,p+"gate_up",f+first,f+first+count,0,h,false);auto hidden=silu(g)*u;
            return std::make_pair(w.project_base_slice(hidden,p+"out",0,h,first,first+count,false),hidden);};
        const auto before_prefetch=runtime.metrics();
        runtime.stage(0,67,sources);
        auto pa=runtime.run(0,px,full,cancelled,nullptr,partial,[&](int next){
            std::vector<ane::FfnWeight> result;if(next==1)for(const auto&s:sources)result.push_back({s,std::nullopt,std::nullopt});return result;});
        runtime.stage(1,67,sources);auto pb=runtime.run(1,px,full,cancelled,nullptr,partial);
        check(mx::all(pa==pb).item<bool>(),"source-identical prefetched bank output changed");
        auto after_prefetch=runtime.metrics();
        check(after_prefetch.runtime_weight_prefetch_submissions==before_prefetch.runtime_weight_prefetch_submissions+1&&
              after_prefetch.runtime_weight_prefetch_hits==before_prefetch.runtime_weight_prefetch_hits+1,"model did not consume prefetched bank");
        // Failure in the SECOND ANE chunk must select the whole-operation GPU
        // callback, not combine valid GPU partial with poisoned ANE scratch.
        std::vector<float> poison(67*h,.1f);poison[50*h]=std::numeric_limits<float>::quiet_NaN();
        auto bad=mx::astype(Tensor(poison.data(),{1,67,h},mx::float32),mx::bfloat16);
        auto gu_parts=mx::split(wg,2,0);mx::eval(gu_parts);
        runtime.begin_request("late-failure");runtime.stage(0,67,{gu_parts[0],gu_parts[1],wd});
        int full_recomputations=0;
        auto recovered=runtime.run(0,bad,[&](const Tensor&input){++full_recomputations;return mx::full(input.shape(),7.f,input.dtype());},
            cancelled,nullptr,[](const Tensor&input,int,int count){
                return std::make_pair(mx::zeros(input.shape(),input.dtype()),mx::zeros({1,input.shape(1),count},input.dtype()));
            });
        check(full_recomputations==1&&mx::all(recovered==Tensor(7.f,mx::bfloat16)).item<bool>()&&runtime.metrics().runtime_failed,
              "late channel failure published a partial result instead of full GPU recomputation");
        std::cout<<"PASS channel MLX/W8 Executor: physical range bit-exact, all rows/tail padding, immutable full source, base/A/B/base, ONE full-hidden down-LoRA, raw_source_relative_l2="<<worst
                 <<" partial_energy_normalized_l2="<<worst_partial_normalized<<"; model quality NOT qualified\n";
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
