#include "../../native/backends/ane_ffn.hpp"
#include "../../native/models/z_image/metal/projection.hpp"
#include "../../native/models/z_image/metal/swiglu_gemm.hpp"
#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
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
        for(auto dtype:{mx::bfloat16,mx::float16}) for(int tile:{16,32}) for(int rows:{33,4128}) {
            auto x=mx::astype(mx::random::normal({1,rows,512},mx::float32,mx::random::key(3))*.1f,dtype);
            auto w=mx::astype(mx::random::normal({96,1024},mx::float32,mx::random::key(4))*.1f,dtype);
            auto compact=mx::contiguous(slice_axis(slice_axis(w,0,7,72),1,512,1024));
            auto wide=z_metal::projection_range(x,w,7,72,512,1024,tile);
            auto packed=z_metal::projection_range(x,compact,0,65,0,512,tile);
            check(mx::all(wide==packed).item<bool>(),"physical dense range vs compact MPP not bit-exact");
            auto wide32=dense_gpu::projection_range(x,w,7,72,512,1024,tile,true);
            auto packed32=dense_gpu::projection_range(x,compact,0,65,0,512,tile,true);
            check(wide32.dtype()==mx::float32 && mx::all(wide32==packed32).item<bool>(),"F32 partial physical/compact leading dimension mismatch");
            check(mx::all(mx::astype(wide32,dtype)==wide).item<bool>(),"F32 partial changed established projection accumulation/output boundary");
            bool bad_tile=false;
            try { z_metal::projection_range(x,w,7,72,512,1024,0); }
            catch(const std::invalid_argument&) { bad_tile=true; }
            check(bad_tile,"invalid MPP range tile accepted");
        }
        std::cout<<"PASS dense GPU F32 partial: original FP16/BF16 operands, physical pitch/tails and exact old final rounding\n";
        for(int rows:{33,4128}) {
            auto x=mx::astype(mx::random::normal({1,rows,3840},mx::float32,mx::random::key(13))*.1f,mx::bfloat16);
            auto g=mx::astype(mx::random::normal({10240,3840},mx::float32,mx::random::key(14))*.01f,mx::bfloat16);
            auto u=mx::astype(mx::random::normal({10240,3840},mx::float32,mx::random::key(15))*.01f,mx::bfloat16);
            auto snap_g=mx::copy(g),snap_u=mx::copy(u);mx::eval({x,g,u,snap_g,snap_u});
            for (int count : {128,384,512}) for (int tile : {128,256}) {
                constexpr int first=256;
                auto compact_g=mx::contiguous(slice_axis(g,0,first,first+count));
                auto compact_u=mx::contiguous(slice_axis(u,0,first,first+count));
                auto wide=z_metal::swiglu_dual_gemm_range(x,g,u,first,count,tile);
                auto compact=z_metal::swiglu_dual_gemm_range(x,compact_g,compact_u,0,count,tile);
                check(mx::all(wide==compact).item<bool>(),"dual SwiGLU physical vs compact range not bit-exact");
                if (tile==128) {
                    auto up=z_metal::projection_range(x,u,first,first+count,0,3840);
                    auto separate=z_metal::swiglu_gemm_range(x,g,up,first,count);
                    check(mx::all(wide==separate).item<bool>(),"dual128 changed separate gate/up BF16 rounding");
                }
            }
            bool bad_tile=false,bad_range=false;
            try { z_metal::swiglu_dual_gemm_range(x,g,u,0,512,64); }
            catch(const std::invalid_argument&) { bad_tile=true; }
            try { z_metal::swiglu_dual_gemm_range(x,g,u,10112,256); }
            catch(const std::invalid_argument&) { bad_range=true; }
            check(bad_tile&&bad_range,"dual SwiGLU invalid range/tile accepted");
            check(mx::all(g==snap_g).item<bool>()&&mx::all(u==snap_u).item<bool>(),"dual SwiGLU mutated base weights");
            std::cout<<"PASS dual GPU channel head: immutable physical/compact ranges, tails, tiles and BF16 separate-projection parity\n";
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
        setenv("TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE","1",1);
        ane::HybridFfn runtime(argv[1],h,f,512u<<20,cancelled,true);
        check(runtime.available()&&runtime.channel_split()&&runtime.ane_channels()==512&&runtime.gpu_channels()==512,runtime.reason());
        const int fg=runtime.gpu_channels(),fa=runtime.ane_channels();
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
            const auto range_identity=ane::HybridFfn::executor_configuration_identity();
            setenv("TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE","0",1);
            check(range_identity!=ane::HybridFfn::executor_configuration_identity(),"LoRA correction policy absent from executor identity");
            setenv("TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE","2",1);
            bool bad_range_policy=false;
            try { ane::HybridFfn rejected(argv[1],h,f,1,cancelled,true); }
            catch(const std::exception&) { bad_range_policy=true; }
            check(bad_range_policy,"invalid LoRA channel policy reported GPU success");
            setenv("TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE","1",1);
            const auto before_defer_identity=ane::HybridFfn::executor_configuration_identity();
            setenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN","1",1);
            check(before_defer_identity!=ane::HybridFfn::executor_configuration_identity(),"deferred policy absent from identity");
            bool invalid_defer=false;
            try { ane::HybridFfn bad(argv[1],h,f,1,cancelled,true); }
            catch(const std::exception&) {invalid_defer=true;}
            check(invalid_defer,"defer without fixed async accepted");
            setenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN","2",1);
            invalid_defer=false;
            try { ane::HybridFfn bad(argv[1],h,f,1,cancelled,true); }
            catch(const std::exception&) {invalid_defer=true;}
            check(invalid_defer,"malformed defer flag accepted");
            unsetenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN");
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
                // Shared model/calibration factories against the previous
                // request-local compiled bodies, including base/A/B/base.
                auto old_full=mx::compile([&](const std::vector<Tensor> &a) {return std::vector<Tensor>{full(a[0])};});
                auto new_full=qwen21::runtime_ffn::full(w,p);
                check(mx::all(old_full({x})[0]==new_full({x})[0]).item<bool>(),"shared Qwen full GPU graph changed arithmetic");
                for(int begin:{0,fg}) {
                    const int count=begin?fa:fg;
                    auto old_head=mx::compile([&,begin,count](const std::vector<Tensor> &a) {
                        auto g=w.project_slice(a[0],p+"gate_up",begin,begin+count,0,h,false);
                        auto u=w.project_slice(a[0],p+"gate_up",f+begin,f+begin+count,0,h,false);
                        auto hidden=silu(g)*u;
                        return std::vector<Tensor>{w.project_base_slice(hidden,p+"out",0,h,begin,begin+count,false),hidden};
                    });
                    auto newer=qwen21::runtime_ffn::channels(w,p,begin,count,h,f)({x});
                    auto older=old_head({x});
                    check(mx::all(newer[0]==older[0]).item<bool>() && mx::all(newer[1]==older[1]).item<bool>(),
                        "shared Qwen physical channel graph changed base/hidden arithmetic");
                    auto old_delta=mx::compile([&,begin,count](const std::vector<Tensor> &a) {
                        return std::vector<Tensor>{mx::contiguous(w.lora_delta_slice(a[0],p+"gate_up",begin,begin+count,0,h)),
                            mx::contiguous(w.lora_delta_slice(a[0],p+"gate_up",f+begin,f+begin+count,0,h))};
                    });
                    auto new_delta=qwen21::runtime_ffn::corrections(w,p,begin,count,h,f)({x});
                    auto prior=old_delta({x});
                    check(mx::all(new_delta[0]==prior[0]).item<bool>() && mx::all(new_delta[1]==prior[1]).item<bool>(),
                        "shared Qwen correction graph changed LoRA intersections/rounding");
                }
                auto halves=mx::split(w.project(x,p+"gate_up"),2,-1);auto full_hidden=silu(halves[0])*halves[1];
                auto down_base=w.project_base_slice(full_hidden,p+"out",0,h,0,f,false);
                auto old_down=mx::compile([&](const std::vector<Tensor> &a) {
                    auto delta=w.lora_delta_slice(a[0],p+"out",0,h,0,f);
                    return std::vector<Tensor>{mx::astype(mx::astype(a[1],mx::float32)+mx::astype(delta,mx::float32),a[1].dtype())};
                });
                auto down=qwen21::runtime_ffn::down_add(w,p,h,f);
                check(mx::all(down({full_hidden,down_base})[0]==old_down({full_hidden,down_base})[0]).item<bool>(),
                    "shared Qwen down graph changed ONE-full-hidden correction");
                int down_calls=0,full_gate_calls=0,range_gate_calls=0;
                ane::HybridFfn::Adapter adapter{
                    [&](const Tensor&input){++full_gate_calls;return std::make_pair(w.lora_delta_slice(input,p+"gate_up",0,f,0,h),
                                                                 w.lora_delta_slice(input,p+"gate_up",f,2*f,0,h));},
                    [&](const Tensor&hidden,const Tensor&partial){
                        ++down_calls;check(hidden.shape()==mx::Shape({1,rows,f}),"down-LoRA did not receive joined FULL hidden");
                        auto delta=w.lora_delta_slice(hidden,p+"out",0,h,0,f);
                        return mx::astype(mx::astype(partial,mx::float32)+mx::astype(delta,mx::float32),partial.dtype());
                    },
                    [&](const Tensor&input,int first,int count){
                        ++range_gate_calls;check(first==512&&count==512,"ANE correction received wrong physical range");
                        return std::make_pair(w.lora_delta_slice(input,p+"gate_up",first,first+count,0,h),
                                              w.lora_delta_slice(input,p+"gate_up",f+first,f+first+count,0,h));
                    }};
                auto gu=mx::split(w.at(p+"gate_up.weight"),2,0);mx::eval(gu);
                runtime.stage(0,rows,{gu[0],gu[1],w.at(p+"out.weight")});
                const auto a8_before=runtime.metrics().runtime_weight_a8_prefetches;
                auto channel_gpu=[&](const Tensor&input,int first,int count){
                        auto g=w.project_slice(input,p+"gate_up",first,first+count,0,h,false);
                        auto u=w.project_slice(input,p+"gate_up",f+first,f+first+count,0,h,false);
                        auto hidden=silu(g)*u;
                        return std::make_pair(w.project_base_slice(hidden,p+"out",0,h,first,first+count,false),hidden);
                    };
                auto y=runtime.run(0,x,full,cancelled,strength?&adapter:nullptr,channel_gpu);
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
                check(full_gate_calls==0&&range_gate_calls==(strength?1:0),"narrow correction computed full gate/up or ran in base");
                if(strength) {
                    auto legacy=adapter;legacy.gate_up_channels={};
                    full_gate_calls=range_gate_calls=down_calls=0;
                    runtime.stage(0,rows,{gu[0],gu[1],w.at(p+"out.weight")});
                    auto legacy_y=runtime.run(0,x,full,cancelled,&legacy,channel_gpu);
                    check(mx::all(y==legacy_y).item<bool>(),"narrow and full channel corrections not bit-exact");
                    check(full_gate_calls==1&&range_gate_calls==0&&down_calls==1,"legacy correction or full-hidden down callback contract changed");
                }
                if(strength==0) {if(base)check(mx::all(*base==y).item<bool>(),"channel base/A/B/base state leaked");else base=y;}
                check(mx::all(w.at(p+"gate_up.weight")==snap_g).item<bool>()&&mx::all(w.at(p+"out.weight")==snap_d).item<bool>(),"channel staging mutated base weights");
            }
        }
        std::cout<<"PASS shared compiled Qwen FFN graph factories: full/channel/corrections/down, base/A/B/base\n";
        auto metrics=runtime.metrics();check(metrics.runtime_weight_channel_blocks==12&&metrics.runtime_weight_device_io_calls>=12,"channel execution counters missing");
        check(metrics.runtime_weight_lora_channel_range_calls==4&&metrics.runtime_weight_lora_channel_full_calls==4,
              "actual narrow/full LoRA correction receipts missing");
        if(metrics.runtime_weight_code_cache.enabled && metrics.runtime_weight_code_cache.native_surface_storage) {
            const auto &cache=metrics.runtime_weight_code_cache;
            check(cache.surface_bind_hits>0 && cache.copy_hits==0 && cache.ready_entries>0 &&
                  cache.live_capacity_bytes<=cache.budget_bytes && cache.peak_capacity_bytes<=cache.budget_bytes,
                  "Private channel lacks real native-surface binding/capacity evidence");
            std::cout<<"PASS Private native cached bindings: base/A/B/base, no cache copies and bounded readers\n";
        }
        std::cout<<"PASS channel LoRA range callback: base/A/B/base, exact legacy parity, all-row tails and ONE full-hidden down correction\n";
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
        {
            setenv("TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE","0",1);
            ane::HybridFfn legacy_policy(argv[1],h,f,512u<<20,cancelled,true);
            setenv("TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE","1",1);
            int full_calls=0,down_calls=0;
            ane::HybridFfn::Adapter controlled{
                [&](const Tensor&input){++full_calls;return std::make_pair(mx::zeros({1,input.shape(1),f},input.dtype()),
                                                                         mx::zeros({1,input.shape(1),f},input.dtype()));},
                [&](const Tensor&hidden,const Tensor&base){++down_calls;check(hidden.shape()==mx::Shape({1,67,f}),"off policy lost full hidden");return base;},
                [](const Tensor&,int,int)->std::pair<Tensor,Tensor>{throw std::runtime_error("off policy called range callback");}};
            legacy_policy.begin_request("explicit-range-off");legacy_policy.stage(0,67,sources);
            auto got=legacy_policy.run(0,px,full,cancelled,&controlled,partial);mx::eval(got);
            check(full_calls==1&&down_calls==1&&legacy_policy.metrics().runtime_weight_lora_channel_full_calls==1&&
                  legacy_policy.metrics().runtime_weight_lora_channel_range_calls==0,"explicit off policy did not execute legacy corrections");
        }
        {
            runtime.begin_request("malformed-channel-correction");runtime.stage(0,67,sources);
            ane::HybridFfn::Adapter malformed{
                [](const Tensor&input){return std::make_pair(mx::zeros({1,input.shape(1),f},input.dtype()),mx::zeros({1,input.shape(1),f},input.dtype()));},
                [](const Tensor&,const Tensor&base){return base;},
                [](const Tensor&input,int,int){return std::make_pair(mx::zeros({1,input.shape(1),f},input.dtype()),mx::zeros({1,input.shape(1),f},input.dtype()));}};
            bool rejected=false;
            try { runtime.run(0,px,full,cancelled,&malformed,partial); }
            catch(const std::invalid_argument&error) {rejected=std::string(error.what()).find("correction geometry mismatch")!=std::string::npos;}
            runtime.drain();
            check(rejected,"narrow callback published a full-width or malformed correction");
        }
        const auto before_prefetch=runtime.metrics();
        {
            setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);
            ane::HybridFfn routed(argv[1],h,f,512u<<20,cancelled,true);
            unsetenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC");
            routed.set_gpu_layers({2});routed.begin_request("explicit-gpu-block");
            const auto skip=routed.plan_block(2,67);
            check(skip.mode==ane::RowScheduler::Mode::Gpu && !skip.measured() &&
                routed.metrics().runtime_weight_forced_gpu_blocks==1 && !routed.metrics().runtime_calls,
                "explicit GPU block still issued an ANE plan/call");
            bool rejected=false;
            try{routed.stage(2,67,sources);}catch(const std::invalid_argument&){rejected=true;}
            check(rejected,"explicit GPU block staged private scratch");
            rejected=false;try{routed.set_gpu_layers({2,2});}catch(const std::invalid_argument&){rejected=true;}
            check(rejected && routed.metrics().runtime_weight_gpu_layers==std::vector<int>{2},
                "invalid GPU policy changed the active routing");
            auto restored=full(px);mx::eval(restored); // family full-GPU path, no bridge
            routed.set_gpu_layers({});routed.begin_request("back-to-hybrid");
            check(routed.plan_block(0,67).mode==ane::RowScheduler::Mode::HybridUntimed,"GPU override leaked after removal");
            rejected=false;try{routed.set_gpu_layers({2});}catch(const std::invalid_argument&){rejected=true;}
            check(rejected,"GPU policy changed while an operation owned its plan");
            routed.stage(0,67,sources);auto resumed=routed.run(0,px,full,cancelled,nullptr,partial);mx::eval(resumed);
            check(routed.metrics().runtime_calls>0 && !routed.metrics().runtime_failed,
                "GPU override removal broke resumed hybrid execution");
            std::cout<<"PASS explicit GPU layer policy: full-family GPU plan, no bridge/staging, immutable policy and hybrid resume\n";
        }
        runtime.stage(0,67,sources);
        auto pa=runtime.run(0,px,full,cancelled,nullptr,partial,[&](int next){
            std::vector<ane::FfnWeight> result;if(next==1)for(const auto&s:sources)result.push_back({s,std::nullopt,std::nullopt});return result;});
        runtime.stage(1,67,sources);auto pb=runtime.run(1,px,full,cancelled,nullptr,partial);
        check(mx::all(pa==pb).item<bool>(),"source-identical prefetched bank output changed");
        auto after_prefetch=runtime.metrics();
        check(after_prefetch.runtime_weight_prefetch_submissions==before_prefetch.runtime_weight_prefetch_submissions+1&&
              after_prefetch.runtime_weight_prefetch_hits==before_prefetch.runtime_weight_prefetch_hits+1,"model did not consume prefetched bank");
        for(bool deferred:{false,true}) {
            setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);
            setenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN",deferred?"1":"0",1);
            ane::HybridFfn asynchronous(argv[1],h,f,512u<<20,cancelled,true);
            unsetenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC");
            unsetenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN");
            auto execute=[&](const Tensor&input,const ane::HybridFfn::Gpu&fallback) {
                auto plan=asynchronous.plan_block(0,input.shape(1));
                check(plan.mode==ane::RowScheduler::Mode::HybridUntimed&&!plan.measured()&&plan.chunks==1,
                      "fixed async channel plan still forces a measured GPU fence");
                asynchronous.stage(0,input.shape(1),sources);
                return asynchronous.run(0,input,fallback,cancelled,nullptr,partial);
            };
            auto got=execute(px,full);
            check(mx::all(got==pb).item<bool>(),"fixed async channel changed output");
            auto saved=mx::copy(got);mx::eval(saved);
            bool caught=false;
            try {execute(px,[](const Tensor&)->Tensor{throw std::runtime_error("not called");});}
            catch(const std::runtime_error&) {caught=true;}
            // A healthy channel split does not call its full-GPU fallback.
            check(!caught&&mx::all(saved==got).item<bool>(),"fixed async channel selected full GPU or lost output ownership");
            const auto counters=asynchronous.metrics();
            check(counters.runtime_weight_async_hybrid_blocks==2&&counters.runtime_weight_untimed_hybrid_blocks==2&&
                  counters.runtime_weight_gpu_seconds==0&&counters.runtime_weight_join_seconds==0,
                  "fixed async channel receipt counts a timed GPU branch");
            check(counters.runtime_weight_deferred_join_enabled==deferred&&
                  counters.runtime_weight_deferred_join_blocks==(deferred?2u:0u),
                  "deferred join policy/actual counter mismatch");
            // Do not consume the first result before the next ANE request
            // reuses y. Returned joins retain independent head/tail owners.
            auto retained=execute(px,full);
            auto different=mx::full(px.shape(),-.4f,px.dtype());
            auto next=execute(different,full);
            mx::eval({retained,next});
            check(mx::all(retained==pb).item<bool>()&&!mx::all(retained==next).item<bool>(),
                  "pending join borrowed reusable ANE output scratch");
            auto bad=mx::concatenate({mx::full({1,33,h},.1f,mx::bfloat16),
                mx::full({1,34,h},std::numeric_limits<float>::quiet_NaN(),mx::bfloat16)},1);
            int fallbacks=0;
            auto recovered=execute(bad,[&](const Tensor&input){++fallbacks;return mx::full(input.shape(),7.f,input.dtype());});
            check(fallbacks==1&&mx::all(recovered==Tensor(7.f,mx::bfloat16)).item<bool>()&&asynchronous.metrics().runtime_failed,
                  "fixed async late chunk published partial scratch");
            check(mx::all(saved==got).item<bool>(),"failed async channel invalidated retained output");
            check(asynchronous.metrics().runtime_weight_deferred_join_blocks==(deferred?4u:0u),
                  "failed ANE chunk counted as a deferred successful join");
            std::cout<<"PASS fixed async channel: identical output, untimed receipts, ownership and full late-chunk GPU recomputation\n";
            if(deferred)std::cout<<"PASS deferred channel join: delayed consumption after y reuse and complete failure fallback\n";
        }
        // Exercise cleanup while work is in flight, not only constructor
        // gates or failures before ANE submission. Failed attempts must never
        // publish/count a deferred join, even with full-hidden down-LoRA.
        for(bool deferred:{false,true}) for(int failure=0;failure<5;++failure) {
            setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);
            setenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN",deferred?"1":"0",1);
            ane::HybridFfn op(argv[1],h,f,512u<<20,cancelled,true);
            unsetenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC");
            unsetenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN");
            int down_calls=0,full_calls=0;
            ane::HybridFfn::Adapter correction{
                [&](const Tensor&input){return std::make_pair(mx::zeros({1,input.shape(1),f},input.dtype()),
                                                             mx::zeros({1,input.shape(1),f},input.dtype()));},
                [&](const Tensor&z,const Tensor&base)->Tensor{
                    ++down_calls;
                    check(z.shape()==mx::Shape({1,67,f}),"failure fixture did not join full hidden");
                    if(failure==2)throw std::runtime_error("channel down callback failure");
                    return base;}};
            auto input=failure>=3?mx::concatenate({mx::full({1,33,h},.1f,mx::bfloat16),
                mx::full({1,34,h},std::numeric_limits<float>::quiet_NaN(),mx::bfloat16)},1):px;
            auto head=[&](const Tensor&x,int first,int count)->std::pair<Tensor,Tensor>{
                if(failure==1)throw std::runtime_error("channel GPU callback failure");
                auto result=partial(x,first,count);
                if(failure==0)cancelled.store(true);
                return result;
            };
            auto fallback=[&](const Tensor&x)->Tensor{
                ++full_calls;
                if(failure==4)throw std::runtime_error("complete GPU fallback failure");
                return mx::full(x.shape(),7.f,x.dtype());
            };
            op.plan_block(0,67);op.stage(0,67,sources);
            bool caught=false;
            try {
                auto result=op.run(0,input,fallback,cancelled,failure>=2?&correction:nullptr,head);
                check(failure==3&&mx::all(result==Tensor(7.f,mx::bfloat16)).item<bool>(),
                      "failed channel operation published partial output or swallowed an exception");
            } catch(const Cancelled&) {
                check(failure==0,"unexpected cancellation");caught=true;
            } catch(const std::runtime_error&error) {
                const std::string expected=failure==1?"channel GPU callback failure":
                    failure==2?"channel down callback failure":"complete GPU fallback failure";
                check((failure==1||failure==2||failure==4)&&error.what()==expected,"wrong channel exception");
                caught=true;
            }
            cancelled.store(false);op.drain();
            check(caught==(failure!=3)&&full_calls==(failure>=3?1:0)&&down_calls==(failure==2?1:0),
                  "channel failure callback counts changed");
            check(op.metrics().runtime_weight_deferred_join_blocks==0&&op.metrics().runtime_weight_channel_blocks==0,
                  "failed channel operation counted a successful join");
            if(failure<3) {
                op.plan_block(0,67);op.stage(0,67,sources);
                auto recovered=op.run(0,px,full,cancelled,nullptr,partial);mx::eval(recovered);
                check(mx::all(recovered==pb).item<bool>()&&op.metrics().runtime_weight_channel_blocks==1&&
                      op.metrics().runtime_weight_deferred_join_blocks==(deferred?1u:0u),
                      "channel executor was not reusable after cancellation/callback exception");
            } else {
                check(op.metrics().runtime_failed&&op.metrics().runtime_weight_fallback_blocks==1,
                      "late LoRA chunk failure did not retire ANE and select whole GPU fallback");
            }
        }
        std::cout<<"PASS channel failure cleanup: eager/deferred cancellation, GPU/down exceptions, late LoRA full fallback and fallback exception\n";
        // Independent lazy output lifetime: neither executor destruction nor
        // later GPU work may invalidate a pending join/full-hidden down-LoRA.
        for(auto dtype:{mx::bfloat16,mx::float16,mx::float32}) for(bool adapter_enabled:{false,true}) {
            auto fused=mx::astype(wg,dtype),down=mx::astype(wd,dtype);mx::eval({fused,down});
            auto halves=mx::split(fused,2,0);mx::eval(halves);
            std::vector<Tensor> typed_sources{halves[0],halves[1],down};
            Weights typed;typed.bind_arrays({p+"gate_up.weight",p+"out.weight"},{fused,down});
            auto input=mx::full({1,67,h},.25f,dtype);
            auto gpu_full=[&](const Tensor&x){auto q=mx::split(typed.project(x,p+"gate_up"),2,-1);
                return typed.project(silu(q[0])*q[1],p+"out");};
            auto gpu_range=[&](const Tensor&x,int first,int count){
                auto g=typed.project_base_slice(x,p+"gate_up",first,first+count,0,h,false);
                auto u=typed.project_base_slice(x,p+"gate_up",f+first,f+first+count,0,h,false);
                auto z=silu(g)*u;return std::make_pair(typed.project_base_slice(z,p+"out",0,h,first,first+count,false),z);};
            int down_calls=0;
            ane::HybridFfn::Adapter correction{
                [&](const Tensor&x){return std::make_pair(mx::full({1,x.shape(1),f},.01f,dtype),
                                                        mx::full({1,x.shape(1),f},.02f,dtype));},
                [&](const Tensor&z,const Tensor&base){++down_calls;
                    check(z.shape()==mx::Shape({1,67,f}),"deferred down correction did not get full hidden");
                    return base+mx::astype(mx::sum(mx::astype(z,mx::float32),-1,true)*.001f,dtype);}};
            auto execute=[&](ane::HybridFfn&op,const ane::HybridFfn::Adapter*ad){
                op.plan_block(0,67);op.stage(0,67,typed_sources);
                return op.run(0,input,gpu_full,cancelled,ad,gpu_range);};
            setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);
            setenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN","0",1);
            ane::HybridFfn eager(argv[1],h,f,512u<<20,cancelled,true);
            auto expected=execute(eager,adapter_enabled?&correction:nullptr);mx::eval(expected);
            std::optional<Tensor> retained;
            setenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN","1",1);
            {
                ane::HybridFfn lazy(argv[1],h,f,512u<<20,cancelled,true);
                retained=execute(lazy,adapter_enabled?&correction:nullptr);
                check(lazy.metrics().runtime_weight_deferred_join_blocks==1,"typed lazy join not executed");
            }
            unsetenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC");
            unsetenv("TURBOCIDER_RUNTIME_ANE_DEFER_CHANNEL_JOIN");
            mx::eval(*retained);
            check(mx::all(*retained==expected).item<bool>()&&down_calls==(adapter_enabled?2:0),
                  "deferred typed output changed or lost owners after executor destruction");
        }
        std::cout<<"PASS deferred typed lifetime: BF16/FP16/FP32 base/one-full-hidden correction after executor destruction\n";
        // F32 partials are a GPU restore/join option, not a changed hidden
        // or LoRA rank ABI. The base boundary still precedes ONE down-LoRA.
        setenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","1",1);
        setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC","1",1);
        for(int rows:{17,67})for(float strength:{0.f,.75f,-.5f,0.f}) {
            w.clear();w.bind_arrays({p+"gate_up.weight",p+"out.weight"},{wg,wd});
            w.apply_loras(strength?std::vector<LoRAAsset>{{adapter_path.string(),strength}}:std::vector<LoRAAsset>{},
                "transformer",[](const std::string&,int,int){},cancelled,true);
            auto input=mx::astype(mx::random::normal({1,rows,h},mx::float32,mx::random::key(rows))*.3f,mx::bfloat16);
            auto full=qwen21::runtime_ffn::full(w,p);
            auto head=qwen21::runtime_ffn::channels(w,p,0,fg,h,f,true);
            auto delta=qwen21::runtime_ffn::corrections(w,p,fg,fa,h,f);
            auto full_delta=qwen21::runtime_ffn::corrections(w,p,0,f,h,f);
            auto down=qwen21::runtime_ffn::down_add(w,p,h,f);
            int down_count=0;
            ane::HybridFfn::Adapter correction{
                [&](const Tensor&x) {auto result=full_delta({x});return std::make_pair(result[0],result[1]);},
                [&](const Tensor&hidden,const Tensor&base) {
                    ++down_count;check(hidden.shape()==mx::Shape({1,rows,f}) && hidden.dtype()==mx::bfloat16 &&
                        base.dtype()==mx::bfloat16,"F32 join widened hidden or skipped canonical base rounding before down-LoRA");
                    return down({hidden,base})[0];
                },[&](const Tensor&x,int first,int count) {
                    check(first==fg && count==fa,"F32 correction range changed");
                    auto result=delta({x});return std::make_pair(result[0],result[1]);
                }};
            ane::HybridFfn op(argv[1],h,f,512u<<20,cancelled,true);
            check(op.available() && op.fp32_channel_join() && op.metrics().runtime_weight_source_recipe.ends_with("+fp32-partial-join-v1"),
                "F32 channel contract not selected/reported");
            auto halves=mx::split(wg,2,0);mx::eval(halves);
            op.plan_block(0,rows);op.stage(0,rows,{halves[0],halves[1],wd});
            auto output=op.run(0,input,[&](const Tensor&x){return full({x})[0];},cancelled,strength?&correction:nullptr,
                [&](const Tensor&x,int first,int count) {
                    check(first==0 && count==fg,"F32 GPU head physical range changed");
                    auto result=head({x});return std::make_pair(result[0],result[1]);
                });
            mx::eval(output);const auto metrics=op.metrics();
            check(output.shape()==input.shape() && output.dtype()==input.dtype() && mx::all(mx::isfinite(output)).item<bool>() &&
                down_count==(strength?1:0) && metrics.runtime_calls==uint64_t(rows==17?1:3) &&
                metrics.runtime_weight_fallback_blocks==0 && metrics.runtime_weight_overflow_retries==0,
                "F32 channel join lost rows, down-LoRA, or actual Private execution");
        }
        w.clear();w.bind_arrays({p+"gate_up.weight",p+"out.weight"},{wg,wd});
        {
            ane::HybridFfn op(argv[1],h,f,512u<<20,cancelled,true);
            std::vector<float> values(67*h,.1f);values[50*h]=std::numeric_limits<float>::quiet_NaN();
            auto input=mx::astype(Tensor(values.data(),{1,67,h},mx::float32),mx::bfloat16);
            auto halves=mx::split(wg,2,0);mx::eval(halves);op.plan_block(0,67);op.stage(0,67,{halves[0],halves[1],wd});
            int recomputed=0;
            auto result=op.run(0,input,[&](const Tensor&x){++recomputed;return mx::full(x.shape(),7.f,x.dtype());},cancelled,nullptr,
                [](const Tensor&x,int,int count) {return std::make_pair(mx::zeros(x.shape(),mx::float32),
                    mx::zeros({1,x.shape(1),count},x.dtype()));});
            check(recomputed==1 && mx::all(result==Tensor(7.f,mx::bfloat16)).item<bool>() && op.metrics().runtime_failed,
                "F32 late chunk failure published partial scratch");
        }
        unsetenv("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN");unsetenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC");
        std::cout<<"PASS F32 channel join: base/A/B/base, original BF16 hidden/base-before-LoRA, ONE down correction and full late-failure GPU recomputation\n";
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
