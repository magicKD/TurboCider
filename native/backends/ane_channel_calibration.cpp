#include "ane_ffn.hpp"
#include "ane_backend.hpp"
#include "ane_calibration_timing.hpp"
#include "../platform/apple/platform.hpp"
#include "../runtime/build_identity.hpp"
#include <sys/sysctl.h>
#include <unistd.h>
#include <map>
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
#include "private/ane_calibration.hpp"
#include "private/ane_mil.hpp"
#endif

namespace tc::ane {
namespace {
std::string os_build() {
    char value[128]{};
    size_t length = sizeof(value);
    return sysctlbyname("kern.osversion", value, &length, nullptr, 0) == 0 ? value : "";
}
bool calibration_prefetch() {
    const char *raw = std::getenv("TURBOCIDER_PRIVATE_ANE_PREFETCH");
    require(!raw || std::string(raw) == "0" || std::string(raw) == "1", "private ANE prefetch requires 0 or 1");
    return raw && std::string(raw) == "1";
}
#ifdef TURBOCIDER_ENABLE_PRIVATE_ANE
DeviceMatrixView matrix(const Tensor &value, int rows, int columns) {
    require(value.flags().row_contiguous && value.buffer().ptr() && value.offset() >= 0 &&
        value.size() == size_t(rows)*columns && (value.dtype() == mx::bfloat16 || value.dtype() == mx::float16 || value.dtype()==mx::float32),
        "calibration matrix requires produced contiguous BF16/FP16/FP32 storage");
    return {const_cast<void *>(value.buffer().ptr()), value.buffer_size(), size_t(value.offset()), rows, columns,
        size_t(columns)*value.itemsize(), value.dtype() == mx::bfloat16 ? DType::BF16 : value.dtype()==mx::float16?DType::FP16:DType::FP32,
        std::make_shared<Tensor>(value)};
}
struct Bank {
    private_api::Surface g, sg, u, su, d, sd;
    Bank(private_api::Device &device, int h, int f)
        : g(device,f,h,private_api::Element::I8), sg(device,f,1,private_api::Element::FP16),
          u(device,f,h,private_api::Element::I8), su(device,f,1,private_api::Element::FP16),
          d(device,h,f,private_api::Element::I8), sd(device,h,1,private_api::Element::FP16) {}
    uint64_t bytes() const { return g.bytes()+sg.bytes()+u.bytes()+su.bytes()+d.bytes()+sd.bytes(); }
};
void numerical_trial(const Tensor &candidate, const Tensor &reference, ChannelTrialEvidence &trial) {
    require(candidate.shape() == reference.shape(), "calibration trial output shape mismatch");
    auto a = mx::astype(candidate,mx::float32), b = mx::astype(reference,mx::float32);
    auto finite = mx::all(mx::isfinite(a)) & mx::all(mx::isfinite(b));
    auto delta = mx::sum(mx::square(a-b)), aa = mx::sum(mx::square(a));
    auto bb = mx::sum(mx::square(b)), dot = mx::sum(a*b);
    mx::eval({finite,delta,aa,bb,dot});
    require(trial.observe_quality(delta.item<float>(),bb.item<float>(),aa.item<float>(),dot.item<float>(),finite.item<bool>()),
        "calibration trial rejects nonfinite values/statistics or zero reference/output energy");
}
#endif
} // namespace

ChannelSelection HybridFfn::calibrate_channels(const std::filesystem::path &manifest, int h, int width,
    size_t budget, std::atomic<bool> &cancelled, bool lora, const CalibrationWorkload &workload) {
    checkpoint(cancelled);
    const auto policy = configured_backend();
    const bool fp32_partial=configured_fp32_channel_join();
    require(policy.allow_private && (policy.preferred == BackendPreference::Private || policy.preferred == BackendPreference::Auto),
        "automatic ANE channels require an explicitly authorized Private backend");
    const std::string path = std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH") ?
        std::getenv("TURBOCIDER_PRIVATE_ANE_DATA_PATH") : "fp16";
    require(path == "w8a8", "native channel calibration currently requires the Sylvester W8A8 recipe");
    require(!std::getenv("TURBOCIDER_PRIVATE_ANE_GPU_IO") || std::string(std::getenv("TURBOCIDER_PRIVATE_ANE_GPU_IO")) == "1",
        "native channel calibration requires Private GPU I/O");
    auto shape = runtime_template_shape(manifest);
    require(shape.kind == Kind::SwiGLU && shape.hidden == h && shape.width == width &&
        workload.layers >= 5 && workload.rows > 0 && workload.rows <= 4224 && workload.weights && workload.gpu &&
        workload.channel_gpu && (workload.dtype == mx::bfloat16 || workload.dtype == mx::float16),
        "invalid model-supplied channel calibration workload/geometry");
    auto report = std::make_shared<ChannelCalibrationReport>();
    report->actual_rows=workload.rows;report->hidden=h;report->width=width;
    report->bucket_rows=shape.rows;report->layer_count=workload.layers;
    report->lora=lora;
    report->gpu_retention_layers=lora?1:4;
    auto select=[&](int channels,bool passed,const std::string &status,const std::string &reason) {
        report->selected_channels=channels;report->trial_passed=passed;
        report->status=status;report->reason=reason;
        return ChannelSelection{channels,false,passed,reason,std::make_shared<const ChannelCalibrationReport>(*report)};
    };
    if (lora && (!shape.lora_inputs || workload.adapter_identity.empty() || !workload.adapter))
        return select(0,false,"unsupported","adapter calibration requires an identity, activation-input graph and complete family callbacks; optimized GPU-only");
    require(lora || workload.adapter_identity.empty(),"base calibration cannot carry an unbound adapter identity");
    if (workload.rows > shape.rows)
        return select(0,false,"unsupported","automatic calibration requires a bucket covering the complete FFN rows; optimized GPU-only");
    shape.lora_inputs = lora;
    require(!fp32_partial || workload.dtype==mx::bfloat16,"F32 partial calibration requires original BF16 activations");
    const bool prefetch = calibration_prefetch();
    ChannelCalibrationIdentity identity{workload.model_sha256, workload.adapter_identity, workload.encoding,
        workload.dtype == mx::bfloat16 ? "bf16" : "fp16", "private_ane",
        std::string(lora?"sylvester-dh-b128-b512-rne-norm-f16-v2-lora":"sylvester-dh-b128-b512-rne-norm-f16-v2-base")+
            (fp32_partial?"+fp32-partial-join-v1":""),
        device_info().gpu, os_build(), runtime_build_identity(), executor_configuration_identity()+workload.gpu_configuration,
        std::string(lora?"prepared-channel-streamed-gpu-v2-b":"prepared-channel-base-v1-b")+
            std::to_string(shape.rows)+"-l"+std::to_string(workload.layers)+(fp32_partial?"-fp32-partial-join-v1":""), workload.source_generation,
        workload.rows, h, width, shape.tile_k, shape.tile_n, prefetch};
    require(identity.valid() && !workload.source_owners.empty() &&
        std::none_of(workload.source_owners.begin(),workload.source_owners.end(),[](const auto &p){return p.expired();}),
        "native channel calibration requires content/runtime/geometry identities and live source generations");
    report->identity=identity;
#ifndef TURBOCIDER_ENABLE_PRIVATE_ANE
    return select(0,false,"unsupported","Private backend omitted from this build; optimized GPU-only");
#else
    static ChannelSelectionCache cache;
    if (auto saved = cache.find(identity)) return *saved; // constructor rechecks actual graph admission
    try {
        const std::array<int,5> depth{0,(workload.layers-1)/4,(workload.layers-1)/2,
            3*(workload.layers-1)/4,workload.layers-1};
        report->sampled_depths.assign(depth.begin(),depth.end());
        std::array<std::vector<FfnWeight>,5> sources;
        std::array<std::array<DeviceWeightView,3>,5> views;
        std::array<std::optional<Adapter>,5> adapters;
        for (size_t layer=0;layer<depth.size();++layer) {
            sources[layer]=workload.weights(depth[layer]);
            require(sources[layer].size()==3,"calibration requires complete gate/up/down sources");
            std::vector<Tensor> ready;
            for(const auto &w:sources[layer]) {
                require(w.transform==FfnWeight::Transform::None,"Sylvester calibration cannot consume a ConvRot source recipe");
                ready.push_back(w.values);if(w.scales)ready.push_back(*w.scales);if(w.offsets)ready.push_back(*w.offsets);
            }
            mx::eval(ready);
            for(int i=0;i<3;++i)views[layer][i]=calibration_source(sources[layer][i]);
            if(lora) {
                adapters[layer]=workload.adapter(depth[layer]);
                require(adapters[layer]->gate_up && adapters[layer]->gate_up_channels && adapters[layer]->down_and_add,
                    "LoRA calibration requires full/range gate-up and ONE full-hidden down callbacks");
            }
        }
        const uint64_t gpu_upper=uint64_t(workload.rows)*(8ull*h+6ull*width)*2+(128ull<<20);
        if(!admit_memory(observe_runtime_memory(mx::get_active_memory()),{4ull<<30,budget},0,gpu_upper).allowed())
            throw MemoryBudgetError("native calibration full GPU scratch admission denied");
        auto input=mx::astype(mx::random::normal({1,workload.rows,h},mx::float32,mx::random::key(17))*.25f,workload.dtype);
        auto padded=mx::contiguous(workload.rows==shape.rows?input:mx::concatenate(
            {input,mx::zeros({1,shape.rows-workload.rows,h},workload.dtype)},1));
        mx::eval({input,padded});
        auto pad_correction=[&](const Tensor &value,int columns) {
            require(value.shape()==mx::Shape({1,workload.rows,columns}) && value.dtype()==workload.dtype,
                "calibration adapter correction shape/precision mismatch");
            return mx::contiguous(workload.rows==shape.rows?value:mx::concatenate(
                {value,mx::zeros({1,shape.rows-workload.rows,columns},workload.dtype)},1));
        };
        std::vector<Tensor> full;
        const auto baseline=measure_full_gpu_calibration([&]{full.clear();},[&](int count) {
            for(int layer=0;layer<count;++layer)full.push_back(workload.gpu(depth[layer],input));
            mx::async_eval(full);
        },[&]{mx::eval(full);checkpoint(cancelled);});
        report->baseline=baseline;
        full.clear();mx::synchronize();
        const auto sampling_observation=observe_runtime_memory(mx::get_active_memory());
        std::map<int,bool> admitted_points;
        const auto sampling=plan_channel_sampling(width,[&](int fa) {
            if(const auto prior=admitted_points.find(fa);prior!=admitted_points.end())return prior->second;
            const auto memory=plan_native_channel_calibration_memory(workload.rows,shape.rows,h,width,fa,lora,lora,getpagesize(),fp32_partial);
            if(!memory)return false;
            const auto decision=admit_memory(sampling_observation,{4ull<<30,budget},0,memory->estimated_bytes);
            report->memory_admissions.push_back({fa,memory->surface_bytes,memory->gpu_scratch_upper_bytes,
                memory->gpu_restore_bytes,memory->input_bytes,memory->internal_allowance_bytes,
                memory->estimated_bytes,budget,decision.headroom_bytes,decision.allowed(),
                memory_denial_reason(decision.denial,sampling_observation)});
            admitted_points.emplace(fa,decision.allowed());return decision.allowed();
        });
        if(!sampling)return select(0,false,"rejected","complete calibration arenas cannot admit two independent channel points; optimized GPU-only");
        report->memory_limited_points=sampling->memory_limited;
        const auto &shares=sampling->channels;
        std::array<CalibrationPoint,2> points;
        using namespace private_api;
        for(size_t sampled=0;sampled<shares.size();++sampled) {
            checkpoint(cancelled);
            const int fa=shares[sampled],fg=width-fa;
            auto candidate_shape=shape;candidate_shape.width=fa;
            const auto spec=w8_swiglu_program(candidate_shape,20260930,1.f);
            const auto memory=plan_native_channel_calibration_memory(workload.rows,shape.rows,h,width,fa,lora,lora,getpagesize(),fp32_partial);
            require(bool(memory),"native calibration complete memory plan invalid");
            const auto observed=observe_runtime_memory(mx::get_active_memory());
            const auto admission=admit_memory(observed,{4ull<<30,budget},0,memory->estimated_bytes);
            if(!admission.allowed())throw MemoryBudgetError("native calibration complete arena admission denied ("+
                memory_denial_reason(admission.denial,observed)+"; estimated_bytes="+std::to_string(memory->estimated_bytes)+")");
            const uint64_t head_upper=memory->gpu_scratch_upper_bytes;
            Device device;Program program(device,spec.mil,spec.constants,default_cache_directory());
            std::vector<Bank> banks;std::vector<Surface> outputs;
            std::vector<Surface> frozen_dg,frozen_du;
            for(int layer=0;layer<4;++layer) {
                banks.emplace_back(device,h,fa);auto &b=banks.back();
                std::array<std::optional<QuantStage>,3> stages;
                std::exception_ptr error;
                try {
                    stages[0]=device.stage_w8(views[layer][0],{fg,fa,0,h,128},b.g,b.sg);
                    stages[1]=device.stage_w8(views[layer][1],{fg,fa,0,h,128},b.u,b.su);
                    stages[2]=device.stage_w8(views[layer][2],{0,h,fg,fa,512},b.d,b.sd);
                } catch(...) {error=std::current_exception();}
                bool ok=true;
                for(auto &stage:stages)if(stage)try{ok=stage->finish().ok&&ok;}catch(...){if(!error)error=std::current_exception();}
                if(error)std::rethrow_exception(error);
                if(!ok)throw CapabilityError("native calibration W8 staging failed");
                outputs.emplace_back(device,spec.packed_rows,shape.rows,Element::FP16);
                if(lora) {
                    auto deltas=adapters[layer]->gate_up_channels(input,fg,fa);
                    std::pair<Tensor,Tensor> c{pad_correction(deltas.first,fa),pad_correction(deltas.second,fa)};
                    mx::eval({c.first,c.second});
                    frozen_dg.emplace_back(device,fa,shape.rows,Element::FP16);
                    frozen_du.emplace_back(device,fa,shape.rows,Element::FP16);
                    auto upload=device.prepare_gpu_transfer({{matrix(c.first,shape.rows,fa),frozen_dg.back(),0},
                        {matrix(c.second,shape.rows,fa),frozen_du.back(),0}},
                        {{frozen_dg.back(),std::nullopt,0,DType::FP16,1.f},
                         {frozen_du.back(),std::nullopt,0,DType::FP16,1.f}});
                    upload.submit();
                    if(!upload.finish().ok || upload.validation_flags())throw CapabilityError("LoRA calibration immutable correction upload failed");
                }
            }
            Surface x(device,h,shape.rows,Element::I8),tx(device,1,shape.rows,Element::FP16);
            DeviceWeightView activation{const_cast<void *>(padded.buffer().ptr()),padded.buffer_size(),size_t(padded.offset()),
                size_t(h)*2,shape.rows,h,DeviceWeightEncoding::Dense,
                workload.dtype==mx::bfloat16?DType::BF16:DType::FP16,32,{}, {},std::make_shared<Tensor>(padded)};
            auto packed=device.stage_w8(activation,{0,shape.rows,0,h,128,20260930,true},x,tx);
            if(!packed.finish().ok)throw CapabilityError("native calibration A8 staging failed");
            auto bindings=[&](int count) {
                std::vector<CalibrationBindings> result;
                for(int layer=0;layer<count;++layer) {
                    auto &b=banks[layer];
                    CalibrationBindings row{{{"x",x},{"tx",tx},{"wg",b.g},{"sg",b.sg},
                        {"wu",b.u},{"su",b.su},{"wd",b.d}},{{"y",outputs[layer]}}};
                    if(lora){row.inputs.emplace_back("dg",frozen_dg[layer]);row.inputs.emplace_back("du",frozen_du[layer]);}
                    result.push_back(std::move(row));
                }
                return result;
            };
            uint64_t timeline=0;
            CalibrationBatch freeze(device,program,bindings(4),timeline);
            if(!freeze.measure(true).ok)throw CapabilityError("native calibration initial snapshots failed");
            std::vector<Tensor> tails,tail_hidden;
            std::vector<W8GpuCalibrationLayer> traffic;
            uint64_t resident=x.bytes()+tx.bytes()+input.nbytes()+padded.nbytes()+head_upper;
            for(const auto &b:banks)resident+=b.bytes();for(const auto &y:outputs)resident+=y.bytes();
            if(lora) {
                for(const auto &s:frozen_dg)resident+=s.bytes();for(const auto &s:frozen_du)resident+=s.bytes();
            }
            for(int layer=0;layer<5;++layer) {
                W8GpuCalibrationLayer item;
                item.weights={{{views[layer][0],{fg,fa,0,h,128}}, {views[layer][1],{fg,fa,0,h,128}},
                    {views[layer][2],{0,h,fg,fa,512}}}};
                if(layer<4) {
                    tails.push_back(lora && layer?tails.front():mx::zeros({1,shape.rows,h},fp32_partial?mx::float32:workload.dtype));
                    mx::eval(tails.back());if(!lora || !layer)resident+=tails.back().nbytes();
                    item.activation=matrix(padded,shape.rows,h);
                    item.restoration.push_back({outputs[layer].slice_rows(0,h),matrix(tails.back(),shape.rows,h),0,
                        fp32_partial?DType::FP32:workload.dtype==mx::bfloat16?DType::BF16:DType::FP16,1.f,banks[layer].sd,outputs[layer].slice_rows(h,1)});
                    if(lora) {
                        tail_hidden.push_back(layer?tail_hidden.front():mx::zeros({1,shape.rows,fa},workload.dtype));
                        mx::eval(tail_hidden.back());if(!layer)resident+=tail_hidden.back().nbytes();
                        item.restoration.push_back({outputs[layer].slice_rows(h+1,fa),matrix(tail_hidden.back(),shape.rows,fa),
                            0,workload.dtype==mx::bfloat16?DType::BF16:DType::FP16,1.f});
                        item.correction_producer=[&,layer]() {
                            auto delta=adapters[layer]->gate_up_channels(input,fg,fa);
                            auto g=pad_correction(delta.first,fa),u=pad_correction(delta.second,fa);mx::eval({g,u});
                            return std::make_pair(matrix(g,shape.rows,fa),matrix(u,shape.rows,fa));
                        };
                        item.correction_drain=[] {mx::synchronize();};
                    }
                }
                traffic.push_back(std::move(item));
            }
            W8GpuCalibrationWork work(device,candidate_shape,std::move(traffic),{4ull<<30,budget},resident,mx::get_active_memory());
            std::vector<Tensor> heads,head_hidden,joined;
            auto series=measure_w8_channel_point(device,program,{bindings(1),bindings(4)},work,timeline,
                double(fa)/width,prefetch,[&]{heads.clear();head_hidden.clear();joined.clear();checkpoint(cancelled);},[&](int layer,int first) {
                    if(lora){heads.clear();head_hidden.clear();joined.clear();}
                    auto result=workload.channel_gpu(depth[layer],input,0,first);heads.push_back(result.first);
                    if(lora)head_hidden.push_back(result.second);mx::async_eval(heads.back());
                },[&]{mx::eval(heads);if(lora)mx::eval(head_hidden);},[&](int layer) {
                    const int current=lora?0:layer;
                    auto merged=mx::astype(mx::astype(heads[current],mx::float32)+
                        mx::astype(slice_axis(tails[layer],1,0,workload.rows),mx::float32),workload.dtype);
                    if(lora)merged=adapters[layer]->down_and_add(mx::concatenate({head_hidden[current],
                        slice_axis(tail_hidden[layer],1,0,workload.rows)},-1),merged);
                    joined.push_back(merged);mx::async_eval(joined.back());
                },[&]{mx::eval(joined);checkpoint(cancelled);},2,7,lora);
            points[sampled]=series.point;
            report->points.push_back(std::move(series));
        }
        report->complete=true;
        const auto fit=ChannelCostModel::fit(points[0],points[1]);
        if(!fit)return select(0,false,"rejected","independent complete-traffic evidence rejected by bandwidth fit");
        auto proposal=fit->select(width,512,baseline.layer_seconds,[&](int fa) {
            const auto plan=plan_gpu_calibration_memory(shape.rows,h,fa,lora,{},getpagesize());
            const uint64_t head=uint64_t(workload.rows)*(8ull*h+6ull*(width-fa))*2+(128ull<<20)+
                (lora?uint64_t(shape.rows)*(4ull*fa+4ull*width)*2:0)+(fp32_partial?uint64_t(workload.rows)*h*4:0);
            return plan && admit_memory(observe_runtime_memory(mx::get_active_memory()),{4ull<<30,budget},0,plan->estimated_bytes+head).allowed();
        });
        report->proposed_channels=proposal.ane_channels;report->predicted_layer_seconds=proposal.predicted_seconds;
        if(!proposal.ane_channels)return select(0,false,"gpu_only",proposal.reason);
        // Fresh actual inference executor, not the independent measurement
        // adapter. It must stage/launch/restore/join through run_channels().
        HybridFfn candidate(manifest,h,width,budget,cancelled,lora,nullptr,proposal.ane_channels);
        if(!candidate.available())return select(0,false,"rejected","proposed graph failed actual admission/self-test");
        candidate.scheduler_=std::make_unique<RowScheduler>(shape.rows,1,PartitionAxis::IntermediateChannels);
        candidate.fixed_async_=true;candidate.profile_=false;candidate.defer_channel_join_=false;
        candidate.begin_request(workload.adapter_identity);
        candidate.scheduler_=std::make_unique<RowScheduler>(shape.rows,1,PartitionAxis::IntermediateChannels);
        auto run_candidate=[&](int index) {
            candidate.plan_block(index,workload.rows);
            candidate.stage_weights(index,workload.rows,sources[index]);
            return candidate.run(index,input,[&](const Tensor &v){return workload.gpu(depth[index],v);},cancelled,
                lora?&*adapters[index]:nullptr,
                [&](const Tensor &v,int first,int count){return workload.channel_gpu(depth[index],v,first,count);},
                [&](int next){return next<5?sources[next]:std::vector<FfnWeight>{};});
        };
        auto &trial=report->trial.emplace();trial.relative_l2=0;trial.cosine=1;
        for(int layer=0;layer<4;++layer) {
            auto reference=workload.gpu(depth[layer],input),actual=run_candidate(layer);
            numerical_trial(actual,reference,trial);
        }
        const auto before=candidate.metrics();
        for(int sweep=0;sweep<9;++sweep)for(int position=0;position<2;++position) {
            const bool hybrid=(sweep+position)%2;
            std::vector<Tensor> results;
            mx::synchronize();checkpoint(cancelled);
            const auto start=Clock::now();
            try {
                for(int layer=0;layer<4;++layer)results.push_back(hybrid?run_candidate(layer):workload.gpu(depth[layer],input));
                mx::eval(results);candidate.drain();
            } catch(...) {
                const auto first=std::current_exception();try{candidate.drain();}catch(...){}
                try{mx::eval(results);}catch(...){}std::rethrow_exception(first);
            }
            const double span=std::chrono::duration<double>(Clock::now()-start).count();
            if(sweep>=2)(hybrid?trial.candidate_seconds:trial.gpu_seconds).push_back(span);
        }
        const auto after=candidate.metrics();
        trial.calls=after.runtime_calls-before.runtime_calls;
        trial.fallbacks=after.runtime_weight_fallback_blocks;
        trial.retries=after.runtime_weight_overflow_retries;
        trial.completed=!after.runtime_failed && trial.calls==9*4;
        if(!trial.accepts())return select(0,false,"rejected","actual complete-runtime window or independent FFN numerical trial rejected candidate");
        auto selected=select(proposal.ane_channels,true,"accepted","accepted complete-runtime FFN candidate; E2E qualification still required");
        cache.admit(identity,selected,trial,workload.source_owners);
        return selected;
    } catch(const Cancelled &) { throw; }
      catch(const std::exception &error) { return select(0,false,"rejected",std::string("calibration rejected; optimized GPU-only: ")+error.what()); }
#endif
}
} // namespace tc::ane
