// Opt-in generated complete native DiT/Euler comparison. No checkpoint, MLX
// tensor/operator, image encoder, VAE or approximation of the Transformer.
#include "../../native/models/qwen21/native_denoiser.hpp"
#include "../../native/models/qwen21/native_memory_plan.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>

namespace mg=tc::metal_graph;
namespace ng=tc::qwen21::native_graph;
namespace {
constexpr uint32_t hidden=128,width=1024,layers=2,target_rows=32,text_rows=7;
constexpr uint64_t seed=2026100817ULL;
constexpr size_t guard=64;
constexpr unsigned char canary=0x6d;
constexpr double maximum_rrms=0.08;
constexpr const char* source_revision="generated-qwen-h128-f1024-l2-bf16-padded-v1";
constexpr ng::DenoiserGeometry geometry{{hidden,1,128,width},layers,64};
using Clock=std::chrono::steady_clock;
void check(bool condition,const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Exception,class Function> void rejects(Function&& function,const char* message) {
    bool rejected=false;
    try { function(); } catch (const Exception&) { rejected=true; }
    check(rejected,message);
}
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
uint16_t narrow(float value) {
    auto bits=std::bit_cast<uint32_t>(value);
    return uint16_t((bits+0x7fffU+((bits>>16)&1U))>>16);
}
float expand(uint16_t value) { return std::bit_cast<float>(uint32_t(value)<<16); }
struct Generator {
    uint64_t state=seed;
    float unit() {
        state^=state>>12; state^=state<<25; state^=state>>27;
        return 2.f*(float(uint32_t((state*2685821657736338717ULL)>>40))*(1.f/16777216.f))-1.f;
    }
};
uint64_t hash(const mg::Buffer& buffer) {
    check(buffer.contents()!=nullptr,"source hashing requires Shared produced backing");
    uint64_t value=14695981039346656037ULL;
    const auto* data=static_cast<const unsigned char*>(buffer.contents());
    for (size_t i=0;i<buffer.size_bytes();++i) { value^=data[i]; value*=1099511628211ULL; }
    return value;
}
struct Source {
    std::string label;
    mg::Buffer buffer;
    mg::MatrixView view;
    uint64_t digest;
};
// These immutable CPU-produced Shared sources exist before each observed
// phase baseline. They do not pretend to have per-array phase ledger claims.
// The exact same views/owners feed both candidates; no MLX/second model copy.
struct Fixture {
    mg::Backend source_backend;
    Generator random;
    std::vector<Source> sources;
    ng::DenoiserWeights weights;
    mg::MatrixView latents,text;

    mg::MatrixView matrix(std::string label,uint32_t rows,uint32_t columns,float amplitude,float center=0.f) {
        const size_t pitch=size_t(columns+16)*2;
        auto buffer=source_backend.allocate(guard+size_t(rows)*pitch+guard,label,mg::StorageMode::Shared);
        std::memset(buffer.contents(),canary,buffer.size_bytes());
        mg::MatrixView value{buffer.view(guard,size_t(rows-1)*pitch+size_t(columns)*2,mg::ScalarType::BF16),
                            rows,columns,pitch};
        for (uint32_t row=0;row<rows;++row) {
            auto* line=reinterpret_cast<uint16_t*>(static_cast<std::byte*>(value.storage.contents())+row*pitch);
            for (uint32_t col=0;col<columns;++col) line[col]=narrow(center+amplitude*random.unit());
        }
        const auto digest=hash(buffer);
        sources.push_back({std::move(label),buffer,value,digest});
        return value;
    }
    ng::Projection projection(const std::string& label,uint32_t out,uint32_t in,float gain=.65f) {
        return {matrix(label,out,in,gain*std::sqrt(3.f/float(in))),{}};
    }
    Fixture() {
        weights.image_input=projection("image-input",hidden,64);
        weights.time_input=projection("time-input",hidden,256);
        weights.time_output=projection("time-output",hidden,hidden);
        weights.text_norm=matrix("text-norm",1,hidden,.125f);
        weights.text_input=projection("text-input",hidden,hidden);
        weights.text_output=projection("text-output",hidden,hidden);
        weights.modulation=projection("modulation",4*hidden,hidden,.35f);
        weights.output_scale=projection("output-scale",hidden,hidden,.5f);
        weights.output=projection("output",64,hidden);
        for (uint32_t layer=0;layer<layers;++layer) {
            const auto prefix="layer-"+std::to_string(layer)+"-";
            ng::BlockWeights block;
            block.query=projection(prefix+"query",hidden,hidden);
            block.key=projection(prefix+"key",hidden,hidden);
            block.value=projection(prefix+"value",hidden,hidden);
            block.attention_output=projection(prefix+"attention-output",hidden,hidden);
            block.query_norm=matrix(prefix+"query-norm",1,128,.125f,1.f);
            block.key_norm=matrix(prefix+"key-norm",1,128,.125f,1.f);
            auto fused=projection(prefix+"fused-gate-up",2*width,hidden).weight;
            block.gate=ng::matrix_rows(fused,0,width);
            block.up=ng::matrix_rows(fused,width,width);
            block.down=projection(prefix+"down",hidden,width).weight;
            weights.blocks.push_back(std::move(block));
        }
        latents=matrix("input-latents",target_rows,64,.35f);
        text=matrix("input-text",text_rows,hidden,.35f);
        verify();
    }
    ng::DenoiserInput input(uint64_t generation) const {
        ng::DenoiserInput result;
        result.latents=latents; result.text=text; result.height=4; result.width=8;
        result.identity={source_revision,"generated-text-and-latents-v1",{},generation};
        result.sigmas={1.f,.7f,0.f};
        return result;
    }
    void verify() const {
        for (const auto& source:sources) {
            check(hash(source.buffer)==source.digest,"read-only source or padding changed: "+source.label);
            const auto* data=static_cast<const unsigned char*>(source.buffer.contents());
            const size_t start=source.view.storage.offset_bytes(),pitch=source.view.row_stride_bytes;
            for (size_t i=0;i<source.buffer.size_bytes();++i) {
                const bool logical=i>=start && (i-start)/pitch<source.view.rows &&
                    (i-start)%pitch<size_t(source.view.columns)*2;
                if (!logical) check(data[i]==canary,"source allocation guard or pitch padding changed");
            }
        }
    }
};

struct Trial {
    std::shared_ptr<Fixture> source;
    std::unique_ptr<ng::NativePhaseMemory> phase;
    std::unique_ptr<mg::Backend> backend;
    std::optional<ng::NativeDenoiser> model;
    std::unique_ptr<ng::DenoiserRun> run;
    void release_native() { run.reset(); model.reset(); backend.reset(); }
};
struct Outcome {
    std::vector<uint16_t> values;
    ng::DenoiserHybridMetrics hybrid;
    tc::MemoryAdmissionMetrics memory;
    double prepare_seconds=0,step_seconds[2]{};
};
void success(const mg::Completion& status,const char* message) {
    check(status.ok && status.physically_complete && !status.timed_out &&
          status.completed_buffers==status.submitted_buffers,message);
}
void zero_ledger(const tc::MemoryAdmissionMetrics& m) {
    check(m.execution_state=="succeeded" && !m.tainted && !m.worker_quarantined &&
          m.ledger_storage_bytes==0 && m.ledger_reserved_bytes==0 && m.ledger_pending_release_bytes==0 &&
          m.ledger_cached_bytes==0 && m.ledger_unknown_bytes==0 && m.ledger_storage_count==0,
          "native phase did not finish with a clean zero ledger");
}
Outcome run_route(const std::shared_ptr<Fixture>& source,bool hybrid,const std::filesystem::path& cache,
                  bool register_weight_staging=false) {
    check(hybrid || !register_weight_staging,"register staging is only valid for the hybrid candidate");
    auto trial=std::make_unique<Trial>(); trial->source=source;
    Outcome outcome;
    try {
        source->verify();
        const auto baseline=tc::observe_process_memory();
        check(baseline.available,"real process baseline unavailable");
        const auto started=Clock::now();
        const uint64_t generation=hybrid?2:1;
        const ng::NativePhaseShape shape{geometry,target_rows,text_rows,{},hybrid};
        trial->phase=std::make_unique<ng::NativePhaseMemory>(shape,source_revision,baseline,
            ng::NativePhaseLimits{},0x5148573146495854ULL,generation);
        const auto planned_backings=trial->phase->plan().backing_count;
        mg::BackendOptions backend_options; backend_options.allocation=trial->phase->hooks();
        trial->backend=std::make_unique<mg::Backend>(std::move(backend_options));
        check(trial->backend->native_device()==source->source_backend.native_device(),"source/candidate Metal device mismatch");
        ng::NativeDenoiserExecutionOptions options;
        options.hybrid_ffn=hybrid; options.ane_program_cache=hybrid?cache:std::filesystem::path{};
        options.allocation=trial->phase->hooks();
        options.register_weight_staging=hybrid && register_weight_staging;
        trial->model.emplace(ng::NativeDenoiser::prepare(*trial->backend,source->weights,source_revision,geometry,std::move(options)));
        const auto before_rejections=trial->phase->context().admission().snapshot();
        rejects<std::invalid_argument>([&] {
            auto bad=source->input(generation); bad.identity.source_revision="foreign-source";
            (void)trial->model->begin(std::move(bad));
        },"foreign source identity was accepted");
        rejects<std::invalid_argument>([&] {
            auto bad=source->input(generation); bad.identity.adapter_revision="unsupported-adapter";
            (void)trial->model->begin(std::move(bad));
        },"BaseOnly silently accepted adapter identity");
        const auto after_rejections=trial->phase->context().admission().snapshot();
        check(before_rejections.storage_count==after_rejections.storage_count &&
              before_rejections.site_allocation_count==after_rejections.site_allocation_count,
              "invalid begin allocated request resources before identity rejection");
        trial->run=trial->model->begin(source->input(generation));
        outcome.prepare_seconds=seconds(started);
        success(trial->run->physical_status(),"unsubmitted run has pending work");
        check(trial->run->physical_status().submitted_buffers==0 && trial->run->completed_steps()==0,
              "begin unexpectedly submitted device work");
        rejects<std::logic_error>([&] { (void)trial->run->final_latents(); },"final latent published before steps");
        rejects<std::invalid_argument>([&] { (void)trial->run->drain(std::chrono::milliseconds(-1)); },"negative drain accepted");
        rejects<std::logic_error>([&] { (void)trial->run->step(std::chrono::milliseconds(0)); },"zero step deadline accepted");
        check(trial->run->state()==ng::DenoiserState::Prepared && trial->run->completed_steps()==0,
              "rejected interface call mutated unsubmitted run");
        success(trial->run->drain(std::chrono::milliseconds(0)),"zero drain changed idle state");
        for (size_t step=0;step<2;++step) {
            trial->phase->context().checkpoint("generated_before_step");
            const auto start=Clock::now();
            const auto complete=trial->run->step(std::chrono::seconds(30));
            outcome.step_seconds[step]=seconds(start);
            success(complete,"complete DiT step physical domain failed");
            success(trial->run->physical_status(),"step physical status failed");
            success(trial->run->drain(std::chrono::milliseconds(0)),"observational drain changed completed step");
            check(trial->run->completed_steps()==step+1,"step transaction count mismatch");
            const auto metrics=trial->run->hybrid_metrics();
            check(metrics.enabled==hybrid && metrics.successful_ffns==(hybrid&&step?2U:0U) &&
                  metrics.completed_hybrid_steps==(hybrid&&step?1U:0U),"prefill/steady hybrid receipt count mismatch");
            check(metrics.weight_staging==(register_weight_staging?"register":"generic"),
                  "actual weight-staging receipt differs from the explicitly requested candidate");
            check(metrics.device_timed_ffns<=metrics.successful_ffns,
                  "device timing coverage exceeds successful hybrid FFNs");
            check(std::isfinite(metrics.producer_gpu_span_seconds) && metrics.producer_gpu_span_seconds>=0 &&
                  std::isfinite(metrics.gpu_complement_span_seconds) && metrics.gpu_complement_span_seconds>=0,
                  "device GPU spans are nonfinite or negative");
            if (!metrics.device_timed_ffns)
                check(metrics.producer_gpu_span_seconds==0 && metrics.gpu_complement_span_seconds==0,
                      "GPU span totals have no covered successful FFN");
            if (hybrid && step) check(complete.gpu_seconds==0,"hybrid invented whole-step GPU timing");
            if (!step) rejects<std::logic_error>([&] { (void)trial->run->final_latents(); },"partial schedule published latent");
            source->verify();
            trial->phase->context().checkpoint("generated_after_step");
        }
        check(trial->run->state()==ng::DenoiserState::Complete,"complete schedule not published");
        outcome.hybrid=trial->run->hybrid_metrics();
        if (hybrid) check(outcome.hybrid.partial_bytes==2*size_t(target_rows)*hidden*4 && outcome.hybrid.surface_bytes>0,
                          "hybrid partial/surface payload receipt missing");
        {
            const auto result=trial->run->final_latents();
            check(result.rows==target_rows && result.columns==64 && result.storage.contents() &&
                  result.storage.scalar_type()==mg::ScalarType::BF16,"final latent shape/readiness contract");
            for (uint32_t row=0;row<target_rows;++row) {
                const auto* data=reinterpret_cast<const uint16_t*>(static_cast<const std::byte*>(result.storage.contents())+row*result.row_stride_bytes);
                for (uint32_t col=0;col<64;++col) {
                    check(std::isfinite(expand(data[col])),"nonfinite completed latent");
                    outcome.values.push_back(data[col]);
                }
            }
        }
        rejects<std::logic_error>([&] { (void)trial->run->step(); },"Complete run accepted another device step");
        success(trial->run->drain(std::chrono::milliseconds(1)),"completed drain failed");
        check(trial->run->completed_steps()==2 && trial->run->state()==ng::DenoiserState::Complete,
              "post-completion rejection changed committed result");
        const auto live=trial->phase->context().admission().snapshot();
        check(live.reserved_bytes==0 && live.unknown_bytes==0 && live.storage_count==planned_backings,
              "live native/Surface backings differ from the authoritative phase plan");
        source->verify();
        trial->release_native();
        trial->phase->finish_after_release(true);
        outcome.memory=trial->phase->context().take_report().metrics;
        zero_ledger(outcome.memory);
        std::cout<<std::setprecision(12)<<"route="<<(hybrid?"native-hybrid":"native-gpu")
            <<" steps=2 hybrid_ffns="<<outcome.hybrid.successful_ffns
            <<" hybrid_steps="<<outcome.hybrid.completed_hybrid_steps
            <<" prepare_seconds="<<outcome.prepare_seconds
            <<" prefill_seconds="<<outcome.step_seconds[0]<<" steady_seconds="<<outcome.step_seconds[1]
            <<" producer_seconds="<<outcome.hybrid.producer_seconds
            <<" submit_ane_seconds="<<outcome.hybrid.submit_ane_seconds
            <<" branch_join_seconds="<<outcome.hybrid.branch_join_seconds
            <<" consumer_seconds="<<outcome.hybrid.consumer_seconds
            <<" hybrid_total_seconds="<<outcome.hybrid.total_seconds
            <<" weight_staging="<<outcome.hybrid.weight_staging
            <<" producer_gpu_span_seconds="<<outcome.hybrid.producer_gpu_span_seconds
            <<" gpu_complement_span_seconds="<<outcome.hybrid.gpu_complement_span_seconds
            <<" device_timed_ffns="<<outcome.hybrid.device_timed_ffns
            <<" device_timing_eligible_ffns="<<outcome.hybrid.successful_ffns
            <<" device_timing_coverage="<<(outcome.hybrid.successful_ffns?
                double(outcome.hybrid.device_timed_ffns)/double(outcome.hybrid.successful_ffns):0.)
            <<" gpu_span_scope=covered_successful_ffn_ticket_intervals_including_gaps"
            <<" gpu_spans_additive=0 ane_device_duration_available=0"
            <<" surface_bytes="<<outcome.hybrid.surface_bytes<<" partial_bytes="<<outcome.hybrid.partial_bytes
            <<" phase_planned_backings="<<planned_backings<<" actual_site_allocations="<<outcome.memory.ledger_site_allocation_count
            <<" baseline_bytes="<<baseline.process_footprint_bytes
            <<" phase_peak_committed_bytes="<<outcome.memory.ledger_peak_committed_bytes
            <<" source_and_padding=unchanged physical_evidence=denoiser_aggregate ledger_released=1\n";
        return outcome;
    } catch (...) {
        // Never destroy source/phase owners while any producer/ANE/GPU/restore
        // may access them. No timeout injection, retry or second device run.
        bool physical=false;
        try { physical=!trial->run || trial->run->physical_status().physically_complete; } catch (...) {}
        if (physical) trial->release_native();
        auto disposition=tc::MemoryFailureDisposition::None;
        if (trial->phase) disposition=trial->phase->fail_after_drain(physical,"generated complete denoiser fixture failed");
        if (!physical || (disposition!=tc::MemoryFailureDisposition::None && disposition!=tc::MemoryFailureDisposition::Clean))
            (void)trial.release();
        throw;
    }
}
void compare(const Outcome& gpu,const Outcome& hybrid) {
    check(gpu.values.size()==size_t(target_rows)*64 && hybrid.values.size()==gpu.values.size(),"candidate output size mismatch");
    double error=0,reference=0,actual=0,dot=0,maximum=0,peak=0;
    size_t changed=0;
    for (size_t i=0;i<gpu.values.size();++i) {
        const double a=expand(gpu.values[i]),b=expand(hybrid.values[i]);
        check(std::isfinite(a) && std::isfinite(b),"comparison has nonfinite latent");
        error+=(a-b)*(a-b); reference+=a*a; actual+=b*b; dot+=a*b;
        maximum=std::max(maximum,std::abs(a-b)); peak=std::max(peak,std::abs(a));
        changed+=gpu.values[i]!=hybrid.values[i];
    }
    check(reference>0 && actual>0 && peak>0,"degenerate generated denoiser output");
    const double rrms=std::sqrt(error/reference),relative=maximum/peak;
    std::cout<<std::setprecision(12)<<"rrms="<<rrms<<" max_abs="<<maximum<<" reference_peak="<<peak
        <<" max_over_reference_peak="<<relative<<" cosine="<<dot/std::sqrt(reference*actual)
        <<" finite="<<gpu.values.size()<<" changed_bf16="<<changed
        <<" screening_rrms_limit="<<maximum_rrms<<" max_over_reference_peak_role=diagnostic_only\n";
    // The user-approved 5–8% allowance is a tensor RRMS screening reference,
    // not a per-element peak-normalized bound or a full-model quality grant.
    // Finite, source/padding, publication and physical-completion checks above
    // remain mandatory regardless of this numeric screening result.
    check(rrms<=maximum_rrms,"generated full denoiser exceeds fixed 8% RRMS screening limit");
}
} // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=3 || (std::string_view(argv[1])!="--tiny-device" &&
                       std::string_view(argv[1])!="--tiny-register-device")) {
            std::cerr<<"usage: native-qwen-hybrid-denoiser-component "
                <<"{--tiny-device|--tiny-register-device} <fresh-owned-absolute-ANE-cache>\n";
            return 2;
        }
        const bool register_weight_staging=std::string_view(argv[1])=="--tiny-register-device";
        const std::filesystem::path cache(argv[2]);
        check(cache.is_absolute() && cache!=cache.root_path() && !std::filesystem::exists(cache),
              "ANE cache must be an unused independently owned absolute path");
        auto source=std::make_shared<Fixture>();
        const auto gpu=run_route(source,false,{});
        const auto hybrid=run_route(source,true,cache,register_weight_staging);
        compare(gpu,hybrid); source->verify();
        std::cout<<"native_qwen_hybrid_denoiser_component=PASS seed="<<seed
            <<" gpu_runs=1 hybrid_runs=1 total_steps=4 immutable_backings="<<source->sources.size()
            <<" requested_weight_staging="<<(register_weight_staging?"register":"generic")
            <<" actual_weight_staging="<<hybrid.hybrid.weight_staging
            <<" H=128 F=1024 L=2 target_rows=32 text_rows=7 models_loaded=0 refs=0 adapters=0"
            <<" oracle=native_gpu performance_qualified=0 full_model_quality_qualified=0"
            <<" quality_scope=generated_complete_denoiser_only\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"native_qwen_hybrid_denoiser_component=FAIL "<<error.what()<<'\n'; return 1;
    }
}
