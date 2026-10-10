#include "native_hybrid_ffn.hpp"
#include "native_ane_handoff_state.hpp"
#include "native_step_ops.hpp"
#include "native_weight_scale_cache_contract.hpp"
#include "../private/ane_mil.hpp"

#include <atomic>
#include <limits>
#include <thread>
#include <cmath>
#include <bit>

namespace tc::metal_graph {
namespace {
using Clock = std::chrono::steady_clock;
namespace ag = tc::ane::gpu;
namespace pa = tc::ane::private_api;
std::atomic<uint64_t> next_domain{1};
void require(bool ok, const std::string& diagnostic) {
    if (!ok) throw std::runtime_error(diagnostic);
}
double seconds(Clock::time_point since) { return std::chrono::duration<double>(Clock::now()-since).count(); }
std::chrono::milliseconds remaining(Clock::time_point end) {
    return detail::ane_remaining(end,Clock::now());
}
size_t extent(const MatrixView& m) {
    require(bool(m.storage), "native hybrid empty retained matrix");
    return dense_ffn::validate_layout({m.rows,m.columns,m.row_stride_bytes,
        m.storage.size_bytes(),m.storage.offset_bytes(),m.storage.scalar_type()});
}
void matrix(const MatrixView& m,uint32_t rows,uint32_t columns,ScalarType type) {
    (void)extent(m);
    require(m.rows==rows && m.columns==columns && m.storage.scalar_type()==type,
            "native hybrid logical matrix shape/dtype mismatch");
}
void distinct(const MatrixView& a,const MatrixView& b) {
    if (a.storage.buffer().native_handle()==b.storage.buffer().native_handle() &&
        dense_ffn::ranges_overlap(a.storage.offset_bytes(),extent(a),b.storage.offset_bytes(),extent(b)))
        throw std::invalid_argument("native hybrid writable matrix aliases source/partial");
}
void validate_plan(const NativeHybridFfnPlan& p,const NativeHybridFfnOptions& o) {
    matrix(p.input,o.rows,o.hidden,ScalarType::BF16);
    matrix(p.gate,o.width,o.hidden,ScalarType::BF16);
    matrix(p.up,o.width,o.hidden,ScalarType::BF16);
    matrix(p.down,o.hidden,o.width,ScalarType::BF16);
    matrix(p.gpu_partial,o.rows,o.hidden,ScalarType::FP32);
    matrix(p.ane_partial,o.rows,o.hidden,ScalarType::FP32);
    matrix(p.published_output,o.rows,o.hidden,ScalarType::BF16);
    require(!p.gpu_ffn.empty() && !p.after_join.empty(), "native hybrid empty branch/consumer graph");
    for(const auto* output:{&p.gpu_partial,&p.ane_partial,&p.published_output}) {
        for(const auto* source:{&p.input,&p.gate,&p.up,&p.down}) distinct(*output,*source);
    }
    distinct(p.gpu_partial,p.ane_partial);
    distinct(p.gpu_partial,p.published_output); distinct(p.ane_partial,p.published_output);
    require(o.weight_scale_cache_entries ? p.weight_scale_cache_slot<o.weight_scale_cache_entries :
        p.weight_scale_cache_slot==UINT32_MAX,"native hybrid weight scale cache slot is outside its request plan");
}
WeightScaleCacheKey scale_key(const NativeHybridFfnPlan& p,const NativeHybridFfnOptions& o) {
    WeightScaleCacheKey key;
    key.immutable_source_revision=o.weight_scale_source_revision;
    key.source_generation=o.weight_scale_source_generation;
    key.recipe.hidden=o.hidden;key.recipe.width=o.width;key.recipe.ane_channels=o.ane_channels;
    key.recipe.first_channel=o.width-o.ane_channels;
    key.recipe.gate_norm_bits=std::bit_cast<uint32_t>(1.f/std::sqrt(128.f));
    key.recipe.down_norm_bits=std::bit_cast<uint32_t>(1.f/std::sqrt(512.f));
    key.recipe.staging=uint8_t(o.register_weight_staging);
    const std::array<MatrixView,3> sources{p.gate,p.up,p.down};
    for(size_t i=0;i<sources.size();++i) {
        const auto& s=sources[i];
        key.sources[i]={s.storage.buffer().allocation_identity(),
            reinterpret_cast<uintptr_t>(s.storage.buffer().native_handle()),s.storage.offset_bytes(),
            s.storage.size_bytes(),s.rows,s.columns,s.row_stride_bytes,s.storage.scalar_type()};
    }
    validate_scale_cache_key(key);
    return key;
}
} // namespace

struct NativeHybridFfn::Impl {
    Backend* backend;
    NativeHybridFfnOptions options;
    const std::thread::id owner=std::this_thread::get_id();
    uint64_t domain=next_domain.fetch_add(1), generation=0;
    ag::Device device{false,false}; // Surface allocator only; no legacy submit.
    PreparedAneStages stages;
    NativeFfn gpu;
    NativeStepOps finite_ops;
    Buffer invalid_buffer;
    BufferView invalid;
    std::optional<pa::Program> program;
    std::vector<NamedAneSurface> inputs, outputs;
    AneWeightBankViews bank;
    AneActivationViews activation;
    struct ScaleEntry {
        Buffer compact;
        AneCompactWeightScales slices;
        WeightScaleEntryState state;
        // Strong original Buffer/Data owners, no source content copies. The
        // exact key uses their real allocation control blocks plus full views.
        std::array<MatrixView,3> retained_sources;
    };
    std::vector<ScaleEntry> scale_cache;
    size_t allocated_surfaces=0;
    bool retired=false, running=false, validated=false;
    std::string error;

    struct Job {
        std::unique_ptr<NativeAneHandoff> handoff;
        std::optional<Ticket> gpu, consumer;
        bool gpu_attempted=false, consumer_attempted=false;
        std::optional<AneStageStatus> producer_status, restore_status;
        // Preserve immutable source/partial owners even after producer alone
        // completes. They cannot escape before the last consumer completes.
        std::optional<NativeHybridFfnPlan> retained_plan;
    };
    std::unique_ptr<Job> job;

    Impl(Backend& b,NativeHybridFfnOptions o)
        :backend(&b),options(std::move(o)),
         stages(PreparedAneStages::prepare(b,{options.rows,options.hidden,options.width,options.ane_channels},
             options.register_weight_staging ? AneWeightStaging::SylvesterRegister : AneWeightStaging::Generic,
             options.weight_scale_cache_entries!=0)),
         gpu(NativeFfn::prepare_parallel(b,{options.hidden,options.width,options.width-options.ane_channels})),
         finite_ops(NativeStepOps::prepare(b)),
         invalid_buffer(b.allocate(sizeof(uint32_t),"tc-native-hybrid-finite-status")),
         invalid(invalid_buffer.view(0,sizeof(uint32_t),ScalarType::UInt32)) {
        require(domain && device.native_device()==b.native_device(), "native hybrid Metal device/domain mismatch");
        scale_cache.reserve(options.weight_scale_cache_entries);
        for(uint32_t i=0;i<options.weight_scale_cache_entries;++i) {
            ScaleEntry entry;
            const size_t part=size_t(options.ane_channels)*2;
            entry.compact=b.allocate(compact_weight_scale_bytes(options.hidden,options.ane_channels),
                "tc-native-hybrid-weight-scale-cache",StorageMode::Private);
            entry.slices={entry.compact.view(0,part,ScalarType::FP16),
                entry.compact.view(part,part,ScalarType::FP16),
                entry.compact.view(2*part,size_t(options.hidden)*2,ScalarType::FP16)};
            scale_cache.push_back(std::move(entry));
        }
        const auto add=[&](std::string name,uint32_t rows,uint32_t columns,ag::Element type,bool output=false) {
            ag::Surface surface(device,rows,columns,type,options.surface_allocation,"tc-native-hybrid-"+name);
            require(allocated_surfaces<=SIZE_MAX-surface.bytes(),"native hybrid surface accounting overflow");
            allocated_surfaces+=surface.bytes();
            auto imported=import_ane_surface(b,surface);
            (output?outputs:inputs).emplace_back(std::move(name),std::move(surface));
            return imported;
        };
        bank.g=add("wg",options.ane_channels,options.hidden,ag::Element::I8);
        bank.sg=add("sg",options.ane_channels,1,ag::Element::FP16);
        bank.u=add("wu",options.ane_channels,options.hidden,ag::Element::I8);
        bank.su=add("su",options.ane_channels,1,ag::Element::FP16);
        bank.d=add("wd",options.hidden,options.ane_channels,ag::Element::I8);
        // sd is restored on GPU, not an ANE binding. Keep it in a separate owner.
        ag::Surface sd(device,options.hidden,1,ag::Element::FP16,options.surface_allocation,"tc-native-hybrid-sd");
        allocated_surfaces+=sd.bytes(); bank.sd=import_ane_surface(b,sd);
        activation.x=add("x",options.hidden,options.rows,ag::Element::I8);
        activation.tx=add("tx",1,options.rows,ag::Element::FP16);
        (void)add("y",options.hidden+1,options.rows,ag::Element::FP16,true);
        tc::ane::GraphShape shape{tc::ane::Kind::SwiGLU,int(options.rows),int(options.hidden),int(options.ane_channels),0,0,false};
        const auto spec=pa::w8_swiglu_program(shape,20260930,1.f,tc::ane::W8Basis::SylvesterDH,0,0,false);
        require(spec.packed_rows==int(options.hidden+1),"native hybrid requires exact BaseOnly ANE output ABI");
        program.emplace(device,spec.mil,spec.constants,options.program_cache);
        require(program->inputs().size()==7 && program->outputs()==std::vector<std::string>{"y"},
                "native hybrid unexpected BaseOnly ANE program ABI");
    }
    void on_owner() const { require(std::this_thread::get_id()==owner,"native hybrid controlling calls must use the preparation thread"); }
    NativeHybridFfnStatus collect() const {
        NativeHybridFfnStatus result;
        result.retired=retired; result.running=running; result.generation=generation; result.error=error;
        if (!job) return result;
        result.evidence.handoff=job->handoff->status();
        result.evidence.gpu_attempted=job->gpu_attempted;
        result.evidence.consumer_attempted=job->consumer_attempted;
        if(job->gpu) result.evidence.gpu=job->gpu->status();
        if(job->consumer) result.evidence.consumer=job->consumer->status();
        result.all_work_physically_complete=hybrid_all_physically_complete(result.evidence);
        result.ok=hybrid_publishable(result.evidence,retired,validated);
        return result;
    }
    void retire(std::string reason) {
        retired=true; running=false; validated=false;
        for(auto& entry:scale_cache) entry.state.retire();
        if(error.empty()) error=std::move(reason);
    }
};

void NativeHybridFfn::validate_options(const NativeHybridFfnOptions& o) {
    native_ane_stage_contract::validate_geometry({o.rows,o.hidden,o.width,o.ane_channels});
    require(!o.program_cache.empty() && o.program_cache.is_absolute(),"native hybrid requires an explicit absolute program cache");
    require(bool(o.surface_allocation.reserve)==bool(o.surface_allocation.allocated),"native hybrid surface memory hooks must be paired");
    require(o.weight_scale_cache_entries<=32,"native hybrid weight scale cache exceeds request-local layer capacity");
    require(o.weight_scale_cache_entries ? !o.weight_scale_source_revision.empty() &&
        o.weight_scale_source_generation!=0 : o.weight_scale_source_revision.empty() &&
        o.weight_scale_source_generation==0,"native hybrid weight scale cache requires an exact enabled source epoch");
}
NativeHybridFfn NativeHybridFfn::prepare(Backend& b,NativeHybridFfnOptions o) {
    validate_options(o);
    return NativeHybridFfn(std::make_unique<Impl>(b,std::move(o)));
}
NativeHybridFfn::NativeHybridFfn(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
NativeHybridFfn::NativeHybridFfn(NativeHybridFfn&&) noexcept=default;
NativeHybridFfn& NativeHybridFfn::operator=(NativeHybridFfn&& other) noexcept {
    if(this!=&other) { preserve_pending(); impl_=std::move(other.impl_); }
    return *this;
}
void NativeHybridFfn::preserve_pending() noexcept {
    if(!impl_) return;
    try {
        if(!impl_->collect().all_work_physically_complete) (void)impl_.release();
    } catch(...) { (void)impl_.release(); }
}
NativeHybridFfn::~NativeHybridFfn() noexcept { preserve_pending(); }
const NativeFfn& NativeHybridFfn::gpu_complement() const {
    require(bool(impl_),"native hybrid empty owner"); impl_->on_owner(); return impl_->gpu;
}
size_t NativeHybridFfn::surface_bytes() const { require(bool(impl_),"native hybrid empty owner"); return impl_->allocated_surfaces; }
size_t NativeHybridFfn::weight_scale_cache_payload_bytes() const {
    require(bool(impl_),"native hybrid empty owner");
    return impl_->scale_cache.size()*compact_weight_scale_bytes(impl_->options.hidden,impl_->options.ane_channels);
}
uint32_t NativeHybridFfn::weight_scale_cache_capacity() const {
    require(bool(impl_),"native hybrid empty owner"); return uint32_t(impl_->scale_cache.size());
}
const std::string& NativeHybridFfn::program_key() const { require(bool(impl_),"native hybrid empty owner"); return impl_->program->cache_key(); }
NativeHybridFfnStatus NativeHybridFfn::status() const {
    require(bool(impl_),"native hybrid empty owner"); impl_->on_owner(); return impl_->collect();
}
NativeHybridFfnStatus NativeHybridFfn::drain(std::chrono::milliseconds timeout) {
    require(bool(impl_) && timeout.count()>=0,"native hybrid invalid physical drain"); impl_->on_owner();
    const auto end=detail::ane_deadline(timeout,Clock::now());
    auto result=impl_->collect();
    while(!result.all_work_physically_complete && Clock::now()<end) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); result=impl_->collect();
    }
    return result; // No force signals, no logical resurrection, no slot reuse.
}

NativeHybridFfnResult NativeHybridFfn::execute(NativeHybridFfnPlan plan,std::chrono::milliseconds timeout) {
    require(bool(impl_) && timeout.count()>0,"native hybrid requires a positive execution deadline");
    return execute_until(std::move(plan),detail::ane_deadline(timeout,Clock::now()));
}
NativeHybridFfnResult NativeHybridFfn::execute_until(NativeHybridFfnPlan plan,Deadline end,const std::function<void()>& concurrent_prepare) {
    require(bool(impl_) && Clock::now()<end,"native hybrid requires an unexpired absolute execution deadline");
    auto& p=*impl_; p.on_owner();
    require(!p.retired && !p.running && p.collect().all_work_physically_complete,"native hybrid slot retired/running/physically live");
    require(p.generation!=UINT64_MAX,"native hybrid job generation exhausted");
    // Invalid caller matrices leave the previous generation completely intact.
    validate_plan(plan,p.options);
    std::optional<WeightScaleCacheKey> cache_key;
    Impl::ScaleEntry* scale_entry=nullptr;
    bool cache_hit=false;
    if(p.options.weight_scale_cache_entries) {
        cache_key.emplace(scale_key(plan,p.options));
        scale_entry=&p.scale_cache[plan.weight_scale_cache_slot];
        cache_hit=scale_entry->state.hit(*cache_key);
    }
    NativeHybridFfnResult result;
    const auto started=Clock::now();
    p.running=true; p.validated=false; ++p.generation;
    // Each generation owns a new handoff/status identity. A previous successful
    // consumer has physically completed before the single bank can be reused.
    p.job.reset();
    try {
        auto job=std::make_unique<Impl::Job>();
        const NativeAneJobIdentity identity{p.domain,1,p.generation,p.generation,p.generation};
        job->handoff=std::make_unique<NativeAneHandoff>(*p.backend,*p.program,p.inputs,p.outputs,identity);
        job->producer_status.emplace(AneStageStatus::producer(*p.backend,identity));
        job->retained_plan.emplace(std::move(plan));
        p.job=std::move(job);
        auto& j=*p.job; auto& binding=*j.retained_plan;
        // Record every producer stage before committing anything. Original
        // gate/up offsets and full down row pitch are retained by these views.
        if(scale_entry && !cache_hit) {
            scale_entry->state.begin(std::move(*cache_key),p.generation);
            scale_entry->retained_sources={binding.gate,binding.up,binding.down};
        }
        p.stages.encode_weight_bank(binding.before_ffn,binding.gate,binding.up,binding.down,p.bank,*j.producer_status,
            cache_hit?&scale_entry->slices:nullptr,scale_entry && !cache_hit?&scale_entry->slices:nullptr);
        p.stages.encode_activation(binding.before_ffn,binding.input,p.activation,*j.producer_status);
        NativeStepOps::validate_finite(binding.gpu_partial,p.invalid);
        NativeStepOps::validate_finite(binding.ane_partial,p.invalid);
        NativeStepOps::validate_finite(binding.published_output,p.invalid);
        require(Clock::now()<end,"native hybrid producer recording deadline");
        const auto producer_start=Clock::now();
        j.handoff->submit_producer(std::move(binding.before_ffn));
        auto ready=j.handoff->finish_producer_until(end);
        j.producer_status->validate_after(ready);
        result.producer_seconds=seconds(producer_start);
        result.producer_gpu_span_seconds=j.handoff->status().producer_gpu_span_seconds;
        // ANE launch fence: driver submission returns before the GPU complement
        // is committed. No result/event is treated as an arithmetic certificate.
        const auto ane_start=Clock::now();
        j.handoff->submit_ane_until(std::move(ready),end);
        result.submit_ane_seconds=seconds(ane_start);
        require(Clock::now()<end,"native hybrid GPU branch submission deadline");
        const auto branch_start=Clock::now();
        j.gpu_attempted=true;
        try { j.gpu.emplace(p.backend->submit(std::move(binding.gpu_ffn))); }
        catch(...) { j.gpu_attempted=false; throw; } // Backend throws only before submission.
        // Submitted devices run while the host prepares the next immutable
        // source window. A preparation error is delayed until BOTH branches
        // and the consumer have physically finished; it cannot free owners.
        std::exception_ptr preparation_error;
        try {if(concurrent_prepare)concurrent_prepare();}catch(...){preparation_error=std::current_exception();}
        auto readable=j.handoff->finish_ane_until(end);
        const auto gpu=j.gpu->wait(remaining(end));
        require(gpu.ok && gpu.physically_complete && !gpu.timed_out && Clock::now()<end,
                "native hybrid GPU complement failed/deadline: "+gpu.error);
        result.branch_join_seconds=seconds(branch_start);
        if (std::isfinite(gpu.gpu_seconds) && gpu.gpu_seconds > 0)
            result.gpu_complement_span_seconds=gpu.gpu_seconds;
        const auto consumer_start=Clock::now();
        j.restore_status.emplace(AneStageStatus::restore(*p.backend,readable,*j.producer_status));
        CommandGraph consumer;
        p.stages.encode_restore(consumer,readable,*j.producer_status,binding.ane_partial,*j.restore_status);
        consumer.append_graph(std::move(binding.after_join));
        p.finite_ops.encode_reset_invalid_flag(consumer,p.invalid);
        p.finite_ops.encode_finite(consumer,binding.gpu_partial,p.invalid);
        p.finite_ops.encode_finite(consumer,binding.ane_partial,p.invalid);
        p.finite_ops.encode_finite(consumer,binding.published_output,p.invalid);
        require(Clock::now()<end,"native hybrid consumer recording deadline");
        j.consumer_attempted=true;
        try { j.consumer.emplace(p.backend->submit(std::move(consumer))); }
        catch(...) { j.consumer_attempted=false; throw; }
        const auto consumed=j.consumer->wait(remaining(end));
        require(consumed.ok && consumed.physically_complete && !consumed.timed_out && Clock::now()<end,
                "native hybrid restore/join failed/deadline: "+consumed.error);
        j.restore_status->validate_after(*j.consumer);
        NativeStepOps::require_finite(*j.consumer,p.invalid);
        result.consumer_seconds=seconds(consumer_start);
        if(preparation_error)std::rethrow_exception(preparation_error);
        p.validated=true; p.running=false;
        require(p.collect().ok,"native hybrid aggregate refused publication");
        // Producer flags alone are insufficient for an entry publication.
        // Consumer finite, sealed restore and aggregate physical completion
        // have all succeeded. Any later failure permanently retires the run.
        if(scale_entry) {
            if(!cache_hit) scale_entry->state.publish(p.generation);
            result.weight_scale_cache_hit=cache_hit;
            result.weight_scale_cache_miss=!cache_hit;
        }
        // Ticket graphs have collected actual callbacks; owned matrices no
        // longer need an additional full source-view retention by this slot.
        j.retained_plan.reset();
    } catch(const std::exception& e) { p.retire(e.what()); }
      catch(...) { p.retire("native hybrid unknown execution failure"); }
    result.total_seconds=seconds(started); result.completion=p.collect();
    return result;
}

} // namespace tc::metal_graph
