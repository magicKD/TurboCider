#include "runtime_private_qkv.hpp"
// Model coordination is pure C++; Apple API ownership remains in the backend.
#include "../../backends/ane_memory.hpp"
#include <mlx/allocator.h>
#include <algorithm>
#include <cmath>
#include <exception>
#if TURBOCIDER_ENABLE_PRIVATE_ANE
#include "../../backends/private/ane_program.hpp"
#include "../../backends/private/ane_mil.hpp"
#endif

namespace tc::qwen21 {
namespace {
double span(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
constexpr int H=4096,N=12288,B=512,M=1024;
}
#if TURBOCIDER_ENABLE_PRIVATE_ANE
namespace pa=ane::private_api;
namespace {
ane::DeviceMatrixView matrix(const Tensor &a,int rows,int cols,int begin=0) {
    require(a.is_available()&&a.flags().row_contiguous&&a.dtype()==mx::bfloat16&&a.buffer().ptr()&&
        begin>=0&&rows>0&&size_t(begin+rows)*cols<=a.size(),"Private QKV needs ready owned BF16 matrices");
    return {const_cast<void*>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset())+size_t(begin)*cols*2,rows,cols,
        size_t(cols)*2,ane::DType::BF16,std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
}
struct QkvWork {
    pa::Device device{true,true,0};
    pa::Surface x{device,H,B,pa::Element::I8},tx{device,1,B,pa::Element::FP16};
    pa::Surface w{device,N,H,pa::Element::I8},sw{device,N,1,pa::Element::FP16},y{device,N,B,pa::Element::FP16};
    std::unique_ptr<pa::Program> program;
    std::vector<pa::QuantStage> pending;
    uint64_t timeline=0;
    int layer=-1;
    bool running=false,current_prefetched=false;
    Clock::time_point stage_started;
    Clock::time_point prefetch_started;
    struct Future {int layer,rows;std::string generation;std::array<Tensor,3> weights;Clock::time_point started;};
    std::optional<Future> future;
    QkvWork(){pending.reserve(3);}
    std::array<ane::DeviceWeightView,3> views(const std::array<Tensor,3>&weights) {
        std::array<ane::DeviceWeightView,3> result;
        for(int part=0;part<3;++part){const auto &a=weights[part];require(a.shape()==mx::Shape{H,H},"original Q/K/V matrices required");
            auto v=matrix(a,H,H);result[part]={v.buffer,v.buffer_bytes,v.offset_bytes,v.row_stride_bytes,H,H,
                ane::DeviceWeightEncoding::Dense,ane::DType::BF16,32,{},{},v.owner,v.allocation_identity};result[part].immutable_generation=true;}
        return result;
    }
    void enqueue(std::array<ane::DeviceWeightView,3> sources,QwenPrivateQkvPrefetchMetrics &stats,bool ahead) {
        require(!running&&pending.empty(),"QKV bank still has an active reader/producer");
        for(int part=0;part<3;++part){pending.push_back(device.stage_w8(std::move(sources[part]),{0,H,0,H,128,20260930,false},
            w.slice_rows(part*H,H),sw.slice_rows(part*H,H)));if(ahead)++stats.producer_submissions;}
    }
    bool join() {
        bool valid=true;std::exception_ptr failure;
        for(auto &ticket:pending)try{auto c=ticket.finish();valid=valid&&c.ok&&!c.timed_out&&ticket.validation_flags()==0;}
            catch(...){if(!failure)failure=std::current_exception();}
        pending.clear();if(failure)std::rethrow_exception(failure);return valid;
    }
    bool retire(QwenPrivateQkvPrefetchMetrics &stats) {
        bool ok=false;std::exception_ptr error;try{ok=join();}catch(...){error=std::current_exception();}
        if(future){++stats.discards;future.reset();}
        if(current_prefetched){++stats.retired_claims;current_prefetched=false;}
        layer=-1;if(error)std::rethrow_exception(error);return ok;
    }
    ~QkvWork(){try{join();}catch(...) {}}
};
#endif
struct RuntimePrivateQkv::Impl {
    QwenPrivateQkvPrefetchMetrics stats;
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    std::unique_ptr<QkvWork> work;
#endif
};

RuntimePrivateQkv::RuntimePrivateQkv(size_t budget,std::atomic<bool>&cancelled):budget_(budget) {
    checkpoint(cancelled);
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    try {
        // Book before allocating. The fixed estimate includes room for driver
        // staging/scratch, not just the explicit IOSurfaces; not a measured RSS.
        metrics_.estimated_bytes=uint64_t(288)<<20;
        auto o=ane::observe_runtime_memory(mx::get_active_memory());
        auto admission=ane::admit_memory(o,{uint64_t(4)<<30,budget_},0,metrics_.estimated_bytes);
        if(!admission.allowed())throw ane::MemoryBudgetError("Private QKV admission: "+ane::memory_denial_reason(admission.denial,o));
        const auto started=Clock::now();impl_=std::make_unique<Impl>();impl_->work=std::make_unique<QkvWork>();auto &p=*impl_->work;
        metrics_.slot_bytes=p.x.bytes()+p.tx.bytes()+p.w.bytes()+p.sw.bytes()+p.y.bytes();
        require(metrics_.slot_bytes<=metrics_.estimated_bytes,"Private QKV slot estimate exceeded");
        p.program=std::make_unique<pa::Program>(p.device,
            pa::w8_matmul_program({ane::Kind::Matmul,B,H,N,4096,512,false}),std::span<const uint8_t>{},pa::default_cache_directory());
        auto names=p.program->inputs();std::sort(names.begin(),names.end());
        require(names==std::vector<std::string>{"w","x"}&&p.program->outputs()==std::vector<std::string>{"y"},"Private QKV ABI mismatch");
        metrics_.program_key=p.program->cache_key();metrics_.setup_seconds=span(started);
    }catch(const std::exception &e){fail(e.what());}
#else
    fail("Private QKV requires a private-enabled native build");
#endif
    checkpoint(cancelled);
}
RuntimePrivateQkv::~RuntimePrivateQkv(){try{drain();}catch(...) {}}
RuntimePrivateQkv::RuntimePrivateQkv(const std::string &reason):budget_(0){fail(reason);}
void RuntimePrivateQkv::disable(const std::string &reason){if(!metrics_.failed)fail(reason);}
bool RuntimePrivateQkv::available()const{
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    return impl_&&impl_->work&&!metrics_.failed;
#else
    return false;
#endif
}
void RuntimePrivateQkv::fail(const std::string &reason){
    metrics_.failed=true;metrics_.failure_reason=reason;
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    if(impl_&&impl_->work){try{(void)impl_->work->retire(impl_->stats);}catch(...){}impl_->work.reset();}
#endif
}
void RuntimePrivateQkv::drain(){
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    if(impl_&&impl_->work)try{if(!impl_->work->retire(impl_->stats)){++impl_->stats.failures;fail("Private QKV staging failed during drain");}}
        catch(const std::exception &e){++impl_->stats.failures;fail(e.what());}
#endif
}
void RuntimePrivateQkv::begin_request(const std::string &generation){
    begin_request(generation,false);
}
void RuntimePrivateQkv::begin_request(const std::string &generation,bool prefetch){
    drain();
    if(!impl_&&prefetch)impl_=std::make_unique<Impl>();
    if(impl_)impl_->stats.requested=prefetch;
    if(source_generation_!=generation){source_generation_=generation;metrics_.validation_passed=false;}
    if(!available())return;
    auto o=ane::observe_runtime_memory(mx::get_active_memory());
    auto a=ane::admit_memory(o,{uint64_t(4)<<30,budget_},metrics_.estimated_bytes,0);
    if(!a.allowed())fail("Private QKV resident admission: "+ane::memory_denial_reason(a.denial,o));
}
bool RuntimePrivateQkv::plan(int rows){
    return plan(rows,true);
}
bool RuntimePrivateQkv::plan(int rows,bool permit_private){
    const bool selected=eligible(rows,permit_private);if(!selected)++metrics_.gpu_blocks;return selected;
}
bool RuntimePrivateQkv::eligible(int rows,bool permit_private)const{return permit_private&&available()&&rows==M;}
QwenPrivateQkvPrefetchMetrics RuntimePrivateQkv::prefetch_metrics()const{
    auto value=impl_?impl_->stats:QwenPrivateQkvPrefetchMetrics{};value.enabled=value.requested&&available();
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    value.future_pending=impl_&&impl_->work&&bool(impl_->work->future);
#endif
    return value;
}
bool RuntimePrivateQkv::prefetch(int layer,int rows,std::array<Tensor,3> weights){
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    if(!eligible(rows)||!impl_->stats.requested)return false;
    auto &p=*impl_->work;if(p.running||!p.pending.empty()||p.future)return false;
    ++impl_->stats.attempts;
    try{
        require(layer>=0&&layer<32&&!source_generation_.empty(),"QKV future needs bounded layer and immutable source generation");
        auto views=p.views(weights);const auto started=Clock::now();p.enqueue(std::move(views),impl_->stats,true);
        ++impl_->stats.submissions;p.future.emplace(QkvWork::Future{layer,rows,source_generation_,std::move(weights),started});return true;
    }catch(const std::exception &e){++impl_->stats.failures;fail(e.what());return false;}
#else
    (void)layer;(void)rows;(void)weights;return false;
#endif
}
void RuntimePrivateQkv::stage(int layer,int rows,std::array<Tensor,3> weights){
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    if(!available())return;
    try{
        auto &p=*impl_->work;require(layer>=0&&layer<32&&rows==M&&!p.running,"Private QKV stage geometry/active-reader mismatch");
        if(p.future){auto &f=*p.future;bool same=f.layer==layer&&f.rows==rows&&f.generation==source_generation_;
            for(int i=0;i<3;++i)same=same&&f.weights[i].id()==weights[i].id();
            if(same){require(p.pending.size()==3,"QKV future lost producers");p.prefetch_started=f.started;p.future.reset();
                p.current_prefetched=true;++impl_->stats.claims;p.layer=layer;p.stage_started=Clock::now();return;}
            require(p.retire(impl_->stats),"QKV mismatched future did not drain");
        }
        require(p.pending.empty(),"Private QKV stage has outstanding producers");p.layer=layer;p.stage_started=Clock::now();
        p.enqueue(p.views(weights),impl_->stats,false);
    }catch(const std::exception &e){fail(e.what());}
#else
    (void)layer;(void)rows;(void)weights;
#endif
}
Tensor RuntimePrivateQkv::run(int layer,const Tensor &input,const Gpu &gpu,std::atomic<bool>&cancelled){
    checkpoint(cancelled);
    if(!available()){++metrics_.fallback_blocks;return gpu(input);}
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    try {
        auto &p=*impl_->work;require(layer==p.layer&&input.shape()==mx::Shape{1,M,H}&&p.pending.size()==3&&!p.running,"Private QKV staged layer/input mismatch");
        p.running=true;struct Running{bool &flag;~Running(){flag=false;}} running{p.running};
        mx::eval(input); // independent W producers can overlap norm/upstream completion
        auto weight_wait=Clock::now();const bool ready_weights=p.join();impl_->stats.weight_join_wait_seconds+=span(weight_wait);
        require(ready_weights,"Private QKV weight staging failed");++metrics_.weight_stages;
        if(p.current_prefetched){++impl_->stats.weight_completions;impl_->stats.producer_lifetime_seconds+=span(p.prefetch_started);}
        metrics_.stage_seconds+=span(p.stage_started);
        auto start=Clock::now();auto v=matrix(input,B,H,M-B);
        ane::DeviceWeightView a{v.buffer,v.buffer_bytes,v.offset_bytes,v.row_stride_bytes,B,H,
            ane::DeviceWeightEncoding::Dense,ane::DType::BF16,32,{},{},v.owner,v.allocation_identity};
        auto activation=p.device.stage_w8(a,{0,B,0,H,128,20260930,true},p.x,p.tx);auto produced=activation.finish();
        require(produced.ok&&!produced.timed_out&&activation.validation_flags()==0,"Private QKV A8 staging failed");
        metrics_.activation_seconds+=span(start);checkpoint(cancelled);
        auto allocation=mx::allocator::malloc(size_t(B)*N*2);require(allocation.ptr(),"Private QKV output allocation failed");
        Tensor tail(allocation,{1,B,N},mx::bfloat16);auto output=matrix(tail,B,N);
        require(p.timeline<=UINT64_MAX-2,"Private QKV timeline exhausted");const auto ready=++p.timeline,done=++p.timeline;
        auto transfer=p.device.prepare_transfer({},{{p.y,output,0,ane::DType::BF16,1.f,p.sw,p.tx}},ready,done);
        std::pair<std::string,pa::Surface> inputs[]{{"x",p.x},{"w",p.w}},outputs[]{{"y",p.y}};
        auto ticket=p.program->enqueue(inputs,outputs,ready,done,transfer.failure_callback());
        std::optional<Tensor> head;
        try{transfer.submit();head=gpu(slice_axis(input,1,0,M-B));mx::async_eval(*head);}
        catch(...){try{(void)ticket.finish();(void)transfer.finish();}catch(...){}throw;}
        start=Clock::now();auto evaluated=ticket.finish();auto restored=transfer.finish();
        require(evaluated.ok&&evaluated.physical_complete&&!evaluated.retired&&!evaluated.timed_out&&
            restored.ok&&!restored.timed_out&&transfer.validation_flags()==0,"Private QKV predict/restore failed");
        metrics_.join_seconds+=span(start);++metrics_.predictions;
        auto result=mx::concatenate({*head,tail},1);mx::eval(result);checkpoint(cancelled);
        if(!metrics_.validation_passed){
            auto reference=gpu(input);auto x=mx::astype(result,mx::float32),y=mx::astype(reference,mx::float32);
            auto error=mx::sqrt(mx::sum(mx::square(x-y))/mx::maximum(mx::sum(mx::square(y)),Tensor(1e-30f)));
            metrics_.validation_relative_l2=error.item<float>();++metrics_.validation_calls;
            require(std::isfinite(metrics_.validation_relative_l2)&&metrics_.validation_relative_l2<.02,
                "Private QKV first real projection failed numeric validation");
            metrics_.validation_passed=true;
        }
        ++metrics_.completed_blocks;if(p.current_prefetched)++impl_->stats.completed_hits;p.current_prefetched=false;p.layer=-1;return result;
    }catch(const std::exception &e){
        if(cancelled.load())throw;
        if(impl_&&impl_->work&&impl_->work->current_prefetched)++impl_->stats.failures;
        fail(e.what());++metrics_.fallback_blocks;return gpu(input);
    }
#else
    (void)layer;++metrics_.fallback_blocks;return gpu(input);
#endif
}
} // namespace tc::qwen21
