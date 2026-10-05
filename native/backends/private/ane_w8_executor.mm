#include "ane_w8_executor.hpp"
#include "ane_program.hpp"
#include "ane_mil.hpp"
#include "../ane_memory.hpp"
#include "../ane_w8a8_math.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <tuple>

namespace tc::ane {
namespace {
using Clock = std::chrono::steady_clock;
using private_api::Surface;
using private_api::Element;
double elapsed(Clock::time_point x) { return std::chrono::duration<double>(Clock::now() - x).count(); }
void check(bool ok, const char *reason) { if (!ok) throw std::runtime_error(reason); }
bool same_matrix(const DeviceMatrixView &a,const DeviceMatrixView &b) {
    return std::tie(a.buffer,a.buffer_bytes,a.offset_bytes,a.rows,a.cols,a.row_stride_bytes,a.dtype)==
           std::tie(b.buffer,b.buffer_bytes,b.offset_bytes,b.rows,b.cols,b.row_stride_bytes,b.dtype);
}
bool same_region(const DeviceWeightRegion &a,const DeviceWeightRegion &b) {
    const auto &x=a.source,&y=b.source;const auto &p=a.selection,&q=b.selection;
    const auto same_optional=[](const auto&a,const auto&b){return bool(a)==bool(b)&&(!a||same_matrix(*a,*b));};
    return std::tie(x.buffer,x.buffer_bytes,x.offset_bytes,x.row_stride_bytes,x.rows,x.cols,x.encoding,x.dense_dtype,x.group_size)==
           std::tie(y.buffer,y.buffer_bytes,y.offset_bytes,y.row_stride_bytes,y.rows,y.cols,y.encoding,y.dense_dtype,y.group_size)&&
        same_optional(x.scales,y.scales)&&same_optional(x.offsets,y.offsets)&&
        std::tie(p.row_begin,p.rows,p.column_begin,p.columns,p.rotation_block,p.rotation_seed,p.transpose)==
        std::tie(q.row_begin,q.rows,q.column_begin,q.columns,q.rotation_block,q.rotation_seed,q.transpose);
}
class Worker {
    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> job_;
    std::exception_ptr error_;
    bool busy_ = false, stop_ = false;
    std::thread thread_;
  public:
    Worker() : thread_([this] {
        for (;;) {
            std::function<void()> job;
            { std::unique_lock lock(mutex_); cv_.wait(lock, [&] { return stop_ || job_; });
                if (stop_) return; job = std::exchange(job_, {}); }
            std::exception_ptr error;
            @autoreleasepool { try { job(); } catch (...) { error = std::current_exception(); } }
            { std::lock_guard lock(mutex_); busy_ = false; error_ = error; } cv_.notify_all();
        }
    }) {}
    ~Worker() { try { join(); } catch (...) {} { std::lock_guard lock(mutex_); stop_ = true; } cv_.notify_all(); thread_.join(); }
    void join() { std::unique_lock lock(mutex_); cv_.wait(lock, [&] { return !busy_; }); auto error = std::exchange(error_, {}); if (error) std::rethrow_exception(error); }
    void submit(std::function<void()> job) { join(); { std::lock_guard lock(mutex_); job_ = std::move(job); busy_ = true; } cv_.notify_all(); }
};
struct Buffer {
    id<MTLBuffer> value;
    std::shared_ptr<void> owner;
    Buffer(id<MTLDevice> d, size_t bytes) {
        value = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!value) throw MemoryBudgetError("W8 self-test buffer allocation failed");
        owner = {(__bridge_retained void *)value, [](void *p) { CFRelease(p); }}; std::memset(value.contents, 0, bytes);
    }
    DeviceWeightView weight(int rows, int cols) { return {(__bridge void *)value, value.length, 0, size_t(cols) * 4, rows, cols,
        DeviceWeightEncoding::Dense, DType::FP32, 32, {}, {}, owner}; }
    DeviceMatrixView matrix(int rows, int cols, DType dtype = DType::FP32) { return {(__bridge void *)value, value.length, 0, rows, cols,
        size_t(cols) * (dtype == DType::FP32 ? 4 : 2), dtype, owner}; }
};
}
struct PrivateW8Graph::Impl {
    GraphShape shape;
    private_api::Device device;
    std::unique_ptr<private_api::Program> program;
    // Exactly two source-independent banks, reused across every layer/step.
    struct Bank {
        Surface g, sg, u, su, d, sd;
        bool ready = false;
        Bank(private_api::Device &dev, const GraphShape &s) : g(dev,s.width,s.hidden,Element::I8), sg(dev,s.width,1,Element::FP16),
            u(dev,s.width,s.hidden,Element::I8), su(dev,s.width,1,Element::FP16), d(dev,s.hidden,s.width,Element::I8), sd(dev,s.hidden,1,Element::FP16) {}
        size_t bytes() const { return g.bytes()+sg.bytes()+u.bytes()+su.bytes()+d.bytes()+sd.bytes(); }
    };
    std::array<std::unique_ptr<Bank>, 2> banks;
    // Two bounded A8 slots, independent of the two W banks. A slot is reused
    // only after its ANE consumer has joined; the other may stage the next row
    // chunk while that consumer runs. No full-sequence activation copy.
    std::array<std::unique_ptr<Surface>, 2> x, tx;
    std::unique_ptr<Surface> dg, du, y;
    bool a8_lookahead = false;
    int current = -1;
    bool verified = false;
    std::atomic<bool> disabled{false};
    float headroom = 1.f;
    uint64_t timeline = 0;
    size_t allocated = 0, estimate = 0;
    double load_time = 0;
    std::filesystem::path cache;
    RunResult result, future_result;
    std::mutex submission_mutex;
    std::condition_variable submission_cv;
    bool launch_fence=false, submitted=false, launch_finished=false;
    int future_bank = -1;
    bool future_pending = false;
    std::vector<DeviceWeightRegion> future_sources; // allocation leases + immutable source identity
    Worker staging_worker;
    Worker worker; // destroyed first, draining before any source/slot release
    void build(float scale) {
        auto emitted = private_api::w8_swiglu_program(shape, 20260930, scale);
        program = std::make_unique<private_api::Program>(device, emitted.mil, emitted.constants, cache);
        headroom = scale;
    }
    void fill_bank(int target,const std::vector<DeviceWeightRegion> &weights) {
        check(weights.size() == 3, "W8 SwiGLU requires gate/up/down");
        const auto &s = shape;
        const auto &gsel = weights[0].selection, &usel = weights[1].selection, &dsel = weights[2].selection;
        check(gsel.rows == s.width && gsel.columns == s.hidden && usel.rows == s.width && usel.columns == s.hidden &&
            dsel.rows == s.hidden && dsel.columns == s.width && gsel.rotation_block == 128 && usel.rotation_block == 128 &&
            dsel.rotation_block == 512 && !gsel.transpose && !usel.transpose && !dsel.transpose &&
            gsel.rotation_seed == 20260930 && usel.rotation_seed == 20260930 && dsel.rotation_seed == 20260930,
            "W8 source projection selection/recipe mismatch");
        auto &b = *banks[target]; b.ready = false;
        auto g = device.stage_w8(weights[0].source, gsel, b.g,b.sg);
        auto u = device.stage_w8(weights[1].source, usel, b.u,b.su);
        auto d = device.stage_w8(weights[2].source, dsel, b.d,b.sd);
        // Wait ALL producers even on the first failure; source owners and slot
        // leases must not be released while another GPU encoder still uses them.
        const auto gr = g.finish(), ur = u.finish(), dr = d.finish();
        check(gr.ok && ur.ok && dr.ok, "W8 GPU weight staging failed");
        b.ready = true;
    }
    void stage(std::vector<DeviceWeightRegion> weights) {
        const int next=current<0?0:current^1;
        fill_bank(next,weights);current=next;
    }
    void run(DeviceMatrixView input, DeviceMatrixView output, std::optional<DeviceAdapterInput> adapter,
             const std::function<void()> &first_submit={}) {
        const auto &s = shape;
        check(!disabled && current >= 0 && banks[current]->ready && input.rows > 0 && input.rows % s.rows == 0 && input.cols == s.hidden &&
            output.rows == input.rows && output.cols == s.hidden && (output.dtype == DType::FP16 || output.dtype == DType::BF16), "W8 launch geometry/slots unavailable");
        check(!adapter || (s.lora_inputs && adapter->gate.rows == input.rows && adapter->up.rows == input.rows && adapter->hidden.rows == input.rows &&
            adapter->gate.cols == s.width && adapter->up.cols == s.width && adapter->hidden.cols == s.width && adapter->hidden.dtype == output.dtype), "W8 LoRA geometry mismatch");
        check(input.buffer != output.buffer && (!adapter || (adapter->hidden.buffer != output.buffer && adapter->hidden.buffer != input.buffer &&
            adapter->gate.buffer != output.buffer && adapter->up.buffer != output.buffer)), "W8 source/output aliases");
        if (s.lora_inputs && !adapter) for (auto *slot : {dg.get(),du.get()}) std::memset(slot->data(),0,slot->rows()*slot->pitch());
        auto &b = *banks[current];
        std::array<std::optional<private_api::QuantStage>, 2> activations;
        // Drain every submitted producer even on cancellation, a late input
        // failure, or a failed ANE/epilogue. Never allow recovery to overwrite
        // a slot still used by a previous GPU producer. A timed-out producer
        // disables this executor; its ticket retains the resources itself.
        struct ActivationDrain {
            decltype(activations) &tickets;
            std::atomic<bool> &disabled;
            ~ActivationDrain() {
                for (auto &ticket : tickets) if (ticket) {
                    try { if (ticket->finish().timed_out) disabled = true; }
                    catch (...) { disabled = true; }
                }
            }
        } drain{activations, disabled};
        auto stage_activation = [&](int row, int slot) {
            DeviceWeightView activation{input.buffer,input.buffer_bytes,input.offset_bytes + size_t(row)*(input.row_stride_bytes ? input.row_stride_bytes : size_t(s.hidden)*(input.dtype==DType::FP32?4:2)),
                input.row_stride_bytes,s.rows,s.hidden,DeviceWeightEncoding::Dense,input.dtype,32,{}, {},input.owner};
            activations[slot] = device.stage_w8(activation,{0,s.rows,0,s.hidden,128,20260930,true},*x[slot],*tx[slot]);
        };
        for (int row = 0; row < input.rows; row += s.rows) {
            const int slot = a8_lookahead ? (row / s.rows) % 2 : 0;
            const auto activation_start = Clock::now();
            if (!activations[slot]) stage_activation(row, slot);
            const auto wait_start = Clock::now();
            const auto staged_x = activations[slot]->finish();
            result.activation_wait_seconds += elapsed(wait_start);
            if (staged_x.timed_out) disabled = true;
            check(staged_x.ok, "W8 activation staging failed");
            activations[slot].reset();
            result.input_seconds += elapsed(activation_start); // GPU-stage exposed host span
            bool prepared_next = false;
            for (;;) {
                check(timeline <= UINT64_MAX-2, "W8 timeline exhausted");
                const auto ready = ++timeline, done = ++timeline;
                std::vector<std::pair<std::string,Surface>> inputs{{"x",*x[slot]},{"tx",*tx[slot]},{"wg",b.g},{"sg",b.sg},{"wu",b.u},{"su",b.su},{"wd",b.d}};
                std::vector<private_api::Upload> uploads;
                if (s.lora_inputs) { inputs.emplace_back("dg",*dg); inputs.emplace_back("du",*du);
                    if (adapter) { uploads.push_back({adapter->gate,*dg,row}); uploads.push_back({adapter->up,*du,row}); } }
                auto norm = y->slice_rows(0,s.hidden), hs = y->slice_rows(s.hidden,1);
                std::vector<private_api::Download> downloads{{norm,output,row,output.dtype,headroom,b.sd,hs}};
                if (s.lora_inputs) downloads.push_back({y->slice_rows(s.hidden+1,s.width),adapter ? std::optional<DeviceMatrixView>(adapter->hidden) : std::nullopt,row,output.dtype,headroom});
                const auto submit_start = Clock::now();
                auto io = device.prepare_transfer(std::move(uploads),std::move(downloads),ready,done);
                std::pair<std::string,Surface> outputs[]{{"y",*y}};
                auto request = program->enqueue(inputs,outputs,ready,done,io.failure_callback());
                try { io.submit(); } catch (...) { request.finish(std::chrono::milliseconds(0)); disabled=true; throw; }
                if(row==0 && first_submit)first_submit();
                result.input_seconds += elapsed(submit_start); // submission, not kernel duration
                // The leading ready producer is committed and current ANE
                // request is in flight. Fill ONLY the alternate A8 slot;
                // headroom retries retain current x/tx and never resubmit the
                // future producer. Defer staging errors until BOTH current
                // consumers have joined, keeping their tickets tracked.
                std::exception_ptr future_error;
                if (a8_lookahead && !prepared_next && row + s.rows < input.rows) {
                    try { stage_activation(row + s.rows, slot ^ 1); ++result.activation_prefetches; }
                    catch (...) { future_error = std::current_exception(); }
                    prepared_next = true;
                }
                const auto predict_start = Clock::now();
                const auto eval = request.finish(); ++result.calls;
                result.prediction_seconds += elapsed(predict_start); // exposed request wait
                const auto output_start = Clock::now();
                const auto copied = io.finish();
                result.output_seconds += elapsed(output_start); // exposed GPU epilogue wait
                if (!eval.ok || !copied.ok) { disabled=true; throw CapabilityError("W8 evaluation/GPU epilogue failed"); }
                if (future_error) std::rethrow_exception(future_error);
                const auto flags = io.validation_flags();
                check(!(flags&1), "W8 correction input overflow/nonfinite");
                if (flags&2) {
                    check(headroom<4096, "W8 exhausted headroom"); build(headroom*4); ++result.overflow_retries; continue;
                }
                check(!(flags&4), "W8 restored output exceeds target dtype");
                result.copied_output_bytes += size_t(s.rows)*(s.hidden+(adapter?s.width:0))*2; break;
            }
        }
        result.headroom_scale=headroom;
    }
};
PrivateW8Graph::PrivateW8Graph(GraphShape shape,size_t budget,const std::filesystem::path &cache) : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        auto start=Clock::now(); auto &p=*impl_; p.shape=shape;
        const char *lookahead=std::getenv("TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD");
        check(!lookahead || std::string(lookahead)=="0" || std::string(lookahead)=="1", "private ANE A8 lookahead requires 0 or 1");
        p.a8_lookahead=lookahead && std::string(lookahead)=="1";
        auto spec=private_api::w8_swiglu_program(shape,20260930,1.f);
        p.estimate=uint64_t(shape.hidden)*shape.width*6 + uint64_t(shape.rows)*(8ull*shape.width+12ull*shape.hidden) + spec.constants.size()*2 + (128ull<<20) + private_api::scale_cache_budget_bytes;
        if (p.a8_lookahead) p.estimate += uint64_t(shape.hidden + 1) * ((uint64_t(shape.rows) + 63) / 64 * 64 + 128);
        if (p.estimate>budget) throw MemoryBudgetError("W8 graph/banks exceed memory budget");
        const auto decision=admit_memory(observe_runtime_memory(0),{uint64_t(4)<<30,budget},0,p.estimate);
        if (!decision.allowed()) throw MemoryBudgetError("W8 system memory admission denied");
        p.cache=cache;
        if (p.cache.empty()) { NSString *dir=NSSearchPathForDirectoriesInDomains(NSCachesDirectory,NSUserDomainMask,YES).firstObject;
            check(dir!=nil,"W8 cache unavailable"); p.cache=std::filesystem::path(dir.UTF8String)/"TurboCider/ane/private"; }
        for (auto &bank:p.banks) { bank=std::make_unique<Impl::Bank>(p.device,shape); p.allocated+=bank->bytes(); }
        auto add=[&](std::unique_ptr<Surface>&slot,int rows,int cols,Element element) { slot=std::make_unique<Surface>(p.device,rows,cols,element);p.allocated+=slot->bytes(); };
        for (int slot=0;slot<(p.a8_lookahead?2:1);++slot) {
            add(p.x[slot],shape.hidden,shape.rows,Element::I8);add(p.tx[slot],1,shape.rows,Element::FP16);
        }
        add(p.y,spec.packed_rows,shape.rows,Element::FP16);
        if (shape.lora_inputs) {add(p.dg,shape.width,shape.rows,Element::FP16);add(p.du,shape.width,shape.rows,Element::FP16);}
        if (p.allocated>budget || p.allocated>p.estimate) throw MemoryBudgetError("W8 actual allocations exceed admission");
        p.program=std::make_unique<private_api::Program>(p.device,spec.mil,spec.constants,p.cache);p.load_time=elapsed(start);
        const char*fence=std::getenv("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE");
        check(!fence||std::string(fence)=="0"||std::string(fence)=="1","private ANE launch fence requires 0 or 1");
        p.launch_fence=fence&&std::string(fence)=="1";
    }
}
PrivateW8Graph::~PrivateW8Graph()=default;
const GraphShape &PrivateW8Graph::shape() const{return impl_->shape;}
size_t PrivateW8Graph::slot_bytes() const{return impl_->allocated;}
size_t PrivateW8Graph::estimated_bytes() const{return impl_->estimate;}
double PrivateW8Graph::load_seconds() const{return impl_->load_time;}
std::string PrivateW8Graph::weight_recipe() const{return w8a8_recipe;}
WeightCacheStats PrivateW8Graph::weight_cache_stats() const{return impl_->device.scale_cache_stats();}
StagePipelineStats PrivateW8Graph::stage_pipeline_stats() const{return impl_->device.stage_pipeline_stats();}
bool PrivateW8Graph::device_submission_fence_enabled() const{return impl_->launch_fence;}
bool PrivateW8Graph::activation_lookahead_enabled() const{return impl_->a8_lookahead;}
void PrivateW8Graph::stage_weights(std::vector<WeightView>){throw CapabilityError("W8 requires explicit GPU weight bindings");}
void PrivateW8Graph::launch(MatrixView,uint16_t*,size_t,DType,std::optional<AdapterInput>){throw CapabilityError("W8 requires explicit GPU I/O bindings");}
void PrivateW8Graph::stage_device_weights(std::vector<DeviceWeightView> sources){
    check(sources.size() == 3, "W8 SwiGLU requires gate/up/down");
    const auto &s = impl_->shape;
    check(sources[0].rows == s.width && sources[0].cols == s.hidden && sources[1].rows == s.width &&
          sources[1].cols == s.hidden && sources[2].rows == s.hidden && sources[2].cols == s.width,
          "W8 full weight geometry mismatch (use regions for a physical source slice)");
    stage_device_weight_regions({{std::move(sources[0]),{0,s.width,0,s.hidden,128}},
        {std::move(sources[1]),{0,s.width,0,s.hidden,128}}, {std::move(sources[2]),{0,s.hidden,0,s.width,512}}});
}
void PrivateW8Graph::stage_device_weight_regions(std::vector<DeviceWeightRegion> sources){
    discard_prefetched_weights();
    auto &p=*impl_;p.worker.join();check(p.verified&&!p.disabled,"W8 self-test required/executor disabled");p.result={};
    if(p.current>=0)p.banks[p.current]->ready=false;
    p.worker.submit([&p,sources=std::move(sources)]{auto start=Clock::now();try{p.stage(sources);p.result.ok=true;}catch(const std::exception&e){p.result.error=e.what();}p.result.stage_seconds=elapsed(start);});
}
void PrivateW8Graph::prefetch_device_weight_regions(std::vector<DeviceWeightRegion> sources) {
    auto &p=*impl_;
    check(p.verified && !p.disabled && p.current>=0 && p.banks[p.current]->ready,"W8 prefetch requires a ready current bank");
    check(!p.future_pending,"W8 exactly one future bank may be prepared");
    p.future_bank=p.current^1;p.future_sources=std::move(sources);p.future_result={};p.future_pending=true;
    p.staging_worker.submit([&p]{
        const auto start=Clock::now();
        try { p.fill_bank(p.future_bank,p.future_sources);p.future_result.ok=true; }
        catch(const std::exception&e){p.future_result.error=e.what();}
        p.future_result.stage_seconds=elapsed(start);
    });
}
std::optional<RunResult> PrivateW8Graph::activate_prefetched_weights(std::span<const DeviceWeightRegion> expected) {
    auto &p=*impl_;
    if(!p.future_pending || expected.size()!=p.future_sources.size() ||
       !std::equal(expected.begin(),expected.end(),p.future_sources.begin(),same_region))return std::nullopt;
    // Both reuse fences: current ANE/epilogue consumer and future GPU producer.
    p.worker.join();p.staging_worker.join();
    if(p.current>=0)p.banks[p.current]->ready=false;
    p.result=p.future_result;
    if(p.result.ok)p.current=p.future_bank;else p.current=-1;
    p.future_pending=false;p.future_bank=-1;p.future_sources.clear();
    return p.result;
}
void PrivateW8Graph::discard_prefetched_weights() {
    auto &p=*impl_;
    if(!p.future_pending)return;
    p.staging_worker.join();
    p.banks[p.future_bank]->ready=false;p.future_pending=false;p.future_bank=-1;p.future_sources.clear();
}
RunResult PrivateW8Graph::wait_stage(){return finish();}
void PrivateW8Graph::launch_device(DeviceMatrixView input,DeviceMatrixView output,std::optional<DeviceAdapterInput> adapter){
    auto &p=*impl_;p.worker.join();const auto stage=p.result.stage_seconds;p.result={};p.result.stage_seconds=stage;
    {std::lock_guard lock(p.submission_mutex);p.submitted=false;p.launch_finished=false;}
    p.worker.submit([&p,input=std::move(input),output=std::move(output),adapter=std::move(adapter)]{
        auto start=Clock::now();
        auto notify=[&p]{ {std::lock_guard lock(p.submission_mutex);p.submitted=true;}p.submission_cv.notify_all(); };
        try{check(p.verified,"W8 self-test required");p.run(input,output,adapter,notify);p.result.ok=true;}
        catch(const std::exception&e){p.result.error=e.what();}
        p.result.total_seconds=elapsed(start);
        {std::lock_guard lock(p.submission_mutex);p.launch_finished=true;}p.submission_cv.notify_all();
    });
    if(p.launch_fence) {
        // Only the first request/ready producer submission, not ANE output.
        // Error paths notify as finished so malformed input cannot deadlock
        // the owning model thread before its ordinary finish/fallback.
        std::unique_lock lock(p.submission_mutex);
        if(!p.submission_cv.wait_for(lock,std::chrono::seconds(45),[&]{return p.submitted||p.launch_finished;})) {
            lock.unlock();p.disabled=true;p.worker.join();p.result.ok=false;p.result.error="W8 first-submission deadline exceeded";
        }
    }
}
RunResult PrivateW8Graph::finish(){auto&p=*impl_;try{p.worker.join();}catch(const std::exception&e){p.result.ok=false;p.result.error=e.what();}return p.result;}
bool PrivateW8Graph::self_test(std::string &error){
    discard_prefetched_weights();
    auto &p=*impl_;p.worker.join();p.verified=false;
    p.worker.submit([&p]{
        const auto&s=p.shape;auto gpu=MTLCreateSystemDefaultDevice();
        Buffer g(gpu,size_t(s.width)*s.hidden*4),u(gpu,size_t(s.width)*s.hidden*4),d(gpu,size_t(s.hidden)*s.width*4),
            x(gpu,size_t(s.rows)*s.hidden*4),y(gpu,size_t(s.rows)*s.hidden*2);
        for(int r=0;r<s.rows;++r)for(int c=0;c<s.hidden;++c)static_cast<float*>(x.value.contents)[r*s.hidden+c]=((r*3+c*7)%17-8)/8.f;
        for(float scale:{.125f,-.25f,.125f}){
            std::memset(g.value.contents,0,g.value.length);std::memset(u.value.contents,0,u.value.length);std::memset(d.value.contents,0,d.value.length);
            for(int r=0;r<s.width;++r){static_cast<float*>(g.value.contents)[size_t(r)*s.hidden+r%s.hidden]=scale;static_cast<float*>(u.value.contents)[size_t(r)*s.hidden+r%s.hidden]=scale;}
            for(int r=0;r<s.hidden;++r)static_cast<float*>(d.value.contents)[size_t(r)*s.width+r%s.width]=scale;
            p.stage({{g.weight(s.width,s.hidden),{0,s.width,0,s.hidden,128}},
                {u.weight(s.width,s.hidden),{0,s.width,0,s.hidden,128}},
                {d.weight(s.hidden,s.width),{0,s.hidden,0,s.width,512}}});
            p.result={};p.run(x.matrix(s.rows,s.hidden),y.matrix(s.rows,s.hidden,DType::BF16),std::nullopt);
            double diff=0,norm=0;
            for(int r=0;r<s.rows;++r)for(int c=0;c<s.hidden;++c){const int input=(c%s.width)%s.hidden;const float v=static_cast<const float*>(x.value.contents)[r*s.hidden+input]*scale;
                const float expected=v/(1+std::exp(-v))*v*scale;const float actual=std::bit_cast<float>(uint32_t(static_cast<const uint16_t*>(y.value.contents)[r*s.hidden+c])<<16);
                check(std::isfinite(actual),"W8 self-test nonfinite");
                // Aggregate L2 alone can hide a missing/corrupt final row in
                // a large bucket. Retain that original gate AND independently
                // bound every scalar against this sparse source oracle.
                check(std::abs(actual-expected)<=.0003f+.08f*std::abs(expected),
                      "W8 self-test pointwise source mismatch");
                diff+=double(actual-expected)*(actual-expected);norm+=double(expected)*expected;}
            check(norm>0&&std::sqrt(diff/norm)<.05,"W8 numerical/weight-switch self-test failed");
        }
        p.verified=true;for(auto&b:p.banks)b->ready=false;p.current=-1;p.result={};
    });
    try{p.worker.join();error.clear();return true;}catch(const std::exception&e){error=e.what();return false;}
}
}
