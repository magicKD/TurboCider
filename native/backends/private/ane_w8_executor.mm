#include "ane_w8_executor.hpp"
#include "ane_program.hpp"
#include "ane_mil.hpp"
#include "../ane_memory.hpp"
#include "../ane_w8a8_math.hpp"
#include "../ane_backend.hpp"
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
        std::tie(p.row_begin,p.rows,p.column_begin,p.columns,p.rotation_block,p.rotation_seed,p.transpose,p.basis,p.activation_group_size)==
        std::tie(q.row_begin,q.rows,q.column_begin,q.columns,q.rotation_block,q.rotation_seed,q.transpose,q.basis,q.activation_group_size);
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
    W8Basis basis = W8Basis::SylvesterDH;
    int activation_group_size = 0;
    int hidden_group_size = 0;
    bool bf16_value_boundaries = false;
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
        check(!bf16_value_boundaries || scale==1.f,"BF16 value-boundary recipe cannot change headroom; whole-operation GPU recompute required");
        auto emitted = private_api::w8_swiglu_program(shape, basis==W8Basis::ComfyH256?0:20260930, scale, basis,activation_group_size,hidden_group_size,bf16_value_boundaries);
        program = std::make_unique<private_api::Program>(device, emitted.mil, emitted.constants, cache);
        headroom = scale;
    }
    void fill_bank(int target,const std::vector<DeviceWeightRegion> &weights) {
        check(weights.size() == 3, "W8 SwiGLU requires gate/up/down");
        const auto &s = shape;
        const bool comfy=basis==W8Basis::ComfyH256;
        const int up_block=comfy?256:128,down_block=comfy?256:512;
        const uint64_t seed=comfy?0:20260930;
        const auto &gsel = weights[0].selection, &usel = weights[1].selection, &dsel = weights[2].selection;
        check(gsel.rows == s.width && gsel.columns == s.hidden && usel.rows == s.width && usel.columns == s.hidden &&
            dsel.rows == s.hidden && dsel.columns == s.width && gsel.rotation_block == up_block && usel.rotation_block == up_block &&
            dsel.rotation_block == down_block && !gsel.transpose && !usel.transpose && !dsel.transpose &&
            gsel.rotation_seed == seed && usel.rotation_seed == seed && dsel.rotation_seed == seed &&
            gsel.basis==basis && usel.basis==basis && dsel.basis==basis &&
            gsel.activation_group_size==0 && usel.activation_group_size==0 && dsel.activation_group_size==0,
            "W8 source projection selection/recipe mismatch");
        auto &b = *banks[target]; b.ready = false;
        std::array<std::optional<private_api::QuantStage>,3> producers;
        struct ProducerDrain {
            decltype(producers) &tickets;
            std::atomic<bool> &disabled;
            ~ProducerDrain() {
                for(auto &ticket:tickets)if(ticket) {
                    try {if(ticket->finish().timed_out)disabled=true;}
                    catch(...) {disabled=true;}
                }
            }
        } drain{producers,disabled};
        producers[0]=device.stage_w8(weights[0].source,gsel,b.g,b.sg);
        producers[1]=device.stage_w8(weights[1].source,usel,b.u,b.su);
        producers[2]=device.stage_w8(weights[2].source,dsel,b.d,b.sd);
        // Wait ALL producers even on the first failure; source owners and slot
        // leases must not be released while another GPU encoder still uses them.
        const auto gr = producers[0]->finish(), ur = producers[1]->finish(), dr = producers[2]->finish();
        if(gr.timed_out || ur.timed_out || dr.timed_out)disabled=true;
        for(auto &producer:producers)producer.reset(); // callbacks retain timed-out GPU resources
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
            output.rows == input.rows && output.cols == s.hidden &&
            (output.dtype == DType::FP16 || output.dtype == DType::BF16 ||
                (output.dtype == DType::FP32 && basis==W8Basis::ComfyH256 && !adapter)), "W8 launch geometry/slots unavailable");
        check(!adapter || (s.lora_inputs && adapter->gate.rows == input.rows && adapter->up.rows == input.rows && adapter->hidden.rows == input.rows &&
            adapter->gate.cols == s.width && adapter->up.cols == s.width && adapter->hidden.cols == s.width && adapter->hidden.dtype == output.dtype), "W8 LoRA geometry mismatch");
        check(!bf16_value_boundaries || input.dtype==DType::BF16,"BF16 value-boundary recipe requires original BF16 activation input");
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
            const bool comfy=basis==W8Basis::ComfyH256;
            activations[slot] = device.stage_w8(activation,{0,s.rows,0,s.hidden,comfy?256:128,comfy?0u:20260930u,true,basis,activation_group_size},*x[slot],*tx[slot]);
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
                if(basis==W8Basis::ComfyH256)downloads[0].row_scale_policy=private_api::RowScalePolicy::SignedFinite;
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
                result.copied_output_bytes += size_t(s.rows)*s.hidden*(output.dtype==DType::FP32?4:2)+
                    size_t(s.rows)*(adapter?s.width:0)*2; break;
            }
        }
        result.headroom_scale=headroom;
    }
};
PrivateW8Graph::PrivateW8Graph(GraphShape shape,size_t budget,const std::filesystem::path &cache,W8Basis basis) : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        auto start=Clock::now(); auto &p=*impl_; p.shape=shape;p.basis=basis;
        p.bf16_value_boundaries=configured_convrot_bf16_boundaries();
        const char *group=std::getenv("TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE");
        check(!group || std::string(group)=="0" || std::string(group)=="256","private ANE A8 group size requires 0 or 256");
        p.activation_group_size=group && std::string(group)=="256"?256:0;
        p.hidden_group_size=p.activation_group_size;
        const char *scope=std::getenv("TURBOCIDER_PRIVATE_ANE_A8_GROUP_SCOPE");
        check(!scope || std::string(scope)=="both" || std::string(scope)=="input" || std::string(scope)=="hidden",
              "private ANE A8 group scope requires input, hidden or both");
        check(!scope || p.activation_group_size==256,"A8 group scope requires explicit group size 256");
        if(scope && std::string(scope)=="input")p.hidden_group_size=0;
        if(scope && std::string(scope)=="hidden")p.activation_group_size=0;
        check(!(p.activation_group_size || p.hidden_group_size) || basis==W8Basis::ComfyH256,"group A8 requires the explicit Comfy direct-code basis");
        const char *lookahead=std::getenv("TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD");
        check(!lookahead || std::string(lookahead)=="0" || std::string(lookahead)=="1", "private ANE A8 lookahead requires 0 or 1");
        p.a8_lookahead=lookahead && std::string(lookahead)=="1";
        auto spec=private_api::w8_swiglu_program(shape,basis==W8Basis::ComfyH256?0:20260930,1.f,basis,p.activation_group_size,p.hidden_group_size,p.bf16_value_boundaries);
        p.estimate=uint64_t(shape.hidden)*shape.width*6 + uint64_t(shape.rows)*(8ull*shape.width+12ull*shape.hidden) + spec.constants.size()*2 + (128ull<<20) + private_api::scale_cache_budget_bytes;
        // Extra compiler/pointwise workspace allowance, not a physical ANE
        // scratch-size observation. End-to-end footprint still needs audit.
        if(p.bf16_value_boundaries)p.estimate+=uint64_t(shape.rows)*shape.width*2*16+(64ull<<20);
        if (p.a8_lookahead) p.estimate += uint64_t(shape.hidden + 1) * ((uint64_t(shape.rows) + 63) / 64 * 64 + 128);
        if(p.activation_group_size)p.estimate+=uint64_t(shape.hidden/256-1)*
            ((uint64_t(shape.rows)*2+63)/64*64+128)*(p.a8_lookahead?2:1)+(64u<<10);
        if (p.estimate>budget) throw MemoryBudgetError("W8 graph/banks exceed memory budget");
        const auto decision=admit_memory(observe_runtime_memory(0),{uint64_t(4)<<30,budget},0,p.estimate);
        if (!decision.allowed()) throw MemoryBudgetError("W8 system memory admission denied");
        p.cache=cache;
        if (p.cache.empty()) { NSString *dir=NSSearchPathForDirectoriesInDomains(NSCachesDirectory,NSUserDomainMask,YES).firstObject;
            check(dir!=nil,"W8 cache unavailable"); p.cache=std::filesystem::path(dir.UTF8String)/"TurboCider/ane/private"; }
        for (auto &bank:p.banks) { bank=std::make_unique<Impl::Bank>(p.device,shape); p.allocated+=bank->bytes(); }
        auto add=[&](std::unique_ptr<Surface>&slot,int rows,int cols,Element element) { slot=std::make_unique<Surface>(p.device,rows,cols,element);p.allocated+=slot->bytes(); };
        for (int slot=0;slot<(p.a8_lookahead?2:1);++slot) {
            add(p.x[slot],shape.hidden,shape.rows,Element::I8);add(p.tx[slot],p.activation_group_size?shape.hidden/256:1,shape.rows,Element::FP16);
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
std::string PrivateW8Graph::weight_recipe() const {
    if(impl_->bf16_value_boundaries)return convrot_bf16_value_recipe;
    if(impl_->activation_group_size && impl_->hidden_group_size)return convrot_group_w8a8_recipe;
    if(impl_->activation_group_size)return convrot_input_group_w8a8_recipe;
    if(impl_->hidden_group_size)return convrot_hidden_group_w8a8_recipe;
    return impl_->basis==W8Basis::ComfyH256?convrot_w8a8_recipe:w8a8_recipe;
}
std::string PrivateW8Graph::data_path() const{return impl_->basis==W8Basis::ComfyH256?"w8a8_convrot":"w8a8_hadamard";}
WeightCacheStats PrivateW8Graph::weight_cache_stats() const{return impl_->device.scale_cache_stats();}
StagePipelineStats PrivateW8Graph::stage_pipeline_stats() const{return impl_->device.stage_pipeline_stats();}
bool PrivateW8Graph::device_submission_fence_enabled() const{return impl_->launch_fence;}
bool PrivateW8Graph::activation_lookahead_enabled() const{return impl_->a8_lookahead;}
bool PrivateW8Graph::supports_fp32_device_output() const{return impl_->basis==W8Basis::ComfyH256;}
int PrivateW8Graph::activation_group_size() const{return impl_->activation_group_size;}
int PrivateW8Graph::hidden_activation_group_size() const{return impl_->hidden_group_size;}
void PrivateW8Graph::stage_weights(std::vector<WeightView>){throw CapabilityError("W8 requires explicit GPU weight bindings");}
void PrivateW8Graph::launch(MatrixView,uint16_t*,size_t,DType,std::optional<AdapterInput>){throw CapabilityError("W8 requires explicit GPU I/O bindings");}
void PrivateW8Graph::stage_device_weights(std::vector<DeviceWeightView> sources){
    check(sources.size() == 3, "W8 SwiGLU requires gate/up/down");
    const auto &s = impl_->shape;
    check(sources[0].rows == s.width && sources[0].cols == s.hidden && sources[1].rows == s.width &&
          sources[1].cols == s.hidden && sources[2].rows == s.hidden && sources[2].cols == s.width,
          "W8 full weight geometry mismatch (use regions for a physical source slice)");
    const bool comfy=impl_->basis==W8Basis::ComfyH256;
    const int up=comfy?256:128,down=comfy?256:512;const uint64_t seed=comfy?0:20260930;
    stage_device_weight_regions({{std::move(sources[0]),{0,s.width,0,s.hidden,up,seed,false,impl_->basis}},
        {std::move(sources[1]),{0,s.width,0,s.hidden,up,seed,false,impl_->basis}},
        {std::move(sources[2]),{0,s.hidden,0,s.width,down,seed,false,impl_->basis}}});
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
        p.result.headroom_start_scale=p.headroom;
        auto start=Clock::now();
        auto notify=[&p]{ {std::lock_guard lock(p.submission_mutex);p.submitted=true;}p.submission_cv.notify_all(); };
        try{check(p.verified,"W8 self-test required");p.run(input,output,adapter,notify);p.result.ok=true;}
        catch(const std::exception&e){p.result.error=e.what();}
        p.result.headroom_scale=p.headroom;
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
            x(gpu,size_t(s.rows)*s.hidden*4),y(gpu,size_t(s.rows)*s.hidden*2),
            sg(gpu,size_t(s.width)*4),su(gpu,size_t(s.width)*4),sd(gpu,size_t(s.hidden)*4);
        for(int r=0;r<s.rows;++r)for(int c=0;c<s.hidden;++c) {
            const float value=((r*3+c*7)%17-8)/8.f;
            if(p.bf16_value_boundaries)static_cast<uint16_t*>(x.value.contents)[r*s.hidden+c]=gguf::float_to_bf16_rne(value);
            else static_cast<float*>(x.value.contents)[r*s.hidden+c]=value;
        }
        for(float scale:{.125f,-.25f,.125f}){
            std::memset(g.value.contents,0,g.value.length);std::memset(u.value.contents,0,u.value.length);std::memset(d.value.contents,0,d.value.length);
            if(p.basis==W8Basis::ComfyH256) {
                auto source=[&](Buffer &codes,Buffer &scales,int rows,int cols) {
                    for(int r=0;r<rows;++r) {
                        const int index=r%cols,begin=index/256*256;
                        for(int c=0;c<256;++c)static_cast<int8_t*>(codes.value.contents)[size_t(r)*cols+begin+c]=int8_t(64*comfy_h256_sign(index%256,c));
                        static_cast<float*>(scales.value.contents)[r]=scale/1024.f;
                    }
                    auto w=codes.weight(rows,cols);w.encoding=DeviceWeightEncoding::ConvrotQ8Signed;
                    w.row_stride_bytes=cols;w.scales=scales.matrix(rows,1);return w;
                };
                p.stage({{source(g,sg,s.width,s.hidden),{0,s.width,0,s.hidden,256,0,false,p.basis}},
                    {source(u,su,s.width,s.hidden),{0,s.width,0,s.hidden,256,0,false,p.basis}},
                    {source(d,sd,s.hidden,s.width),{0,s.hidden,0,s.width,256,0,false,p.basis}}});
            } else {
                for(int r=0;r<s.width;++r){static_cast<float*>(g.value.contents)[size_t(r)*s.hidden+r%s.hidden]=scale;static_cast<float*>(u.value.contents)[size_t(r)*s.hidden+r%s.hidden]=scale;}
                for(int r=0;r<s.hidden;++r)static_cast<float*>(d.value.contents)[size_t(r)*s.width+r%s.width]=scale;
                p.stage({{g.weight(s.width,s.hidden),{0,s.width,0,s.hidden,128}},
                    {u.weight(s.width,s.hidden),{0,s.width,0,s.hidden,128}},
                    {d.weight(s.hidden,s.width),{0,s.hidden,0,s.width,512}}});
            }
            p.result={};p.run(x.matrix(s.rows,s.hidden,p.bf16_value_boundaries?DType::BF16:DType::FP32),y.matrix(s.rows,s.hidden,DType::BF16),std::nullopt);
            double diff=0,norm=0;
            for(int r=0;r<s.rows;++r)for(int c=0;c<s.hidden;++c){const int input=(c%s.width)%s.hidden;
                const float xv=p.bf16_value_boundaries?((r*3+input*7)%17-8)/8.f:static_cast<const float*>(x.value.contents)[r*s.hidden+input];const float v=xv*scale;
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
