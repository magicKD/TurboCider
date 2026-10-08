#include "ane_public_w8.hpp"
#include "ane_gpu.hpp"
#include "ane_memory.hpp"
#include "ane_w8a8_math.hpp"
#include "ane_backend.hpp"
#include "ane_weight_identity.hpp"
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#include <dispatch/dispatch.h>
#include <map>
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
using gpu::Surface;
using gpu::Element;
double elapsed(Clock::time_point x) { return std::chrono::duration<double>(Clock::now() - x).count(); }
void check(bool ok, const char *reason) { if (!ok) throw std::runtime_error(reason); }
// Public API binding over the SAME shared GPU surfaces used by Private.
struct CoreArray {
    gpu::Surface owner;
    CVPixelBufferRef pixel=nullptr;
    MLMultiArray *array=nil;
    explicit CoreArray(gpu::Surface value):owner(std::move(value)) {
        check(!owner.is_view(),"Public Core ML bindings require whole surfaces");
        check(CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault,(IOSurfaceRef)owner.native_iosurface(),nullptr,&pixel)==kCVReturnSuccess && pixel,
              "Public Core ML cannot wrap shared IOSurface");
        array=[[MLMultiArray alloc] initWithPixelBuffer:pixel shape:@[@(owner.rows()),@(owner.columns())]];
        if(!array || array.dataType!=(owner.element()==gpu::Element::I8?MLMultiArrayDataTypeInt8:MLMultiArrayDataTypeFloat16)) {
            CVPixelBufferRelease(pixel);pixel=nullptr;throw CapabilityError("Public Core ML shared surface dtype mismatch");
        }
    }
    ~CoreArray(){array=nil;if(pixel)CVPixelBufferRelease(pixel);}
};
struct PublicRequest {
    std::function<gpu::Completion()> invoke;
    gpu::Completion finish(std::chrono::milliseconds = std::chrono::seconds(30)) {
        check(bool(invoke),"Public Core ML request already consumed");
        auto call=std::exchange(invoke,{});return call();
    }
};
class PublicProgram {
    RuntimeArtifactSnapshot artifact_;
    gpu::Device device_;
    gpu::Surface headroom_;
    MLModel *model_=nil;
    std::map<std::pair<std::string,void*>,std::unique_ptr<CoreArray>> arrays_;
    CoreArray &bind(const std::string &name,const gpu::Surface &surface,bool output=false) {
        const auto key=std::pair{name,surface.native_iosurface()};
        auto at=arrays_.find(key);
        if(at==arrays_.end()) {
            check(arrays_.size()<24,"Public Core ML binding cache exceeded fixed banks/slots");
            auto value=std::make_unique<CoreArray>(surface);
            auto descriptions=output?model_.modelDescription.outputDescriptionsByName:model_.modelDescription.inputDescriptionsByName;
            auto expected=descriptions[@(name.c_str())].multiArrayConstraint;
            check(expected && expected.dataType==value->array.dataType && [expected.shape isEqual:value->array.shape],
                "Public Core ML compiled feature differs from shared W8 ABI");
            at=arrays_.emplace(key,std::move(value)).first;
        }
        return *at->second;
    }
  public:
    PublicProgram(gpu::Device device,RuntimeArtifactSnapshot artifact):artifact_(std::move(artifact)),device_(std::move(device)),
        headroom_(device_,1,1,gpu::Element::FP16) {
        MLModelConfiguration *config=[MLModelConfiguration new];config.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
        NSError *error=nil;
        model_=[MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(artifact_.compiled_model.c_str())] configuration:config error:&error];
        if(!model_)throw CapabilityError(std::string("Public W8 model load failed: ")+(error?error.localizedDescription.UTF8String:"unknown"));
        check(model_.modelDescription.inputDescriptionsByName.count==(artifact_.shape.lora_inputs?10u:8u) &&
              model_.modelDescription.outputDescriptionsByName.count==1,"Public W8 feature count mismatch");
        set_headroom(1.f);
    }
    ~PublicProgram(){arrays_.clear();model_=nil;}
    size_t extra_bytes()const{return headroom_.bytes();}
    void set_headroom(float value) {
        check(std::isfinite(value) && value>=1 && value<=4096 && std::log2(value)==std::floor(std::log2(value)),
              "Public W8 headroom invalid");
        *static_cast<uint16_t*>(headroom_.data())=gguf::float_to_fp16_rne(value);
    }
    PublicRequest enqueue(std::span<const std::pair<std::string,gpu::Surface>> inputs,
        std::span<const std::pair<std::string,gpu::Surface>> outputs,uint64_t ready,uint64_t done,std::function<void()> failure) {
        check(outputs.size()==1 && outputs[0].first=="y","Public W8 output ABI mismatch");
        NSMutableDictionary *values=[NSMutableDictionary dictionary];
        for(const auto &[name,surface]:inputs)values[@(name.c_str())]=[MLFeatureValue featureValueWithMultiArray:bind(name,surface).array];
        values[@"headroom"]=[MLFeatureValue featureValueWithMultiArray:bind("headroom",headroom_).array];
        auto &target=bind("y",outputs[0].second,true);
        NSError *error=nil;
        MLDictionaryFeatureProvider *features=[[MLDictionaryFeatureProvider alloc] initWithDictionary:values error:&error];
        check(features!=nil,"Public W8 feature binding failed");
        MLPredictionOptions *options=[MLPredictionOptions new];options.outputBackings=@{@"y":target.array};
        // Do not enqueue a Metal ready-wait BEHIND the output CB waiting done:
        // Core ML runs synchronously on this persistent executor worker. A
        // public shared-event listener meets its dependency without that cycle.
        return {[this,features,options,target_ptr=&target,ready,done,failure=std::move(failure)] {
            try {
              @try {
                auto event=(__bridge id<MTLSharedEvent>)device_.shared_event();
                struct Ready {std::mutex mutex;std::condition_variable cv;bool signaled=false;};
                auto state=std::make_shared<Ready>();
                if(event.signaledValue<ready) {
                    MTLSharedEventListener *listener=[[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_get_global_queue(QOS_CLASS_DEFAULT,0)];
                    [event notifyListener:listener atValue:ready block:^(id<MTLSharedEvent>,uint64_t) {
                        {std::lock_guard lock(state->mutex);state->signaled=true;}state->cv.notify_all();
                    }];
                    std::unique_lock lock(state->mutex);
                    if(!state->cv.wait_for(lock,std::chrono::seconds(30),[&]{return state->signaled;}))
                        throw CapabilityError("Public W8 GPU producer deadline exceeded");
                }
                NSError *error=nil;
                id<MLFeatureProvider> prediction=[model_ predictionFromFeatures:features options:options error:&error];
                if(!prediction)throw CapabilityError(std::string("Public W8 prediction failed: ")+(error?error.localizedDescription.UTF8String:"unknown"));
                MLMultiArray *result=[prediction featureValueForName:@"y"].multiArrayValue;
                check(result && result.dataType==MLMultiArrayDataTypeFloat16 && [result.shape isEqual:target_ptr->array.shape],
                      "Public W8 prediction output ABI changed");
                if(result.pixelBuffer!=target_ptr->pixel) {
                    const size_t rs=result.strides[0].unsignedLongLongValue,cs=result.strides[1].unsignedLongLongValue;
                    check(rs && cs,"Public W8 output stride invalid");__block bool copied=false;
                    [result getBytesWithHandler:^(const void *bytes,NSInteger size) {
                        const auto &surface=target_ptr->owner;
                        const size_t extent=size_t(surface.rows()-1)*rs+size_t(surface.columns()-1)*cs+1;
                        if(!bytes || size<0 || size_t(size)/2<extent)return;
                        for(uint32_t row=0;row<surface.rows();++row) {
                            auto *dst=reinterpret_cast<uint16_t*>(static_cast<char*>(surface.data())+row*surface.pitch());
                            for(uint32_t col=0;col<surface.columns();++col)dst[col]=static_cast<const uint16_t*>(bytes)[row*rs+col*cs];
                        }
                        copied=true;
                    }];
                    check(copied,"Public W8 output backing copy failed");
                }
                device_.release_after_failure(done); // Core ML API completion, not physical overlap evidence
                return gpu::Completion{true,false,{}};
              } @catch(NSException *exception) {
                throw CapabilityError(std::string("Public W8 Objective-C prediction exception: ")+
                    (exception.reason.UTF8String?:"unknown"));
              }
            }catch(const std::exception &error) {
                if(failure)failure();device_.release_after_failure(done);
                return gpu::Completion{false,false,error.what()};
            }
        }};
    }
};
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
struct PublicW8Graph::Impl {
    GraphShape shape;
    W8Basis basis = W8Basis::SylvesterDH;
    int activation_group_size = 0;
    int hidden_group_size = 0;
    bool bf16_value_boundaries = false;
    gpu::Device device{true,true};
    std::unique_ptr<PublicProgram> program;
    RuntimeArtifactSnapshot artifact;
    // Exactly two source-independent banks, reused across every layer/step.
    struct Bank {
        Surface g, sg, u, su, d, sd;
        bool ready = false;
        Bank(gpu::Device &dev, const GraphShape &s) : g(dev,s.width,s.hidden,Element::I8), sg(dev,s.width,1,Element::FP16),
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
    void build(float scale) { program->set_headroom(scale);headroom=scale; }
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
        std::array<std::optional<gpu::QuantStage>,3> producers;
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
                output.dtype == DType::FP32), "W8 launch geometry/slots unavailable");
        check(!adapter || (s.lora_inputs && adapter->gate.rows == input.rows && adapter->up.rows == input.rows && adapter->hidden.rows == input.rows &&
            adapter->gate.cols == s.width && adapter->up.cols == s.width && adapter->hidden.cols == s.width &&
            (output.dtype==DType::FP32 ? (adapter->hidden.dtype==DType::BF16 || adapter->hidden.dtype==DType::FP16) :
                adapter->hidden.dtype==output.dtype)), "W8 LoRA geometry mismatch");
        check(!bf16_value_boundaries || input.dtype==DType::BF16,"BF16 value-boundary recipe requires original BF16 activation input");
        check(input.buffer != output.buffer && (!adapter || (adapter->hidden.buffer != output.buffer && adapter->hidden.buffer != input.buffer &&
            adapter->gate.buffer != output.buffer && adapter->up.buffer != output.buffer)), "W8 source/output aliases");
        if (s.lora_inputs && !adapter) for (auto *slot : {dg.get(),du.get()}) std::memset(slot->data(),0,slot->rows()*slot->pitch());
        auto &b = *banks[current];
        std::array<std::optional<gpu::QuantStage>, 2> activations;
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
                std::vector<gpu::Upload> uploads;
                if (s.lora_inputs) { inputs.emplace_back("dg",*dg); inputs.emplace_back("du",*du);
                    if (adapter) { uploads.push_back({adapter->gate,*dg,row}); uploads.push_back({adapter->up,*du,row}); } }
                auto norm = y->slice_rows(0,s.hidden), hs = y->slice_rows(s.hidden,1);
                std::vector<gpu::Download> downloads{{norm,output,row,output.dtype,headroom,b.sd,hs}};
                if(basis==W8Basis::ComfyH256)downloads[0].row_scale_policy=gpu::RowScalePolicy::SignedFinite;
                if (s.lora_inputs) downloads.push_back({y->slice_rows(s.hidden+1,s.width),adapter ? std::optional<DeviceMatrixView>(adapter->hidden) : std::nullopt,
                    row,adapter?adapter->hidden.dtype:output.dtype==DType::FP32?input.dtype:output.dtype,headroom});
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
                if (!eval.ok || !copied.ok) { disabled=true; throw CapabilityError("Public W8 evaluation/GPU epilogue failed"); }
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
PublicW8Graph::PublicW8Graph(const std::filesystem::path &manifest,size_t budget,std::optional<GraphGeometry> expected)
    :impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        auto start=Clock::now();auto &p=*impl_;p.artifact=snapshot_runtime_w8a8(manifest);p.shape=p.artifact.shape;p.basis=p.artifact.basis;
        const auto &shape=p.shape;
        check(shape.kind==Kind::SwiGLU && shape.rows>0 && shape.rows<=4224 && shape.hidden>0 && shape.hidden<=4096 &&
              shape.hidden%128==0 && shape.width>0 && shape.width<=16384 && shape.width%512==0 &&
              (p.basis!=W8Basis::ComfyH256 || shape.hidden%256==0),"Public W8 geometry unsupported");
        if(expected)check(expected->kind==shape.kind && expected->hidden==shape.hidden && expected->width==shape.width &&
                          (!expected->require_lora_inputs || shape.lora_inputs),"Public W8 model geometry/LoRA ABI mismatch");
        const int rotation_block=p.basis==W8Basis::ComfyH256?256:512;
        p.estimate=uint64_t(shape.hidden)*shape.width*6+uint64_t(shape.rows)*(8ull*shape.width+12ull*shape.hidden)+
                   uint64_t(shape.width)*rotation_block*4+(128ull<<20)+gpu::scale_cache_budget_bytes+
                   p.device.weight_code_cache_budget_bytes();
        if(p.estimate>budget || !admit_memory(observe_runtime_memory(0),{uint64_t(4)<<30,budget},0,p.estimate).allowed())
            throw MemoryBudgetError("Public W8 graph/banks/system memory admission denied");
        for(auto &bank:p.banks){bank=std::make_unique<Impl::Bank>(p.device,shape);p.allocated+=bank->bytes();}
        auto add=[&](std::unique_ptr<Surface> &slot,int rows,int cols,Element element){
            slot=std::make_unique<Surface>(p.device,rows,cols,element);p.allocated+=slot->bytes();
        };
        add(p.x[0],shape.hidden,shape.rows,Element::I8);add(p.tx[0],1,shape.rows,Element::FP16);
        add(p.y,shape.hidden+1+(shape.lora_inputs?shape.width:0),shape.rows,Element::FP16);
        if(shape.lora_inputs){add(p.dg,shape.width,shape.rows,Element::FP16);add(p.du,shape.width,shape.rows,Element::FP16);}
        p.program=std::make_unique<PublicProgram>(p.device,p.artifact);p.allocated+=p.program->extra_bytes();
        if(p.allocated>budget || p.allocated>p.estimate)throw MemoryBudgetError("Public W8 actual surfaces exceed admission");
        p.launch_fence=true;p.load_time=elapsed(start);
    }
}
PublicW8Graph::~PublicW8Graph()=default;
const GraphShape &PublicW8Graph::shape() const{return impl_->shape;}
size_t PublicW8Graph::slot_bytes() const{return impl_->allocated;}
size_t PublicW8Graph::estimated_bytes() const{return impl_->estimate;}
double PublicW8Graph::load_seconds() const{return impl_->load_time;}
std::string PublicW8Graph::weight_recipe() const {
    if(impl_->bf16_value_boundaries)return convrot_bf16_value_recipe;
    if(impl_->activation_group_size && impl_->hidden_group_size)return convrot_group_w8a8_recipe;
    if(impl_->activation_group_size)return convrot_input_group_w8a8_recipe;
    if(impl_->hidden_group_size)return convrot_hidden_group_w8a8_recipe;
    return std::string(impl_->basis==W8Basis::ComfyH256?convrot_w8a8_recipe:w8a8_recipe)+"+public-int8-io-v1";
}
std::string PublicW8Graph::data_path() const{return impl_->basis==W8Basis::ComfyH256?"w8a8_convrot":"w8a8_hadamard";}
WeightCacheStats PublicW8Graph::weight_cache_stats() const{return impl_->device.scale_cache_stats();}
WeightCodeCacheReport PublicW8Graph::weight_code_cache_stats() const{return impl_->device.weight_code_cache_stats();}
StagePipelineStats PublicW8Graph::stage_pipeline_stats() const{return impl_->device.stage_pipeline_stats();}
bool PublicW8Graph::device_submission_fence_enabled() const{return impl_->launch_fence;}
bool PublicW8Graph::activation_lookahead_enabled() const{return impl_->a8_lookahead;}
bool PublicW8Graph::supports_fp32_device_output() const{return true;}
int PublicW8Graph::activation_group_size() const{return impl_->activation_group_size;}
int PublicW8Graph::hidden_activation_group_size() const{return impl_->hidden_group_size;}
void PublicW8Graph::stage_weights(std::vector<WeightView>){throw CapabilityError("W8 requires explicit GPU weight bindings");}
void PublicW8Graph::launch(MatrixView,uint16_t*,size_t,DType,std::optional<AdapterInput>){throw CapabilityError("W8 requires explicit GPU I/O bindings");}
void PublicW8Graph::stage_device_weights(std::vector<DeviceWeightView> sources){
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
void PublicW8Graph::stage_device_weight_regions(std::vector<DeviceWeightRegion> sources){
    discard_prefetched_weights();
    auto &p=*impl_;p.worker.join();check(p.verified&&!p.disabled,"W8 self-test required/executor disabled");p.result={};
    if(p.current>=0)p.banks[p.current]->ready=false;
    p.worker.submit([&p,sources=std::move(sources)]{auto start=Clock::now();try{p.stage(sources);p.result.ok=true;}catch(const std::exception&e){p.result.error=e.what();}p.result.stage_seconds=elapsed(start);});
}
void PublicW8Graph::prefetch_device_weight_regions(std::vector<DeviceWeightRegion> sources) {
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
std::optional<RunResult> PublicW8Graph::activate_prefetched_weights(std::span<const DeviceWeightRegion> expected) {
    auto &p=*impl_;
    if(!p.future_pending || expected.size()!=p.future_sources.size() ||
       !std::equal(expected.begin(),expected.end(),p.future_sources.begin(),same_weight_region))return std::nullopt;
    // Both reuse fences: current ANE/epilogue consumer and future GPU producer.
    p.worker.join();p.staging_worker.join();
    if(p.current>=0)p.banks[p.current]->ready=false;
    p.result=p.future_result;
    if(p.result.ok)p.current=p.future_bank;else p.current=-1;
    p.future_pending=false;p.future_bank=-1;p.future_sources.clear();
    return p.result;
}
void PublicW8Graph::discard_prefetched_weights() {
    auto &p=*impl_;
    if(!p.future_pending)return;
    p.staging_worker.join();
    p.banks[p.future_bank]->ready=false;p.future_pending=false;p.future_bank=-1;p.future_sources.clear();
}
RunResult PublicW8Graph::wait_stage(){return finish();}
void PublicW8Graph::launch_device(DeviceMatrixView input,DeviceMatrixView output,std::optional<DeviceAdapterInput> adapter){
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
RunResult PublicW8Graph::finish(){auto&p=*impl_;try{p.worker.join();}catch(const std::exception&e){p.result.ok=false;p.result.error=e.what();}return p.result;}
bool PublicW8Graph::self_test(std::string &error){
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
