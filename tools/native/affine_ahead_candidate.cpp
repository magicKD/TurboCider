#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <mlx/backend/metal/device.h>
#include <mlx/primitives.h>
#include "affine_ahead_candidate.hpp"
#include <chrono>
#include <atomic>
#include <optional>
#include <thread>

namespace tc::research {
namespace {
using Clock=std::chrono::steady_clock;
std::atomic<uint64_t> ahead_target_epoch{1};
struct Target {Tensor dense,status;affine_finite::Geometry geometry;int bits,group;};
MTL::ComputePipelineState *producer_pipeline(mx::Stream stream,mx::Dtype dtype,int bits,int group) {
    auto &device=mx::metal::device(stream.device);
    const std::string type=dtype==mx::float16?"half":"bfloat";
    const std::string name="tc-research-affine-ahead-v2-"+type+"-b"+std::to_string(bits)+"-g"+std::to_string(group);
    auto *library=device.get_library(name,mx::CompileOptions{},[&] {
        return "#include <metal_stdlib>\nusing namespace metal;\nusing T="+type+";\n#define BITS "+
            std::to_string(bits)+"\n#define GROUP "+std::to_string(group)+"\n"+R"metal(
            struct Params {uint words,threads,flags;};
            kernel void tc_affine_ahead(const device uint *words [[buffer(0)]],const device T *scales [[buffer(1)]],
                const device T *biases [[buffer(2)]],device T *dense [[buffer(3)]],device uint *status [[buffer(4)]],
                constant Params &p [[buffer(5)]],uint id [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
                constexpr uint PACK=32/BITS;bool bad=false;
                #pragma unroll
                for(uint b=0;b<4;++b) {
                    uint index=id*4+b;
                    if(index<p.words) {
                        uint word=words[index],meta=index*PACK/GROUP;T scale=scales[meta],bias=biases[meta];
                        #pragma unroll
                        for(uint j=0;j<PACK;++j) {
                            uchar code=uchar((word>>(j*BITS))&((1u<<BITS)-1));T value=scale*code+bias;
                            dense[index*PACK+j]=value;bad=bad || !isfinite(float(value));
                        }
                    }
                }
                uint failures=simd_sum(uint(bad));if(lane==0 && id/32<p.flags)status[id/32]=failures;
            }
        )metal";
    });
    auto *pipeline=device.get_kernel("tc_affine_ahead",library);
    require(pipeline->maxTotalThreadsPerThreadgroup()>=256,"affine ahead shader cannot use declared TG256");
    return pipeline;
}
class AheadDecode final:public mx::Primitive {
    Target target_;
  public:
    AheadDecode(mx::Stream stream,Target target):mx::Primitive(stream),target_(std::move(target)) {}
    const char *name() const override {return "TcResearchAffineAheadDecodeV1";}
    bool is_equivalent(const mx::Primitive &) const override {return false;}
    void eval_cpu(const std::vector<Tensor> &,std::vector<Tensor> &) override {throw std::invalid_argument("affine ahead needs Metal GPU");}
    void eval_gpu(const std::vector<Tensor> &inputs,std::vector<Tensor> &outputs) override {
        require(inputs.size()==3 && outputs.size()==2,"affine ahead decode arity changed");
        outputs[0].copy_shared_buffer(target_.dense);outputs[1].copy_shared_buffer(target_.status);
        auto *pipeline=producer_pipeline(stream(),outputs[0].dtype(),target_.bits,target_.group);
        auto &encoder=mx::metal::get_command_encoder(stream());encoder.set_compute_pipeline_state(pipeline);
        for(int i=0;i<3;++i)encoder.set_input_array(inputs[size_t(i)],i);
        encoder.set_output_array(outputs[0],3);encoder.set_output_array(outputs[1],4);
        struct Params{uint32_t words,threads,flags;};const auto &g=target_.geometry;
        encoder.set_bytes(Params{uint32_t(g.words),uint32_t(g.threads),uint32_t(g.flags)},5);
        encoder.dispatch_threads(MTL::Size((g.threads+255)/256*256,1,1),MTL::Size(256,1,1));
    }
};
}
struct AffineAheadWindow::Impl {
    struct Job {uint64_t id,generation;std::string ticket;affine_finite::Packed source;std::vector<Tensor> outputs;Tensor maximum;};
    MemoryLedger &ledger;size_t limit;mx::Stream stream;std::thread::id owner=std::this_thread::get_id();
    std::array<std::optional<Job>,2> jobs;Stats counters;uint64_t next=1;
    Impl(MemoryLedger &value,size_t slots):ledger(value),limit(slots),stream(mx::new_stream(mx::Device(mx::Device::gpu))) {}
    void check_owner() const {require(owner==std::this_thread::get_id(),"affine ahead requires its original owner thread");}
    void await(Job &job) {
        const auto start=Clock::now();mx::eval({job.outputs[0],job.outputs[1],job.maximum});
        require(job.outputs[0].is_available() && job.outputs[1].is_available() && job.maximum.is_available(),
            "affine ahead publication before job-specific completion");
        counters.wait_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
    }
};
AffineAheadWindow::AffineAheadWindow(MemoryLedger &ledger,size_t slots) {
    require(slots && slots<=2,"affine ahead needs one/two logical slots");impl_=std::make_unique<Impl>(ledger,slots);
}
AffineAheadWindow::~AffineAheadWindow() {try{drain();}catch(...){std::terminate();}}
uint64_t AffineAheadWindow::submit(const affine_finite::Packed &source,int bits,int group,const std::string &ticket,uint64_t generation) {
    auto &s=*impl_;s.check_owner();require(!ticket.empty() && ticket.size()<=512 && generation,"affine ahead needs immutable source ticket/generation");
    require(s.counters.pending<s.limit && s.next<UINT64_MAX,"affine ahead logical window full or ticket overflow");
    const auto g=affine_finite::geometry(source,bits,group,4);
    for(const auto &input:source)require(input.is_available() && input.data_shared_ptr() && input.offset()>=0 && input.buffer_size()>=
        uint64_t(input.offset())*input.itemsize()+input.nbytes(),"affine ahead source must already be completed/prepared");
    // Compile/qualify on the owner before any allocation or async dispatch.
    // MLX worker exceptions are not a usable early capability fallback.
    (void)producer_pipeline(s.stream,source[1].dtype(),bits,group);
    auto allocate=[&](uint64_t bytes,uint64_t upper,const mx::Shape &shape,mx::Dtype dtype) {
        const auto epoch=ahead_target_epoch.fetch_add(1,std::memory_order_relaxed);
        require(epoch && epoch<UINT64_MAX,"affine ahead allocation epoch exhausted");
        // A source may keep generation1 across many immutable reads. Fresh
        // target identities need their OWN allocation epoch, including a
        // recycled allocator handle and independently retiring callbacks.
        return streaming::gguf_storage::allocate(s.ledger,bytes,upper,shape,dtype,MemoryClass::ConversionScratch,epoch);
    };
    Target target{allocate(g.dense_bytes,g.dense_upper,{g.rows,g.columns},source[1].dtype()),
        allocate(g.status_bytes,g.status_upper,{g.flags},mx::uint32),g,bits,group};
    require(target.dense.buffer().ptr()!=target.status.buffer().ptr(),"affine ahead target/target alias");
    for(const auto &input:source)require(input.buffer().ptr()!=target.dense.buffer().ptr() && input.buffer().ptr()!=target.status.buffer().ptr(),
        "affine ahead source/target alias");
    auto outputs=Tensor::make_arrays({{g.rows,g.columns},{g.flags}},{source[1].dtype(),mx::uint32},
        std::make_shared<AheadDecode>(s.stream,target),{source[0],source[1],source[2]});
    auto maximum=mx::max(outputs[1],{},false,s.stream);
    const uint64_t id=s.next++;
    auto index=size_t(0);while(index<s.limit && s.jobs[index])++index;
    require(index<s.limit,"affine ahead pending count/slots disagree");
    // Store every owner before scheduling; a failed submit still drains its
    // possibly scheduled outputs. Targets are fresh, never a recycled view.
    s.jobs[index].emplace(Impl::Job{id,generation,ticket,source,std::move(outputs),maximum});
    ++s.counters.pending;s.counters.peak_pending=std::max(s.counters.peak_pending,s.counters.pending);
    try{mx::async_eval({s.jobs[index]->outputs[0],s.jobs[index]->outputs[1],s.jobs[index]->maximum});++s.counters.submitted;}
    catch(...){drain();throw;}
    return id;
}
Tensor AffineAheadWindow::take(uint64_t id) {
    auto &s=*impl_;s.check_owner();size_t index=0;while(index<s.limit && (!s.jobs[index] || s.jobs[index]->id!=id))++index;
    require(index<s.limit,"affine ahead unknown/already-consumed ticket");auto &job=*s.jobs[index];
    try{s.await(job);}catch(...){drain();throw;}
    const bool finite=job.maximum.item<uint32_t>()==0;
    // A failed operation publishes nothing. Retire command-buffer owners as
    // well as the per-array event before rolling back its backing claims.
    if(!finite)mx::synchronize(s.stream);
    for(auto &output:job.outputs){output.detach();output.set_siblings({},0);}
    auto dense=job.outputs[0];s.jobs[index].reset();--s.counters.pending;
    if(!finite){++s.counters.failed;throw std::runtime_error("affine ahead rejects nonfinite decoded coefficients");}
    ++s.counters.published;return dense;
}
void AffineAheadWindow::drain() {
    auto &s=*impl_;s.check_owner();
    for(auto &job:s.jobs)if(job)s.await(*job);
    mx::synchronize(s.stream);
    for(auto &job:s.jobs)if(job){for(auto &output:job->outputs){output.detach();output.set_siblings({},0);}
        job.reset();--s.counters.pending;++s.counters.drained;}
}
AffineAheadWindow::Stats AffineAheadWindow::stats() const {impl_->check_owner();return impl_->counters;}
} // namespace tc::research
