#include "native_prefill.hpp"
#if TURBOCIDER_ENABLE_PRIVATE_ANE
#include "../../backends/metal_graph/native_prefill_ffn.hpp"
#include "../../backends/metal_graph/mlx_bridge.hpp"
namespace tc::qwen21 {
struct NativePrefill::Impl {
    metal_graph::Backend backend;
    std::unique_ptr<metal_graph::NativePrefillFfn> ffn;
    std::filesystem::path cache;
    double cold_seconds=0;
    NativePrefillExecutionMetrics metrics;
    explicit Impl(const std::filesystem::path& path):cache(path){
        auto start=Clock::now();metal_graph::NativeHybridFfnOptions o;o.rows=2688;o.program_cache=path;o.register_weight_staging=true;
        ffn=metal_graph::NativePrefillFfn::prepare(backend,std::move(o));cold_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    }
    metal_graph::MatrixView view(const Tensor& t){auto v=metal_graph::import_mlx_matrix(backend,t);return {v.storage,v.rows,v.columns,v.row_stride_bytes};}
};
NativePrefill::NativePrefill(const std::filesystem::path& cache):impl_(std::make_unique<Impl>(cache)){}
NativePrefill::~NativePrefill()=default;
bool NativePrefill::healthy() const{return impl_->ffn->healthy();}
const std::filesystem::path& NativePrefill::cache() const{return impl_->cache;}
double NativePrefill::cold_preparation_seconds() const{return impl_->cold_seconds;}
void NativePrefill::begin_request(bool prepared){require(healthy(),"native prefill Session resources retired; restart before reuse");impl_->ffn->reset_statistics();impl_->metrics={};
    impl_->metrics.preparation_reused=prepared;impl_->metrics.preparation_seconds=prepared?0:impl_->cold_seconds;impl_->metrics.prewarm_seconds=prepared?impl_->cold_seconds:0;
}
Tensor NativePrefill::execute(const Tensor& x,const Tensor& gu,const Tensor& down,std::atomic<bool>& stop){
    checkpoint(stop);require(healthy() && gu.shape()==mx::Shape{24576,4096} && down.shape()==mx::Shape{4096,12288} && x.ndim()==3 && x.shape(0)==1 && x.shape(2)==4096,"native prefill FFN source/sequence geometry mismatch");
    auto gate=slice_axis(gu,0,0,12288),up=slice_axis(gu,0,12288,24576);auto start=Clock::now();mx::eval(x,gate,up,down);mx::synchronize();impl_->metrics.input_ready_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
    auto out=metal_graph::allocate_mlx_matrix(x.shape(1),4096,metal_graph::ScalarType::BF16,true);
    impl_->ffn->execute(impl_->view(x),impl_->view(gate),impl_->view(up),impl_->view(down),impl_->view(out));checkpoint(stop);return out;
}
NativePrefillExecutionMetrics NativePrefill::metrics() const {auto m=impl_->metrics;const auto s=impl_->ffn->statistics();
    m.calls=s.calls;m.chunks=s.chunks;m.valid_rows=s.valid_rows;m.padded_rows=s.padded_rows;m.bucket_rows=impl_->ffn->bucket_rows();
    m.producer_seconds=s.producer_seconds;m.branch_seconds=s.branch_seconds;m.consumer_seconds=s.consumer_seconds;m.total_seconds=s.total_seconds;m.physically_complete=healthy();return m;
}
}
#else
namespace tc::qwen21 {
struct NativePrefill::Impl {};
NativePrefill::NativePrefill(const std::filesystem::path&){throw std::runtime_error("native prefill ANE requires a private-enabled build");}
NativePrefill::~NativePrefill()=default;
bool NativePrefill::healthy() const{return false;}
const std::filesystem::path& NativePrefill::cache() const{static const std::filesystem::path empty;return empty;}
double NativePrefill::cold_preparation_seconds() const{return 0;}
void NativePrefill::begin_request(bool){throw std::runtime_error("native prefill ANE unavailable");}
Tensor NativePrefill::execute(const Tensor&,const Tensor&,const Tensor&,std::atomic<bool>&){throw std::runtime_error("native prefill ANE unavailable");}
NativePrefillExecutionMetrics NativePrefill::metrics() const{return {};}
}
#endif
