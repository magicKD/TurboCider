#include "native_prefill_ffn.hpp"
#include "native_linear.hpp"
#include <algorithm>

namespace tc::metal_graph {
namespace {
struct Rows { uint32_t valid, bucket, columns, input_stride, output_stride, pad; };
static_assert(sizeof(Rows)==24);
constexpr std::string_view row_source=R"metal(
#include <metal_stdlib>
using namespace metal;
struct Rows { uint valid,bucket,columns,input_stride,output_stride,pad; };
kernel void prefill_rows(device const ushort* x [[buffer(0)]],device ushort* y [[buffer(1)]],constant Rows& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint row=i/p.columns,col=i%p.columns;
    if(row>=(p.pad?p.bucket:p.valid))return;
    // ushort preserves the exact BF16 bit pattern, including signed zero.
    y[ulong(row)*p.output_stride+col]=(p.pad && row>=p.valid)?ushort(0):x[ulong(row)*p.input_stride+col];
}
)metal";
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
MatrixView matrix(Backend& b,uint32_t rows,uint32_t cols,ScalarType type,const char* label){
    const size_t bytes=size_t(rows)*cols*scalar_bytes(type);auto a=b.allocate(bytes,label,StorageMode::Private);
    return {a.view(0,bytes,type),rows,cols,size_t(cols)*scalar_bytes(type)};
}
void validate(const MatrixView& m,uint32_t rows,uint32_t cols,ScalarType type){
    (void)dense_linear::validate_layout({m.rows,m.columns,m.row_stride_bytes,m.storage.size_bytes(),m.storage.offset_bytes(),m.storage.scalar_type()});
    require(m.rows==rows && m.columns==cols && m.storage.scalar_type()==type,"prefill FFN shape/type mismatch");
}
MatrixView subrows(const MatrixView& m,uint32_t first,uint32_t count){
    require(first<m.rows && count && count<=m.rows-first,"prefill FFN row view outside source");
    const size_t bytes=size_t(count-1)*m.row_stride_bytes+size_t(m.columns)*scalar_bytes(m.storage.scalar_type());
    return {m.storage.buffer().view(m.storage.offset_bytes()+size_t(first)*m.row_stride_bytes,bytes,m.storage.scalar_type()),count,m.columns,m.row_stride_bytes};
}
}
struct NativePrefillFfn::Impl {
    Backend* backend;
    NativeHybridFfnOptions options;
    NativeHybridFfn hybrid;
    Pipeline row_copy;
    MatrixView input, hidden, gpu, ane, joined;
    NativePrefillFfnStats stats;
    Impl(Backend& b,NativeHybridFfnOptions o):backend(&b),options(o),hybrid(NativeHybridFfn::prepare(b,std::move(o))){
        auto lib=b.compile_library(row_source,"native-prefill-zero-pad-crop");row_copy=b.prepare_pipeline(lib,"prefill_rows");
        input=matrix(b,options.rows,options.hidden,ScalarType::BF16,"prefill-padded-input");
        hidden=matrix(b,options.rows,options.width-options.ane_channels,ScalarType::BF16,"prefill-gpu-hidden");
        gpu=matrix(b,options.rows,options.hidden,ScalarType::FP32,"prefill-gpu-partial");
        ane=matrix(b,options.rows,options.hidden,ScalarType::FP32,"prefill-ane-partial");
        joined=matrix(b,options.rows,options.hidden,ScalarType::BF16,"prefill-joined-output");
    }
    void copy(CommandGraph& g,const MatrixView& x,const MatrixView& y,uint32_t valid,bool pad){
        const Rows p{valid,options.rows,options.hidden,uint32_t(x.row_stride_bytes/2),uint32_t(y.row_stride_bytes/2),uint32_t(pad)};
        g.dispatch(row_copy,{x.storage,y.storage},p,{(size_t(pad?options.rows:valid)*options.hidden+255)/256,1,1},{256,1,1});
    }
};
std::unique_ptr<NativePrefillFfn> NativePrefillFfn::prepare(Backend& b,NativeHybridFfnOptions o){
    NativeHybridFfn::validate_options(o);
    require(o.weight_scale_cache_entries==0,"prefill source lifetime is layer-local; scale cache requires a separate qualified epoch");
    return std::unique_ptr<NativePrefillFfn>(new NativePrefillFfn(std::make_unique<Impl>(b,std::move(o))));
}
NativePrefillFfn::NativePrefillFfn(std::unique_ptr<Impl> p):impl_(std::move(p)){}
NativePrefillFfn::~NativePrefillFfn()=default;
NativePrefillFfnStats NativePrefillFfn::statistics() const{return impl_->stats;}
uint32_t NativePrefillFfn::bucket_rows() const{return impl_->options.rows;}
bool NativePrefillFfn::healthy() const {const auto s=impl_->hybrid.status();return impl_->backend->healthy() && !s.retired && !s.running && s.all_work_physically_complete;}
void NativePrefillFfn::reset_statistics(){require(healthy(),"cannot reuse unhealthy native prefill FFN");impl_->stats={};}
void NativePrefillFfn::execute(MatrixView x,MatrixView gate,MatrixView up,MatrixView down,MatrixView out,CommandGraph before,CommandGraph after,std::chrono::milliseconds timeout,const std::function<void()>& concurrent_prepare){
    auto& p=*impl_;const auto& o=p.options;
    require(healthy() && x.rows && x.rows<=16384 && timeout.count()>0 && timeout<=std::chrono::minutes(5),"native prefill FFN invalid state/row count/deadline");
    validate(x,x.rows,o.hidden,ScalarType::BF16);validate(out,x.rows,o.hidden,ScalarType::BF16);
    validate(gate,o.width,o.hidden,ScalarType::BF16);validate(up,o.width,o.hidden,ScalarType::BF16);validate(down,o.hidden,o.width,ScalarType::BF16);
    // Conservatively reject shared backing: no caller may overwrite unread
    // later chunks, checkpoint weights or the persistent scratch bank.
    for(const auto& source:{x,gate,up,down})require(source.storage.buffer().native_handle()!=out.storage.buffer().native_handle(),"native prefill output aliases a source");
    const auto start=std::chrono::steady_clock::now(),deadline=start+timeout;
    for(uint32_t first=0;first<x.rows;first+=o.rows){const uint32_t count=std::min(o.rows,x.rows-first);
        NativeHybridFfnPlan plan;
        if(first==0)plan.before_ffn.append_graph(std::move(before));
        p.copy(plan.before_ffn,subrows(x,first,count),p.input,count,true);
        plan.input=p.input;plan.gate=gate;plan.up=up;plan.down=down;
        plan.gpu_partial=p.gpu;plan.ane_partial=p.ane;plan.published_output=p.joined;
        const auto& gpu=p.hybrid.gpu_complement();gpu.encode(plan.gpu_ffn,{p.input,gate,up,down,p.hidden,p.gpu,0});
        gpu.encode_join_partials(plan.after_join,p.gpu,p.ane,p.joined);
        p.copy(plan.after_join,p.joined,subrows(out,first,count),count,false);
        if(first+count==x.rows)plan.after_join.append_graph(std::move(after));
        const auto result=p.hybrid.execute_until(std::move(plan),deadline,first+count==x.rows?concurrent_prepare:std::function<void()>{});
        if(!result.completion.ok || !result.completion.all_work_physically_complete)
            throw std::runtime_error("native prefill FFN retired: "+result.completion.error);
        ++p.stats.chunks;p.stats.valid_rows+=count;p.stats.padded_rows+=o.rows-count;
        p.stats.producer_seconds+=result.producer_seconds;p.stats.branch_seconds+=result.branch_join_seconds;p.stats.consumer_seconds+=result.consumer_seconds;
    }
    ++p.stats.calls;p.stats.total_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
} // namespace tc::metal_graph
