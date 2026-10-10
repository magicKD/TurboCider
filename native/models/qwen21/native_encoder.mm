#include "native_encoder.hpp"
#include "../../backends/metal_graph/native_attention.hpp"
#include "../../backends/metal_graph/native_linear.hpp"
#include "../../backends/metal_graph/mlx_bridge.hpp"
#include "../../platform/apple/platform.hpp"
#include "../../runtime/memory_accounting.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#if TURBOCIDER_ENABLE_PRIVATE_ANE
#include "../../backends/metal_graph/native_prefill_ffn.hpp"
#endif

namespace tc::qwen21 {
namespace {
using namespace metal_graph;
// Explicit, copied POD ABI shared by encoder pointwise kernels.
struct P { uint32_t rows=0,columns=0,stride=0,out_stride=0,heads=0,dim=0,padded=0;
    uint32_t xf32=0,wf32=0,bf32=0,rms=0,rope=0,delta=0,gh=0,gw=0,side=0;
    float epsilon=1e-6f; };
static_assert(sizeof(P)==68);
constexpr std::string_view kernels=R"metal(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct EP { uint rows,columns,stride,out_stride,heads,dim,padded,xf32,wf32,bf32,rms,rope,delta,gh,gw,side; float epsilon; };
inline float rd(device const uchar* a,ulong i,uint f) { return f?reinterpret_cast<device const float*>(a)[i]:float(reinterpret_cast<device const bfloat*>(a)[i]); }
inline void wr(device uchar* a,ulong i,float x,uint f) { if(f)reinterpret_cast<device float*>(a)[i]=x;else reinterpret_cast<device bfloat*>(a)[i]=bfloat(x); }
kernel void en_norm(device const uchar* x [[buffer(0)]],device const uchar* w [[buffer(1)]],device const uchar* b [[buffer(2)]],device uchar* y [[buffer(3)]],constant EP& p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    threadgroup float shared[8];float sum=0;
    for(uint c=t;c<p.columns;c+=256){float v=rd(x,ulong(row)*p.stride+c,p.xf32);sum+=p.rms?v*v:v;}
    float s=simd_sum(sum);if((t&31)==0)shared[t/32]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=0;for(uint i=0;i<8;++i)mean+=shared[i];mean/=p.columns;
    float inv;
    if(p.rms)inv=rsqrt(mean+p.epsilon);
    else {
        threadgroup_barrier(mem_flags::mem_threadgroup);float sq=0;
        for(uint c=t;c<p.columns;c+=256){float v=rd(x,ulong(row)*p.stride+c,p.xf32)-mean;sq+=v*v;}
        float ss=simd_sum(sq);if((t&31)==0)shared[t/32]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);
        float variance=0;for(uint i=0;i<8;++i)variance+=shared[i];inv=rsqrt(variance/p.columns+p.epsilon);
    }
    for(uint c=t;c<p.columns;c+=256){float v=rd(x,ulong(row)*p.stride+c,p.xf32);float r=(p.rms?v:v-mean)*inv*rd(w,c,p.wf32);if(!p.rms)r+=rd(b,c,p.bf32);wr(y,ulong(row)*p.out_stride+c,r,p.xf32);}
}
kernel void en_pack(device const uchar* x [[buffer(0)]],device const uchar* w [[buffer(1)]],device const float* trig [[buffer(2)]],device uchar* y [[buffer(3)]],constant EP& p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    uint row=group.x,head=group.y;threadgroup float shared[8];float sum=0;
    if(p.rms){for(uint c=t;c<p.dim;c+=256){float v=rd(x,ulong(row)*p.stride+head*p.dim+c,p.xf32);sum+=v*v;}}
    float s=simd_sum(sum);if((t&31)==0)shared[t/32]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(uint i=0;i<8;++i)total+=shared[i];float inv=rsqrt(total/p.dim+p.epsilon);
    for(uint c=t;c<p.padded;c+=256){
        float v=0;
        if(c<p.dim){v=rd(x,ulong(row)*p.stride+head*p.dim+c,p.xf32);
            if(p.rms){v=(v*inv)*rd(w,c,p.wf32);if(!p.xf32)v=float(bfloat(v));}
            if(p.rope){uint other=c<p.dim/2?c+p.dim/2:c-p.dim/2;float z=rd(x,ulong(row)*p.stride+head*p.dim+other,p.xf32);
                if(p.rms){z=(z*inv)*rd(w,other,p.wf32);if(!p.xf32)z=float(bfloat(z));}
                float co=trig[ulong(row)*p.dim*2+c],si=trig[ulong(row)*p.dim*2+p.dim+c];
                if(!p.xf32){co=float(bfloat(co));si=float(bfloat(si));v=float(bfloat(v*co));z=float(bfloat(z*si));}
                else {v=v*co;z=z*si;}
                v=v+(c<p.dim/2?-z:z);
            }
        }
        wr(y,(ulong(head)*p.rows+row)*p.padded+c,v,p.xf32);
    }
}
kernel void en_unpack(device const uchar* x [[buffer(0)]],device uchar* y [[buffer(1)]],constant EP& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.columns)return;uint row=i/p.columns,c=i%p.columns,head=c/p.dim,d=c%p.dim;
    wr(y,ulong(row)*p.out_stride+c,rd(x,(ulong(head)*p.rows+row)*p.padded+d,p.xf32),p.xf32);
}
kernel void en_add(device const uchar* x [[buffer(0)]],device const uchar* b [[buffer(1)]],device const uchar* d [[buffer(2)]],device uchar* y [[buffer(3)]],constant EP& p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.columns)return;uint row=i/p.columns,c=i%p.columns;
    float v=rd(x,ulong(row)*p.stride+c,p.xf32)+rd(b,ulong(row)*p.out_stride+c,p.xf32);
    if(!p.xf32)v=float(bfloat(v));if(p.delta){v+=rd(d,i,p.xf32);if(!p.xf32)v=float(bfloat(v));}wr(y,ulong(row)*p.out_stride+c,v,p.xf32);
}
kernel void en_gelu(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],constant EP& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.columns)return;float v=x[i];y[i]=.5f*v*(1.f+tanh(.7978845608028654f*(v+.044715f*v*v*v)));
}
kernel void en_cast_weight(device const uchar* x [[buffer(0)]],device uchar* y [[buffer(1)]],constant EP& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.columns)return;wr(y,i,rd(x,ulong(i/p.columns)*p.stride+i%p.columns,p.xf32),p.wf32);
}
kernel void en_position(device const float* x [[buffer(0)]],device const uchar* table [[buffer(1)]],device float* y [[buffer(2)]],constant EP& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.columns)return;uint r=i/p.columns,c=i%p.columns,block=r/4,inner=r%4;
    uint gy=block/(p.gw/2)*2+inner/2,gx=block%(p.gw/2)*2+inner%2;
    float yy=float(gy)*(float(p.side-1)/float(p.gh-1)),xx=float(gx)*(float(p.side-1)/float(p.gw-1));
    uint y0=uint(yy),x0=uint(xx),y1=min(y0+1,p.side-1),x1=min(x0+1,p.side-1);float dy=yy-y0,dx=xx-x0;
    float pos=rd(table,(ulong(y0)*p.side+x0)*p.columns+c,p.wf32)*((1-dy)*(1-dx));
    pos+=rd(table,(ulong(y0)*p.side+x1)*p.columns+c,p.wf32)*((1-dy)*dx);
    pos+=rd(table,(ulong(y1)*p.side+x0)*p.columns+c,p.wf32)*(dy*(1-dx));
    pos+=rd(table,(ulong(y1)*p.side+x1)*p.columns+c,p.wf32)*(dy*dx);y[i]=x[i]+pos;
}
)metal";
constexpr std::string_view f32_linear=R"metal(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;using namespace mpp::tensor_ops;
#if EBF
using EI=bfloat;
#else
using EI=float;
#endif
struct LP {uint rows,input_stride,weight_stride,output_stride;};
kernel void en_float_linear(device const EI* x [[buffer(0)]],device const EI* w [[buffer(1)]],device float* y [[buffer(2)]],device const uchar* bias [[buffer(3)]],constant LP& p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]]) {
    uint row=group.y*32,col=group.x*128;
    auto a=tensor(const_cast<device EI*>(x),dextents<int,2>{EK,int(p.rows)},array<int,2>{1,int(p.input_stride)});
    auto b=tensor(const_cast<device EI*>(w),dextents<int,2>{EK,EN},array<int,2>{1,int(p.weight_stride)});
    auto aa=a.slice(0,row),bb=b.slice(0,col);
    matmul2d<matmul2d_descriptor(32,128,EK,false,true,false),execution_simdgroups<4>> op;
    auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
    for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto q=acc.get_multidimensional_index(i);
        if(row+q[1]<p.rows && col+q[0]<EN){float v=acc[i];if(EB)v+=EFB?reinterpret_cast<device const float*>(bias)[col+q[0]]:float(reinterpret_cast<device const bfloat*>(bias)[col+q[0]]);y[ulong(row+q[1])*p.output_stride+col+q[0]]=v;}}
}
)metal";
MatrixView imported(Backend& b,const Tensor& value){auto v=import_mlx_matrix(b,value);return {v.storage,v.rows,v.columns,v.row_stride_bytes};}
MatrixView matrix(Backend& b,uint32_t rows,uint32_t cols,ScalarType type,std::string_view label){
    require(rows && rows<=16384 && cols && cols<=32768,"encoder workspace shape exceeds bounded domain");
    auto n=size_t(rows)*cols*scalar_bytes(type);auto a=b.allocate(n,label,StorageMode::Private);return {a.view(0,n,type),rows,cols,size_t(cols)*scalar_bytes(type)};
}
// Native allocation rows may combine heads; checked total extent is bounded separately.
HeadView heads_alloc(Backend& b,uint32_t n,uint32_t rows,uint32_t dim,ScalarType type){
    require(n>0 && n<=32 && rows>0 && rows<=4096 && (dim==128 || dim==80),"invalid encoder head allocation");
    const auto pitch=size_t(dim)*scalar_bytes(type),bytes=size_t(n)*rows*pitch;
    auto a=b.allocate(bytes,"encoder-heads",StorageMode::Private);return {a.view(0,bytes,type),n,rows,dim,pitch,size_t(rows)*pitch};
}
MatrixView reshape_matrix(const MatrixView& m,uint32_t rows,uint32_t cols){require(size_t(rows)*cols==size_t(m.rows)*m.columns && m.row_stride_bytes==m.columns*scalar_bytes(m.storage.scalar_type()),"encoder reshape requires compact storage");return {m.storage,rows,cols,cols*scalar_bytes(m.storage.scalar_type())};}
MatrixView column_view(const MatrixView& m,uint32_t first,uint32_t cols){require(first+cols<=m.columns,"encoder column view outside source");auto e=scalar_bytes(m.storage.scalar_type());auto bytes=size_t(m.rows-1)*m.row_stride_bytes+size_t(cols)*e;return {m.storage.buffer().view(m.storage.offset_bytes()+first*e,bytes,m.storage.scalar_type()),m.rows,cols,m.row_stride_bytes};}
P params(const MatrixView& x,const MatrixView& out){return {.rows=x.rows,.columns=x.columns,.stride=uint32_t(x.row_stride_bytes/scalar_bytes(x.storage.scalar_type())),.out_stride=uint32_t(out.row_stride_bytes/scalar_bytes(out.storage.scalar_type())),.xf32=uint32_t(x.storage.scalar_type()==ScalarType::FP32)};}
void require_matrix(const MatrixView& m,uint32_t rows,uint32_t cols){(void)dense_linear::validate_layout({m.rows,m.columns,m.row_stride_bytes,m.storage.size_bytes(),m.storage.offset_bytes(),m.storage.scalar_type()});require(m.rows==rows && m.columns==cols,"encoder source shape mismatch");}
}
struct NativeEncoder::Impl {
    Backend backend;
    NativeEncoderStats stats;
    bool vision_bf16=native_encoder_vision_bf16();
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    std::unique_ptr<NativePrefillFfn> hybrid_ffn;
#endif
    Pipeline norm,pack,unpack,add,gelu,cast,position;
    std::map<std::pair<uint32_t,uint32_t>,NativeLinear> language_linear;
    std::map<std::tuple<uint32_t,uint32_t,bool,bool>,Pipeline> float_linear;
    std::map<std::pair<uint32_t,uint32_t>,NativeFfn> ffn;
    std::optional<NativeAttention> language_attention,vision_attention;
    Impl(){auto start=Clock::now();auto lib=backend.compile_library(kernels,"qwen3vl-native-encoder-ops",{LanguageVersion::Metal4_0});
        norm=backend.prepare_pipeline(lib,"en_norm");pack=backend.prepare_pipeline(lib,"en_pack");unpack=backend.prepare_pipeline(lib,"en_unpack");add=backend.prepare_pipeline(lib,"en_add");gelu=backend.prepare_pipeline(lib,"en_gelu");cast=backend.prepare_pipeline(lib,"en_cast_weight");position=backend.prepare_pipeline(lib,"en_position");stats.preparation_seconds+=std::chrono::duration<double>(Clock::now()-start).count();}
    MatrixView weight(Weights& w,const std::string& name){auto a=w.at(name);if(a.ndim()==1)a=mx::reshape(a,{1,int(a.size())});else if(a.ndim()>2)a=mx::reshape(a,{a.shape(0),-1});require(a.dtype()==mx::bfloat16 || a.dtype()==mx::float32,"native encoder requires original BF16/FP32 sources");mx::eval(a);return imported(backend,a);}
    void ready(Weights& w,const std::vector<std::string>& prefixes){auto start=Clock::now();std::vector<Tensor> values;
        for(const auto& key:w.sorted_keys())if(std::any_of(prefixes.begin(),prefixes.end(),[&](const std::string& p){return key.starts_with(p);}))values.push_back(w.at(key));
        require(!values.empty(),"native encoder source window is empty");mx::eval(values);stats.source_ready_seconds+=std::chrono::duration<double>(Clock::now()-start).count();}
    void record(CommandGraph& g,Pipeline op,std::vector<BufferView> buffers,P p,bool reduce=false,uint32_t heads=1){
        if(reduce)g.dispatch(op,std::move(buffers),p,{p.rows,heads,1},{256,1,1});
        else g.dispatch(op,std::move(buffers),p,{(size_t(p.rows)*p.columns+255)/256,1,1},{256,1,1});
    }
    void normalization(CommandGraph& g,MatrixView x,MatrixView w,std::optional<MatrixView> bias,MatrixView y,bool rms,float eps=1e-6f){require_matrix(w,1,x.columns);require_matrix(y,x.rows,x.columns);if(bias)require_matrix(*bias,1,x.columns);auto p=params(x,y);p.rms=rms;p.wf32=w.storage.scalar_type()==ScalarType::FP32;p.bf32=bias && bias->storage.scalar_type()==ScalarType::FP32;p.epsilon=eps;record(g,norm,{x.storage,w.storage,bias?bias->storage:w.storage,y.storage},p,true);}
    void residual(CommandGraph& g,MatrixView x,MatrixView branch,MatrixView y,std::optional<MatrixView> delta={}){require_matrix(branch,x.rows,x.columns);require_matrix(y,x.rows,x.columns);require(x.storage.scalar_type()==branch.storage.scalar_type() && y.storage.scalar_type()==x.storage.scalar_type(),"encoder residual precision mismatch");if(delta)require_matrix(*delta,x.rows,x.columns);auto p=params(x,y);p.delta=bool(delta);record(g,add,{x.storage,branch.storage,delta?delta->storage:x.storage,y.storage},p);}
    void pack_heads(CommandGraph& g,MatrixView x,HeadView y,uint32_t dim,std::optional<MatrixView> normw,MatrixView trig,bool rope,float eps=1e-6f){require_matrix(x,y.tokens,y.heads*dim);require(dim<=y.dimension && dim%2==0 && x.storage.scalar_type()==y.storage.scalar_type(),"encoder pack geometry/precision mismatch");if(normw)require_matrix(*normw,1,dim);if(rope){require_matrix(trig,x.rows,dim*2);require(trig.storage.scalar_type()==ScalarType::FP32,"encoder trig must be FP32");}auto p=params(x,x);p.heads=y.heads;p.dim=dim;p.padded=y.dimension;p.rms=bool(normw);p.rope=rope;p.wf32=normw && normw->storage.scalar_type()==ScalarType::FP32;p.epsilon=eps;
        record(g,pack,{x.storage,normw?normw->storage:x.storage,trig.storage,y.storage},p,true,y.heads);}
    void unpack_heads(CommandGraph& g,HeadView x,MatrixView y,uint32_t dim){auto p=params(y,y);p.heads=x.heads;p.dim=dim;p.padded=x.dimension;record(g,unpack,{x.storage,y.storage},p);}
    void linear(CommandGraph& g,MatrixView x,MatrixView w,std::optional<MatrixView> bias,MatrixView y){require_matrix(w,y.columns,x.columns);require_matrix(y,x.rows,w.rows);
        if(x.storage.scalar_type()==ScalarType::BF16){require(w.storage.scalar_type()==ScalarType::BF16,"language original dense weight must be BF16");auto key=std::pair{x.columns,y.columns};auto it=language_linear.find(key);require(it!=language_linear.end(),"language PSO must be prepared before graph execution");it->second.encode(g,{x,w,y,bias});return;}
        require(x.storage.scalar_type()==ScalarType::FP32 && y.storage.scalar_type()==ScalarType::FP32,"vision projection requires FP32 model boundary");
        MatrixView xx=x,ww=w;const auto operand=vision_bf16?ScalarType::BF16:ScalarType::FP32;
        if(vision_bf16){xx=matrix(backend,x.rows,x.columns,operand,"encoder-current-bf16-input");auto pp=params(x,xx);pp.wf32=0;record(g,cast,{x.storage,xx.storage},pp);}
        if(w.storage.scalar_type()!=operand){ww=matrix(backend,w.rows,w.columns,operand,"encoder-current-converted-weight");auto pp=params(w,ww);pp.wf32=!vision_bf16;record(g,cast,{w.storage,ww.storage},pp);}
        auto key=std::tuple{x.columns,y.columns,bool(bias),bool(bias && bias->storage.scalar_type()==ScalarType::FP32)};auto it=float_linear.find(key);require(it!=float_linear.end(),"vision PSO must be prepared before graph execution");
        dense_linear::Parameters p{x.rows,uint32_t(xx.row_stride_bytes/scalar_bytes(operand)),uint32_t(ww.row_stride_bytes/scalar_bytes(operand)),uint32_t(y.row_stride_bytes/4)};
        g.dispatch(it->second,{xx.storage,ww.storage,y.storage,bias?bias->storage:w.storage},p,{(size_t(y.columns)+127)/128,(size_t(x.rows)+31)/32,1},{128,1,1});
    }
    void prepare_linear(Weights& w,const std::string& prefix,bool fp32){const auto& a=w.at(prefix+".weight");uint32_t n=a.shape(0),k=uint32_t(a.size()/n);auto start=Clock::now();
        if(fp32){require(k%16==0 && n%16==0 && k<=12288 && n<=16384,"vision dense projection requires bounded16 geometry");bool bias=w.has(prefix+".bias"),fb=bias && w.at(prefix+".bias").dtype()==mx::float32;auto key=std::tuple{k,n,bias,fb};if(float_linear.contains(key))return;
            auto src="#define EBF "+std::to_string(vision_bf16)+"\n#define EK "+std::to_string(k)+"\n#define EN "+std::to_string(n)+"\n#define EB "+std::to_string(bias)+"\n#define EFB "+std::to_string(fb)+"\n"+std::string(f32_linear);auto l=backend.compile_library(src,"qwen3vl-fp32-linear",{LanguageVersion::Metal4_0});float_linear.emplace(key,backend.prepare_pipeline(l,"en_float_linear"));}
        else{auto key=std::pair{k,n};if(language_linear.contains(key))return;language_linear.emplace(key,NativeLinear::prepare(backend,{k,n}));}
        stats.preparation_seconds+=std::chrono::duration<double>(Clock::now()-start).count();}
    void projection(CommandGraph& g,Weights& w,const std::string& p,MatrixView x,MatrixView y){linear(g,x,weight(w,p+".weight"),w.has(p+".bias")?std::optional{weight(w,p+".bias")}:std::nullopt,y);}
    void complete(CommandGraph g,std::atomic<bool>& stop,const std::function<void()>& future={}){checkpoint(stop);stats.dispatches+=std::count_if(g.steps().begin(),g.steps().end(),[](const GraphStep& s){return s.kind==GraphStep::Kind::Dispatch;});++stats.graph_submissions;
        auto ticket=backend.submit(std::move(g));
        // A future CPU/source error must not release current GPU owners early.
        std::exception_ptr future_error;try{if(future)future();}catch(...){future_error=std::current_exception();}
        auto start=Clock::now();auto result=ticket.wait();stats.gpu_wait_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
        require(result.ok && result.physically_complete,"native encoder GPU completion failed: "+result.error);checkpoint(stop);if(future_error)std::rethrow_exception(future_error);stats.physically_complete=true;}
    MatrixView trig(const std::vector<float>& values,uint32_t rows,uint32_t dim){require(values.size()==size_t(rows)*dim*2,"encoder trig extent mismatch");auto buffer=backend.allocate(values.size()*4,"encoder-rope-trig");std::memcpy(buffer.contents(),values.data(),values.size()*4);return {buffer.view(0,values.size()*4,ScalarType::FP32),rows,dim*2,size_t(dim)*8};}
    bool future_allowed(){auto mem=observe_process_memory();const bool allowed=mem.available && mem.system_available_bytes>(uint64_t(5)<<30);if(!allowed)++stats.skipped_future_windows;return allowed;}
};
NativeEncoder::NativeEncoder():impl_(std::make_unique<Impl>()){}
NativeEncoder::~NativeEncoder()=default;
NativeEncoderStats NativeEncoder::statistics() const{return impl_->stats;}
bool NativeEncoder::healthy() const{
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    if(impl_->hybrid_ffn && !impl_->hybrid_ffn->healthy())return false;
#endif
    return impl_->backend.healthy();
}
void NativeEncoder::reset_statistics(){require(healthy(),"cannot reuse unhealthy native encoder");impl_->stats={};
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    if(impl_->hybrid_ffn)impl_->hybrid_ffn->reset_statistics();
#endif
}

Tensor NativeEncoder::language(Weights& weights,const Tensor& embeddings,const Tensor& positions,int valid,
    const std::vector<Tensor>& deltas,const Event& event,std::atomic<bool>& stop,TextConfig c){
    auto& p=*impl_;p.stats.physically_complete=false;
    require(embeddings.ndim()==3 && c.head_dim==128 && c.heads>0 && c.heads<=32 && c.kv_heads>0 && c.heads%c.kv_heads==0 && c.layers>0 && c.layers<=36 && std::isfinite(c.theta) && c.theta>0 && std::isfinite(c.epsilon) && c.epsilon>0 && deltas.size()<=size_t(c.layers) && c.mrope_sections==std::array{24,20,20},"unsupported native Qwen3-VL language geometry");
    const uint32_t rows=embeddings.shape(1),h=c.heads*c.head_dim;
    require(embeddings.shape()==mx::Shape{1,int(rows),int(h)} && embeddings.dtype()==mx::bfloat16 && positions.shape()==mx::Shape{3,int(rows)} && positions.dtype()==mx::int32 && rows>0 && rows<=4096 && valid==int(rows),"native language encoder requires unpadded BF16 batch1 rows1..4096 and int32 mRoPE positions");
    require(!weights.has_runtime_loras(),"native encoder cannot ignore runtime encoder LoRA");
    const auto base=weights.has("model.language_model.layers.0.input_layernorm.weight")?std::string("model.language_model."):std::string("model.");
    const auto first=base+"layers.0";const uint32_t f=weights.at(first+".mlp.gate_proj.weight").shape(0);
    auto begin=Clock::now();
    if(!p.language_attention || p.language_attention->geometry().heads!=uint32_t(c.heads) || p.language_attention->geometry().kv_heads!=uint32_t(c.kv_heads))p.language_attention=NativeAttention::prepare(p.backend,{uint32_t(c.heads),128,1,uint32_t(c.kv_heads),ScalarType::BF16});
    const bool hybrid=!native_encoder_ane_cache().empty();
#if TURBOCIDER_ENABLE_PRIVATE_ANE
    if(hybrid){require(h==4096 && f==12288 && rows<=1152,"native encoder ANE candidate requires Qwen FFN rows1..1152");
        if(!p.hybrid_ffn){NativeHybridFfnOptions options;options.rows=1152;options.program_cache=native_encoder_ane_cache();options.register_weight_staging=true;
            p.hybrid_ffn=NativePrefillFfn::prepare(p.backend,std::move(options));}}
#else
    require(!hybrid,"native encoder ANE requires a private-enabled build");
#endif
    auto ffkey=std::pair{h,f};if(!hybrid && !p.ffn.contains(ffkey))p.ffn.emplace(ffkey,NativeFfn::prepare(p.backend,{h,f,f}));
    p.stats.preparation_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
    for(const char* op:{"q_proj","k_proj","v_proj","o_proj"})p.prepare_linear(weights,first+".self_attn."+op,false);
    std::vector<Tensor> inputs{embeddings,positions};for(const auto& d:deltas){require(d.shape()==embeddings.shape() && d.dtype()==mx::bfloat16,"native deepstack must preserve full BF16 embedding shape");inputs.push_back(d);}mx::eval(inputs);
    auto input=imported(p.backend,embeddings);std::vector<MatrixView> delta;for(const auto& d:deltas)delta.push_back(imported(p.backend,d));
    auto pos=positions.data<int32_t>();std::vector<float> tr(size_t(rows)*256);
    for(uint32_t r=0;r<rows;++r)for(uint32_t d=0;d<128;++d){uint32_t i=d%64,axis=0;if(i%3==1 && i<60)axis=1;if(i%3==2 && i<60)axis=2;float a=float(pos[axis*rows+r])*std::pow(c.theta,-float(2*i)/128.f);tr[size_t(r)*256+d]=std::cos(a);tr[size_t(r)*256+128+d]=std::sin(a);}
    auto trig=p.trig(tr,rows,128);
    auto x=matrix(p.backend,rows,h,ScalarType::BF16,"encoder-language-state");auto out=allocate_mlx_matrix(rows,h,ScalarType::BF16,true);auto output=imported(p.backend,out);
    auto n=matrix(p.backend,rows,h,ScalarType::BF16,"encoder-language-norm"),qr=matrix(p.backend,rows,h,ScalarType::BF16,"encoder-raw-query"),kr=matrix(p.backend,rows,c.kv_heads*128,ScalarType::BF16,"encoder-raw-key"),vr=matrix(p.backend,rows,c.kv_heads*128,ScalarType::BF16,"encoder-raw-value");
    auto q=heads_alloc(p.backend,c.heads,rows,128,ScalarType::BF16),k=heads_alloc(p.backend,c.kv_heads,rows,128,ScalarType::BF16),v=heads_alloc(p.backend,c.kv_heads,rows,128,ScalarType::BF16),a=heads_alloc(p.backend,c.heads,rows,128,ScalarType::BF16);
    auto ar=matrix(p.backend,rows,h,ScalarType::BF16,"encoder-attended-row"),ap=matrix(p.backend,rows,h,ScalarType::BF16,"encoder-attention-projection"),res=matrix(p.backend,rows,h,ScalarType::BF16,"encoder-attention-residual"),fh=hybrid?MatrixView{}:matrix(p.backend,rows,f,ScalarType::BF16,"encoder-ffn-hidden"),fo=matrix(p.backend,rows,h,ScalarType::BF16,"encoder-ffn-output");
    MatrixView current=input;bool ready=false;
    for(int i=0;i<c.layers;i+=hybrid?1:2){int count=std::min(hybrid?1:2,c.layers-i);std::vector<std::string> prefixes;for(int j=0;j<count;++j)prefixes.push_back(base+"layers."+std::to_string(i+j)+".");if(!ready)p.ready(weights,prefixes);ready=false;CommandGraph graph;
        for(int j=0;j<count;++j){int layer=i+j;auto pre=base+"layers."+std::to_string(layer);auto destination=(layer==c.layers-1)?output:x;
            p.normalization(graph,current,p.weight(weights,pre+".input_layernorm.weight"),{},n,true,c.epsilon);
            p.projection(graph,weights,pre+".self_attn.q_proj",n,qr);p.projection(graph,weights,pre+".self_attn.k_proj",n,kr);p.projection(graph,weights,pre+".self_attn.v_proj",n,vr);
            p.pack_heads(graph,qr,q,128,p.weight(weights,pre+".self_attn.q_norm.weight"),trig,true,c.epsilon);p.pack_heads(graph,kr,k,128,p.weight(weights,pre+".self_attn.k_norm.weight"),trig,true,c.epsilon);p.pack_heads(graph,vr,v,128,{},trig,false);
            p.language_attention->encode(graph,{q,k,v,a,{true,0,1.f/std::sqrt(128.f)}});p.unpack_heads(graph,a,ar,128);p.projection(graph,weights,pre+".self_attn.o_proj",ar,ap);
            p.residual(graph,current,ap,res);p.normalization(graph,res,p.weight(weights,pre+".post_attention_layernorm.weight"),{},n,true,c.epsilon);
            const auto gate=p.weight(weights,pre+".mlp.gate_proj.weight"),up=p.weight(weights,pre+".mlp.up_proj.weight"),down=p.weight(weights,pre+".mlp.down_proj.weight");
#if TURBOCIDER_ENABLE_PRIVATE_ANE
            if(hybrid){
                CommandGraph after;p.residual(after,res,fo,destination,size_t(layer)<delta.size()?std::optional{delta[layer]}:std::nullopt);
                checkpoint(stop);
                p.stats.dispatches+=std::count_if(graph.steps().begin(),graph.steps().end(),[](const GraphStep& s){return s.kind==GraphStep::Kind::Dispatch;});
                p.stats.dispatches+=std::count_if(after.steps().begin(),after.steps().end(),[](const GraphStep& s){return s.kind==GraphStep::Kind::Dispatch;});
                p.hybrid_ffn->execute(n,gate,up,down,fo,std::move(graph),std::move(after),std::chrono::seconds(30),[&]{
                    if(layer+1<c.layers && p.future_allowed()) {p.ready(weights,{base+"layers."+std::to_string(layer+1)+"."});ready=true;++p.stats.future_weight_windows;}
                });
                checkpoint(stop);
                p.stats.physically_complete=true;const auto stats=p.hybrid_ffn->statistics();
                p.stats.ane_ffns=stats.calls;p.stats.ane_chunks=stats.chunks;p.stats.ane_padded_rows=stats.padded_rows;
                p.stats.ane_producer_seconds=stats.producer_seconds;p.stats.ane_branch_seconds=stats.branch_seconds;p.stats.ane_consumer_seconds=stats.consumer_seconds;
                // Every W1 chunk submits producer, complementary GPU and consumer.
                p.stats.graph_submissions+=3;
            }else
#endif
            {
                p.ffn.at(ffkey).encode(graph,{n,gate,up,down,fh,fo,0});
                p.residual(graph,res,fo,destination,size_t(layer)<delta.size()?std::optional{delta[layer]}:std::nullopt);
            }
            current=destination;
        }
        auto future=[&]{if(i+count<c.layers && p.future_allowed()){std::vector<std::string> next;for(int j=i+count;j<std::min(i+count+(hybrid?1:2),c.layers);++j)next.push_back(base+"layers."+std::to_string(j)+".");p.ready(weights,next);ready=true;++p.stats.future_weight_windows;}};
        if(!hybrid)p.complete(std::move(graph),stop,future);
        for(int j=0;j<count;++j){weights.erase_prefix(prefixes[j]);++p.stats.language_layers;if(event)event("qwen21_text_encode",i+j+1,c.layers);}
    }
    if(c.final_norm){p.ready(weights,{base+"norm."});auto raw=allocate_mlx_matrix(rows,h,ScalarType::BF16,true);auto dest=imported(p.backend,raw);CommandGraph graph;
        p.normalization(graph,output,p.weight(weights,base+"norm.weight"),{},dest,true,c.epsilon);p.complete(std::move(graph),stop);return raw;}
    return out;
}

VisionFeatures NativeEncoder::vision(Weights& weights,const Tensor& patches,int gh,int gw,const Event& event,std::atomic<bool>& stop,VisionConfig c){
    auto& p=*impl_;p.stats.physically_complete=false;
    require(gh>=2 && gw>=2 && gh<=4096 && gw<=4096 && int64_t(gh)*gw<=4096 && c.hidden>0 && c.heads>0 && c.heads<=32 && c.hidden%c.heads==0 && c.patch>0 && c.patch<=16 && c.position_side>0 && c.position_side<=64,"unsupported native visual geometry");
    require(std::is_sorted(c.deepstack_layers.begin(),c.deepstack_layers.end()) && std::adjacent_find(c.deepstack_layers.begin(),c.deepstack_layers.end())==c.deepstack_layers.end(),"native DeepStack layers must be unique and sorted");
    for(int layer:c.deepstack_layers)require(layer>=0 && layer<c.layers,"native DeepStack layer out of range");
    const uint32_t rows=gh*gw,h=c.hidden,dim=h/c.heads;
    require(!c.pe_qwen35 && !c.pe_fp32 && !c.pe_compensated_projection && c.merge==2 && c.temporal==2 && c.channels==3 && dim==72 && c.layers>0 && c.layers<=27 && gh>=2 && gw>=2 && gh%2==0 && gw%2==0 && rows<=4096 && patches.shape()==mx::Shape{int(rows),c.channels*c.temporal*c.patch*c.patch} && patches.dtype()==mx::float32,"native vision requires Qwen3-VL FP32 patches, D72, bounded4096 rows and merge2");
    require(!weights.has_runtime_loras(),"native vision cannot ignore runtime encoder LoRA");
    auto start=Clock::now();if(!p.vision_attention || p.vision_attention->geometry().heads!=uint32_t(c.heads))p.vision_attention=NativeAttention::prepare(p.backend,{uint32_t(c.heads),80,1,0,ScalarType::FP32});p.stats.preparation_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
    p.prepare_linear(weights,"model.visual.patch_embed.proj",true);
    const std::string block="model.visual.blocks.0";for(const char* name:{"attn.qkv","attn.proj","mlp.linear_fc1","mlp.linear_fc2"})p.prepare_linear(weights,block+"."+name,true);
    const auto f=weights.at(block+".mlp.linear_fc1.weight").shape(0);
    for(const auto& prefix:{std::string("model.visual.merger")}){p.prepare_linear(weights,prefix+".linear_fc1",true);p.prepare_linear(weights,prefix+".linear_fc2",true);}
    for(size_t i=0;i<c.deepstack_layers.size();++i){auto pre="model.visual.deepstack_merger_list."+std::to_string(i);p.prepare_linear(weights,pre+".linear_fc1",true);p.prepare_linear(weights,pre+".linear_fc2",true);}
    mx::eval(patches);auto input=imported(p.backend,patches);auto x=matrix(p.backend,rows,h,ScalarType::FP32,"encoder-vision-state"),y=matrix(p.backend,rows,h,ScalarType::FP32,"encoder-vision-second-state"),n=matrix(p.backend,rows,h,ScalarType::FP32,"encoder-vision-norm");
    auto qkv=matrix(p.backend,rows,3*h,ScalarType::FP32,"encoder-vision-qkv");auto q=heads_alloc(p.backend,c.heads,rows,80,ScalarType::FP32),k=heads_alloc(p.backend,c.heads,rows,80,ScalarType::FP32),v=heads_alloc(p.backend,c.heads,rows,80,ScalarType::FP32),a=heads_alloc(p.backend,c.heads,rows,80,ScalarType::FP32);
    auto ar=matrix(p.backend,rows,h,ScalarType::FP32,"encoder-vision-attended"),ap=matrix(p.backend,rows,h,ScalarType::FP32,"encoder-vision-projection"),res=matrix(p.backend,rows,h,ScalarType::FP32,"encoder-vision-residual"),mlp=matrix(p.backend,rows,f,ScalarType::FP32,"encoder-vision-mlp"),activated=matrix(p.backend,rows,f,ScalarType::FP32,"encoder-vision-activated");
    std::vector<float> tr(size_t(rows)*144);for(uint32_t r=0;r<rows;++r){uint32_t blockrow=r/4,inner=r%4;uint32_t gy=blockrow/(gw/2)*2+inner/2,gx=blockrow%(gw/2)*2+inner%2;
        for(uint32_t d=0;d<72;++d){uint32_t z=d%36;float angle=float(z<18?gy:gx)*std::pow(10000.f,-float(2*(z%18))/36.f);tr[size_t(r)*144+d]=std::cos(angle);tr[size_t(r)*144+72+d]=std::sin(angle);}}
    auto trig=p.trig(tr,rows,72);p.ready(weights,{"model.visual.patch_embed.","model.visual.pos_embed."});CommandGraph initial;p.projection(initial,weights,"model.visual.patch_embed.proj",input,y);auto table=p.weight(weights,"model.visual.pos_embed.weight");require_matrix(table,c.position_side*c.position_side,h);auto pp=params(y,x);pp.gh=gh;pp.gw=gw;pp.side=c.position_side;pp.wf32=table.storage.scalar_type()==ScalarType::FP32;p.record(initial,p.position,{y.storage,table.storage,x.storage},pp);p.complete(std::move(initial),stop);
    std::vector<Tensor> deep;std::optional<Tensor> final;
    auto merge=[&](CommandGraph& graph,MatrixView hidden,const std::string& pre,bool postshuffle){uint32_t width=h*4;auto merged=reshape_matrix(hidden,rows/4,width);auto nn=matrix(p.backend,postshuffle?rows/4:rows,postshuffle?width:h,ScalarType::FP32,"encoder-merger-norm");p.normalization(graph,postshuffle?merged:hidden,p.weight(weights,pre+".norm.weight"),p.weight(weights,pre+".norm.bias"),nn,false);if(!postshuffle)nn=reshape_matrix(nn,rows/4,width);
        uint32_t f1=weights.at(pre+".linear_fc1.weight").shape(0),result_width=weights.at(pre+".linear_fc2.weight").shape(0);auto first=matrix(p.backend,rows/4,f1,ScalarType::FP32,"encoder-merger-projection"),gel=matrix(p.backend,rows/4,f1,ScalarType::FP32,"encoder-merger-gelu");p.projection(graph,weights,pre+".linear_fc1",nn,first);p.record(graph,p.gelu,{first.storage,gel.storage},params(first,gel));auto result=allocate_mlx_matrix(rows/4,result_width,ScalarType::FP32,false);p.projection(graph,weights,pre+".linear_fc2",gel,imported(p.backend,result));return result;};
    bool ready=false;MatrixView current=x;
    for(int i=0;i<c.layers;i+=2){int count=std::min(2,c.layers-i);std::vector<std::string> prefixes;
        for(int j=0;j<count;++j){int layer=i+j;prefixes.push_back("model.visual.blocks."+std::to_string(layer)+".");auto found=std::find(c.deepstack_layers.begin(),c.deepstack_layers.end(),layer);if(found!=c.deepstack_layers.end())prefixes.push_back("model.visual.deepstack_merger_list."+std::to_string(found-c.deepstack_layers.begin())+".");if(layer==c.layers-1)prefixes.push_back("model.visual.merger.");}
        if(!ready)p.ready(weights,prefixes);ready=false;CommandGraph graph;
        for(int j=0;j<count;++j){int layer=i+j;auto pre="model.visual.blocks."+std::to_string(layer);auto dest=current.storage.buffer().native_handle()==x.storage.buffer().native_handle()?y:x;
            p.normalization(graph,current,p.weight(weights,pre+".norm1.weight"),p.weight(weights,pre+".norm1.bias"),n,false);p.projection(graph,weights,pre+".attn.qkv",n,qkv);
            p.pack_heads(graph,column_view(qkv,0,h),q,72,{},trig,true);p.pack_heads(graph,column_view(qkv,h,h),k,72,{},trig,true);p.pack_heads(graph,column_view(qkv,h*2,h),v,72,{},trig,false);p.vision_attention->encode(graph,{q,k,v,a,{false,0,1.f/std::sqrt(72.f)}});p.unpack_heads(graph,a,ar,72);p.projection(graph,weights,pre+".attn.proj",ar,ap);p.residual(graph,current,ap,res);
            p.normalization(graph,res,p.weight(weights,pre+".norm2.weight"),p.weight(weights,pre+".norm2.bias"),n,false);p.projection(graph,weights,pre+".mlp.linear_fc1",n,mlp);p.record(graph,p.gelu,{mlp.storage,activated.storage},params(mlp,activated));p.projection(graph,weights,pre+".mlp.linear_fc2",activated,ap);p.residual(graph,res,ap,dest);current=dest;
            auto found=std::find(c.deepstack_layers.begin(),c.deepstack_layers.end(),layer);if(found!=c.deepstack_layers.end())deep.push_back(merge(graph,current,"model.visual.deepstack_merger_list."+std::to_string(found-c.deepstack_layers.begin()),true));if(layer==c.layers-1)final=merge(graph,current,"model.visual.merger",false);
        }
        p.complete(std::move(graph),stop,[&]{if(i+count<c.layers && p.future_allowed()){std::vector<std::string> next;for(int j=i+count;j<std::min(i+count+2,c.layers);++j){next.push_back("model.visual.blocks."+std::to_string(j)+".");auto found=std::find(c.deepstack_layers.begin(),c.deepstack_layers.end(),j);if(found!=c.deepstack_layers.end())next.push_back("model.visual.deepstack_merger_list."+std::to_string(found-c.deepstack_layers.begin())+".");if(j==c.layers-1)next.push_back("model.visual.merger.");}p.ready(weights,next);ready=true;++p.stats.future_weight_windows;}});
        for(int j=0;j<count;++j){weights.erase_prefix("model.visual.blocks."+std::to_string(i+j)+".");++p.stats.vision_layers;if(event)event("qwen21_vision_encode",i+j+1,c.layers);}
    }
    require(final.has_value() && deep.size()==c.deepstack_layers.size(),"incomplete native visual tower");return {*final,std::move(deep)};
}
} // namespace tc::qwen21
