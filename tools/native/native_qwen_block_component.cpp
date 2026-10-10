// Explicit --tiny-device mode only. Synthetic tensors; no model or App load.
// Oracle implements model operations on CPU with visible BF16 boundaries.
#include "../../native/models/qwen21/native_block.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace tc::metal_graph;
using namespace tc::qwen21::native_graph;
namespace {
void check(bool p,const char* message) { if(!p) throw std::runtime_error(message); }
uint16_t bf16(float x) {
    auto b=std::bit_cast<uint32_t>(x);
    return uint16_t((b+0x7fffU+((b>>16)&1U))>>16);
}
float from_bf16(uint16_t x) { return std::bit_cast<float>(uint32_t(x)<<16); }
float round_bf16(float x) { return from_bf16(bf16(x)); }
float load(const MatrixView& v,uint32_t r,uint32_t c) {
    auto* data=static_cast<const std::byte*>(v.storage.contents())+size_t(r)*v.row_stride_bytes;
    return v.storage.scalar_type()==ScalarType::FP32 ? reinterpret_cast<const float*>(data)[c] :
        from_bf16(reinterpret_cast<const uint16_t*>(data)[c]);
}
float load(const HeadView& v,uint32_t h,uint32_t r,uint32_t d) {
    auto* data=static_cast<const std::byte*>(v.storage.contents())+size_t(h)*v.head_stride_bytes+
        size_t(r)*v.token_stride_bytes;
    return from_bf16(reinterpret_cast<const uint16_t*>(data)[d]);
}
void store(const MatrixView& v,uint32_t r,uint32_t c,float value) {
    auto* data=static_cast<std::byte*>(v.storage.contents())+size_t(r)*v.row_stride_bytes;
    if (v.storage.scalar_type()==ScalarType::FP32) reinterpret_cast<float*>(data)[c]=value;
    else reinterpret_cast<uint16_t*>(data)[c]=bf16(value);
}
MatrixView matrix(Backend& b,uint32_t rows,uint32_t columns,ScalarType type=ScalarType::BF16) {
    const size_t pitch=size_t(columns+8)*scalar_bytes(type),bytes=size_t(rows)*pitch;
    auto allocation=b.allocate(bytes+80,"native-tiny-matrix");
    std::memset(allocation.contents(),0x5a,allocation.size_bytes());
    return {allocation.view(16,bytes,type),rows,columns,pitch};
}
HeadView heads(Backend& b,uint32_t n,uint32_t rows) {
    const size_t token=136*2,head=size_t(rows+3)*token,bytes=size_t(n)*head;
    auto allocation=b.allocate(bytes+80,"native-tiny-heads");
    std::memset(allocation.contents(),0x5a,allocation.size_bytes());
    return {allocation.view(16,bytes,ScalarType::BF16),n,rows,128,token,head};
}
void fill(const MatrixView& v,uint32_t salt,float divisor=64.f) {
    for(uint32_t r=0;r<v.rows;++r) for(uint32_t c=0;c<v.columns;++c)
        store(v,r,c,float(int((r*7+c*13+salt)%17)-8)/divisor);
}
void matrix_canaries(const MatrixView& v) {
    const auto* data=static_cast<const unsigned char*>(v.storage.buffer().contents());
    for(size_t i=0;i<v.storage.offset_bytes();++i) check(data[i]==0x5a,"matrix prefix modified");
    for(size_t i=v.storage.offset_bytes()+v.storage.size_bytes();i<v.storage.buffer().size_bytes();++i)
        check(data[i]==0x5a,"matrix suffix modified");
    data=static_cast<const unsigned char*>(v.storage.contents());
    for(uint32_t r=0;r<v.rows;++r) for(size_t i=size_t(v.columns)*scalar_bytes(v.storage.scalar_type());i<v.row_stride_bytes;++i)
        check(data[size_t(r)*v.row_stride_bytes+i]==0x5a,"matrix row padding modified");
}
void head_canaries(const HeadView& v) {
    const auto* buffer=static_cast<const unsigned char*>(v.storage.buffer().contents());
    for(size_t i=0;i<v.storage.offset_bytes();++i)check(buffer[i]==0x5a,"head prefix modified");
    for(size_t i=v.storage.offset_bytes()+v.storage.size_bytes();i<v.storage.buffer().size_bytes();++i)
        check(buffer[i]==0x5a,"head suffix modified");
    const auto* data=static_cast<const unsigned char*>(v.storage.contents());
    for(uint32_t h=0;h<v.heads;++h) for(size_t i=0;i<v.head_stride_bytes;++i) {
        const size_t row=i/v.token_stride_bytes,col=i%v.token_stride_bytes;
        if(row>=v.tokens || col>=256) check(data[size_t(h)*v.head_stride_bytes+i]==0x5a,"head padding modified");
    }
}
using Snapshot=std::pair<BufferView,std::vector<std::byte>>;
Snapshot snapshot(BufferView v) {
    auto* data=static_cast<const std::byte*>(v.contents());
    return {v,std::vector<std::byte>(data,data+v.size_bytes())};
}
void unchanged(const Snapshot& s) {
    check(std::memcmp(s.first.contents(),s.second.data(),s.second.size())==0,"readonly source changed");
}
std::vector<float> values(const MatrixView& v) {
    std::vector<float> out(size_t(v.rows)*v.columns);
    for(uint32_t r=0;r<v.rows;++r) for(uint32_t c=0;c<v.columns;++c) out[size_t(r)*v.columns+c]=load(v,r,c);
    return out;
}
struct Error { double max_abs=0,rrms=0; };
Error compare(const std::vector<float>& a,const std::vector<float>& expected,const char* message) {
    check(a.size()==expected.size(),"oracle extent mismatch");
    Error e; double sq=0,ref=0;
    for(size_t i=0;i<a.size();++i) {
        check(std::isfinite(a[i]),"native block nonfinite");
        const double d=double(a[i])-expected[i];
        e.max_abs=std::max(e.max_abs,std::abs(d)); sq+=d*d; ref+=double(expected[i])*expected[i];
    }
    e.rrms=std::sqrt(sq/std::max(ref,1e-30));
    std::cout<<"oracle="<<message<<" count="<<a.size()<<" max_abs="<<e.max_abs<<" rrms="<<e.rrms<<'\n';
    check(e.rrms<0.025 && e.max_abs<0.032,"native block component oracle tolerance exceeded");
    return e;
}
std::vector<float> linear(const std::vector<float>& x,uint32_t rows,uint32_t k,const Projection& p) {
    const auto n=p.weight.rows;
    std::vector<float> out(size_t(rows)*n);
    for(uint32_t r=0;r<rows;++r) for(uint32_t c=0;c<n;++c) {
        float sum=0;
        for(uint32_t i=0;i<k;++i) sum+=x[size_t(r)*k+i]*load(p.weight,c,i);
        sum=round_bf16(sum);
        if(p.bias) sum=round_bf16(sum+load(*p.bias,0,c));
        out[size_t(r)*n+c]=sum;
    }
    return out;
}
std::vector<float> norm_mod(const std::vector<float>& x,uint32_t rows,uint32_t h,const MatrixView& scale) {
    std::vector<float> out(x.size());
    for(uint32_t r=0;r<rows;++r) {
        float sum=0,sq=0;
        for(uint32_t c=0;c<h;++c) sum+=x[size_t(r)*h+c];
        const float mean=sum/h;
        for(uint32_t c=0;c<h;++c) { const float v=x[size_t(r)*h+c]-mean; sq+=v*v; }
        const float inv=1/std::sqrt(sq/h+1e-6f);
        for(uint32_t c=0;c<h;++c) {
            const auto i=size_t(r)*h+c;
            out[i]=round_bf16(round_bf16((x[i]-mean)*inv)*round_bf16(1+load(scale,scale.rows==1?0:r,c)));
        }
    }
    return out;
}
std::vector<float> residual(const std::vector<float>& x,const std::vector<float>& y,
                           uint32_t rows,uint32_t h,const MatrixView& gate) {
    std::vector<float> out(x.size());
    for(uint32_t r=0;r<rows;++r) for(uint32_t c=0;c<h;++c) {
        const auto i=size_t(r)*h+c;
        out[i]=round_bf16(x[i]+round_bf16(round_bf16(std::tanh(load(gate,gate.rows==1?0:r,c)))*y[i]));
    }
    return out;
}
// Oracle head data are canonical [head,row,dim], independent of device strides.
std::vector<float> pack(const std::vector<float>& x,uint32_t rows,uint32_t n) {
    std::vector<float> out(x.size());
    for(uint32_t h=0;h<n;++h) for(uint32_t r=0;r<rows;++r) for(uint32_t d=0;d<128;++d)
        out[(size_t(h)*rows+r)*128+d]=x[size_t(r)*n*128+h*128+d];
    return out;
}
std::vector<float> unpack(const std::vector<float>& x,uint32_t rows,uint32_t n) {
    std::vector<float> out(x.size());
    for(uint32_t h=0;h<n;++h) for(uint32_t r=0;r<rows;++r) for(uint32_t d=0;d<128;++d)
        out[size_t(r)*n*128+h*128+d]=x[(size_t(h)*rows+r)*128+d];
    return out;
}
std::vector<float> qk_rope(const std::vector<float>& input,uint32_t rows,uint32_t n,
    const MatrixView& weight,const MatrixView& cos,const MatrixView& sin) {
    std::vector<float> out(input.size());
    for(uint32_t h=0;h<n;++h) for(uint32_t r=0;r<rows;++r) {
        const size_t base=size_t(r)*n*128+h*128;
        float sq=0; for(uint32_t d=0;d<128;++d) sq+=input[base+d]*input[base+d];
        float inv=1/std::sqrt(sq/128+1e-6f);
        for(uint32_t d=0;d<128;d+=2) {
            float a=round_bf16(round_bf16(input[base+d]*inv)*load(weight,0,d));
            float b=round_bf16(round_bf16(input[base+d+1]*inv)*load(weight,0,d+1));
            const float c=load(cos,r,d/2),s=load(sin,r,d/2);
            const size_t dst=(size_t(h)*rows+r)*128+d;
            out[dst]=round_bf16(a*c-b*s); out[dst+1]=round_bf16(b*c+a*s);
        }
    }
    return out;
}
std::vector<float> head_values(const HeadView& v) {
    std::vector<float> out(size_t(v.heads)*v.tokens*128);
    for(uint32_t h=0;h<v.heads;++h) for(uint32_t r=0;r<v.tokens;++r) for(uint32_t d=0;d<128;++d)
        out[(size_t(h)*v.tokens+r)*128+d]=load(v,h,r,d);
    return out;
}
std::vector<float> oracle(const BlockBindings& b,const BlockWorkspace& w,uint32_t n) {
    const auto rows=b.input.rows,h=n*128;
    auto x=values(b.input),input=norm_mod(x,rows,h,b.modulation[0]);
    auto qraw=linear(input,rows,h,b.weights.query),kraw=linear(input,rows,h,b.weights.key);
    auto vraw=linear(input,rows,h,b.weights.value);
    auto q=qk_rope(qraw,rows,n,b.weights.query_norm,b.cosine,b.sine);
    auto k=qk_rope(kraw,rows,n,b.weights.key_norm,b.cosine,b.sine);
    auto v=pack(vraw,rows,n);
    compare(values(w.normalized_attention),input,"norm-attn");
    compare(values(w.raw_query),qraw,"linear-q");
    compare(head_values(w.query),q,"qk-rope-q");
    compare(head_values(w.key),k,"qk-rope-k");
    compare(head_values(w.value),v,"pack-v");
    uint32_t kvrows=rows;
    if(b.phase==BlockPhase::SteadyDecode && b.prefix_tokens) {
        kvrows+=b.prefix_tokens;
        std::vector<float> joined_k(size_t(n)*kvrows*128),joined_v(joined_k.size());
        for(uint32_t hd=0;hd<n;++hd) for(uint32_t r=0;r<kvrows;++r) for(uint32_t d=0;d<128;++d) {
            const auto j=(size_t(hd)*kvrows+r)*128+d;
            joined_k[j]=r<b.prefix_tokens?load(b.prefix->key,hd,r,d):k[(size_t(hd)*rows+r-b.prefix_tokens)*128+d];
            joined_v[j]=r<b.prefix_tokens?load(b.prefix->value,hd,r,d):v[(size_t(hd)*rows+r-b.prefix_tokens)*128+d];
        }
        k=std::move(joined_k); v=std::move(joined_v);
        compare(head_values(w.joined_key),k,"prefix-concat-k");
        compare(head_values(w.joined_value),v,"prefix-concat-v");
    } else if(b.prefix_tokens) {
        for(uint32_t hd=0;hd<n;++hd) for(uint32_t r=0;r<b.prefix_tokens;++r) for(uint32_t d=0;d<128;++d) {
            check(load(b.prefix->key,hd,r,d)==load(w.key,hd,r,d),"prefix K cache exact copy mismatch");
            check(load(b.prefix->value,hd,r,d)==load(w.value,hd,r,d),"prefix V cache exact copy mismatch");
        }
    }
    std::vector<float> attention(size_t(n)*rows*128);
    for(uint32_t hd=0;hd<n;++hd) for(uint32_t r=0;r<rows;++r) {
        uint32_t visible=kvrows;
        if(b.phase==BlockPhase::PrefixBuild) for(const auto& segment:b.segments)
            if(int(r)>=segment.start && int(r)<segment.end) visible=segment.causal?r+1:uint32_t(segment.end);
        std::vector<float> scores(visible); float maximum=-INFINITY;
        for(uint32_t j=0;j<visible;++j) {
            float dot=0; for(uint32_t d=0;d<128;++d) dot+=q[(size_t(hd)*rows+r)*128+d]*k[(size_t(hd)*kvrows+j)*128+d];
            scores[j]=dot*0.08838834764831844f; maximum=std::max(maximum,scores[j]);
        }
        float total=0; for(auto& score:scores) { score=std::exp(score-maximum); total+=score; }
        for(uint32_t d=0;d<128;++d) {
            float sum=0; for(uint32_t j=0;j<visible;++j) sum+=scores[j]/total*v[(size_t(hd)*kvrows+j)*128+d];
            attention[(size_t(hd)*rows+r)*128+d]=round_bf16(sum);
        }
    }
    compare(head_values(w.attended),attention,"tiled-attention");
    auto projected=linear(unpack(attention,rows,n),rows,h,b.weights.attention_output);
    auto after=residual(x,projected,rows,h,b.modulation[1]);
    auto ffninput=norm_mod(after,rows,h,b.modulation[2]);
    auto gate=linear(ffninput,rows,h,{b.weights.gate,{}}),up=linear(ffninput,rows,h,{b.weights.up,{}});
    for(size_t i=0;i<gate.size();++i) {
        // CPU checks stage boundaries, not exact Metal fast::exp at BF16 ties.
        float exponential=round_bf16(std::exp(std::abs(gate[i])));
        float y=round_bf16(1.f/round_bf16(1.f+exponential));
        gate[i]=round_bf16(round_bf16(gate[i]*round_bf16(gate[i]<0?y:1-y))*up[i]);
    }
    auto feed=linear(gate,rows,b.weights.gate.rows,{b.weights.down,{}});
    auto out=residual(after,feed,rows,h,b.modulation[3]);
    compare(values(w.projected_attention),projected,"attn-out-projection");
    compare(values(w.attention_residual),after,"attn-residual");
    compare(values(w.ffn_hidden),gate,"ffn-hidden");
    compare(values(w.ffn_output),feed,"ffn-output");
    compare(values(b.output),out,"block-output");
    return out;
}
void linear_case(Backend& backend,uint32_t k,uint32_t n,bool with_bias,ScalarType output_type) {
    auto prepared=NativeLinear::prepare(backend,{k,n});
    LinearBindings b{matrix(backend,33,k),matrix(backend,n,k),matrix(backend,33,n,output_type),{}};
    if(with_bias) b.bias=matrix(backend,1,n);
    fill(b.input,1);fill(b.weight,3);if(b.bias)fill(*b.bias,5,256);
    std::vector<Snapshot> read{snapshot(b.input.storage),snapshot(b.weight.storage)};
    if(b.bias)read.push_back(snapshot(b.bias->storage));
    std::vector<float> expected(size_t(b.output.rows)*n);
    for(uint32_t r=0;r<b.output.rows;++r)for(uint32_t c=0;c<n;++c) {
        float sum=0;for(uint32_t i=0;i<k;++i)sum+=load(b.input,r,i)*load(b.weight,c,i);
        if(output_type==ScalarType::BF16)sum=round_bf16(sum);
        if(b.bias)sum+=load(*b.bias,0,c);
        expected[size_t(r)*n+c]=output_type==ScalarType::BF16?round_bf16(sum):sum;
    }
    CommandGraph graph;prepared.encode(graph,b);
    auto completed=backend.submit(std::move(graph)).wait();
    check(completed.ok && completed.physically_complete,"native linear did not physically complete");
    const auto e=compare(values(b.output),expected,"linear-64-tail-bias");
    check(e.max_abs<1e-6,"native linear exact powers-of-two oracle mismatch");
    matrix_canaries(b.input);matrix_canaries(b.weight);matrix_canaries(b.output);
    if(b.bias)matrix_canaries(*b.bias);
    for(const auto& s:read)unchanged(s);
    std::cout<<"linear_case=PASS K="<<k<<" N="<<n<<" bias="<<with_bias<<" f32="<<(output_type==ScalarType::FP32)<<'\n';
}
BlockBindings bindings(Backend& b,uint32_t rows,uint32_t n,BlockPhase phase,uint32_t prefix) {
    const uint32_t h=n*128,f=256;
    BlockBindings x;
    x.phase=phase;x.prefix_tokens=prefix;x.input=matrix(b,rows,h);x.output=matrix(b,rows,h);
    fill(x.input,1,16);
    for(uint32_t i=0;i<4;++i) { x.modulation[i]=matrix(b,phase==BlockPhase::PrefixBuild?rows:1,h);fill(x.modulation[i],i+2,64); }
    x.cosine=matrix(b,rows,64,ScalarType::FP32);x.sine=matrix(b,rows,64,ScalarType::FP32);
    for(uint32_t r=0;r<rows;++r)for(uint32_t d=0;d<64;++d) {
        const float angle=float((r+prefix)*3+d)/512;
        store(x.cosine,r,d,std::cos(angle));store(x.sine,r,d,std::sin(angle));
    }
    Projection* projections[]={&x.weights.query,&x.weights.key,&x.weights.value,&x.weights.attention_output};
    for(uint32_t i=0;i<4;++i) { projections[i]->weight=matrix(b,h,h);fill(projections[i]->weight,3+i,128); }
    x.weights.query_norm=matrix(b,1,128);x.weights.key_norm=matrix(b,1,128);
    for(uint32_t d=0;d<128;++d){store(x.weights.query_norm,0,d,1+float(int(d%7)-3)/16);store(x.weights.key_norm,0,d,1+float(int(d%5)-2)/16);}
    x.weights.gate=matrix(b,f,h);x.weights.up=matrix(b,f,h);x.weights.down=matrix(b,h,f);
    fill(x.weights.gate,9,128);fill(x.weights.up,11,128);fill(x.weights.down,13,128);
    if(prefix) {
        x.prefix=PrefixKv{heads(b,n,prefix),heads(b,n,prefix)};
        if(phase==BlockPhase::SteadyDecode) for(uint32_t hd=0;hd<n;++hd)for(uint32_t r=0;r<prefix;++r)for(uint32_t d=0;d<128;++d) {
            const size_t ko=size_t(hd)*x.prefix->key.head_stride_bytes+size_t(r)*x.prefix->key.token_stride_bytes+d*2;
            const size_t vo=size_t(hd)*x.prefix->value.head_stride_bytes+size_t(r)*x.prefix->value.token_stride_bytes+d*2;
            reinterpret_cast<uint16_t*>(static_cast<void*>(static_cast<std::byte*>(x.prefix->key.storage.contents())+ko))[0]=bf16(float(int((r+d+hd)%13)-6)/32);
            reinterpret_cast<uint16_t*>(static_cast<void*>(static_cast<std::byte*>(x.prefix->value.storage.contents())+vo))[0]=bf16(float(int((r+d+hd)%11)-5)/64);
        }
    }
    if(phase==BlockPhase::PrefixBuild) {
        // Two causal text segments exercise nonzero logical query offsets.
        if(prefix>1) x.segments={{0,2,true,0,-1},{2,int(prefix),true,2,-1},{int(prefix),int(rows),false,-1,0}};
        else x.segments={{0,int(rows),false,-1,0}};
    }
    return x;
}
void block_case(Backend& backend,uint32_t rows,uint32_t n,BlockPhase phase,uint32_t prefix) {
    auto prepared=NativeBlock::prepare(backend,{n*128,n,128,256});
    auto b=bindings(backend,rows,n,phase,prefix);
    auto w=prepared.allocate_workspace(backend,rows,prefix,phase,StorageMode::Shared);
    std::vector<Snapshot> read{snapshot(b.input.storage),snapshot(b.cosine.storage),snapshot(b.sine.storage),
        snapshot(b.weights.query.weight.storage),snapshot(b.weights.key.weight.storage),snapshot(b.weights.value.weight.storage),
        snapshot(b.weights.attention_output.weight.storage),snapshot(b.weights.query_norm.storage),snapshot(b.weights.key_norm.storage),
        snapshot(b.weights.gate.storage),snapshot(b.weights.up.storage),snapshot(b.weights.down.storage)};
    for(const auto& m:b.modulation)read.push_back(snapshot(m.storage));
    if(b.prefix && phase==BlockPhase::SteadyDecode) {
        read.push_back(snapshot(b.prefix->key.storage));read.push_back(snapshot(b.prefix->value.storage));
    }
    // Reject a late FFN binding or cross-step alias before modifying any
    // previously encoded outer graph. Rejection must not submit GPU work.
    CommandGraph invalid; auto sentinel=backend.make_event();invalid.signal(sentinel,1);
    auto bad=b;bad.weights.down.columns--;
    bool rejected=false;try{prepared.encode(invalid,bad,w);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected && invalid.steps().size()==1,"late binding left partial block in graph");
    bad=b;bad.weights.down.storage=w.raw_query.storage;
    rejected=false;try{prepared.encode(invalid,bad,w);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected && invalid.steps().size()==1,"cross-step immutable alias accepted");
    CommandGraph graph;prepared.encode(graph,b,w);const size_t steps=graph.steps().size();
    const auto result=backend.submit(std::move(graph)).wait();
    check(result.ok && result.physically_complete,"native Qwen block did not physically complete");
    (void)oracle(b,w,n);
    matrix_canaries(b.input);matrix_canaries(b.output);matrix_canaries(b.cosine);matrix_canaries(b.sine);
    for(const auto& m:b.modulation)matrix_canaries(m);
    if(b.prefix){head_canaries(b.prefix->key);head_canaries(b.prefix->value);}
    for(const auto& s:read)unchanged(s);
    std::cout<<"block_case=PASS rows="<<rows<<" heads="<<n<<" prefix="<<prefix
             <<" phase="<<(phase==BlockPhase::PrefixBuild?"prefill":"steady")
             <<" steps="<<steps<<" workspace="<<w.reserved_bytes<<" gpu_seconds="<<result.gpu_seconds<<'\n';
}
}
int main(int argc,char** argv) {
    if(argc!=2 || std::string_view(argv[1])!="--tiny-device") {
        std::cout<<"usage: native-qwen-block-component --tiny-device (synthetic only)\n";return 2;
    }
    try {
        Backend backend;std::cout<<"device="<<backend.device_name()<<" models=0 app_launches=0\n";
        linear_case(backend,64,64,false,ScalarType::BF16);
        linear_case(backend,64,192,true,ScalarType::BF16);
        linear_case(backend,192,64,true,ScalarType::FP32);
        block_case(backend,37,1,BlockPhase::PrefixBuild,5);
        block_case(backend,33,1,BlockPhase::SteadyDecode,5);
        block_case(backend,3,2,BlockPhase::SteadyDecode,0);
        std::cout<<"native_qwen_block_component=PASS full_model=0 e2e=0\n";
    } catch(const std::exception& e) { std::cerr<<"native_qwen_block_component=FAIL "<<e.what()<<'\n';return 1; }
}
