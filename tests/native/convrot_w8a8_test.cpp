#include "../../native/third_party/cider/w8a8_primitive.h"
#include <mlx/mlx.h>
#ifndef TC_CONVROT_PRIMITIVE_ONLY
#include "../../native/backends/mlx.hpp"
#include "turbocider/turbocider.h"
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using mx::array;
void check(bool ok,const char* reason) { if(!ok)throw std::runtime_error(reason); }
std::vector<float> values(const array& a) {
    auto f=mx::contiguous(mx::astype(a,mx::float32));mx::eval(f);
    return {f.data<float>(),f.data<float>()+f.size()};
}
bool same_bytes(const array& a,const array& b) {
    if(a.shape()!=b.shape() || a.dtype()!=b.dtype())return false;
    // Compare original dtype bytes in logical order, including signed zero.
    // Floating conversion or raw traversal of a strided view can hide errors.
    auto first=mx::view(mx::contiguous(a),mx::uint8);
    auto second=mx::view(mx::contiguous(b),mx::uint8);
    mx::eval(first,second);
    return std::memcmp(first.data<uint8_t>(),second.data<uint8_t>(),first.nbytes())==0;
}
double relative(const std::vector<float>& actual,const std::vector<float>& expected) {
    check(actual.size()==expected.size(),"output geometry changed");
    double error=0,denominator=0;
    for(size_t i=0;i<actual.size();++i) {
        check(std::isfinite(actual[i]),"nonfinite W8A8 output");
        error+=std::pow(double(actual[i])-expected[i],2);
        denominator+=std::pow(double(expected[i]),2);
    }
    return std::sqrt(error/std::max(denominator,1e-30));
}
std::vector<float> oracle(const array& x,const array& w,const array& scales,bool quantize) {
    auto a=values(x),q=values(w),s=values(scales);
    const int m=x.shape(0),k=x.shape(1),n=w.shape(0);
    std::vector<float> out(size_t(m)*n);
    for(int row=0;row<m;++row) {
        float peak=0;for(int c=0;c<k;++c)peak=std::max(peak,std::abs(a[size_t(row)*k+c]));
        const float scale=peak==0?1:peak/127;
        for(int col=0;col<n;++col) {
            int64_t dot=0;double full=0;
            for(int c=0;c<k;++c) {
                const float v=a[size_t(row)*k+c];
                const int weight=int(q[size_t(col)*k+c]);
                dot+=int64_t(std::clamp(std::nearbyint(v/scale),-127.f,127.f))*weight;
                full+=double(v)*weight;
            }
            out[size_t(row)*n+col]=quantize ? float(float(dot)*scale*s[col]) : float(full*s[col]);
        }
    }
    return out;
}
std::pair<array,array> weights(int n,int k) {
    std::vector<int8_t> codes(size_t(n)*k);
    std::vector<float> scales(n);
    for(size_t i=0;i<codes.size();++i) {
        uint32_t bits=uint32_t(i)+71u;
        bits=(bits^(bits>>16))*0x7feb352du;
        bits=(bits^(bits>>15))*0x846ca68bu;
        codes[i]=int8_t(int((bits^(bits>>16))>>24)-128);
    }
    for(int i=0;i<n;++i)scales[i]=float(i%11-5)*.00012345f;
    return {array(codes.data(),{n,k},mx::int8),array(scales.data(),{n,1},mx::float32)};
}
array input(int m,int k,mx::Dtype dtype,bool exact,bool huge=false) {
    std::vector<float> source(size_t(m)*k);
    for(size_t i=0;i<source.size();++i)source[i]=exact ? float(int(i%255)-127)/8.f :
        float(std::sin(double(i)*.73521)*.31+std::cos(double(i)*.017)*.1);
    if(huge)for(auto& v:source)v*=131072.f;
    // Explicit zero row exercises zero/negative source row scales together.
    if(m>1)std::fill_n(source.begin(),k,0.f);
    return mx::astype(array(source.data(),{m,k},mx::float32),dtype);
}
void primitive_cases() {
    double max_oracle=0,max_approximation=0;
    auto before=cider::convrot_w8a8_stats();uint64_t calls=0,rows=0;
    for(const auto dtype:{mx::bfloat16,mx::float16,mx::float32})
        for(const auto shape:{std::array{1,33,256},std::array{33,129,512},std::array{65,1,768},std::array{129,128,256}}) {
            const auto [m,n,k]=shape;auto [w,s]=weights(n,k);
            for(bool exact:{true,false}) {
                auto x=input(m,k,dtype,exact);
                auto y=cider::convrot_linear(x,w,s);mx::eval(y);
                check(y.dtype()==mx::float32,"ConvRot primitive rounded before model I/O boundary");
                const auto result=values(y);
                max_oracle=std::max(max_oracle,relative(result,oracle(x,w,s,true)));
                max_approximation=std::max(max_approximation,relative(result,oracle(x,w,s,false)));
                ++calls;rows+=m;
            }
        }
    // BF16 values outside FP16 range must not overflow inside the primitive.
    auto [w,s]=weights(17,256);auto large=input(3,256,mx::bfloat16,true,true);
    auto result=cider::convrot_linear(large,w,s);mx::eval(result);
    check(relative(values(result),oracle(large,w,s,true))<2e-6,"BF16 activation was narrowed through FP16");
    ++calls;rows+=3;
    // Strided input/weight views, compiled execution, changed inputs, and no
    // pointer-identity cache of prior activation codes.
    auto x=input(33,512,mx::bfloat16,false);
    auto cut=mx::slice(x,{0,0},{33,256});auto wcut=mx::slice(w,{1,0},{16,256});auto scut=mx::slice(s,{1,0},{16,1});
    auto fn=mx::compile([wcut,scut](const std::vector<array>& a){return std::vector<array>{cider::convrot_linear(a[0],wcut,scut)};});
    for(float factor:{1.f,2.f}) {
        auto changed=cut*array(factor,mx::bfloat16);auto y=fn({changed})[0];mx::eval(y);
        const auto error=relative(values(y),oracle(changed,wcut,scut,true));
        if(error>=2e-6) {
            auto eager=cider::convrot_linear(changed,wcut,scut);mx::eval(eager);
            std::cerr<<"compiled_error="<<error<<" eager_error="<<relative(values(eager),oracle(changed,wcut,scut,true))
                <<" compiled_eager="<<relative(values(y),values(eager))<<" primitive_max="<<max_oracle<<'\n';
        }
        check(error<2e-6,"compiled/strided source projection changed");
        ++calls;rows+=33;
    }
    check(max_oracle<2e-6,"INT8 TensorOps differs from independent CPU integer-dot oracle");
    if(max_approximation>=.025)std::cerr<<"max_cpu_a8_rel="<<max_oracle<<" max_float_reference_rel="<<max_approximation<<'\n';
    check(max_approximation<.025,"synthetic A8 projection error exceeds 2.5 percent");
    bool rejected=false;try{(void)cider::convrot_linear(large,mx::astype(w,mx::uint8),s);}catch(const std::exception&){rejected=true;}
    check(rejected,"unsigned/packed source accidentally accepted");
    // An invalid activation must remain visible, including with a zero
    // checkpoint row coefficient; it cannot become a plausible zero token.
    for(float invalid:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        std::vector<float> bad(256,1.f);bad[177]=invalid;
        auto out=values(cider::convrot_linear(array(bad.data(),{1,256},mx::float32),w,s));
        check(std::all_of(out.begin(),out.end(),[](float v){return !std::isfinite(v);}),
            "invalid activation was silently converted to a finite projection");
        ++calls;++rows;
    }
    auto after=cider::convrot_w8a8_stats();
    check(after.projections-before.projections==calls && after.activation_rows-before.activation_rows==rows,
        "actual GPU W8A8 receipt differs from evaluated projections");
    std::cout<<"primitive_calls="<<calls<<" rows="<<rows<<" max_cpu_a8_rel="<<max_oracle<<" max_float_reference_rel="<<max_approximation<<'\n';
}
void typed_output_cases() {
    const std::array<mx::Dtype,3> dtypes{mx::bfloat16,mx::float16,mx::float32};
    const auto stream=mx::to_stream(mx::StreamOrDevice{});
    const cider::ConvrotW8A8Linear fp32(stream,mx::float32),fp32_again(stream,mx::float32),
        bf16(stream,mx::bfloat16),fp16(stream,mx::float16);
    check(fp32.is_equivalent(fp32_again) && !fp32.is_equivalent(bf16) &&
          !fp32.is_equivalent(fp16) && !bf16.is_equivalent(fp16),
          "ConvRot primitive equivalence omitted its output dtype");
    uint64_t checked=0;
    auto check_cast=[&](const array& x,const array& w,const array& s,const array& baseline,mx::Dtype dtype) {
        const auto actual=cider::convrot_linear_typed(x,w,s,dtype);
        check(actual.shape()==baseline.shape() && actual.dtype()==dtype,
              "typed ConvRot output geometry/dtype changed");
        check(same_bytes(actual,mx::astype(baseline,dtype)),
              "typed dequant store differs from the original F32 projection then cast");
        ++checked;
    };
    // Both GEMM selectors, their M=64 boundary, unaligned N, and a partial
    // 512-wide K iteration. The original F32 path remains CPU-oracle checked.
    for(const auto input_dtype:dtypes)
        for(const auto shape:{std::array{1,129,256},std::array{33,129,512},
                              std::array{64,127,768},std::array{65,1,768},std::array{129,128,256}}) {
            const auto [m,n,k]=shape;const auto [w,s]=weights(n,k);
            const auto x=input(m,k,input_dtype,false);
            const auto baseline=cider::convrot_linear(x,w,s);mx::eval(baseline);
            check(baseline.dtype()==mx::float32 && relative(values(baseline),oracle(x,w,s,true))<2e-6,
                  "typed-output test lost the original F32 integer-dot contract");
            for(const auto output_dtype:dtypes)check_cast(x,w,s,baseline,output_dtype);
        }
    // Actual DiT channel dimensions, including the 24576-row fused gate/up
    // matrix. Inputs/codes are synthetic; this exercises real Metal dispatch,
    // physical row pitch and complete output arrays without loading a model.
    for(const auto shape:{std::array{1,4096,4096},std::array{17,24576,4096},std::array{65,4096,12288}}) {
        const auto [m,n,k]=shape;const auto [w,s]=weights(n,k);
        const auto x=input(m,k,mx::bfloat16,false);
        const auto baseline=cider::convrot_linear(x,w,s);mx::eval(baseline);
        for(const auto dtype:dtypes)check_cast(x,w,s,baseline,dtype);
    }
    // A compiled graph publishes three dtypes for the same source operands.
    // Replay changed inputs to reject stale outputs and unsafe dtype CSE.
    auto [w,s]=weights(131,512);
    const auto wcut=mx::slice(w,{1,0},{130,256}),scut=mx::slice(s,{1,0},{130,1});
    auto compiled=mx::compile([wcut,scut](const std::vector<array>& a) {
        return std::vector<array>{cider::convrot_linear_typed(a[0],wcut,scut,mx::bfloat16),
            cider::convrot_linear_typed(a[0],wcut,scut,mx::float16),
            cider::convrot_linear_typed(a[0],wcut,scut,mx::float32)};
    });
    const auto wide=input(65,512,mx::bfloat16,false);
    const auto cut=mx::slice(wide,{0,0},{65,256});
    for(float factor:{1.f,2.f}) {
        const auto changed=factor==1.f ? cut : cut*array(factor,mx::bfloat16);
        const auto baseline=cider::convrot_linear(changed,wcut,scut);
        const auto actual=compiled({changed});mx::eval(actual);
        for(size_t i=0;i<dtypes.size();++i) {
            check(actual[i].dtype()==dtypes[i] && same_bytes(actual[i],mx::astype(baseline,dtypes[i])),
                  "compiled typed ConvRot mixed dtype/strided/changed-input result changed");
            ++checked;
        }
    }
    // BF16 values above FP16 range stay wide until the final output cast.
    // F16 output may overflow, exactly as an explicit F32-to-F16 cast does.
    const auto large=mx::full({2,256},131072.f,mx::bfloat16);
    const auto positive_codes=mx::full({3,256},127,mx::int8);
    const std::array<float,3> edge_scales{1.f,0.f,-0.f};
    const auto edge_s=array(edge_scales.data(),{3,1},mx::float32);
    const auto large_baseline=cider::convrot_linear(large,positive_codes,edge_s);
    for(const auto dtype:dtypes)check_cast(large,positive_codes,edge_s,large_baseline,dtype);
    // Nonfinite activation rows must remain nonfinite, including zero source
    // scales. Compare cast bytes as well as classification; never hide NaNs
    // behind an allclose or a plausible finite-zero projection.
    for(float invalid:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        std::vector<float> source(256,1.f);source[177]=invalid;
        const auto bad=array(source.data(),{1,256},mx::float32);
        const auto baseline=cider::convrot_linear(bad,positive_codes,edge_s);
        for(const auto dtype:dtypes) {
            check_cast(bad,positive_codes,edge_s,baseline,dtype);
            const auto result=values(cider::convrot_linear_typed(bad,positive_codes,edge_s,dtype));
            check(std::all_of(result.begin(),result.end(),[](float v){return !std::isfinite(v);}),
                  "typed ConvRot hid a nonfinite activation with a zero scale");
        }
    }
    // The new API must not pull bias into FP32 before the output cast. Each
    // fixture lands on an RNE midpoint where those two orders differ.
    std::vector<int8_t> one(256,0);one[0]=1;
    const auto one_code=array(one.data(),{1,256},mx::int8);
    const auto ones=mx::ones({1,256},mx::bfloat16);
    for(const auto dtype:{mx::bfloat16,mx::float16}) {
        const float half_ulp=dtype==mx::bfloat16 ? 0.00390625f : 0.00048828125f;
        const auto scale=mx::full({1,1},1.f+half_ulp,mx::float32);
        const auto bias=mx::full({1},half_ulp,dtype);
        const auto baseline=cider::convrot_linear(ones,one_code,scale);
        const auto expected=mx::astype(baseline,dtype)+bias;
        const auto actual=cider::convrot_linear_typed(ones,one_code,scale,dtype)+bias;
        const auto wrong=mx::astype(baseline+mx::astype(bias,mx::float32),dtype);
        check(same_bytes(actual,expected) && !same_bytes(expected,wrong),
              "external typed bias rounding boundary changed or midpoint fixture is ineffective");
        ++checked;
    }
    bool rejected=false;
    try {(void)cider::convrot_linear_typed(ones,one_code,mx::ones({1,1},mx::float32),mx::int8);}
    catch(const std::invalid_argument&) {rejected=true;}
    check(rejected,"typed ConvRot accepted a non-floating output dtype");
    std::cout<<"typed_output_cases="<<checked<<" eager_compiled_full_bytes=PASS f32_api_unchanged=true bias_after_cast=true\n";
}
#ifndef TC_CONVROT_PRIMITIVE_ONLY
void shared_weights_cases() {
    using namespace tc;configure_streams();Weights source;
    auto [w,s]=weights(33,512);auto tag=mx::zeros({1},mx::uint8);
    auto bias=mx::full({33},.125f,mx::bfloat16);
    source.bind_arrays({"p.weight","p.weight_scale","p.comfy_quant","p.bias"},{w,s,tag,bias});
    check(!source.convrot_w8a8(),"Weights owner flag must be configured explicitly");
    source.set_convrot_w8a8(true);check(source.at("p.weight").dtype()==mx::int8,"original codes mutated");
    auto x=mx::reshape(input(33,512,mx::bfloat16,false),{1,33,512});
    auto full=source.project(x,"p");auto many=source.project_many(x,{"p","p"});mx::eval(full,many[0],many[1]);
    check(values(full)==values(many[0]) && values(full)==values(many[1]),"shared projection or bias boundary changed");
    auto slice=source.project_base_slice(x,"p",1,32,0,512,false);
    auto partial=source.project_base_slice_fp32(x,"p",1,32,0,512);mx::eval(slice,partial);
    check(values(slice)==values(mx::astype(partial,x.dtype())),"FP32 partial was narrowed before the join");
    // A nonzero column offset must retain physical row pitch and the original
    // coefficient. Compare against a separately rebound complete H256 source.
    Weights right;right.bind_arrays({"right.weight","right.weight_scale","right.comfy_quant"},
        {mx::slice(w,{1,256},{32,512}),mx::slice(s,{1,0},{32,1}),tag});
    right.set_convrot_w8a8(true);
    auto xright=mx::slice(x,{0,0,256},{1,33,512});
    auto sliced=source.project_range(xright,"p",1,32,256,512);
    auto sliced_fp32=source.project_range_fp32(xright,"p",1,32,256,512);
    auto rebound=right.project(xright,"right");mx::eval(sliced,sliced_fp32,rebound);
    check(values(sliced)==values(rebound) && values(sliced)==values(mx::astype(sliced_fp32,x.dtype())),
        "column slice lost original ConvRot pitch or FP32 output boundary");
    check(source.at("p.weight").id()==w.id() && source.at("p.weight_scale").id()==s.id(),"source identity changed");
    bool rejected=false;try{source.pack_convrot_q8(128,mx::float32);}catch(const std::exception&){rejected=true;}
    check(rejected,"packed recipe replaced live raw W8A8 source");
    source.set_convrot_w8a8(false);auto old=source.project(x,"p");mx::eval(old);
    check(relative(values(full),values(old))<.025,"full ConvRot projection exceeds approximation budget");
    std::cout<<"shared_weights=PASS explicit_owner_flag=true source_unchanged=true\n";
}
void qwen_lora_cases(const std::filesystem::path &adapter, bool has_w8a8) {
    using namespace tc;
    configure_streams();
    constexpr int m=3, k=256, width=32, rank=2;
    const std::string mlp="transformer_blocks.0.img_mlp.";
    const std::string fused=mlp+"gate_up";
    const auto [codes,scales]=weights(2*width,k);
    const auto tag=mx::zeros({1},mx::uint8);
    const auto x=mx::reshape(mx::astype(mx::arange(m*k,mx::int32)%17-8,mx::bfloat16)*
                            Tensor(1.f/32.f,mx::bfloat16),{1,m,k});
    const auto x_values=values(x);
    std::unordered_map<std::string,Tensor> factors;
    for(int half=0;half<2;++half) {
        const std::string target=mlp+(half ? "proj" : "gate_layer");
        factors.emplace(target+".lora_A.weight",mx::reshape(
            mx::astype((mx::arange(rank*k,mx::int32)*(half+2))%13-6,mx::float32)/32.f,{rank,k}));
        factors.emplace(target+".lora_B.weight",mx::reshape(
            mx::astype((mx::arange(width*rank,mx::int32)*(half+3))%11-5,mx::float32)/64.f,{width,rank}));
        factors.emplace(target+".alpha",Tensor(half ? 1.f : 2.f));
    }
    mx::save_safetensors(adapter.string(),factors);
    const auto no_event=[](const std::string&,int,int){};
    std::atomic<bool> cancel{false};
    auto init=[&](int mode) {
        Weights source;
        source.bind_arrays({fused+".weight",fused+".weight_scale",fused+".comfy_quant"},{codes,scales,tag});
        if(mode==1) source.pack_convrot_q8(128,mx::float32);
        if(mode==2) source.set_convrot_w8a8(true);
        return source;
    };
    for(int mode=0;mode<(has_w8a8?3:2);++mode) {
        auto source=init(mode);
        const auto base=source.project(x,fused);
        const auto base_values=values(base);
        const auto master=source.at(fused+".weight").id();
        const auto partial=mode ? values(source.project_base_slice_fp32(x,fused,0,width,0,k)) : std::vector<float>{};
        for(float strength:{0.f,.75f}) {
            check(source.apply_loras({{adapter.string(),strength,"transformer"}},"transformer",no_event,cancel,true,true)==2,
                  "Qwen split gate/up aliases did not bind to both ConvRot halves");
            const auto result=values(source.project(x,fused));
            if(strength==0.f) {
                check(result==base_values,"strength-zero ConvRot LoRA changed the projection");
                continue;
            }
            auto expected=base_values;
            for(int half=0;half<2;++half) {
                const std::string target=mlp+(half ? "proj" : "gate_layer");
                const auto a=values(factors.at(target+".lora_A.weight"));
                const auto b=values(factors.at(target+".lora_B.weight"));
                for(int row=0;row<m;++row) {
                    double low[rank]={};
                    for(int r=0;r<rank;++r)
                        for(int c=0;c<k;++c) low[r]+=double(x_values[row*k+c])*a[r*k+c];
                    for(int out=0;out<width;++out) {
                        double delta=0;
                        for(int r=0;r<rank;++r) delta+=low[r]*b[out*rank+r];
                        expected[row*2*width+half*width+out]+=float(delta*strength*(half?.5:1.));
                    }
                }
            }
            check(result==values(mx::astype(Tensor(expected.data(),{1,m,2*width},mx::float32),mx::bfloat16)) &&
                      result!=base_values,"ConvRot LoRA differs from unrotated-input, separate-half CPU reference");
        }
        check(source.at(fused+".weight").id()==master,"online ConvRot LoRA replaced source weight storage");
        check(!mode || values(source.project_base_slice_fp32(x,fused,0,width,0,k))==partial,
              "checkpoint-only channel partial consumed online LoRA");
    }
    factors.insert_or_assign(mlp+"proj.lora_A.weight",mx::zeros({rank,k/4},mx::float32));
    mx::save_safetensors(adapter.string(),factors);
    auto invalid=init(1);
    bool denied=false;
    try { invalid.apply_loras({{adapter.string(),1.f,"transformer"}},"transformer",no_event,cancel,true,true); }
    catch(const std::exception&) { denied=true; }
    check(denied && !invalid.has_runtime_loras() && invalid.bytes()==0,
          "physical packed width accepted as LoRA width, or invalid binding remained visible");
    std::cout<<"qwen_convrot_lora=PASS raw=true packed=true w8a8="<<(has_w8a8?"true":"false")<<'\n';
}
#endif
int main(int argc,char** argv) { try {
#ifndef TC_CONVROT_PRIMITIVE_ONLY
    char* raw_identity=tc_runtime_build_identity();check(raw_identity,"native identity missing");
    const std::string identity(raw_identity);tc_string_free(raw_identity);
    check(identity==TC_TEST_BOUND_NATIVE_ID,"ConvRot Weights ABI requires the matching library");
#else
    (void)argc;(void)argv;
#endif
    const bool has_w8a8=cider::convrot_w8a8_available();
#ifndef TC_CONVROT_PRIMITIVE_ONLY
    if(argc==2) qwen_lora_cases(argv[1],has_w8a8);
#endif
    if(!has_w8a8){std::cout<<"SKIP M5 TensorOps unavailable\n";return 0;}
    mx::set_default_device(mx::Device::gpu);primitive_cases();typed_output_cases();
#ifndef TC_CONVROT_PRIMITIVE_ONLY
    shared_weights_cases();
#endif
    std::cout<<"PASS ConvRot original-code W8A8 CPU oracle, dtype/range, padding, compiled and GPU receipts\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
