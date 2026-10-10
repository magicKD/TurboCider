// Opt-in tiny full-DiT comparison against the existing MLX Transformer. No
// checkpoint files, approximate Transformer oracle, VAE or image generation.
#include "../../native/models/qwen21/transformer.hpp"
#include "../../native/models/qwen21/mlx_native_denoiser.hpp"
#include "../../native/backends/metal_graph/mlx_bridge.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>

using namespace tc;
namespace mg = tc::metal_graph;
namespace ng = tc::qwen21::native_graph;

namespace {
constexpr int hidden=128, head_count=1, dimensions=128, layers=2, ffn=256, channels=64;
constexpr double maximum_rrms=0.01, maximum_abs=0.01;
constexpr const char* source_revision="tiny-qwen-bf16-deterministic-padded-v1";

void check(bool condition,const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Exception,class Function>
void rejects(Function&& fn,const char* message) {
    bool rejected=false;
    try { fn(); } catch (const Exception&) { rejected=true; }
    check(rejected,message);
}
uint32_t mix(uint32_t value) {
    value^=value>>16; value*=0x7feb352dU;
    value^=value>>15; value*=0x846ca68bU;
    return value^(value>>16);
}
mg::MatrixView matrix(mg::Backend& backend,const Tensor& value) {
    const auto imported=mg::import_mlx_matrix(backend,value);
    return {imported.storage,imported.rows,imported.columns,imported.row_stride_bytes};
}
uint64_t hash_bytes(const mg::MatrixView& full) {
    check(full.storage.contents()!=nullptr,"read-only integrity inspection needs Shared produced backing");
    uint64_t hash=14695981039346656037ULL;
    const auto* bytes=static_cast<const unsigned char*>(full.storage.contents());
    for (size_t i=0;i<full.storage.size_bytes();++i) {
        hash^=bytes[i]; hash*=1099511628211ULL;
    }
    return hash;
}
struct Guard {
    std::string label;
    Tensor backing;
    mg::MatrixView full;
    uint64_t hash;
};
class Fixture {
public:
    explicit Fixture(mg::Backend& backend):backend_(backend) {}
    Tensor padded(const std::string& label,int rows,int columns,uint32_t salt,
                  float amplitude,float center=0.f) {
        const int physical_rows=rows+2, physical_columns=columns+8;
        // One leading/trailing row plus three leading/five trailing columns.
        // The logical source has a nonzero byte offset and a larger row pitch.
        std::vector<float> values(size_t(physical_rows)*physical_columns,0.15625f);
        for (int r=0;r<rows;++r) for (int c=0;c<columns;++c) {
            const auto bits=mix(salt+uint32_t(r)*131071U+uint32_t(c)*8191U);
            const float unit=float(bits&65535U)/32768.f-1.f;
            values[size_t(r+1)*physical_columns+c+3]=center+amplitude*unit;
        }
        auto backing=mx::astype(Tensor(values.data(),{physical_rows,physical_columns},mx::float32),mx::bfloat16);
        auto logical=mx::slice(backing,{1,3},{rows+1,columns+3});
        mx::eval(backing,logical);
        auto full=matrix(backend_,backing);
        const auto visible=matrix(backend_,logical);
        check(visible.storage.offset_bytes()>0 && visible.row_stride_bytes>size_t(columns)*2,
              "fixture must retain nonzero source offset and row padding");
        guards_.push_back({label,backing,full,hash_bytes(full)});
        return logical;
    }
    void verify() const {
        for (const auto& guard:guards_)
            check(hash_bytes(guard.full)==guard.hash,
                  "source or padding modified: "+guard.label);
    }
    size_t count() const noexcept { return guards_.size(); }
private:
    mg::Backend& backend_;
    std::vector<Guard> guards_;
};

void make_weights(Weights& weights,Fixture& fixture) {
    std::vector<std::string> names;
    std::vector<Tensor> tensors;
    uint32_t salt=1701;
    auto projection=[&](const std::string& name,int out,int in,float gain=0.65f) {
        names.push_back(name+".weight");
        tensors.push_back(fixture.padded(name,out,in,salt++,gain*std::sqrt(3.f/float(in))));
    };
    auto norm=[&](const std::string& name,int width,float center) {
        names.push_back(name+".weight");
        tensors.push_back(mx::reshape(fixture.padded(name,1,width,salt++,0.125f,center),{width}));
    };
    projection("time_text_embed.timestep_embedder.linear_1",hidden,256);
    projection("time_text_embed.timestep_embedder.linear_2",hidden,hidden);
    projection("modulation.1",4*hidden,hidden,0.35f);
    projection("img_in",hidden,channels);
    norm("txt_in.text_norm",hidden,0.f);
    projection("txt_in.in_layer",hidden,hidden);
    projection("txt_in.out_layer",hidden,hidden);
    projection("norm_out.linear",hidden,hidden,0.5f);
    projection("proj_out",channels,hidden);
    for (int i=0;i<layers;++i) {
        const auto prefix="transformer_blocks."+std::to_string(i);
        for (const auto* name:{"to_q","to_k","to_v","to_out.0"})
            projection(prefix+".attn."+name,hidden,hidden);
        norm(prefix+".attn.norm_q",dimensions,1.f);
        norm(prefix+".attn.norm_k",dimensions,1.f);
        projection(prefix+".img_mlp.gate_up",2*ffn,hidden);
        projection(prefix+".img_mlp.out",hidden,ffn);
    }
    mx::eval(tensors);
    weights.bind_arrays(names,tensors);
    check(!weights.has_runtime_loras(),"tiny native qualification is BaseOnly");
    fixture.verify();
}
qwen21::TransformerConfig mlx_config() {
    qwen21::TransformerConfig config;
    config.layers=layers; config.heads=head_count; config.head_dim=dimensions;
    config.channels=channels; config.context_dim=hidden;
    config.rope_axes={16,56,56}; config.epsilon=1e-6f;
    return config;
}
struct Case {
    const char* name;
    int height,width,text_rows;
    std::vector<qwen21::ReferenceGeometry> references;
};
constexpr double relative_error_floor=1e-6;
uint32_t bf16_order(uint16_t bits) {
    // Monotonic finite-value order; -0 and +0 intentionally share one point.
    return bits&0x8000U ? 0x8000U-(bits&0x7fffU) : 0x8000U+bits;
}
uint16_t bf16_from_order(uint32_t order) {
    return order>=0x8000U ? uint16_t(order-0x8000U) : uint16_t(0x8000U|(0x8000U-order));
}
float decode_bf16(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits)<<16); }
uint32_t bf16_distance(uint16_t a,uint16_t b) {
    const auto x=bf16_order(a),y=bf16_order(b);
    return x>=y ? x-y : y-x;
}
uint16_t exact_bf16_bits(float value) {
    const auto bits=std::bit_cast<uint32_t>(value);
    check((bits&65535U)==0,"FP32 reference did not exactly widen a BF16 value");
    return uint16_t(bits>>16);
}
struct ErrorPoint {
    uint32_t row=0,column=0;
    float reference=0,native=0;
    uint16_t reference_bits=0,native_bits=0;
    double absolute=0,relative=0;
    uint32_t ulps=0;
};
struct Comparison {
    double rrms=0,max_abs=0,mean_abs=0,reference_peak=0;
    size_t count=0;
    uint32_t maximum_ulps=0;
    std::vector<double> absolute,relative;
    std::array<size_t,6> absolute_bins{},relative_bins{},ulp_bins{};
    std::vector<ErrorPoint> largest;
};
size_t magnitude_bin(double value,const std::array<double,5>& limits) {
    for (size_t i=0;i<limits.size();++i) if (value<=limits[i]) return i;
    return limits.size();
}
double percentile(const std::vector<double>& sorted,double fraction) {
    check(!sorted.empty(),"empty error distribution");
    return sorted[size_t(std::ceil(fraction*double(sorted.size()-1)))];
}
template<size_t N> void print_counts(const std::array<size_t,N>& values) {
    std::cout << '[';
    for (size_t i=0;i<N;++i) { if(i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
}
std::string hex_bits(uint16_t bits) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::setw(4) << std::setfill('0') << bits;
    return stream.str();
}
void print_finite_or_null(double value) {
    if(std::isfinite(value)) std::cout << value;
    else std::cout << "null";
}
void print_point(const ErrorPoint& point) {
    const auto order=bf16_order(point.reference_bits);
    const double lower=double(point.reference)-decode_bf16(bf16_from_order(order-1));
    const double upper=double(decode_bf16(bf16_from_order(order+1)))-point.reference;
    std::cout << "{\"row\":" << point.row << ",\"column\":" << point.column
        << ",\"reference\":" << point.reference << ",\"native\":" << point.native
        << ",\"reference_bf16\":\"" << hex_bits(point.reference_bits)
        << "\",\"native_bf16\":\"" << hex_bits(point.native_bits)
        << "\",\"abs_error\":" << point.absolute << ",\"relative_error\":" << point.relative
        << ",\"bf16_ulps\":" << point.ulps << ",\"reference_lower_ulp_size\":";
    print_finite_or_null(lower);
    std::cout << ",\"reference_upper_ulp_size\":"; print_finite_or_null(upper);
    std::cout << '}';
}
void print_error_diagnostics(const Comparison& result) {
    std::cout << ",\"count\":" << result.count << ",\"reference_peak\":" << result.reference_peak
        << ",\"relative_denominator_floor\":" << relative_error_floor
        << ",\"absolute_error\":{\"mean\":" << result.mean_abs
        << ",\"p50\":" << percentile(result.absolute,0.5) << ",\"p95\":" << percentile(result.absolute,0.95)
        << ",\"p99\":" << percentile(result.absolute,0.99)
        << ",\"bin_upper_bounds\":[0,0.001,0.005,0.01,0.02,null],\"counts\":";
    print_counts(result.absolute_bins);
    std::cout << "},\"relative_error\":{\"p50\":" << percentile(result.relative,0.5)
        << ",\"p95\":" << percentile(result.relative,0.95) << ",\"p99\":" << percentile(result.relative,0.99)
        << ",\"max\":" << result.relative.back()
        << ",\"bin_upper_bounds\":[0,0.001,0.005,0.01,0.05,null],\"counts\":";
    print_counts(result.relative_bins);
    std::cout << "},\"bf16_ulp_error\":{\"signed_zero\":\"coalesced\",\"max\":" << result.maximum_ulps
        << ",\"bin_upper_bounds\":[0,1,2,4,8,null],\"counts\":";
    print_counts(result.ulp_bins);
    std::cout << "},\"largest_abs_errors\":[";
    for(size_t i=0;i<result.largest.size();++i) { if(i) std::cout << ','; print_point(result.largest[i]); }
    std::cout << ']';
}
void diagnostic_cpu_checks() {
    check(bf16_distance(exact_bf16_bits(2.f),exact_bf16_bits(2.015625f))==1,
          "2..4 BF16 one-ULP size should be 0.015625");
    check(bf16_distance(exact_bf16_bits(1.9921875f),exact_bf16_bits(2.f))==1,
          "lower spacing at 2 is 0.0078125");
    check(bf16_distance(exact_bf16_bits(-2.f),exact_bf16_bits(-2.015625f))==1,
          "negative BF16 order reversed incorrectly");
    check(bf16_distance(0x8000,0)==0 && bf16_distance(0x8001,1)==2,"signed-zero BF16 order contract");
    for(uint16_t bits:{uint16_t(0x3f80),uint16_t(0xbf80),uint16_t(0x0001),uint16_t(0x8001)})
        check(bf16_from_order(bf16_order(bits))==bits,"BF16 order roundtrip");
}
Comparison compare(const mg::MatrixView& native,const Tensor& expected) {
    check(native.storage.scalar_type()==mg::ScalarType::BF16 && native.storage.contents(),
          "native final output must be physically completed Shared BF16");
    check(expected.dtype()==mx::bfloat16,"reference Euler must still publish BF16 latents");
    auto reference=mx::astype(expected,mx::float32);
    mx::eval(reference);
    check(reference.flags().row_contiguous,"error diagnostics require contiguous widened reference");
    check(reference.size()==size_t(native.rows)*native.columns,"final latent element count mismatch");
    const float* values=reference.data<float>();
    const auto* bytes=static_cast<const std::byte*>(native.storage.contents());
    double squares=0,energy=0;
    Comparison result;
    for (uint32_t r=0;r<native.rows;++r) {
        const auto* row=reinterpret_cast<const uint16_t*>(bytes+size_t(r)*native.row_stride_bytes);
        for (uint32_t c=0;c<native.columns;++c) {
            const float actual=std::bit_cast<float>(uint32_t(row[c])<<16);
            const float oracle=values[size_t(r)*native.columns+c];
            check(std::isfinite(actual) && std::isfinite(oracle),"nonfinite final latent");
            const double difference=double(actual)-oracle;
            squares+=difference*difference; energy+=double(oracle)*oracle;
            const double absolute=std::abs(difference),relative=absolute/std::max(std::abs(double(oracle)),relative_error_floor);
            const auto reference_bits=exact_bf16_bits(oracle);
            const auto ulps=bf16_distance(reference_bits,row[c]);
            result.max_abs=std::max(result.max_abs,absolute); result.mean_abs+=absolute;
            result.reference_peak=std::max(result.reference_peak,std::abs(double(oracle)));
            result.maximum_ulps=std::max(result.maximum_ulps,ulps); ++result.count;
            result.absolute.push_back(absolute); result.relative.push_back(relative);
            ++result.absolute_bins[magnitude_bin(absolute,{0,0.001,0.005,0.01,0.02})];
            ++result.relative_bins[magnitude_bin(relative,{0,0.001,0.005,0.01,0.05})];
            ++result.ulp_bins[magnitude_bin(double(ulps),{0,1,2,4,8})];
            result.largest.push_back({r,c,oracle,actual,reference_bits,row[c],absolute,relative,ulps});
        }
    }
    result.rrms=std::sqrt(squares/std::max(energy,1e-30));
    result.mean_abs/=double(result.count);
    std::sort(result.absolute.begin(),result.absolute.end());
    std::sort(result.relative.begin(),result.relative.end());
    std::stable_sort(result.largest.begin(),result.largest.end(),[](const auto& a,const auto& b) {
        return a.absolute!=b.absolute ? a.absolute>b.absolute : a.ulps>b.ulps;
    });
    result.largest.resize(std::min<size_t>(5,result.largest.size()));
    return result;
}
void run_silu_component() {
    // This mode creates no Transformer, denoiser, checkpoint or model weight.
    // Compare the actual compiled tc::silu primitive to the independent kernel.
    mg::Backend backend;
    constexpr uint32_t rows=8,columns=128,physical_columns=columns+8;
    std::vector<float> values(size_t(rows+2)*physical_columns,0.15625f);
    const std::array<float,16> boundaries{-100.f,-90.f,-16.f,-4.5f,-2.f,-1.f,-0.5f,-0.f,
        0.f,0.5f,1.f,2.f,4.5f,16.f,90.f,100.f};
    for(uint32_t r=0;r<rows;++r) for(uint32_t c=0;c<columns;++c) {
        const auto i=size_t(r)*columns+c;
        values[size_t(r+1)*physical_columns+c+3]=i<boundaries.size()?boundaries[i]:
            float(int(i)-512)/32.f;
    }
    auto backing=mx::astype(Tensor(values.data(),{int(rows+2),int(physical_columns)},mx::float32),mx::bfloat16);
    auto input=mx::slice(backing,{1,3},{int(rows+1),int(columns+3)});
    auto expected=tc::silu(input);
    mx::eval(backing,input,expected);
    const auto source=matrix(backend,input),full=matrix(backend,backing);
    const auto original_hash=hash_bytes(full);
    constexpr size_t prefix=32,pitch=physical_columns*sizeof(uint16_t),suffix=64;
    auto output=backend.allocate(prefix+size_t(rows)*pitch+suffix,"silu-component-output",mg::StorageMode::Shared);
    auto* bytes=static_cast<unsigned char*>(output.contents());
    check(bytes!=nullptr,"SiLU component needs Shared output");
    std::fill_n(bytes,output.size_bytes(),0x5a);
    const mg::MatrixView destination{output.view(prefix,size_t(rows)*pitch,mg::ScalarType::BF16),rows,columns,pitch};
    auto ops=mg::NativeQwenOps::prepare(backend,{columns,1,128});
    mg::CommandGraph graph;
    ops.encode_silu(graph,source,destination);
    auto ticket=backend.submit(std::move(graph));
    const auto completion=ticket.wait();
    check(completion.ok && completion.physically_complete && !completion.timed_out,
          "SiLU component did not physically complete");
    check(hash_bytes(full)==original_hash,"SiLU modified input or its padding");
    for(size_t i=0;i<prefix;++i) check(bytes[i]==0x5a,"SiLU output prefix canary changed");
    for(size_t i=prefix+size_t(rows)*pitch;i<output.size_bytes();++i)
        check(bytes[i]==0x5a,"SiLU output suffix canary changed");
    for(uint32_t r=0;r<rows;++r) for(size_t c=columns*sizeof(uint16_t);c<pitch;++c)
        check(bytes[prefix+size_t(r)*pitch+c]==0x5a,"SiLU output row padding changed");
    const auto result=compare(destination,expected);
    const bool accepted=result.maximum_ulps==0;
    std::cout << std::setprecision(10)
        << "{\"component\":\"silu\",\"reference\":\"actual-compiled-tc-silu\",\"recipe\":\""
        << mg::NativeQwenOps::conditioning_recipe() << "\",\"rrms\":" << result.rrms
        << ",\"max_abs\":" << result.max_abs << ",\"required_bf16_ulps\":0,\"accepted\":"
        << (accepted?"true":"false");
    print_error_diagnostics(result);
    std::cout << "}\n";
    check(accepted,"SiLU differs from actual MLX primitive; keep this a component failure");
    std::cout << "native_silu_component=PASS full_dit_qualification=none real_model_loads=0 e2e_images=0\n";
}
constexpr std::string_view silu_math_diagnostic_source=R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void tc_silu_math_diagnostic(device const bfloat* input [[buffer(0)]],
    device float* output [[buffer(1)]],constant uint& count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=count) return;
    float x=float(input[i]);
    float exponentials[3]={metal::exp(metal::abs(x)),
        metal::fast::exp(metal::abs(x)),metal::precise::exp(metal::abs(x))};
    for(uint mode=0;mode<3;++mode) {
        bfloat e=bfloat(exponentials[mode]);
        bfloat d=bfloat(1.f+float(e));
        bfloat y=bfloat(1.f/float(d));
        bfloat s=x<0.f?y:bfloat(1.f-float(y));
        bfloat result=bfloat(x*float(s));
        uint offset=i*18+mode*6;
        output[offset]=exponentials[mode]; output[offset+1]=float(e);
        output[offset+2]=float(d); output[offset+3]=float(y);
        output[offset+4]=float(s); output[offset+5]=float(result);
    }
}
)metal";
void print_stage(std::string_view name,float value,bool is_bf16) {
    check(std::isfinite(value),"SiLU math diagnostic contains a nonfinite intermediate");
    std::ostringstream bits;
    bits << "0x" << std::hex << std::setw(is_bf16?4:8) << std::setfill('0');
    if(is_bf16) bits << exact_bf16_bits(value);
    else bits << std::bit_cast<uint32_t>(value);
    std::cout << '"' << name << "\":{\"value\":" << value << ",\"bits\":\""
              << bits.str() << "\",\"dtype\":\"" << (is_bf16?"bf16":"fp32") << "\"}";
}
void run_silu_math_diagnostic() {
    const std::array<float,8> values{-6.75f,-6.8125f,-6.84375f,-6.875f,-6.9375f,6.8125f,6.84375f,6.875f};
    constexpr uint32_t count=values.size(),stages=6,modes=3;
    auto input=mx::astype(Tensor(values.data(),{1,int(count)},mx::float32),mx::bfloat16);
    auto exponential=mx::exp(mx::abs(input));
    auto denominator=Tensor(1.f,mx::bfloat16)+exponential;
    auto reciprocal=Tensor(1.f,mx::bfloat16)/denominator;
    auto sigmoid=mx::sigmoid(input);
    auto compiled_silu=tc::silu(input);
    std::vector<Tensor> results{mx::exp(mx::astype(mx::abs(input),mx::float32)),
        exponential,denominator,reciprocal,sigmoid,compiled_silu,input*sigmoid};
    mx::eval(results);
    // Widen only after the actual MLX primitive/compiled outputs materialize.
    for(auto& result:results) result=mx::astype(result,mx::float32);
    mx::eval(results); mx::eval(input);
    mg::Backend backend;
    const auto source=matrix(backend,input);
    const auto source_hash=hash_bytes(source);
    constexpr size_t prefix=32,payload=count*stages*modes*sizeof(float),suffix=64;
    auto buffer=backend.allocate(prefix+payload+suffix,"silu-math-diagnostic",mg::StorageMode::Shared);
    auto* bytes=static_cast<unsigned char*>(buffer.contents());
    check(bytes!=nullptr,"SiLU math diagnostic needs Shared output");
    std::fill_n(bytes,buffer.size_bytes(),0x5a);
    const auto output=buffer.view(prefix,payload,mg::ScalarType::FP32);
    auto library=backend.compile_library(silu_math_diagnostic_source,"silu-math-diagnostic-only");
    auto pipeline=backend.prepare_pipeline(library,"tc_silu_math_diagnostic");
    mg::CommandGraph graph;
    graph.dispatch(pipeline,{source.storage,output},count,{1,1,1},{32,1,1});
    const auto completion=backend.submit(std::move(graph)).wait();
    check(completion.ok && completion.physically_complete && !completion.timed_out,
          "SiLU math diagnostic did not physically complete");
    check(hash_bytes(source)==source_hash,"SiLU math diagnostic modified its input");
    for(size_t i=0;i<prefix;++i) check(bytes[i]==0x5a,"SiLU math diagnostic prefix canary changed");
    for(size_t i=prefix+payload;i<buffer.size_bytes();++i)
        check(bytes[i]==0x5a,"SiLU math diagnostic suffix canary changed");
    const auto* gpu=static_cast<const float*>(output.contents());
    const std::array<std::string_view,7> names{"exp_fp32","exp_bf16","denom_bf16","recip_bf16",
        "sigmoid_bf16","silu_bf16","materialized_silu_bf16"};
    const std::array<std::string_view,3> routes{"native_default_exp","native_fast_exp","native_precise_exp"};
    std::cout << std::setprecision(12);
    for(uint32_t i=0;i<count;++i) {
        std::cout << "{\"diagnostic\":\"silu-math\",\"index\":" << i << ',';
        print_stage("input",values[i],true);
        std::cout << ",\"native_compile_options\":\"safe+precise-functions\",\"mlx\":{";
        for(size_t j=0;j<results.size();++j) {
            if(j) std::cout << ',';
            print_stage(names[j],results[j].data<float>()[i],j!=0);
        }
        std::cout << '}';
        for(uint32_t mode=0;mode<modes;++mode) {
            std::cout << ",\"" << routes[mode] << "\":{";
            for(uint32_t stage=0;stage<stages;++stage) {
                if(stage) std::cout << ',';
                print_stage(names[stage],gpu[i*stages*modes+mode*stages+stage],stage!=0);
            }
            std::cout << '}';
        }
        std::cout << ",\"qualification\":\"none\"}\n";
    }
    std::cout << "native_silu_math=OBSERVED qualification=none real_model_loads=0 e2e_images=0\n";
}
bool run_case(mg::Backend& backend,const ng::NativeDenoiser& denoiser,
              const Weights& weights,Fixture& fixture,const Case& item,bool observe_only) {
    auto latent=mx::expand_dims(fixture.padded(std::string(item.name)+"-latent",
        item.height*item.width,channels,7103,0.5f),0);
    auto text=mx::expand_dims(fixture.padded(std::string(item.name)+"-text",
        item.text_rows,hidden,7207,0.7f),0);
    std::vector<qwen21::ReferenceLatents> references;
    for (size_t i=0;i<item.references.size();++i) {
        const auto g=item.references[i];
        auto value=mx::expand_dims(fixture.padded(std::string(item.name)+"-reference-"+std::to_string(i),
            g.height*g.width,channels,7309+uint32_t(i),0.6f),0);
        mx::eval(value);
        references.push_back({value,g});
    }
    mx::eval(latent,text);
    const std::vector<float> schedule{1.f,0.7f,0.f};
    ng::DenoiserInput input;
    input.latents=matrix(backend,latent); input.text=matrix(backend,text);
    input.height=item.height; input.width=item.width; input.sigmas=schedule;
    input.identity={source_revision,std::string("tiny-condition-")+item.name,"",1};
    for (const auto& reference:references)
        input.references.push_back({matrix(backend,reference.latents),reference.geometry});
    auto wrong_source=input; wrong_source.identity.source_revision="wrong-source";
    rejects<std::invalid_argument>([&]{(void)denoiser.begin(wrong_source);},"source mismatch begin was accepted");
    auto adapter=input; adapter.identity.adapter_revision="not-base";
    rejects<std::invalid_argument>([&]{(void)denoiser.begin(adapter);},"adapter begin was silently accepted");

    // The reference is the actual model implementation, including prefix KV,
    // conditioning GELU and BF16 schedule delta/product/residual operations.
    qwen21::Transformer transformer(weights,mlx_config());
    Tensor oracle=latent;
    for (size_t step=0;step+1<schedule.size();++step) {
        auto noise=transformer.forward(oracle,text,schedule[step],item.height,item.width,true,nullptr,references);
        check(noise.dtype()==mx::bfloat16 && oracle.dtype()==mx::bfloat16,"reference Transformer/Euler dtype changed");
        oracle=oracle+noise*Tensor(schedule[step+1]-schedule[step],oracle.dtype());
        mx::eval(oracle);
        check(mx::all(mx::isfinite(oracle)).item<bool>(),"MLX step returned nonfinite latent");
        check(transformer.cached_layers()==layers,"MLX prefix did not cover both tiny layers");
        fixture.verify();
    }
    // Every MLX producer has completed before the independent native queue.
    auto run=denoiser.begin(input);
    check(run->state()==ng::DenoiserState::Prepared && run->completed_steps()==0,"invalid initial run state");
    for(const auto& status:{run->physical_status(),run->drain(std::chrono::milliseconds(0)),
                           run->drain(std::chrono::milliseconds(1))})
        check(status.ok && status.physically_complete && !status.timed_out &&
              status.submitted_buffers==0 && status.completed_buffers==0 && status.error.empty(),
              "unsubmitted run must report no outstanding physical work");
    rejects<std::invalid_argument>([&]{(void)run->drain(std::chrono::milliseconds(-1));},
                                   "negative drain timeout was accepted before submission");
    check(run->state()==ng::DenoiserState::Prepared && run->completed_steps()==0 && backend.healthy(),
          "unsubmitted drain/status published or retired the run");
    rejects<std::logic_error>([&]{(void)run->final_latents();},"early final_latents was accepted");
    for (size_t step=0;step+1<schedule.size();++step) {
        const auto completion=run->step();
        // step() includes the real per-step finite flag acceptance before it
        // publishes completed_steps; no test-only approximation replaces it.
        check(completion.ok && completion.physically_complete && !completion.timed_out,
              "native step did not physically finish and pass its finite gate");
        check(run->completed_steps()==step+1,"native step publication count mismatch");
        const auto prior_state=run->state();
        for(const auto& status:{run->physical_status(),run->drain(std::chrono::milliseconds(0)),
                               run->drain(std::chrono::milliseconds(1))})
            check(status.ok && status.physically_complete && !status.timed_out && status.error.empty() &&
                  status.submitted_buffers==completion.submitted_buffers &&
                  status.completed_buffers==completion.completed_buffers &&
                  status.completed_buffers==status.submitted_buffers && status.submitted_buffers>0,
                  "successful status/drain changed the completed submission receipt");
        rejects<std::invalid_argument>([&]{(void)run->drain(std::chrono::milliseconds(-1));},
                                       "negative drain timeout was accepted after completion");
        check(run->state()==prior_state && run->completed_steps()==step+1 && backend.healthy(),
              "successful or rejected drain changed result publication/state");
        fixture.verify();
        if (step+2<schedule.size()) {
            check(run->state()==ng::DenoiserState::Prepared,"intermediate step state mismatch");
            rejects<std::logic_error>([&]{(void)run->final_latents();},"partial trajectory was published");
        }
    }
    check(run->state()==ng::DenoiserState::Complete,"final state is not Complete");
    const auto result=compare(run->final_latents(),oracle);
    rejects<std::logic_error>([&]{(void)run->step();},"step after Complete was accepted");
    check(run->state()==ng::DenoiserState::Complete,"rejected extra step changed completed state");
    fixture.verify();
    const bool accepted=result.rrms<=maximum_rrms && result.max_abs<=maximum_abs;
    const auto memory=run->memory();
    std::cout << std::setprecision(10)
        << "{\"case\":\"" << item.name << "\",\"steps\":2,\"layers\":2,\"reference\":\"existing-mlx-transformer\""
        << ",\"rrms\":" << result.rrms << ",\"max_abs\":" << result.max_abs
        << ",\"rrms_limit\":" << maximum_rrms << ",\"max_abs_limit\":" << maximum_abs
        << ",\"request_bytes\":" << memory.request_bytes << ",\"block_arena_bytes\":" << memory.block_arena_bytes
        << ",\"prefix_bytes\":" << memory.prefix_bytes << ",\"source_padding_hashes\":" << fixture.count()
        << ",\"mode\":\"" << (observe_only?"diagnostic-only":"strict")
        << "\",\"reference_euler\":\"delta->BF16;noise*delta->BF16;latent+product->BF16\""
        << ",\"conditioning_recipe\":\"" << mg::NativeQwenOps::conditioning_recipe()
        << "\",\"ffn_recipe\":\"" << mg::NativeFfn::precision_recipe() << '"'
        << ",\"drain_contract\":\"no-submission/status/zero/success/negative-checked\""
        << ",\"strict_threshold_passed\":" << (accepted?"true":"false")
        << ",\"accepted\":" << (accepted && !observe_only?"true":"false");
    print_error_diagnostics(result);
    std::cout << "}\n";
    if(!observe_only) check(accepted,std::string(item.name)+" exceeded fixed tiny latent tolerance; investigate without widening");
    return accepted;
}
void reject_transformer_overrides() {
    for (const auto* key:{"TURBOCIDER_QWEN21_METAL_QK_ROPE","TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE",
                          "TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION","TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC",
                          "TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS","TURBOCIDER_QWEN21_PROFILE_GPU_OPS",
                          "TURBOCIDER_QWEN21_PROFILE_PREFILL_SEGMENTS"}) {
        const char* value=std::getenv(key);
        check(!value || !*value || std::string_view(value)=="0",std::string("unset diagnostic reference override: ")+key);
    }
}
} // namespace

int main(int argc,char** argv) try {
    diagnostic_cpu_checks();
    if(argc==2 && std::string_view(argv[1])=="--cpu-diagnostics") {
        std::cout << "bf16_error_diagnostic_contract=PASS device_created=0\n";
        return 0;
    }
    if(argc==2 && std::string_view(argv[1])=="--device-silu") {
        run_silu_component();
        return 0;
    }
    if(argc==2 && std::string_view(argv[1])=="--device-silu-math") {
        run_silu_math_diagnostic();
        return 2; // Observation only; no component or full-DiT qualification.
    }
    if (argc==1) {
        std::cout << "usage: native-qwen-denoiser-component --device [base|ref1|ref2|all] [--observe-only]\n"
                     "       native-qwen-denoiser-component --device-silu\n"
                     "       native-qwen-denoiser-component --device-silu-math\n"
                     "       native-qwen-denoiser-component --cpu-diagnostics\n"
                     "No device created. Uses tiny generated weights; no model files.\n";
        return 0;
    }
    check(argc>=2 && argc<=4 && std::string_view(argv[1])=="--device","expected --device [base|ref1|ref2|all] [--observe-only]");
    std::string selected="all";
    bool observe_only=false,have_case=false;
    for(int i=2;i<argc;++i) {
        if(std::string_view(argv[i])=="--observe-only") {
            check(!observe_only,"duplicate --observe-only"); observe_only=true;
        } else {
            check(!have_case,"multiple tiny cases selected"); selected=argv[i]; have_case=true;
        }
    }
    check(selected=="all" || selected=="base" || selected=="ref1" || selected=="ref2","unknown tiny case");
    reject_transformer_overrides();
    mg::Backend backend;
    Fixture fixture(backend);
    Weights weights;
    make_weights(weights,fixture);
    const ng::DenoiserGeometry geometry{{hidden,head_count,dimensions,ffn},layers,64};
    // Import only; weights were explicitly produced and are held by Fixture.
    auto native_weights=ng::import_base_denoiser_weights(backend,weights,geometry);
    auto denoiser=ng::NativeDenoiser::prepare(backend,std::move(native_weights),source_revision,geometry);
    const std::array<Case,3> cases{{{"base",2,2,3,{}},
        {"ref1",2,3,4,{{3,1,1}}}, {"ref2",3,2,5,{{1,3,2},{2,1,2}}}}};
    size_t failed_threshold_cases=0;
    for (const auto& item:cases) if (selected=="all" || selected==item.name)
        failed_threshold_cases+=!run_case(backend,denoiser,weights,fixture,item,observe_only);
    fixture.verify();
    if(observe_only) {
        std::cout << "tiny_native_denoiser=OBSERVED qualification=none failed_threshold_cases="
                  << failed_threshold_cases << " real_model_loads=0 e2e_images=0\n";
        return 2; // Explicit diagnostic outcome, never a qualification PASS.
    }
    std::cout << "tiny_native_denoiser=PASS real_model_loads=0 e2e_images=0\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "tiny_native_denoiser=FAIL " << error.what() << '\n';
    return 1;
}
