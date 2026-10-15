// Bounded native FFN correctness probe. CPU-only by default; device compilation
// and tiny tensor dispatches require separate explicit command-line modes.
#include "../../native/backends/metal_graph/native_ffn.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tc::metal_graph;
namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
uint16_t bf16(float value) {
    auto bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return uint16_t((bits >> 16) | 0x40U);
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
float decode(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }
float rounded(float value) { return decode(bf16(value)); }
float sigmoid_oracle(float value) {
    // Structural CPU oracle: std::exp does not prove exact Metal fast::exp
    // bits near a BF16 midpoint. The actual MLX SiLU device probe covers that.
    const float exponential=rounded(std::exp(std::abs(value)));
    const float y=rounded(1.0f/rounded(1.0f+exponential));
    return value<0 ? y : rounded(1.0f-y);
}
float load(const MatrixView& m, uint32_t row, uint32_t column) {
    auto* bytes = static_cast<const std::byte*>(m.storage.contents()) + size_t(row)*m.row_stride_bytes;
    if (m.storage.scalar_type() == ScalarType::FP32) return reinterpret_cast<const float*>(bytes)[column];
    return decode(reinterpret_cast<const uint16_t*>(bytes)[column]);
}
void fill(const MatrixView& m, uint32_t salt) {
    for (uint32_t row=0; row<m.rows; ++row) {
        auto* dst = reinterpret_cast<uint16_t*>(static_cast<std::byte*>(m.storage.contents())+size_t(row)*m.row_stride_bytes);
        for (uint32_t col=0; col<m.columns; ++col)
            dst[col] = bf16(float(int((row*7+col*13+salt)%17)-8)/128.0f);
    }
}
float hidden_oracle(const FfnBindings& b, uint32_t row, uint32_t channel) {
    float gate=0,up=0;
    for (uint32_t k=0; k<b.input.columns; ++k) {
        const float input=load(b.input,row,k);
        gate += input*load(b.gate,b.first_channel+channel,k);
        up += input*load(b.up,b.first_channel+channel,k);
    }
    gate=rounded(gate); up=rounded(up);
    float sigmoid=sigmoid_oracle(gate);
    return rounded(rounded(gate*sigmoid)*up);
}
std::vector<float> oracle_hidden(const FfnBindings& b) {
    std::vector<float> values(size_t(b.hidden.rows)*b.hidden.columns);
    for (uint32_t r=0; r<b.hidden.rows; ++r)
        for (uint32_t c=0; c<b.hidden.columns; ++c)
            values[size_t(r)*b.hidden.columns+c]=hidden_oracle(b,r,c);
    return values;
}
std::vector<float> oracle_down(const FfnBindings& b, const std::vector<float>& hidden) {
    std::vector<float> values(size_t(b.output.rows)*b.output.columns);
    for (uint32_t row=0; row<b.output.rows; ++row) {
        for (uint32_t col=0; col<b.output.columns; ++col) {
            float total=0;
            for (uint32_t c=0; c<b.hidden.columns; ++c)
                total += hidden[size_t(row)*b.hidden.columns+c]*load(b.down,col,b.first_channel+c);
            values[size_t(row)*b.output.columns+col]=
                b.output.storage.scalar_type()==ScalarType::FP32 ? total : rounded(total);
        }
    }
    return values;
}

struct Error { double max_abs=0,relative_rmse=0; size_t finite=0; };
Error compare(const MatrixView& actual, const std::vector<float>& expected) {
    Error error;
    double squares=0, reference=0;
    for (uint32_t r=0; r<actual.rows; ++r) for (uint32_t c=0; c<actual.columns; ++c) {
        float a=load(actual,r,c),e=expected[size_t(r)*actual.columns+c];
        check(std::isfinite(a),"native FFN nonfinite output");
        ++error.finite;
        double diff=double(a)-double(e);
        error.max_abs=std::max(error.max_abs,std::abs(diff));
        squares+=diff*diff; reference+=double(e)*double(e);
    }
    error.relative_rmse=std::sqrt(squares/std::max(reference,1e-30));
    return error;
}

MatrixView matrix(Backend& backend,uint32_t rows,uint32_t columns,ScalarType type,const char* label) {
    size_t pitch=size_t(columns+8)*scalar_bytes(type), bytes=size_t(rows)*pitch;
    auto allocation=backend.allocate(16+bytes+64,label);
    check(allocation.contents()!=nullptr,"component requires Shared CPU-fill buffer");
    std::memset(allocation.contents(),0x5a,allocation.size_bytes());
    return {allocation.view(16,bytes,type),rows,columns,pitch};
}
void canaries(const MatrixView& view) {
    const auto* data=static_cast<const unsigned char*>(view.storage.buffer().contents());
    for (size_t i=0; i<view.storage.offset_bytes(); ++i) check(data[i]==0x5a,"prefix canary modified");
    for (size_t i=view.storage.offset_bytes()+view.storage.size_bytes(); i<view.storage.buffer().size_bytes(); ++i)
        check(data[i]==0x5a,"suffix canary modified");
    const auto* content=static_cast<const unsigned char*>(view.storage.contents());
    const auto columns=size_t(view.columns)*scalar_bytes(view.storage.scalar_type());
    for (uint32_t r=0; r<view.rows; ++r)
        for (size_t c=columns; c<view.row_stride_bytes; ++c)
            check(content[size_t(r)*view.row_stride_bytes+c]==0x5a,"row padding modified");
}

void cpu_checks() {
    for (auto g : {dense_ffn::Geometry{3840,10240,6144},{4096,12288,8192},{128,256,128}}) {
        auto source=dense_ffn::source(g);
        check(source.find("#define TC_FFN_H "+std::to_string(g.hidden))!=std::string::npos,"source hidden specialization");
        check(source.find("#define TC_FFN_C "+std::to_string(g.channels))!=std::string::npos,"source channels specialization");
    }
    size_t rejected=0;
    for (auto g : {dense_ffn::Geometry{0,256,128},{128,256,0},{128,256,384},{128,256,127},{4224,256,128}}) {
        try { dense_ffn::validate_geometry(g); }
        catch (const std::invalid_argument&) { ++rejected; }
    }
    check(rejected==5,"invalid geometry accepted");
    check(bf16(1.00390625f)==0x3f80 && bf16(1.01171875f)==0x3f82,"BF16 RNE oracle ties");
    check(rounded(0.0f)==0.0f && std::isfinite(rounded(-0.125f)),"BF16 oracle finite boundary");
    check(rounded(-2.f*sigmoid_oracle(-2.f))==-0.2392578125f &&
          rounded(0.5f*sigmoid_oracle(0.5f))==0.3125f,"MLX BF16 sigmoid internal boundaries");
    check(NativeFfn::precision_recipe().ends_with("-v3"),"compiled MLX exp requires FFN v3 recipe");
    std::cout<<"cpu_contract=PASS valid_geometry=3 invalid_geometry=5 bf16_rne=PASS sigmoid_bf16_internal=PASS recipe="
             <<NativeFfn::precision_recipe()<<" device_created=0\n";
}

void compile_checks(Backend& backend) {
    // Compile full and GPU-complement dimensions, without allocating any
    // model-sized matrices or recording/submitting commands.
    for (auto g : {dense_ffn::Geometry{3840,10240,10240},{3840,10240,6144},
                   {4096,12288,12288},{4096,12288,8192}}) {
        const auto started=std::chrono::steady_clock::now();
        const auto ffn=NativeFfn::prepare(backend,g);
        const auto prepare_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::cout<<"pipeline_compile=PASS hidden="<<ffn.geometry().hidden
                 <<" width="<<g.width<<" channels="<<g.channels<<" pipelines=3 prepare_seconds="<<prepare_seconds<<'\n';
    }
}

void tiny_case(Backend& backend,uint32_t rows,uint32_t first,uint32_t count,ScalarType output_type) {
    constexpr uint32_t h=128,f=256;
    const auto prepare_started=std::chrono::steady_clock::now();
    auto prepared=NativeFfn::prepare(backend,{h,f,count});
    const auto prepare_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-prepare_started).count();
    FfnBindings b;
    b.input=matrix(backend,rows,h,ScalarType::BF16,"ffn-probe-input");
    // Fused source deliberately keeps two offsets and original padded pitch.
    size_t pitch=(h+8)*2, half_bytes=size_t(f)*pitch;
    auto fused=backend.allocate(16+2*half_bytes+64,"ffn-probe-fused-gate-up");
    std::memset(fused.contents(),0x5a,fused.size_bytes());
    b.gate={fused.view(16,half_bytes,ScalarType::BF16),f,h,pitch};
    b.up={fused.view(16+half_bytes,half_bytes,ScalarType::BF16),f,h,pitch};
    b.down=matrix(backend,h,f,ScalarType::BF16,"ffn-probe-down");
    b.hidden=matrix(backend,rows,count,ScalarType::BF16,"ffn-probe-hidden");
    b.output=matrix(backend,rows,h,output_type,"ffn-probe-output");
    b.first_channel=first;
    fill(b.input,1); fill(b.gate,3); fill(b.up,7); fill(b.down,11);
    auto snapshot=[](const Buffer& buffer) {
        const auto* begin=static_cast<const std::byte*>(buffer.contents());
        return std::vector<std::byte>(begin,begin+buffer.size_bytes());
    };
    const auto input_before=snapshot(b.input.storage.buffer());
    const auto fused_before=snapshot(fused);
    const auto down_before=snapshot(b.down.storage.buffer());
    auto hidden_expected=oracle_hidden(b);
    auto output_expected=oracle_down(b,hidden_expected);
    CommandGraph graph;
    prepared.encode(graph,b);
    check(graph.steps().size()==2,"FFN must record exactly two dispatches");
    auto completion=backend.submit(std::move(graph)).wait();
    check(completion.ok && completion.physically_complete,"tiny FFN command did not complete successfully");
    const auto hidden_error=compare(b.hidden,hidden_expected);
    const auto output_error=compare(b.output,output_expected);
    check(hidden_error.max_abs<=2e-4 && hidden_error.relative_rmse<=0.012,"gate/up oracle error exceeds tolerance");
    check(output_error.max_abs<=2e-4 && output_error.relative_rmse<=0.018,"down oracle error exceeds tolerance");
    // Isolate down GEMM from epilogue error by using the actual BF16 hidden.
    std::vector<float> actual_hidden(hidden_expected.size());
    for (uint32_t r=0; r<rows; ++r) for (uint32_t c=0; c<count; ++c)
        actual_hidden[size_t(r)*count+c]=load(b.hidden,r,c);
    auto down_only_error=compare(b.output,oracle_down(b,actual_hidden));
    check(down_only_error.max_abs<=2e-5 && down_only_error.relative_rmse<=0.002,"isolated down GEMM oracle mismatch");
    canaries(b.hidden); canaries(b.output);
    check(input_before==snapshot(b.input.storage.buffer()) && fused_before==snapshot(fused) &&
          down_before==snapshot(b.down.storage.buffer()),"read-only FFN source modified");
    size_t rejected=0;
    for (unsigned kind=0; kind<3; ++kind) {
        auto bad=b;
        if (kind==0) bad.first_channel=f-count+1;
        if (kind==1) bad.hidden.storage=b.input.storage;
        if (kind==2) bad.down.row_stride_bytes=2;
        CommandGraph untouched;
        try { prepared.encode(untouched,bad); }
        catch (const std::invalid_argument&) { ++rejected; }
        check(untouched.empty(),"invalid whole FFN binding left partial graph commands");
    }
    check(rejected==3,"FFN slice/alias/pitch validation missed invalid binding");
    std::cout<<"tiny_ffn=PASS rows="<<rows<<" first="<<first<<" channels="<<count
             <<" output="<<(output_type==ScalarType::FP32?"fp32":"bf16")
             <<" hidden_max_abs="<<hidden_error.max_abs<<" hidden_relative_rmse="<<hidden_error.relative_rmse
             <<" output_max_abs="<<output_error.max_abs<<" output_relative_rmse="<<output_error.relative_rmse
             <<" down_isolated_max_abs="<<down_only_error.max_abs
             <<" finite_values="<<hidden_error.finite+output_error.finite
             <<" prepare_seconds="<<prepare_seconds
             <<" gpu_seconds="<<completion.gpu_seconds<<" canaries=PASS rejected=3\n";
}

} // namespace

int main(int argc,char** argv) {
    try {
        const std::string mode=argc==2 ? argv[1] : "--cpu-only";
        check(mode=="--cpu-only" || mode=="--compile-only" || mode=="--small-tensors","unknown probe mode");
        cpu_checks();
        if (mode=="--cpu-only") return 0;
        Backend backend;
        std::cout<<std::setprecision(9)<<"device="<<backend.device_name()<<" recipe="<<NativeFfn::precision_recipe()<<'\n';
        if (mode=="--compile-only") {
            compile_checks(backend);
            std::cout<<"compile_only=PASS command_buffers=0 dispatches=0 models_loaded=0\n";
        } else {
            tiny_case(backend,3,0,256,ScalarType::BF16);
            tiny_case(backend,33,128,128,ScalarType::BF16);
            tiny_case(backend,3,128,128,ScalarType::FP32);
            std::cout<<"small_tensors=PASS cases=3 dispatches=6 models_loaded=0 no_performance_qualification=1\n";
        }
    } catch (const std::exception& e) {
        std::cerr<<"native_ffn_probe=FAIL "<<e.what()<<'\n';
        return 1;
    }
}
