#include "../../native/backends/metal_graph/kernels/native_lora.hpp"
#include "../../native/backends/metal_graph/kernels/dense_ffn.hpp"

#include <array>
#include <cctype>
#include <iostream>
#include <regex>
#include <type_traits>

using namespace tc::metal_graph::native_lora;
namespace {
size_t checks=0;
void check(bool value,const char* message) {
    ++checks;
    if(!value)throw std::runtime_error(message);
}
size_t occurrences(std::string_view value,std::string_view needle) {
    size_t count=0,position=0;
    while((position=value.find(needle,position))!=std::string_view::npos) {
        ++count;position+=needle.size();
    }
    return count;
}
struct Kernel { std::string signature,body; };
Kernel kernel(const std::string& source,std::string_view name) {
    const std::string marker="kernel void "+std::string(name)+"(";
    check(occurrences(source,marker)==1,"kernel declaration must be unique");
    const auto start=source.find(marker),body_start=source.find('{',start);
    check(body_start!=std::string::npos,"kernel body missing");
    size_t depth=1,end=body_start+1;
    for(;end<source.size() && depth;++end) {
        if(source[end]=='{')++depth;
        else if(source[end]=='}')--depth;
    }
    check(depth==0,"unbalanced kernel source");
    return {source.substr(start,body_start-start),source.substr(body_start+1,end-body_start-2)};
}
void bindings(const Kernel& value,unsigned buffer_count,std::string_view params) {
    const std::regex pattern(R"(\[\[buffer\(([0-9]+)\)\]\])");
    unsigned index=0;
    for(std::sregex_iterator it(value.signature.begin(),value.signature.end(),pattern),end;it!=end;++it) {
        check(std::stoul((*it)[1].str())==index,"buffer bindings must be contiguous and unique");
        ++index;
    }
    check(index==buffer_count,"unexpected kernel buffer count");
    check(value.signature.find(std::string(params)+"& p [[buffer("+
          std::to_string(buffer_count-1)+")]]")!=std::string::npos,"POD parameters must be last buffer");
}
std::string compact(std::string_view value) {
    std::string result;
    for(const auto character:value)
        if(!std::isspace(static_cast<unsigned char>(character)))result+=character;
    return result;
}
std::string silu_rounding(std::string_view source) {
    const auto first=source.find("bfloat exponential"),last=source.find("bfloat activated",first);
    check(first!=std::string_view::npos && last!=std::string_view::npos,"SiLU BF16 boundaries missing");
    const auto end=source.find(';',last);
    check(end!=std::string_view::npos,"SiLU activated boundary missing");
    return compact(source.substr(first,end-first+1));
}
uint64_t fingerprint(std::string_view value) {
    uint64_t result=14695981039346656037ULL;
    for(const auto character:value) {
        result^=static_cast<unsigned char>(character);
        result*=1099511628211ULL;
    }
    return result;
}
std::string decoder(std::string_view source,std::string_view name) {
    const auto first=source.find("inline float "+std::string(name)+"("),open=source.find('{',first);
    const auto end=source.find('}',open);
    check(first!=std::string_view::npos && open!=std::string_view::npos && end!=std::string_view::npos,
          "factor decoder missing");
    return compact(source.substr(open+1,end-open-1));
}
void sg_index_contract() {
    // Integer indexing/coverage contract only. This does not execute or model
    // Metal's matrix reduction order, fragment lane layout or synchronization.
    for(int phase=0;phase<2;++phase) {
        std::array<int,8*32> left{};
        std::array<int,32*32> factor{};
        for(int row=0;row<8;++row)
            for(int k=0;k<32;++k)left[size_t(row*32+k)]=(row+1)*3+k%7-3+phase;
        for(int n=0;n<32;++n)
            for(int k=0;k<32;++k)
                factor[size_t(n*32+k)]=(n+1)*2-(k%11)*3+(n*k)%5+phase*(n%3-k%4);
        std::array<int64_t,8*32> result{};
        std::array<unsigned,8*32> visits{};
        for(int sg=0;sg<4;++sg)
            for(int row=0;row<8;++row)
                for(int n=0;n<8;++n) {
                    int64_t sum=0;
                    for(int sub_k=0;sub_k<32;sub_k+=8)
                        for(int k=0;k<8;++k)
                            sum+=int64_t(left[size_t(row*32+sub_k+k)])*
                                factor[size_t((sg*8+n)*32+sub_k+k)];
                    const size_t at=size_t(row*32+sg*8+n);
                    result[at]=sum;++visits[at];
                }
        bool wrong_transpose_differs=false;
        for(int row=0;row<8;++row)
            for(int column=0;column<32;++column) {
                int64_t expected=0,wrong_transpose=0;
                for(int k=0;k<32;++k) {
                    expected+=int64_t(left[size_t(row*32+k)])*factor[size_t(column*32+k)];
                    wrong_transpose+=int64_t(left[size_t(row*32+k)])*factor[size_t(k*32+column)];
                }
                const auto at=size_t(row*32+column);
                check(visits[at]==1 && result[at]==expected,
                      "four SG column slices and K8 chunks must cover the pitched result tile exactly once");
                wrong_transpose_differs=wrong_transpose_differs || expected!=wrong_transpose;
            }
        check(wrong_transpose_differs,"non-symmetric factor must detect a missing right-matrix transpose");
    }
}
void sg_source_contract(const std::string& scalar) {
    const auto shader=sg_mma_source();
    check(shader==sg_mma_source_body,"candidate uses its separate inline source body");
    check(scalar.size()==5931 && fingerprint(scalar)==0xd6c67c3ecb9e7553ULL,
          "default scalar shader remains byte-for-byte unchanged");
    check(scalar.find("sgmma")==std::string::npos && scalar.find("simdgroup_float8x8")==std::string::npos,
          "candidate entry points are absent from the default five-PSO source");
    check(occurrences(shader,"kernel void ")==2 && shader.find("tc_native_lora_copy")==std::string::npos &&
          shader.find("tc_native_lora_bias")==std::string::npos && shader.find("tc_native_lora_silu_up")==std::string::npos,
          "the opt-in library contains only the two candidate GEMM entry points");
    check(occurrences(shader,"#include")==2 && shader.find("#include <metal_stdlib>")!=std::string::npos &&
          shader.find("#include <metal_simdgroup_matrix>")!=std::string::npos &&
          shader.find("#include \"")==std::string::npos && shader.find("../")==std::string::npos,
          "candidate includes only the public standalone Metal headers");
    for(const auto forbidden:{"half","relaxed_precision","function_constant","thread_elements",
                              "reinterpret_cast","MetalPerformancePrimitives","metal::fma("})
        check(shader.find(forbidden)==std::string::npos,"candidate must not narrow or use private fragment layouts");
    check(shader.find("#pragma clang fp contract(off)")<shader.find("kernel void"),
          "candidate also keeps scale/add separate");
    check(decoder(shader,"tc_lora_sg_factor")==decoder(scalar,"tc_lora_factor"),
          "SG BF16/FP32 decoder is identical to the scalar precision boundary");
    const auto low=kernel(shader,"tc_native_lora_low_sgmma");
    const auto apply=kernel(shader,"tc_native_lora_apply_sgmma");
    bindings(low,4,"TCLoRALowParams");bindings(apply,4,"TCLoRAApplyParams");
    check(low.signature.find("device float* low [[buffer(2)]]")!=std::string::npos &&
          apply.signature.find("device const float* low [[buffer(0)]]")!=std::string::npos,
          "candidate keeps the global low backing FP32");
    for(const auto* value:{&low,&apply}) {
        check(value->signature.find("uint tid [[thread_index_in_threadgroup]]")!=std::string::npos &&
              value->signature.find("uint sg [[simdgroup_index_in_threadgroup]]")!=std::string::npos,
              "candidate receives thread and SG indices without assuming fragment lane mapping");
        check(value->body.find("simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.f)")!=std::string::npos &&
              value->body.find("simdgroup_float8x8 left,right")!=std::string::npos &&
              occurrences(value->body,"simdgroup_multiply_accumulate(acc,left,right,acc)")==1,
              "all matrix operands and accumulators are float8x8 initialized independently of output");
        check(value->body.find("for(uint sub_k=0;sub_k<32;sub_k+=8)")!=std::string::npos &&
              occurrences(value->body,"simdgroup_load(")==2 &&
              value->body.find("simdgroup_store(acc,result_tile+sg*8,32,ulong2(0),false)")!=std::string::npos,
              "K32 uses four K8 chunks and each of four SGs stores an eight-column pitched slice");
        check(occurrences(value->body,"threadgroup_barrier(mem_flags::mem_threadgroup)")==3 &&
              value->body.find("return")==std::string::npos && value->body.find("break")==std::string::npos &&
              value->body.find("continue")==std::string::npos,"all candidate threads reach three barrier call sites");
        const auto store=value->body.find("simdgroup_store("),last=value->body.rfind("threadgroup_barrier(");
        const auto read=value->body.find("result_tile[local_row*32+lane]");
        check(store<last && last<read && value->body.substr(0,last).find("if(")==std::string::npos,
              "result store finishes before scalar reads and every barrier is in uniform control flow");
        check(value->body.find("row_first=group.y*8")!=std::string::npos &&
              value->body.find("lane=tid&31, local_row=tid/32")!=std::string::npos,
              "candidate preserves the original eight-row/32-column threadgroup mapping");
    }
    check(low.body.find("threadgroup float x_tile[8*32], a_tile[32*32], result_tile[8*32]")!=std::string::npos &&
          apply.body.find("threadgroup float low_tile[8*32], b_tile[32*32], result_tile[8*32]")!=std::string::npos &&
          sg_mma_threadgroup_bytes==sizeof(float)*(8*32+32*32+8*32),"candidate shared layout is exactly six KiB");
    check(low.body.find("simdgroup_load(left,x_tile+sub_k,32,ulong2(0),false)")!=std::string::npos &&
          low.body.find("simdgroup_load(right,a_tile+sg*8*32+sub_k,32,ulong2(0),true)")!=std::string::npos &&
          apply.body.find("simdgroup_load(left,low_tile+sub_k,32,ulong2(0),false)")!=std::string::npos &&
          apply.body.find("simdgroup_load(right,b_tile+sg*8*32+sub_k,32,ulong2(0),true)")!=std::string::npos,
          "both right factor tiles are transposed logically, preserving produced source pitches");
    check(low.body.find("row<p.rows && column<p.input_columns")!=std::string::npos &&
          low.body.find("rank<p.rank && column<p.input_columns")!=std::string::npos &&
          low.body.find("rank<p.rank && row<p.rows")!=std::string::npos &&
          low.body.find("rank<p.rank && row+4<p.rows")!=std::string::npos && low.body.find("bfloat(")==std::string::npos,
          "candidate low retains load/store tail masks without an intermediate BF16 cast");
    check(apply.body.find("row<p.rows && rank<p.rank")!=std::string::npos &&
          apply.body.find("column<p.output_count && rank<p.rank")!=std::string::npos &&
          apply.body.find("column<p.output_count && row<p.rows")!=std::string::npos &&
          apply.body.find("column<p.output_count && row+4<p.rows")!=std::string::npos &&
          apply.body.find("tc_lora_sg_factor(b,column,rank,p.b_pitch,p.b_type)")!=std::string::npos,
          "candidate apply retains all masks and the resolved B slice");
    check(apply.body.find("const float delta=result_tile[local_row*32+lane]*p.effective_scale;")!=std::string::npos &&
          apply.body.find("const float delta=result_tile[(local_row+4)*32+lane]*p.effective_scale;")!=std::string::npos &&
          occurrences(apply.body,"output[index]=bfloat(float(output[index])+delta);")==2 &&
          occurrences(apply.body,"bfloat(")==2 && occurrences(apply.body,"+p.first_output+column")==2,
          "each candidate adapter preserves independent FP32 scaling, selected output offset and one BF16 cast");
    check(shader.find("uint rows, input_columns, rank, input_pitch, a_pitch, low_pitch, a_type, padding;")!=std::string::npos &&
          shader.find("uint rows, rank, output_count, first_output, low_pitch, b_pitch, output_pitch, b_type;")!=std::string::npos &&
          occurrences(shader,"float effective_scale;")==1,"candidate parameters retain the same checked host POD ABI");
    sg_index_contract();
}
}

int main() {
    try {
        const std::string shader=source();
        check(shader==source_body,"source assembly must use the actual inline shader body");
        check(occurrences(shader,"#include")==1 && shader.find("#include <metal_stdlib>")!=std::string::npos,
              "standalone shader has no repository-file dependency");
        check(shader.find("function_constant")==std::string::npos,"bounded shapes are runtime parameters");
        check(shader.find("half")==std::string::npos && shader.find("matmul2d")==std::string::npos,
              "compact-rank arithmetic cannot use FP16 or dense MPP accumulation");
        check(occurrences(shader,"kernel void ")==5,"only the five prepared LoRA entry points are present");
        check(shader.find("#pragma clang fp contract(off)")<shader.find("kernel void"),
              "scale/add remain separate while reductions use explicit FP32 fma");
        check(shader.find("source+ulong(row)*pitch")!=std::string::npos &&
              shader.find("if(fp32)return ((device const float*)p)[column];")!=std::string::npos &&
              shader.find("as_type<float>(uint(((device const ushort*)p)[column])<<16)")!=std::string::npos,
              "factor decoding preserves byte pitch and raw BF16/FP32 values");

        const auto low=kernel(shader,"tc_native_lora_low");
        const auto apply=kernel(shader,"tc_native_lora_apply");
        const auto copy=kernel(shader,"tc_native_lora_copy");
        const auto bias=kernel(shader,"tc_native_lora_bias");
        const auto silu=kernel(shader,"tc_native_lora_silu_up");
        bindings(low,4,"TCLoRALowParams");bindings(apply,4,"TCLoRAApplyParams");
        bindings(copy,3,"TCLoRAElementParams");bindings(bias,3,"TCLoRAElementParams");
        bindings(silu,4,"TCLoRAElementParams");
        check(low.signature.find("device float* low [[buffer(2)]]")!=std::string::npos &&
              apply.signature.find("device const float* low [[buffer(0)]]")!=std::string::npos,
              "the low backing is FP32 across both kernels");
        for(const auto* value:{&low,&apply}) {
            check(value->signature.find("uint tid [[thread_index_in_threadgroup]]")!=std::string::npos,
                  "tiled kernels receive a linear local thread index");
            check(occurrences(value->body,"threadgroup_barrier(mem_flags::mem_threadgroup)")==2 &&
                  value->body.find("return")==std::string::npos,"all 128 threads reach both tile barriers");
            check(occurrences(value->body,"metal::fma(")==2 &&
                  value->body.find("float acc0=0.f, acc1=0.f;")!=std::string::npos,
                  "each thread accumulates two FP32 rows");
            check(value->body.find("local_row=tid/32")!=std::string::npos &&
                  value->body.find("(local_row+4)*32+i")!=std::string::npos,
                  "four local thread rows cover the eight-row output tile");
        }
        check(low.body.find("threadgroup float x_tile[8*32], a_tile[32*32]")!=std::string::npos &&
              apply.body.find("threadgroup float low_tile[8*32], b_tile[32*32]")!=std::string::npos,
              "shared tiles require five KiB of FP32 storage per tiled kernel");
        check(low.body.find("row<p.rows && column<p.input_columns")!=std::string::npos &&
              low.body.find("rank<p.rank && column<p.input_columns")!=std::string::npos &&
              low.body.find("rank<p.rank && row<p.rows")!=std::string::npos &&
              low.body.find("rank<p.rank && row+4<p.rows")!=std::string::npos,
              "M/K/rank tails mask both loads and stores");
        check(low.body.find("bfloat(")==std::string::npos &&
              low.body.find("low[ulong(row)*(p.low_pitch/4)+rank]=acc0")!=std::string::npos &&
              low.body.find("low[ulong(row+4)*(p.low_pitch/4)+rank]=acc1")!=std::string::npos,
              "no BF16 cast occurs at the low boundary");
        check(apply.body.find("row<p.rows && rank<p.rank")!=std::string::npos &&
              apply.body.find("column<p.output_count && rank<p.rank")!=std::string::npos &&
              apply.body.find("column<p.output_count && row<p.rows")!=std::string::npos &&
              apply.body.find("column<p.output_count && row+4<p.rows")!=std::string::npos,
              "M/rank/output tails mask both apply passes");
        check(occurrences(apply.body,"+p.first_output+column")==2 &&
              apply.body.find("tc_lora_factor(b,column,rank,p.b_pitch,p.b_type)")!=std::string::npos,
              "B is a resolved slice and output writes retain its selected range");
        check(apply.body.find("const float delta=acc0*p.effective_scale;")!=std::string::npos &&
              apply.body.find("const float delta=acc1*p.effective_scale;")!=std::string::npos &&
              occurrences(apply.body,"output[index]=bfloat(float(output[index])+delta);")==2 &&
              occurrences(apply.body,"bfloat(")==2,
              "each adapter scales FP32 delta then casts the previous BF16 output exactly once");
        check(copy.signature.find("device const ushort* base")!=std::string::npos &&
              copy.signature.find("device ushort* output")!=std::string::npos &&
              copy.body.find("p.output_pitch/2")!=std::string::npos && copy.body.find("p.first_pitch/2")!=std::string::npos,
              "base copy preserves BF16 bits with independent physical pitches");
        check(bias.body.find("output[at]=bfloat(float(output[at])+float(bias[index.x]));")!=std::string::npos &&
              occurrences(bias.body,"bfloat(")==1,"bias is one separate BF16 operation per output element");
        check(silu_rounding(silu.body)==silu_rounding(tc::metal_graph::dense_ffn::source_body),
              "pre-SiLU gate/up uses existing NativeFfn BF16 operation boundaries");
        check(compact(silu.body).find("bfloat(float(activated)*float(u));")!=std::string::npos &&
              compact(tc::metal_graph::dense_ffn::source_body).find("bfloat(float(activated)*float(u));")!=std::string::npos,
              "hidden product retains its final BF16 boundary");

        check(std::is_standard_layout_v<LowParameters> && sizeof(LowParameters)==32 &&
              offsetof(LowParameters,input_pitch)==12 && offsetof(LowParameters,a_type)==24,
              "low host POD field ABI");
        check(std::is_standard_layout_v<ApplyParameters> && sizeof(ApplyParameters)==36 &&
              offsetof(ApplyParameters,first_output)==12 && offsetof(ApplyParameters,effective_scale)==32,
              "apply host POD field ABI");
        check(std::is_standard_layout_v<ElementParameters> && sizeof(ElementParameters)==20 &&
              offsetof(ElementParameters,output_pitch)==16,"element host POD field ABI");
        check(shader.find("uint rows, input_columns, rank, input_pitch, a_pitch, low_pitch, a_type, padding;")!=std::string::npos &&
              shader.find("uint rows, rank, output_count, first_output, low_pitch, b_pitch, output_pitch, b_type;")!=std::string::npos &&
              shader.find("float effective_scale;")!=std::string::npos &&
              shader.find("uint rows, columns, first_pitch, second_pitch, output_pitch;")!=std::string::npos,
              "MSL POD field order matches the host parameters");
        check(precision_recipe=="runtime-lora-tiled8x32-fp32-fma-low-fp32-scaled-ordered-bf16-bias-last-v1",
              "precision recipe identifies explicit FP32 reduction and ordered BF16 boundaries");
        sg_source_contract(shader);
        std::cout<<"native_lora_source_cpu=PASS checks="<<checks
                 <<" device_created=0 gpu_submissions=0 model_loads=0\n";
    } catch(const std::exception& error) {
        std::cerr<<"native_lora_source_cpu=FAIL "<<error.what()<<'\n';
        return 1;
    }
}
