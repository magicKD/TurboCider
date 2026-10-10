#include "../../native/backends/metal_graph/kernels/dense_gate_up.hpp"
#include "../../native/backends/metal_graph/kernels/dense_linear.hpp"

#include <array>
#include <cctype>
#include <iostream>
#include <regex>

using namespace tc::metal_graph;
namespace {
size_t checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
size_t occurrences(std::string_view text, std::string_view needle) {
    size_t count = 0, position = 0;
    while ((position = text.find(needle,position)) != std::string_view::npos) {
        ++count; position += needle.size();
    }
    return count;
}
std::string compact(std::string_view text) {
    std::string result;
    for (const auto character : text)
        if (!std::isspace(static_cast<unsigned char>(character))) result += character;
    return result;
}
std::string replace(std::string text, std::string_view from, std::string_view to) {
    size_t position = 0;
    while ((position = text.find(from,position)) != std::string::npos) {
        text.replace(position,from.size(),to); position += to.size();
    }
    return text;
}
std::string paired_projection_body(std::string_view source) {
    const auto first = source.find("auto a = tensor("), last = source.find("op.run(aa,uu,au);");
    check(first != std::string_view::npos && last != std::string_view::npos,
          "paired base MPP algorithm missing");
    return compact(source.substr(first,last-first+std::string_view("op.run(aa,uu,au);").size()));
}
void source_contract() {
    const std::string source(dense_gate_up::source_body);
    check(occurrences(source,"kernel void ") == 1,"prepare must require only one paired PSO");
    check(occurrences(source,"kernel void tc_native_gate_up_bf16(") == 1,"paired entrypoint name");
    const auto signature = source.substr(0,source.find("uint row = group.y"));
    const std::regex pattern(R"(\[\[buffer\(([0-9]+)\)\]\])");
    unsigned index = 0;
    for (std::sregex_iterator it(signature.begin(),signature.end(),pattern),end; it != end; ++it) {
        check(std::stoul((*it)[1].str()) == index,"contiguous buffer binding indices");
        ++index;
    }
    check(index == 6,"five buffers and last POD binding");
    check(source.find("TCNativeGateUpParams& p [[buffer(5)]]") != std::string::npos,
          "POD binding ABI");
    check(sizeof(dense_gate_up::Parameters) == 28 &&
          offsetof(dense_gate_up::Parameters,rows) == 0 &&
          offsetof(dense_gate_up::Parameters,first_channel) == 4 &&
          offsetof(dense_gate_up::Parameters,input_stride) == 8 &&
          offsetof(dense_gate_up::Parameters,gate_stride) == 12 &&
          offsetof(dense_gate_up::Parameters,up_stride) == 16 &&
          offsetof(dense_gate_up::Parameters,gate_output_stride) == 20 &&
          offsetof(dense_gate_up::Parameters,up_output_stride) == 24,"seven uint POD layout");
    const auto old_body = paired_projection_body(dense_ffn::source_body);
    const auto new_body = replace(replace(paired_projection_body(source),"TC_GATE_UP_H","TC_FFN_H"),
                                  "TC_GATE_UP_C","TC_FFN_C");
    check(old_body == new_body,"paired MPP projection arithmetic must remain identical to dense FFN");
    check(occurrences(source,"op.run(") == 2,"two independent base projections remain");
    check(source.find("execution_simdgroups<4>") != std::string::npos,"four SIMD groups");
    check(source.find("decltype(gg),float>()") != std::string::npos &&
          source.find("decltype(uu),float>()") != std::string::npos,"two FP32 accumulation fragments");
    const auto formatted = compact(source);
    check(formatted.find("gate_output[ulong(row+coord[1])*p.gate_output_stride+col+coord[0]]=bfloat(ag[i]);") !=
          std::string::npos,"gate FP32 to BF16 boundary and independent physical output pitch");
    check(formatted.find("up_output[ulong(row+coord[1])*p.up_output_stride+col+coord[0]]=bfloat(au[i]);") !=
          std::string::npos,"up FP32 to BF16 boundary and independent physical output pitch");
    check(source.find("exp(") == std::string::npos && source.find("activated") == std::string::npos &&
          source.find("effective_scale") == std::string::npos,"no activation or LoRA fusion in base primitive");
    check(source.find("first_channel)*p.gate_stride") != std::string::npos &&
          source.find("first_channel)*p.up_stride") != std::string::npos,
          "channel source offset retains each original physical pitch");
    check(occurrences(source,"device const bfloat*") == 3,"all checkpoint/input sources read-only");
    check(source.find("row + coord[1] >= p.rows") != std::string::npos,
          "dynamic M32 tails must be masked before writes");
    check(source.find("col + coord[0] >= TC_GATE_UP_C") != std::string::npos,
          "output-local channel range mask");
    check(dense_linear::source_body.find("matmul2d_descriptor(32,128,TC_LINEAR_K,false,true,false)") !=
          std::string::npos,"split NativeLinear uses same MPP tile/reduction descriptor");
}

void dispatch_index_contract() {
    // Integer coverage contract only, not an oracle for MPP fragment execution
    // or rounding. Real-device numerical evidence belongs to the opt-in probe.
    for (const uint32_t rows : {1U,31U,32U,33U}) {
        for (const uint32_t channels : {128U,256U}) {
            const size_t gate_pitch = channels+8, up_pitch = channels+12;
            std::vector<unsigned> gate(size_t(rows)*gate_pitch),up(size_t(rows)*up_pitch);
            for (uint32_t group_y = 0; group_y < (rows+31)/32; ++group_y) {
                for (uint32_t group_x = 0; group_x < channels/128; ++group_x) {
                    for (uint32_t y = 0; y < 32; ++y) {
                        for (uint32_t x = 0; x < 128; ++x) {
                            const auto row = group_y*32+y,col = group_x*128+x;
                            if (row >= rows || col >= channels) continue;
                            ++gate[size_t(row)*gate_pitch+col];
                            ++up[size_t(row)*up_pitch+col];
                        }
                    }
                }
            }
            bool okay = true;
            for (uint32_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < gate_pitch; ++col)
                    okay &= gate[size_t(row)*gate_pitch+col] == unsigned(col < channels);
                for (size_t col = 0; col < up_pitch; ++col)
                    okay &= up[size_t(row)*up_pitch+col] == unsigned(col < channels);
            }
            check(okay,"tile coverage writes both logical matrices once and leaves padding untouched");
        }
    }
}
}

int main() {
    try {
        source_contract(); dispatch_index_contract();
        std::cout << "native_gate_up_source=PASS checks=" << checks
                  << " device_created=0 shader_compiled=0 model_loads=0\n";
    } catch (const std::exception& error) {
        std::cerr << "native_gate_up_source=FAIL " << error.what() << '\n';
        return 1;
    }
}
