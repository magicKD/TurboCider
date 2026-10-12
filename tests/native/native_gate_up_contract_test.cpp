#include "../../native/backends/metal_graph/native_gate_up.hpp"

#include <array>
#include <iostream>
#include <limits>

using namespace tc::metal_graph;
using namespace tc::metal_graph::dense_gate_up;
namespace {
size_t checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected,"invalid paired gate/up binding accepted");
}
Layout matrix(uint32_t rows, uint32_t columns, uintptr_t identity,
              size_t padding = 8, size_t offset = 16) {
    const size_t pitch = (size_t(columns)+padding)*2;
    return {rows,columns,pitch,size_t(rows-1)*pitch+size_t(columns)*2,
        offset,ScalarType::BF16,identity};
}

void geometry_contract() {
    for (const auto geometry : {Geometry{4096,12288,12288},Geometry{4096,12288,6144},
                               Geometry{3840,10240,10240},Geometry{3840,10240,6144},
                               Geometry{128,256,128}}) {
        const auto shader = dense_gate_up::source(geometry);
        check(shader.find("#define TC_GATE_UP_H "+std::to_string(geometry.hidden)) !=
              std::string::npos,"prepare H specialization missing");
        check(shader.find("#define TC_GATE_UP_C "+std::to_string(geometry.channels)) !=
              std::string::npos,"prepare C specialization missing");
    }
    for (const auto geometry : {Geometry{0,128,128},Geometry{128,0,128},Geometry{128,128,0},
                               Geometry{127,128,128},Geometry{128,127,128},Geometry{128,256,127},
                               Geometry{4224,128,128},Geometry{128,12416,128},Geometry{128,128,256}})
        rejects([&] { (void)dense_gate_up::source(geometry); });
    check(!NativeGateUp::supports_bias(),"paired primitive must explicitly exclude bias");
    check(NativeGateUp::precision_recipe() != NativeFfn::precision_recipe(),
          "paired base primitive must have its own recipe");
}

void layout_contract() {
    const auto valid = matrix(33,128,1);
    check(validate_layout(valid) == 32*272+256,"physical pitch/last row extent");
    for (const auto type : {ScalarType::Bytes,ScalarType::FP16,ScalarType::FP32,
                           ScalarType::UInt32,ScalarType::Int8}) {
        auto bad = valid; bad.type = type;
        rejects([&] { (void)validate_layout(bad); });
    }
    auto bad = valid; bad.native_identity = 0; rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.rows = 0; rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.columns = 0; rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.rows = uint32_t(std::numeric_limits<int>::max())+1;
    rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.columns = uint32_t(std::numeric_limits<int>::max())+1;
    rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.offset_bytes = 1; rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.row_stride_bytes = 255; rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.row_stride_bytes = 254; rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.row_stride_bytes = (size_t(std::numeric_limits<int>::max())+1)*2;
    rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.view_size_bytes--;
    rejects([&] { (void)validate_layout(bad); });
    bad = valid; bad.offset_bytes = std::numeric_limits<size_t>::max()-1;
    rejects([&] { (void)validate_layout(bad); });
    auto maximum = matrix(uint32_t(std::numeric_limits<int>::max()),128,1);
    check(validate_layout(maximum) == size_t(maximum.rows-1)*272+256,
          "bounded MPP signed row extent");
}

void bindings_contract() {
    const Geometry geometry{128,256,128};
    for (const uint32_t rows : {1U,31U,32U,33U,1024U,1056U}) {
        const auto input = matrix(rows,128,1,12,18);
        const auto gate = matrix(256,128,2,14,24);
        const auto up = matrix(256,128,2,18,24+256*284); // Same original fused source backing.
        const auto gout = matrix(rows,128,3,20,30);
        const auto uout = matrix(rows,128,4,22,38);
        for (const uint32_t first : {0U,13U,128U}) {
            const auto checked = validate_bindings(geometry,input,gate,up,gout,uout,first);
            check(checked.parameters.rows == rows && checked.parameters.first_channel == first &&
                  checked.parameters.input_stride == 140 && checked.parameters.gate_stride == 142 &&
                  checked.parameters.up_stride == 146 && checked.parameters.gate_output_stride == 148 &&
                  checked.parameters.up_output_stride == 150,"independent physical source/output pitches");
            // The first-channel offset is applied only within the already
            // offset-bound source pointer. Both outputs retain local columns.
            const size_t gate_last = size_t(first+127)*gate.row_stride_bytes+127*2+2;
            const size_t up_last = size_t(first+127)*up.row_stride_bytes+127*2+2;
            check(gate_last <= checked.gate_bytes && up_last <= checked.up_bytes,
                  "selected source rows remain within full physical source extent");
            check(checked.gate_output_bytes == size_t(rows-1)*296+256 &&
                  checked.up_output_bytes == size_t(rows-1)*300+256,
                  "output extent excludes tail padding");
        }
    }
    const auto input = matrix(33,128,1),gate = matrix(256,128,2),up = matrix(256,128,3);
    const auto gout = matrix(33,128,4),uout = matrix(33,128,5);
    rejects([&] { (void)validate_bindings(geometry,input,gate,up,gout,uout,129); });
    rejects([&] { (void)validate_bindings(geometry,input,gate,up,gout,uout,UINT32_MAX); });
    auto wrong = input; wrong.columns--;
    rejects([&] { (void)validate_bindings(geometry,wrong,gate,up,gout,uout,0); });
    wrong = gate; wrong.rows = 128; // Passing only the selected slice is ambiguous and rejected.
    rejects([&] { (void)validate_bindings(geometry,input,wrong,up,gout,uout,128); });
    wrong = up; wrong.rows = 128;
    rejects([&] { (void)validate_bindings(geometry,input,gate,wrong,gout,uout,128); });
    wrong = gate; wrong.columns--;
    rejects([&] { (void)validate_bindings(geometry,input,wrong,up,gout,uout,0); });
    wrong = up; wrong.columns--;
    rejects([&] { (void)validate_bindings(geometry,input,gate,wrong,gout,uout,0); });
    wrong = gout; wrong.rows--;
    rejects([&] { (void)validate_bindings(geometry,input,gate,up,wrong,uout,0); });
    wrong = uout; wrong.columns--;
    rejects([&] { (void)validate_bindings(geometry,input,gate,up,gout,wrong,0); });
    for (unsigned operand = 0; operand < 5; ++operand) {
        auto all = std::array{input,gate,up,gout,uout};
        all[operand].view_size_bytes--;
        rejects([&] { (void)validate_bindings(geometry,all[0],all[1],all[2],all[3],all[4],0); });
        all = {input,gate,up,gout,uout}; all[operand].type = ScalarType::FP32;
        rejects([&] { (void)validate_bindings(geometry,all[0],all[1],all[2],all[3],all[4],0); });
    }

    for (unsigned output = 3; output < 5; ++output) {
        for (unsigned read = 0; read < 3; ++read) {
            auto all = std::array{input,gate,up,gout,uout};
            all[output].native_identity = all[read].native_identity;
            all[output].offset_bytes = all[read].offset_bytes;
            rejects([&] { (void)validate_bindings(geometry,all[0],all[1],all[2],all[3],all[4],0); });
        }
    }
    wrong = uout; wrong.native_identity = gout.native_identity;
    wrong.offset_bytes = gout.offset_bytes;
    rejects([&] { (void)validate_bindings(geometry,input,gate,up,gout,wrong,0); });
    wrong.offset_bytes = gout.offset_bytes+validate_layout(gout)-2;
    rejects([&] { (void)validate_bindings(geometry,input,gate,up,gout,wrong,0); });
    wrong.offset_bytes = gout.offset_bytes+validate_layout(gout);
    (void)validate_bindings(geometry,input,gate,up,gout,wrong,0);
    check(true,"disjoint output views may share one retained backing");
    auto overlapping_reads = up;
    overlapping_reads.native_identity = gate.native_identity;
    overlapping_reads.offset_bytes = gate.offset_bytes;
    (void)validate_bindings(geometry,input,gate,overlapping_reads,gout,uout,0);
    check(true,"immutable source alias is safe");
}
}

int main() {
    try {
        geometry_contract(); layout_contract(); bindings_contract();
        std::cout << "native_gate_up_contract=PASS checks=" << checks
                  << " device_created=0 model_loads=0\n";
    } catch (const std::exception& error) {
        std::cerr << "native_gate_up_contract=FAIL " << error.what() << '\n';
        return 1;
    }
}
