#pragma once
#include "../ane_runtime.hpp"
#include <string>

namespace tc::ane::private_api {
// Native fixed micrograph emitter, no externally supplied MIL and no embedded
// checkpoint weights. FP16 infrastructure baseline; NOT a W8A8 emitter.
std::string fp16_program(const GraphShape &shape);
// Native INT8 representation inputs with normalized FP16 output. Scales are
// restored by the GPU in FP32, not multiplied into a potentially overflowing
// ANE FP16 output. This does NOT assert native INT8 hardware arithmetic.
std::string w8_matmul_program(const GraphShape &shape);
struct W8FfnProgram {
    std::string mil;
    std::vector<uint8_t> constants;
    float headroom = 64.f;
    int packed_rows = 0; // down normalized + hidden scale + optional corrected hidden
};
W8FfnProgram w8_swiglu_program(const GraphShape &, uint64_t rotation_seed = 20260930, float headroom = 64.f,
                             W8Basis basis = W8Basis::SylvesterDH, int activation_group_size = 0);
}
