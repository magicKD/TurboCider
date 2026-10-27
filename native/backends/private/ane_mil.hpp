#pragma once
#include "../ane_runtime.hpp"
#include "../ane_w8_projection.hpp"
#include "../ane_w8_activation.hpp"
#include <string>

namespace tc::metal_graph::native_ane_lora_contract { class Spec; }

namespace tc::ane::private_api {
// Native fixed micrograph emitter, no externally supplied MIL and no embedded
// checkpoint weights. FP16 infrastructure baseline; NOT a W8A8 emitter.
std::string fp16_program(const GraphShape &shape,bool bf16_value_boundaries=false);
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
                             W8Basis basis = W8Basis::SylvesterDH, int activation_group_size = 0,
                             int hidden_group_size = -1, bool bf16_value_boundaries = false);
// Explicit research-only K reduction recipe. The original entry point and
// descriptor/default bytes stay unchanged. Shape/weight ABI stays fixed.
W8FfnProgram w8_swiglu_projection_program(const GraphShape &,uint64_t,float,W8Basis,int,int,bool,W8Projection);
// Independent activation expression experiment; legacy entry points keep
// their exact signatures/default bytes. No checkpoint constants or new ABI.
W8FfnProgram w8_swiglu_activation_program(const GraphShape &,uint64_t,float,W8Basis,int,int,bool,W8Projection,W8Activation);
// Explicit base-only family: one shared constants blob and maximum physical
// token pitch, several fixed logical rows. Existing emitters stay byte stable.
struct W8FfnFamily {W8FfnProgram program;std::vector<int> rows;};
W8FfnFamily w8_swiglu_function_family_program(const GraphShape &,std::span<const int>,W8Activation);
// Native hybrid LoRA has a distinct immutable ABI and finite-boundary recipe.
// No model/factor constants: dg/du and, for Rank128, Adown remain runtime inputs.
// This only emits MIL; it does not prove compilation/placement or create a job.
W8FfnProgram native_lora_program(const tc::metal_graph::native_ane_lora_contract::Spec&);
}
