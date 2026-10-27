#pragma once
#include "runtime_private_qkv_options.hpp"

namespace tc::qwen21 {
inline bool configured_gpu_dual_swiglu_request(const Request&r) {
    const auto raw=std::getenv("TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU");
    if(!binary_option_or_unset(raw))throw std::invalid_argument("Qwen GPU dual SwiGLU requires0 or1");
    if(!option_enabled(raw))return false;
    const auto channels=std::getenv("TURBOCIDER_PRIVATE_ANE_CHANNELS");
    const auto activation=std::getenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION");
    if(!private_qkv_joint_request(r)||!option_enabled(std::getenv("TURBOCIDER_QWEN21_PRIVATE_QKV"))||
        !option_enabled(std::getenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM"))||!channels||std::string_view(channels)!="6144"||
        !activation||std::string_view(activation)!="sigmoid-complement")
        throw std::invalid_argument("GPU dual SwiGLU requires explicit joint resident512 base40 fixed6144 sigmoid-complement runtime");
    return true;
}
// Scalar-only source-bound qualification. No tensors/weights retained; once
// disabled it never retries/recompiles within this owner. Explicit unload/new
// owner resets the state, not a new conditioning/prompt/reference length.
struct RuntimeGpuDualState {
    std::string generation,failure_reason;
    bool validated=false,disabled=false;
    uint64_t validation_calls=0;
    double validation_seconds=0;
    void fail(const std::string&reason){disabled=true;if(failure_reason.empty())failure_reason=reason;}
    void bind(const std::string&source) {
        if(source.empty())fail("GPU dual source identity is empty");
        else if(generation.empty())generation=source;
        else if(generation!=source)fail("GPU dual source generation changed; new owner required");
    }
    bool available()const{return !disabled;}
    bool healthy()const{return validated&&!disabled;}
    void accept(){if(!disabled)validated=true;}
};
}
