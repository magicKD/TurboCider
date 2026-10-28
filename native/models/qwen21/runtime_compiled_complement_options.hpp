#pragma once
#include "runtime_private_qkv_options.hpp"

namespace tc::qwen21 {
inline bool configured_compiled_complement_request(const Request &r) {
    const char *raw=std::getenv("TURBOCIDER_QWEN21_COMPILED_COMPLEMENT");
    if(!binary_option_or_unset(raw))throw std::invalid_argument("Qwen compiled complement requires0 or1");
    if(!option_enabled(raw))return false;
    const auto channels=std::getenv("TURBOCIDER_PRIVATE_ANE_CHANNELS");
    const auto activation=std::getenv("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION");
    const auto projection=std::getenv("TURBOCIDER_PRIVATE_ANE_W8_PROJECTION");
    if(!private_qkv_joint_request(r) || !option_enabled(std::getenv("TURBOCIDER_QWEN21_PRIVATE_QKV")) ||
       !option_enabled(std::getenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM")) ||
       !channels || std::string_view(channels)!="6144" || !activation || std::string_view(activation)!="sigmoid-complement" ||
       (projection && std::string_view(projection)!="descriptor") ||
       option_enabled(std::getenv("TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU")) ||
       std::getenv("TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS"))
        throw std::invalid_argument("compiled complement requires explicit joint base40 resident512 fixed6144 sigmoid-complement descriptor without GPU-dual/family");
    return true;
}
// Source-bound scalar state only. The compiled function takes X and both W
// arrays explicitly; no layer weights are cached or captured here.
struct RuntimeCompiledComplementState {
    std::string generation,failure_reason;
    bool validated=false,disabled=false;
    uint64_t validation_calls=0,traces=0;
    double validation_seconds=0;
    void fail(const std::string &reason){disabled=true;if(failure_reason.empty())failure_reason=reason;}
    void bind(const std::string &source) {
        if(source.empty())fail("compiled complement source identity is empty");
        else if(generation.empty())generation=source;
        else if(generation!=source)fail("compiled complement source generation changed; new owner required");
    }
    bool available()const{return !disabled;}
    bool healthy()const{return validated&&!disabled;}
    void accept(){if(!disabled)validated=true;}
};
}
