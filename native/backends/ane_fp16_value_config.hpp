#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tc::ane {
inline constexpr const char *fp16_bf16_value_recipe=
    "fp16-swiglu-compact-bf16-values-canonical-zero-guarded-v1";
inline bool configured_fp16_bf16_values() {
    const char *raw=std::getenv("TURBOCIDER_PRIVATE_ANE_FP16_BF16_VALUES");
    if(raw && std::string(raw)!="0" && std::string(raw)!="1")
        throw std::runtime_error("private FP16 BF16 value policy requires 0 or 1");
    return raw && std::string(raw)=="1";
}
} // namespace tc::ane
