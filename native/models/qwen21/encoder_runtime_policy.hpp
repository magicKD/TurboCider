#pragma once
#include "../../core/contracts.hpp"
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <stdexcept>

namespace tc::qwen21::encoder_runtime {
enum class GpuRecipe { LegacyFp32Constants, MppDynamicBf16 };
inline GpuRecipe gpu_recipe() {
    // Process fixed so cached conditioning cannot cross precision/layout recipes.
    static const auto recipe=[] {
        const char* raw=std::getenv("TURBOCIDER_QWEN21_ENCODER_RUNTIME_GPU");
        const std::string_view value=raw?raw:"mpp";
        if(value=="mpp")return GpuRecipe::MppDynamicBf16;
        if(value=="legacy")return GpuRecipe::LegacyFp32Constants;
        throw std::invalid_argument("encoder runtime GPU recipe requires mpp or legacy");
    }();
    return recipe;
}
inline std::string gpu_recipe_identity() {
    return gpu_recipe()==GpuRecipe::MppDynamicBf16?"mpp-bf16-operands-fp32-partial-dynamic-v1":"legacy-fp32-constants-v1";
}
// Process-fixed research route, separate from the unsupported frozen encoder
// manifest request. Never enables private APIs or changes default GPU routing.
inline const std::filesystem::path& diagnostic_manifest() {
    static const std::filesystem::path path=[] {
        const char* value=std::getenv("TURBOCIDER_QWEN21_ENCODER_RUNTIME_MANIFEST");
        if(!value || std::string_view(value).empty())return std::filesystem::path{};
        std::filesystem::path p(value);
        if(!p.is_absolute() || p!=p.lexically_normal() || p.extension()!=".json" || p.native().size()>4096)
            throw std::invalid_argument("encoder runtime manifest requires a bounded normalized absolute JSON path");
        return p;
    }();
    return path;
}
inline bool supported_request(const Request& r) {
    return r.model=="qwen-image-2.1" && r.operation=="image.edit" && r.execution=="gpu" &&
        r.width==512 && r.height==512 && r.inputs.size()>=1 && r.inputs.size()<=3 &&
        (r.qwen21_reference_size==512 || r.qwen21_reference_size==1024) &&
        r.hybrid_mlp_mode=="auto" && r.encoder_ane_manifest.empty() && r.ane_manifest.empty() &&
        r.qwen21_denoiser_backend=="mlx" && r.allow_approximation && !r.prompt_enhance && !r.prompt_enhance_edit_experimental;
}
inline bool eligible_rows(int rows) {
    if(rows<1 || rows>32768)throw std::invalid_argument("encoder runtime rows outside 1..32768");
    // One prefill has little amortization. Small edit/text sequences stay GPU.
    return rows>=1024;
}
inline void validate_bucket(int rows) {
    if(rows<512 || rows>2048 || rows%128)
        throw std::invalid_argument("encoder runtime program bucket must be 512..2048 in 128-row steps");
}
inline bool configured_bool(const char* key) {
    const char* raw=std::getenv(key);
    if(raw && std::string_view(raw)!="0" && std::string_view(raw)!="1")
        throw std::invalid_argument(std::string(key)+" requires 0 or 1");
    return raw && std::string_view(raw)=="1";
}
inline int select_bucket(int rows,int descriptor_rows) {
    validate_bucket(descriptor_rows);
    if(rows && !eligible_rows(rows))return descriptor_rows;
    const char* raw=std::getenv("TURBOCIDER_QWEN21_ENCODER_RUNTIME_BUCKET");
    const std::string_view value=raw?raw:"descriptor";
    if(value=="descriptor")return descriptor_rows;
    if(value=="1152")return 1152;
    if(value=="1664")return 1664;
    if(value!="balanced")throw std::invalid_argument("encoder bucket requires descriptor/1152/1664/balanced");
    if(!rows)throw std::invalid_argument("balanced encoder bucket requires actual sequence rows");
    // Only two bounded program geometries; never one program per prompt.
    const int small=(rows+1151)/1152*1152,large=(rows+1663)/1664*1664;
    return (rows+1663)/1664<(rows+1151)/1152 && large<=small?1664:1152;
}
inline std::string process_recipe() {
    std::string value=gpu_recipe_identity();
    for(const char* key:{"TURBOCIDER_QWEN21_ENCODER_RUNTIME_BUCKET","TURBOCIDER_QWEN21_ENCODER_RUNTIME_REUSE",
        "TURBOCIDER_QWEN21_ENCODER_S1_PROFILE"}) {
        const char* raw=std::getenv(key);const std::string part=raw?raw:"<unset>";
        value+=":"+std::to_string(part.size())+":"+part;
    }
    return value;
}
} // namespace tc::qwen21::encoder_runtime
