#pragma once
#include "../../core/contracts.hpp"
#include "diagnostic_options.hpp"
#include "../../platform/apple/platform.hpp"
#include <cstdlib>

namespace tc::qwen21 {
// Qualified small-canvas BF16 route: consume the language encoder one layer
// at a time, then stream DiT layers on every step, then load the VAE. Larger canvases,
// multimodal conditioning and PE retain their existing estimate.
inline bool layer_staged_hybrid_t2i(const Request &r, const DeviceInfo &device) {
    if (!device.optimizations().qwen21_layer_streaming) return false;
    // Existing diagnostic caches/fusions retain weights or change execution
    // geometry. Keep their established memory estimate and resident route.
    for (const char *flag : {"TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN",
            "TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC",
            "TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC"})
        if (option_enabled(std::getenv(flag))) return false;
    return r.model == "qwen-image-2.1" && r.residency == "component_staged" &&
        r.operation == "image.generate" && r.inputs.empty() &&
        r.execution == "gpu_ane" && r.qwen21_w8a8 &&
        r.allow_approximation && !r.ane_manifest.empty() &&
        r.loras.empty() && !r.prompt_enhance && !r.qwen21_gpu_w8a16 &&
        r.qwen21_gpu_full_ffn_blocks.empty() && r.width == 512 && r.height == 512 &&
        r.steps >= 2;
}
inline bool layer_staged_t2i(const Request &r, const DeviceInfo &device) {
    if (!device.optimizations().qwen21_layer_streaming) return false;
    if (layer_staged_hybrid_t2i(r, device)) return true;
    return r.model == "qwen-image-2.1" && r.residency == "component_staged" &&
        r.operation == "image.generate" && r.inputs.empty() &&
        (r.execution == "gpu" || r.execution == "auto") &&
        r.loras.empty() && !r.prompt_enhance && !r.allow_approximation &&
        r.width <= 512 && r.height <= 512;
}
} // namespace tc::qwen21
