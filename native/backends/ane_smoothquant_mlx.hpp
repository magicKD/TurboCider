#pragma once

#include "ane_smoothquant.hpp"
#include "ane_ffn.hpp"
#include "ane_backend.hpp"

namespace tc::ane::smoothquant {
inline std::string configured_value(const char *key, const char *fallback) {
    const char *value = std::getenv(key);
    return value && *value ? value : fallback;
}
inline std::string requested_path() {
    return configured_value("TURBOCIDER_PRIVATE_ANE_S1_PROFILE", "");
}
// Fail before expensive model work when an explicit profile could be ignored.
// The App's typed worker profiles deliberately do not expose this experiment
// until a real candidate end-to-end run has been accepted.
inline void validate_request(const Request &request) {
    if (requested_path().empty()) return;
    require((request.model == "z-image-turbo" || request.model == "qwen-image-2.1") &&
            request.hybrid_mlp_mode == "runtime" && request.execution == "gpu_ane" &&
            request.allow_approximation && request.width == 512 && request.height == 512,
            "experimental S1 requires an explicit local dense 512px GPU+ANE Runtime request");
    const auto policy = configured_backend();
    require(policy.allow_private && (policy.preferred == BackendPreference::Private ||
            policy.preferred == BackendPreference::Auto) &&
            configured_value("TURBOCIDER_PRIVATE_ANE_DATA_PATH", "fp16") == "w8a8" &&
            configured_value("TURBOCIDER_PRIVATE_ANE_GPU_IO", "0") == "1" &&
            private_channel_count(request.model == "z-image-turbo" ? 10240 : 12288) > 0,
            "experimental S1 requires authorized private W8A8 device channel staging");
    require(configured_value("TURBOCIDER_ANE_CALIBRATION_DIR", "").empty(),
            "capture and an experimental S1 profile must use separate requests");
    require(request.model != "qwen-image-2.1" ||
            configured_value("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV", "0") != "1",
            "experimental S1 is not qualified with resident prefix KV reuse");
}
inline Binding request_binding(const Request &request, const std::filesystem::path &checkpoint,
                               const std::string &adapter_identity, int hidden) {
    Binding binding;
    binding.experimental = true; binding.checkpoint = checkpoint;
    binding.model_id = request.model; binding.hidden = hidden;
    binding.width = request.width; binding.height = request.height; binding.total_steps = request.steps;
    binding.reference_count = int(request.inputs.size());
    binding.reference_size = request.inputs.empty() ? 0 : request.qwen21_reference_size;
    binding.configured_backend = configured_value("TURBOCIDER_ANE_BACKEND", "public");
    binding.runtime_recipe = configured_value("TURBOCIDER_PRIVATE_ANE_DATA_PATH", "fp16") +
        ":channels=" + configured_value("TURBOCIDER_PRIVATE_ANE_CHANNELS", "auto");
    for (size_t i = 0; i < request.loras.size(); ++i)
        binding.loras.push_back({std::filesystem::path(request.loras[i].path).filename().string(),
            fingerprint_adapter(adapter_identity,i),request.loras[i].strength});
    return binding;
}
inline std::vector<Tensor> scale_tensors(const ExperimentalProfile &profile, int hidden) {
    std::vector<Tensor> scales;
    scales.reserve(32);
    for (int layer = 0; layer < 32; ++layer) {
        const auto values = profile.scales_for(layer);
        require(values.size() == size_t(hidden), "experimental S1 hidden dimension mismatch");
        scales.emplace_back(values.data(),mx::Shape{1,hidden},mx::float32);
    }
    return scales;
}
inline void bind_request(HybridFfn &runtime, const Request &request,
                         const std::filesystem::path &checkpoint,
                         const std::string &adapter_identity, int hidden) {
    const auto path = requested_path();
    if (path.empty()) {
        runtime.set_smoothquant({});
        runtime.begin_request(adapter_identity);
        return;
    }
    validate_request(request);
    const auto profile = ExperimentalProfile::load(path,
        request_binding(request,checkpoint,adapter_identity,hidden));
    runtime.set_smoothquant(profile.content_digest(),[&] { return scale_tensors(profile,hidden); });
    // Same path with changed contents must never reuse an old cost model.
    runtime.begin_request(adapter_identity + ":smoothquant-s1:" + profile.content_digest());
}
} // namespace tc::ane::smoothquant
