#pragma once
#include "ane_runtime.hpp"
#include <cstdlib>

namespace tc::ane {
enum class BackendPreference { Public, Private, Auto, Off };
struct BackendPolicy { BackendPreference preferred = BackendPreference::Public; bool allow_private = false; };
inline bool configured_convrot_bf16_boundaries() {
    const char *raw=std::getenv("TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES");
    if(raw && std::string(raw)!="0" && std::string(raw)!="1")
        throw std::runtime_error("ConvRot BF16 value boundaries require 0 or 1");
    return raw && std::string(raw)=="1";
}
inline int parse_private_channel_count(int full_width, const char *raw) {
    if (!raw) return 0;
    const std::string value(raw);
    if (value == "auto") return -1; // must be resolved by native calibration, never a graph width
    if (value.empty() || value.size() > 5 || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_CHANNELS requires 0 or aligned intermediate channels");
    const int channels = std::stoi(value);
    if (channels && (full_width % 512 || channels % 512 || channels >= full_width))
        throw std::runtime_error("private ANE channels must be a positive 512 multiple smaller than the full FFN width");
    return channels;
}
inline int private_channel_count(int full_width) {
    return parse_private_channel_count(full_width,std::getenv("TURBOCIDER_PRIVATE_ANE_CHANNELS"));
}
inline int resolved_private_channel_count(int full_width, std::optional<int> calibrated = std::nullopt,
                                         std::optional<int> explicit_override = std::nullopt) {
    if(explicit_override) {
        if(calibrated)throw std::runtime_error("explicit channel override cannot be combined with calibration");
        const auto text=std::to_string(*explicit_override);
        return parse_private_channel_count(full_width,text.c_str());
    }
    const int configured = private_channel_count(full_width);
    if (!calibrated) {
        if (configured < 0) throw std::runtime_error("automatic ANE channels require a native calibration workload");
        return configured;
    }
    const int channels = *calibrated;
    if (channels < 0 || (channels && (full_width % 512 || channels % 512 || channels >= full_width)))
        throw std::runtime_error("calibrated ANE channels must be zero or a positive aligned partial width");
    if (configured >= 0 && configured != channels)
        throw std::runtime_error("calibration cannot override an explicit ANE channel configuration");
    return channels;
}
inline BackendPolicy configured_backend() {
    BackendPolicy policy;
    if (const char *allow = std::getenv("TURBOCIDER_ALLOW_PRIVATE_ANE")) {
        const std::string value(allow);
        if (value != "0" && value != "1") throw std::runtime_error("TURBOCIDER_ALLOW_PRIVATE_ANE requires 0 or 1");
        policy.allow_private = value == "1";
    }
    const std::string value = std::getenv("TURBOCIDER_ANE_BACKEND") ? std::getenv("TURBOCIDER_ANE_BACKEND") : "public";
    if (value == "public") policy.preferred = BackendPreference::Public;
    else if (value == "private") policy.preferred = BackendPreference::Private;
    else if (value == "auto") policy.preferred = BackendPreference::Auto;
    else if (value == "off") policy.preferred = BackendPreference::Off;
    else throw std::runtime_error("TURBOCIDER_ANE_BACKEND requires public/private/auto/off");
    if (policy.preferred == BackendPreference::Private && !policy.allow_private)
        throw std::runtime_error("private ANE requires TURBOCIDER_ALLOW_PRIVATE_ANE=1");
    return policy;
}
struct BuiltExecutor {
    std::unique_ptr<Executor> executor;
    bool self_test_passed = false;
    double self_test_seconds = 0;
    std::string fallback_reason;
};
BuiltExecutor build_runtime_executor(const std::filesystem::path &manifest, size_t budget,
                                    GraphGeometry expected, BackendPolicy policy,
                                    std::optional<int> calibrated_channels = std::nullopt,
                                    std::optional<int> channel_override = std::nullopt);
// Explicit Private native geometry only. The original template is still
// validated against the full model; no Public artifact is resized/relabelled.
BuiltExecutor build_runtime_executor(const std::filesystem::path &,size_t,GraphGeometry,BackendPolicy,
                                    std::optional<int>,std::optional<int>,std::optional<int> bucket_override);
}
