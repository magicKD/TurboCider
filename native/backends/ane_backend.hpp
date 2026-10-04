#pragma once
#include "ane_runtime.hpp"
#include <cstdlib>

namespace tc::ane {
enum class BackendPreference { Public, Private, Auto, Off };
struct BackendPolicy { BackendPreference preferred = BackendPreference::Public; bool allow_private = false; };
inline int private_channel_count(int full_width) {
    const char *raw = std::getenv("TURBOCIDER_PRIVATE_ANE_CHANNELS");
    if (!raw) return 0;
    const std::string value(raw);
    if (value.empty() || value.size() > 5 || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_CHANNELS requires 0 or aligned intermediate channels");
    const int channels = std::stoi(value);
    if (channels && (full_width % 512 || channels % 512 || channels >= full_width))
        throw std::runtime_error("private ANE channels must be a positive 512 multiple smaller than the full FFN width");
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
                                    GraphGeometry expected, BackendPolicy policy);
}
