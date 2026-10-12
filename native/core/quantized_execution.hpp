#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tc {
struct QuantizedExecutionConfig {
    std::optional<bool> enabled, allow_requantization;
    std::optional<uint32_t> schema_version, prefetch_layers, persistent_dense_layers;
    std::optional<std::string> mode, source_residency, decode_backend, precision_profile;
    std::optional<std::string> granularity, oversized_layer_policy, ane_compute;
    bool active() const noexcept { return enabled.value_or(false); }
    bool specified() const noexcept {
        return enabled || schema_version || mode || source_residency || decode_backend || precision_profile ||
            granularity || prefetch_layers || persistent_dense_layers || oversized_layer_policy || ane_compute || allow_requantization;
    }
};
void validate_quantized_execution(const QuantizedExecutionConfig &);
} // namespace tc
