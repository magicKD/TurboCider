#pragma once
#include "vae.hpp"
#include "../../runtime/streaming/qwen_bf16_metrics.hpp"
#include <limits>

namespace tc::qwen21 {
// Only the independent streamed generation executor calls this, before any
// source array materialization. The counts are logical bytes, not RSS savings.
inline QwenStreamVaeDecodeResidencyMetrics retain_streamed_vae_decode_weights(
        Weights &weights, std::atomic<bool> &cancelled) {
    checkpoint(cancelled);
    {
        // Preserve full-table shape and canonical-alias validation, including
        // unused encoder fields. Original mode creates only lazy views; it
        // never enters the optional FP16 evaluation/nonfinite-check branch.
        VAE metadata_validation(weights, VAE::DecodeConvolution::Original, false);
    }
    QwenStreamVaeDecodeResidencyMetrics metrics;
    std::vector<std::string> dropped;
    for (const auto &key : weights.sorted_keys()) {
        const auto name = VAE::canonical_key(key);
        const bool keep = key.find("time_conv") == std::string::npos &&
            (name.starts_with("decoder.") || name.starts_with("post_quant_conv."));
        auto &count = keep ? metrics.retained_tensors : metrics.dropped_tensors;
        auto &total = keep ? metrics.retained_logical_bytes : metrics.dropped_logical_bytes;
        const auto bytes = uint64_t(weights.at(key).nbytes());
        require(count < std::numeric_limits<uint64_t>::max() &&
                bytes <= std::numeric_limits<uint64_t>::max() - total,
                "Qwen streamed VAE logical byte count overflow");
        ++count;
        total += bytes;
        if (!keep) dropped.push_back(key);
    }
    checkpoint(cancelled);
    // Preflight/counting completes before mutation. erase preserves retained
    // source identities, raw legacy keys, dtypes and saved weight metadata.
    for (const auto &key : dropped) weights.erase(key);
    metrics.metadata_preflight = true;
    return metrics;
}
} // namespace tc::qwen21
