#pragma once

// Diagnostic only. A same-process observer receives the actual arguments and
// compiled callable of an ordinary GPU block; it must not escape these borrowed
// references. No observer is installed on the normal generation path.
#include "../../backends/mlx.hpp"
#include "sequence.hpp"

namespace tc::qwen21 {

using DiagnosticGpuBlock =
    std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;

struct DiagnosticBlockObservation {
    int layer;
    bool steady;
    int prefix_tokens;
    const Weights& weights;
    const std::vector<SequenceSegment>& segments;
    const std::vector<Tensor>& arguments;
    const std::vector<Tensor>& outputs;
    const DiagnosticGpuBlock& gpu_block;
};

using DiagnosticBlockObserver = std::function<void(const DiagnosticBlockObservation&)>;

// Only the bounded diagnostic driver interprets this exception as success.
// Production request handling must still unwind and drain its normal owners.
struct DiagnosticBlockReplayComplete final : std::runtime_error {
    DiagnosticBlockReplayComplete()
        : std::runtime_error("bounded Qwen block replay completed before image decode") {}
};

} // namespace tc::qwen21
