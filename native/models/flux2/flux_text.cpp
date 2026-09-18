#include "flux.hpp"
#include "../../components/text/qwen3.hpp"
#include <algorithm>

namespace tc {
Tensor Flux::encode(const Tokens &tokens, const Event &event,
                     std::atomic<bool> &cancelled, HybridSession *encoder_hybrid) {
    Weights weights;
    if (public_stream_lease_) {
        std::vector<std::string> artifacts;
        for (const auto &file : public_stream_lease_->descriptor().files)
            if (file.logical_id.starts_with("text_encoder/") &&
                file.logical_id.ends_with(".safetensors"))
                artifacts.push_back(file.logical_id);
        std::sort(artifacts.begin(), artifacts.end());
        weights.load_lease(public_stream_lease_, artifacts, event, cancelled);
    } else {
        weights.load(root_ / "text_encoder", event, cancelled);
    }
    if (!active_loras_.empty())
        weights.apply_loras(active_loras_, "text_encoder", event, cancelled);
    return components::qwen3_conditioning(
        tokens, weights, components::Qwen3Conditioning::flux_klein(), event, cancelled,
        encoder_hybrid);
}
} // namespace tc
