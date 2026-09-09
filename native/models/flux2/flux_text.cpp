#include "flux.hpp"
#include "../../components/text/qwen3.hpp"

namespace tc {
Tensor Flux::encode(const Tokens &tokens, const Event &event,
                     std::atomic<bool> &cancelled) {
    Weights weights;
    weights.load(root_ / "text_encoder", event, cancelled);
    if (!active_loras_.empty())
        weights.apply_loras(active_loras_, "text_encoder", event, cancelled);
    return components::qwen3_conditioning(
        tokens, weights, components::Qwen3Conditioning::flux_klein(), event, cancelled);
}
} // namespace tc
