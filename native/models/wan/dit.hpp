#pragma once

#include "checkpoint.hpp"

namespace tc::wan {

// Pure tensor stages: ownership, sampling and cancellation belong to the
// session. The explicit path is also the numerical oracle for graph fusion.
class DiT {
    const Checkpoint &weights_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)> compiled_;
    std::vector<mx::Shape> signature_;
    Tensor normalize(const Tensor &) const;
    Tensor rms_normalize(const Tensor &, const std::string &) const;
    Tensor attention(const Tensor &, const Tensor &, const Tensor &) const;
    Tensor rotary(const Tensor &, const Tensor &, const Tensor &) const;
    std::vector<Tensor> pre_ffn(const Tensor &, const Tensor &, const Tensor &,
                                const Tensor &, const Tensor &, int) const;

  public:
    using SplitFFN = std::function<Tensor(int, const Tensor &, const Tensor &, const Tensor &)>;
    explicit DiT(const Checkpoint &weights) : weights_(weights) {}
    DiT(const DiT &) = delete;
    DiT &operator=(const DiT &) = delete;
    Tensor patch_embed(const Tensor &) const;
    // Returns time embedding, per-block modulation, projected text context.
    std::vector<Tensor> condition(const Tensor &timestep, const Tensor &text) const;
    Tensor block(const Tensor &, const Tensor &context, const Tensor &modulation,
                 const Tensor &cosine, const Tensor &sine, int index) const;
    Tensor output(const Tensor &, const Tensor &time_embedding,
                  const mx::Shape &latent_shape) const;
    Tensor forward(const Tensor &latent, const Tensor &text, const Tensor &timestep,
                   const Tensor &cosine, const Tensor &sine,
                   const Event &, std::atomic<bool> &) const;
    Tensor forward_compiled(const Tensor &latent, const Tensor &text, const Tensor &timestep,
                            const Tensor &cosine, const Tensor &sine,
                            const Event &, std::atomic<bool> &);
    Tensor forward_hybrid(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &,
                          const SplitFFN &, const Event &, std::atomic<bool> &) const;
};

} // namespace tc::wan
