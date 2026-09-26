#pragma once
#include "../../backends/mlx.hpp"
#include <array>
#include "sequence.hpp"

namespace tc::qwen21 {
struct ReferenceLatents {
    Tensor latents; // [1,height*width,channels], normalized VAE latents
    ReferenceGeometry geometry;
};

struct TransformerConfig {
    int layers = 32;
    int heads = 32;
    int head_dim = 128;
    int channels = 64;
    int context_dim = 4096;
    std::array<int, 3> rope_axes{16, 56, 56};
    float epsilon = 1e-6f;
    int hidden() const { return heads * head_dim; }
};

// One request owns a Transformer. Prefix tensors are deliberately not shared
// across requests, weights, guidance branches, or different image conditions.
class Transformer {
  public:
    Transformer(const Weights &, TransformerConfig = {});
    Transformer(const Transformer &) = delete;
    Transformer &operator=(const Transformer &) = delete;
    void reset();
    Tensor forward(const Tensor &latents, const Tensor &text, float timestep,
                   int latent_height, int latent_width, bool cache_prefix = true,
                   std::unordered_map<std::string, Tensor> *trace = nullptr,
                   const std::vector<ReferenceLatents> &references = {});
    size_t cached_layers() const { return prefix_.size(); }
    // Experimental decode-only split: prefill remains exact GPU so cached
    // conditioning is unchanged. Caller owns the callback's runtime/session.
    using DecodeMLP = std::function<Tensor(int, const Tensor &)>;
    void set_decode_mlp(DecodeMLP fn) { decode_mlp_ = std::move(fn); decode_blocks_.clear(); }

  private:
    struct KV { Tensor key, value; };
    const Weights &weights_;
    TransformerConfig config_;
    std::vector<KV> prefix_;
    std::optional<Tensor> cached_text_;
    std::vector<Tensor> cached_references_;
    std::vector<ReferenceGeometry> reference_geometry_;
    SequenceGeometry sequence_;
    std::optional<Tensor> cosine_, sine_;
    int text_length_ = 0, height_ = 0, width_ = 0;
    using BlockFunction = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;
    std::vector<BlockFunction> prefill_blocks_, decode_blocks_;
    DecodeMLP decode_mlp_;
    Tensor embedding(float timestep, mx::Dtype) const;
    void geometry(int text_length, int height, int width, const std::vector<ReferenceGeometry> &);
};
} // namespace tc::qwen21
