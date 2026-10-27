#pragma once
#include "../../backends/mlx.hpp"
#include <memory>
#include <unordered_map>

namespace tc::components {
// Native port of madebyollin/taesd. Public latents use the diffusion model's
// normalized latent space; images are NCHW in [0,1] (RGBA for Qwen 2.1).
class TinyAutoencoder {
  public:
    TinyAutoencoder(const std::filesystem::path &, const std::string &model, bool encoder = true,
                    bool decoder = true);
    Tensor encode(const Tensor &, const Event &, std::atomic<bool> &) const;
    Tensor decode(const Tensor &, const Event &, std::atomic<bool> &) const;
    int channels() const {
        return qwen_ ? 64 : 16;
    }
    int scale() const {
        return qwen_ ? 16 : 8;
    }

  private:
    bool qwen_;
    std::unordered_map<std::string, Tensor> weights_;
    const Tensor &get(const std::string &) const;
    Tensor conv(const Tensor &, const std::string &, int stride = 1) const;
    Tensor block(const Tensor &, const std::string &) const;
};

// One bounded, replace-in-place preview, separate from final output publication.
// Disabled requests do no model loading, file I/O or GPU preview work.
class TinyPreview {
  public:
    TinyPreview(const Request &, const Event &, std::atomic<bool> &);
    ~TinyPreview();
    bool due(int step) const;
    void publish(const Tensor &nchw, int completed);

  private:
    const Request &request_;
    const Event &event_;
    std::atomic<bool> &cancelled_;
    std::unique_ptr<TinyAutoencoder> decoder_;
    std::filesystem::path path_;
};
} // namespace tc::components
