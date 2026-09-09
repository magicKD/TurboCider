#pragma once

#include "../../components/weights/affine.hpp"

namespace tc::wan {

struct DiTConfig {
    int heads = 12;
    int head_dim = 128;
    int layers = 30;
    int ffn_dim = 8960;
    int frequency_dim = 256;
    int text_dim = 4096;
    int channels = 16;
    std::vector<int> patch = {1, 2, 2};
    float epsilon = 1e-6f;
    int hidden() const { return heads * head_dim; }
};

// Native loader for the version-1 MLX Wan checkpoint. Conversion from upstream
// weights is an offline concern. Loading never imports Python or downloads data.
class Checkpoint {
    std::unordered_map<std::string, Tensor> arrays_;
    std::unordered_map<std::string, components::AffineMatrix> quantized_;
    DiTConfig config_;

  public:
    void load(const std::filesystem::path &root, const Event &, std::atomic<bool> &);
    const DiTConfig &config() const { return config_; }
    const Tensor &at(const std::string &) const;
    bool has(const std::string &) const;
    Tensor linear(const Tensor &, const std::string &) const;
    const components::AffineMatrix &affine(const std::string &) const;
    size_t bytes() const;
    void clear();
};

} // namespace tc::wan
