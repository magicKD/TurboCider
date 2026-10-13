#pragma once
#include "../../backends/mlx.hpp"
#include "geometry.hpp"
#include <map>

namespace tc::seedvr2 {
using FFNRoute = std::function<Tensor(int, const Tensor &)>;

// An unquantized checkpoint bank. The original checkpoint layout is retained;
// VAE convolution filters are converted once to MLX's channel-last layout.
class TensorBank {
    std::unordered_map<std::string, Tensor> tensors_;
  public:
    void load(const std::filesystem::path &, mx::Dtype compute_dtype,
              bool convolution_filters = false);
    const Tensor &at(const std::string &) const;
    bool has(const std::string &) const;
    Tensor linear(const Tensor &, const std::string &) const;
    void clear() { tensors_.clear(); }
    size_t bytes() const;
};

class DiT {
    struct WindowGroup {
        std::array<int, 3> shape;
        int count;
        Tensor indices, cosine, sine;
    };
    struct Layout {
        std::vector<WindowGroup> groups;
        Tensor reverse, text_cosine, text_sine;
        int windows;
    };
    TensorBank weights_;
    mx::Dtype dtype_ = mx::bfloat16;
    std::map<std::array<int, 5>, Layout> layouts_;
    Layout make_layout(int frames, int height, int width, int text_tokens, bool shifted) const;
    const Layout &layout(int frames, int height, int width, int text_tokens, bool shifted);
    Tensor modulate(const Tensor &, const Tensor &, int layer,
                    const std::string &prefix, bool output, bool learned = true) const;
    Tensor feed_forward(const Tensor &, const std::string &) const;
    std::vector<Tensor> attention(const Tensor &, const Tensor &, int block, const Layout &) const;
  public:
    void load(const std::filesystem::path &, mx::Dtype = mx::bfloat16);
    size_t bytes() const { return weights_.bytes(); }
    mx::Dtype dtype() const { return dtype_; }
    Tensor time_embedding(float timestep) const;
    std::vector<Tensor> embed(const Tensor &latent, const Tensor &condition) const;
    std::vector<Tensor> block(const Tensor &video, const Tensor &text,
                             const Tensor &time, int index,
                             int frames, int height, int width, const FFNRoute & = {});
    Tensor ffn_partial(const Tensor &, int block, int first_channel, int last_channel) const;
    Tensor ffn_full(const Tensor &, int block) const;
    Tensor output(const Tensor &video, const Tensor &time, const mx::Shape &) const;
    // BCHTW external layout; matching sampling noise can be supplied by callers.
    Tensor forward(const Tensor &latent, const Tensor &condition, float timestep,
                   const Event &, std::atomic<bool> &, const std::filesystem::path &dump = {},
                   const FFNRoute & = {});
    void clear() { weights_.clear(); layouts_.clear(); }
};

} // namespace tc::seedvr2
