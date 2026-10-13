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
    std::string linear_recipe_ = "mlx";
    mx::Dtype compute_dtype_ = mx::bfloat16;
  public:
    void load(const std::filesystem::path &, mx::Dtype compute_dtype,
              bool convolution_filters = false);
    const Tensor &at(const std::string &) const;
    bool has(const std::string &) const;
    Tensor linear(const Tensor &, const std::string &) const;
    const std::string &linear_recipe() const { return linear_recipe_; }
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
    mx::Dtype dtype_ = mx::float16;
    std::string rms_recipe_ = "mlx";
    std::string time_recipe_ = "host_fp32";
    std::string attention_recipe_ = "fp16";
    std::string rope_recipe_ = "fp16_angles";
    std::string residual_recipe_ = "fp32";
    std::map<std::array<int, 5>, Layout> layouts_;
    Layout make_layout(int frames, int height, int width, int text_tokens, bool shifted) const;
    const Layout &layout(int frames, int height, int width, int text_tokens, bool shifted);
    Tensor modulate(const Tensor &, const Tensor &, int layer,
                    const std::string &prefix, bool output, bool learned = true,
                    bool rms_interface_roundtrip = true) const;
    Tensor feed_forward(const Tensor &, const std::string &) const;
    Tensor rms_norm(const Tensor &, const std::optional<Tensor> &weight = {}) const;
    std::vector<Tensor> attention(const Tensor &, const Tensor &, int block, const Layout &,
                                 std::unordered_map<std::string,Tensor> *trace = nullptr) const;
  public:
    void load(const std::filesystem::path &, mx::Dtype = mx::float16);
    size_t bytes() const { return weights_.bytes(); }
    mx::Dtype dtype() const { return dtype_; }
    const std::string &rms_recipe() const { return rms_recipe_; }
    const std::string &time_recipe() const { return time_recipe_; }
    const std::string &attention_recipe() const { return attention_recipe_; }
    const std::string &rope_recipe() const { return rope_recipe_; }
    const std::string &linear_recipe() const { return weights_.linear_recipe(); }
    const std::string &residual_recipe() const { return residual_recipe_; }
    bool stable_fp16_profile() const {
        return dtype_ == mx::float16 && residual_recipe_ == "fp32" &&
            time_recipe_ == "host_fp32" && attention_recipe_ == "fp16" &&
            rope_recipe_ == "fp16_angles" && weights_.linear_recipe() == "mlx";
    }
    Tensor time_embedding(float timestep, const std::filesystem::path &dump = {}) const;
    std::vector<Tensor> embed(const Tensor &latent, const Tensor &condition) const;
    std::vector<Tensor> block(const Tensor &video, const Tensor &text,
                             const Tensor &time, int index,
                             int frames, int height, int width, const FFNRoute & = {},
                             const std::filesystem::path &dump = {});
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
