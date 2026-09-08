#pragma once
#include "common.hpp"
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/io.h>
#include <mlx/memory.h>
#include <optional>
#include <unordered_map>
#include <regex>
#include <functional>
namespace tc {
void configure_streams();
namespace mx = mlx::core;
using Tensor = mx::array;
class Weights {
    struct RuntimeLoRA {
        Tensor down;
        Tensor up;
        float scale = 1.f;
        int output_start = 0;
        int output_end = 0;
    };
    std::unordered_map<std::string, Tensor> values_;
    std::unordered_map<std::string, std::vector<RuntimeLoRA>> runtime_loras_;

  public:
    void load(const std::filesystem::path &, const Event &, std::atomic<bool> &);
    void load_file(const std::filesystem::path &, const std::string &prefix = "");
    void load_gguf_file(const std::filesystem::path &);
    void remap_keys(const std::function<std::string(const std::string &)> &);
    void fuse_keys(const std::string &, const std::vector<std::string> &, int axis);
    void cast_unquantized_float32(mx::Dtype);
    size_t pack_convrot_q8();
    void dequantize(const std::vector<std::string> &);
    const Tensor &at(const std::string &) const;
    bool has(const std::string &) const;
    void erase(const std::string &);
    void erase_prefix(const std::string &);
    bool quantized(const std::string &) const;
    bool convrot(const std::string &) const;
    bool has_runtime_loras() const { return !runtime_loras_.empty(); }
    Tensor project(const Tensor &, const std::string &) const;
    Tensor project_range(const Tensor &, const std::string &, int row_start, int row_end,
                        int col_start, int col_end) const;
    void clear();
    size_t bytes() const;
    void materialize();
    size_t apply_loras(const std::vector<LoRAAsset> &, const std::string &, const Event &,
                      std::atomic<bool> &, bool inference_time = false);
};
Tensor linear(const Tensor &, const Weights &, const std::string &);
Tensor silu(const Tensor &);
Tensor rms(const Tensor &, const Tensor &, float eps);
Tensor norm(const Tensor &);
Tensor slice_axis(const Tensor &, int axis, int start, int stop);
Tensor heads(const Tensor &, int count, int dim);
Tensor attend(const Tensor &, const Tensor &, const Tensor &, bool fp32 = false,
              const std::optional<Tensor> &mask = {}, bool force_fused = false);
Tensor rope_pairs(const Tensor &, const Tensor &, const Tensor &);
std::vector<Tensor> rope_pairs_pair(const Tensor &, const Tensor &,
                                    const Tensor &, const Tensor &);
Tensor euler_step(const Tensor &, const Tensor &, float dt);
std::vector<float> flux_gpu_sigmas(int image_tokens, int steps);
} // namespace tc
