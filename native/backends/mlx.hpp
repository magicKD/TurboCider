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
    bool metal_convrot_ = false;

  public:
    void load(const std::filesystem::path &, const Event &, std::atomic<bool> &);
    void load_file(const std::filesystem::path &, const std::string &prefix = "");
    void load_gguf_file(const std::filesystem::path &);
    void remap_keys(const std::function<std::string(const std::string &)> &);
    void fuse_keys(const std::string &, const std::vector<std::string> &, int axis);
    void cast_unquantized_float32(mx::Dtype);
    void set_metal_convrot(bool enabled) { metal_convrot_ = enabled; }
    bool metal_convrot() const { return metal_convrot_; }
    std::vector<std::string> sorted_keys() const;
    void bind_arrays(const std::vector<std::string> &,
                     const std::vector<Tensor> &, size_t offset = 0);
    // Pack Comfy signed tensor-wise INT8 ConvRot rows into MLX affine Q8.
    // The no-argument overload preserves the historical Z-Image g32 profile
    // and its FP32 diagnostic environment switch. LTX explicitly selects
    // g64/FP32 scales to match its MLX reference.
    size_t pack_convrot_q8();
    size_t pack_convrot_q8(int group_size, mx::Dtype scale_dtype);
    size_t pack_comfy_nvfp4();
    // Quantize an aligned dense matrix range into MLX affine W8. The logical
    // shape of the stored matrix becomes precisely the selected GPU shard.
    void quantize_dense_range(const std::string &prefix, int row_start, int row_end,
                              int col_start, int col_end, int group_size = 32);
    // Offline channel routing: gather exactly the GPU-owned FFN channels
    // before packing W8; no gather/scatter is left in the inference graph.
    void quantize_dense_indices(const std::string &prefix,
                                const std::vector<int> &channel_indexes,
                                int axis, int group_size = 32);
    // Experimental routed BF16 complement: retain exactly the GPU-owned
    // channels, in source order, without changing their storage dtype.
    void select_dense_indices(const std::string &prefix,
                              const std::vector<int> &channel_indexes, int axis);
    void dequantize(const std::vector<std::string> &);
    const Tensor &at(const std::string &) const;
    bool has(const std::string &) const;
    void erase(const std::string &);
    void erase_prefix(const std::string &);
    bool quantized(const std::string &) const;
    bool convrot(const std::string &) const;
    bool nvfp4(const std::string &) const;
    bool has_runtime_loras() const { return !runtime_loras_.empty(); }
    Tensor project(const Tensor &, const std::string &) const;
    std::vector<Tensor> project_many(const Tensor &,
                                     const std::vector<std::string> &) const;
    // Project one aligned matrix slice. The input contains exactly the
    // selected column range; this is used by exact tensor-parallel MLP
    // branches without materializing the full gate/up activation.
    Tensor project_slice(const Tensor &, const std::string &, int row_start,
                         int row_end, int col_start, int col_end,
                         bool add_bias = true) const;
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
              const std::optional<Tensor> &mask = {}, bool force_fused = false,
              const std::string &mask_mode = "");
Tensor rope_pairs(const Tensor &, const Tensor &, const Tensor &);
std::vector<Tensor> rope_pairs_pair(const Tensor &, const Tensor &,
                                    const Tensor &, const Tensor &);
Tensor euler_step(const Tensor &, const Tensor &, float dt);
std::vector<float> flux_gpu_sigmas(int image_tokens, int steps);
} // namespace tc
