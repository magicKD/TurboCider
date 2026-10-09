#pragma once
#include "common.hpp"
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/io.h>
#include <mlx/io/load.h>
#include <mlx/memory.h>
#include <optional>
#include <unordered_map>
#include <regex>
#include <functional>
#include <memory>
#include <vector>
namespace tc {
namespace streaming { class SourceLease; }
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
    // Lease-backed MLX load primitives may read lazily. Keep their duplicate
    // descriptors alive until every array owned by this weight set is gone.
    std::vector<std::shared_ptr<mlx::core::io::Reader>> lease_readers_;
    bool metal_convrot_ = false;
    bool affine_fp32_mpp_ = false;
    bool affine_bf16_fp32_partial_ = false;
    bool runtime_lora_fp16_ = false;
    bool runtime_lora_bf16_fp32_ranks_ = false;
    bool runtime_lora_b_epilogue_ = false;
    bool runtime_lora_b_eligible(const Tensor &,const Tensor &) const;
    Tensor runtime_lora_scaled_b(const Tensor &,const Tensor &,int,int,float) const;
    Tensor runtime_lora_add_b(const Tensor &,const Tensor &,int,int,float,const Tensor &,int) const;
    Tensor runtime_lora_rank(const Tensor &,const Tensor &,int,int) const;
    Tensor add_runtime_projection_loras(const Tensor &,Tensor,const std::string &) const;
    Tensor project_slice_rank_impl(const Tensor &,const std::string &,int,int,int,int,bool,
                                  const std::vector<Tensor> *) const;
    Tensor lora_delta_slice_rank_impl(const Tensor *,const mx::Shape &,mx::Dtype,const std::string &,int,int,int,int,
                                     std::optional<mx::Dtype>,const std::vector<Tensor> *) const;
    size_t apply_loras_impl(const std::vector<LoRAAsset> &,const std::string &,const Event &,
        std::atomic<bool> &,bool,const std::shared_ptr<const streaming::SourceLease> &,
        const std::vector<std::string> &);

  public:
    void load(const std::filesystem::path &, const Event &, std::atomic<bool> &);
    // Load safetensors through duplicate descriptors owned by one
    // request-scoped SourceLease. The fd-backed MLX readers remain alive with
    // this object, so lazy arrays cannot reopen a mutable model pathname.
    void load_lease(const std::shared_ptr<const streaming::SourceLease> &,
                    const std::vector<std::string> &, const Event &,
                    std::atomic<bool> &);
    void load_file(const std::filesystem::path &, const std::string &prefix = "");
    void load_gguf_file(const std::filesystem::path &);
    void remap_keys(const std::function<std::string(const std::string &)> &);
    void fuse_keys(const std::string &, const std::vector<std::string> &, int axis);
    void cast_unquantized_float32(mx::Dtype);
    void set_metal_convrot(bool enabled) { metal_convrot_ = enabled; }
    bool metal_convrot() const { return metal_convrot_; }
    // Owner-thread request snapshot; never mutate while a compiled closure or
    // submitted projection is reading this Weights generation. Default off.
    void set_affine_fp32_mpp(bool enabled) {
        require(!enabled || !affine_bf16_fp32_partial_,"affine partial recipes are mutually exclusive");
        affine_fp32_mpp_=enabled;
    }
    bool affine_fp32_mpp() const { return affine_fp32_mpp_; }
    // Explicit approximate ConvRot base-down only: preserve original BF16
    // packed QMM, then widen its rounded output to F32 for the channel join.
    // No widened metadata/weights or per-shard down-LoRA is introduced.
    void set_affine_bf16_fp32_partial(bool enabled) {
        require(!enabled || !affine_fp32_mpp_,"affine partial recipes are mutually exclusive");
        affine_bf16_fp32_partial_=enabled;
    }
    bool affine_bf16_fp32_partial() const { return affine_bf16_fp32_partial_; }
    // Experimental Qwen21 student only: narrow LoRA matmuls while retaining
    // FP32 accumulation with the BF16 base projection.
    void set_runtime_lora_fp16(bool enabled) { runtime_lora_fp16_ = enabled; }
    // Request-owner snapshot. Original BF16 operands, F32 rank output and
    // unchanged F32 B/delta arithmetic; never an FP16-rank substitution.
    void set_runtime_lora_bf16_fp32_ranks(bool enabled) { runtime_lora_bf16_fp32_ranks_=enabled; }
    bool runtime_lora_bf16_fp32_ranks() const { return runtime_lora_bf16_fp32_ranks_; }
    void set_runtime_lora_b_epilogue(bool enabled) { runtime_lora_b_epilogue_=enabled; }
    bool runtime_lora_b_epilogue() const { return runtime_lora_b_epilogue_; }
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
    // branches without materializing the full gate/up activation. Runtime
    // LoRA projections intersect both the selected input columns and output
    // rows (including separately trained fused gate/up row ranges).
    Tensor project_slice(const Tensor &, const std::string &, int row_start,
                         int row_end, int col_start, int col_end,
                         bool add_bias = true) const;
    // Immutable checkpoint contribution only, for partial down reductions.
    // Down-LoRA must be applied once to joined hidden, not rounded per slice.
    Tensor project_base_slice(const Tensor &, const std::string &, int row_start,
                              int row_end, int col_start, int col_end,
                              bool add_bias = true) const;
    // Explicit base-only F32 partial, never a widened checkpoint bank or
    // per-shard down-LoRA. Original operands/hidden dtype stays unchanged.
    Tensor project_base_slice_fp32(const Tensor &, const std::string &, int row_start,
                                  int row_end, int col_start, int col_end) const;
    // Runtime adapter contribution only; Core ML supplies the frozen base
    // gate/up projection. An optional output dtype allows the experimental
    // Z-Image bridge to avoid BF16 rounding before its FP16 Core ML input.
    Tensor lora_delta_slice(const Tensor &, const std::string &, int row_start,
                            int row_end, int col_start, int col_end,
                            std::optional<mx::Dtype> output_dtype = std::nullopt) const;
    // Request/operation-local ranks over the SAME bound adapter generation.
    // Do not retain across adapter rebinds or input/column-range changes.
    std::vector<Tensor> lora_input_ranks(const Tensor &,const std::string &,int col_start,int col_end) const;
    size_t lora_rank_count(const std::string &prefix) const {
        auto found=runtime_loras_.find(prefix);return found==runtime_loras_.end()?0:found->second.size();
    }
    size_t lora_rank_width(const std::string &prefix) const {
        size_t width=0;auto found=runtime_loras_.find(prefix);
        if(found!=runtime_loras_.end())for(const auto &adapter:found->second)width+=adapter.down.shape(0);
        return width;
    }
    Tensor project_slice_with_ranks(const Tensor &,const std::string &,const std::vector<Tensor> &,
        int row_start,int row_end,int col_start,int col_end,bool add_bias=false) const;
    Tensor lora_delta_slice_with_ranks(const Tensor &,const std::string &,const std::vector<Tensor> &,
        int row_start,int row_end,int col_start,int col_end,std::optional<mx::Dtype> output_dtype=std::nullopt) const;
    // Metadata-only logical input: no concatenated full-hidden allocation.
    // Same operation/adapter generation and full column interval as the ranks.
    Tensor lora_delta_from_ranks(const mx::Shape &,mx::Dtype,const std::string &,const std::vector<Tensor> &,
        int row_start,int row_end,int col_start,int col_end,std::optional<mx::Dtype> output_dtype=std::nullopt) const;
    Tensor project_range(const Tensor &, const std::string &, int row_start, int row_end,
                        int col_start, int col_end) const;
    Tensor project_range_fp32(const Tensor &, const std::string &, int row_start, int row_end,
                             int col_start, int col_end) const;
    void clear();
    size_t bytes() const;
    void materialize();
    size_t apply_loras(const std::vector<LoRAAsset> &, const std::string &, const Event &,
                      std::atomic<bool> &, bool inference_time = false);
    // Content-verified, held-fd adapter inputs. Logical ids match adapters in
    // order; readers never reopen a mutable path. Materialize bound low-rank
    // sources and revalidate the lease before publishing a successful bind.
    size_t apply_loras_leased(const std::vector<LoRAAsset> &,const std::string &,const Event &,
        std::atomic<bool> &,bool,const std::shared_ptr<const streaming::SourceLease> &,
        const std::vector<std::string> &);
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
