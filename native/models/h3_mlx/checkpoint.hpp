#pragma once

#include "../../components/weights/affine.hpp"

#include <array>
#include <unordered_map>

namespace tc::h3_mlx {

struct DiTConfig {
    int hidden_size = 5376;
    int num_layers = 50;
    int refiner_layers = 2;
    int num_heads = 56;
    int head_dim = 128;
    int ffn_dim = 14336;
    int latent_channels = 24;
    int audio_latent_channels = 32;
    int text_dim = 5120;
    int frequency_dim = 256;
    int time_embed_dim = 2688;
    int rope_frequency_dim = 16;
    std::array<int, 3> patch_size{1, 2, 2};
    float rope_theta = 10000.f;
    float norm_epsilon = 1e-5f;
    float qk_norm_epsilon = 1e-5f;
    float final_norm_epsilon = 1e-5f;

    int patch_dim() const {
        return latent_channels * patch_size[0] * patch_size[1] * patch_size[2];
    }
};

struct QuantizationSpec {
    std::string mode = "affine";
    int bits = 6;
    int group_size = 64;
};

struct VDNCheckpointConfig {
    bool enabled = false;
    int version = 0;
    int softmax_chunk = 0;
    int softmax_radius = 0;
    int linear_head_dim = 0;
    bool anchor_rows = false;
    bool anchor_columns = false;
    bool a_fp32 = false;
    bool bridge_alpha = false;
    bool enable_text_state = false;
    bool enable_softmax_gate = false;
    bool conv_k = false;
    bool conv_v = false;
    std::string delta_rule;
    std::string branch_sha256;
    uint64_t branch_bytes = 0;
};

struct CheckpointIdentity {
    int format_version = 1;
    int steps = 4;
    int denoise_steps = 4;
    float video_shift = 12.f;
    float audio_shift = 3.f;
    bool vsa_capable = false;
    int vsa_gate_matrices = 0;
    std::string source_sha256;
    std::string source_repository;
    std::vector<float> adaln_timesteps;
    VDNCheckpointConfig vdn;
};

class Checkpoint {
    std::unordered_map<std::string, Tensor> arrays_;
    std::unordered_map<std::string, components::AffineMatrix> quantized_;
  std::unordered_map<std::string, Tensor> vdn_arrays_;
  DiTConfig config_;
  QuantizationSpec quantization_;
  CheckpointIdentity identity_;
  int affine_dq_gemm_min_rows_ = 768;
  bool experimental_fused_qkv_ = false;
  mutable uint64_t quantized_matmul_calls_ = 0;
  mutable uint64_t dequantized_gemm_calls_ = 0;

  public:
    void load(const std::filesystem::path &root, const Event &, std::atomic<bool> &);
    void clear();
    const DiTConfig &config() const { return config_; }
    const QuantizationSpec &quantization() const { return quantization_; }
    const CheckpointIdentity &identity() const { return identity_; }
    const Tensor &at(const std::string &) const;
    bool has(const std::string &) const;
    const Tensor &vdn_at(const std::string &) const;
    bool vdn_has(const std::string &) const;
    bool is_quantized(const std::string &prefix) const;
    mx::Dtype projection_dtype(const std::string &prefix) const;
    Tensor linear(const Tensor &, const std::string &prefix,
                  bool prefer_wide_dense = true) const;
    Tensor linear_fused(const Tensor &, const std::vector<std::string> &prefixes,
                        bool prefer_wide_dense = true) const;
    void set_affine_dq_gemm_min_rows(int);
    int affine_dq_gemm_min_rows() const { return affine_dq_gemm_min_rows_; }
    void set_experimental_fused_qkv(bool enabled) {
        experimental_fused_qkv_ = enabled;
    }
    bool experimental_fused_qkv() const { return experimental_fused_qkv_; }
    void reset_dispatch_metrics() const;
    uint64_t quantized_matmul_calls() const { return quantized_matmul_calls_; }
    uint64_t dequantized_gemm_calls() const { return dequantized_gemm_calls_; }
    size_t bytes() const;
};

} // namespace tc::h3_mlx
