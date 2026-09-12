#pragma once

#include "geometry.hpp"
#include "../../backends/mlx.hpp"
#include "../../core/tokenizer.hpp"

#include <unordered_map>

namespace tc::h3_mlx {

struct ConditionerConfig {
    int hidden_size = 5120;
    int num_layers = 64;
    int num_heads = 64;
    int kv_heads = 8;
    int head_dim = 128;
    int intermediate_size = 25600;
    int vocabulary_size = 151936;
    float norm_epsilon = 1e-6f;
    float rope_theta = 5000000.f;
    std::array<int, 3> mrope_sections{24, 20, 20};
};

ConditionerConfig load_conditioner_config(const std::filesystem::path &);

class ShardIndex {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit ShardIndex(const std::filesystem::path &);
    ~ShardIndex();
    ShardIndex(const ShardIndex &) = delete;
    ShardIndex &operator=(const ShardIndex &) = delete;
    Tensor tensor(const std::string &) const;
    Tensor rows(const std::string &, const std::vector<int> &) const;
    bool has(const std::string &) const;
};

struct ConditioningResult {
    Tensor hidden_states;
    std::vector<int32_t> token_tags;
    int tokens = 0;
};

using ConditionerDebugTensors = std::unordered_map<std::string, Tensor>;

class Conditioner {
    static constexpr int output_layers = 50;
    ConditionerConfig config_;
    ShardIndex weights_;
    Tokenizer tokenizer_;
    std::filesystem::path component_root_;
    std::filesystem::path tokenizer_root_;

    Tensor layer(int, const Tensor &, const Tensor &, const Tensor &,
                 ConditionerDebugTensors *) const;

  public:
    Conditioner(const std::filesystem::path &component_root,
                const std::filesystem::path &tokenizer_root);
    const ConditionerConfig &config() const { return config_; }
    ConditioningResult encode_prompt(const std::string &, const Event &,
                                      std::atomic<bool> &,
                                      std::vector<Tensor> *debug_layers = nullptr,
                                      ConditionerDebugTensors *debug_tensors = nullptr) const;
    ConditioningResult encode_tokens(const std::vector<int> &, const Event &,
                                      std::atomic<bool> &,
                                      std::vector<Tensor> *debug_layers = nullptr,
                                      ConditionerDebugTensors *debug_tensors = nullptr) const;
    ConditioningResult encode_prompt_cached(const std::string &,
                                             const std::filesystem::path &,
                                             const Event &,
                                             std::atomic<bool> &) const;
};

} // namespace tc::h3_mlx
