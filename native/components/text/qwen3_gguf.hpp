#pragma once

#include "qwen3.hpp"
#include "../../runtime/streaming/gguf_weight_pager.hpp"
#include "../../runtime/session.hpp"

namespace tc::components {
struct Qwen3GgufConfig {
    uint32_t hidden = 2560, intermediate = 9728, vocabulary = 151936, layers = 36;
    uint32_t heads = 32, kv_heads = 8, head_dim = 128, context_length = 40960;
    float rope_theta = 1000000.f, epsilon = 1e-6f;
};
Qwen3GgufConfig read_qwen3_gguf_config(int fd, uint64_t bytes);
void verify_qwen3_gguf_tokenizer(int fd, uint64_t bytes, int gguf_fd,
                                const gguf::Directory &, const Qwen3GgufConfig &);

struct Qwen3GgufPlan {
    streaming::Descriptor descriptor;
    streaming::Layout layout;
    StreamingConfig config;
    std::map<std::string, std::string> names;
    uint64_t packed_capacity = 0, gather_capacity = 0, read_capacity = 0;
};
Qwen3GgufPlan describe_qwen3_gguf(std::shared_ptr<const streaming::SourceLease>,
                                const Qwen3GgufConfig &, const Tokens &, uint32_t prefetch, gguf::DecodeOptions = {},
                                const std::string &source_residency = "packed_resident");

// Reusable verified CPU metadata only: held source/config/tokenizer fds,
// configuration and tokenizer. No Tensor, weight bank, execution graph,
// cancellation reference or fill worker is retained across requests.
class Qwen3GgufPreparedSource {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit Qwen3GgufPreparedSource(std::unique_ptr<Impl>);
  public:
    static std::shared_ptr<const Qwen3GgufPreparedSource> prepare(
        const std::filesystem::path &gguf,const std::filesystem::path &config,
        const std::filesystem::path &tokenizer_json,std::atomic<bool> &);
    ~Qwen3GgufPreparedSource();
    bool reusable_for(const std::filesystem::path &gguf,const std::filesystem::path &config,
                      const std::filesystem::path &tokenizer_json) const;
    void revalidate() const;
    Tokens tokenize(const std::string &,bool dynamic) const;
    std::shared_ptr<const streaming::SourceLease> lease() const;
    const Qwen3GgufConfig &config() const;
    std::string execution_identity(uint32_t prefetch,uint64_t managed_budget,gguf::DecodeOptions,
                                  const std::string &residency) const;
};

// Experimental source-bound Qwen3-4B consumer; no Core ML or LoRA route.
// StageExecutor owns fill/reader/cancel/drain; all parameter math is shared
// with the resident Qwen3 consumer, with an explicit each-layer eval boundary.
class Qwen3GgufEncoder {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    Qwen3GgufEncoder(const std::filesystem::path &gguf, const std::filesystem::path &config,
                    const std::filesystem::path &tokenizer_json, uint32_t prefetch,
                    uint64_t managed_budget, const Event &, std::atomic<bool> &, gguf::DecodeOptions = {},
                    const std::string &source_residency = "packed_resident");
    Qwen3GgufEncoder(std::shared_ptr<const Qwen3GgufPreparedSource>,uint32_t prefetch,
                    uint64_t managed_budget,const Event &,std::atomic<bool> &,gguf::DecodeOptions = {},
                    const std::string &source_residency = "packed_resident");
    ~Qwen3GgufEncoder();
    Tokens tokenize(const std::string &, bool dynamic);
    Tensor encode(const Tokens &);
    bool drain_safely() noexcept;
    std::string identity() const;
    QuantizedExecutionMetrics metrics() const;
};
} // namespace tc::components
