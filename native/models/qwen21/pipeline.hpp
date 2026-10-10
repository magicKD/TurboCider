#pragma once
#include "../../runtime/session.hpp"
#include "../../backends/mlx.hpp"
#include "hybrid.hpp"
#include "transformer.hpp"
#include "conditioning_cache.hpp"
#include "encoder_residency.hpp"
#include "gguf_weights.hpp"
#include "../../backends/ane_ffn.hpp"
#include "../../backends/ane_qkv.hpp"

namespace tc::qwen21 {
class Session final : public ModelSession {
  public:
    explicit Session(const std::filesystem::path &);
    LoadResult load(const Event &, std::atomic<bool> &) override;
    void unload() override;
    RunResult prepare(const Request &, bool, const Event &, std::atomic<bool> &) override;
    RunResult generate(const Request &, const Event &, std::atomic<bool> &) override;
  private:
    std::filesystem::path root_;
    std::filesystem::path transformer_source_,encoder_source_;
    Weights transformer_, vae_;
    std::unique_ptr<GgufComponent> transformer_gguf_,encoder_gguf_;
    // Diagnostic shallow views of QKV matrices replacing their three source
    // weights in transformer_. Never retained on a route without opt-in.
    std::vector<Tensor> fused_qkv_weights_;
    ConditioningCache conditioning_cache_;
    std::unique_ptr<Weights> encoder_weights_;
    std::string encoder_weight_identity_;
    uint64_t encoder_weight_loads_ = 0;
    // Optional one-entry executor only; never retains the complete encoder
    // weights. Drain before request-owned text sources die or identity changes.
    std::unique_ptr<ane::HybridFfn> encoder_runtime_;
    std::string encoder_runtime_identity_;
    // Explicit resident experiment; owns one prefix KV bank at most. The
    // transformer must be destroyed before its referenced weights are cleared.
    std::unique_ptr<Transformer> cached_prefix_transformer_;
    std::string cached_prefix_runtime_;
    float cached_prefix_sigma_ = -1.f;
    std::unique_ptr<HybridSession> hybrid_;
    std::unique_ptr<HybridMLP> hybrid_mlp_;
    std::unique_ptr<ane::HybridFfn> runtime_ffn_;
    std::unique_ptr<ane::HybridQkv> runtime_qkv_;
    std::string runtime_manifest_;
    std::string qkv_manifest_;
    std::string hybrid_manifest_;
    std::string hybrid_runtime_options_;
    std::string active_lora_identity_;
    size_t lora_applied_projections_ = 0;
    // Destroy the Transformer before invalidating its weights or callback
    // owners, and discard the identity used to admit cross-request KV reuse.
    void clear_prefix_cache();
    void prepare_transformer(const Request &, const Event &, std::atomic<bool> &,
                             bool experimental_adapter, bool fused_qkv, bool lora_fp16);
    RunResult run(const Request &, const Event &, std::atomic<bool> &, bool warmup, bool prepare_only);
    RunResult run_bf16_streamed(const Request &,const Event &,std::atomic<bool> &,bool warmup,bool prepare_only);
};
} // namespace tc::qwen21
