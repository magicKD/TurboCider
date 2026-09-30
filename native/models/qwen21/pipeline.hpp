#pragma once
#include "../../runtime/session.hpp"
#include "../../backends/mlx.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include "hybrid.hpp"
#include "transformer.hpp"

namespace tc::qwen21 {
class Session final : public ModelSession {
  public:
    explicit Session(const std::filesystem::path &);
    LoadResult load(const Event &, std::atomic<bool> &) override;
    void unload() override;
    RunResult prepare(const Request &, bool, const Event &, std::atomic<bool> &) override;
    RunResult generate(const Request &, const Event &, std::atomic<bool> &) override;
  private:
    struct CachedEditCondition {
        std::string prompt;
        int reference_size = 0;
        std::vector<std::string> image_sha256;
        Tensor text;
        std::vector<int> image_slots;
        std::vector<ReferenceLatents> reference_latents;
    };
    std::filesystem::path root_;
    Weights transformer_, vae_;
    // Diagnostic shallow views of QKV matrices replacing their three source
    // weights in transformer_. Never retained on a route without opt-in.
    std::vector<Tensor> fused_qkv_weights_;
    std::optional<Tensor> cached_text_;
    std::string cached_prompt_;
    std::optional<CachedEditCondition> cached_edit_;
    // Explicit resident experiment; owns one prefix KV bank at most. The
    // transformer must be destroyed before its referenced weights are cleared.
    std::unique_ptr<Transformer> cached_prefix_transformer_;
    std::string cached_prefix_runtime_;
    float cached_prefix_sigma_ = -1.f;
    std::unique_ptr<HybridSession> hybrid_;
    std::optional<streaming::SourceFileIdentity> hybrid_source_identity_;
    std::unique_ptr<HybridMLP> hybrid_mlp_;
    std::string hybrid_manifest_;
    std::string hybrid_runtime_options_;
    std::string active_lora_identity_;
    size_t lora_applied_projections_ = 0;
    // Destroy the Transformer before invalidating its weights or callback
    // owners, and discard the identity used to admit cross-request KV reuse.
    void clear_prefix_cache();
    RunResult run(const Request &, const Event &, std::atomic<bool> &, bool warmup, bool prepare_only);
};
} // namespace tc::qwen21
