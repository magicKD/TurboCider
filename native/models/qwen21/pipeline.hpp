#pragma once
#include "../../runtime/session.hpp"
#include "../../backends/mlx.hpp"
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
    std::optional<Tensor> cached_text_;
    std::string cached_prompt_;
    std::optional<CachedEditCondition> cached_edit_;
    std::unique_ptr<HybridSession> hybrid_;
    std::unique_ptr<HybridMLP> hybrid_mlp_;
    std::string hybrid_manifest_;
    std::string hybrid_runtime_options_;
    std::string active_lora_identity_;
    size_t lora_applied_projections_ = 0;
    RunResult run(const Request &, const Event &, std::atomic<bool> &, bool warmup, bool prepare_only);
};
} // namespace tc::qwen21
