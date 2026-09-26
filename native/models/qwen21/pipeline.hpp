#pragma once
#include "../../runtime/session.hpp"
#include "../../backends/mlx.hpp"
#include "hybrid.hpp"

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
    Weights transformer_, vae_;
    std::optional<Tensor> cached_text_;
    std::string cached_prompt_;
    std::unique_ptr<HybridSession> hybrid_;
    std::unique_ptr<HybridMLP> hybrid_mlp_;
    std::string hybrid_manifest_;
    RunResult run(const Request &, const Event &, std::atomic<bool> &, bool warmup, bool prepare_only);
};
} // namespace tc::qwen21
