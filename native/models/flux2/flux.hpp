#pragma once
#include "../../runtime/session.hpp"
#include "../../core/tokenizer.hpp"
#include "../../backends/mlx.hpp"
namespace tc {
class HybridSession;
class Flux : public ModelSession {
    std::filesystem::path root_;
    Tokenizer tokenizer_;
    std::unique_ptr<HybridSession> hybrid_;
    Weights transformer_, vae_;
    std::string cached_prompt_;
    bool cached_dynamic_ = true;
    std::optional<Tensor> cached_conditioning_;

  public:
    std::string select_acceleration(Request &, int, const Event &, std::atomic<bool> &);
    bool conditioning(const Request &, const Tokens &, const Event &, std::atomic<bool> &);
    RunResult run(const Request &, const Event &, std::atomic<bool> &, bool);
    RunResult prepare(const Request &, bool, const Event &, std::atomic<bool> &) override;
    explicit Flux(const std::filesystem::path &);
    ~Flux();
    LoadResult load(const Event &, std::atomic<bool> &) override;
    void unload() override;
    Tensor encode(const Tokens &, const Event &, std::atomic<bool> &);
    Tensor denoise(const Tensor &, const Tensor &, float, int, int, const Event &,
                   std::atomic<bool> &, const std::vector<float> &reference_ids = {},
                   bool compile_blocks = false);
    Tensor encode_image(const Tensor &, const Event &, std::atomic<bool> &);
    Tensor decode(const Tensor &, int, int, const Event &, std::atomic<bool> &,
                  const std::string &);
    RunResult generate(const Request &, const Event &, std::atomic<bool> &) override;
};
} // namespace tc
