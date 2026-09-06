#pragma once

#include "../../backends/mlx.hpp"
#include "../../core/tokenizer.hpp"
#include "../../runtime/session.hpp"
#include "../../backends/coreml.hpp"

namespace tc {

class ZImage final : public ModelSession {
    std::filesystem::path root_;
    std::filesystem::path text_path_, transformer_path_, transformer_checkpoint_, vae_path_;
    bool diffusers_layout_ = false;
    Tokenizer tokenizer_;
    Weights text_encoder_;
    Weights transformer_;
    Weights vae_;
    std::optional<Tensor> cached_conditioning_;
    std::string cached_prompt_;
    bool cached_dynamic_ = true;
    std::vector<LoRAAsset> active_loras_;
    std::string cached_lora_identity_;
    size_t lora_applied_projections_ = 0;
    std::unique_ptr<HybridSession> hybrid_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)> hybrid_gpu_graph_;
    int hybrid_gpu_mlp_start_ = -1;

    void select_loras(const Request &);
    Tensor encode_text(const Tokens &, const Event &, std::atomic<bool> &);
    Tensor denoise(const Tensor &, const Tensor &, float, float, int, int,
                   const Event &, std::atomic<bool> &);
    Tensor decode(const Tensor &, int, int, const Event &, std::atomic<bool> &);
    bool conditioning(const Request &, const Event &, std::atomic<bool> &);
    std::string select_acceleration(Request &, int, const Event &, std::atomic<bool> &);
    RunResult run(const Request &, const Event &, std::atomic<bool> &, bool warmup);

  public:
    explicit ZImage(const std::filesystem::path &);
    ~ZImage() override = default;
    LoadResult load(const Event &, std::atomic<bool> &) override;
    void unload() override;
    RunResult prepare(const Request &, bool, const Event &, std::atomic<bool> &) override;
    RunResult generate(const Request &, const Event &, std::atomic<bool> &) override;
};

} // namespace tc
