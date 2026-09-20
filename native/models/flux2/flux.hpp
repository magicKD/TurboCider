#pragma once
#include "../../runtime/session.hpp"
#include "../../core/tokenizer.hpp"
#include "../../backends/mlx.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include <unordered_map>
namespace tc {
class HybridSession;
class FluxExactStream;
class Flux : public ModelSession {
    std::filesystem::path root_;
    std::string model_id_;
    int hidden_ = 0, heads_ = 0, dual_layers_ = 0, single_layers_ = 0;
    mutable Tokenizer tokenizer_;
    std::unique_ptr<HybridSession> hybrid_;
    std::unique_ptr<HybridSession> encoder_hybrid_;
    std::unique_ptr<FluxExactStream> exact_stream_;
    uint64_t exact_stream_generation_ = 0;
    bool streaming_quarantined_ = false;
    mutable bool streaming_content_identity_ = false;
    std::vector<streaming::SourceFileIdentity> streaming_source_files() const;
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    bool test_fail_drain_ = false;
#endif
    Weights transformer_, vae_;
    std::string cached_prompt_;
    std::string cached_encoder_manifest_;
    bool cached_dynamic_ = true;
    std::optional<Tensor> cached_conditioning_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)> hybrid_gpu_graph_;
    int hybrid_gpu_mlp_start_ = -1;
    std::vector<LoRAAsset> active_loras_;
    std::string cached_lora_identity_;
    struct LoRAFileHash {
        std::uintmax_t bytes = 0;
        std::filesystem::file_time_type mtime{};
        std::string sha256;
    };
    std::unordered_map<std::string, LoRAFileHash> lora_hash_cache_;
    std::shared_ptr<const streaming::SourceLease> public_stream_lease_;
    std::unique_ptr<Tokenizer> public_stream_tokenizer_;
    bool public_component_cache_ = false;
    uint64_t public_stream_target_bytes_ = 0;
    void reset_public_component_cache();

  public:
    std::string select_acceleration(Request &, int, const Event &, std::atomic<bool> &);
    void select_loras(const Request &);
    bool conditioning(const Request &, const Tokens &, const Event &, std::atomic<bool> &);
    RunResult run(const Request &, const Event &, std::atomic<bool> &, bool);
    RunResult prepare(const Request &, bool, const Event &, std::atomic<bool> &) override;
    explicit Flux(const std::filesystem::path &, std::string model_id);
    ~Flux();
    bool streaming_quarantined() const noexcept override { return streaming_quarantined_; }
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    void test_set_streaming_drain_failure(bool value) override { test_fail_drain_ = value; }
#endif
    LoadResult load(const Event &, std::atomic<bool> &) override;
    void unload() override;
    Tensor encode(const Tokens &, const Event &, std::atomic<bool> &, HybridSession * = nullptr);
    Tensor denoise(const Tensor &, const Tensor &, float, int, int, const Event &,
                   std::atomic<bool> &, const std::vector<float> &reference_ids = {},
                   bool compile_blocks = false,
                   FluxExactStream *exact_stream = nullptr,
                   uint32_t stream_pass = 0);
    Tensor encode_image(const Tensor &, const Event &, std::atomic<bool> &);
    Tensor decode(const Tensor &, int, int, const Event &, std::atomic<bool> &,
                  const std::string &);
    RunResult generate(const Request &, const Event &, std::atomic<bool> &) override;
    std::shared_ptr<const streaming::SourceLease>
    verify_streaming_sources(std::atomic<bool> &) override;
    std::shared_ptr<const streaming::ModelStreamingProbe>
    probe_public_streaming(
        const streaming::PublicResolveInput &) const override;
    std::shared_ptr<const streaming::ModelStreamingSnapshot>
    compile_public_streaming(
        std::shared_ptr<const streaming::ModelStreamingProbe>,
        const streaming::StreamingPresetRecord &) const override;
    RunResult generate_resolved(
        std::shared_ptr<const streaming::ResolvedRequestExecution>,
        const Event &, std::atomic<bool> &) override;
};
} // namespace tc
