#pragma once

#include "../../backends/mlx.hpp"
#include "../../core/tokenizer.hpp"
#include "../../runtime/session.hpp"
#include "../../backends/coreml.hpp"
#include "../../backends/ane_ffn.hpp"
#include "weight_stream.hpp"
#include "../../runtime/device_optimizations.hpp"

namespace tc {

namespace components { class Qwen3GgufEncoder; }

class ZImageExactStream;
class ZImageGgufStream;
namespace streaming { class GgufPackedBank; }
using ZImageBlockObserver=std::function<void(const std::string &,const Tensor &)>;

class ZImage final : public ModelSession {
    std::filesystem::path root_;
    std::filesystem::path text_path_, transformer_path_, transformer_checkpoint_, vae_path_;
    std::string model_id_ = "z-image-turbo";
    bool diffusers_layout_ = false, gguf_transformer_ = false, convrot_transformer_ = false;
    bool nvfp4_transformer_ = false;
    bool runtime_convrot_ = false;
    bool gguf_direct_import_ = false;
    bool gguf_compile_packed_ = false;
    bool gguf_gpu_f16_ = false;
    bool gguf_gpu_f16_mpp_ = false;
    bool gguf_qmm_f16_ = false;
    bool gguf_f16_refiners_ = false;
    bool gguf_ref_mpp_dynamic_ = false;
    uint64_t gguf_allocator_cache_bytes_ = 0;
    bool gguf_validate_blocks_ = false;
    bool gguf_retain_packed_ = false;
    std::unique_ptr<MemoryLedger> gguf_packed_ledger_;
    std::unique_ptr<streaming::GgufPackedBank> gguf_packed_bank_;
    DeviceOptimizations optimizations_;
    mutable Tokenizer tokenizer_;
    Weights text_encoder_;
    std::unique_ptr<components::Qwen3GgufEncoder> encoder_gguf_;
    std::string cached_encoder_gguf_identity_;
    std::optional<QuantizedExecutionMetrics> cached_encoder_gguf_metrics_;
    Weights transformer_;
    Weights vae_;
    std::unique_ptr<ZImageWeightStream> weight_stream_;
    std::unique_ptr<ZImageExactStream> exact_stream_;
    std::unique_ptr<ZImageGgufStream> gguf_stream_;
    std::string stream_configuration_;
    uint64_t exact_stream_generation_ = 0;
    bool streaming_quarantined_ = false;
    mutable bool streaming_content_identity_ = false;
    std::vector<streaming::SourceFileIdentity> streaming_source_files() const;
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    bool test_fail_drain_ = false;
#endif
    std::optional<Tensor> cached_conditioning_;
    std::optional<HybridMetrics> cached_encoder_hybrid_metrics_;
    std::string cached_prompt_;
    std::string cached_encoder_manifest_;
    bool cached_dynamic_ = true;
    std::vector<LoRAAsset> active_loras_;
    std::string cached_lora_identity_;
    std::string active_lora_strategy_ = "none";
    size_t lora_applied_projections_ = 0;
    std::unique_ptr<HybridSession> hybrid_;
    std::unique_ptr<ane::HybridFfn> runtime_ffn_;
    std::string runtime_manifest_;
    std::unique_ptr<HybridSession> encoder_hybrid_;
    std::function<std::vector<Tensor>(const std::vector<Tensor> &)> hybrid_gpu_graph_;
    int hybrid_gpu_mlp_start_ = -1;
    // Bound only for one public exact generate call. Legacy/private paths keep
    // the empty value and retain their existing path-based construction.
    std::shared_ptr<const streaming::SourceLease> public_stream_lease_;
    std::unique_ptr<Tokenizer> public_stream_tokenizer_;
    bool public_component_cache_ = false;
    uint64_t public_stream_target_bytes_ = 0;
    int gpu_w8_suffix_start_ = -1;
    int gpu_w8_group_size_ = 0;
    std::string gpu_w8_manifest_;
    std::string gpu_bf16_route_manifest_;

    void select_loras(const Request &);
    void load_vae(const Event &, std::atomic<bool> &);
    Tensor encode_text(const Tokens &, const Event &, std::atomic<bool> &);
    Tensor denoise(const Tensor &, const Tensor &, float, float, int, int,
                   const Event &, std::atomic<bool> &, std::vector<Tensor> * = nullptr,
                   bool source_reference=false,const ZImageBlockObserver &observe={});
    Tensor decode(const Tensor &, int, int, const Event &, std::atomic<bool> &);
    bool conditioning(const Request &, const Event &, std::atomic<bool> &);
    std::string select_acceleration(Request &, int, const Event &, std::atomic<bool> &);
    RunResult run(const Request &, const Event &, std::atomic<bool> &, bool warmup,
                  bool load_only = false);

  public:
    explicit ZImage(const std::filesystem::path &);
    ZImage(const std::filesystem::path &, std::string,
           const std::filesystem::path &transformer_checkpoint);
    ~ZImage() override;
    bool streaming_quarantined() const noexcept override { return streaming_quarantined_; }
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    void test_set_streaming_drain_failure(bool value) override { test_fail_drain_ = value; }
#endif
    LoadResult load(const Event &, std::atomic<bool> &) override;
    void unload() override;
    RunResult prepare(const Request &, bool, const Event &, std::atomic<bool> &) override;
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
