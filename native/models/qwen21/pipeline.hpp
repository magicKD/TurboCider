#pragma once
#include "../../runtime/session.hpp"
#include "../../backends/mlx.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include "hybrid.hpp"
#include "transformer.hpp"
#include "conditioning.hpp"
#include "../../backends/ane_ffn.hpp"
#include "../../backends/ane_qkv.hpp"

namespace tc::qwen21 {
class NativeSession;
class EncoderRuntimeCache;
class NativeEncoder;
class NativePrefill;
class Session final : public ModelSession {
  public:
    explicit Session(const std::filesystem::path &);
    ~Session() override;
    bool streaming_quarantined() const noexcept override;
    void bind_memory_context(MemoryExecutionContext *) override;
    void unbind_memory_context() noexcept override;
    MemoryDrainResult drain_memory_completions(
        MemoryExecutionContext &, std::chrono::milliseconds) override;
    LoadResult load(const Event &, std::atomic<bool> &) override;
    void unload() override;
    RunResult prepare(const Request &, bool, const Event &, std::atomic<bool> &) override;
    RunResult generate(const Request &, const Event &, std::atomic<bool> &) override;
    // Idle same-process diagnostic only; out-of-line symbol rejects stale
    // libraries at probe link time rather than accepting a mismatched ABI.
    void set_diagnostic_block_observer(DiagnosticBlockObserver);
    using DiagnosticPrefillObserver=std::function<void(int,const Tensor&,const Weights&)>;
    // Borrowed real uncached FFN input; observer must finish/drain its bounded
    // component and throw DiagnosticBlockReplayComplete before image decoding.
    void set_diagnostic_prefill_observer(DiagnosticPrefillObserver);
    // Explicit model-selection/idle prewarm. Cost is returned and retained in
    // request metadata; it must not be described as free cold-start speedup.
    double prewarm_native_prefill(const std::filesystem::path&);
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
    // Explicit native DiT request owner; retained on an unresolved physical
    // drain, including the source descriptors and original MLX Data owners.
    std::unique_ptr<NativeSession> native_session_;
    std::unique_ptr<EncoderRuntimeCache> encoder_runtime_cache_;
    std::unique_ptr<NativeEncoder> native_encoder_;
    std::unique_ptr<NativePrefill> native_prefill_;
    MemoryExecutionContext *memory_context_ = nullptr;
    bool native_quarantined_ = false;
    DiagnosticBlockObserver diagnostic_block_observer_;
    DiagnosticPrefillObserver diagnostic_prefill_observer_;
    Weights transformer_, vae_;
    // Diagnostic shallow views of QKV matrices replacing their three source
    // weights in transformer_. Never retained on a route without opt-in.
    std::vector<Tensor> fused_qkv_weights_;
    std::optional<Tensor> cached_text_;
    bool cached_text_source_leased_ = false;
    std::string cached_prompt_;
    std::optional<CachedEditCondition> cached_edit_;
    struct CachedReference {
        std::string sha256;
        int resolution;
        VisualReference visual;
        Tensor latent;
        int height, width;
    };
    // One bounded group, evaluated values only; never retains encoder weights.
    // Slots belong to the current prompt and are deliberately not cached here.
    std::vector<CachedReference> cached_references_;
    std::optional<streaming::SourceLeaseDescriptor> cached_encoder_identity_;
    // Explicit resident experiment; owns one prefix KV bank at most. The
    // transformer must be destroyed before its referenced weights are cleared.
    std::unique_ptr<Transformer> cached_prefix_transformer_;
    std::string cached_prefix_runtime_;
    float cached_prefix_sigma_ = -1.f;
    // Compact, evaluated K/V only: unlike a Transformer this does not capture
    // DiT weights and may outlive their release in component-staged sessions.
    std::optional<Transformer::PrefixSnapshot> cached_prefix_snapshot_;
    std::string cached_snapshot_runtime_;
    std::unique_ptr<HybridSession> hybrid_;
    std::optional<streaming::SourceFileIdentity> hybrid_source_identity_;
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
    void clear_prefix_snapshot();
    RunResult run(const Request &, const Event &, std::atomic<bool> &, bool warmup, bool prepare_only);
};
} // namespace tc::qwen21
