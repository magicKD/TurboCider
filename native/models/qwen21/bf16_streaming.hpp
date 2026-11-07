#pragma once
#include "../../backends/mlx.hpp"
#include "../../runtime/streaming/mlx_weight_pager.hpp"
#include "../../runtime/streaming/slot_pool.hpp"
#include "../../runtime/streaming/io_executor.hpp"
#include "../../runtime/streaming/qwen_bf16_metrics.hpp"
#include "../../core/tokenizer.hpp"

namespace tc::qwen21 {
struct StreamEmbeddingSource {
    streaming::FieldSpec codes, scales;
};
bool bf16_streaming_enabled();
void bf16_stream_scope(const std::function<void()> &);
enum class StreamWeightMode { Bf16, Bf16ConvRotCompanion, ConvRotFloat, ConvRotW8A8 };
struct Bf16StreamPlan {
    std::shared_ptr<const streaming::SourceLease> lease;
    streaming::Descriptor descriptor;
    streaming::Layout layout;
    uint32_t layers=0,prefix=0;
    uint32_t cache_budget_prefix=0;
    bool cache_limited_by_memory=false;
    std::shared_ptr<const Bf16StreamPlan> minimum_plan;
    double verification_seconds=0;
    StreamWeightMode mode = StreamWeightMode::Bf16;
    std::optional<StreamEmbeddingSource> embedding;
};
// Parses the held-fd safetensors header, validates the complete directory and
// describes only consumed language/fixed/layer fields. No payload allocation.
Bf16StreamPlan bf16_stream_plan(const std::filesystem::path &,bool encoder,
    uint32_t layers,uint32_t prefix,uint32_t passes,std::atomic<bool> &cancel,
    StreamWeightMode mode = StreamWeightMode::Bf16);
uint64_t bf16_stream_managed_capacity(const Bf16StreamPlan &);
// Retains an exact raw DiT prefix within both the requested weight budget and
// the caller's physical-headroom policy. Encoder layers are consumed once.
Bf16StreamPlan bf16_stream_cache_plan(Bf16StreamPlan, uint64_t budget,
    const std::function<bool(uint64_t)> &admit = {});

// Private diagnostic adapter, not a public catalog authority or hard RAM cap.
// CPU refill jobs never allocate/evaluate MLX arrays. Owner graph arguments are
// dynamic, and every submitted GPU last reader completes before slot retirement.
class Bf16StreamBank final {
    Bf16StreamPlan plan_;
    streaming::MlxWeightPager pager_;
    Weights resident_,current_;
    std::optional<Bf16StreamPlan> cache_plan_;
    std::unique_ptr<streaming::MlxWeightPager> cache_pager_;
    Weights cached_;
    streaming::MlxWeightPagerMetrics retired_cache_metrics_;
    struct FloatProjection {
        std::string name;
        Tensor codes, scales, biases;
    };
    std::vector<FloatProjection> float_projections_;
    std::atomic<bool> &cancel_;
    streaming::CompletionMailbox mailbox_{8};
    std::unique_ptr<streaming::SlotSafetyTracker> safety_;
    std::unique_ptr<streaming::IoExecutor> io_;
    struct Job { Bf16StreamBank *bank=nullptr;const streaming::Group *group=nullptr; };
    std::array<Job,2> jobs_{};
    std::array<std::optional<tc_stream_slot_ticket_v1>,2> pending_{};
    std::optional<tc_stream_slot_ticket_v1> current_ticket_;
    uint32_t pass_=0,next_layer_=0;
    bool pass_live_=false,layer_live_=false,finished_=false;
    QwenBf16StreamStageMetrics metrics_;
    void schedule(uint32_t);
    void consume();
    void bind_float_layer(uint32_t layer);
    const streaming::StageLayout &execution_layout() const;
  public:
    Bf16StreamBank(Bf16StreamPlan,uint64_t managed_weight_budget,std::atomic<bool> &);
    ~Bf16StreamBank();
    Bf16StreamBank(const Bf16StreamBank &)=delete;
    Bf16StreamBank &operator=(const Bf16StreamBank &)=delete;
    const Weights &resident() const {return resident_;}
    // Gather only requested vocabulary rows from the authenticated held fd.
    // The returned BF16 embeddings own their storage before any layer refill.
    Tensor embedding_rows(const Tokens &);
    int embedding_vocabulary() const;
    void begin_pass(uint32_t);
    const Weights &acquire(int layer);
    // Caller must have synchronously evaluated ALL outputs/prefix readers.
    void retire(int layer);
    void finish_pass();
    // At a drained pass boundary, free optional cached raw layers. The
    // original fixed weights and two refill buffers remain allocated.
    bool evict_cached_layers();
    void finish();
    QwenBf16StreamStageMetrics metrics() const;
};
}
