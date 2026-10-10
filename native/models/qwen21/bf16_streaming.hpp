#pragma once
#include "../../backends/mlx.hpp"
#include "../../runtime/streaming/mlx_weight_pager.hpp"
#include "../../runtime/streaming/slot_pool.hpp"
#include "../../runtime/streaming/io_executor.hpp"
#include "../../runtime/streaming/qwen_bf16_metrics.hpp"

namespace tc::qwen21 {
bool bf16_streaming_enabled();
void bf16_stream_scope(const std::function<void()> &);
struct Bf16StreamPlan {
    std::shared_ptr<const streaming::SourceLease> lease;
    streaming::Descriptor descriptor;
    streaming::Layout layout;
    uint32_t layers=0,prefix=0;
    double verification_seconds=0;
};
// Parses the held-fd safetensors header, validates the complete directory and
// describes only consumed language/fixed/layer fields. No payload allocation.
Bf16StreamPlan bf16_stream_plan(const std::filesystem::path &,bool encoder,
    uint32_t layers,uint32_t prefix,uint32_t passes,std::atomic<bool> &cancel);

// Private diagnostic adapter, not a public catalog authority or hard RAM cap.
// CPU refill jobs never allocate/evaluate MLX arrays. Owner graph arguments are
// dynamic, and every submitted GPU last reader completes before slot retirement.
class Bf16StreamBank final {
    Bf16StreamPlan plan_;
    streaming::MlxWeightPager pager_;
    Weights resident_,current_;
    std::atomic<bool> &cancel_;
    streaming::CompletionMailbox mailbox_{8};
    std::unique_ptr<streaming::SlotSafetyTracker> safety_;
    std::unique_ptr<streaming::IoExecutor> io_;
    struct Job { Bf16StreamBank *bank=nullptr;const streaming::Group *group=nullptr; };
    std::array<Job,2> jobs_{};
    std::array<std::optional<tc_stream_slot_ticket_v1>,2> pending_{};
    std::optional<tc_stream_slot_ticket_v1> current_ticket_;
    uint32_t pass_=0,next_layer_=0;
    bool pass_live_=false,finished_=false;
    QwenBf16StreamStageMetrics metrics_;
    void schedule(uint32_t);
    void consume();
  public:
    Bf16StreamBank(Bf16StreamPlan,uint64_t managed_weight_budget,std::atomic<bool> &);
    ~Bf16StreamBank();
    Bf16StreamBank(const Bf16StreamBank &)=delete;
    Bf16StreamBank &operator=(const Bf16StreamBank &)=delete;
    const Weights &resident() const {return resident_;}
    void begin_pass(uint32_t);
    const Weights &acquire(int layer);
    // Caller must have synchronously evaluated ALL outputs/prefix readers.
    void retire(int layer);
    void finish_pass();
    void finish();
    QwenBf16StreamStageMetrics metrics() const;
};
}
