#pragma once
#include "hybrid_layout.hpp"
#include "../../backends/coreml.hpp"
#include "../../runtime/streaming/context.hpp"
namespace tc {
struct ZHybridBranchCompletion {
    uint32_t step = 0, branch = 0, rows = 0;
    uint64_t coreml_sequence = 0;
};
// Internal single-owner-thread denoiser stage. This is not public admission or
// image quality authority. Cancellation signal is owned, not borrowed.
class ZImageHybridStream final {
  public:
    ZImageHybridStream(std::shared_ptr<const streaming::SourceLease>,
        std::shared_ptr<const z_image::VerifiedCoreMLBundleLease>,
        const StreamingConfig &, const z_image::StreamingWorkload &,
        uint64_t budget, uint64_t activation_reserve, Event,
        std::shared_ptr<std::atomic<bool>> cancelled, uint64_t request_generation);
    ~ZImageHybridStream();
    ZImageHybridStream(const ZImageHybridStream &) = delete;
    ZImageHybridStream &operator=(const ZImageHybridStream &) = delete;
    Tensor transform(const Tensor &latent, const Tensor &caption, float sigma, uint32_t step);
    std::vector<float> sigmas() const;
    void finish();
    bool drain_safely() noexcept;
    bool failed() const noexcept;
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
    void test_set_drain_failure(bool);
#endif
    const streaming::Layout &layout() const;
    streaming::ExecutionCounters counters() const;
    HybridMetrics hybrid_metrics() const;
    std::shared_ptr<const streaming::ActualStageReceipt> receipt() const;
    const std::vector<ZHybridBranchCompletion> &branches() const;
    // Internal z_transformer hooks, valid only during transform on owner thread.
    Tensor encode_noise(uint32_t branch, const Tensor &, const Tensor &freqs, const Tensor &temb);
    void run_main(uint32_t pass, Tensor &, const Tensor &freqs, const Tensor &temb);
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tc
