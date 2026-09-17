#pragma once

#include "../../runtime/session.hpp"
#include "../../runtime/streaming/context.hpp"
#include "../../runtime/streaming/mlx_weight_pager.hpp"
#include "streaming_descriptor.hpp"

#include <memory>

namespace tc {

class FluxExactStream {
  public:
    FluxExactStream(const std::filesystem::path &transformer_directory,
                    const std::string &model_id,
                    const StreamingConfig &config,
                    const flux2::StreamingWorkload &workload,
                    Weights &resident, const Event &event,
                    std::atomic<bool> &cancelled,
                    uint64_t request_generation);
    ~FluxExactStream();

    FluxExactStream(const FluxExactStream &) = delete;
    FluxExactStream &operator=(const FluxExactStream &) = delete;

    void run_pass(uint32_t pass, uint32_t step, Tensor &image,
                  Tensor &context, const std::vector<Tensor> &image_modulation,
                  const std::vector<Tensor> &text_modulation,
                  const std::vector<Tensor> &single_modulation,
                  const Tensor &cosine, const Tensor &sine,
                  int text_tokens, int total_tokens);
    void finish();

    const flux2::StreamingPlanView &plan() const;
    const streaming::MlxWeightPagerMetrics &pager_metrics() const;
    streaming::ExecutionCounters counters() const;
    const char *implementation() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tc
