#pragma once
#include "mlx.hpp"
#include "../runtime/session.hpp"
namespace tc {
// Public Core ML only. The framework may execute portions on CPU.
class HybridSession {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    std::string manifest;
    int rows = 0;
    double load_seconds = 0;
    HybridSession(const std::filesystem::path &, const std::filesystem::path &model, int tokens,
                  const Event &, std::atomic<bool> &, int warmups = 0, int policy_rows = 0);
    ~HybridSession();
    Tensor predict(int block, const Tensor &input);
    HybridMetrics metrics() const;
};
} // namespace tc
