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
    int rows = 0, mlp_width = 0, ane_mlp_start = 0, ane_mlp_end = 0;
    bool checkpoint_sha_verified = false;
    double load_seconds = 0;
    HybridSession(const std::filesystem::path &, const std::filesystem::path &model, int tokens,
                  const Event &, std::atomic<bool> &, int warmups = 0);
    ~HybridSession();
    Tensor predict(int block, const Tensor &input);
    HybridMetrics metrics() const;
};
} // namespace tc
