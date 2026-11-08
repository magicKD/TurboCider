#pragma once
#include "../../backends/mlx.hpp"
#include "cider_receipt.hpp"
#include <filesystem>
#include <functional>
#include <memory>

namespace tc::qwen21 {
inline constexpr const char *cider_recipe = "dit-pc-w8a8-absmax127-nk128-m32-v1";
// One request owns its counters and cancellation callback; tensor banks remain
// with the streamed transformer/FFN owners and are released before VAE decode.
class CiderCache {
  public:
    CiderCache(const std::filesystem::path &root, const std::string &verified_sha256,
               std::function<void()> checkpoint, bool smoothquant_convrot = false);
    std::vector<Tensor> pack(const Tensor &weight, const std::string &key);
    const CiderCacheReceipt &receipt() const { return receipt_; }
  private:
    CiderCacheReceipt receipt_;
    std::function<void()> checkpoint_;
    bool smoothquant_convrot_ = false;
};
Tensor cider_project(const Tensor &input, const std::vector<Tensor> &args, size_t offset);
}
