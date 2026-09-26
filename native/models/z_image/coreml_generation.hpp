#pragma once
#include "../../runtime/streaming/source_lease.hpp"

namespace tc::z_image {

// Internal import primitive, not hybrid execution authority. The tree must be
// self-contained with nonempty regular files (SourceLease requirement); cache
// management lock files are not model payload. Semantic manifest/parent/partition validation is a separate
// responsibility of the future VerifiedCoreMLBundleLease factory.
class CoreMLGeneration final {
  public:
    static std::shared_ptr<const CoreMLGeneration> import_tree(
        const std::filesystem::path &source,
        const std::filesystem::path &private_parent,
        const std::atomic<bool> *cancelled = nullptr);
    ~CoreMLGeneration();
    CoreMLGeneration(const CoreMLGeneration &) = delete;
    CoreMLGeneration &operator=(const CoreMLGeneration &) = delete;
    const std::filesystem::path &root() const noexcept;
    std::string_view content_digest() const noexcept;
    uint64_t copied_bytes() const noexcept;
    // Call around path-based Core ML loading and after drain. Checks file and
    // directory generations as well as the complete relative entry set.
    void revalidate() const;
  private:
    struct State;
    explicit CoreMLGeneration(std::unique_ptr<State>);
    std::unique_ptr<State> state_;
};
} // namespace tc::z_image
