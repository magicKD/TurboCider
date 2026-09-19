#pragma once

#include "preset_catalog.hpp"

#include <atomic>
#include <memory>

namespace tc::streaming {

// Catalog access is intentionally abstracted from the engine orchestration so
// host tests can exercise a complete reviewed-record resolution without
// adding a writable catalog entry point to the release C ABI. Providers must
// return an immutable snapshot for the duration of one resolve/revalidate
// call. Returning shared ownership prevents a mutable test/staging provider
// from invalidating the exact catalog view between preflight and selection.
// Production uses a process-lifetime compiled catalog.
class StreamingCatalogProvider {
  public:
    virtual ~StreamingCatalogProvider() = default;
    virtual std::shared_ptr<const StreamingPresetCatalog>
    snapshot() const = 0;
};

const StreamingCatalogProvider &production_streaming_catalog_provider();

#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
// Test/calibration-only provider.  The installed snapshot is replaced as one
// immutable shared_ptr; readers that already captured a snapshot continue to
// see the old catalog until their resolve/revalidate call completes.  This
// class is intentionally unavailable from release builds.
class TestStreamingCatalogProvider final : public StreamingCatalogProvider {
  public:
    TestStreamingCatalogProvider();

    std::shared_ptr<const StreamingPresetCatalog>
    snapshot() const override;

    void install(std::shared_ptr<const StreamingPresetCatalog>);
    void clear();

  private:
    mutable std::shared_ptr<const StreamingPresetCatalog> catalog_;
};
#endif

} // namespace tc::streaming
