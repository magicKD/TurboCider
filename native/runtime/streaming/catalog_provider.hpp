#pragma once

#include "preset_catalog.hpp"

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

} // namespace tc::streaming
