#pragma once

#include "preset_catalog.hpp"

namespace tc::streaming {

// Catalog access is intentionally abstracted from the engine orchestration so
// host tests can exercise a complete reviewed-record resolution without
// adding a writable catalog entry point to the release C ABI. Providers must
// return an immutable snapshot for the duration of one resolve/revalidate
// call. Production uses a process-lifetime compiled catalog.
class StreamingCatalogProvider {
  public:
    virtual ~StreamingCatalogProvider() = default;
    virtual const StreamingPresetCatalog &catalog() const noexcept = 0;
};

const StreamingCatalogProvider &production_streaming_catalog_provider();

} // namespace tc::streaming
