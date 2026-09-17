#include "catalog_provider.hpp"

namespace tc::streaming {
namespace {

class ProductionStreamingCatalogProvider final
    : public StreamingCatalogProvider {
  public:
    const StreamingPresetCatalog &catalog() const noexcept override {
        return production_streaming_preset_catalog();
    }
};

} // namespace

const StreamingCatalogProvider &production_streaming_catalog_provider() {
    static const ProductionStreamingCatalogProvider provider;
    return provider;
}

} // namespace tc::streaming
