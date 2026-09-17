#include "catalog_provider.hpp"

namespace tc::streaming {
namespace {

class ProductionStreamingCatalogProvider final
    : public StreamingCatalogProvider {
  public:
    std::shared_ptr<const StreamingPresetCatalog>
    snapshot() const override {
        static const auto catalog =
            std::make_shared<const StreamingPresetCatalog>(
                production_streaming_preset_catalog());
        return catalog;
    }
};

} // namespace

const StreamingCatalogProvider &production_streaming_catalog_provider() {
    static const ProductionStreamingCatalogProvider provider;
    return provider;
}

} // namespace tc::streaming
