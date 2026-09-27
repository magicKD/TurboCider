#include "catalog_provider.hpp"

#include <utility>

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

#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
TestStreamingCatalogProvider::TestStreamingCatalogProvider()
    : catalog_(std::make_shared<const StreamingPresetCatalog>(
          StreamingPresetCatalog{"tc-streaming-test-empty-v1", {}})) {}

std::shared_ptr<const StreamingPresetCatalog>
TestStreamingCatalogProvider::snapshot() const {
    return std::atomic_load_explicit(&catalog_, std::memory_order_acquire);
}

void TestStreamingCatalogProvider::install(
        std::shared_ptr<const StreamingPresetCatalog> catalog) {
    if (!catalog)
        throw std::invalid_argument("streaming_test_catalog_null");
    std::atomic_store_explicit(&catalog_, std::move(catalog),
                               std::memory_order_release);
}

void TestStreamingCatalogProvider::clear() {
    install(std::make_shared<const StreamingPresetCatalog>(
        StreamingPresetCatalog{"tc-streaming-test-empty-v1", {}}));
}
#endif

} // namespace tc::streaming
