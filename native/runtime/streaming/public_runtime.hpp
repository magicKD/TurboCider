#pragma once

#include "catalog_provider.hpp"
#include "resolved_request.hpp"

#include <memory>
#include <string>

namespace tc {
class ModelSession;
}

namespace tc::streaming {

// Pure C++ orchestration for the public preset authority path. Locking,
// request parsing, platform device discovery and result serialization remain
// at the API boundary. The coordinator itself never acquires the global GPU
// lock or creates a DeviceLease.
class PublicStreamingCoordinator final {
  public:
    PublicStreamingCoordinator(
        ModelSession &, std::string engine_model_id,
        std::string execution_container,
        const StreamingCatalogProvider &);

    // Cheap request/engine/catalog checks. API callers invoke this before
    // make_plan so an empty production catalog remains fail-closed before
    // model validation or metadata probing.
    void preflight(const Request &) const;

    // The request must already contain the normalized result of make_plan.
    // This performs metadata-only probe, deterministic selection, immutable
    // snapshot compilation and internal authority creation.
    std::shared_ptr<const ResolvedRequestExecution> resolve_normalized(
        Request, const StreamingDeviceIdentity &) const;

    // Called after the global GPU lock is held and immediately before model
    // execution. It replays the exact selector, verifies the source lease and
    // checks that the internal authority still matches device/layout/runtime.
    void revalidate(const ResolvedRequestExecution &,
                    const StreamingDeviceIdentity &) const;

  private:
    ModelSession &session_;
    std::string engine_model_id_;
    std::string execution_container_;
    const StreamingCatalogProvider &catalog_provider_;
};

} // namespace tc::streaming
