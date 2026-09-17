#pragma once

#include "resolved_request.hpp"

namespace tc::streaming {

class PublicPresetResolver final {
  public:
    static SelectedStreamingPreset select(
        const StreamingSelector &,
        const ModelStreamingProbe &,
        const StreamingDeviceIdentity &,
        const StreamingPresetCatalog &);

    static ResolvedStreamingSelection authorize(
        const SelectedStreamingPreset &,
        const ModelStreamingProbe &,
        const ModelStreamingSnapshot &,
        const StreamingDeviceIdentity &);
};

} // namespace tc::streaming
