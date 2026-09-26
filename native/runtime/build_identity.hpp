#pragma once

namespace tc {
// Generated from build inputs, never supplied by a request or catalog.
// The manifest beside the binary describes the exact fingerprint scope.
const char *runtime_build_identity() noexcept;
}
