#pragma once

namespace tc {
// Generated from build inputs, never supplied by a request or catalog.
// The manifest beside the binary describes the exact fingerprint scope.
const char *runtime_build_identity() noexcept;
// Catalog compatibility excludes test hooks/audit counters only. The complete
// identity above continues to bind worker transport and executable generations.
const char *catalog_runtime_identity() noexcept;
}
