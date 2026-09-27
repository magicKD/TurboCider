#pragma once

#include "../../core/contracts.hpp"

namespace tc::streaming {

// Cheap, request-only validation shared by the planner and the exact public
// resolver. It must remain free of catalog access, model metadata probing,
// filesystem reads, GPU state and backend construction.
void validate_public_streaming_request(const Request &);

} // namespace tc::streaming
