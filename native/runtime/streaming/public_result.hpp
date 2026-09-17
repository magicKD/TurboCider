#pragma once

#include "../session.hpp"
#include "resolved_request.hpp"

namespace tc::streaming {

// Verifies that the adapter's observed execution is exactly the layout and
// lifecycle authorized by the reviewed preset, then attaches immutable public
// selection metrics to the result. This is called only for active public
// selectors; private candidates and the default path never enter it.
void verify_and_attach_public_streaming_result(
    const ResolvedRequestExecution &, RunResult &);

} // namespace tc::streaming
