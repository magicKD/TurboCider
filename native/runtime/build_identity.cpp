#include "build_identity.hpp"
#include "turbocider_runtime_build_generated.hpp"

namespace tc {
const char *runtime_build_identity() noexcept {
    return TURBOCIDER_RUNTIME_BUILD_ID;
}
}
