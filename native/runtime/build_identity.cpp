#include "build_identity.hpp"
#include "turbocider_runtime_build_generated.hpp"

namespace tc {
const char *runtime_build_identity() noexcept {
    return TURBOCIDER_RUNTIME_BUILD_ID;
}
const char *catalog_runtime_identity() noexcept {
    return TURBOCIDER_CATALOG_RUNTIME_ID;
}
}
