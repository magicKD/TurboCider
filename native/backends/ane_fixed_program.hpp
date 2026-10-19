#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace tc::ane {
// Policy only; no device/model work. Captured once by each executor owner.
inline bool parse_fixed_program(const char *raw) {
    if (!raw || std::string_view(raw)=="0") return false;
    if (std::string_view(raw)=="1") return true;
    throw std::runtime_error("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM requires0 or1");
}
inline bool configured_fixed_program() {
    return parse_fixed_program(std::getenv("TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM"));
}
inline bool program_creation_allowed(bool fixed, bool already_created) {
    return !fixed || !already_created;
}
} // namespace tc::ane
