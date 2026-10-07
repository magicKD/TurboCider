#pragma once

#include "ane_runtime.hpp"
#import <CoreML/CoreML.h>

namespace tc::ane {
// Older supported SDKs have no public INT8 feature type. Preserve the
// legacy FP16 and Private routes without inventing an undeclared enum value.
inline MLMultiArrayDataType public_w8_array_type(bool int8) {
    if (!int8) return MLMultiArrayDataTypeFloat16;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) return MLMultiArrayDataTypeInt8;
#endif
    throw CapabilityError("Public W8A8 INT8 I/O requires macOS 26 and a macOS 26 SDK build");
}
} // namespace tc::ane
