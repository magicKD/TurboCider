#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shared C/MLX policy. Return -1 for an invalid explicit request. */
static inline int ltx_convrot_w8a8_requested(int supported,
                                           char *error, size_t error_size) {
    const char *value = getenv("TURBOCIDER_LTX_CONVROT_W8A8");
    const char *message = NULL;
    if (value && strcmp(value, "0") && strcmp(value, "1"))
        message = "TURBOCIDER_LTX_CONVROT_W8A8 requires 0 or 1";
    int enabled = value ? !strcmp(value, "1") : supported;
    if (!message && enabled && !supported)
        message = "LTX W8A8 requires M5+ and macOS 26.4+";
    if (!message && getenv("TURBOCIDER_LTX_RAW_CONVROT")) {
        if (value && enabled)
            message = "LTX raw ConvRot and explicit W8A8 are incompatible";
        enabled = 0;
    }
    if (message) {
        if (error && error_size) snprintf(error, error_size, "%s", message);
        return -1;
    }
    return enabled;
}
