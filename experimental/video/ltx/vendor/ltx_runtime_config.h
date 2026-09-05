#pragma once
/* Library routes never inherit process environment tuning or subprocess flags. */
static inline const char *ltx_runtime_getenv(const char *key) {(void)key;return 0;}
