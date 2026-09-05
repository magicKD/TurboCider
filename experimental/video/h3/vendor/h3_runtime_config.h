#pragma once
#ifdef __cplusplus
extern "C" {
#endif
/* Immutable during inference, including native worker threads. Never reads
 * process environment. Config changes are serialized by the engine coordinator. */
const char *h3_runtime_getenv(const char *key);
#ifdef __cplusplus
}
#endif
