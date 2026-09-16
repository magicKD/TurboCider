#ifndef H3_MEMORY_H
#define H3_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    H3_HOST_MEMORY_CONDITIONING = 1,
    H3_HOST_MEMORY_LATENT = 2,
    H3_HOST_MEMORY_DECODED_F32 = 3,
    H3_HOST_MEMORY_OUTPUT = 4,
    H3_HOST_MEMORY_STAGING = 5,
    H3_HOST_MEMORY_CONTROL = 6,
    H3_HOST_MEMORY_UNKNOWN = 255
} h3_host_memory_class;

typedef struct {
    uint32_t struct_size;
    uint32_t version;
    void *user;
    int (*reserve)(void *user, uint32_t memory_class,
                   uint64_t upper_bytes, const char *tag,
                   void **token, char *error, size_t error_size);
    int (*commit)(void *user, void *token, uint64_t allocator_domain,
                  uint64_t handle, uint64_t actual_bytes,
                  uint64_t generation, char *error, size_t error_size);
    void (*cancel)(void *user, void *token);
    void (*release)(void *user, void *token);
} h3_host_memory_hooks;

/* Internal options propagated to independently owned H3 subcomponents.  The
 * hooks and bridge remain owned by the top-level caller for the complete
 * generation call. */
typedef struct {
    uint32_t struct_size;
    uint32_t version;
    const h3_host_memory_hooks *hooks;
    uint64_t allocator_domain;
    uint64_t generation;
} h3_host_memory_options;

#ifdef __cplusplus
}
#endif

#endif
