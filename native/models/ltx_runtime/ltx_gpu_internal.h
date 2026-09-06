#ifndef LTX_GPU_INTERNAL_H
#define LTX_GPU_INTERNAL_H

#include "ltx_gpu.h"

/* Non-owning Objective-C object pointers used by sibling .m modules. */
void *ltx_gpu_native_device(const ltx_gpu *gpu);
void *ltx_gpu_native_queue(const ltx_gpu *gpu);
void *ltx_gpu_buffer_native(const ltx_gpu_buffer *buffer);

#endif
