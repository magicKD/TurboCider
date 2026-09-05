#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ltx_native_denoiser ltx_native_denoiser;
typedef int (*ltx_native_progress)(const char *phase,int current,int total,void *opaque);
typedef struct {
    const char *checkpoint;
    uint32_t width,height,frames,fps;
    uint64_t seed;
    int parallel_av;
    const char *mlp_directories[2];
    const char *v2a_directories[2];
    const char *kv_directory;
    const char *qkv_directories[2];
} ltx_native_options;
ltx_native_denoiser *ltx_native_create(const ltx_native_options*,ltx_native_progress,void*,char*,size_t);
void ltx_native_free(ltx_native_denoiser*);
/* All inputs are BF16. Stage 1 receives seeded noise, stage 2 receives the
 * normalized upsampled stage-1 latent. Each stage updates video/audio in place
 * only after successful completion; cancelled runs leave caller buffers intact. */
int ltx_native_run(ltx_native_denoiser*,int stage,
    uint16_t *video,size_t video_elements,uint16_t *audio,size_t audio_elements,
    const uint16_t *video_text,const uint16_t *audio_text,const uint16_t *mask,
    uint32_t text_rows,const uint16_t *first_frame,float strength,
    ltx_native_progress,void*,char*,size_t);
#ifdef __cplusplus
}
#endif
