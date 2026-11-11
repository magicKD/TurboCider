/* Isolated real-weight connector timing and repeated-output equality. */
#include "ltx_connector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static void *read_input(const char *directory, const char *name, size_t *bytes) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", directory, name) >= (int)sizeof(path)) return NULL;
    struct stat info;
    if (stat(path, &info) || info.st_size <= 0) return NULL;
    *bytes = (size_t)info.st_size;
    FILE *f = fopen(path, "rb");
    void *p = malloc(*bytes);
    if (!f || !p) { if (f) fclose(f); free(p); return NULL; }
    int ok = fread(p, 1, *bytes, f) == *bytes;
    fclose(f);
    if (!ok) { free(p); return NULL; }
    return p;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: ltx-connector-probe CHECKPOINT SHADER RAW_DIRECTORY OUTPUT_DIRECTORY WARM_RUNS\n");
        return 2;
    }
    char *end = NULL;
    long warm = strtol(argv[5], &end, 10);
    if (!end || *end || warm < 1 || warm > 50) return 2;
    char error[2048] = {0};
    ltx_st_header header = {0};
    ltx_st_mapping mapping = {0};
    ltx_gpu *gpu = NULL;
    ltx_connector *connector = NULL;
    ltx_gpu_buffer *vi = NULL, *ai = NULL, *vo = NULL, *ao = NULL;
    size_t vb = 0, ab = 0;
    void *video = read_input(argv[3], "raw_video_context.bf16", &vb);
    void *audio = read_input(argv[3], "raw_audio_context.bf16", &ab);
    void *actual = NULL, *reference = NULL;
    int ok = 0;
    double started = now(), gpu_seconds = 0, load_seconds = 0, samples[51] = {0};
    if (!video || !audio || vb % (4096u * 2u) || ab % (2048u * 2u) ||
        vb / 4096u != ab / 2048u || vb / 8192u > 4096u) goto cleanup;
    uint32_t input_rows = (uint32_t)(vb / 8192u);
    gpu = ltx_gpu_create(argv[2], error, sizeof(error));
    gpu_seconds = now() - started;
    started = now();
    if (!gpu || !ltx_st_read_header(argv[1], &header, error, sizeof(error)) ||
        !ltx_st_map_open(&header, &mapping, error, sizeof(error))) goto cleanup;
    connector = ltx_connector_load(&header, &mapping, gpu, "model.diffusion_model", error, sizeof(error));
    load_seconds = now() - started;
    if (!connector) goto cleanup;
    uint32_t rows = ltx_connector_output_rows(connector, input_rows);
    size_t vout = (size_t)rows * 4096u * 2u, aout = (size_t)rows * 2048u * 2u;
    vi = ltx_gpu_buffer_new_copy(gpu, video, vb, error, sizeof(error));
    ai = ltx_gpu_buffer_new_copy(gpu, audio, ab, error, sizeof(error));
    vo = ltx_gpu_buffer_new(gpu, vout, error, sizeof(error));
    ao = ltx_gpu_buffer_new(gpu, aout, error, sizeof(error));
    actual = malloc(vout + aout); reference = malloc(vout + aout);
    if (!vi || !ai || !vo || !ao || !actual || !reference) goto cleanup;
    for (long run = 0; run <= warm; ++run) {
        started = now();
        if (!ltx_connector_run_bf16(connector, vo, ao, vi, ai, input_rows, error, sizeof(error))) goto cleanup;
        samples[run] = now() - started;
        // Correctness reads and comparisons stay outside the compute timing.
        if (!ltx_gpu_buffer_read(vo, actual, vout, error, sizeof(error)) ||
            !ltx_gpu_buffer_read(ao, (char *)actual + vout, aout, error, sizeof(error))) goto cleanup;
        if (!run) memcpy(reference, actual, vout + aout);
        else if (memcmp(reference, actual, vout + aout)) {
            snprintf(error, sizeof(error), "connector changed repeated output at run %ld", run); goto cleanup;
        }
    }
    if (mkdir(argv[4], 0755)) goto cleanup;
    for (int i = 0; i < 2; ++i) {
        char path[4096];
        if (snprintf(path, sizeof(path), "%s/%s_context.bf16", argv[4], i ? "audio" : "video") >= (int)sizeof(path)) goto cleanup;
        FILE *f = fopen(path, "wb");
        if (!f) goto cleanup;
        const size_t n = i ? aout : vout;
        const void *p = (char *)actual + (i ? vout : 0);
        int written = fwrite(p, 1, n, f) == n;
        int closed = fclose(f);
        if (!written || closed) goto cleanup;
    }
    printf("{\"complete\":true,\"scope\":\"isolated connector, original coefficients, output reads outside compute timing\",\"input_rows\":%u,\"output_rows\":%u,\"gpu_create_seconds\":%.9f,\"weight_load_seconds\":%.9f,\"first_seconds\":%.9f,\"all_repeated_outputs_byte_exact\":true,\"warm_seconds\":[", input_rows, rows, gpu_seconds, load_seconds, samples[0]);
    for (long i = 1; i <= warm; ++i) printf("%s%.9f", i > 1 ? "," : "", samples[i]);
    printf("]}\n");
    ok = 1;
cleanup:
    free(reference); free(actual); free(video); free(audio);
    ltx_gpu_buffer_free(ao); ltx_gpu_buffer_free(vo); ltx_gpu_buffer_free(ai); ltx_gpu_buffer_free(vi);
    ltx_connector_free(connector); ltx_st_map_close(&mapping); ltx_st_free_header(&header); ltx_gpu_free(gpu);
    if (!ok) fprintf(stderr, "%s\n", error[0] ? error : "connector probe input/output failure");
    return ok ? 0 : 1;
}
