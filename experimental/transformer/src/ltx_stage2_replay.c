/* Fixed-geometry real LTX replay: 704x448, 97 frames, seed 42.
 * Optional arguments enable Stage-1/upsampling and video decoding. The native
 * denoiser stays resident; request-layer work is not executed.
 * Stage-1 audio is the Stage-2 input; final Stage-2 audio is a reference only.
 */
#include "../../../native/models/ltx_runtime/ltx_native.h"
#include "../../../native/models/ltx_runtime/ltx_rng.h"
#include "../../../native/models/ltx_runtime/ltx_mlx_video_vae.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <mach/mach.h>
#import <Foundation/Foundation.h>

static double now(void);

/* Read-only process/host snapshots, not an attribution of host-wide paging
 * to this process. Failed queries are explicit rather than reported as zeros. */
static void memory_snapshot(long run, const char *phase) {
    const char *enabled = getenv("TURBOCIDER_LTX_REPLAY_MEMORY");
    if (!enabled || strcmp(enabled, "1") != 0) return;
    task_vm_info_data_t task = {0};
    mach_msg_type_number_t task_count = TASK_VM_INFO_COUNT;
    kern_return_t task_status = task_info(mach_task_self(), TASK_VM_INFO,
        (task_info_t)&task, &task_count);
    vm_statistics64_data_t host = {0};
    mach_msg_type_number_t host_count = HOST_VM_INFO64_COUNT;
    mach_port_t host_port = mach_host_self();
    kern_return_t host_status = host_statistics64(host_port, HOST_VM_INFO64,
        (host_info64_t)&host, &host_count);
    mach_port_deallocate(mach_task_self(), host_port);
    struct rusage usage = {0};
    int usage_status = getrusage(RUSAGE_SELF, &usage);
    printf("{\"kind\":\"memory\",\"run\":%ld,\"phase\":\"%s\",\"task_status\":%d,\"host_status\":%d,\"usage_status\":%d",
        run, phase, task_status, host_status, usage_status);
    if (task_status == KERN_SUCCESS)
        printf(",\"resident_bytes\":%llu,\"phys_footprint_bytes\":%llu,\"compressed_bytes\":%llu,\"internal_bytes\":%llu,\"external_bytes\":%llu",
            (unsigned long long)task.resident_size,
            (unsigned long long)task.phys_footprint,
            (unsigned long long)task.compressed,
            (unsigned long long)task.internal,
            (unsigned long long)task.external);
    if (host_status == KERN_SUCCESS)
        printf(",\"host_pageins\":%llu,\"host_pageouts\":%llu,\"host_swapins\":%llu,\"host_swapouts\":%llu,\"host_compressions\":%llu,\"host_decompressions\":%llu",
            (unsigned long long)host.pageins, (unsigned long long)host.pageouts,
            (unsigned long long)host.swapins, (unsigned long long)host.swapouts,
            (unsigned long long)host.compressions, (unsigned long long)host.decompressions);
    if (!usage_status)
        printf(",\"minor_faults\":%ld,\"major_faults\":%ld", usage.ru_minflt, usage.ru_majflt);
    printf("}\n");
    fflush(stdout);
}

static double run_inprocess_decoder(const char *checkpoint,
                                    const void *latent, size_t bytes) {
    const size_t elements = (size_t)3 * 97 * 448 * 704;
    char error[4096] = {0};
    double started = now();
    ltx_mlx_video_vae *vae = ltx_mlx_video_vae_create(checkpoint, error, sizeof(error));
    uint16_t *pixels = vae ? malloc(elements * sizeof(uint16_t)) : NULL;
    int ok = pixels && ltx_mlx_video_vae_decode_tokens_bf16(
        vae, pixels, elements, latent, bytes / 2, 1, 13, 14, 22, error, sizeof(error));
    if (!ok) fprintf(stderr, "in-process decoder: %s\n", error);
    free(pixels);
    ltx_mlx_video_vae_free(vae);
    ltx_mlx_video_vae_clear_cache();
    return ok ? now() - started : -1.0;
}

static double run_video_decoder(const char *helper, const char *checkpoint,
                                const void *latent, size_t bytes, int readback) {
    @autoreleasepool {
    NSString *root = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"ltx-replay-%@", NSUUID.UUID.UUIDString]];
    if (![[NSFileManager defaultManager] createDirectoryAtPath:root
        withIntermediateDirectories:NO attributes:nil error:NULL]) return -1.0;
    NSString *input = [root stringByAppendingPathComponent:@"in.bf16"];
    NSString *output = [root stringByAppendingPathComponent:@"out.bf16"];
    if (![[NSData dataWithBytes:latent length:bytes] writeToFile:input atomically:YES]) {
        [[NSFileManager defaultManager] removeItemAtPath:root error:NULL];
        return -1.0;
    }
    NSTask *task = [NSTask new];
    task.launchPath = [NSString stringWithUTF8String:helper];
    task.arguments = @[[NSString stringWithUTF8String:checkpoint], input,
        @"13", @"14", @"22", output];
    double started = now();
    BOOL ok = NO;
    @try {
        NSError *error = nil;
        if ([task launchAndReturnError:&error]) {
            [task waitUntilExit];
            ok = task.terminationStatus == 0;
        } else {
            fprintf(stderr, "decoder launch: %s\n", error.description.UTF8String);
        }
    }
    @catch (NSException *exception) {
        fprintf(stderr, "decoder exception: %s\n", exception.description.UTF8String);
    }
    [task release];
    /* Match the parent's readback, but omit RGB conversion and video export. */
    if (ok && readback) {
        NSData *pixels = [NSData dataWithContentsOfFile:output];
        ok = pixels && pixels.length == (size_t)3 * 97 * 448 * 704 * 2;
    } else if (ok) {
        struct stat info;
        ok = stat(output.fileSystemRepresentation, &info) == 0 &&
             info.st_size == (off_t)3 * 97 * 448 * 704 * 2;
    }
    double elapsed = now() - started;
    [[NSFileManager defaultManager] removeItemAtPath:root error:NULL];
    return ok ? elapsed : -1.0;
    }
}

static void *read_tensor(const char *directory, const char *name, size_t bytes) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", directory, name) >= (int)sizeof(path))
        return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    void *p = malloc(bytes);
    int ok = p && fread(p, 1, bytes, f) == bytes && fgetc(f) == EOF && !ferror(f);
    fclose(f);
    if (!ok) { fprintf(stderr, "invalid tensor: %s\n", path); free(p); return NULL; }
    return p;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc != 7 && argc != 9 && argc != 11) {
        fprintf(stderr, "usage: %s CHECKPOINT ABSOLUTE_SHADER CONDITIONING_DIR DUMP_DIR ANE_DIR_OR_DASH RUNS [UPSAMPLER VIDEO_VAE [DECODER DECODER_CHECKPOINT]]\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    long runs = strtol(argv[6], &end, 10);
    if (!end || *end || runs < 1 || runs > 20) return 2;
    const char *decode_mode = getenv("TURBOCIDER_LTX_REPLAY_VIDEO_MODE");
    if (!decode_mode) decode_mode = "readback";
    if (strcmp(decode_mode, "readback") != 0 &&
        strcmp(decode_mode, "stat-only") != 0 &&
        strcmp(decode_mode, "in-process") != 0) {
        fprintf(stderr, "invalid replay video mode: %s\n", decode_mode);
        return 2;
    }
    const char *rebuild_flag = getenv("TURBOCIDER_LTX_REPLAY_REBUILD_DENOISER");
    const int rebuild_after_decode = rebuild_flag && strcmp(rebuild_flag, "1") == 0;
    /* Validate all optional files before loading the expensive Transformer. */
    const int file_args[] = {1, 2, 7, 8, 9, 10};
    for (unsigned i = 0; i < sizeof(file_args) / sizeof(file_args[0]); ++i) {
        int arg = file_args[i];
        if (arg >= argc) continue;
        struct stat info;
        if (stat(argv[arg], &info) != 0 || !S_ISREG(info.st_mode)) {
            fprintf(stderr, "missing regular input file: %s\n", argv[arg]);
            return 2;
        }
    }
    const char *pool_flag = getenv("TURBOCIDER_LTX_REPLAY_DRAIN_POOL");
    const int drain_pool = pool_flag && strcmp(pool_flag, "1") == 0;
    const size_t vb = 4004u * 128u * 2u, ab = 101u * 128u * 2u;
    const size_t stage1_bytes = 1001u * 128u * 2u;
    void *p[11] = {0};
    ltx_native_denoiser *ctx = NULL;
    int status = 1;
    p[0] = read_tensor(argv[3], "video_context.bf16", 1024u * 4096u * 2u);
    p[1] = read_tensor(argv[3], "audio_context.bf16", 1024u * 2048u * 2u);
    p[2] = read_tensor(argv[3], "text_mask.bf16", 1024u * 2u);
    p[3] = read_tensor(argv[4], "stage2_input_video.bf16", vb);
    p[4] = read_tensor(argv[4], "stage1_audio.bf16", ab);
    p[5] = read_tensor(argv[4], "stage2_video.bf16", vb);
    p[6] = read_tensor(argv[4], "stage2_audio.bf16", ab);
    p[7] = malloc(vb); p[8] = malloc(ab);
    for (int i = 0; i < 9; ++i) if (!p[i]) goto done;
    if (argc == 9 || argc == 11) {
        p[9] = malloc(stage1_bytes);
        p[10] = read_tensor(argv[4], "stage1_video.bf16", stage1_bytes);
        if (!p[9] || !p[10]) goto done;
    }
    ltx_native_options options = {0};
    options.checkpoint = argv[1]; options.shader_source = argv[2];
    options.width = 704; options.height = 448; options.frames = 97; options.fps = 24;
    options.parallel_av = 1;
    if (strcmp(argv[5], "-") != 0) {
        options.mlp_directories[1] = argv[5];
        options.ane_variant = "int8_pc";
        options.ane_mlp_stage_mask = 2;
        options.ane_mlp_block_count = 48;
        options.preload_ane_stage2 = 1;
    }
    char error[4096] = {0};
    double started = now();
    ctx = ltx_native_create(&options, NULL, NULL, error, sizeof(error));
    if (!ctx) { fprintf(stderr, "%s\n", error); goto done; }
    printf("{\"kind\":\"load\",\"seconds\":%.9f,\"rows\":4004,\"stage\":2,\"decoder\":%s,\"drain_pool\":%s}\n", now() - started, argc == 11 ? "true" : "false", drain_pool ? "true" : "false");
    fflush(stdout);
    printf("{\"kind\":\"decoder_config\",\"mode\":\"%s\",\"enabled\":%s}\n",
        decode_mode, argc == 11 ? "true" : "false");
    fflush(stdout);
    status = 0;
    for (long run = 0; run < runs; ++run) {
        memory_snapshot(run, "before_denoise");
        /* MRC in this diagnostic only. @finally also drains on failure/break.
         * Native model ownership stays unchanged; only autoreleased temporaries
         * created by this thread during the round acquire a request-like pool. */
        NSAutoreleasePool *pool = drain_pool ? [[NSAutoreleasePool alloc] init] : nil;
        @try {
        memcpy(p[7], p[3], vb); memcpy(p[8], p[4], ab);
        if (argc == 9 || argc == 11) {
            ltx_rng vr, ar;
            ltx_rng_seed(&vr, 10042, 0); ltx_rng_seed(&ar, 44, 0);
            ltx_rng_fill_normal_bf16(&vr, p[9], stage1_bytes / 2);
            ltx_rng_fill_normal_bf16(&ar, p[8], ab / 2);
            started = now();
            int ok = ltx_native_run(ctx, 1, 42, p[9], stage1_bytes / 2,
                p[8], ab / 2, p[0], p[1], p[2], 1024, NULL, 0,
                NULL, NULL, error, sizeof(error));
            double stage1_seconds = now() - started;
            if (!ok) { fprintf(stderr, "%s\n", error); status = 1; break; }
            int exact = memcmp(p[9], p[10], stage1_bytes) == 0 &&
                        memcmp(p[8], p[4], ab) == 0;
            printf("{\"kind\":\"stage1\",\"run\":%ld,\"seconds\":%.9f,\"reference_exact\":%s}\n",
                run, stage1_seconds, exact ? "true" : "false");
            fflush(stdout);
            if (!exact) { status = 3; break; }
            started = now();
            ok = ltx_native_upsample_stage2(ctx, argv[7], argv[8], p[7], vb / 2,
                p[9], stage1_bytes / 2, error, sizeof(error));
            if (!ok) { fprintf(stderr, "%s\n", error); status = 1; break; }
            exact = memcmp(p[7], p[3], vb) == 0;
            printf("{\"kind\":\"upsample\",\"run\":%ld,\"seconds\":%.9f,\"reference_exact\":%s}\n",
                run, now() - started, exact ? "true" : "false");
            fflush(stdout);
            if (!exact) { status = 3; break; }
        }
        started = now();
        int ok = ltx_native_run(ctx, 2, 42, p[7], vb / 2, p[8], ab / 2,
            p[0], p[1], p[2], 1024, NULL, 0, NULL, NULL, error, sizeof(error));
        double seconds = now() - started;
        if (!ok) { fprintf(stderr, "%s\n", error); status = 1; break; }
        int video_exact = memcmp(p[7], p[5], vb) == 0;
        int audio_exact = memcmp(p[8], p[6], ab) == 0;
        printf("{\"kind\":\"stage2\",\"run\":%ld,\"seconds\":%.9f,\"video_reference_exact\":%s,\"audio_reference_exact\":%s}\n",
            run, seconds, video_exact ? "true" : "false", audio_exact ? "true" : "false");
        fflush(stdout);
        /* The supplied reference must match the selected GPU/ANE route. */
        if (!video_exact || !audio_exact) { status = 3; break; }
        memory_snapshot(run, "after_stage2");
        if (argc == 11) {
            double decoded = strcmp(decode_mode, "in-process") == 0 ?
                run_inprocess_decoder(argv[10], p[7], vb) :
                run_video_decoder(argv[9], argv[10], p[7], vb,
                                  strcmp(decode_mode, "stat-only") != 0);
            printf("{\"kind\":\"video_decode\",\"run\":%ld,\"seconds\":%.9f,\"success\":%s}\n",
                run, decoded, decoded >= 0.0 ? "true" : "false");
            fflush(stdout);
            if (decoded < 0.0) { status = 1; break; }
            memory_snapshot(run, "after_video_decode");
            if (rebuild_after_decode && run + 1 < runs) {
                double rebuild_started = now();
                ltx_native_free(ctx);
                ctx = ltx_native_create(&options, NULL, NULL, error, sizeof(error));
                if (!ctx) { fprintf(stderr, "%s\n", error); status = 1; break; }
                printf("{\"kind\":\"denoiser_rebuild\",\"run\":%ld,\"seconds\":%.9f}\n",
                       run, now() - rebuild_started);
                fflush(stdout);
                memory_snapshot(run, "after_denoiser_rebuild");
            }
        }
        } @finally {
            [pool drain];
        }
    }
done:
    ltx_native_free(ctx);
    for (int i = 0; i < 11; ++i) free(p[i]);
    return status;
}
