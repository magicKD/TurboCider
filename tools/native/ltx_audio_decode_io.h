#ifndef TURBOCIDER_LTX_AUDIO_DECODE_IO_H
#define TURBOCIDER_LTX_AUDIO_DECODE_IO_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shared file I/O for the standalone LTX audio decoding tools. Callers keep
 * their shape-specific diagnostics and own the returned input allocation. */
static void fail(const char *message) {
    fprintf(stderr, "error: %s\n", message);
    exit(1);
}

static void *read_file(const char *path, size_t expected_bytes,
                       const char *shape_error, const char *allocation_error,
                       const char *read_error) {
    FILE *stream = fopen(path, "rb");
    if (!stream) fail(strerror(errno));
    if (fseek(stream, 0, SEEK_END) != 0) fail("cannot seek input");
    long length = ftell(stream);
    if (length < 0 || (size_t)length != expected_bytes) {
        fail(shape_error);
    }
    rewind(stream);
    void *data = malloc(expected_bytes);
    if (!data) fail(allocation_error);
    if (fread(data, 1, expected_bytes, stream) != expected_bytes) {
        fail(read_error);
    }
    fclose(stream);
    return data;
}

static void write_file(const char *path, const void *data, size_t bytes,
                       const char *write_error) {
    FILE *stream = fopen(path, "wb");
    if (!stream) fail(strerror(errno));
    if (fwrite(data, 1, bytes, stream) != bytes || fclose(stream) != 0) {
        fail(write_error);
    }
}

#endif
