#pragma once

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void h3_video_vae_config_error(
    char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static inline int h3_video_vae_parse_float_array(
    const char *json, const char *key, float *values, size_t count,
    char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        h3_video_vae_config_error(error, error_size, "video VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            h3_video_vae_config_error(error, error_size, "video VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                h3_video_vae_config_error(error, error_size, "video VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            h3_video_vae_config_error(error, error_size, "video VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static inline int h3_video_vae_load_normalization(
    const char *weight_directory, float *mean, float *deviation,
    size_t channels, const char *invalid_deviation_error,
    char *error, size_t error_size) {
    size_t path_size = strlen(weight_directory) + strlen("/../config.json") + 1;
    char *path = malloc(path_size);
    if (!path) {
        h3_video_vae_config_error(error, error_size, "out of memory resolving video VAE config");
        return 0;
    }
    snprintf(path, path_size, "%s/../config.json", weight_directory);
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        h3_video_vae_config_error(
            error, error_size, "cannot open video VAE config %s: %s", path, strerror(errno));
        if (file) fclose(file);
        free(path);
        return 0;
    }
    long end = ftell(file);
    if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        h3_video_vae_config_error(error, error_size, "invalid video VAE config %s", path);
        fclose(file);
        free(path);
        return 0;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        h3_video_vae_config_error(error, error_size, "cannot read video VAE config %s", path);
        free(json);
        fclose(file);
        free(path);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    free(path);
    int ok = h3_video_vae_parse_float_array(
                 json, "latents_mean", mean, channels, error, error_size) &&
             h3_video_vae_parse_float_array(
                 json, "latents_std", deviation, channels, error, error_size);
    free(json);
    if (ok) for (size_t channel = 0; channel < channels; channel++) {
        if (deviation[channel] <= 0.0f) {
            h3_video_vae_config_error(error, error_size, "%s", invalid_deviation_error);
            return 0;
        }
    }
    return ok;
}
