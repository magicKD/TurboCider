#include "ltx_weights.h"

#import <Foundation/Foundation.h>

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ltx_weights_fail(char *error, size_t error_size,
                            const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static uint32_t ltx_u32(NSDictionary *object, NSString *key,
                        uint32_t fallback) {
    id value = object[key];
    if (![value isKindOfClass:[NSNumber class]]) return fallback;
    unsigned long long parsed = [(NSNumber *)value unsignedLongLongValue];
    return parsed <= UINT32_MAX ? (uint32_t)parsed : fallback;
}

static int ltx_bool(NSDictionary *object, NSString *key, int fallback) {
    id value = object[key];
    return [value isKindOfClass:[NSNumber class]] ?
        [(NSNumber *)value boolValue] : fallback;
}

static int ltx_name_contains(const char *name, const char *needle) {
    return name && needle && strstr(name, needle) != NULL;
}

static int ltx_parse_block_index(const char *name, uint32_t *block) {
    const char *marker = strstr(name, "transformer_blocks.");
    if (!marker) return 0;
    marker += strlen("transformer_blocks.");
    if (*marker < '0' || *marker > '9') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(marker, &end, 10);
    if (errno || end == marker || *end != '.' || value > UINT32_MAX) return 0;
    *block = (uint32_t)value;
    return 1;
}

static void ltx_scan_tensor(const ltx_st_tensor *tensor,
                            ltx_checkpoint_info *info,
                            unsigned char blocks[256]) {
    if ((unsigned)tensor->dtype <= LTX_DTYPE_F64)
        info->dtype_counts[tensor->dtype]++;
    uint32_t block = 0;
    if (ltx_parse_block_index(tensor->name, &block) && block < 256u)
        blocks[block] = 1u;
    if (ltx_name_contains(tensor->name, ".comfy_quant"))
        info->comfy_quantized = 1;
    if (tensor->dtype == LTX_DTYPE_I8)
        info->quantized_int8 = 1;

    if (ltx_name_contains(tensor->name,
            "transformer_blocks.0.ff.net.0.proj.weight") &&
        tensor->ndim == 2u) {
        info->video_ff_dim = (uint32_t)tensor->shape[0];
    }
    if (ltx_name_contains(tensor->name,
            "transformer_blocks.0.audio_ff.net.0.proj.weight") &&
        tensor->ndim == 2u) {
        info->audio_ff_dim = (uint32_t)tensor->shape[0];
    }
    if (ltx_name_contains(tensor->name,
            "transformer_blocks.0.ff.net.0.proj.bias"))
        info->ff_bias = 1;
    if (ltx_name_contains(tensor->name,
            "transformer_blocks.0.audio_ff.net.0.proj.bias"))
        info->audio_ff_bias = 1;
}

int ltx_checkpoint_inspect(const char *path, ltx_checkpoint_info *info,
                           char *error, size_t error_size) {
    if (!path || !info)
        return ltx_weights_fail(error, error_size,
                                "missing checkpoint inspection argument");
    memset(info, 0, sizeof(*info));
    snprintf(info->path, sizeof(info->path), "%s", path);
    snprintf(info->model_class, sizeof(info->model_class), "%s",
             "AVTransformer3DModel");
    info->num_layers = 48u;
    info->video_dim = 4096u;
    info->audio_dim = 2048u;
    info->video_heads = 32u;
    info->audio_heads = 32u;
    info->video_head_dim = 128u;
    info->audio_head_dim = 64u;
    info->video_channels = 128u;
    info->audio_channels = 128u;
    info->audio_ff_bias = 1;
    info->split_rope = 1;
    info->appears_distilled = ltx_name_contains(path, "distilled");

    ltx_st_header header;
    if (!ltx_st_read_header(path, &header, error, error_size)) return 0;
    info->tensor_count = header.tensor_count;

    if (header.metadata_config) {
        @autoreleasepool {
            NSData *data = [NSData dataWithBytes:header.metadata_config
                                          length:strlen(header.metadata_config)];
            NSError *json_error = nil;
            id root_value = [NSJSONSerialization JSONObjectWithData:data
                                                            options:0
                                                              error:&json_error];
            if ([root_value isKindOfClass:[NSDictionary class]]) {
                NSDictionary *root = (NSDictionary *)root_value;
                id transformer_value = root[@"transformer"];
                NSDictionary *transformer =
                    [transformer_value isKindOfClass:[NSDictionary class]] ?
                    transformer_value : root;
                NSString *model_class = transformer[@"_class_name"];
                if ([model_class isKindOfClass:[NSString class]])
                    snprintf(info->model_class, sizeof(info->model_class),
                             "%s", model_class.UTF8String);
                info->num_layers = ltx_u32(transformer, @"num_layers",
                                           info->num_layers);
                info->video_dim = ltx_u32(transformer, @"cross_attention_dim",
                                          info->video_dim);
                info->audio_dim = ltx_u32(transformer,
                                          @"audio_cross_attention_dim",
                                          info->audio_dim);
                info->video_heads = ltx_u32(transformer,
                                            @"num_attention_heads",
                                            info->video_heads);
                info->audio_heads = ltx_u32(transformer,
                                            @"audio_num_attention_heads",
                                            info->audio_heads);
                info->video_head_dim = ltx_u32(transformer,
                                               @"attention_head_dim",
                                               info->video_head_dim);
                info->audio_head_dim = ltx_u32(transformer,
                                               @"audio_attention_head_dim",
                                               info->audio_head_dim);
                info->video_channels = ltx_u32(transformer, @"in_channels",
                                               info->video_channels);
                info->audio_channels = ltx_u32(transformer,
                                               @"audio_in_channels",
                                               info->audio_channels);
                info->ff_bias = ltx_bool(transformer, @"ff_bias",
                                         info->ff_bias);
                info->audio_ff_bias = ltx_bool(transformer, @"audio_ff_bias",
                                               info->audio_ff_bias);
                info->keyframe_absolute_positions = ltx_bool(
                    transformer, @"use_keyframes_abs_pos_embedding", 0);
                NSString *rope_type = transformer[@"rope_type"];
                if ([rope_type isKindOfClass:[NSString class]])
                    info->split_rope = [rope_type isEqualToString:@"split"];
                NSString *precision = transformer[@"frequencies_precision"];
                if ([precision isKindOfClass:[NSString class]])
                    info->double_precision_rope =
                        [precision isEqualToString:@"float64"];
            }
        }
    }

    unsigned char blocks[256] = {0};
    for (size_t index = 0; index < header.tensor_count; index++)
        ltx_scan_tensor(&header.tensors[index], info, blocks);
    for (uint32_t index = 0; index < 256u; index++)
        if (blocks[index]) info->transformer_blocks_found++;
    if (!info->video_ff_dim) info->video_ff_dim = info->video_dim * 4u;
    if (!info->audio_ff_dim) info->audio_ff_dim = info->audio_dim * 4u;

    ltx_st_free_header(&header);
    return 1;
}

int ltx_checkpoint_validate_transformer(const ltx_checkpoint_info *info,
                                        char *error, size_t error_size) {
    if (!info)
        return ltx_weights_fail(error, error_size, "missing checkpoint info");
    if (strcmp(info->model_class, "AVTransformer3DModel"))
        return ltx_weights_fail(error, error_size,
                                "checkpoint is not AVTransformer3DModel");
    if (info->num_layers != 48u || info->transformer_blocks_found != 48u)
        return ltx_weights_fail(error, error_size,
                                "checkpoint does not contain all 48 blocks");
    if (info->video_dim != 4096u || info->audio_dim != 2048u ||
        info->video_heads != 32u || info->audio_heads != 32u ||
        info->video_head_dim != 128u || info->audio_head_dim != 64u)
        return ltx_weights_fail(error, error_size,
                                "unsupported LTX transformer geometry");
    if (info->video_channels != 128u || info->audio_channels != 128u)
        return ltx_weights_fail(error, error_size,
                                "unsupported LTX patch channel geometry");
    if (!info->tensor_count)
        return ltx_weights_fail(error, error_size, "empty checkpoint");
    return 1;
}

int ltx_linear_weight_resolve(const ltx_st_header *header,
                              const ltx_st_mapping *mapping,
                              const char *prefix,
                              ltx_linear_weight_info *info,
                              char *error, size_t error_size) {
    if (!header || !mapping || !prefix || !info)
        return ltx_weights_fail(error, error_size,
                                "missing linear weight lookup argument");
    memset(info, 0, sizeof(*info));

    char name[8192];
    int length = snprintf(name, sizeof(name), "%s.weight", prefix);
    if (length < 0 || (size_t)length >= sizeof(name))
        return ltx_weights_fail(error, error_size,
                                "linear weight prefix is too long");
    info->weight = ltx_st_find(header, name);
    if (!info->weight)
        return ltx_weights_fail(error, error_size,
                                "missing linear weight %s", name);
    if (info->weight->ndim != 2u ||
        !info->weight->shape[0] || !info->weight->shape[1] ||
        info->weight->shape[0] > UINT32_MAX ||
        info->weight->shape[1] > UINT32_MAX)
        return ltx_weights_fail(error, error_size,
                                "unsupported linear weight shape for %s", name);
    info->output_dim = (uint32_t)info->weight->shape[0];
    info->input_dim = (uint32_t)info->weight->shape[1];

    snprintf(name, sizeof(name), "%s.bias", prefix);
    info->bias = ltx_st_find(header, name);
    if (info->bias &&
        (info->bias->ndim != 1u ||
         info->bias->shape[0] != info->output_dim ||
         (info->bias->dtype != LTX_DTYPE_BF16 &&
          info->bias->dtype != LTX_DTYPE_F16 &&
          info->bias->dtype != LTX_DTYPE_F32)))
        return ltx_weights_fail(error, error_size,
                                "unsupported linear bias for %s", prefix);

    if (info->weight->dtype == LTX_DTYPE_BF16 ||
        info->weight->dtype == LTX_DTYPE_F16 ||
        info->weight->dtype == LTX_DTYPE_F32)
        return 1;
    if (info->weight->dtype != LTX_DTYPE_I8)
        return ltx_weights_fail(error, error_size,
                                "unsupported linear weight dtype %s for %s",
                                ltx_dtype_name(info->weight->dtype), prefix);

    info->quantized_int8 = 1;
    snprintf(name, sizeof(name), "%s.weight_scale", prefix);
    info->weight_scale = ltx_st_find(header, name);
    if (!info->weight_scale || info->weight_scale->dtype != LTX_DTYPE_F32 ||
        !((info->weight_scale->ndim == 1u &&
           info->weight_scale->shape[0] == info->output_dim) ||
          (info->weight_scale->ndim == 2u &&
           info->weight_scale->shape[0] == info->output_dim &&
           info->weight_scale->shape[1] == 1u)))
        return ltx_weights_fail(error, error_size,
                                "invalid INT8 weight scale for %s", prefix);

    snprintf(name, sizeof(name), "%s.comfy_quant", prefix);
    info->quant_metadata = ltx_st_find(header, name);
    if (!info->quant_metadata) return 1;
    if (info->quant_metadata->dtype != LTX_DTYPE_U8 ||
        info->quant_metadata->ndim != 1u)
        return ltx_weights_fail(error, error_size,
                                "invalid Comfy quant metadata for %s", prefix);

    size_t metadata_bytes = 0;
    const void *metadata = ltx_st_map_tensor(mapping, info->quant_metadata,
                                             &metadata_bytes, error,
                                             error_size);
    if (!metadata || !metadata_bytes || metadata_bytes > 4096u)
        return ltx_weights_fail(error, error_size,
                                "invalid Comfy quant metadata payload for %s",
                                prefix);
    @autoreleasepool {
        NSData *data = [NSData dataWithBytes:metadata length:metadata_bytes];
        NSError *json_error = nil;
        id value = [NSJSONSerialization JSONObjectWithData:data
                                                   options:0
                                                     error:&json_error];
        if (![value isKindOfClass:[NSDictionary class]]) {
            const char *description = json_error ?
                json_error.localizedDescription.UTF8String : NULL;
            return ltx_weights_fail(error, error_size,
                                    "invalid Comfy quant JSON for %s: %s",
                                    prefix,
                                    description ? description :
                                                  "root is not an object");
        }
        NSDictionary *object = (NSDictionary *)value;
        id convrot = object[@"convrot"];
        if ([convrot isKindOfClass:[NSNumber class]])
            info->convrot = [(NSNumber *)convrot boolValue];
        id group_size = object[@"convrot_groupsize"];
        if ([group_size isKindOfClass:[NSNumber class]]) {
            unsigned long long parsed =
                [(NSNumber *)group_size unsignedLongLongValue];
            if (parsed <= UINT32_MAX)
                info->convrot_group_size = (uint32_t)parsed;
        }
    }
    if (info->convrot && info->convrot_group_size != 256u)
        return ltx_weights_fail(error, error_size,
                                "unsupported ConvRot group size %u for %s",
                                info->convrot_group_size, prefix);
    if (info->convrot && info->input_dim % info->convrot_group_size)
        return ltx_weights_fail(error, error_size,
                                "ConvRot group size does not divide K for %s",
                                prefix);
    return 1;
}
