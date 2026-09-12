#include "ltx_weights.h"

#import <Foundation/Foundation.h>

#include <errno.h>
#include <math.h>
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

    uint64_t metadata_bytes64 = info->quant_metadata->data_end -
        info->quant_metadata->data_begin;
    if (!metadata_bytes64 || metadata_bytes64 > 4096u)
        return ltx_weights_fail(error, error_size,
                                "invalid Comfy quant metadata payload for %s",
                                prefix);
    size_t metadata_bytes = (size_t)metadata_bytes64;
    unsigned char metadata[4096];
    if (!ltx_st_read_mapped_data(
            mapping, info->quant_metadata, metadata, metadata_bytes,
            error, error_size)) return 0;
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

static const ltx_st_tensor *ltx_gemma_require_tensor(
        const ltx_st_header *header, const char *name, ltx_dtype dtype,
        uint32_t ndim, uint64_t first, uint64_t second,
        char *error, size_t error_size) {
    const ltx_st_tensor *tensor = ltx_st_find(header, name);
    if (!tensor) {
        ltx_weights_fail(error, error_size,
                         "missing Gemma tensor %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim ||
        (ndim > 0u && tensor->shape[0] != first) ||
        (ndim > 1u && tensor->shape[1] != second)) {
        ltx_weights_fail(error, error_size,
                         "unexpected Gemma tensor shape/dtype for %s", name);
        return NULL;
    }
    return tensor;
}

static int ltx_gemma_require_linear(
        const ltx_st_header *header, const ltx_st_mapping *mapping,
        const char *prefix, uint32_t input_dim, uint32_t output_dim,
        int *all_convrot, char *error, size_t error_size) {
    ltx_linear_weight_info linear;
    if (!ltx_linear_weight_resolve(header, mapping, prefix, &linear,
                                   error, error_size)) return 0;
    if (linear.input_dim != input_dim || linear.output_dim != output_dim)
        return ltx_weights_fail(error, error_size,
                                "unexpected Gemma linear geometry for %s",
                                prefix);
    if (!linear.quantized_int8 || !linear.convrot ||
        linear.convrot_group_size != 256u) {
        *all_convrot = 0;
        return ltx_weights_fail(error, error_size,
                                "Gemma linear is not ConvRot INT8/256: %s",
                                prefix);
    }
    if (linear.bias)
        return ltx_weights_fail(error, error_size,
                                "unsupported Gemma linear bias for %s",
                                prefix);
    return 1;
}

int ltx_gemma_checkpoint_inspect(const char *path,
                                 ltx_gemma_checkpoint_info *info,
                                 char *error, size_t error_size) {
    if (!path || !info)
        return ltx_weights_fail(error, error_size,
                                "missing Gemma checkpoint inspection argument");
    memset(info, 0, sizeof(*info));
    snprintf(info->path, sizeof(info->path), "%s", path);
    ltx_st_header header;
    if (!ltx_st_read_header(path, &header, error, error_size)) return 0;
    info->tensor_count = header.tensor_count;
    if (!header.metadata_gemma_config) {
        ltx_st_free_header(&header);
        return ltx_weights_fail(error, error_size,
                                "Gemma checkpoint has no gemma_config metadata");
    }

    __block NSArray *layer_types = nil;
    @autoreleasepool {
        NSData *data = [NSData dataWithBytes:header.metadata_gemma_config
                                      length:strlen(header.metadata_gemma_config)];
        NSError *json_error = nil;
        id raw = [NSJSONSerialization JSONObjectWithData:data
                                                 options:0
                                                   error:&json_error];
        NSDictionary *root = [raw isKindOfClass:NSDictionary.class] ? raw : nil;
        NSDictionary *text = [root[@"text_config"]
            isKindOfClass:NSDictionary.class] ? root[@"text_config"] : nil;
        if (!root || !text ||
            ![root[@"model_type"] isEqual:@"gemma4_unified"] ||
            ![text[@"model_type"] isEqual:@"gemma4_unified_text"]) {
            ltx_st_free_header(&header);
            return ltx_weights_fail(error, error_size,
                                    "invalid Gemma4 unified metadata");
        }
        info->vocab_size = ltx_u32(text, @"vocab_size", 0u);
        info->hidden_size = ltx_u32(text, @"hidden_size", 0u);
        info->intermediate_size = ltx_u32(text, @"intermediate_size", 0u);
        info->num_layers = ltx_u32(text, @"num_hidden_layers", 0u);
        info->attention_heads = ltx_u32(text, @"num_attention_heads", 0u);
        info->sliding_kv_heads =
            ltx_u32(text, @"num_key_value_heads", 0u);
        info->full_kv_heads =
            ltx_u32(text, @"num_global_key_value_heads", 0u);
        info->sliding_head_dim = ltx_u32(text, @"head_dim", 0u);
        info->full_head_dim = ltx_u32(text, @"global_head_dim", 0u);
        info->attention_k_eq_v =
            [text[@"attention_k_eq_v"] isKindOfClass:NSNumber.class] ?
            [text[@"attention_k_eq_v"] boolValue] : 0;
        NSDictionary *rope = [text[@"rope_parameters"]
            isKindOfClass:NSDictionary.class] ? text[@"rope_parameters"] : nil;
        NSDictionary *sliding_rope = [rope[@"sliding_attention"]
            isKindOfClass:NSDictionary.class] ? rope[@"sliding_attention"] : nil;
        NSDictionary *full_rope = [rope[@"full_attention"]
            isKindOfClass:NSDictionary.class] ? rope[@"full_attention"] : nil;
        if (ltx_u32(text, @"sliding_window", 0u) != 1024u ||
            ltx_u32(text, @"num_kv_shared_layers", UINT32_MAX) != 0u ||
            ![text[@"hidden_activation"] isEqual:@"gelu_pytorch_tanh"] ||
            ![text[@"use_bidirectional_attention"] isEqual:@"vision"] ||
            ![sliding_rope[@"rope_type"] isEqual:@"default"] ||
            ![full_rope[@"rope_type"] isEqual:@"proportional"] ||
            fabs([text[@"rms_norm_eps"] doubleValue] - 1e-6) > 1e-12 ||
            fabs([sliding_rope[@"rope_theta"] doubleValue] - 10000.0) > 1e-6 ||
            fabs([full_rope[@"rope_theta"] doubleValue] - 1000000.0) > 1e-6 ||
            fabs([full_rope[@"partial_rotary_factor"] doubleValue] - 0.25) >
                1e-12) {
            ltx_st_free_header(&header);
            return ltx_weights_fail(error, error_size,
                                    "unsupported Gemma4 attention/norm parameters");
        }
        NSArray *raw_types = text[@"layer_types"];
        if (![raw_types isKindOfClass:NSArray.class] ||
            raw_types.count != info->num_layers) {
            ltx_st_free_header(&header);
            return ltx_weights_fail(error, error_size,
                                    "Gemma layer schedule mismatch");
        }
        layer_types = [raw_types copy];
    }
    if (info->vocab_size != 262144u || info->hidden_size != 3840u ||
        info->intermediate_size != 15360u || info->num_layers != 48u ||
        info->attention_heads != 16u || info->sliding_kv_heads != 8u ||
        info->full_kv_heads != 1u || info->sliding_head_dim != 256u ||
        info->full_head_dim != 512u || !info->attention_k_eq_v) {
        ltx_st_free_header(&header);
        return ltx_weights_fail(error, error_size,
                                "unsupported Gemma4 text geometry");
    }

    const uint32_t stacked_hidden =
        (info->num_layers + 1u) * info->hidden_size;
    info->projection_video_dim = 4096u;
    info->projection_audio_dim = 2048u;
    info->projection_input_dim = stacked_hidden;
    if (!ltx_gemma_require_tensor(
            &header, "model.embed_tokens.weight", LTX_DTYPE_BF16, 2u,
            info->vocab_size, info->hidden_size, error, error_size) ||
        !ltx_gemma_require_tensor(
            &header, "model.norm.weight", LTX_DTYPE_BF16, 1u,
            info->hidden_size, 0u, error, error_size) ||
        !ltx_gemma_require_tensor(
            &header, "text_embedding_projection.video_aggregate_embed.weight",
            LTX_DTYPE_BF16, 2u, info->projection_video_dim,
            stacked_hidden, error, error_size) ||
        !ltx_gemma_require_tensor(
            &header, "text_embedding_projection.video_aggregate_embed.bias",
            LTX_DTYPE_BF16, 1u, info->projection_video_dim, 0u,
            error, error_size) ||
        !ltx_gemma_require_tensor(
            &header, "text_embedding_projection.audio_aggregate_embed.weight",
            LTX_DTYPE_BF16, 2u, info->projection_audio_dim,
            stacked_hidden, error, error_size) ||
        !ltx_gemma_require_tensor(
            &header, "text_embedding_projection.audio_aggregate_embed.bias",
            LTX_DTYPE_BF16, 1u, info->projection_audio_dim, 0u,
            error, error_size)) {
        ltx_st_free_header(&header);
        return 0;
    }

    ltx_st_mapping mapping;
    if (!ltx_st_map_open(&header, &mapping, error, error_size)) {
        ltx_st_free_header(&header);
        return 0;
    }
    int ok = 1;
    int all_convrot = 1;
    char name[512];
    for (uint32_t layer = 0; layer < info->num_layers && ok; layer++) {
        NSString *type = layer_types[layer];
        int sliding = [type isEqual:@"sliding_attention"];
        if (!sliding && ![type isEqual:@"full_attention"]) {
            ok = ltx_weights_fail(error, error_size,
                                  "unknown Gemma attention layer type");
            break;
        }
        if (sliding != (layer % 6u != 5u)) {
            ok = ltx_weights_fail(error, error_size,
                                  "unexpected Gemma sliding/full layer schedule");
            break;
        }
        uint32_t head_dim =
            sliding ? info->sliding_head_dim : info->full_head_dim;
        uint32_t kv_heads =
            sliding ? info->sliding_kv_heads : info->full_kv_heads;
        uint32_t query_dim = info->attention_heads * head_dim;
        uint32_t key_value_dim = kv_heads * head_dim;
#define LTX_GEMMA_NAME(FORMAT, ...) \
        snprintf(name, sizeof(name), FORMAT, __VA_ARGS__)
#define LTX_GEMMA_VECTOR(SUFFIX, DIMENSION) \
        (LTX_GEMMA_NAME("model.layers.%u.%s", layer, (SUFFIX)), \
         ltx_gemma_require_tensor(&header, name, LTX_DTYPE_BF16, 1u, \
                                  (DIMENSION), 0u, error, error_size) != NULL)
#define LTX_GEMMA_LINEAR(SUFFIX, INPUT, OUTPUT) \
        (LTX_GEMMA_NAME("model.layers.%u.%s", layer, (SUFFIX)), \
         ltx_gemma_require_linear(&header, &mapping, name, (INPUT), (OUTPUT), \
                                  &all_convrot, error, error_size))
        ok = LTX_GEMMA_VECTOR("input_layernorm.weight", info->hidden_size) &&
             LTX_GEMMA_VECTOR("post_attention_layernorm.weight", info->hidden_size) &&
             LTX_GEMMA_VECTOR("pre_feedforward_layernorm.weight", info->hidden_size) &&
             LTX_GEMMA_VECTOR("post_feedforward_layernorm.weight", info->hidden_size) &&
             LTX_GEMMA_VECTOR("layer_scalar", 1u) &&
             LTX_GEMMA_VECTOR("self_attn.q_norm.weight", head_dim) &&
             LTX_GEMMA_VECTOR("self_attn.k_norm.weight", head_dim) &&
             LTX_GEMMA_LINEAR("self_attn.q_proj", info->hidden_size, query_dim) &&
             LTX_GEMMA_LINEAR("self_attn.k_proj", info->hidden_size, key_value_dim) &&
             LTX_GEMMA_LINEAR("self_attn.o_proj", query_dim, info->hidden_size) &&
             LTX_GEMMA_LINEAR("mlp.gate_proj", info->hidden_size,
                              info->intermediate_size) &&
             LTX_GEMMA_LINEAR("mlp.up_proj", info->hidden_size,
                              info->intermediate_size) &&
             LTX_GEMMA_LINEAR("mlp.down_proj", info->intermediate_size,
                              info->hidden_size);
        LTX_GEMMA_NAME("model.layers.%u.self_attn.v_proj.weight", layer);
        if (ok && sliding) {
            ok = LTX_GEMMA_LINEAR("self_attn.v_proj", info->hidden_size,
                                  key_value_dim);
        } else if (ok && ltx_st_find(&header, name)) {
            ok = ltx_weights_fail(error, error_size,
                                  "full Gemma attention unexpectedly has v_proj");
        }
#undef LTX_GEMMA_LINEAR
#undef LTX_GEMMA_VECTOR
#undef LTX_GEMMA_NAME
    }
    ltx_st_map_close(&mapping);
    if (ok) {
        info->quantized_int8 = 1;
        info->all_convrot_group_256 = all_convrot;
    }
    ltx_st_free_header(&header);
    return ok;
}

int ltx_gemma_checkpoint_validate(const ltx_gemma_checkpoint_info *info,
                                  char *error, size_t error_size) {
    if (!info)
        return ltx_weights_fail(error, error_size,
                                "missing Gemma checkpoint info");
    if (!info->tensor_count || info->num_layers != 48u ||
        info->vocab_size != 262144u || info->hidden_size != 3840u ||
        info->intermediate_size != 15360u ||
        info->projection_input_dim != 188160u ||
        info->projection_video_dim != 4096u ||
        info->projection_audio_dim != 2048u ||
        !info->attention_k_eq_v || !info->quantized_int8 ||
        !info->all_convrot_group_256)
        return ltx_weights_fail(error, error_size,
                                "unsupported or incomplete Gemma checkpoint");
    return 1;
}
