#include "ltx_streaming_layout.h"
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int fail(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return 0;
}

static int tensor_valid(const ltx_st_header *h, const ltx_st_tensor *t) {
    if (!h || !t || !t->name || !t->ndim || t->ndim > 8u ||
        t->data_end <= t->data_begin || t->file_offset > h->file_size)
        return 0;
    uint64_t bytes = ltx_dtype_size(t->dtype);
    if (!bytes) return 0;
    for (uint32_t i = 0; i < t->ndim; ++i) {
        if (!t->shape[i] || t->shape[i] > UINT64_MAX / bytes) return 0;
        bytes *= t->shape[i];
    }
    if (h->header_size > UINT64_MAX - 8u ||
        t->data_begin > UINT64_MAX - (h->header_size + 8u)) return 0;
    return bytes == t->data_end - t->data_begin && bytes <= SIZE_MAX &&
        t->file_offset == h->header_size + 8u + t->data_begin &&
        bytes <= h->file_size - t->file_offset;
}

static int add_bytes(uint64_t *total, uint64_t bytes) {
    if (bytes > UINT64_MAX - *total) return 0;
    *total += bytes;
    return 1;
}

static int field(ltx_stream_block_layout *out, const ltx_st_tensor *t,
                 uint32_t kind, uint32_t object, uint32_t part,
                 uint64_t bytes, char *error, size_t size) {
    if (!bytes || bytes > SIZE_MAX || out->field_count >= LTX_STREAM_MAX_FIELDS ||
        !add_bytes(kind == LTX_STREAM_TABLE_BASE ? &out->cpu_bytes : &out->gpu_bytes, bytes) ||
        (kind != LTX_STREAM_TABLE_ROW &&
         !add_bytes(&out->source_read_bytes, t->data_end - t->data_begin)))
        return fail(error, size, "LTX streaming field capacity overflow");
    out->fields[out->field_count++] = (ltx_stream_field){t, bytes, kind, object, part};
    return 1;
}

static const ltx_st_tensor *lookup(const ltx_st_header *h, const char *prefix,
                                 const char *suffix, char *error, size_t size) {
    char name[1024];
    int n = snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        fail(error, size, "LTX streaming tensor name overflow");
        return NULL;
    }
    const ltx_st_tensor *t = ltx_st_find(h, name);
    if (!tensor_valid(h, t)) {
        fail(error, size, "invalid LTX streaming tensor %s", name);
        return NULL;
    }
    return t;
}

static int linear(const ltx_st_header *h, const ltx_st_mapping *m,
                  const char *prefix, uint32_t index, ltx_stream_block_layout *out,
                  char *error, size_t size) {
    /* Check metadata range before the existing small-payload resolver reads it. */
    const ltx_st_tensor *metadata = lookup(h, prefix, "comfy_quant", error, size);
    if (!metadata || metadata->dtype != LTX_DTYPE_U8 || metadata->ndim != 1u ||
        metadata->data_end - metadata->data_begin > 4096u)
        return fail(error, size, "invalid LTX ConvRot metadata: %s", prefix);
    ltx_linear_weight_info *l = &out->linears[index];
    if (!ltx_linear_weight_resolve(h, m, prefix, l, error, size)) return 0;
    if (!l->quantized_int8 || !l->convrot || l->convrot_group_size != 256u ||
        !tensor_valid(h, l->weight) || l->weight->dtype != LTX_DTYPE_I8 ||
        !tensor_valid(h, l->weight_scale) || l->weight_scale->dtype != LTX_DTYPE_F32 ||
        (l->bias && (!tensor_valid(h, l->bias) || l->bias->dtype != LTX_DTYPE_BF16)))
        return fail(error, size, "unsupported streaming ConvRot INT8 linear: %s", prefix);
    const ltx_st_tensor *tensors[] = {l->weight, l->weight_scale, l->bias};
    for (uint32_t p = 0; p < 3; ++p)
        if (tensors[p] && !field(out, tensors[p], LTX_STREAM_LINEAR, index, p,
                                tensors[p]->data_end - tensors[p]->data_begin, error, size)) return 0;
    return add_bytes(&out->metadata_read_bytes, metadata->data_end - metadata->data_begin) ||
        fail(error, size, "LTX metadata byte count overflow");
}

static int attention(const ltx_st_header *h, const ltx_st_mapping *m,
                     const char *prefix, uint32_t index, ltx_stream_block_layout *out,
                     char *error, size_t size) {
    static const char *projection[] = {"to_q", "to_k", "to_v", "to_out.0"};
    static const char *aux[] = {"q_norm.weight", "k_norm.weight", "to_gate_logits.weight", "to_gate_logits.bias"};
    for (uint32_t p = 0; p < 4; ++p) {
        char name[1024];
        int n = snprintf(name, sizeof(name), "%s.%s", prefix, projection[p]);
        if (n < 0 || (size_t)n >= sizeof(name)) return fail(error, size, "LTX projection name overflow");
        if (!linear(h, m, name, index * 4u + p, out, error, size)) return 0;
    }
    const ltx_linear_weight_info *l = &out->linears[index * 4u];
    ltx_stream_attention_geometry *g = &out->attentions[index];
    *g = (ltx_stream_attention_geometry){l[0].input_dim, l[1].input_dim, l[0].output_dim, l[3].output_dim, 0, 0};
    if (l[1].output_dim != g->inner_dim || l[2].input_dim != g->key_value_dim ||
        l[2].output_dim != g->inner_dim || l[3].input_dim != g->inner_dim)
        return fail(error, size, "incompatible LTX projection geometry: %s", prefix);
    const ltx_st_tensor *t[4];
    for (uint32_t p = 0; p < 4; ++p) {
        t[p] = lookup(h, prefix, aux[p], error, size);
        if (!t[p] || t[p]->dtype != LTX_DTYPE_BF16) return fail(error, size, "invalid LTX norm/gate: %s", prefix);
    }
    if (t[0]->ndim != 1 || t[0]->shape[0] != g->inner_dim ||
        t[1]->ndim != 1 || t[1]->shape[0] != g->inner_dim ||
        t[2]->ndim != 2 || t[2]->shape[1] != g->query_dim ||
        t[2]->shape[0] > UINT32_MAX || t[3]->ndim != 1 || t[3]->shape[0] != t[2]->shape[0])
        return fail(error, size, "incompatible LTX norm/gate geometry: %s", prefix);
    g->heads = (uint32_t)t[2]->shape[0];
    if (!g->heads || g->inner_dim % g->heads) return fail(error, size, "invalid LTX head count: %s", prefix);
    g->head_dim = g->inner_dim / g->heads;
    for (uint32_t p = 0; p < 4; ++p)
        if (!field(out, t[p], LTX_STREAM_ATTENTION_AUX, index, p,
                   t[p]->data_end - t[p]->data_begin, error, size)) return 0;
    return 1;
}

int ltx_stream_describe_block(const ltx_st_header *h, const ltx_st_mapping *m,
                             uint32_t block, ltx_stream_block_layout *out,
                             char *error, size_t size) {
    if (!out) return fail(error, size, "missing LTX streaming layout output");
    memset(out, 0, sizeof(*out));
    if (!h || !m || !m->descriptor_open || m->descriptor < 0 ||
        m->bytes != h->file_size || block >= LTX_STREAM_BLOCKS)
        return fail(error, size, "invalid LTX streaming describe arguments");
    static const char *attentions[] = {"attn1", "audio_attn1", "attn2", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"};
    static const char *mlps[] = {"ff.net.0.proj", "ff.net.2", "audio_ff.net.0.proj", "audio_ff.net.2"};
    static const char *tables[] = {"scale_shift_table", "scale_shift_table", "audio_scale_shift_table",
        "prompt_scale_shift_table", "audio_prompt_scale_shift_table", "scale_shift_table_a2v_ca_video",
        "scale_shift_table_a2v_ca_video", "scale_shift_table_a2v_ca_audio"};
    static const uint32_t rows[] = {9, 9, 9, 2, 2, 5, 5, 5};
    char prefix[128], name[1024];
    snprintf(prefix, sizeof(prefix), "model.diffusion_model.transformer_blocks.%u", block);
    out->block = block;
    for (uint32_t a = 0; a < LTX_STREAM_ATTENTION_COUNT; ++a) {
        snprintf(name, sizeof(name), "%s.%s", prefix, attentions[a]);
        if (!attention(h, m, name, a, out, error, size)) goto failed;
    }
    for (uint32_t p = 0; p < 4; ++p) {
        snprintf(name, sizeof(name), "%s.%s", prefix, mlps[p]);
        if (!linear(h, m, name, 24u + p, out, error, size)) goto failed;
    }
    const ltx_stream_attention_geometry *a = out->attentions;
    const uint32_t vd = a[0].query_dim, ad = a[1].query_dim;
    const uint32_t qd[] = {vd, ad, vd, ad, vd, ad};
    const uint32_t kd[] = {vd, ad, vd, ad, ad, vd};
    for (uint32_t i = 0; i < LTX_STREAM_ATTENTION_COUNT; ++i)
        if (a[i].query_dim != qd[i] || a[i].key_value_dim != kd[i] || a[i].output_dim != qd[i]) {
            fail(error, size, "incompatible LTX block %u attention geometry", block);
            goto failed;
        }
    for (uint32_t i = 0; i < 2; ++i) {
        const ltx_linear_weight_info *l = &out->linears[24u + 2u * i];
        if (l[0].input_dim != (i ? ad : vd) || l[1].input_dim != l[0].output_dim || l[1].output_dim != l[0].input_dim) {
            fail(error, size, "incompatible LTX block %u MLP geometry", block);
            goto failed;
        }
    }
    for (uint32_t i = 0; i < LTX_STREAM_TABLE_COUNT; ++i) {
        uint32_t columns = (i == 2 || i == 4 || i == 7) ? ad : vd;
        const ltx_st_tensor *t = lookup(h, prefix, tables[i], error, size);
        if (!t || t->dtype != LTX_DTYPE_F32 || t->ndim != 2 || t->shape[0] != rows[i] || t->shape[1] != columns) {
            fail(error, size, "incompatible LTX block %u parameter table %u", block, i);
            goto failed;
        }
        out->table_rows[i] = rows[i]; out->table_columns[i] = columns;
        uint64_t read_bytes = t->data_end - t->data_begin;
        if (read_bytes > out->scratch_bytes) out->scratch_bytes = read_bytes;
        if (!field(out, t, LTX_STREAM_TABLE_BASE, i, 0, read_bytes / 2u, error, size)) goto failed;
        for (uint32_t row = 0; row < rows[i]; ++row)
            if (!field(out, t, LTX_STREAM_TABLE_ROW, i, row, (uint64_t)columns * 2u, error, size)) goto failed;
    }
    if (out->cpu_bytes > UINT64_MAX - out->gpu_bytes) {
        fail(error, size, "LTX block capacity overflow");
        goto failed;
    }
    return 1;
failed:
    memset(out, 0, sizeof(*out));
    return 0;
}

int ltx_stream_blocks_compatible(const ltx_stream_block_layout *a, const ltx_stream_block_layout *b) {
    if (!a || !b || !a->field_count || a->field_count != b->field_count ||
        a->field_count > LTX_STREAM_MAX_FIELDS ||
        a->gpu_bytes != b->gpu_bytes || a->cpu_bytes != b->cpu_bytes || a->scratch_bytes != b->scratch_bytes)
        return 0;
    for (uint32_t i = 0; i < LTX_STREAM_LINEAR_COUNT; ++i)
        if (a->linears[i].input_dim != b->linears[i].input_dim || a->linears[i].output_dim != b->linears[i].output_dim ||
            a->linears[i].convrot_group_size != b->linears[i].convrot_group_size) return 0;
    for (uint32_t i = 0; i < LTX_STREAM_ATTENTION_COUNT; ++i) {
        const ltx_stream_attention_geometry *x = &a->attentions[i], *y = &b->attentions[i];
        if (x->query_dim != y->query_dim || x->key_value_dim != y->key_value_dim || x->inner_dim != y->inner_dim ||
            x->output_dim != y->output_dim || x->heads != y->heads || x->head_dim != y->head_dim) return 0;
    }
    for (uint32_t i = 0; i < LTX_STREAM_TABLE_COUNT; ++i)
        if (a->table_rows[i] != b->table_rows[i] || a->table_columns[i] != b->table_columns[i]) return 0;
    for (uint32_t i = 0; i < a->field_count; ++i) {
        const ltx_stream_field *x = &a->fields[i], *y = &b->fields[i];
        if (x->bytes != y->bytes || x->kind != y->kind || x->object != y->object || x->part != y->part ||
            !x->source || !y->source || x->source->dtype != y->source->dtype || x->source->ndim != y->source->ndim ||
            x->source->ndim > 8u) return 0;
        for (uint32_t d = 0; d < x->source->ndim; ++d)
            if (x->source->shape[d] != y->source->shape[d]) return 0;
    }
    return 1;
}

static int read_tensor(const ltx_st_mapping *m, const ltx_st_tensor *t, void *data,
                       size_t bytes, ltx_stream_cancel_query cancelled, const void *user,
                       uint64_t *read_bytes, char *error, size_t size) {
    if (!t || t->data_end < t->data_begin || t->data_end - t->data_begin != bytes ||
        t->file_offset > m->bytes || bytes > m->bytes - t->file_offset ||
        t->file_offset > INT64_MAX || bytes > (uint64_t)INT64_MAX - t->file_offset)
        return fail(error, size, "invalid LTX streaming source range");
    size_t done = 0;
    while (done < bytes) {
        if (cancelled && cancelled(user)) return fail(error, size, "LTX streaming fill cancelled");
        size_t chunk = bytes - done;
        if (chunk > 8u * 1024u * 1024u) chunk = 8u * 1024u * 1024u;
        ssize_t n = pread(m->descriptor, (unsigned char *)data + done, chunk, (off_t)(t->file_offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return fail(error, size, "LTX streaming read %s: %s", t->name, n ? strerror(errno) : "unexpected EOF");
        done += (size_t)n;
        if (!add_bytes(read_bytes, (uint64_t)n)) return fail(error, size, "LTX read counter overflow");
    }
    return 1;
}

int ltx_stream_fill_block(const ltx_stream_block_layout *layout, const ltx_st_mapping *m,
                         void *const *destinations, const size_t *capacities,
                         void *scratch, size_t scratch_bytes,
                         ltx_stream_cancel_query cancelled, const void *user,
                         uint64_t *content_bytes, uint64_t *source_read_bytes,
                         char *error, size_t size) {
    if (content_bytes) *content_bytes = 0;
    if (source_read_bytes) *source_read_bytes = 0;
    if (!layout || !layout->field_count || layout->field_count > LTX_STREAM_MAX_FIELDS ||
        !m || !m->descriptor_open || m->descriptor < 0 || !destinations || !capacities ||
        !scratch || layout->scratch_bytes > scratch_bytes || !content_bytes || !source_read_bytes)
        return fail(error, size, "invalid LTX streaming fill arguments");
    /* Validate every destination before writing any of them. */
    for (uint32_t i = 0; i < layout->field_count; ++i)
        if (!destinations[i] || !layout->fields[i].bytes || layout->fields[i].bytes > capacities[i])
            return fail(error, size, "LTX streaming destination capacity mismatch at field %u", i);
    void *bases[LTX_STREAM_TABLE_COUNT] = {0};
    uint64_t total = 0;
    for (uint32_t i = 0; i < layout->field_count; ++i) {
        const ltx_stream_field *f = &layout->fields[i];
        if (cancelled && cancelled(user)) return fail(error, size, "LTX streaming fill cancelled");
        if (f->kind == LTX_STREAM_TABLE_ROW) {
            if (f->object >= LTX_STREAM_TABLE_COUNT || !bases[f->object] ||
                f->part >= layout->table_rows[f->object] || f->bytes != (uint64_t)layout->table_columns[f->object] * 2u)
                return fail(error, size, "invalid LTX streaming table row binding");
            memcpy(destinations[i], (const unsigned char *)bases[f->object] + (size_t)f->part * (size_t)f->bytes, (size_t)f->bytes);
        } else if (f->kind == LTX_STREAM_TABLE_BASE) {
            if (f->object >= LTX_STREAM_TABLE_COUNT || f->bytes > scratch_bytes / 2u ||
                !layout->table_rows[f->object] || layout->table_rows[f->object] > 9u ||
                f->bytes != (uint64_t)layout->table_columns[f->object] * layout->table_rows[f->object] * 2u)
                return fail(error, size, "invalid LTX streaming parameter table binding");
            if (!read_tensor(m, f->source, scratch, (size_t)f->bytes * 2u, cancelled, user, source_read_bytes, error, size)) return 0;
            for (size_t n = 0; n < (size_t)f->bytes / 2u; ++n) {
                uint32_t bits;
                memcpy(&bits, (const unsigned char *)scratch + n * 4u, sizeof(bits));
                bits += 0x7fffu + ((bits >> 16u) & 1u);
                uint16_t value = (uint16_t)(bits >> 16u);
                memcpy((unsigned char *)destinations[i] + n * 2u, &value, sizeof(value));
            }
            bases[f->object] = destinations[i];
        } else if (f->kind == LTX_STREAM_LINEAR || f->kind == LTX_STREAM_ATTENTION_AUX) {
            if (!read_tensor(m, f->source, destinations[i], (size_t)f->bytes, cancelled, user, source_read_bytes, error, size)) return 0;
        } else return fail(error, size, "unknown LTX streaming field kind");
        if (!add_bytes(&total, f->bytes)) return fail(error, size, "LTX content byte count overflow");
    }
    if (cancelled && cancelled(user)) return fail(error, size, "LTX streaming fill cancelled");
    if (total != layout->gpu_bytes + layout->cpu_bytes || *source_read_bytes != layout->source_read_bytes)
        return fail(error, size, "LTX streaming content/read byte mismatch");
    *content_bytes = total;
    return 1;
}
