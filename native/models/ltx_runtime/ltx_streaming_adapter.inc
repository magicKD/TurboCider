/* Private bridge included by ltx_blocks.c to share the existing numerical
 * helpers without exporting or copying kernels. No legacy caller uses it. */
typedef struct ltx_exact_stream {
    ltx_stream_block_layout layouts[LTX_STREAM_BLOCKS];
    ltx_stream_slot slots[LTX_MAX_REFILL_SLOTS];
    block_weights views[LTX_MAX_REFILL_SLOTS]; /* borrowed; never free_block_weights */
    tc_stream_executor *executor;
    pthread_t owner;
    ltx_gpu *gpu, *audio_gpu;
    const ltx_st_mapping *mapping;
    block_weights *prefix;
    uint32_t prefix_count, slot_count, pass, expected_passes;
    uint64_t queue_sequence, source_bytes[LTX_MAX_REFILL_SLOTS];
    int finished, poisoned;
    /* Error-only retention until a later owner drain proves storage safe. */
    ltx_gpu_buffer *quarantine_buffers[24];
    void *quarantine_hosts[4];
    uint32_t quarantine_buffer_count;
    block_workspace quarantine_workspace;
    block_rope quarantine_rope;
    /* Borrowed only during run_pass; all model-side readers join before return. */
    ltx_transformer_conditioning_values *conditioning;
    const block_rope *rope;
    block_workspace *workspace;
    const ltx_gpu_buffer *video_text, *audio_text, *text_mask;
    uint32_t video_rows, audio_rows, text_rows;
    block_timing *timing;
} ltx_exact_stream;

static int ltx_exact_owner(const ltx_exact_stream *s, char *error, size_t size) {
    if (pthread_equal(s->owner, pthread_self())) return 1;
    if (error && size) snprintf(error, size, "LTX exact owner thread violation");
    return 0;
}

static void ltx_exact_bind_slot(ltx_stream_slot *slot, block_weights *w) {
    memset(w, 0, sizeof(*w));
    attention_weights *attn[] = {&w->video_self, &w->audio_self, &w->video_text,
        &w->audio_text, &w->audio_to_video, &w->video_to_audio};
    gpu_linear *linears[LTX_STREAM_LINEAR_COUNT];
    for (unsigned i = 0; i < 6; ++i) {
        linears[i * 4] = &attn[i]->query; linears[i * 4 + 1] = &attn[i]->key;
        linears[i * 4 + 2] = &attn[i]->value; linears[i * 4 + 3] = &attn[i]->output;
        const ltx_stream_attention_geometry *g = &slot->construction->attentions[i];
        attn[i]->query_dim = g->query_dim; attn[i]->key_value_dim = g->key_value_dim;
        attn[i]->inner_dim = g->inner_dim; attn[i]->output_dim = g->output_dim;
        attn[i]->heads = g->heads; attn[i]->head_dim = g->head_dim;
    }
    linears[24] = &w->video_mlp.fc1; linears[25] = &w->video_mlp.fc2;
    linears[26] = &w->audio_mlp.fc1; linears[27] = &w->audio_mlp.fc2;
    for (unsigned i = 0; i < LTX_STREAM_LINEAR_COUNT; ++i) {
        const ltx_linear_weight_info *l = &slot->construction->linears[i];
        linears[i]->input_dim = l->input_dim; linears[i]->output_dim = l->output_dim;
        linears[i]->weight_bytes = (size_t)(l->weight->data_end - l->weight->data_begin);
    }
    mlp_weights *mlps[] = {&w->video_mlp, &w->audio_mlp};
    for (unsigned i = 0; i < 2; ++i) {
        mlps[i]->input_dim = mlps[i]->fc1.input_dim;
        mlps[i]->hidden_dim = mlps[i]->fc1.output_dim;
        mlps[i]->output_dim = mlps[i]->fc2.output_dim;
    }
    parameter_table *tables[] = {&w->video_adaln, &w->video_adaln_conditioned, &w->audio_adaln,
        &w->video_prompt, &w->audio_prompt, &w->av_video, &w->av_video_conditioned, &w->av_audio};
    for (unsigned i = 0; i < LTX_STREAM_TABLE_COUNT; ++i) {
        tables[i]->rows = slot->construction->table_rows[i];
        tables[i]->columns = slot->construction->table_columns[i];
    }
    for (unsigned i = 0; i < slot->construction->field_count; ++i) {
        const ltx_stream_field *f = &slot->construction->fields[i];
        if (f->kind == LTX_STREAM_LINEAR) {
            gpu_linear *l = linears[f->object];
            if (f->part == 0) l->weight = slot->buffers[i];
            else if (f->part == 1) l->scale = slot->buffers[i];
            else l->bias = slot->buffers[i];
        } else if (f->kind == LTX_STREAM_ATTENTION_AUX) {
            attention_weights *a = attn[f->object];
            if (f->part == 0) a->query_norm = slot->buffers[i];
            else if (f->part == 1) a->key_norm = slot->buffers[i];
            else if (f->part == 2) a->gate_weight = slot->buffers[i];
            else a->gate_bias = slot->buffers[i];
        } else if (f->kind == LTX_STREAM_TABLE_BASE) tables[f->object]->base_values = slot->destinations[i];
        else tables[f->object]->row[f->part] = slot->buffers[i];
    }
}

static int ltx_exact_allocate(void *user, uint32_t index, uint64_t capacity, char *error, size_t size) {
    ltx_exact_stream *s = user;
    if (index >= s->slot_count) { snprintf(error, size, "invalid LTX exact slot index"); return 0; }
    const ltx_stream_block_layout *layout = &s->layouts[s->prefix_count + index];
    if (capacity != layout->gpu_bytes + layout->cpu_bytes) {
        snprintf(error, size, "LTX actual slot capacity differs from compiled plan"); return 0;
    }
    if (!ltx_stream_slot_create(s->gpu, layout, &s->slots[index], error, size)) return 0;
    ltx_exact_bind_slot(&s->slots[index], &s->views[index]);
    if (block_resident_bytes(&s->views[index]) != capacity) {
        snprintf(error, size, "LTX bound slot bytes differ from metadata"); return 0;
    }
    return 1;
}

static void ltx_exact_destroy_pool(void *user) {
    ltx_exact_stream *s = user;
    for (unsigned i = 0; i < LTX_MAX_REFILL_SLOTS; ++i) {
        ltx_stream_slot_destroy(&s->slots[i]);
        memset(&s->views[i], 0, sizeof(s->views[i]));
    }
}

static int ltx_exact_fill(void *user, const tc_stream_slot_ticket_v1 *ticket,
                          const tc_stream_group_v1 *group, tc_stream_cancel_query_v1 cancel,
                          const void *cancel_user, uint64_t *bytes, char *error, size_t size) {
    ltx_exact_stream *s = user;
    if (ticket->slot >= s->slot_count || group->block_count != 1 || group->blocks[0] >= LTX_STREAM_BLOCKS) {
        snprintf(error, size, "invalid LTX exact fill ticket"); return 0;
    }
    ltx_stream_slot *slot = &s->slots[ticket->slot];
    uint64_t read = 0;
    int ok = ltx_stream_fill_block(&s->layouts[group->blocks[0]], s->mapping,
        slot->destinations, slot->capacities, slot->scratch, slot->scratch_bytes,
        cancel, cancel_user, bytes, &read, error, size);
    if (read > UINT64_MAX - s->source_bytes[ticket->slot]) {
        snprintf(error, size, "LTX source read counter overflow"); return 0;
    }
    s->source_bytes[ticket->slot] += read;
    return ok;
}

static int ltx_exact_compute(ltx_exact_stream *s, block_weights *w, uint32_t block, char *error, size_t size) {
    return !report_progress("ltx_block", (int)block, LTX_STREAM_BLOCKS, error, size) &&
        run_block(s->gpu, w, block, s->rope, s->workspace, s->video_text, s->audio_text,
                  s->text_mask, s->video_rows, s->audio_rows, s->text_rows, s->timing, error, size);
}

static int ltx_exact_prefix(void *user, uint32_t pass, char *error, size_t size) {
    ltx_exact_stream *s = user;
    if (pass != s->pass || !s->workspace) { snprintf(error, size, "unbound LTX pass"); return 0; }
    for (uint32_t i = 0; i < s->prefix_count; ++i)
        if (!ltx_exact_compute(s, &s->prefix[i], i, error, size)) return 0;
    return 1;
}

static int ltx_exact_prepare(void *user, const tc_stream_slot_ticket_v1 *ticket,
                             const tc_stream_group_v1 *group, char *error, size_t size) {
    (void)group;
    ltx_exact_stream *s = user;
    if (!s->conditioning || ticket->slot >= s->slot_count) { snprintf(error, size, "unbound LTX conditioning"); return 0; }
    block_weights *w = &s->views[ticket->slot];
    ltx_transformer_conditioning_values *v = s->conditioning;
    parameter_table *tables[] = {&w->video_adaln, &w->audio_adaln, &w->video_prompt, &w->audio_prompt,
                                 &w->av_video, &w->av_video, &w->av_audio, &w->av_audio};
    const uint16_t *parameters[] = {v->video_adaln, v->audio_adaln, v->video_prompt, v->audio_prompt,
                                    v->av_video, v->a2v_gate, v->av_audio, v->v2a_gate};
    const unsigned first[] = {0, 0, 0, 0, 0, 4, 0, 4}, counts[] = {9, 9, 2, 2, 4, 1, 4, 1};
    ltx_stream_slot *slot = &s->slots[ticket->slot];
    uint16_t *row_values = slot->scratch;
    for (unsigned i = 0; i < 8; ++i) {
        parameter_table *table = tables[i];
        size_t row_bytes = (size_t)table->columns * sizeof(uint16_t);
        if (!parameters[i] || row_bytes > slot->scratch_bytes) { snprintf(error, size, "LTX conditioning scratch mismatch"); return 0; }
        for (unsigned row = 0; row < counts[i]; ++row) {
            const uint16_t *base = table->base_values + (size_t)(first[i] + row) * table->columns;
            for (uint32_t c = 0; c < table->columns; ++c)
                row_values[c] = f32_to_bf16(bf16_to_f32(base[c]) + bf16_to_f32(parameters[i][(size_t)row * table->columns + c]));
            if (!ltx_gpu_buffer_write(table->row[first[i] + row], row_values, row_bytes, error, size)) return 0;
        }
    }
    return 1;
}

static int ltx_exact_encode(void *user, const tc_stream_slot_ticket_v1 *ticket,
                            const tc_stream_group_v1 *group, const tc_stream_completion_sink_v1 *sink,
                            tc_stream_reader_set_v1 *readers, char *error, size_t size) {
    ltx_exact_stream *s = user;
    if (!ltx_exact_compute(s, &s->views[ticket->slot], group->blocks[0], error, size)) return 0;
    /* run_block joins its audio tasks. Both finish helpers wait unless a batch
     * is active; all supported model batches end inside run_block. Check this
     * contract explicitly: no additional device-wide wait or marker command. */
    if (!ltx_gpu_streaming_boundary_completed(s->gpu) ||
        (s->audio_gpu && !ltx_gpu_streaming_boundary_completed(s->audio_gpu)) ||
        s->queue_sequence == UINT64_MAX) {
        snprintf(error, size, "LTX exact block has an unproven reader boundary"); return 0;
    }
    ++s->queue_sequence;
    readers->count = s->audio_gpu ? 2u : 1u;
    readers->fences[0] = (tc_stream_reader_fence_v1){LTX_GPU_MEMORY_QUEUE_VIDEO, s->queue_sequence};
    readers->fences[1] = (tc_stream_reader_fence_v1){LTX_GPU_MEMORY_QUEUE_AUDIO, s->queue_sequence};
    for (uint32_t i = 0; i < readers->count; ++i)
        if (!sink->post(sink->user, ticket, readers->fences[i], 0)) {
            snprintf(error, size, "LTX completion mailbox overflow"); return 0;
        }
    return 1;
}

static int ltx_exact_drain(void *user, char *error, size_t size) {
    ltx_exact_stream *s = user;
    int video = s->gpu ? ltx_gpu_drain(s->gpu, error, size) : 1;
    int audio = s->audio_gpu ? ltx_gpu_drain(s->audio_gpu, error, size) : 1;
    return video && audio;
}

static int ltx_exact_release_safe(ltx_exact_stream *s) {
    char detail[1024] = {0};
    return ltx_exact_drain(s, detail, sizeof(detail));
}

static void ltx_exact_keep_buffer(ltx_exact_stream *s, ltx_gpu_buffer *buffer) {
    if (!buffer) return;
    /* At most seven sampler and ten stage buffers; only one failing run can
     * enter quarantine because poisoned blocks all subsequent runs. */
    if (s->quarantine_buffer_count >= 24u) abort();
    s->quarantine_buffers[s->quarantine_buffer_count++] = buffer;
}

static void ltx_exact_release_quarantine(ltx_exact_stream *s) {
    for (uint32_t i = 0; i < s->quarantine_buffer_count; ++i)
        ltx_gpu_buffer_free(s->quarantine_buffers[i]);
    s->quarantine_buffer_count = 0;
    for (uint32_t i = 0; i < 4; ++i) { free(s->quarantine_hosts[i]); s->quarantine_hosts[i] = NULL; }
    free_workspace(&s->quarantine_workspace);
    free_block_rope(&s->quarantine_rope);
}

/* Structural validation is shared by v1/v2 and runs before metadata I/O or GPU
 * construction. It never treats an invalid ABI as a best-effort legacy plan. */
static int ltx_exact_validate_options(const ltx_native_options *options,
                                      const ltx_native_streaming_options_v1 *o,
                                      char *error, size_t size) {
    if (!options || !o || o->struct_size != sizeof(*o) || o->version != 1u) {
        snprintf(error, size, "invalid LTX exact options ABI"); return 0;
    }
    if (!options->checkpoint || !options->checkpoint[0] || options->fps != 24u ||
        options->memory_budget_bytes || options->max_refill_slots || options->memory_hooks ||
        options->schedule_hooks || options->sol_stage1 || options->sol_stage2 || options->sparse_mode ||
        options->preload_ane_stage2 || options->release_full_gpu_mlp ||
        options->detach_ane_stage1 || options->detach_ane_stage2 || options->release_blocks_final_step ||
        options->mlp_directories[0] || options->mlp_directories[1] ||
        options->v2a_directories[0] || options->v2a_directories[1] || options->kv_directory ||
        options->qkv_directories[0] || options->qkv_directories[1] ||
        options->ane_mlp_fused_residual || options->ane_mlp_fused_adaln_pack) {
        snprintf(error, size, "invalid/unsupported LTX exact layout options"); return 0;
    }
    const tc_stream_stage_plan_v1 *p = o->plan;
    if (!p || p->struct_size != sizeof(*p) || p->version != TC_STREAM_SLOT_ABI_V1 ||
        !p->request_generation || !p->slot_count || p->slot_count > LTX_MAX_REFILL_SLOTS ||
        !p->groups || !p->slot_capacity_bytes || !p->io_workers || p->io_workers > p->slot_count ||
        p->prefetch_distance >= p->slot_count || !o->resident_prefix_blocks ||
        o->resident_prefix_blocks >= LTX_STREAM_BLOCKS ||
        p->group_count != LTX_STREAM_BLOCKS - o->resident_prefix_blocks || p->slot_count > p->group_count) {
        snprintf(error, size, "invalid LTX exact plan ABI/fields"); return 0;
    }
    size_t stage1, stage2;
    (void)ltx_distilled_stage1_sigmas(&stage1); (void)ltx_distilled_stage2_sigmas(&stage2);
    if (stage1 < 2 || stage2 < 2 || p->pass_count != stage1 + stage2 - 2u) {
        snprintf(error, size, "LTX exact plan has wrong pass count"); return 0;
    }
    uint64_t pool_bytes = 0, pass_bytes = 0;
    for (uint32_t i = 0; i < p->slot_count; ++i) {
        uint64_t bytes = p->slot_capacity_bytes[i];
        if (!bytes || bytes > SIZE_MAX || bytes > UINT64_MAX - pool_bytes) {
            snprintf(error, size, "invalid LTX exact slot capacity/overflow"); return 0;
        }
        pool_bytes += bytes;
    }
    for (uint32_t i = 0; i < p->group_count; ++i) {
        const tc_stream_group_v1 *g = &p->groups[i];
        if (g->group != i || g->slot != i % p->slot_count || g->block_count != 1 || !g->blocks ||
            g->blocks[0] != o->resident_prefix_blocks + i || !g->content_bytes ||
            g->content_bytes > p->slot_capacity_bytes[g->slot] ||
            g->content_bytes > UINT64_MAX - pass_bytes) {
            snprintf(error, size, "invalid LTX exact group/binding at %u", i); return 0;
        }
        pass_bytes += g->content_bytes;
    }
    if (pass_bytes > UINT64_MAX / p->pass_count) {
        snprintf(error, size, "LTX exact content counter overflow"); return 0;
    }
    return 1;
}

/* Metadata-dependent geometry/capacity check, after the structural validator. */
static int ltx_exact_validate_plan(const ltx_native_streaming_options_v1 *o,
                                   const ltx_stream_block_layout *layouts, char *error, size_t size) {
    const tc_stream_stage_plan_v1 *p = o->plan;
    for (uint32_t i = 0; i < LTX_STREAM_BLOCKS; ++i)
        if (!ltx_stream_blocks_compatible(&layouts[0], &layouts[i])) {
            snprintf(error, size, "LTX exact plan requires homogeneous block geometry"); return 0;
        }
    const uint64_t bytes = layouts[0].gpu_bytes + layouts[0].cpu_bytes;
    for (uint32_t i = 0; i < p->slot_count; ++i)
        if (p->slot_capacity_bytes[i] != bytes) { snprintf(error, size, "LTX exact capacity mismatch"); return 0; }
    for (uint32_t i = 0; i < p->group_count; ++i) {
        const tc_stream_group_v1 *g = &p->groups[i];
        if (g->group != i || g->slot != i % p->slot_count || g->block_count != 1 || !g->blocks ||
            g->blocks[0] != o->resident_prefix_blocks + i || g->content_bytes != bytes) {
            snprintf(error, size, "LTX exact group/binding mismatch at %u", i); return 0;
        }
    }
    return 1;
}

static int ltx_exact_begin(ltx_exact_stream *s, const tc_stream_stage_plan_v1 *plan, char *error, size_t size) {
    tc_stream_adapter_v1 ops = {sizeof(ops), TC_STREAM_SLOT_ABI_V1, s,
        ltx_exact_allocate, ltx_exact_destroy_pool, ltx_exact_fill, ltx_exact_prefix,
        ltx_exact_prepare, ltx_exact_encode, ltx_exact_drain};
    return tc_stream_executor_create_v1(plan, &ops, &s->executor, error, size);
}

static int run_exact_block_stack(ltx_exact_stream *s, uint32_t step,
    ltx_transformer_conditioning_values *conditioning, const block_rope *rope, block_workspace *workspace,
    const ltx_gpu_buffer *video_text, const ltx_gpu_buffer *audio_text, const ltx_gpu_buffer *mask,
    uint32_t video_rows, uint32_t audio_rows, uint32_t text_rows, block_timing *timing, char *error, size_t size) {
    if (s->finished || s->poisoned) { snprintf(error, size, "LTX exact request already finished/failed"); return 0; }
    s->conditioning = conditioning; s->rope = rope; s->workspace = workspace;
    s->video_text = video_text; s->audio_text = audio_text; s->text_mask = mask;
    s->video_rows = video_rows; s->audio_rows = audio_rows; s->text_rows = text_rows; s->timing = timing;
    int ok = tc_stream_executor_run_pass(s->executor, s->pass, step, error, size);
    if (ok) ++s->pass;
    else s->poisoned = 1;
    s->conditioning = NULL; s->rope = NULL; s->workspace = NULL;
    s->video_text = NULL; s->audio_text = NULL; s->text_mask = NULL; s->timing = NULL;
    return ok;
}
