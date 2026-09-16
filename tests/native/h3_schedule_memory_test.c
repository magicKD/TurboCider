#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Include the implementation so the test can construct the otherwise opaque
 * schedule and exercise its host-accounted readback path without a Metal
 * device or a checkpoint fixture. External GPU/weight operations are stubbed
 * below; the production source itself is what is under test. */
#include "../../native/models/h3_runtime/h3_dit_schedule.c"

struct h3_gpu {
    int unused;
};

struct h3_gpu_tensor {
    uint16_t *values;
    size_t elements;
};

struct h3_weight_store {
    int unused;
};

typedef struct {
    uint64_t reserved;
    uint64_t committed;
    uint64_t reserve_calls;
    uint64_t commit_calls;
    uint64_t cancel_calls;
    uint64_t release_calls;
    uint64_t last_upper;
    uint32_t last_class;
    const char *last_tag;
    int reject_reserve;
} test_memory_state;

typedef struct {
    test_memory_state *state;
    uint64_t upper;
    uint64_t actual;
    int committed;
} test_memory_token;

static int fail_readback;

static int test_reserve(void *opaque, uint32_t memory_class,
                        uint64_t upper_bytes, const char *tag,
                        void **token, char *error, size_t error_size) {
    test_memory_state *state = opaque;
    if (token) *token = NULL;
    if (!state || !token || !upper_bytes || state->reject_reserve) {
        if (error && error_size)
            snprintf(error, error_size, "test reserve rejected");
        return 0;
    }
    test_memory_token *value = calloc(1, sizeof(*value));
    if (!value) return 0;
    value->state = state;
    value->upper = upper_bytes;
    state->reserved += upper_bytes;
    state->reserve_calls++;
    state->last_upper = upper_bytes;
    state->last_class = memory_class;
    state->last_tag = tag;
    *token = value;
    return 1;
}

static int test_commit(void *opaque, void *opaque_token,
                       uint64_t allocator_domain, uint64_t handle,
                       uint64_t actual_bytes, uint64_t generation,
                       char *error, size_t error_size) {
    (void)opaque;
    test_memory_token *token = opaque_token;
    if (!token || token->committed || !allocator_domain || !handle ||
        !generation || !actual_bytes || actual_bytes > token->upper) {
        if (error && error_size)
            snprintf(error, error_size, "test commit rejected");
        return 0;
    }
    assert(token->state->reserved >= token->upper);
    token->state->reserved -= token->upper;
    token->state->committed += actual_bytes;
    token->state->commit_calls++;
    token->actual = actual_bytes;
    token->committed = 1;
    return 1;
}

static void test_cancel(void *opaque, void *opaque_token) {
    (void)opaque;
    test_memory_token *token = opaque_token;
    if (!token) return;
    assert(!token->committed);
    assert(token->state->reserved >= token->upper);
    token->state->reserved -= token->upper;
    token->state->cancel_calls++;
    free(token);
}

static void test_release(void *opaque, void *opaque_token) {
    (void)opaque;
    test_memory_token *token = opaque_token;
    assert(token && token->committed);
    assert(token->state->committed >= token->actual);
    token->state->committed -= token->actual;
    token->state->release_calls++;
    free(token);
}

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static h3_gpu_tensor *make_tensor(size_t elements) {
    h3_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    assert(tensor);
    tensor->values = calloc(elements, sizeof(*tensor->values));
    assert(tensor->values);
    tensor->elements = elements;
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_new_classified(
        h3_gpu *gpu, size_t elements, h3_gpu_dtype dtype,
        h3_gpu_memory_class memory_class, const char *tag) {
    (void)gpu;
    (void)dtype;
    (void)memory_class;
    (void)tag;
    return make_tensor(elements);
}

h3_gpu_tensor *h3_gpu_tensor_from_f32_classified(
        h3_gpu *gpu, const float *values, size_t elements,
        h3_gpu_memory_class memory_class, const char *tag) {
    (void)gpu;
    (void)memory_class;
    (void)tag;
    h3_gpu_tensor *tensor = make_tensor(elements);
    for (size_t index = 0; index < elements; index++)
        tensor->values[index] = bf16(values[index]);
    return tensor;
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    free(tensor->values);
    free(tensor);
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    if (fail_readback || !tensor || !values || elements != tensor->elements)
        return 0;
    memcpy(values, tensor->values, elements * sizeof(*values));
    return 1;
}

int h3_gpu_begin(h3_gpu *gpu) {
    (void)gpu;
    return 1;
}

int h3_gpu_submit(h3_gpu *gpu) {
    (void)gpu;
    return 1;
}

const char *h3_gpu_error(const h3_gpu *gpu) {
    (void)gpu;
    return "test GPU error";
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    (void)gpu; (void)output; (void)input; (void)weight; (void)bias;
    (void)rows; (void)input_dim; (void)output_dim;
    return 1;
}

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    (void)gpu; (void)output; (void)input; (void)elements;
    return 1;
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    (void)gpu; (void)output; (void)input; (void)elements;
    return 1;
}

int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    (void)gpu; (void)output; (void)input; (void)weight; (void)bias;
    (void)rows; (void)input_dim; (void)output_dim;
    return 1;
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    (void)gpu; (void)output; (void)input; (void)elements;
    return 1;
}

static size_t shape_elements(int ndim, const uint64_t *shape) {
    size_t result = 1;
    for (int index = 0; index < ndim; index++) result *= (size_t)shape[index];
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(
        const h3_weight_store *store, h3_gpu *gpu, const char *name,
        int ndim, const uint64_t *shape, char *error, size_t error_size) {
    (void)store; (void)gpu; (void)name; (void)error; (void)error_size;
    return make_tensor(shape_elements(ndim, shape));
}

h3_gpu_tensor *h3_weight_load_f32(
        const h3_weight_store *store, h3_gpu *gpu, const char *name,
        int ndim, const uint64_t *shape, char *error, size_t error_size) {
    (void)store; (void)gpu; (void)name; (void)error; (void)error_size;
    return make_tensor(shape_elements(ndim, shape));
}

static void fill_gate_values(h3_gpu_tensor *tensor) {
    assert(tensor && tensor->elements == BLOCK_OUTPUT);
    for (uint32_t modality = 0; modality < H3_DIT_MODALITIES; modality++)
        for (uint32_t slot = 2; slot <= 5; slot += 3) {
            size_t base = ((size_t)modality * H3_DIT_ADALN_SLOTS + slot) *
                          H3_DIT_HIDDEN;
            for (uint32_t column = 0; column < H3_DIT_HIDDEN; column++)
                tensor->values[base + column] = bf16(1.0f);
        }
}

int main(void) {
    test_memory_state state = {0};
    h3_host_memory_hooks hooks = {
        sizeof(hooks), 1u, &state, test_reserve, test_commit,
        test_cancel, test_release};
    h3_dit_schedule schedule = {0};
    schedule.host_memory = (h3_host_memory_options){
        sizeof(schedule.host_memory), 1u, &hooks, 19u, 23u};
    schedule.steps = 1;
    schedule.time_rows = 1;
    uint32_t video_rows[] = {0};
    uint32_t audio_rows[] = {0};
    schedule.video_rows = video_rows;
    schedule.audio_rows = audio_rows;
    schedule.blocks[0] = make_tensor(BLOCK_OUTPUT);
    fill_gate_values(schedule.blocks[0]);

    assert(h3_dit_schedule_gate_score(&schedule, 0) == 1.0);
    assert(state.reserve_calls == 1 && state.commit_calls == 1 &&
           state.release_calls == 1);
    assert(state.reserved == 0 && state.committed == 0);
    assert(state.last_class == H3_HOST_MEMORY_STAGING);
    assert(strcmp(state.last_tag, "h3.schedule.gate_readback") == 0);
    assert(state.last_upper == (uint64_t)BLOCK_OUTPUT * sizeof(uint16_t));

    double scores[H3_DIT_MODALITIES] = {0};
    assert(h3_dit_schedule_gate_scores(
        &schedule, 0, scores, H3_DIT_MODALITIES));
    for (size_t index = 0; index < H3_DIT_MODALITIES; index++)
        assert(scores[index] == 1.0);

    double branch_scores[H3_DIT_MODALITIES * 2] = {0};
    assert(h3_dit_schedule_gate_branch_scores(
        &schedule, 0, branch_scores, H3_DIT_MODALITIES * 2));
    for (size_t index = 0; index < H3_DIT_MODALITIES * 2; index++)
        assert(branch_scores[index] == 1.0);
    assert(state.reserve_calls == 3 && state.commit_calls == 3 &&
           state.release_calls == 3);

    fail_readback = 1;
    assert(h3_dit_schedule_gate_score(&schedule, 0) < 0.0);
    assert(state.reserve_calls == 4 && state.commit_calls == 4 &&
           state.release_calls == 4);
    assert(state.reserved == 0 && state.committed == 0);
    fail_readback = 0;

    state.reject_reserve = 1;
    assert(!h3_dit_schedule_gate_scores(
        &schedule, 0, scores, H3_DIT_MODALITIES));
    assert(state.reserve_calls == 4 && state.committed == 0 &&
           state.reserved == 0);

    h3_gpu_tensor_free(schedule.blocks[0]);
    puts("H3 schedule memory tests passed");
    return 0;
}
