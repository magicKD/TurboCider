#include "../../native/core/stream_slot_c.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
struct Model {
    bool standard_exception = false;
    bool safe_to_destroy = false;
    unsigned allocations = 0;
    unsigned destroys = 0;
};

int allocate(void *user, uint32_t, uint64_t, char *, size_t) {
    auto &model = *static_cast<Model *>(user);
    ++model.allocations;
    if (model.standard_exception)
        throw std::runtime_error("injected allocation failure");
    throw 42; // Defensively exercise catch(...) at the C bridge boundary.
}

void destroy(void *user) {
    auto &model = *static_cast<Model *>(user);
    assert(model.safe_to_destroy);
    ++model.destroys;
}

int fill(void *, const tc_stream_slot_ticket_v1 *, const tc_stream_group_v1 *,
         tc_stream_cancel_query_v1, const void *, uint64_t *, char *, size_t) {
    assert(false && "failed construction must not dispatch fills");
    return 0;
}

int prefix(void *, uint32_t, char *, size_t) { return 1; }
int prepare(void *, const tc_stream_slot_ticket_v1 *, const tc_stream_group_v1 *,
            char *, size_t) { return 1; }
int encode(void *, const tc_stream_slot_ticket_v1 *, const tc_stream_group_v1 *,
           const tc_stream_completion_sink_v1 *, tc_stream_reader_set_v1 *,
           char *, size_t) { return 1; }
int drain(void *user, char *, size_t) {
    return static_cast<Model *>(user)->safe_to_destroy ? 1 : 0;
}

int create(uint32_t version, Model &model, tc_stream_executor **executor,
           char *error, size_t size) {
    const uint32_t block = 0;
    const uint64_t capacity = 8;
    if (version == TC_STREAM_SLOT_ABI_V2) {
        const tc_stream_group_v2 group{0, 0, 0, 1, &block, capacity};
        const tc_stream_pool_plan_v2 pool{sizeof(pool), version, 0, 1, &capacity};
        const tc_stream_stage_plan_v2 plan{
            sizeof(plan), version, 1, 1, 1, 0, 1, 1, 1, &pool, 1, &group};
        const tc_stream_adapter_v2 ops{
            sizeof(ops), version, &model,
            [](void *user, uint32_t, uint32_t slot, uint64_t bytes, char *detail, size_t count) {
                return allocate(user, slot, bytes, detail, count);
            },
            [](void *user, uint32_t) { destroy(user); },
            fill, prefix, prepare, encode, drain};
        return tc_stream_executor_create_v2(&plan, &ops, executor, error, size);
    }
    const tc_stream_group_v1 group{0, 0, 1, &block, capacity};
    const tc_stream_adapter_v1 ops{
        sizeof(ops), TC_STREAM_SLOT_ABI_V1, &model, allocate, destroy,
        fill, prefix, prepare, encode, drain};
    if (version == TC_STREAM_SLOT_ABI_V3) {
        const tc_stream_stage_plan_v3 plan{
            sizeof(plan), version, 1, 0, 1, 0, 1, 1, TC_STREAM_PASS_RELOAD_V3,
            1, &capacity, 1, &group};
        return tc_stream_executor_create_v3(&plan, &ops, executor, error, size);
    }
    const tc_stream_stage_plan_v1 plan{
        sizeof(plan), version, 1, 0, 1, 0, 1, 1, 1, &capacity, 1, &group};
    return tc_stream_executor_create_v1(&plan, &ops, executor, error, size);
}

void check_create_failure(uint32_t version, const char *unknown_error,
                          bool standard_exception, bool safe_at_failure) {
    Model model{standard_exception, safe_at_failure};
    tc_stream_executor *executor = nullptr;
    char error[1024]{};

    assert(!create(version, model, &executor, error, sizeof(error)));
    assert(model.allocations == 1);
    assert(std::strcmp(error, standard_exception ? "injected allocation failure" : unknown_error) == 0);
    if (safe_at_failure) {
        assert(!executor && model.destroys == 1);
    } else {
        assert(executor && model.destroys == 0);
        auto *retained = executor;
        assert(!tc_stream_executor_destroy(&executor, error, sizeof(error)));
        assert(executor == retained && model.destroys == 0);
        std::thread wrong_owner([&] {
            assert(!tc_stream_executor_destroy(&executor, error, sizeof(error)));
            assert(std::strstr(error, "streaming_owner_violation"));
        });
        wrong_owner.join();
        assert(executor == retained && model.destroys == 0);
        model.safe_to_destroy = true;
        assert(tc_stream_executor_destroy(&executor, error, sizeof(error)));
        assert(!executor && model.destroys == 1);
    }
    assert(tc_stream_executor_destroy(&executor, error, sizeof(error)));
    assert(model.destroys == 1);
}
} // namespace

int main() {
    for (bool standard_exception : {false, true})
        for (bool safe_at_failure : {false, true}) {
            check_create_failure(TC_STREAM_SLOT_ABI_V1, "streaming unknown create error",
                                 standard_exception, safe_at_failure);
            check_create_failure(TC_STREAM_SLOT_ABI_V2, "streaming unknown v2 create error",
                                 standard_exception, safe_at_failure);
            check_create_failure(TC_STREAM_SLOT_ABI_V3, "streaming unknown v3 create error",
                                 standard_exception, safe_at_failure);
        }
    std::cout << "PASS v1/v2/v3 C bridge construction failures: standard/unknown exceptions, "
                 "safe cleanup, quarantine ownership and retry\n";
}
