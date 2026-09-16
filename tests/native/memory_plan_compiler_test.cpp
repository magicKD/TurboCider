#include "memory_plan.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

using namespace tc;

namespace {

constexpr const char *checkpoint =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

AllocationSiteSpec site(std::string id, MemoryClass memory_class,
                        bool aliasable = false) {
    AllocationSiteSpec result;
    result.site_id = std::move(id);
    result.component = "synthetic";
    result.stage = "test";
    result.memory_class = memory_class;
    result.lifetime = AllocationLifetime::Stage;
    result.provenance = UpperProvenance::ExactShapeFormula;
    result.aliasable = aliasable;
    return result;
}

MemoryManifest manifest() {
    MemoryManifest result;
    result.candidate_id = "synthetic_streamed_v1";
    result.checkpoint_digest = checkpoint;
    result.backend_revision = "synthetic-backend-v1";
    result.runtime_revision = "memory-runtime-v3";
    result.sites = {
        site("synthetic.weights", MemoryClass::Weights, true),
        site("synthetic.activation", MemoryClass::Activation),
        site("synthetic.output", MemoryClass::Output),
    };
    result.instances = {
        {"synthetic.weights", 0, 40, 1, 3, 7, "weights.complete"},
        {"synthetic.weights", 1, 30, 3, 4, 7, "weights.complete"},
        {"synthetic.activation", 0, 20, 2, 3, 0,
         "activation.complete"},
        {"synthetic.output", 0, 10, 4, 5, 0, "output.complete"},
    };
    return result;
}

std::vector<MemoryScheduleBindingSpec> schedule() {
    MemoryScheduleBindingSpec begin;
    begin.key.stage = TC_MEMORY_STAGE_ADMISSION;
    begin.key.action = TC_MEMORY_ACTION_END;
    MemoryScheduleBindingSpec compute;
    compute.key.stage = TC_MEMORY_STAGE_DENOISER;
    compute.key.action = TC_MEMORY_ACTION_COMPUTE;
    compute.expected_slot = 0;
    compute.epoch = 2;
    MemoryScheduleBindingSpec terminal;
    terminal.key.stage = TC_MEMORY_STAGE_TERMINAL_DRAIN;
    terminal.key.action = TC_MEMORY_ACTION_END;
    terminal.epoch = 5;
    terminal.required_flags = TC_MEMORY_EVENT_SAFE_POINT |
        TC_MEMORY_EVENT_GPU_DRAINED | TC_MEMORY_EVENT_TERMINAL;
    terminal.checkpoint_after = true;
    return {begin, compute, terminal};
}

template <typename Function>
void expect_rejected(Function function, const char *needle) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &error) {
        rejected = std::string(error.what()).find(needle) != std::string::npos;
    }
    assert(rejected);
}

void test_alias_folding_and_half_open_lifetimes() {
    const auto plan = compile_memory_plan(manifest(), 10, 5, 75);
    assert(plan.complete);
    assert(plan.fits_budget);
    assert(plan.planned_peak_bytes == 75);
    assert(plan.peak_epoch == 2);
    assert(plan.peak_live_set.size() == 2);
    assert(plan.allocation_sites.size() == 3);
    assert(plan.allocation_instances.size() == 4);
    assert(plan.process_baseline_bytes == 10);
    assert(plan.framework_upper_bytes == 5);
    const auto weights = std::find_if(
        plan.allocation_sites.begin(), plan.allocation_sites.end(),
        [](const auto &site) { return site.site_id == "synthetic.weights"; });
    assert(weights != plan.allocation_sites.end());
    assert(weights->memory_class == MemoryClass::Weights);
    assert(weights->maximum_instance_upper_bytes == 40);
    assert(weights->aggregate_instance_upper_bytes == 70);
    assert(weights->instance_count == 2);
    const auto weight_instance = std::find_if(
        plan.allocation_instances.begin(), plan.allocation_instances.end(),
        [](const auto &instance) {
            return instance.site_id == "synthetic.weights" &&
                   instance.instance_id == 1;
        });
    assert(weight_instance != plan.allocation_instances.end());
    assert(weight_instance->upper_bytes == 30);
    assert(weight_instance->live_begin == 3);
    assert(weight_instance->live_end == 4);
    assert(weight_instance->alias_group == 7);
    assert(plan.manifest_digest.size() == 64);
    assert(plan.plan_digest.size() == 64);
}

void test_minimum_minus_one_is_rejected_without_losing_plan() {
    const auto plan = compile_memory_plan(manifest(), 10, 5, 74);
    assert(plan.complete);
    assert(!plan.fits_budget);
    assert(plan.planned_peak_bytes == 75);
    assert(plan.rejection_reason.find("exceeds budget 74") != std::string::npos);
}

void test_input_order_does_not_change_plan_digest() {
    auto first = manifest();
    auto second = first;
    std::reverse(second.sites.begin(), second.sites.end());
    std::reverse(second.instances.begin(), second.instances.end());
    const auto a = compile_memory_plan(first, 10, 5, 75);
    const auto b = compile_memory_plan(second, 10, 5, 75);
    assert(a.manifest_digest == b.manifest_digest);
    assert(a.plan_digest == b.plan_digest);
    assert(a.planned_peak_bytes == b.planned_peak_bytes);
    assert(a.peak_live_set == b.peak_live_set);
}

void test_explicit_epoch_requirement_is_plan_identity() {
    const auto compatibility = compile_memory_plan(
        manifest(), 10, 5, 75, false);
    const auto strict = compile_memory_plan(
        manifest(), 10, 5, 75, true);
    assert(!compatibility.require_explicit_epoch);
    assert(strict.require_explicit_epoch);
    assert(compatibility.plan_digest != strict.plan_digest);
    assert(compatibility.planned_peak_bytes == strict.planned_peak_bytes);
}

void test_schedule_requirement_is_plan_identity() {
    MemoryPlanCompileOptions options;
    options.require_explicit_epoch = true;
    options.require_explicit_schedule = true;
    options.schedule = schedule();
    const auto scheduled = compile_memory_plan(
        manifest(), 10, 5, 75, options);
    const auto unscheduled = compile_memory_plan(
        manifest(), 10, 5, 75, true);
    assert(scheduled.require_explicit_schedule);
    assert(scheduled.schedule_bindings.size() == 3);
    assert(scheduled.schedule_revision.size() == 64);
    assert(scheduled.schedule_schema == "turbocider.memory_schedule.v1");
    assert(scheduled.plan_digest != unscheduled.plan_digest);

    options.require_explicit_epoch = false;
    expect_rejected(
        [&] { (void)compile_memory_plan(
                  manifest(), 10, 5, 75, options); },
        "requires explicit epoch transitions");
    options.require_explicit_epoch = true;

    options.schedule.clear();
    expect_rejected(
        [&] { (void)compile_memory_plan(
                  manifest(), 10, 5, 75, options); },
        "explicit schedule is required but absent");
}

void test_checked_arithmetic_and_baseline_gate() {
    expect_rejected(
        [&] { (void)compile_memory_plan(manifest(), 76, 0, 75); },
        "baseline exceeds");
    auto huge = manifest();
    huge.instances.front().upper_bytes = UINT64_MAX;
    expect_rejected(
        [&] { (void)compile_memory_plan(huge, 10, 5, UINT64_MAX); },
        "memory_plan_overflow");
}

} // namespace

int main() {
    test_alias_folding_and_half_open_lifetimes();
    test_minimum_minus_one_is_rejected_without_losing_plan();
    test_input_order_does_not_change_plan_digest();
    test_explicit_epoch_requirement_is_plan_identity();
    test_schedule_requirement_is_plan_identity();
    test_checked_arithmetic_and_baseline_gate();
    std::cout << "memory plan compiler tests passed\n";
    return 0;
}
