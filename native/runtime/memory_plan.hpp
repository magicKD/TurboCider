#pragma once

#include "memory_manifest.hpp"
#include "memory_schedule.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tc {

enum class ResourceKind : uint8_t {
    Weight,
    Activation,
    Conditioning,
    RefillSlot,
    HostStaging,
    GraphTemporary,
    Output,
    AllocatorCache,
    ProcessBaseline,
};

struct ResourceInterval {
    uint32_t id = 0;
    std::string site_id;
    ResourceKind kind = ResourceKind::Activation;
    uint64_t upper_bytes = 0;
    uint64_t acquire_epoch = 0;
    uint64_t release_epoch = 0;
    uint64_t alias_group = 0;
    bool required = true;
    bool asynchronous_release = false;
};

struct ScheduleEpoch {
    uint64_t id = 0;
    std::string name;
    bool runtime_safe_point = false;
};

struct CompiledAllocationSite {
    std::string site_id;
    MemoryClass memory_class = MemoryClass::UnknownExternal;
    uint64_t maximum_instance_upper_bytes = 0;
    uint64_t aggregate_instance_upper_bytes = 0;
    uint32_t instance_count = 0;
    bool required = true;
    bool asynchronous = false;
};

struct CompiledAllocationInstance {
    std::string site_id;
    uint32_t instance_id = 0;
    MemoryClass memory_class = MemoryClass::UnknownExternal;
    uint64_t upper_bytes = 0;
    uint64_t live_begin = 0;
    uint64_t live_end = 0;
    uint64_t alias_group = 0;
    bool asynchronous = false;
};

struct CompiledMemoryPlan {
    bool complete = false;
    bool fits_budget = false;
    uint64_t process_baseline_bytes = 0;
    uint64_t framework_upper_bytes = 0;
    uint64_t planned_peak_bytes = 0;
    uint64_t peak_epoch = 0;
    bool require_explicit_epoch = false;
    bool require_explicit_schedule = false;
    std::vector<uint32_t> peak_live_set;
    std::vector<ScheduleEpoch> epochs;
    std::vector<CompiledScheduleBinding> schedule_bindings;
    std::vector<CompiledAllocationSite> allocation_sites;
    std::vector<CompiledAllocationInstance> allocation_instances;
    std::string manifest_digest;
    std::string schedule_schema;
    std::string schedule_revision;
    std::string plan_digest;
    std::string rejection_reason;
};

struct MemoryPlanCompileOptions {
    bool require_explicit_epoch = false;
    bool require_explicit_schedule = false;
    std::vector<MemoryScheduleBindingSpec> schedule;
};

CompiledMemoryPlan compile_memory_plan(
    const MemoryManifest &, uint64_t process_baseline_bytes,
    uint64_t framework_upper_bytes, uint64_t budget_bytes,
    bool require_explicit_epoch = false);
CompiledMemoryPlan compile_memory_plan(
    const MemoryManifest &, uint64_t process_baseline_bytes,
    uint64_t framework_upper_bytes, uint64_t budget_bytes,
    const MemoryPlanCompileOptions &);

} // namespace tc
