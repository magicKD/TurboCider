#include "memory_plan.hpp"

#include "../core/common.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

namespace tc {
namespace {

uint64_t checked_add(uint64_t left, uint64_t right, const char *what) {
    require(left <= std::numeric_limits<uint64_t>::max() - right,
            std::string("memory_plan_overflow: ") + what);
    return left + right;
}

struct ActiveResource {
    const ResourceInterval *resource = nullptr;
};

uint64_t alias_key(const ResourceInterval &resource) {
    return resource.alias_group ? resource.alias_group :
                                  (uint64_t(1) << 63) | resource.id;
}

uint64_t live_bytes(const std::map<uint64_t, ActiveResource> &active) {
    std::map<uint64_t, uint64_t> max_by_alias;
    for (const auto &[key, item] : active) {
        (void)key;
        const auto &resource = *item.resource;
        auto found = max_by_alias.find(alias_key(resource));
        if (found == max_by_alias.end() || found->second < resource.upper_bytes)
            max_by_alias[alias_key(resource)] = resource.upper_bytes;
    }
    uint64_t total = 0;
    for (const auto &[key, bytes] : max_by_alias) {
        (void)key;
        total = checked_add(total, bytes, "live resource bytes");
    }
    return total;
}

} // namespace

CompiledMemoryPlan compile_memory_plan(
        const MemoryManifest &manifest, uint64_t process_baseline_bytes,
        uint64_t framework_upper_bytes, uint64_t budget_bytes,
        const MemoryPlanCompileOptions &options) {
    manifest.validate();
    CompiledMemoryPlan result;
    result.manifest_digest = manifest.digest();
    result.process_baseline_bytes = process_baseline_bytes;
    result.framework_upper_bytes = framework_upper_bytes;
    result.require_explicit_epoch = options.require_explicit_epoch;
    result.require_explicit_schedule = options.require_explicit_schedule;
    require(!options.require_explicit_schedule ||
                options.require_explicit_epoch,
            "memory_schedule_invalid: explicit schedule requires explicit epoch transitions");
    require(budget_bytes > 0, "memory_policy_invalid: plan budget is zero");
    require(process_baseline_bytes <= budget_bytes,
            "memory_budget_too_small: process baseline exceeds plan budget");
    require(framework_upper_bytes <= budget_bytes - process_baseline_bytes,
            "memory_budget_too_small: framework upper exceeds plan budget");

    std::vector<const AllocationInstance *> sorted_instances;
    for (const auto &instance : manifest.instances)
        sorted_instances.push_back(&instance);
    std::sort(sorted_instances.begin(), sorted_instances.end(),
              [](const auto *left, const auto *right) {
                  return std::tie(left->site_id, left->instance_id,
                                  left->live_begin, left->live_end) <
                         std::tie(right->site_id, right->instance_id,
                                  right->live_begin, right->live_end);
              });
    std::vector<ResourceInterval> resources;
    resources.reserve(sorted_instances.size());
    uint32_t resource_id = 1;
    for (const auto *instance_pointer : sorted_instances) {
        const auto &instance = *instance_pointer;
        const auto *site = manifest.find_site(instance.site_id);
        require(site != nullptr, "memory_manifest_invalid: missing site");
        ResourceKind kind = ResourceKind::Activation;
        switch (site->memory_class) {
        case MemoryClass::Weights: kind = ResourceKind::Weight; break;
        case MemoryClass::Activation: kind = ResourceKind::Activation; break;
        case MemoryClass::Conditioning: kind = ResourceKind::Conditioning; break;
        case MemoryClass::RefillSlot: kind = ResourceKind::RefillSlot; break;
        case MemoryClass::ConversionScratch:
            kind = ResourceKind::HostStaging; break;
        case MemoryClass::Output: kind = ResourceKind::Output; break;
        case MemoryClass::AllocatorCache:
            kind = ResourceKind::AllocatorCache; break;
        case MemoryClass::CompileTemporary:
            kind = ResourceKind::GraphTemporary; break;
        case MemoryClass::ProcessBaseline:
            kind = ResourceKind::ProcessBaseline; break;
        case MemoryClass::ChildProcessEnvelope:
        case MemoryClass::UnknownExternal:
            kind = ResourceKind::HostStaging; break;
        }
        resources.push_back({resource_id++, instance.site_id, kind,
                             instance.upper_bytes, instance.live_begin,
                             instance.live_end, instance.alias_group,
                             site->required, site->asynchronous});
        result.allocation_instances.push_back({
            instance.site_id, instance.instance_id, site->memory_class,
            instance.upper_bytes, instance.live_begin, instance.live_end,
            instance.alias_group, site->asynchronous});
    }
    for (const auto &site : manifest.sites) {
        CompiledAllocationSite compiled_site;
        compiled_site.site_id = site.site_id;
        compiled_site.memory_class = site.memory_class;
        compiled_site.required = site.required;
        compiled_site.asynchronous = site.asynchronous;
        for (const auto &instance : manifest.instances) {
            if (instance.site_id != site.site_id) continue;
            compiled_site.maximum_instance_upper_bytes = std::max(
                compiled_site.maximum_instance_upper_bytes,
                instance.upper_bytes);
            compiled_site.aggregate_instance_upper_bytes = checked_add(
                compiled_site.aggregate_instance_upper_bytes,
                instance.upper_bytes, "site aggregate upper");
            require(compiled_site.instance_count !=
                        std::numeric_limits<uint32_t>::max(),
                    "memory_plan_overflow: site instance count");
            compiled_site.instance_count++;
        }
        result.allocation_sites.push_back(std::move(compiled_site));
    }
    std::sort(result.allocation_sites.begin(),
              result.allocation_sites.end(),
              [](const auto &left, const auto &right) {
                  return left.site_id < right.site_id;
              });
    std::sort(result.allocation_instances.begin(),
              result.allocation_instances.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.site_id, left.instance_id) <
                         std::tie(right.site_id, right.instance_id);
              });
    std::sort(resources.begin(), resources.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.acquire_epoch, left.release_epoch,
                                  left.site_id, left.id) <
                         std::tie(right.acquire_epoch, right.release_epoch,
                                  right.site_id, right.id);
              });

    std::set<uint64_t> epochs;
    epochs.insert(0);
    std::map<uint64_t, std::vector<const ResourceInterval *>> acquires;
    std::map<uint64_t, std::vector<const ResourceInterval *>> releases;
    for (const auto &resource : resources) {
        require(resource.upper_bytes > 0,
                "memory_manifest_invalid: resource upper is zero");
        require(resource.acquire_epoch < resource.release_epoch,
                "memory_manifest_invalid: resource lifetime is empty");
        epochs.insert(resource.acquire_epoch);
        epochs.insert(resource.release_epoch);
        acquires[resource.acquire_epoch].push_back(&resource);
        releases[resource.release_epoch].push_back(&resource);
    }

    std::map<uint64_t, ActiveResource> active;
    uint64_t peak = checked_add(process_baseline_bytes, framework_upper_bytes,
                                "baseline plus framework upper");
    uint64_t peak_epoch = 0;
    std::vector<uint32_t> peak_live;
    for (uint64_t epoch : epochs) {
        /* Half-open intervals: release is applied before acquire at the same
         * epoch.  A manifest that needs physical overlap must use distinct
         * epochs or distinct alias groups, so it cannot be hidden by ordering. */
        for (const auto *resource : releases[epoch])
            active.erase(resource->id);
        for (const auto *resource : acquires[epoch]) {
            require(active.emplace(resource->id,
                                    ActiveResource{resource}).second,
                    "memory_plan_invalid: duplicate active resource id");
        }
        const uint64_t current = checked_add(
            checked_add(process_baseline_bytes, framework_upper_bytes,
                        "baseline plus framework upper"),
            live_bytes(active), "plan peak");
        if (current > peak) {
            peak = current;
            peak_epoch = epoch;
            peak_live.clear();
            for (const auto &[id, item] : active) {
                (void)item;
                peak_live.push_back(id);
            }
        }
    }
    result.planned_peak_bytes = peak;
    result.peak_epoch = peak_epoch;
    result.peak_live_set = std::move(peak_live);
    result.complete = true;
    result.fits_budget = peak <= budget_bytes;
    if (!result.fits_budget) {
        std::ostringstream reason;
        reason << "planned peak " << peak << " exceeds budget " << budget_bytes;
        result.rejection_reason = reason.str();
    }
    std::vector<uint64_t> valid_epochs;
    valid_epochs.reserve(epochs.size());
    for (uint64_t epoch : epochs) {
        result.epochs.push_back(
            {epoch, "epoch_" + std::to_string(epoch), true});
        valid_epochs.push_back(epoch);
    }
    require(!options.require_explicit_schedule || !options.schedule.empty(),
            "memory_schedule_invalid: explicit schedule is required but absent");
    if (!options.schedule.empty()) {
        auto schedule = compile_memory_schedule(options.schedule, valid_epochs);
        result.schedule_schema = std::move(schedule.schema);
        result.schedule_revision = std::move(schedule.revision);
        result.schedule_bindings = std::move(schedule.bindings);
    }
    std::ostringstream plan_identity;
    plan_identity << "memory-plan-v1|manifest=" << result.manifest_digest
                  << "|baseline=" << process_baseline_bytes
                  << "|framework=" << framework_upper_bytes
                  << "|budget=" << budget_bytes
                  << "|peak=" << result.planned_peak_bytes
                  << "|epoch=" << result.peak_epoch
                  << "|explicit_epoch="
                  << (options.require_explicit_epoch ? 1 : 0)
                  << "|explicit_schedule="
                  << (options.require_explicit_schedule ? 1 : 0)
                  << "|schedule_schema=" << result.schedule_schema
                  << "|schedule_revision=" << result.schedule_revision;
    result.plan_digest = memory_sha256_hex(plan_identity.str());
    return result;
}

CompiledMemoryPlan compile_memory_plan(
        const MemoryManifest &manifest, uint64_t process_baseline_bytes,
        uint64_t framework_upper_bytes, uint64_t budget_bytes,
        bool require_explicit_epoch) {
    MemoryPlanCompileOptions options;
    options.require_explicit_epoch = require_explicit_epoch;
    return compile_memory_plan(
        manifest, process_baseline_bytes, framework_upper_bytes,
        budget_bytes, options);
}

} // namespace tc
