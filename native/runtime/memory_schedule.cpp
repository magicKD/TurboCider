#include "memory_schedule.hpp"

#include "../core/common.hpp"
#include "memory_manifest.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <tuple>

namespace tc {
namespace {

constexpr uint32_t kKnownFlags =
    TC_MEMORY_EVENT_SAFE_POINT | TC_MEMORY_EVENT_GPU_DRAINED |
    TC_MEMORY_EVENT_PREFETCH_ALLOWED | TC_MEMORY_EVENT_TERMINAL;

bool valid_stage(uint32_t value) {
    return value >= TC_MEMORY_STAGE_ADMISSION &&
           value <= TC_MEMORY_STAGE_TERMINAL_DRAIN;
}

bool valid_action(uint32_t value) {
    return value >= TC_MEMORY_ACTION_BEGIN &&
           value <= TC_MEMORY_ACTION_END;
}

bool valid_branch(uint32_t value) {
    return value >= TC_MEMORY_BRANCH_COMMON &&
           value <= TC_MEMORY_BRANCH_EXPORT;
}

const char *stage_name(uint32_t value) {
    switch (value) {
    case TC_MEMORY_STAGE_ADMISSION: return "admission";
    case TC_MEMORY_STAGE_TEXT: return "text";
    case TC_MEMORY_STAGE_CONDITIONING_HANDOFF: return "conditioning_handoff";
    case TC_MEMORY_STAGE_DENOISER_LOAD: return "denoiser_load";
    case TC_MEMORY_STAGE_DENOISER: return "denoiser";
    case TC_MEMORY_STAGE_LATENT_HANDOFF: return "latent_handoff";
    case TC_MEMORY_STAGE_VIDEO_VAE: return "video_vae";
    case TC_MEMORY_STAGE_AUDIO_VAE: return "audio_vae";
    case TC_MEMORY_STAGE_EXPORT: return "export";
    case TC_MEMORY_STAGE_TERMINAL_DRAIN: return "terminal_drain";
    }
    return "invalid";
}

const char *action_name(uint32_t value) {
    switch (value) {
    case TC_MEMORY_ACTION_BEGIN: return "begin";
    case TC_MEMORY_ACTION_PREFETCH: return "prefetch";
    case TC_MEMORY_ACTION_UPLOAD: return "upload";
    case TC_MEMORY_ACTION_COMPUTE: return "compute";
    case TC_MEMORY_ACTION_LAST_USE: return "last_use";
    case TC_MEMORY_ACTION_RETIRE: return "retire";
    case TC_MEMORY_ACTION_DRAIN: return "drain";
    case TC_MEMORY_ACTION_END: return "end";
    }
    return "invalid";
}

void validate_binding(const MemoryScheduleBindingSpec &binding) {
    require(valid_stage(binding.key.stage) &&
                valid_action(binding.key.action) &&
                valid_branch(binding.key.branch),
            "memory_schedule_invalid: event key contains an invalid enum");
    require((binding.required_flags & ~kKnownFlags) == 0,
            "memory_schedule_invalid: event contains unknown flags");
    if (binding.required_flags & TC_MEMORY_EVENT_GPU_DRAINED)
        require(binding.required_flags & TC_MEMORY_EVENT_SAFE_POINT,
                "memory_schedule_invalid: GPU-drained event is not a safe point");
    if (binding.required_flags & TC_MEMORY_EVENT_TERMINAL)
        require((binding.required_flags &
                 (TC_MEMORY_EVENT_SAFE_POINT |
                  TC_MEMORY_EVENT_GPU_DRAINED)) ==
                    (TC_MEMORY_EVENT_SAFE_POINT |
                     TC_MEMORY_EVENT_GPU_DRAINED),
                "memory_schedule_invalid: terminal event must be a drained safe point");
    if (binding.checkpoint_after)
        require(binding.required_flags & TC_MEMORY_EVENT_SAFE_POINT,
                "memory_schedule_invalid: checkpoint event is not a safe point");
}

} // namespace

bool MemoryScheduleEventKey::operator<(
        const MemoryScheduleEventKey &other) const {
    return std::tie(stage, action, step, block, tile, branch) <
           std::tie(other.stage, other.action, other.step, other.block,
                    other.tile, other.branch);
}

std::string MemoryScheduleEventKey::canonical() const {
    std::ostringstream out;
    out << "stage=" << stage << "|action=" << action
        << "|step=" << step << "|block=" << block
        << "|tile=" << tile << "|branch=" << branch;
    return out.str();
}

MemoryScheduleEventKey memory_schedule_event_key(
        const tc_memory_schedule_event_v1 &event) {
    return {event.stage, event.action, event.step, event.block,
            event.tile, event.branch};
}

std::string memory_schedule_event_name(const MemoryScheduleEventKey &key) {
    std::ostringstream out;
    out << stage_name(key.stage) << '.' << action_name(key.action);
    if (key.step != TC_MEMORY_INDEX_NONE) out << ".step" << key.step;
    if (key.block != TC_MEMORY_INDEX_NONE) out << ".block" << key.block;
    if (key.tile != TC_MEMORY_INDEX_NONE) out << ".tile" << key.tile;
    out << ".branch" << key.branch;
    return out.str();
}

CompiledMemorySchedule compile_memory_schedule(
        const std::vector<MemoryScheduleBindingSpec> &bindings,
        const std::vector<uint64_t> &valid_epochs) {
    require(!bindings.empty(),
            "memory_schedule_invalid: schedule binding list is empty");
    require(!valid_epochs.empty(),
            "memory_schedule_invalid: valid epoch list is empty");
    const std::set<uint64_t> epoch_set(valid_epochs.begin(),
                                       valid_epochs.end());
    require(epoch_set.size() == valid_epochs.size(),
            "memory_schedule_invalid: valid epoch list contains duplicates");
    CompiledMemorySchedule result;
    std::set<std::pair<MemoryScheduleEventKey, uint32_t>> identities;
    uint64_t previous_epoch = 0;
    bool first = true;
    std::ostringstream canonical;
    canonical << result.schema;
    result.bindings.reserve(bindings.size());
    for (size_t index = 0; index < bindings.size(); ++index) {
        const auto &binding = bindings[index];
        validate_binding(binding);
        require(epoch_set.count(binding.epoch) != 0,
                "memory_schedule_invalid: event references an unknown epoch");
        require(first || binding.epoch >= previous_epoch,
                "memory_schedule_invalid: event epochs move backwards");
        require(identities.emplace(binding.key,
                                   binding.expected_slot).second,
                "memory_schedule_invalid: duplicate semantic event");
        if (binding.required_flags & TC_MEMORY_EVENT_TERMINAL)
            require(index + 1 == bindings.size(),
                    "memory_schedule_invalid: terminal event is not last");
        CompiledScheduleBinding compiled;
        compiled.sequence = index;
        compiled.key = binding.key;
        compiled.expected_slot = binding.expected_slot;
        compiled.required_flags = binding.required_flags;
        compiled.epoch = binding.epoch;
        compiled.checkpoint_after = binding.checkpoint_after;
        result.bindings.push_back(compiled);
        canonical << "|sequence=" << index
                  << '|' << binding.key.canonical()
                  << "|slot=" << binding.expected_slot
                  << "|flags=" << binding.required_flags
                  << "|epoch=" << binding.epoch
                  << "|checkpoint=" << (binding.checkpoint_after ? 1 : 0);
        previous_epoch = binding.epoch;
        first = false;
    }
    require(result.bindings.back().required_flags & TC_MEMORY_EVENT_TERMINAL,
            "memory_schedule_invalid: schedule has no terminal event");
    result.revision = memory_sha256_hex(canonical.str());
    return result;
}

} // namespace tc
