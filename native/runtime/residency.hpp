#pragma once
#include "session.hpp"
namespace tc {
struct ResidencyPolicy {
    bool retain_images_during_text = false;
    bool release_before_denoise = false;
    bool release_after_denoise = false;
    bool release_after_decode = false;
    static ResidencyPolicy for_request(const Request &, uint64_t physical_memory);
    static void validate_budget(const ExecutionPlan &, uint64_t physical_memory);
};
} // namespace tc
