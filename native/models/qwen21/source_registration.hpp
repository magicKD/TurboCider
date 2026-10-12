#pragma once
#include "../../core/common.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include <cstdlib>
#include <iostream>

namespace tc::qwen21 {

inline bool source_registration_enabled() {
    const auto *flag = std::getenv("TURBOCIDER_QWEN21_SOURCE_REGISTRATION");
    require(!flag || std::string_view(flag) == "0" || std::string_view(flag) == "1",
            "Qwen source registration requires 0 or 1");
    return !flag || std::string_view(flag) == "1";
}

inline std::shared_ptr<const streaming::SourceLease> capture_model_source(
        streaming::SourceFileIdentity source, std::atomic<bool> &cancel) {
    const bool registered = source_registration_enabled();
    auto lease = registered ? streaming::SourceLease::capture_registered({source}, &cancel) :
                              streaming::SourceLease::capture_verified({source}, &cancel);
    const auto bytes = lease->verification_bytes_read();
    std::cerr << "{\"qwen_source_registration\":{\"component\":\"" << source.logical_id
        << "\",\"enabled\":" << (registered ? "true" : "false")
        << ",\"verification_bytes\":" << bytes
        << ",\"verification_cache_hits\":" << lease->verification_cache_hits()
        << ",\"registration_cache_hits\":" << lease->registration_cache_hits()
        << ",\"registration_records_written\":" << lease->registration_records_written()
        << ",\"first_registration_seconds\":" << (bytes && registered ? lease->verification_seconds() : 0)
        << ",\"verification_seconds\":" << lease->verification_seconds() << "}}\n";
    return lease;
}

} // namespace tc::qwen21
