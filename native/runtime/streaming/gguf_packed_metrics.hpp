#pragma once
#include <cstdint>
#include <string>

namespace tc::streaming {
struct GgufPackedBankMetrics {
    std::string source_sha256, plan_digest;
    uint64_t planned_packed_capacity_bytes = 0, packed_capacity_bytes = 0;
    uint64_t read_buffer_capacity_bytes = 0, source_read_bytes = 0, logical_source_bytes = 0;
    uint64_t managed_peak_bytes = 0, output_bytes = 0, verification_bytes = 0;
    uint32_t tensor_count = 0, field_count = 0;
    double load_seconds = 0, read_seconds = 0, decode_seconds = 0;
    bool released_before_vae = false, serial_refiner_eval = false;
};
} // namespace tc::streaming
