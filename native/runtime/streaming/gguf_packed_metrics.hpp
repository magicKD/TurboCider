#pragma once
#include <cstdint>
#include <string>

namespace tc::streaming {
struct GgufPackedBankMetrics {
    std::string source_sha256, plan_digest;
    std::string affine_packing_recipe,affine_packing_backend;
    std::string float_import_recipe;
    uint64_t planned_packed_capacity_bytes = 0, packed_capacity_bytes = 0;
    uint64_t read_buffer_capacity_bytes = 0, source_read_bytes = 0, logical_source_bytes = 0;
    uint64_t managed_peak_bytes = 0, output_bytes = 0, verification_bytes = 0;
    uint32_t tensor_count = 0, field_count = 0;
    double load_seconds = 0, read_seconds = 0, decode_seconds = 0;
    double affine_decode_seconds=0,float_decode_seconds=0;
    bool released_before_vae = false, serial_refiner_eval = false;
    bool session_packed_retention = false, reused_packed_bank = false, compiled_packed_blocks = false;
    bool gpu_f16_compute = false;
    bool gpu_f16_mpp = false;
    bool qmm_f16_compute = false;
    bool f16_refiners = false;
    bool ref_mpp_dynamic = false;
    uint64_t dense_weight_capacity_upper = 0;
    uint64_t allocator_cache_limit_bytes = 0;
};
} // namespace tc::streaming
