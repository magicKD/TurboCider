#pragma once
#include <cstdint>
#include <string>

namespace tc {
struct QwenBf16StreamStageMetrics {
    std::string source_sha256,layout_digest;
    uint64_t source_file_bytes=0,verification_bytes=0,verification_cache_hits=0;
    uint64_t managed_weight_capacity_bytes=0,managed_weight_budget_bytes=0;
    uint64_t resident_source_bytes=0,streamed_source_bytes=0,slot_arrays=0;
    uint64_t fills=0,completed_layers=0,completed_prefix_layers=0,completed_streamed_layers=0;
    uint64_t reader_fences=0,completed_reader_fences=0;
    uint32_t layers=0,prefix=0,slots=0,completed_passes=0;
    double verification_seconds=0,resident_load_seconds=0,streamed_read_seconds=0,wait_seconds=0;
    bool drained=false;
};
struct QwenBf16StreamingMetrics {
    QwenBf16StreamStageMetrics encoder,denoiser;
};
}
