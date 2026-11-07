#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tc {
struct QwenBf16StreamStageMetrics {
    std::string source_sha256,layout_digest,cache_layout_digest;
    std::string source_precision,matmul_mode;
    uint64_t source_file_bytes=0,verification_bytes=0,verification_cache_hits=0;
    uint64_t managed_weight_capacity_bytes=0,managed_weight_budget_bytes=0;
    uint64_t derived_weight_capacity_bytes=0;
    uint64_t gathered_embedding_rows=0,gathered_embedding_source_bytes=0;
    uint64_t resident_source_bytes=0,streamed_source_bytes=0,slot_arrays=0;
    uint64_t fills=0,completed_layers=0,completed_prefix_layers=0,completed_streamed_layers=0;
    uint64_t reader_fences=0,completed_reader_fences=0;
    uint32_t layers=0,prefix=0,slots=0,completed_passes=0;
    uint32_t peak_live_layer_owners=0;
    uint32_t cache_budget_prefix=0;
    bool cache_limited_by_memory=false;
    uint32_t final_prefix=0,cache_evictions=0;
    std::vector<uint32_t> prefix_per_pass;
    double verification_seconds=0,resident_load_seconds=0,streamed_read_seconds=0,wait_seconds=0;
    bool drained=false;
};
struct QwenBf16StreamingMetrics {
    QwenBf16StreamStageMetrics encoder,denoiser;
};
struct QwenConvrotStreamingMetrics {
    QwenBf16StreamStageMetrics encoder,denoiser;
    uint64_t requested_weight_budget_bytes=0;
};
}
