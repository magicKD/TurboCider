#pragma once

#include "mlx_weight_pager.hpp"
#include "../../core/gguf_decode.hpp"
#include "../memory_accounting.hpp"

namespace tc::streaming {

struct GgufWeightPagerMetrics {
    uint64_t packed_source_bytes = 0, packed_capacity_bytes = 0;
    uint64_t source_float_bytes = 0, source_quantized_bytes = 0;
    uint64_t fixed_alias_bytes = 0, dense_pool_capacity_bytes = 0;
    uint64_t maximum_dense_pool_capacity_bytes = 0;
    uint64_t fill_count = 0, decoded_bytes = 0, source_bytes_processed = 0;
    double packed_read_seconds = 0, decode_seconds = 0;
    uint64_t source_logical_bytes = 0, read_buffer_capacity_bytes = 0, source_read_bytes = 0;
    double streamed_read_seconds = 0;
    uint64_t gpu_affine_preparations=0,gpu_affine_output_bytes=0;
    double gpu_prepare_seconds=0;
    uint64_t gpu_fixed_output_bank_bytes=0,gpu_fixed_output_banks=0,maximum_gpu_fixed_output_bank_bytes=0;
};

// Packed-resident source, owner-created immutable raw buffers, and a bounded
// pool of dense outputs. Reuses StageExecutor's state/fence protocol: this
// class does not contain a queue or scheduler. Every physical MLX buffer's
// ledger lease lives in its array Data deleter, including escaped views.
class GgufWeightPager {
  public:
    GgufWeightPager(std::shared_ptr<const SourceLease>, const Descriptor &,
                    const StageDescriptor &, const StageLayout &, MemoryLedger &, gguf::DecodeOptions = {});
    ~GgufWeightPager();
    GgufWeightPager(const GgufWeightPager &) = delete;
    GgufWeightPager &operator=(const GgufWeightPager &) = delete;
    void load_packed(const std::atomic<bool> *cancel = nullptr);
    void load_resident_aliases(Weights &);
    // Explicit packed-source resident field, not a full dense embedding.
    // Gather output is independently owned and charged to the same ledger.
    Tensor gather_rows(const std::string &tensor, std::span<const uint64_t> rows,
                       const std::atomic<bool> *cancel = nullptr);
    void create_pool(const PoolLayout &);
    void destroy_pool(uint32_t) noexcept;
    uint64_t fill(const Group &, const tc_stream_slot_ticket_v1 &, const std::atomic<bool> *);
    Weights bind(const Group &, const tc_stream_slot_ticket_v1 &) const;
    void retire(const Group &,const tc_stream_slot_ticket_v1 &);
    void check_unchanged() const;
    GgufWeightPagerMetrics metrics() const;
    const gguf::Directory &directory(uint32_t artifact) const;
    static constexpr uint64_t buffer_alignment = 16384;
    static constexpr uint64_t default_read_buffer_bytes = 1ull << 20;
  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace tc::streaming
