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
};

// Packed-resident source, owner-created immutable raw buffers, and a bounded
// pool of dense outputs. Reuses StageExecutor's state/fence protocol: this
// class does not contain a queue or scheduler. Every physical MLX buffer's
// ledger lease lives in its array Data deleter, including escaped views.
class GgufWeightPager {
  public:
    GgufWeightPager(std::shared_ptr<const SourceLease>, const Descriptor &,
                    const StageDescriptor &, const StageLayout &, MemoryLedger &);
    ~GgufWeightPager();
    GgufWeightPager(const GgufWeightPager &) = delete;
    GgufWeightPager &operator=(const GgufWeightPager &) = delete;
    void load_packed(const std::atomic<bool> *cancel = nullptr);
    void load_resident_aliases(Weights &);
    void create_pool(const PoolLayout &);
    void destroy_pool(uint32_t) noexcept;
    uint64_t fill(const Group &, const tc_stream_slot_ticket_v1 &, const std::atomic<bool> *);
    Weights bind(const Group &, const tc_stream_slot_ticket_v1 &) const;
    void check_unchanged() const;
    GgufWeightPagerMetrics metrics() const;
    const gguf::Directory &directory(uint32_t artifact) const;
    static constexpr uint64_t buffer_alignment = 16384;
  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace tc::streaming
