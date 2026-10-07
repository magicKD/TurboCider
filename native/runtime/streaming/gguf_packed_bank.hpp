#pragma once

#include "source_lease.hpp"
#include "gguf_packed_metrics.hpp"
#include "../../backends/mlx.hpp"
#include "../../core/gguf_directory.hpp"
#include "../memory_accounting.hpp"

namespace tc::streaming {

// Immutable, compute-ready MLX affine packed bank. Only Q4_0/Q4_1/Q8_0 and
// floating tensors are accepted; raw packed source never becomes a second
// resident model. This is NOT an all-model dense expansion or a RAM guard.
// Construction is metadata-only. load() is transactional and owner-only.
class GgufPackedBank final {
  public:
    GgufPackedBank(std::shared_ptr<const SourceLease>, std::string logical_id,
                   MemoryLedger &, uint64_t read_buffer_bytes = 1ull << 20,bool fused_affine=true,
                   uint64_t raw_window_bytes=0,uint32_t raw_window_entries=6);
    ~GgufPackedBank();
    GgufPackedBank(const GgufPackedBank &) = delete;
    GgufPackedBank &operator=(const GgufPackedBank &) = delete;
    const gguf::Directory &directory() const;
    void load(Weights &, const std::atomic<bool> *cancel = nullptr, const Event & = {});
    void check_unchanged() const;
    GgufPackedBankMetrics metrics() const;
    struct RawMatrix {
        Tensor values; uint32_t type; int columns;
        std::shared_ptr<void> logical_content_identity{};
    };
    // Explicit bounded raw GGML source for ANE staging, NOT affine output.
    // CPU read is synchronous/owner-only; no pending buffer is published.
    // Array Data keeps its ledger claim even after window eviction/destruction.
    RawMatrix raw_matrix(const std::string &,const std::atomic<bool> *cancel=nullptr);
    void clear_raw_window();
  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace tc::streaming
