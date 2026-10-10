#pragma once
#include "native_hybrid_ffn.hpp"

namespace tc::metal_graph {
// Bounded row adapter shared by the native encoder and MLX DiT prefill island.
// Each ANE program has a fixed bucket; every tail is explicitly zero filled on
// GPU before A8 quantization. Only valid rows are copied to the model result.
// One physical slot/scratch set is reused; no speculative reuse after failure.
struct NativePrefillFfnStats {
    uint64_t calls=0, chunks=0, valid_rows=0, padded_rows=0;
    double producer_seconds=0, branch_seconds=0, consumer_seconds=0, total_seconds=0;
};
class NativePrefillFfn {
public:
    static std::unique_ptr<NativePrefillFfn> prepare(Backend&, NativeHybridFfnOptions);
    ~NativePrefillFfn();
    NativePrefillFfn(const NativePrefillFfn&)=delete;
    NativePrefillFfn& operator=(const NativePrefillFfn&)=delete;
    // before/after must belong to the same backend and exact input/output.
    // Output is caller-owned and is published only after all chunks complete.
    void execute(MatrixView input, MatrixView gate, MatrixView up, MatrixView down,
        MatrixView output, CommandGraph before={}, CommandGraph after={},
        std::chrono::milliseconds timeout=std::chrono::seconds(30),
        const std::function<void()>& concurrent_prepare={});
    NativePrefillFfnStats statistics() const;
    bool healthy() const;
    void reset_statistics();
    uint32_t bucket_rows() const;
private:
    struct Impl;
    explicit NativePrefillFfn(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
} // namespace tc::metal_graph
