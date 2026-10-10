#pragma once
#include "affine_dense_finite_candidate.hpp"
#include <memory>
#include <string>

namespace tc::research {
// Owner-thread research window. Two logical jobs, fresh admitted targets,
// immutable prepared source owners; never overwrite an escaped old result.
// A submitted job is NOT a published finite weight or physical overlap proof.
class AffineAheadWindow {
  public:
    struct Stats {
        uint64_t submitted=0,published=0,failed=0,drained=0,pending=0,peak_pending=0;
        double wait_seconds=0;
    };
    explicit AffineAheadWindow(MemoryLedger &ledger,size_t slots=2);
    ~AffineAheadWindow();
    AffineAheadWindow(const AffineAheadWindow &)=delete;
    AffineAheadWindow &operator=(const AffineAheadWindow &)=delete;
    uint64_t submit(const affine_finite::Packed &source,int bits,int group,
                    const std::string &content_ticket,uint64_t source_generation);
    Tensor take(uint64_t submission);
    void drain();
    Stats stats() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tc::research
