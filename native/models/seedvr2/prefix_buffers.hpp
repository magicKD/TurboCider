#pragma once
#include "model.hpp"

namespace tc::seedvr2 {
// Experimental fixed-shape prefix pool. Each row chunk has an independent
// output backing, but all chunks reuse the same MLModel for a given block.
// Inputs must already be materialized before GPU suffix submission. The pool
// must be released only after the joined, separately owned result is ready.
class PrefixBuffers {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    PrefixBuffers(const std::vector<std::filesystem::path> &, int rows, int hidden,
                  const Event &, std::atomic<bool> &, bool cpu_only_control = false);
    ~PrefixBuffers();
    void prepare(size_t chunks);
    Tensor predict(int block, size_t chunk, const Tensor &materialized_input);
    void release_after(const Tensor &owned_join);
    uint64_t calls() const;
    uint64_t copied_bytes() const;
    size_t backing_bytes() const;
    size_t backing_slots() const;
    size_t maximum_chunks() const;
};
} // namespace tc::seedvr2
