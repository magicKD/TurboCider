#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tc::seedvr2 {
// CPU-only state machine: no pool reuse until every chunk was produced and
// the owner explicitly acknowledges its separately owned, completed join.
class PrefixBufferLease {
    std::vector<bool> written_;
    bool active_ = false;
  public:
    void begin(size_t chunks) {
        if (active_ || !chunks || chunks > 256)
            throw std::logic_error("invalid or still-active SeedVR2 prefix buffer lease");
        written_.assign(chunks, false);
        active_ = true;
    }
    void check(size_t chunk) const {
        if (!active_ || chunk >= written_.size() || written_[chunk])
            throw std::logic_error("invalid or already-written SeedVR2 prefix output slot");
    }
    void mark_written(size_t chunk) {
        check(chunk);
        written_[chunk] = true;
    }
    void release() {
        if (!active_)
            throw std::logic_error("SeedVR2 prefix buffer lease is not active");
        for (bool written : written_)
            if (!written)
                throw std::logic_error("SeedVR2 prefix buffer lease has incomplete outputs");
        active_ = false;
    }
    bool active() const { return active_; }
};
} // namespace tc::seedvr2
