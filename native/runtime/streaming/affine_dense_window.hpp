#pragma once

#include "gguf_storage.hpp"
#include <array>
#include <list>

namespace tc::streaming {

// Research consumer for already-produced MLX affine Q4/Q8 (including the
// Q4_0/Q4_1/Q8_0 GGUF importer), NOT a raw-GGUF or inverse-ConvRot decoder.
// The caller's ticket identifies immutable source CONTENT, not an address.
// A reused/mutated packed bank MUST receive a new content ticket/generation.
// Retain original metadata dtype; widening before decode changes the recipe.
// Owner-thread only. Synchronous prepare publishes only completed, finite
// weights. This is bounded retention, NOT proof of physical prefetch overlap.
class AffineDenseWindow {
  public:
    struct Stats {
        uint64_t hits = 0, misses = 0, evictions = 0, decoded_bytes = 0;
        uint64_t retained_bytes = 0, peak_retained_bytes = 0;
    };

    AffineDenseWindow(MemoryLedger &ledger, uint64_t maximum_bytes, size_t slots = 2)
        : ledger_(ledger), maximum_bytes_(maximum_bytes), slots_(slots) {
        require(maximum_bytes && slots && slots <= 2, "affine dense window requires bounded one/two-slot budget");
    }
    AffineDenseWindow(const AffineDenseWindow &) = delete;
    AffineDenseWindow &operator=(const AffineDenseWindow &) = delete;

    Tensor prepare(const std::array<Tensor, 3> &source, int bits, int group,
                   const std::string &content_ticket, uint64_t generation) {
        const auto &[words, scales, biases] = source;
        require(!content_ticket.empty() && content_ticket.size() <= 512 && generation &&
                (bits == 4 || bits == 8) && (group == 32 || group == 64 || group == 128) &&
                words.ndim() == 2 && words.dtype() == mx::uint32 && words.shape(0) > 0 && words.shape(1) > 0 &&
                (scales.dtype() == mx::float16 || scales.dtype() == mx::bfloat16) && biases.dtype() == scales.dtype() &&
                scales.ndim() == 2 && biases.shape() == scales.shape() && scales.shape(0) == words.shape(0) &&
                uint64_t(words.shape(1)) * (32 / bits) == uint64_t(scales.shape(1)) * group,
                "affine dense window requires valid same-dtype Q4/Q8 content ticket");
        const Key key{content_ticket, generation, words.id(), scales.id(), biases.id(), bits, group, scales.dtype()};
        for (auto it = entries_.begin(); it != entries_.end(); ++it) if (it->key == key) {
            ++stats_.hits;
            entries_.splice(entries_.end(), entries_, it);
            return entries_.back().dense;
        }
        ++stats_.misses;
        const uint64_t bytes = gguf::checked_mul(gguf::checked_mul(uint64_t(words.shape(0)),
            uint64_t(words.shape(1)) * (32 / bits)), 2);
        const uint64_t upper = gguf_storage::capacity_upper(bytes);
        require(upper <= maximum_bytes_, "affine dense window matrix exceeds retained budget");
        while (!entries_.empty() && (entries_.size() >= slots_ || upper > maximum_bytes_ - stats_.retained_bytes)) {
            stats_.retained_bytes -= entries_.front().capacity;
            entries_.pop_front();
            ++stats_.evictions;
        }
        auto reservation = ledger_.try_reserve(MemoryClass::ConversionScratch, upper, "affine-dense-window-v1");
        require(reservation.has_value(), "affine dense window admission denied (including escaped readers)");
        // Do not let oversized cached allocator bins invalidate the admitted
        // physical backing upper. Cache hint is restored before consumers.
        gguf_storage::ExactCapacityCacheScope exact;
        auto dense = mx::dequantize(words, scales, biases, group, bits, "affine", std::nullopt, scales.dtype());
        auto finite = mx::all(mx::isfinite(dense));
        mx::eval({dense, finite});
        mx::synchronize();
        require(finite.item<bool>(), "affine dense window rejects nonfinite decoded coefficients");
        dense.detach();
        dense.set_siblings({}, 0);
        auto data = dense.data_shared_ptr();
        const uint64_t actual = mx::allocator::allocator().size(data->buffer);
        require(actual >= bytes && actual <= upper, "affine dense window allocation exceeds admitted upper");
        auto claim = std::make_shared<StorageLease>(reservation->commit({0x544344454e5345ull,
            uint64_t(reinterpret_cast<uintptr_t>(data->buffer.ptr())), actual, generation}));
        auto prior = data->d;
        data->d = [prior, claim](mx::allocator::Buffer buffer) { prior(buffer); };
        // Holding the source arrays also prevents ID recycling after detach;
        // a key is not merely an unretained pointer to a previous allocation.
        entries_.push_back({key, source, dense, actual});
        stats_.decoded_bytes += bytes;
        stats_.retained_bytes += actual;
        stats_.peak_retained_bytes = std::max(stats_.peak_retained_bytes, stats_.retained_bytes);
        return dense;
    }

    const Stats &stats() const { return stats_; }
    void clear() {
        entries_.clear();
        stats_.retained_bytes = 0;
        // Escaped dense views/lazy readers still own their Data/ledger claim.
    }

  private:
    struct Key {
        std::string ticket;
        uint64_t generation;
        uintptr_t words, scales, biases;
        int bits, group;
        mx::Dtype dtype;
        bool operator==(const Key &) const = default;
    };
    struct Entry { Key key; std::array<Tensor, 3> source; Tensor dense; uint64_t capacity; };
    MemoryLedger &ledger_;
    uint64_t maximum_bytes_;
    size_t slots_;
    Stats stats_;
    std::list<Entry> entries_;
};
} // namespace tc::streaming
