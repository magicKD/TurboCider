#pragma once

#include "../../backends/mlx.hpp"
#include "../../core/gguf_directory.hpp"
#include "../memory_accounting.hpp"

namespace tc::streaming::gguf_storage {

inline constexpr uint64_t alignment = 16384;
inline uint64_t capacity_upper(uint64_t bytes) {
    return gguf::checked_add(bytes, alignment - 1) & ~(alignment - 1);
}
struct Backing {
    mx::allocator::Buffer buffer{nullptr};
    StorageLease lease;
    ~Backing() { if (buffer.ptr()) mx::allocator::free(buffer); }
};
// The array Data owns both allocation and ledger claim, including escaped
// views. A pager/bank going out of scope must not release a live reader's claim.
inline Tensor allocate(MemoryLedger &ledger, uint64_t bytes, uint64_t upper,
                       const mx::Shape &shape, mx::Dtype dtype, MemoryClass kind,
                       uint64_t generation) {
    require(bytes && upper >= bytes && upper % alignment == 0, "gguf_storage: invalid backing capacity");
    auto reservation = ledger.try_reserve(kind, upper, "gguf-resident-or-slot-v1");
    require(reservation.has_value(), "gguf_storage: managed backing budget insufficient");
    auto owner = std::make_shared<Backing>();
    owner->buffer = mx::allocator::malloc(size_t(bytes));
    require(owner->buffer.ptr() != nullptr, "gguf_storage: MLX allocation failed");
    const auto actual = mx::allocator::allocator().size(owner->buffer);
    require(actual >= bytes && actual <= upper, "gguf_storage: allocator capacity exceeds compiled upper");
    owner->lease = reservation->commit({0x544347475546ull,
        uint64_t(reinterpret_cast<uintptr_t>(owner->buffer.ptr())), uint64_t(actual), generation});
    return Tensor(owner->buffer, shape, dtype, [owner](mx::allocator::Buffer) {});
}

} // namespace tc::streaming::gguf_storage
