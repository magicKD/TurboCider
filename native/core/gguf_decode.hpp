#pragma once

#include "gguf_directory.hpp"
#include <atomic>
#include <cstddef>
#include <span>

namespace tc::gguf {

class DecodeError final : public std::exception {
  public:
    explicit DecodeError(const char *reason) noexcept : reason_(reason) {}
    const char *what() const noexcept override { return reason_; }
  private:
    const char *reason_;
};

enum class DecodeDType { f32, f16, bf16 };
inline uint32_t dtype_bytes(DecodeDType dtype) {
    switch (dtype) {
    case DecodeDType::f32: return 4;
    case DecodeDType::f16: case DecodeDType::bf16: return 2;
    }
    throw DecodeError("gguf_decode: invalid target dtype");
}
struct PackedMatrix {
    std::span<const std::byte> bytes;
    uint32_t type = 0;
    uint64_t rows = 0, columns = 0;
};
struct DecodeSlice {
    uint64_t row_begin = 0, rows = 0, column_begin = 0, columns = 0;
};
struct DecodeTarget {
    std::span<std::byte> bytes;
    DecodeDType dtype = DecodeDType::bf16;
    uint64_t row_stride = 0, column_stride = 0;
};
struct DecodeReceipt {
    uint64_t source_bytes_processed = 0, elements_written = 0, bytes_written = 0;
    uint64_t scratch_bytes = 0;
    uint64_t simd_blocks = 0;
};
struct DecodeOptions { bool use_simd = true; };

float fp16_to_float(uint16_t value);
uint16_t float_to_fp16_rne(float value);
uint16_t float_to_bf16_rne(float value);
// Explicit same-width importer alias conversion. No allocation; input may be
// partially overwritten on failure and must not be published by the owner.
void bf16_to_fp16_inplace(std::span<std::byte>,const std::atomic<bool> *cancel=nullptr,DecodeOptions = {});

// Synchronous CPU-only; no I/O, MLX objects, heap allocation, or retained spans.
// Caller must discard partial target contents on failure/cancellation. The
// fixed block scratch is <=1024 bytes, independent of matrix/layer size.
DecodeReceipt decode_cpu_into(const PackedMatrix &, const DecodeSlice &,
                              const DecodeTarget &, const std::atomic<bool> *cancel = nullptr,
                              DecodeOptions = {});

// Bounded embedding gather: duplicates and order preserved. row_indices are
// caller-owned; no hidden full-vocabulary materialization or deduplication map.
DecodeReceipt decode_cpu_gather(const PackedMatrix &, std::span<const uint64_t> row_indices,
                                uint64_t column_begin, uint64_t columns,
                                const DecodeTarget &, const std::atomic<bool> *cancel = nullptr,
                                DecodeOptions = {});

} // namespace tc::gguf
