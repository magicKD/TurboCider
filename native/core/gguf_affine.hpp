#pragma once
#include "gguf_decode.hpp"
#include <array>

namespace tc::gguf {
enum class AffinePart { codes, scales, biases };
// Lossless layout conversion for the existing MLX affine Q4/Q8 format.
// No weight/activation quantizer; codes and FP16 scale identity are preserved.
uint64_t pack_native_affine(const PackedMatrix &, AffinePart, std::span<std::byte>,
                            const std::atomic<bool> *cancel = nullptr);
// One synchronous CPU pass; no allocation or I/O. Targets are codes/scales/
// biases and must not overlap each other or the source. Caller discards ALL
// partial fields on failure. Same bytes/rounding as the legacy oracle above.
uint64_t pack_native_affine_all(const PackedMatrix &,const std::array<std::span<std::byte>,3> &,
                               const std::atomic<bool> *cancel=nullptr,DecodeOptions = {});
} // namespace tc::gguf
