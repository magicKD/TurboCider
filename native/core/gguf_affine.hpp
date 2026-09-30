#pragma once
#include "gguf_decode.hpp"

namespace tc::gguf {
enum class AffinePart { codes, scales, biases };
// Lossless layout conversion for the existing MLX affine Q4/Q8 format.
// No weight/activation quantizer; codes and FP16 scale identity are preserved.
uint64_t pack_native_affine(const PackedMatrix &, AffinePart, std::span<std::byte>,
                            const std::atomic<bool> *cancel = nullptr);
} // namespace tc::gguf
