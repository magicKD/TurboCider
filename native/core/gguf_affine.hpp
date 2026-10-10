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

// Q4_K_M is a file recipe mixing Q4_K/Q5_K/Q6_K tensors. Preserve Q4_K
// codes and Q5_K integer codes (expanded to Q8); expand their subgroup
// scale/min coefficients to finite FP16/BF16 affine
// metadata. Q6_K has 16-element scales (unsupported by MLX affine QMM), so
// explicitly requantize ONLY each 32-element group to affine Q8. No full
// dense matrix, heap allocation, I/O or implicit change to the legacy API.
// The coefficient rounding/Q6 recipe must be disclosed/validated by its owner.
uint64_t pack_k_affine_all(const PackedMatrix &,const std::array<std::span<std::byte>,3> &,
                          const std::atomic<bool> *cancel=nullptr,DecodeDType metadata=DecodeDType::f16,
                          DecodeOptions = {});
} // namespace tc::gguf
