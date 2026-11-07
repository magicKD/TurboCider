// W8A8 Linear as mlx Custom Primitive.
// Dispatches quantize + INT8 matmul (prefill) or FP MV (decode) via mlx's CommandEncoder.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <mutex>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace MTL {
class ComputePipelineState;
}

namespace cider {

namespace mx = mlx::core;

// Original Comfy ConvRot codes, with no second weight quantization. The
// caller supplies the already rotated activation; the primitive quantizes
// it per token in FP32 and retains FP32 output until the model's I/O cast.
// Unlike the historical perchannel_linear path, M=1 also executes INT8 GEMM.
bool convrot_w8a8_available();
struct ConvrotW8A8Stats { uint64_t projections = 0, activation_rows = 0; };
ConvrotW8A8Stats convrot_w8a8_stats();
mx::array convrot_linear(const mx::array& rotated,
                        const mx::array& signed_codes,
                        const mx::array& row_scales,
                        mx::StreamOrDevice s = {});
// Fuse only the final model I/O cast into the dequantizing store. Integer
// accumulation and both FP32 scale multiplications match convrot_linear;
// any bias/LoRA/residual must still follow the original typed boundary.
mx::array convrot_linear_typed(const mx::array& rotated,
                              const mx::array& signed_codes,
                              const mx::array& row_scales,
                              mx::Dtype output_dtype,
                              mx::StreamOrDevice s = {});

class ConvrotW8A8Linear : public mx::Primitive {
 public:
  explicit ConvrotW8A8Linear(mx::Stream stream,
                            mx::Dtype output_dtype = mx::float32)
      : mx::Primitive(stream), output_dtype_(output_dtype) {}
  void eval_cpu(const std::vector<mx::array>&,
                std::vector<mx::array>&) override {
    throw std::runtime_error("ConvrotW8A8Linear: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>&,
                std::vector<mx::array>&) override;
  const char* name() const override { return "ConvrotW8A8Linear"; }
  bool is_equivalent(const mx::Primitive& other) const override {
    const auto* typed = dynamic_cast<const ConvrotW8A8Linear*>(&other);
    return typed && output_dtype_ == typed->output_dtype_;
  }

 private:
  mx::Dtype output_dtype_;
};

// ── Custom Primitive ─────────────────────────────────────────────
// Inputs: [x(M,K) float16, w(N,K) int8, scale_w(N) float32, bias(N) float16]
// Output: [y(M,N) float16]
//
// M > 1: quantize activation + INT8 GEMM (prefill)
// M == 1: FP activation × dequant weight MV (decode)
class W8A8Linear : public mx::Primitive {
 public:
  explicit W8A8Linear(mx::Stream stream, const std::string& kernel_dir)
      : mx::Primitive(stream), kernel_dir_(kernel_dir) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("W8A8Linear: CPU not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "W8A8Linear";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    return true;
  }

 private:
  std::string kernel_dir_;
};


// ── Raw INT32 Matmul Primitive ───────────────────────────────────
// Inputs: [A(M,K) int8, B(K,N) int8]
// Output: [C(M,N) int32]   (bit-exact, no dequant)
class Int8MatMulInt32 : public mx::Primitive {
 public:
  explicit Int8MatMulInt32(mx::Stream stream, const std::string& kernel_dir)
      : mx::Primitive(stream), kernel_dir_(kernel_dir) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("Int8MatMulInt32: CPU not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "Int8MatMulInt32";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    return true;
  }

 private:
  std::string kernel_dir_;
};

// ── Public API ───────────────────────────────────────────────────
mx::array perchannel_linear(
    const mx::array& x,       // [M, K] float16
    const mx::array& w,       // [N, K] int8
    const mx::array& scale_w, // [N] float32
    const mx::array& bias,    // [N] float16
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});


// ── Raw INT32 matmul (for bit-exact testing) ─────────────────────
mx::array int8_matmul_int32(
    const mx::array& a,       // [M, K] int8
    const mx::array& b,       // [K, N] int8
    const std::string& kernel_dir,
    mx::StreamOrDevice s = {});

}  // namespace cider
