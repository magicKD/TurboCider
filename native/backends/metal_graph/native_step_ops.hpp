#pragma once

#include "native_ffn.hpp"
#include "kernels/step_ops.hpp"

namespace tc::metal_graph {

class NativeStepOps {
  public:
    // Six reusable PSOs, built before graph encoding. No activation allocation.
    static NativeStepOps prepare(Backend&);
    static constexpr std::string_view precision_recipe() { return step_ops::precision_recipe; }
    static void validate_copy(const MatrixView& source, const MatrixView& destination);
    static void validate_euler_bf16(const MatrixView& latents, const MatrixView& noise,
                                  float dt, const MatrixView& output);
    static void validate_invalid_flag(const BufferView& invalid);
    static void validate_finite(const MatrixView&, const BufferView& invalid);

    // Same BF16 or FP32 type/shape; preserves each operand's physical row stride.
    // Output must not overlap a source, including partial overlap of one backing.
    void encode_copy(CommandGraph&, const MatrixView& source, const MatrixView& destination) const;
    // Pipeline semantics: BF16(latents + BF16(noise * BF16(dt))). No FP32 fused
    // Euler shortcut; output is independent from latents and noise.
    void encode_euler_bf16(CommandGraph&, const MatrixView& latents, const MatrixView& noise,
                           float dt, const MatrixView& output) const;
    // Encode reset FIRST in each run's graph, then generation/update and finite
    // checks. Checks only OR bit 0 and never clear an earlier detected failure.
    // Caller also excludes reset/flag aliases with every other graph operand.
    void encode_reset_invalid_flag(CommandGraph&, const BufferView& invalid) const;
    void encode_finite(CommandGraph&, const MatrixView&, const BufferView& invalid) const;

    // ticket MUST belong to this run's reset/check graph, with no subsequent or
    // concurrent writes to the flag. Read only after real success and physical
    // completion; reject non-Shared storage, failed tickets and any nonzero bit.
    static void require_finite(const Ticket& ticket, const BufferView& invalid);
  private:
    NativeStepOps() = default;
    Pipeline copy_bf16_, copy_f32_, euler_, reset_, finite_bf16_, finite_f32_;
};

} // namespace tc::metal_graph
