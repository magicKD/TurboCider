#pragma once

#include "backend.hpp"
#include "kernels/dense_ffn.hpp"

namespace tc::metal_graph {

// Row-major logical matrix over a retained native allocation. columns excludes
// physical row padding. A Qwen fused [2F,H] uses gate/up views with distinct
// byte offsets and the original leading row stride; down retains its F stride.
struct MatrixView {
    BufferView storage;
    uint32_t rows = 0;
    uint32_t columns = 0;
    size_t row_stride_bytes = 0;
};

struct FfnBindings {
    MatrixView input;       // [M,H] BF16
    MatrixView gate;        // [F,H] BF16
    MatrixView up;          // [F,H] BF16
    MatrixView down;        // [H,F] BF16
    MatrixView hidden;      // [M,C] BF16, compact or physically padded
    MatrixView output;      // [M,H] BF16 or explicit FP32 channel partial
    uint32_t first_channel = 0;
};

class NativeFfn {
public:
    // Creates three MPP PSOs outside the execution path. No allocations of
    // activations/weights, no source copies and no command submission here.
    static NativeFfn prepare(Backend&, dense_ffn::Geometry);
    // Explicit cold preparation of the channel branch and FP32 join PSO.
    // Ordinary full-GPU prepare retains its original three-pipeline bundle.
    static NativeFfn prepare_parallel(Backend&, dense_ffn::Geometry);
    dense_ffn::Geometry geometry() const noexcept { return geometry_; }
    static constexpr std::string_view precision_recipe() noexcept {
        return dense_ffn::precision_recipe;
    }
    // Validates the complete binding before recording either dispatch. The
    // graph owns views and copied POD parameters; Backend owns submit/waits.
    void encode(CommandGraph&, const FfnBindings&) const;
    // Split recording is useful when the ANE complement consumes input and a
    // later graph joins down-output partials. Readiness belongs to the caller.
    void encode_gate_up(CommandGraph&, const MatrixView& input,
                        const MatrixView& gate, const MatrixView& up,
                        const MatrixView& hidden, uint32_t first_channel) const;
    void encode_down(CommandGraph&, const MatrixView& hidden,
                     const MatrixView& down, const MatrixView& output,
                     uint32_t first_channel) const;
    // Both branch results stay FP32 until exactly one model BF16 boundary.
    // No readiness or ANE read permission is inferred by recording this node.
    void encode_join_partials(CommandGraph&, const MatrixView& gpu_partial,
                              const MatrixView& ane_partial, const MatrixView& output) const;
    bool has_parallel_join() const noexcept { return bool(join_); }
    static constexpr std::string_view parallel_recipe() noexcept {
        return dense_ffn::parallel_recipe;
    }
private:
    static NativeFfn prepare_impl(Backend&, dense_ffn::Geometry, bool parallel);
    dense_ffn::Geometry geometry_;
    Pipeline gate_up_, down_bf16_, down_f32_, join_;
};

} // namespace tc::metal_graph
