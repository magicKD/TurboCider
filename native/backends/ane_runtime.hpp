#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <variant>

namespace tc::ane {

enum class Kind { Matmul, SwiGLU, GELU };
enum class DType { FP16, BF16, FP32 };
struct MemoryBudgetError : std::runtime_error { using std::runtime_error::runtime_error; };
struct CapabilityError : std::runtime_error { using std::runtime_error::runtime_error; };

// The model's logical geometry, independent of the selected chunk/tiles.
// Validate it before memory admission, even when the backend must fall back.
struct GraphGeometry {
    Kind kind;
    int hidden, width;
    bool require_lora_inputs = false;
};

// Borrowed immutable row-major matrix. Sources must remain alive and readable
// until wait_stage()/finish(); runtime never changes master model weights.
struct MatrixView {
    const void *data = nullptr;
    size_t bytes = 0;
    int rows = 0, cols = 0;
    size_t row_stride_bytes = 0; // 0 means tightly packed
    DType dtype = DType::FP16;
};

// MLX affine codes, not raw GGUF blocks: unsigned low-bit-first uint32 words,
// logical [out,in], value = scale[group] * code + offset[group]. Legacy MLX
// GGUF imports expose this representation; raw blocks instead use GgufView.
// ConvRot/NVFP4 are different
// formats and must never be passed as affine. All buffers are borrowed.
struct AffineView {
    const void *data = nullptr;
    size_t bytes = 0;
    int rows = 0, cols = 0;
    size_t row_stride_bytes = 0;
    int group_size = 32, bits = 4;
    MatrixView scales;
    std::optional<MatrixView> offsets;
};
// Raw on-disk GGML blocks, not MLX affine. Callers retain immutable packed
// buffers through wait_stage/finish; no intermediate full dense matrix.
struct GgufView {
    const void *data = nullptr;
    size_t bytes = 0;
    int rows = 0, cols = 0;
    size_t row_stride_bytes = 0;
    uint32_t ggml_type = 8;
};
// Original signed rotated codes and FP32 row scales. The staging recipe is
// effective W = diag(scale) * codes * Comfy H256^T; it does not requantize W.
struct ConvrotView {
    const int8_t *data = nullptr;
    size_t bytes = 0;
    int rows = 0, cols = 0;
    size_t row_stride_bytes = 0;
    MatrixView row_scales;
};
// Explicit legacy MLX packed ConvRot, NEVER ordinary affine/GGUF. Codes are
// unsigned q+128; every group's scale/offset must be the same row scale and
// -128*scale. Inverse H uses that stored scale, not original F32 provenance.
struct ConvrotAffineView { AffineView packed; };
using WeightView = std::variant<MatrixView, AffineView, GgufView, ConvrotView, ConvrotAffineView>;

struct GraphShape {
    Kind kind = Kind::SwiGLU;
    int rows = 0, hidden = 0, width = 0;
    int tile_k = 0, tile_n = 0;
    bool lora_inputs = false;
    int output_width() const { return kind == Kind::Matmul ? width : hidden; }
};

// Per-request activation corrections, NEVER merged into weight slots.
// gate/up have the full launch row count and FFN width. The worker scales up
// with its base up weights, and restores hidden before GPU down-LoRA uses it.
struct AdapterInput {
    MatrixView gate, up;
    uint16_t *hidden = nullptr;
    size_t hidden_elements = 0;
};

struct RunResult {
    bool ok = false;
    std::string error;
    double stage_seconds = 0, input_seconds = 0, prediction_seconds = 0;
    double output_seconds = 0, total_seconds = 0;
    uint64_t calls = 0, copied_output_bytes = 0;
    uint64_t overflow_retries = 0;
    float headroom_scale = 1.f;
};

// Explicit opt-in runtime-weight backend; never selected by auto routing.
// See docs/status/runtime-ane-component-2026-09-28.md for validation limits.
// A single fixed-shape, runtime-weight graph and ONE layer's IOSurface slots.
// No model-dependent tensors are stored in the compiled graph. All Core ML
// calls run on one persistent dispatch thread; conversions use a separate
// system pool. Public Core ML CPU+NE policy does not prove hardware residency.
// This low-level executor NEVER returns uncomputed rows as successful output.
// Its model-facing caller MUST recompute the entire requested range on GPU
// after any false result (including partially completed multi-chunk jobs).
// Public calls require one owning host thread; this is not a concurrent queue.
// Source/output buffers must outlive the asynchronous job and finish().
class RuntimeGraph {
  public:
    explicit RuntimeGraph(const std::filesystem::path &manifest,
                          size_t memory_budget_bytes, bool cpu_only = false,
                          bool scalar_staging = false,
                          std::optional<GraphGeometry> expected = std::nullopt);
    ~RuntimeGraph();
    RuntimeGraph(const RuntimeGraph &) = delete;
    RuntimeGraph &operator=(const RuntimeGraph &) = delete;

    const GraphShape &shape() const;
    size_t slot_bytes() const;
    size_t estimated_bytes() const;
    double load_seconds() const;

    // Checks TWO weight sets and nonzero input on this actual shape. Caller
    // must run this before arming a model route; it leaves no user weights.
    bool self_test(std::string &error);

    // Graph input order: matmul {w}; SwiGLU {gate,up,down}; GELU {up,down}.
    // Asynchronous staging can overlap the model's GPU attention.
    void stage(std::vector<MatrixView> weights);
    void stage_weights(std::vector<WeightView> weights);
    // MatMul only: append dense [out,in] source rows directly into ONE weight
    // slot. Q/K/V remain separate checkpoint tensors; no 3 GiB GPU pack bank.
    // Every row must be covered exactly once. Inputs stay borrowed until join.
    void stage_matmul_parts(std::vector<MatrixView> parts);
    RunResult wait_stage();

    // Input rows must be a positive multiple of the graph's chunk rows.
    // Output is caller-owned packed FP16 or BF16, never borrowed from the next
    // job. BF16 can restore large outputs after FP16 headroom rescaling.
    // This deliberately does not depend on MLX and does not evaluate GPU work.
    void launch(MatrixView input, uint16_t *output, size_t output_elements,
                DType output_dtype = DType::FP16,
                std::optional<AdapterInput> adapter = std::nullopt);
    RunResult finish();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tc::ane
