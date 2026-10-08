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
#include <span>
#include "../core/ane_weight_code_cache_report.hpp"

namespace tc::ane {

enum class Kind { Matmul, SwiGLU, GELU };
enum class DType { FP16, BF16, FP32 };
enum class BackendKind { PublicCoreML, PrivateANE };
enum class PartitionAxis { Rows, IntermediateChannels };
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

// Opaque, already-produced Metal storage. No ObjC/MLX dependency in the graph
// contract. Owner keeps the allocation (not just its MTL object) out of the
// caller's allocator cache until finish. Input is immutable during the job;
// output is independent caller-owned storage, published only on success.
struct DeviceMatrixView {
    void *buffer = nullptr;
    size_t buffer_bytes = 0, offset_bytes = 0;
    int rows = 0, cols = 0;
    size_t row_stride_bytes = 0;
    DType dtype = DType::FP16;
    std::shared_ptr<void> owner;
    std::weak_ptr<void> allocation_identity{}; // stable allocation generation, never a raw-address-only cache key
};
struct DeviceAdapterInput { DeviceMatrixView gate, up, hidden; };

enum class DeviceWeightEncoding : uint32_t {
    Dense, AffineQ4, AffineQ8, GgufQ4_0, GgufQ4_K, GgufQ8_0, GgufQ6_K,
    ConvrotQ8Signed, ConvrotQ8Packed
};
enum class W8Basis : uint32_t { SylvesterDH, ComfyH256 };
// Immutable physical source matrix, independent of an executor's W8/FP16
// representation. cols/pitch remain the FULL source row when staging a slice.
struct DeviceWeightView {
    void *buffer = nullptr;
    size_t buffer_bytes = 0, offset_bytes = 0, row_stride_bytes = 0;
    int rows = 0, cols = 0;
    DeviceWeightEncoding encoding = DeviceWeightEncoding::Dense;
    DType dense_dtype = DType::BF16;
    int group_size = 32;
    std::optional<DeviceMatrixView> scales, offsets;
    std::shared_ptr<void> owner;
    std::weak_ptr<void> allocation_identity{};
    bool immutable_generation = false; // opt-in promise: contents/metadata do not change while this generation lives
    // Raw GGUF importer-only promise: complete logical matrix bytes identify
    // one verified source generation/tensor across physical refills. This is
    // NOT an allocation identity or permission to reuse mutable data. A scale
    // cache may weakly retain the tag, never the packed matrix or source lease.
    std::weak_ptr<void> logical_content_identity{};
};
struct W8StageSpec {
    int row_begin = 0, rows = 0, column_begin = 0, columns = 0;
    int rotation_block = 128;
    uint64_t rotation_seed = 20260930;
    bool transpose = false; // A8 channel-major vs W8 out/in
    W8Basis basis = W8Basis::SylvesterDH;
    int activation_group_size = 0; // 0: per-token; 256: explicit Comfy A8 groups
};
// Logical selection over an unchanged physical source. In particular a down
// channel slice keeps the full packed row pitch/metadata geometry and owner.
struct DeviceWeightRegion { DeviceWeightView source; W8StageSpec selection; };

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

// Parse the runtime template's geometry/ABI only. Private executors emit their
// own native MIL and do not load or claim validation of its Core ML artifact.
GraphShape runtime_template_shape(const std::filesystem::path &manifest);
bool runtime_template_w8a8(const std::filesystem::path &manifest);
struct RuntimeArtifactSnapshot {
    GraphShape shape;
    W8Basis basis = W8Basis::SylvesterDH;
    std::filesystem::path compiled_model;
    std::shared_ptr<void> lease;
};
RuntimeArtifactSnapshot snapshot_runtime_w8a8(const std::filesystem::path &manifest);

// Per-request activation corrections, NEVER merged into weight slots.
// gate/up have the full launch row count and FFN width. The worker scales up
// with its base up weights, and restores hidden before GPU down-LoRA uses it.
// hidden has the SAME dtype as launch's output_dtype, including BF16 range.
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
    float headroom_start_scale = 1.f;
    float headroom_scale = 1.f;
    // Successful lookahead submissions, not proof of physical GPU/ANE overlap.
    uint64_t activation_prefetches = 0;
    double activation_wait_seconds = 0;
};
struct WeightCacheStats {
    bool enabled = false;
    uint64_t hits = 0, misses = 0, entries = 0, bytes = 0, evictions = 0;
};
struct StagePipelineStats {
    bool specialized = false;
    uint64_t variants = 0;
};

// Model/scheduler contract, independent of Core ML and private ObjC classes.
// All executors share the same source views, row chunks, adapter corrections,
// output ownership and all-or-GPU failure semantics. One owning host thread.
class Executor {
  public:
    virtual ~Executor() = default;
    virtual BackendKind backend() const = 0;
    virtual const GraphShape &shape() const = 0;
    virtual size_t slot_bytes() const = 0;
    virtual size_t estimated_bytes() const = 0;
    virtual double load_seconds() const = 0;
    virtual bool self_test(std::string &error) = 0;
    virtual void stage_weights(std::vector<WeightView> weights) = 0;
    virtual RunResult wait_stage() = 0;
    virtual void launch(MatrixView input, uint16_t *output, size_t output_elements,
                        DType output_dtype = DType::FP16,
                        std::optional<AdapterInput> adapter = std::nullopt) = 0;
    virtual RunResult finish() = 0;
    virtual bool supports_device_io() const { return false; }
    virtual bool supports_device_weights() const { return false; }
    virtual bool supports_device_weight_regions() const { return false; }
    virtual bool supports_weight_prefetch() const { return false; }
    virtual std::string data_path() const { return "fp16"; }
    virtual std::string weight_recipe() const { return {}; }
    virtual WeightCacheStats weight_cache_stats() const { return {}; }
    virtual WeightCodeCacheReport weight_code_cache_stats() const { return {}; }
    virtual StagePipelineStats stage_pipeline_stats() const { return {}; }
    virtual bool device_submission_fence_enabled() const { return false; }
    virtual bool activation_lookahead_enabled() const { return false; }
    virtual void stage_device_weights(std::vector<DeviceWeightView>) {
        throw CapabilityError("executor has no device-weight staging path");
    }
    virtual void stage_device_weight_regions(std::vector<DeviceWeightRegion>) {
        throw CapabilityError("executor has no physical-source region staging path");
    }
    // Exactly one future bank; source owners remain leased until the producer
    // completes. Activation is allowed only after the current consumer joins.
    virtual void prefetch_device_weight_regions(std::vector<DeviceWeightRegion>) {
        throw CapabilityError("executor has no future weight bank");
    }
    virtual std::optional<RunResult> activate_prefetched_weights(std::span<const DeviceWeightRegion>) { return std::nullopt; }
    virtual void discard_prefetched_weights() {}
    virtual void launch_device(DeviceMatrixView, DeviceMatrixView,
                               std::optional<DeviceAdapterInput> = std::nullopt) {
        throw CapabilityError("executor has no device-buffer I/O path");
    }
    // Base-down partial only. Retain FP32 until a channel join's ONE final
    // model-dtype rounding. Does not imply an FP32 ANE graph or hidden ABI.
    virtual bool supports_fp32_device_output() const { return false; }
    virtual int activation_group_size() const { return 0; }
    virtual int hidden_activation_group_size() const { return 0; }
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
class RuntimeGraph : public Executor {
  public:
    explicit RuntimeGraph(const std::filesystem::path &manifest,
                          size_t memory_budget_bytes, bool cpu_only = false,
                          bool scalar_staging = false,
                          std::optional<GraphGeometry> expected = std::nullopt);
    ~RuntimeGraph() override;
    RuntimeGraph(const RuntimeGraph &) = delete;
    RuntimeGraph &operator=(const RuntimeGraph &) = delete;

    BackendKind backend() const override { return BackendKind::PublicCoreML; }
    const GraphShape &shape() const override;
    size_t slot_bytes() const override;
    size_t estimated_bytes() const override;
    double load_seconds() const override;

    // Checks TWO weight sets and nonzero input on this actual shape. Caller
    // must run this before arming a model route; it leaves no user weights.
    bool self_test(std::string &error) override;

    // Graph input order: matmul {w}; SwiGLU {gate,up,down}; GELU {up,down}.
    // Asynchronous staging can overlap the model's GPU attention.
    void stage(std::vector<MatrixView> weights);
    void stage_weights(std::vector<WeightView> weights) override;
    // MatMul only: append dense [out,in] source rows directly into ONE weight
    // slot. Q/K/V remain separate checkpoint tensors; no 3 GiB GPU pack bank.
    // Every row must be covered exactly once. Inputs stay borrowed until join.
    void stage_matmul_parts(std::vector<MatrixView> parts);
    RunResult wait_stage() override;

    // Input rows must be a positive multiple of the graph's chunk rows.
    // Output is caller-owned packed FP16 or BF16, never borrowed from the next
    // job. BF16 can restore large outputs after FP16 headroom rescaling.
    // This deliberately does not depend on MLX and does not evaluate GPU work.
    void launch(MatrixView input, uint16_t *output, size_t output_elements,
                DType output_dtype = DType::FP16,
                std::optional<AdapterInput> adapter = std::nullopt) override;
    RunResult finish() override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tc::ane
