#pragma once

#include "../../runtime/streaming/layout.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include "../../core/common.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tc::z_image {

struct StreamingWorkload {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t caption_rows = 0;
    uint32_t steps = 0;
    uint32_t ane_mlp_prefix_channels = 0;
    bool fp32_scales = false;
};

enum class StreamingConversion { copy, signed_q8, bf16, scales, biases };

struct StreamingWeightOptions {
    uint32_t mlp_prefix_channels = 0;
    bool fp32_scales = false;
};

// Model-owned materialization plan shared by the descriptor and reader.
// The generic executor sees only source ranges, destination bytes and tickets.
struct StreamingTensor {
    std::string name, suffix, dtype;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0, data_end = 0, file_offset = 0, bytes = 0;
    uint32_t artifact = 0;
    StreamingConversion conversion = StreamingConversion::copy;
    streaming::SourceRange source;
};

struct StreamingSuffixPack {
    uint64_t source_offset = 0, destination_offset = 0;
    uint64_t rows = 0, row_bytes = 0, skip_bytes = 0;
};

// Planning data only: the derived artifact is a recipe identity, not verified
// file contents or a Core ML partition authority. No fd or payload is created.
struct SuffixPackingRecord {
    streaming::SourceRange source;
    uint64_t destination_offset = 0, destination_bytes = 0;
    uint32_t first_gpu_channel = 0;
};
struct GpuSuffixPlan {
    streaming::Descriptor descriptor;
    std::vector<SuffixPackingRecord> packing;
    std::string recipe_digest;
    uint64_t setup_read_bytes = 0, setup_write_bytes = 0;
};

class GpuSuffixSource;

// Header-only view of the single-file Comfy BF16/INT8 transformer.  Construction
// reads the safetensors prefix and JSON header only.  It never reads tensor
// payloads, creates MLX arrays, allocates Metal buffers, or starts workers.
class StreamingMetadata {
  public:
    explicit StreamingMetadata(const std::string &checkpoint,
                               StreamingWeightOptions options = {});
    // Public adapters pass the request-scoped lease captured during probe.
    // This constructor never reopens the named path.
    explicit StreamingMetadata(
        std::shared_ptr<const streaming::SourceLease> lease,
        std::string logical_id = "transformer",
        StreamingWeightOptions options = {});
    ~StreamingMetadata();

    StreamingMetadata(const StreamingMetadata &) = delete;
    StreamingMetadata &operator=(const StreamingMetadata &) = delete;

    streaming::Descriptor describe(const StreamingWorkload &) const;
    // Portable semantic identity for the v2 migration. Requires content proof
    // already captured by the native verifier; never hashes payloads here.
    // The request lease/snapshot remains separate and must still be revalidated.
    streaming::Descriptor describe_verified(const StreamingWorkload &) const;
    // Internal H1 layout preview. Covers fixed noise refiners and all main
    // layers; context refiners remain whole. Execution requires a separately
    // verified hybrid source/owner and cannot use the exact GPU adapter.
    GpuSuffixPlan describe_gpu_suffix(const StreamingWorkload &,
                                     uint32_t first_gpu_channel) const;
    // Requires native verified parent contents. Rebuilds its own recipe;
    // caller-supplied plans never grant source authority. No GPU/Core ML work.
    std::unique_ptr<GpuSuffixSource> materialize_gpu_suffix(
        const StreamingWorkload &, uint32_t first_gpu_channel,
        std::atomic<bool> &cancelled, const Event &event = {}) const;
    void check_unchanged() const;
    const streaming::SourceLease &lease() const;
    std::shared_ptr<const streaming::SourceLease> lease_ptr() const;

    uint32_t block_count() const noexcept { return 30; }
    uint32_t tensors_per_block() const noexcept;
    bool convrot() const noexcept;
    uint64_t scratch_bytes_per_slot() const noexcept;
    uint64_t packed_bytes() const noexcept;
    const StreamingWeightOptions &options() const noexcept;
    const std::vector<StreamingTensor> &fixed_records() const;
    const std::vector<StreamingTensor> &block_records(uint32_t block) const;
    const std::vector<StreamingSuffixPack> &suffix_packs() const;
    uint64_t block_bytes() const noexcept;
    uint64_t fixed_bytes() const noexcept;
    const std::string &snapshot_identity() const noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
    void parse_checkpoint();
    void prepare_materializations();
};

// A completed, request-owned derived file, accessible only through read-only
// duplicates. The recipe descriptor stays unchanged after materialization;
// content_digest is separate execution evidence binding actual packed bytes.
// This does not provide Core ML or public hybrid execution authority.
class GpuSuffixSource final {
  public:
    ~GpuSuffixSource();
    GpuSuffixSource(const GpuSuffixSource &) = delete;
    GpuSuffixSource &operator=(const GpuSuffixSource &) = delete;
    const GpuSuffixPlan &plan() const noexcept;
    const std::string &content_digest() const noexcept;
    uint64_t verification_read_bytes() const noexcept;
    const streaming::SourceFileIdentity &parent_file() const;
    streaming::OwnedSourceFd duplicate_fd(uint32_t artifact) const;
    void check_unchanged() const;

  private:
    friend class StreamingMetadata;
    struct State;
    explicit GpuSuffixSource(std::unique_ptr<State>);
    std::unique_ptr<State> state_;
};

// Private metadata/plan shadow for the existing ZImageWeightStream contract.
// It intentionally exposes no execution bridge yet.
class StreamingPlanView {
  public:
    StreamingPlanView(const std::string &checkpoint,
                      const StreamingConfig &config,
                      const StreamingWorkload &workload);
    StreamingPlanView(std::shared_ptr<const streaming::SourceLease> lease,
                      const StreamingConfig &config,
                      const StreamingWorkload &workload);
    // A native verified lease selects portable content identity; metadata-only
    // leases preserve legacy snapshot identity. Compile and execution rebuild
    // must use the same request lease, never reopen by path to infer identity.

    StreamingPlanView(const StreamingPlanView &) = delete;
    StreamingPlanView &operator=(const StreamingPlanView &) = delete;

    const StreamingMetadata &metadata() const noexcept { return metadata_; }
    const streaming::Descriptor &descriptor() const noexcept {
        return descriptor_;
    }
    const streaming::Layout &layout() const noexcept { return layout_; }
    const streaming::SourceLease &lease() const { return metadata_.lease(); }
    std::shared_ptr<const streaming::SourceLease> lease_ptr() const {
        return metadata_.lease_ptr();
    }

  private:
    StreamingMetadata metadata_;
    streaming::Descriptor descriptor_;
    streaming::Layout layout_;
    void validate() const;
};

} // namespace tc::z_image
