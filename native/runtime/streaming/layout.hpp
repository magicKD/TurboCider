#pragma once

#include "../../core/streaming_contracts.hpp"
#include <vector>

namespace tc::streaming {

inline constexpr uint32_t max_stages = 64, max_blocks = 65536, max_slots = 64;
inline constexpr uint32_t max_fields_per_block = 4096, max_passes = 4096;
inline constexpr uint64_t max_descriptor_fields = 1048576;
inline constexpr uint64_t max_identity_bytes = 16ull << 20;

// These are metadata identities, not execution or memory certifications.
// A snapshot identity (stat/index) must never be presented as a content hash.
enum class SourceIdentityKind { snapshot, content_sha256 };
struct SourceArtifact {
    std::string id, identity;
    uint64_t bytes = 0;
    SourceIdentityKind identity_kind = SourceIdentityKind::snapshot;
};
struct SourceRange {
    uint32_t artifact = 0;
    uint64_t offset = 0, bytes = 0;
    std::string tensor, dtype;
    std::vector<uint64_t> shape;
    bool operator==(const SourceRange &) const = default;
};
struct Materialization {
    std::string format, storage_mode, conversion;
    std::vector<uint64_t> shape;
    std::vector<SourceRange> reads;
    // A same-block earlier field. Derived spans do not read the file again.
    std::string derived_from;
    uint64_t derived_offset = 0;
    bool operator==(const Materialization &) const = default;
};
struct PassSpec {
    uint32_t step = 0;
    std::string phase;
    std::vector<uint64_t> shape;
};

struct FieldSpec {
    std::string name;       // Stable field/binding identity within a block.
    std::string storage_id; // Destination backing identity; NOT a source tensor.
    uint64_t bytes = 0, alignment = 1;
    std::optional<Materialization> materialization = {};
    bool operator==(const FieldSpec &) const = default;
};
struct BlockSpec {
    uint32_t id = 0;
    std::string layout_class;
    std::vector<FieldSpec> fields;
    bool streamable = true;
    bool safe_boundary_after = true;
};
struct StageDescriptor {
    std::string id;
    std::vector<BlockSpec> blocks;
    std::optional<StreamingStageConfig> fixed_policy;
    uint32_t min_slots = 1, max_slots = 3, max_group_size = 1, min_prefix = 0;
    uint32_t pass_count = 1;
    // A content/shape/format/reader identity supplied by the model adapter.
    // An empty identity cannot be used for execution certification.
    std::string adapter_revision;
    std::vector<PassSpec> passes = {};
};
struct Descriptor {
    std::string model, checkpoint_identity, backend_revision;
    std::vector<StageDescriptor> stages;
    std::vector<SourceArtifact> artifacts = {};
    std::map<std::string, std::string> workload = {};
};
struct Group {
    uint32_t id = 0, pool = 0, slot = 0;
    std::vector<uint32_t> blocks;
    // Fixed per-block-position fields: no relocation is assumed.
    std::vector<uint64_t> field_bytes;
    uint64_t bytes = 0;
};
struct SlotLayout {
    std::vector<uint64_t> field_capacity;
    uint64_t capacity_bytes = 0;
};
struct PoolLayout {
    uint32_t id = 0;
    std::string layout_class;
    std::vector<SlotLayout> slots;
    uint64_t capacity_bytes = 0;
};
struct StageLayout {
    std::string id;
    bool resident = false, inherited = false;
    uint32_t prefix = 0, group_size = 0, slot_count = 0, distance = 0, workers = 0;
    uint32_t pass_count = 1;
    std::vector<Group> groups;
    std::vector<PoolLayout> pools;
    uint64_t prefix_bytes = 0, peak_pool_bytes = 0;
    uint64_t suffix_content_bytes_per_pass = 0;
    // Null means unknown, not zero. Reads are logical file bytes, NOT physical
    // disk traffic (which depends on the OS page cache).
    std::optional<uint64_t> source_read_bytes_per_pass = {};
    std::optional<uint64_t> prefix_source_read_bytes = {};
};
struct Layout {
    std::vector<StageLayout> stages;
    // Immutable after compilation. This is not a whole-request memory upper.
    std::string canonical, digest;
    // Complete source/binding/pass metadata does not establish a whole-request
    // memory upper, immutable artifact contents, or execution qualification.
    bool materializations_complete = false;
};

// Pure metadata operation; never probes RAM or changes user parameters.
Layout compile_layout(const StreamingConfig &, const Descriptor &);

} // namespace tc::streaming
