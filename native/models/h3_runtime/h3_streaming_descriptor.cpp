#include "h3_streaming_descriptor.hpp"

#include "../../runtime/memory_manifest.hpp"
#include "h3_streaming_policy.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace tc::h3 {
namespace {

using std::filesystem::path;

void require_metadata(bool ok, const std::string &reason) {
    if (!ok)
        throw std::invalid_argument("h3_streaming_metadata: " + reason);
}

std::string hex_u64(uint64_t value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::string snapshot_fingerprint(const struct stat &status) {
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << status.st_dev << ':' << status.st_ino << ':'
              << status.st_size << ':' << status.st_mtimespec.tv_sec << ':'
              << status.st_mtimespec.tv_nsec << ':'
              << status.st_ctimespec.tv_sec << ':'
              << status.st_ctimespec.tv_nsec;
    return tc::memory_sha256_hex(canonical.str());
}

struct ShardInfo {
    std::string path;
    std::string name;
    uint64_t bytes = 0;
    std::string identity;
};

struct StoreDeleter {
    void operator()(h3_weight_store *store) const noexcept {
        h3_weight_store_free(store);
    }
};

struct MatrixSpec {
    const char *suffix;
    uint64_t rows;
    uint64_t columns;
    const char *field;
};

constexpr MatrixSpec kMatrices[] = {
    {"attn.qkv_proj.weight", H3_DIT_INNER * 3u, H3_DIT_HIDDEN, "qkv"},
    {"attn.out_proj.weight", H3_DIT_HIDDEN, H3_DIT_INNER, "out"},
    {"mlp.fc1.weight", H3_DIT_FFN * 2u, H3_DIT_HIDDEN, "fc1"},
    {"mlp.fc2.weight", H3_DIT_HIDDEN, H3_DIT_FFN, "fc2"},
};

uint64_t checked_matrix_bytes(const MatrixSpec &spec) {
    require_metadata(spec.rows && spec.columns &&
                         spec.rows <= UINT64_MAX / spec.columns,
                     "matrix shape overflow");
    const uint64_t elements = spec.rows * spec.columns;
    require_metadata(elements <= UINT64_MAX / sizeof(uint16_t),
                     "matrix byte count overflow");
    return elements * sizeof(uint16_t);
}

std::string active_ids(const uint8_t *mask) {
    std::ostringstream out;
    bool first = true;
    for (uint32_t block = 0; block < H3_DIT_BLOCKS; ++block) {
        if (!mask[block])
            continue;
        if (!first)
            out << ',';
        first = false;
        out << block;
    }
    return out.str();
}

std::string bool_string(bool value) { return value ? "1" : "0"; }

} // namespace

struct StreamingMetadata::State {
    std::string directory;
    std::unique_ptr<h3_weight_store, StoreDeleter> store;
    std::vector<ShardInfo> shards;
    uint64_t identity = 0;
};

StreamingMetadata::StreamingMetadata(const std::string &transformer_directory)
    : state_(std::make_unique<State>()) {
    require_metadata(!transformer_directory.empty() &&
                         transformer_directory.find('\0') == std::string::npos,
                     "invalid transformer directory");

    std::error_code filesystem_error;
    const path absolute = std::filesystem::absolute(
        path(transformer_directory), filesystem_error);
    require_metadata(!filesystem_error &&
                         std::filesystem::is_directory(absolute,
                                                        filesystem_error) &&
                         !filesystem_error,
                     "transformer directory is missing");
    state_->directory = absolute.lexically_normal().string();

    std::vector<path> shard_paths;
    for (const auto &entry : std::filesystem::directory_iterator(
             state_->directory, filesystem_error)) {
        require_metadata(!filesystem_error,
                         "cannot enumerate transformer directory");
        const path candidate = entry.path();
        if (candidate.extension() != ".safetensors")
            continue;
        std::error_code entry_error;
        require_metadata(std::filesystem::is_regular_file(candidate,
                                                            entry_error) &&
                             !entry_error,
                         "safetensors shard is not a regular file: " +
                             candidate.string());
        shard_paths.push_back(candidate);
    }
    require_metadata(!filesystem_error && !shard_paths.empty(),
                     "no safetensors shards in transformer directory");
    std::sort(shard_paths.begin(), shard_paths.end());

    char error[512] = {};
    state_->store.reset(h3_weight_store_open(
        state_->directory.c_str(), error, sizeof(error)));
    require_metadata(state_->store != nullptr,
                     error[0] ? error : "cannot open H3 weight store");
    require_metadata(h3_weight_store_shards(state_->store.get()) ==
                         shard_paths.size(),
                     "weight-store shard inventory changed during open");

    for (uint32_t index = 0; index < shard_paths.size(); ++index) {
        const path &shard = shard_paths[index];
        struct stat status{};
        require_metadata(::stat(shard.c_str(), &status) == 0 &&
                             S_ISREG(status.st_mode) && status.st_size >= 8,
                         "invalid safetensors shard: " + shard.string());
        state_->shards.push_back({shard.string(), shard.filename().string(),
                                  static_cast<uint64_t>(status.st_size),
                                  snapshot_fingerprint(status)});
        const h3_st_header *header = h3_weight_store_header(
            state_->store.get(), index);
        require_metadata(header && header->path &&
                             path(header->path).lexically_normal() ==
                                 shard.lexically_normal() &&
                             header->file_size ==
                                 static_cast<uint64_t>(status.st_size),
                         "weight-store shard order/identity mismatch");
    }

    require_metadata(h3_weight_store_identity(
                         state_->store.get(), &state_->identity, error,
                         sizeof(error)) != 0,
                     error[0] ? error : "cannot create H3 snapshot identity");
    check_unchanged();
}

StreamingMetadata::~StreamingMetadata() = default;

size_t StreamingMetadata::shard_count() const noexcept {
    return state_ ? state_->shards.size() : 0;
}

uint64_t StreamingMetadata::snapshot_identity() const noexcept {
    return state_ ? state_->identity : 0;
}

void StreamingMetadata::check_unchanged() const {
    require_metadata(state_ && state_->store,
                     "weight store is not available");
    for (const auto &shard : state_->shards) {
        struct stat status{};
        require_metadata(::stat(shard.path.c_str(), &status) == 0 &&
                             S_ISREG(status.st_mode) &&
                             static_cast<uint64_t>(status.st_size) ==
                                 shard.bytes &&
                             snapshot_fingerprint(status) == shard.identity,
                         "checkpoint_changed: H3 metadata snapshot is stale");
    }
    std::error_code scan_error;
    std::vector<std::string> current_paths;
    std::filesystem::directory_iterator iterator(
        state_->directory, scan_error);
    const std::filesystem::directory_iterator end;
    for (; !scan_error && iterator != end; iterator.increment(scan_error)) {
        const path candidate = iterator->path();
        if (candidate.extension() == ".safetensors")
            current_paths.push_back(candidate.lexically_normal().string());
    }
    std::sort(current_paths.begin(), current_paths.end());
    require_metadata(!scan_error && current_paths.size() ==
                         state_->shards.size(),
                     "checkpoint_changed: H3 shard set is stale");
    for (size_t index = 0; index < current_paths.size(); ++index)
        require_metadata(current_paths[index] == state_->shards[index].path,
                         "checkpoint_changed: H3 shard set is stale");
    char error[512] = {};
    uint64_t current = 0;
    require_metadata(h3_weight_store_identity(
                         state_->store.get(), &current, error,
                         sizeof(error)) != 0 && current == state_->identity,
                     "checkpoint_changed: H3 shard inventory is stale");
}

streaming::Descriptor StreamingMetadata::describe(
    const StreamingWorkload &workload) const {
    check_unchanged();
    require_metadata(workload.width && workload.height && workload.frames &&
                         workload.fps && workload.text_rows &&
                         workload.steps >= 2 &&
                         workload.steps <= H3_MAX_STEPS,
                     "workload dimensions/steps are invalid");
    require_metadata(workload.active_blocks >= H3_DIT_BLOCKS / 2u &&
                         workload.active_blocks <= H3_DIT_BLOCKS,
                     "active block count is outside the uniform policy range");
    require_metadata(workload.width % H3_CANVAS_MULTIPLE == 0 &&
                         workload.height % H3_CANVAS_MULTIPLE == 0,
                     "workload canvas must be normalized to 32 pixels");
    int adapted_width = 0, adapted_height = 0;
    require_metadata(h3_adapt_canvas(
                         static_cast<int>(workload.width),
                         static_cast<int>(workload.height), &adapted_width,
                         &adapted_height) &&
                         adapted_width == static_cast<int>(workload.width) &&
                         adapted_height == static_cast<int>(workload.height),
                     "workload canvas is not normalized");
    const h3_temporal_shape temporal = h3_temporal(
        static_cast<int>(workload.frames));
    require_metadata(temporal.frame_count ==
                         static_cast<int>(workload.frames),
                     "workload frame count is not normalized");

    int latent_width = 0, latent_height = 0;
    h3_latent_canvas(static_cast<int>(workload.width),
                     static_cast<int>(workload.height), &latent_width,
                     &latent_height);
    h3_layout_spec layout_spec{
        static_cast<int>(workload.text_rows), temporal.video_t, latent_height,
        latent_width, temporal.audio_t, temporal.frame_count, nullptr, 0,
        nullptr, 0};
    h3_layout layout{};
    char error[512] = {};
    require_metadata(h3_layout_build(&layout_spec, &layout, error,
                                     sizeof(error)) != 0,
                     error[0] ? error : "cannot build H3 workload layout");
    const uint64_t sequence = static_cast<uint64_t>(layout.seq_len);
    h3_layout_free(&layout);
    require_metadata(sequence && sequence <= UINT32_MAX,
                     "H3 workload sequence is outside native range");

    uint8_t active_mask[H3_DIT_BLOCKS] = {};
    require_metadata(h3_stream_uniform_active_mask(
                         H3_DIT_BLOCKS, workload.active_blocks, active_mask,
                         H3_DIT_BLOCKS) != 0,
                     "uniform active-block policy could not be projected");

    streaming::Descriptor descriptor{
        "minimax-h3-turbo", "snapshot:" + hex_u64(state_->identity),
        "h3-c-metal-streaming-v1", {}};
    for (const auto &shard : state_->shards) {
        descriptor.artifacts.push_back(
            {"shard:" + shard.name, shard.identity, shard.bytes,
             streaming::SourceIdentityKind::snapshot});
    }
    descriptor.workload = {
        {"operation", "dit.denoise"},
        {"width", std::to_string(workload.width)},
        {"height", std::to_string(workload.height)},
        {"frames", std::to_string(workload.frames)},
        {"fps", std::to_string(workload.fps)},
        {"text_rows", std::to_string(workload.text_rows)},
        {"steps", std::to_string(workload.steps)},
        {"active_blocks", std::to_string(workload.active_blocks)},
        {"active_block_policy", "uniform-v1"},
        {"active_block_ids", active_ids(active_mask)},
        {"audio_output", bool_string(workload.audio)},
        {"token_reduction", bool_string(workload.token_reduction)},
        {"first_block_cache", bool_string(workload.first_block_cache)},
        {"resident_auxiliary", "norms-adaln-text-workspace-not-in-slots"},
        {"cross_block_fusion", "legacy-template-not-execution-authority"},
        {"quantized_cache", "unsupported"},
    };

    streaming::StageDescriptor stage;
    stage.id = "denoiser";
    stage.adapter_revision = "h3-c-metal-streaming-v1-metadata";
    stage.min_slots = 2;
    stage.max_slots = 2;
    stage.max_group_size = 1;
    stage.min_prefix = 0;
    stage.pass_count = workload.steps;
    for (uint32_t step = 0; step < workload.steps; ++step)
        stage.passes.push_back({step, "denoise", {sequence, H3_DIT_HIDDEN}});

    for (uint32_t block_id = 0; block_id < H3_DIT_BLOCKS; ++block_id) {
        if (!active_mask[block_id])
            continue;
        streaming::BlockSpec block;
        block.id = block_id;
        block.layout_class = "h3-bf16-dit-matrix-v1";
        block.streamable = true;
        block.safe_boundary_after = true;
        for (const MatrixSpec &spec : kMatrices) {
            std::string name = "blocks." + std::to_string(block_id) + "." +
                               spec.suffix;
            const h3_st_header *header = nullptr;
            const h3_st_tensor *tensor = nullptr;
            for (size_t shard_index = 0;
                 shard_index < h3_weight_store_shards(state_->store.get());
                 ++shard_index) {
                const h3_st_header *candidate = h3_weight_store_header(
                    state_->store.get(), shard_index);
                const h3_st_tensor *found = h3_st_find(candidate,
                                                       name.c_str());
                require_metadata(!found || !tensor,
                                 "duplicate H3 matrix across shards: " +
                                     name);
                if (found) {
                    tensor = found;
                    header = candidate;
                }
            }
            require_metadata(tensor && header && header->path,
                             "required H3 matrix is absent: " + name);
            require_metadata(tensor->dtype == H3_DTYPE_BF16 &&
                                 tensor->ndim == 2 &&
                                 tensor->shape[0] == spec.rows &&
                                 tensor->shape[1] == spec.columns,
                             "H3 matrix has wrong dtype or shape: " + name);
            const uint64_t expected = checked_matrix_bytes(spec);
            require_metadata(tensor->data_end >= tensor->data_begin &&
                                 tensor->data_end - tensor->data_begin ==
                                     expected,
                             "H3 matrix byte range mismatch: " + name);
            uint32_t artifact_index = UINT32_MAX;
            for (uint32_t index = 0; index < state_->shards.size(); ++index) {
                if (state_->shards[index].path == header->path) {
                    artifact_index = index;
                    break;
                }
            }
            require_metadata(artifact_index != UINT32_MAX,
                             "H3 matrix shard is outside the snapshot set: " +
                                 name);

            streaming::Materialization materialization;
            materialization.format = "BF16";
            materialization.storage_mode = "metal-shared";
            materialization.conversion = "copy-bf16-v1";
            materialization.shape = {spec.rows, spec.columns};
            materialization.reads.push_back(
                {artifact_index, tensor->file_offset, expected, name,
                 "BF16", {spec.rows, spec.columns}});
            block.fields.push_back(
                {spec.field,
                 "h3.block." + std::to_string(block_id) + "." + spec.field,
                 expected, 256, std::move(materialization)});
        }
        stage.blocks.push_back(std::move(block));
    }
    require_metadata(stage.blocks.size() == workload.active_blocks,
                     "active-block projection count mismatch");
    descriptor.stages.push_back(std::move(stage));
    return descriptor;
}

StreamingPlanView::StreamingPlanView(
    const std::string &transformer_directory, const StreamingConfig &config,
    const StreamingWorkload &workload)
    : metadata_(transformer_directory), descriptor_(metadata_.describe(workload)),
      layout_(streaming::compile_layout(config, descriptor_)) {
    require_metadata(layout_.materializations_complete,
                     "H3 descriptor metadata is incomplete");
    require_metadata(layout_.stages.size() == 1,
                     "H3 exact candidate requires one stage");
    const auto &stage = layout_.stages.front();
    require_metadata(stage.id == "denoiser" && !stage.resident &&
                         stage.group_size == 1 && stage.slot_count == 2 &&
                         stage.pools.size() == 1,
                     "H3 exact candidate requires one K=2/G=1 streamed pool");
    require_metadata(!workload.token_reduction &&
                         !workload.first_block_cache,
                     "H3 exact candidate does not yet model dynamic/fused "
                     "shortcut access");
    require_metadata(stage.prefix <
                         descriptor_.stages.front().blocks.size(),
                     "H3 exact candidate requires a streamed suffix");
    require_metadata(stage.groups.size() ==
                         descriptor_.stages.front().blocks.size() - stage.prefix,
                     "H3 compiled suffix group count mismatch");
    for (const auto &group : stage.groups)
        require_metadata(group.blocks.size() == 1 && group.pool == 0,
                         "H3 exact candidate requires one block per group");
}

} // namespace tc::h3
