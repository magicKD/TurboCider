#import <Foundation/Foundation.h>

#include "../../models/flux2/streaming_descriptor.hpp"
#include "../../runtime/memory_manifest.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace tc::flux2 {
namespace {

constexpr uint64_t kMaxHeaderBytes = 32ull << 20;
constexpr uint64_t kMaxExactJsonInteger = 1ull << 53;
constexpr uint32_t kHeadDimension = 128;

void require_metadata(bool ok, const std::string &reason) {
    if (!ok)
        throw std::invalid_argument("flux_streaming_metadata: " + reason);
}

uint64_t checked_add(uint64_t left, uint64_t right,
                     const char *reason) {
    require_metadata(right <= UINT64_MAX - left, reason);
    return left + right;
}

uint64_t checked_multiply(uint64_t left, uint64_t right,
                          const char *reason) {
    require_metadata(!left || right <= UINT64_MAX / left, reason);
    return left * right;
}

std::string fingerprint(const struct stat &status) {
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << status.st_dev << ':' << status.st_ino << ':'
              << status.st_size << ':' << status.st_mtimespec.tv_sec << ':'
              << status.st_mtimespec.tv_nsec << ':'
              << status.st_ctimespec.tv_sec << ':'
              << status.st_ctimespec.tv_nsec;
    return memory_sha256_hex(canonical.str());
}

struct stat checked_status(const std::string &path) {
    struct stat status{};
    require_metadata(::stat(path.c_str(), &status) == 0 &&
                         S_ISREG(status.st_mode),
                     "snapshot file is not a regular file: " + path);
    return status;
}

void pread_exact(int descriptor, void *destination, size_t bytes,
                 uint64_t offset) {
    auto *output = static_cast<unsigned char *>(destination);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t count = ::pread(
            descriptor, output + done, bytes - done,
            static_cast<off_t>(offset + done));
        if (count < 0 && errno == EINTR)
            continue;
        require_metadata(count > 0,
                         "safetensors header read failed or was truncated");
        done += static_cast<size_t>(count);
    }
}

NSDictionary *json_object(const std::string &path, const char *label) {
    NSData *data = [NSData dataWithContentsOfFile:@(path.c_str())];
    require_metadata(data != nil, std::string("cannot read ") + label);
    NSError *failure = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:data
                                                options:0
                                                  error:&failure];
    require_metadata([parsed isKindOfClass:NSDictionary.class],
                     failure ? failure.localizedDescription.UTF8String :
                               std::string("invalid ") + label);
    return (NSDictionary *)parsed;
}

NSDictionary *json_object(int descriptor, uint64_t bytes,
                          const char *label) {
    require_metadata(descriptor >= 0 && bytes > 0 &&
                         bytes <= kMaxHeaderBytes,
                     std::string("invalid ") + label + " size");
    std::vector<unsigned char> encoded(static_cast<size_t>(bytes));
    pread_exact(descriptor, encoded.data(), encoded.size(), 0);
    NSData *data = [NSData dataWithBytes:encoded.data()
                                  length:encoded.size()];
    NSError *failure = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:data
                                                options:0
                                                  error:&failure];
    require_metadata([parsed isKindOfClass:NSDictionary.class],
                     failure ? failure.localizedDescription.UTF8String :
                               std::string("invalid ") + label);
    return (NSDictionary *)parsed;
}

uint64_t integer(id value, const char *label) {
    require_metadata([value isKindOfClass:NSNumber.class] &&
                         CFGetTypeID((__bridge CFTypeRef)value) !=
                             CFBooleanGetTypeID(),
                     std::string("non-integer ") + label);
    const double number = [value doubleValue];
    require_metadata(std::isfinite(number) && number >= 0 &&
                         number == std::floor(number) &&
                         number <= static_cast<double>(kMaxExactJsonInteger),
                     std::string("invalid ") + label);
    return static_cast<uint64_t>(number);
}

uint32_t config_integer(NSDictionary *object, NSString *key,
                        const char *label) {
    const uint64_t value = integer(object[key], label);
    require_metadata(value <= UINT32_MAX, std::string(label) + " overflows");
    return static_cast<uint32_t>(value);
}

std::vector<uint64_t> shape_vector(NSArray *shape) {
    require_metadata([shape isKindOfClass:NSArray.class] &&
                         shape.count > 0 && shape.count <= 8,
                     "invalid tensor shape");
    std::vector<uint64_t> result;
    result.reserve(shape.count);
    for (id raw_dimension in shape) {
        const uint64_t dimension = integer(raw_dimension, "tensor dimension");
        require_metadata(dimension > 0, "zero tensor dimension");
        result.push_back(dimension);
    }
    return result;
}

std::pair<uint32_t, std::string> block_identity(
    const std::string &name, const std::string &prefix, uint32_t count) {
    require_metadata(name.starts_with(prefix), "invalid block tensor prefix");
    const size_t begin = prefix.size();
    const size_t end = name.find('.', begin);
    require_metadata(end != std::string::npos && end + 1 < name.size(),
                     "invalid block tensor name");
    const std::string number = name.substr(begin, end - begin);
    require_metadata(!number.empty() && number.size() <= 3 &&
                         number.find_first_not_of("0123456789") ==
                             std::string::npos,
                     "invalid block index");
    const unsigned long parsed = std::stoul(number);
    require_metadata(parsed < count && number == std::to_string(parsed),
                     "block index is outside the configured topology");
    return {static_cast<uint32_t>(parsed), name.substr(end + 1)};
}

using ShapeMap = std::map<std::string, std::vector<uint64_t>>;

ShapeMap fixed_shapes(uint64_t hidden, uint64_t context_width) {
    return {
        {"context_embedder.weight", {hidden, context_width}},
        {"double_stream_modulation_img.linear.weight", {hidden * 6, hidden}},
        {"double_stream_modulation_txt.linear.weight", {hidden * 6, hidden}},
        {"norm_out.linear.weight", {hidden * 2, hidden}},
        {"proj_out.weight", {128, hidden}},
        {"single_stream_modulation.linear.weight", {hidden * 3, hidden}},
        {"time_guidance_embed.timestep_embedder.linear_1.weight", {hidden, 256}},
        {"time_guidance_embed.timestep_embedder.linear_2.weight", {hidden, hidden}},
        {"x_embedder.weight", {hidden, 128}},
    };
}

ShapeMap dual_shapes(uint64_t hidden) {
    ShapeMap result;
    for (const auto *name : {
             "attn.add_k_proj.weight", "attn.add_q_proj.weight",
             "attn.add_v_proj.weight", "attn.to_add_out.weight",
             "attn.to_k.weight", "attn.to_out.0.weight",
             "attn.to_q.weight", "attn.to_v.weight"})
        result[name] = {hidden, hidden};
    for (const auto *name : {
             "attn.norm_added_k.weight", "attn.norm_added_q.weight",
             "attn.norm_k.weight", "attn.norm_q.weight"})
        result[name] = {kHeadDimension};
    result["ff.linear_in.weight"] = {hidden * 6, hidden};
    result["ff.linear_out.weight"] = {hidden, hidden * 3};
    result["ff_context.linear_in.weight"] = {hidden * 6, hidden};
    result["ff_context.linear_out.weight"] = {hidden, hidden * 3};
    return result;
}

ShapeMap single_shapes(uint64_t hidden) {
    return {
        {"attn.norm_k.weight", {kHeadDimension}},
        {"attn.norm_q.weight", {kHeadDimension}},
        {"attn.to_out.weight", {hidden, hidden * 4}},
        {"attn.to_qkv_mlp_proj.weight", {hidden * 9, hidden}},
    };
}

struct TensorRecord {
    std::string name;
    std::string suffix;
    std::vector<uint64_t> shape;
    uint32_t artifact = 0;
    uint64_t file_offset = 0;
    uint64_t bytes = 0;
};

struct ArtifactRecord {
    std::string name;
    std::string path;
    std::string identity;
    int descriptor = -1;
    uint64_t file_bytes = 0;
};

void validate_records(const std::vector<TensorRecord> &records,
                      const ShapeMap &expected, const char *label) {
    require_metadata(records.size() == expected.size(),
                     std::string(label) + " tensor count differs");
    for (const auto &record : records) {
        const auto found = expected.find(record.suffix);
        require_metadata(found != expected.end(),
                         std::string(label) + " has an unexpected tensor");
        require_metadata(record.shape == found->second,
                         std::string(label) + " tensor shape differs");
    }
}

uint64_t records_bytes(const std::vector<TensorRecord> &records) {
    uint64_t result = 0;
    for (const auto &record : records)
        result = checked_add(result, record.bytes,
                             "tensor byte count overflow");
    return result;
}

} // namespace

struct StreamingMetadata::State {
    std::shared_ptr<const streaming::SourceLease> lease;
    std::string directory;
    std::string model_id;
    std::string config_path;
    std::string index_path;
    std::string config_identity;
    std::string index_identity;
    std::string identity;
    uint32_t heads = 0;
    uint32_t hidden = 0;
    uint32_t context_width = 0;
    bool indexed = true;
    uint32_t dual_count = 0;
    uint32_t single_count = 0;
    uint64_t fixed_total = 0;
    uint64_t dual_total = 0;
    uint64_t single_total = 0;
    std::vector<ArtifactRecord> artifacts;
    std::vector<TensorRecord> fixed;
    std::vector<std::vector<TensorRecord>> dual;
    std::vector<std::vector<TensorRecord>> single;

    ~State() {
        for (auto &artifact : artifacts)
            if (artifact.descriptor >= 0)
                ::close(artifact.descriptor);
    }
};

StreamingMetadata::StreamingMetadata(
    const std::string &transformer_directory, const std::string &model_id)
    : StreamingMetadata(nullptr, transformer_directory, model_id) {}

StreamingMetadata::StreamingMetadata(
    std::shared_ptr<const streaming::SourceLease> lease,
    const std::string &model_id)
    : StreamingMetadata(std::move(lease), std::string{}, model_id) {}

StreamingMetadata::StreamingMetadata(
    std::shared_ptr<const streaming::SourceLease> lease,
    const std::string &transformer_directory, const std::string &model_id)
    : state_(std::make_unique<State>()) {
    require_metadata(model_id == "flux2-klein-9b" || model_id == "flux2-klein-4b",
                     "only FLUX.2 Klein 4B/9B support exact streaming");
    state_->indexed = model_id == "flux2-klein-9b";
    state_->lease = std::move(lease);
    std::filesystem::path absolute;
    if (state_->lease) {
        state_->directory = state_->lease->file("config.json")
                                .path.parent_path().string();
        absolute = state_->directory;
    } else {
        std::error_code path_error;
        absolute = std::filesystem::absolute(
            transformer_directory, path_error);
        require_metadata(!path_error && std::filesystem::is_directory(absolute),
                         "transformer directory is missing");
        state_->directory = absolute.lexically_normal().string();
    }
    state_->model_id = model_id;
    state_->config_path =
        (absolute / "config.json").lexically_normal().string();
    state_->index_path = (absolute /
        "diffusion_pytorch_model.safetensors.index.json").lexically_normal().string();

    streaming::OwnedSourceFd config_fd;
    streaming::OwnedSourceFd index_fd;
    struct stat config_status{};
    struct stat index_status{};
    if (state_->lease) {
        config_fd = state_->lease->duplicate_fd("config.json");
        require_metadata(::fstat(config_fd.get(), &config_status) == 0 &&
                             S_ISREG(config_status.st_mode), "leased FLUX config is invalid");
        if (state_->indexed) {
            index_fd = state_->lease->duplicate_fd("diffusion_pytorch_model.safetensors.index.json");
            require_metadata(::fstat(index_fd.get(), &index_status) == 0 &&
                                 S_ISREG(index_status.st_mode), "leased FLUX index is invalid");
        }
    } else {
        config_status = checked_status(state_->config_path);
        if (state_->indexed) index_status = checked_status(state_->index_path);
    }
    state_->config_identity = fingerprint(config_status);
    if (state_->indexed) state_->index_identity = fingerprint(index_status);

    @autoreleasepool {
        NSDictionary *config = state_->lease
            ? json_object(config_fd.get(),
                          static_cast<uint64_t>(config_status.st_size),
                          "FLUX transformer config")
            : json_object(state_->config_path, "FLUX transformer config");
        state_->heads = config_integer(config, @"num_attention_heads",
                                       "attention head count");
        const uint32_t head_dimension = config_integer(
            config, @"attention_head_dim", "attention head dimension");
        state_->dual_count = config_integer(config, @"num_layers",
                                            "dual layer count");
        state_->single_count = config_integer(config, @"num_single_layers",
                                              "single layer count");
        state_->hidden = state_->heads * head_dimension;
        state_->context_width = config_integer(config, @"joint_attention_dim", "joint attention width");
        const bool klein4 = !state_->indexed;
        require_metadata(head_dimension == kHeadDimension &&
                             config_integer(config, @"in_channels", "input channels") == 128 &&
                             state_->context_width == (klein4 ? 7680u : 12288u) &&
                             state_->heads == (klein4 ? 24u : 32u) &&
                             state_->hidden == (klein4 ? 3072u : 4096u) &&
                             state_->dual_count == (klein4 ? 5u : 8u) &&
                             state_->single_count == (klein4 ? 20u : 24u),
                         "configuration does not match FLUX.2 Klein model");

        state_->dual.resize(state_->dual_count);
        state_->single.resize(state_->single_count);

        uint64_t indexed_total = 0;
        std::map<std::string, std::string> mapping;
        std::set<std::string> shard_names;
        if (state_->indexed) {
            NSDictionary *index = state_->lease
                ? json_object(index_fd.get(),
                              static_cast<uint64_t>(index_status.st_size),
                              "FLUX safetensors index")
                : json_object(state_->index_path, "FLUX safetensors index");
            NSDictionary *weight_map = index[@"weight_map"];
            NSDictionary *metadata = index[@"metadata"];
            require_metadata([weight_map isKindOfClass:NSDictionary.class] &&
                                 [metadata isKindOfClass:NSDictionary.class],
                             "invalid FLUX safetensors index");
            indexed_total = integer(metadata[@"total_size"],
                                                   "index total_size");
            for (id raw_key in weight_map) {
                require_metadata([raw_key isKindOfClass:NSString.class] &&
                                     [weight_map[raw_key]
                                         isKindOfClass:NSString.class],
                                 "invalid index weight mapping");
                const std::string key = [(NSString *)raw_key UTF8String];
                const std::string shard =
                    [(NSString *)weight_map[raw_key] UTF8String];
                require_metadata(!key.empty() && !shard.empty() &&
                                     std::filesystem::path(shard).filename() ==
                                         std::filesystem::path(shard) &&
                                     std::filesystem::path(shard).extension() ==
                                         ".safetensors",
                                 "invalid index shard path");
                require_metadata(mapping.emplace(key, shard).second,
                                 "duplicate tensor in index");
                shard_names.insert(shard);
            }
            require_metadata(!mapping.empty() && shard_names.size() == 2,
                             "Klein 9B requires two indexed shards");
        } else {
            shard_names.insert("diffusion_pytorch_model.safetensors");
        }

        std::map<std::string, uint32_t> artifact_by_name;
        for (const auto &name : shard_names) {
            ArtifactRecord artifact;
            artifact.name = name;
            artifact.path = state_->lease
                ? state_->lease->file(name).path.string()
                : (absolute / name).lexically_normal().string();
            artifact.descriptor = state_->lease
                ? state_->lease->duplicate_fd(name).release()
                : ::open(artifact.path.c_str(), O_RDONLY | O_CLOEXEC);
            require_metadata(artifact.descriptor >= 0,
                             "cannot open indexed safetensors shard");
            struct stat opened{};
            require_metadata(::fstat(artifact.descriptor, &opened) == 0 &&
                                 S_ISREG(opened.st_mode) && opened.st_size >= 8,
                             "indexed shard is invalid");
            artifact.file_bytes = static_cast<uint64_t>(opened.st_size);
            artifact.identity = fingerprint(opened);
            const uint32_t index_value =
                static_cast<uint32_t>(state_->artifacts.size());
            artifact_by_name.emplace(name, index_value);
            state_->artifacts.push_back(std::move(artifact));
        }

        std::set<std::string> discovered;
        uint64_t discovered_total = 0;
        const ShapeMap fixed_expected = fixed_shapes(state_->hidden, state_->context_width);
        const ShapeMap dual_expected = dual_shapes(state_->hidden);
        const ShapeMap single_expected = single_shapes(state_->hidden);

        for (uint32_t artifact_index = 0;
             artifact_index < state_->artifacts.size(); ++artifact_index) {
            auto &artifact = state_->artifacts[artifact_index];
            std::array<unsigned char, 8> prefix{};
            pread_exact(artifact.descriptor, prefix.data(), prefix.size(), 0);
            uint64_t header_bytes = 0;
            for (uint32_t index_byte = 0; index_byte < prefix.size(); ++index_byte)
                header_bytes |= static_cast<uint64_t>(prefix[index_byte]) <<
                    (8 * index_byte);
            require_metadata(header_bytes && header_bytes <= kMaxHeaderBytes &&
                                 header_bytes <= artifact.file_bytes - 8,
                             "invalid safetensors header length");
            std::vector<unsigned char> encoded(
                static_cast<size_t>(header_bytes));
            pread_exact(artifact.descriptor, encoded.data(), encoded.size(), 8);
            NSData *header_data = [NSData dataWithBytes:encoded.data()
                                                  length:encoded.size()];
            NSError *header_failure = nil;
            id parsed = [NSJSONSerialization JSONObjectWithData:header_data
                                                        options:0
                                                          error:&header_failure];
            require_metadata([parsed isKindOfClass:NSDictionary.class],
                             header_failure ?
                                 header_failure.localizedDescription.UTF8String :
                                 "invalid safetensors header JSON");
            NSDictionary *root = (NSDictionary *)parsed;
            NSArray *keys = [[root allKeys]
                sortedArrayUsingSelector:@selector(compare:)];
            std::vector<std::pair<uint64_t, uint64_t>> intervals;
            const uint64_t payload_bytes =
                artifact.file_bytes - 8 - header_bytes;
            for (id raw_key in keys) {
                require_metadata([raw_key isKindOfClass:NSString.class],
                                 "safetensors tensor name is not a string");
                NSString *key_string = (NSString *)raw_key;
                if ([key_string isEqualToString:@"__metadata__"])
                    continue;
                const std::string name = key_string.UTF8String;
                require_metadata(discovered.insert(name).second,
                                 "duplicate tensor across shards");
                const auto mapped = mapping.find(name);
                require_metadata(!state_->indexed || (mapped != mapping.end() &&
                                     mapped->second == artifact.name),
                                 "index and shard header disagree");
                NSDictionary *record = root[key_string];
                require_metadata([record isKindOfClass:NSDictionary.class] &&
                                     [record[@"dtype"]
                                         isKindOfClass:NSString.class] &&
                                     [record[@"dtype"]
                                         isEqualToString:@"BF16"],
                                 "Klein streaming supports BF16 weights only");
                const std::vector<uint64_t> shape =
                    shape_vector(record[@"shape"]);
                NSArray *offsets = record[@"data_offsets"];
                require_metadata([offsets isKindOfClass:NSArray.class] &&
                                     offsets.count == 2,
                                 "invalid tensor offsets");
                uint64_t bytes = sizeof(uint16_t);
                for (uint64_t dimension : shape)
                    bytes = checked_multiply(
                        bytes, dimension, "tensor shape size overflow");
                const uint64_t begin = integer(offsets[0], "tensor offset");
                const uint64_t end = integer(offsets[1], "tensor offset");
                require_metadata(end >= begin && end - begin == bytes &&
                                     end <= payload_bytes,
                                 "tensor range differs from BF16 shape");
                intervals.emplace_back(begin, end);

                TensorRecord tensor;
                tensor.name = name;
                tensor.shape = shape;
                tensor.artifact = artifact_index;
                tensor.file_offset = checked_add(
                    8 + header_bytes, begin,
                    "tensor file offset overflow");
                tensor.bytes = bytes;
                discovered_total = checked_add(
                    discovered_total, bytes, "checkpoint byte count overflow");

                if (name.starts_with("transformer_blocks.")) {
                    auto [block, suffix] = block_identity(
                        name, "transformer_blocks.", state_->dual_count);
                    tensor.suffix = std::move(suffix);
                    state_->dual[block].push_back(std::move(tensor));
                } else if (name.starts_with("single_transformer_blocks.")) {
                    auto [block, suffix] = block_identity(
                        name, "single_transformer_blocks.",
                        state_->single_count);
                    tensor.suffix = std::move(suffix);
                    state_->single[block].push_back(std::move(tensor));
                } else {
                    tensor.suffix = name;
                    require_metadata(fixed_expected.count(name) != 0,
                                     "checkpoint has an unexpected fixed tensor");
                    state_->fixed.push_back(std::move(tensor));
                }
            }
            std::sort(intervals.begin(), intervals.end());
            uint64_t cursor = 0;
            for (const auto &[begin, end] : intervals) {
                require_metadata(begin == cursor,
                                 "overlapping or noncontiguous shard ranges");
                cursor = end;
            }
            require_metadata(cursor == payload_bytes,
                             "unexpected safetensors shard payload length");
        }

        require_metadata(!state_->indexed || (discovered.size() == mapping.size() &&
                             discovered_total == indexed_total),
                         "index tensor set or total_size differs from shards");
        std::sort(state_->fixed.begin(), state_->fixed.end(),
                  [](const auto &left, const auto &right) {
                      return left.suffix < right.suffix;
                  });
        validate_records(state_->fixed, fixed_expected, "fixed weights");
        state_->fixed_total = records_bytes(state_->fixed);

        for (uint32_t block = 0; block < state_->dual.size(); ++block) {
            auto &records = state_->dual[block];
            std::sort(records.begin(), records.end(),
                      [](const auto &left, const auto &right) {
                          return left.suffix < right.suffix;
                      });
            validate_records(records, dual_expected, "dual block");
            const uint64_t bytes = records_bytes(records);
            if (!block)
                state_->dual_total = bytes;
            require_metadata(bytes == state_->dual_total,
                             "dual block byte counts differ");
        }
        for (uint32_t block = 0; block < state_->single.size(); ++block) {
            auto &records = state_->single[block];
            std::sort(records.begin(), records.end(),
                      [](const auto &left, const auto &right) {
                          return left.suffix < right.suffix;
                      });
            validate_records(records, single_expected, "single block");
            const uint64_t bytes = records_bytes(records);
            if (!block)
                state_->single_total = bytes;
            require_metadata(bytes == state_->single_total,
                             "single block byte counts differ");
        }
    }

    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << state_->model_id << '|' << state_->config_identity << '|'
              << state_->index_identity;
    for (const auto &artifact : state_->artifacts)
        canonical << '|' << artifact.name << ':' << artifact.identity;
    state_->identity = memory_sha256_hex(canonical.str());
    check_unchanged();
}

StreamingMetadata::~StreamingMetadata() = default;

uint32_t StreamingMetadata::hidden_size() const noexcept { return state_->hidden; }
uint32_t StreamingMetadata::head_count() const noexcept { return state_->heads; }

uint32_t StreamingMetadata::dual_block_count() const noexcept {
    return state_ ? state_->dual_count : 0;
}

uint32_t StreamingMetadata::single_block_count() const noexcept {
    return state_ ? state_->single_count : 0;
}

uint64_t StreamingMetadata::dual_block_bytes() const noexcept {
    return state_ ? state_->dual_total : 0;
}

uint64_t StreamingMetadata::single_block_bytes() const noexcept {
    return state_ ? state_->single_total : 0;
}

uint64_t StreamingMetadata::fixed_bytes() const noexcept {
    return state_ ? state_->fixed_total : 0;
}

const std::string &StreamingMetadata::snapshot_identity() const noexcept {
    return state_->identity;
}

const streaming::SourceLease *
StreamingMetadata::source_lease() const noexcept {
    return state_ ? state_->lease.get() : nullptr;
}

std::shared_ptr<const streaming::SourceLease>
StreamingMetadata::lease_ptr() const noexcept {
    return state_ ? state_->lease : nullptr;
}

void StreamingMetadata::check_unchanged() const {
    require_metadata(state_ != nullptr, "metadata state is unavailable");
    if (state_->lease) {
        state_->lease->revalidate_open_files();
        state_->lease->revalidate_paths();
        return;
    }
    require_metadata(fingerprint(checked_status(state_->config_path)) ==
                         state_->config_identity &&
                         (!state_->indexed || fingerprint(checked_status(state_->index_path)) ==
                         state_->index_identity),
                     "checkpoint_changed: FLUX config or index is stale");
    for (const auto &artifact : state_->artifacts) {
        struct stat opened{};
        require_metadata(artifact.descriptor >= 0 &&
                             ::fstat(artifact.descriptor, &opened) == 0 &&
                             S_ISREG(opened.st_mode) &&
                             static_cast<uint64_t>(opened.st_size) ==
                                 artifact.file_bytes &&
                             fingerprint(opened) == artifact.identity &&
                             fingerprint(checked_status(artifact.path)) ==
                                 artifact.identity,
                         "checkpoint_changed: FLUX shard snapshot is stale");
    }
}

streaming::Descriptor StreamingMetadata::describe(
    const StreamingWorkload &workload) const {
    check_unchanged();
    require_metadata(workload.width >= 16 && workload.height >= 16 &&
                         workload.width % 16 == 0 &&
                         workload.height % 16 == 0 &&
                         workload.caption_tokens > 0 && workload.steps > 0 &&
                         workload.steps <= streaming::max_passes,
                     "workload must be normalized before describe");
    const uint64_t image_tokens =
        static_cast<uint64_t>(workload.width / 16) *
        static_cast<uint64_t>(workload.height / 16);
    const uint64_t unified_tokens = checked_add(
        checked_add(image_tokens, workload.caption_tokens,
                    "FLUX token count overflow"),
        workload.reference_tokens, "FLUX token count overflow");
    require_metadata(unified_tokens <= 20000,
                     "workload exceeds the native token workspace budget");

    streaming::Descriptor descriptor{
        state_->model_id, "snapshot:" + state_->identity,
        "flux2-mlx-sharded-bf16-streaming-v1", {}};
    for (const auto &artifact : state_->artifacts)
        descriptor.artifacts.push_back(
            {artifact.name, artifact.identity, artifact.file_bytes,
             streaming::SourceIdentityKind::snapshot});
    descriptor.workload = {
        {"operation", "image.denoise"},
        {"format", "diffusers-bf16-sharded"},
        {"width", std::to_string(workload.width)},
        {"height", std::to_string(workload.height)},
        {"caption_tokens", std::to_string(workload.caption_tokens)},
        {"image_tokens", std::to_string(image_tokens)},
        {"reference_tokens", std::to_string(workload.reference_tokens)},
        {"unified_tokens", std::to_string(unified_tokens)},
        {"hidden", std::to_string(state_->hidden)},
        {"dual_blocks", std::to_string(state_->dual_count)},
        {"single_blocks", std::to_string(state_->single_count)},
        {"steps", std::to_string(workload.steps)},
        {"fixed_tensor_count", std::to_string(state_->fixed.size())},
        {"fixed_bytes", std::to_string(state_->fixed_total)},
        {"resident_auxiliary", "embedders-modulation-output-not-in-slots"},
        {"reader_revision", "flux2-sharded-pread-bf16-v1"},
        {"single_block_boundary", "mlx-eval-required-v1"},
    };

    streaming::StageDescriptor stage;
    stage.id = "denoiser";
    stage.adapter_revision = state_->model_id + "-two-class-v1-metadata";
    stage.min_slots = 2;
    stage.max_slots = 2;
    stage.max_group_size = 1;
    stage.min_prefix = 0;
    stage.pass_count = workload.steps;
    stage.pass_transition = streaming::PassTransition::reload;
    // Flux crosses dual -> single on every denoise pass and returns to dual on
    // the next pass. Keep both class backings request-resident so these
    // barriers do not allocate or destroy MLX/Metal storage in steady state.
    stage.multi_pool_policy = streaming::MultiPoolPolicy::retain_all;
    for (uint32_t step = 0; step < workload.steps; ++step)
        stage.passes.push_back(
            {step, "denoise", {unified_tokens, state_->hidden}});

    auto field = [](const TensorRecord &record,
                    const std::string &storage_id) {
        streaming::Materialization materialization;
        materialization.format = "BF16";
        materialization.storage_mode = "mlx-metal-shared";
        materialization.conversion = "copy-bf16-v1";
        materialization.shape = record.shape;
        materialization.reads.push_back(
            {record.artifact, record.file_offset, record.bytes,
             record.name, "BF16", record.shape});
        return streaming::FieldSpec{
            record.suffix, storage_id, record.bytes, 256,
            std::move(materialization)};
    };
    for (const auto &record : state_->fixed)
        stage.resident_fields.push_back(field(
            record, "flux2.fixed." + record.suffix));

    auto append = [&](uint32_t id, const std::string &layout_class,
                      const std::vector<TensorRecord> &records) {
        streaming::BlockSpec block;
        block.id = id;
        block.layout_class = layout_class;
        block.streamable = true;
        block.safe_boundary_after = true;
        for (const auto &record : records)
            block.fields.push_back(field(
                record, "flux2.block." + std::to_string(id) + "." +
                            record.suffix));
        stage.blocks.push_back(std::move(block));
    };
    for (uint32_t block = 0; block < state_->dual.size(); ++block)
        append(block, state_->model_id + "-dual-bf16-v1",
               state_->dual[block]);
    for (uint32_t block = 0; block < state_->single.size(); ++block)
        append(state_->dual_count + block,
               state_->model_id + "-single-bf16-v1",
               state_->single[block]);
    descriptor.stages.push_back(std::move(stage));
    return descriptor;
}

StreamingPlanView::StreamingPlanView(
    const std::string &transformer_directory, const std::string &model_id,
    const StreamingConfig &config, const StreamingWorkload &workload)
    : metadata_(transformer_directory, model_id),
      descriptor_(metadata_.describe(workload)),
      layout_(streaming::compile_layout(config, descriptor_)) {
    require_metadata(layout_.materializations_complete,
                     "FLUX descriptor metadata is incomplete");
    require_metadata(layout_.stages.size() == 1,
                     "FLUX shadow requires one stage");
    const auto &stage = layout_.stages.front();
    require_metadata(stage.id == "denoiser" && !stage.resident &&
                         stage.prefix == 0 && stage.group_size == 1 &&
                         stage.slot_count == 2 && stage.distance <= 1 &&
                         stage.workers >= 1 && stage.workers <= 2 &&
                         stage.multi_pool_policy ==
                             streaming::MultiPoolPolicy::retain_all &&
                         stage.pass_transition ==
                             streaming::PassTransition::reload &&
                         stage.pools.size() == 2,
                     "FLUX exact streaming requires P0/K2/G1/D0..1/Q1..2 reload");
    const uint32_t dual = metadata_.dual_block_count();
    const uint32_t total = dual + metadata_.single_block_count();
    require_metadata(stage.groups.size() == total,
                     "FLUX compiled suffix differs from the descriptor");
    for (uint32_t index = 0; index < stage.groups.size(); ++index) {
        const auto &group = stage.groups[index];
        const uint32_t expected_pool = index < dual ? 0 : 1;
        const uint64_t expected_bytes = index < dual ?
            metadata_.dual_block_bytes() : metadata_.single_block_bytes();
        require_metadata(group.blocks.size() == 1 &&
                             group.blocks.front() == index &&
                             group.pool == expected_pool &&
                             group.bytes == expected_bytes,
                         "FLUX group/class projection differs from metadata");
    }
}

StreamingPlanView::StreamingPlanView(
    std::shared_ptr<const streaming::SourceLease> lease,
    const std::string &model_id, const StreamingConfig &config,
    const StreamingWorkload &workload)
    : metadata_(std::move(lease), model_id),
      descriptor_(metadata_.describe(workload)),
      layout_(streaming::compile_layout(config, descriptor_)) {
    require_metadata(layout_.materializations_complete,
                     "FLUX descriptor metadata is incomplete");
    require_metadata(layout_.stages.size() == 1,
                     "FLUX shadow requires one stage");
    const auto &stage = layout_.stages.front();
    require_metadata(stage.id == "denoiser" && !stage.resident &&
                         stage.prefix == 0 && stage.group_size == 1 &&
                         stage.slot_count == 2 && stage.distance <= 1 &&
                         stage.workers >= 1 && stage.workers <= 2 &&
                         stage.multi_pool_policy ==
                             streaming::MultiPoolPolicy::retain_all &&
                         stage.pass_transition ==
                             streaming::PassTransition::reload &&
                         stage.pools.size() == 2,
                     "FLUX exact streaming requires P0/K2/G1/D0..1/Q1..2 reload");
    const uint32_t dual = metadata_.dual_block_count();
    const uint32_t total = dual + metadata_.single_block_count();
    require_metadata(stage.groups.size() == total,
                     "FLUX compiled suffix differs from the descriptor");
    for (uint32_t index = 0; index < stage.groups.size(); ++index) {
        const auto &group = stage.groups[index];
        const uint32_t expected_pool = index < dual ? 0 : 1;
        const uint64_t expected_bytes = index < dual ?
            metadata_.dual_block_bytes() : metadata_.single_block_bytes();
        require_metadata(group.blocks.size() == 1 &&
                             group.blocks.front() == index &&
                             group.pool == expected_pool &&
                             group.bytes == expected_bytes,
                         "FLUX group/class projection differs from metadata");
    }
}

} // namespace tc::flux2
