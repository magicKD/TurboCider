#import <Foundation/Foundation.h>

#include "../../models/z_image/streaming_descriptor.hpp"
#include "../../models/z_image/suffix_materialization.hpp"
#include "../../runtime/streaming/canonical_encoding.hpp"
#include "../../runtime/memory_manifest.hpp"
#include <fcntl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace tc::z_image {
namespace {

constexpr uint64_t kMaxHeaderBytes = 16ull << 20;
constexpr uint64_t kMaxExactJsonInteger = 1ull << 53;
constexpr uint64_t kHidden = 3840;

void require_metadata(bool ok, const std::string &reason) {
    if (!ok)
        throw std::invalid_argument("z_image_streaming_metadata: " + reason);
}

uint64_t checked_add(uint64_t left, uint64_t right,
                     const char *reason) {
    require_metadata(right <= UINT64_MAX - left, reason);
    return left + right;
}

std::shared_ptr<const streaming::SourceLease> capture_checkpoint_lease(
        const std::string &checkpoint) {
    require_metadata(!checkpoint.empty() &&
                         checkpoint.find('\0') == std::string::npos,
                     "invalid checkpoint path");
    std::error_code path_error;
    const auto absolute = std::filesystem::absolute(checkpoint, path_error);
    require_metadata(!path_error, "cannot resolve checkpoint path");
    const auto path = absolute.lexically_normal();
    require_metadata(path.extension() == ".safetensors",
                     "Z-Image shadow requires a safetensors checkpoint");
    streaming::SourceFileIdentity file;
    file.logical_id = "transformer";
    file.path = path;
    return streaming::SourceLease::capture(
        std::vector<streaming::SourceFileIdentity>{std::move(file)});
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
                         "checkpoint header read failed or was truncated");
        done += static_cast<size_t>(count);
    }
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

bool fixed_name(const std::string &name) {
    return name.starts_with("noise_refiner.") ||
        name.starts_with("context_refiner.") ||
        name.starts_with("x_embedder.") ||
        name.starts_with("cap_embedder.") ||
        name.starts_with("t_embedder.") ||
        name.starts_with("final_layer.") || name == "x_pad_token" ||
        name == "cap_pad_token";
}

using TensorRecord = StreamingTensor;

std::pair<uint32_t, std::string> layer_identity(const std::string &name) {
    require_metadata(name.starts_with("layers."),
                     "invalid Z-Image layer tensor name");
    const size_t end = name.find('.', 7);
    require_metadata(end != std::string::npos && end + 1 < name.size(),
                     "invalid Z-Image layer tensor name");
    const std::string number = name.substr(7, end - 7);
    require_metadata(!number.empty() && number.size() <= 2 &&
                         number.find_first_not_of("0123456789") ==
                             std::string::npos,
                     "invalid Z-Image layer number");
    const unsigned long parsed = std::stoul(number);
    require_metadata(parsed < 30 && number == std::to_string(parsed),
                     "invalid Z-Image layer number");
    return {static_cast<uint32_t>(parsed), name.substr(end + 1)};
}

uint64_t padded_rows(uint64_t value) {
    require_metadata(value && value <= UINT64_MAX - 31,
                     "workload token count overflow");
    return (value + 31) / 32 * 32;
}

} // namespace

struct StreamingMetadata::State {
    StreamingWeightOptions options;
    bool convrot = false;
    uint64_t packed_bytes = 0;
    std::vector<StreamingSuffixPack> packs;
    std::string path;
    std::string identity;
    std::string logical_id;
    std::shared_ptr<const streaming::SourceLease> lease;
    streaming::OwnedSourceFd descriptor;
    uint64_t file_bytes = 0;
    uint64_t fixed_total = 0;
    uint64_t block_total = 0;
    std::vector<TensorRecord> fixed;
    std::array<std::vector<TensorRecord>, 30> blocks;

};

StreamingMetadata::StreamingMetadata(const std::string &checkpoint,
                                     StreamingWeightOptions options)
    : StreamingMetadata(capture_checkpoint_lease(checkpoint),
                        "transformer", options) {}

StreamingMetadata::StreamingMetadata(
        std::shared_ptr<const streaming::SourceLease> lease,
        std::string logical_id, StreamingWeightOptions options)
    : state_(std::make_unique<State>()) {
    require_metadata(lease != nullptr, "source lease is unavailable");
    require_metadata(!logical_id.empty(), "source logical id is empty");
    const auto &file = lease->file(logical_id);
    require_metadata(file.path.extension() == ".safetensors",
                     "Z-Image shadow requires a safetensors checkpoint");
    state_->options = options;
    state_->path = file.path.string();
    state_->logical_id = std::move(logical_id);
    state_->lease = std::move(lease);
    state_->descriptor = state_->lease->duplicate_fd(state_->logical_id);
    require_metadata(bool(state_->descriptor), "checkpoint fd is unavailable");
    struct stat opened{};
    require_metadata(::fstat(state_->descriptor.get(), &opened) == 0 &&
                         S_ISREG(opened.st_mode) && opened.st_size >= 8,
                     "opened checkpoint is invalid");
    state_->file_bytes = static_cast<uint64_t>(opened.st_size);
    state_->identity = std::string(state_->lease->digest());
    parse_checkpoint();
}

void StreamingMetadata::parse_checkpoint() {
    require_metadata(state_ && state_->descriptor,
                     "checkpoint descriptor is unavailable");

    std::array<unsigned char, 8> prefix{};
    pread_exact(state_->descriptor.get(), prefix.data(), prefix.size(), 0);
    uint64_t header_bytes = 0;
    for (uint32_t index = 0; index < prefix.size(); ++index)
        header_bytes |= static_cast<uint64_t>(prefix[index]) << (8 * index);
    require_metadata(header_bytes && header_bytes <= kMaxHeaderBytes &&
                         header_bytes <= state_->file_bytes - 8,
                     "invalid safetensors header length");
    std::vector<unsigned char> encoded(static_cast<size_t>(header_bytes));
    pread_exact(state_->descriptor.get(), encoded.data(), encoded.size(), 8);

    @autoreleasepool {
        NSData *data = [NSData dataWithBytes:encoded.data()
                                      length:encoded.size()];
        NSError *failure = nil;
        id parsed = [NSJSONSerialization JSONObjectWithData:data
                                                    options:0
                                                      error:&failure];
        require_metadata([parsed isKindOfClass:NSDictionary.class],
                         failure ? failure.localizedDescription.UTF8String :
                                   "invalid safetensors JSON root");
        NSDictionary *root = (NSDictionary *)parsed;
        NSArray *keys = [[root allKeys]
            sortedArrayUsingSelector:@selector(compare:)];
        std::vector<std::pair<uint64_t, uint64_t>> intervals;
        const uint64_t payload_bytes = state_->file_bytes - 8 - header_bytes;
        for (id raw_key in keys) {
            require_metadata([raw_key isKindOfClass:NSString.class],
                             "safetensors key is not a string");
            NSString *key = (NSString *)raw_key;
            if ([key isEqualToString:@"__metadata__"])
                continue;
            const char *utf8 = key.UTF8String;
            require_metadata(utf8 && std::strlen(utf8) ==
                         [key lengthOfBytesUsingEncoding:NSUTF8StringEncoding],
                             "invalid tensor name");
            id raw_record = root[key];
            require_metadata([raw_record isKindOfClass:NSDictionary.class],
                             "tensor record is not an object");
            NSDictionary *record = (NSDictionary *)raw_record;
            require_metadata([record[@"dtype"] isKindOfClass:NSString.class],
                             "invalid tensor dtype");
            const std::string dtype = [record[@"dtype"] UTF8String];
            require_metadata(dtype == "BF16" || dtype == "F32" ||
                                 dtype == "I8" || dtype == "U8",
                             "expected BF16 or INT8 ConvRot tensor dtype");
            state_->convrot |= dtype == "I8";
            NSArray *shape = record[@"shape"];
            NSArray *offsets = record[@"data_offsets"];
            require_metadata([shape isKindOfClass:NSArray.class] &&
                                 shape.count > 0 && shape.count <= 8 &&
                                 [offsets isKindOfClass:NSArray.class] &&
                                 offsets.count == 2,
                             "invalid tensor shape or offsets");

            TensorRecord tensor;
            tensor.name = utf8;
            tensor.dtype = dtype;
            uint64_t bytes = dtype == "F32" ? 4 : dtype == "BF16" ? 2 : 1;
            for (id raw_dimension in shape) {
                const uint64_t dimension = integer(raw_dimension,
                                                   "tensor dimension");
                require_metadata(dimension &&
                                     bytes <= UINT64_MAX / dimension,
                                 "tensor shape size overflow");
                bytes *= dimension;
                tensor.shape.push_back(dimension);
            }
            tensor.data_begin = integer(offsets[0], "tensor offset");
            tensor.data_end = integer(offsets[1], "tensor offset");
            require_metadata(tensor.data_end >= tensor.data_begin &&
                                 tensor.data_end - tensor.data_begin == bytes &&
                                 tensor.data_end <= payload_bytes,
                             "tensor range differs from dtype/shape");
            tensor.file_offset = checked_add(
                8 + header_bytes, tensor.data_begin,
                "tensor file offset overflow");
            tensor.bytes = bytes;
            intervals.emplace_back(tensor.data_begin, tensor.data_end);

            if (tensor.name.starts_with("layers.")) {
                auto [block, suffix] = layer_identity(tensor.name);
                tensor.suffix = std::move(suffix);
                state_->blocks[block].push_back(std::move(tensor));
            } else {
                require_metadata(fixed_name(tensor.name),
                                 "checkpoint is not the Comfy Z-Image layout");
                state_->fixed_total = checked_add(
                    state_->fixed_total, tensor.bytes,
                    "fixed tensor byte count overflow");
                state_->fixed.push_back(std::move(tensor));
            }
        }

        std::sort(intervals.begin(), intervals.end());
        uint64_t cursor = 0;
        for (const auto &[begin, end] : intervals) {
            require_metadata(begin == cursor,
                             "overlapping or noncontiguous tensor ranges");
            cursor = end;
        }
        require_metadata(cursor == payload_bytes,
                         "unexpected safetensors payload length");
    }

    require_metadata(std::any_of(
                         state_->fixed.begin(), state_->fixed.end(),
                         [](const TensorRecord &record) {
                             return record.name == "x_embedder.weight";
                         }),
                     "missing Comfy Z-Image embedding weights");
    for (auto &block : state_->blocks)
        std::sort(block.begin(), block.end(),
                  [](const TensorRecord &left, const TensorRecord &right) {
                      return left.suffix < right.suffix;
                  });
    const size_t expected_fields = state_->convrot ? 25 : 13;
    require_metadata(state_->blocks[0].size() == expected_fields,
                     "Z-Image requires 13 BF16 or 25 ConvRot source tensors per block");
    for (uint32_t block = 0; block < state_->blocks.size(); ++block) {
        require_metadata(state_->blocks[block].size() == expected_fields,
                         "Z-Image shadow requires 30 dense blocks");
        uint64_t block_bytes = 0;
        for (uint32_t field = 0; field < state_->blocks[block].size(); ++field) {
            const auto &expected = state_->blocks[0][field];
            const auto &actual = state_->blocks[block][field];
            require_metadata(actual.suffix == expected.suffix &&
                                 actual.shape == expected.shape &&
                                 actual.dtype == expected.dtype &&
                                 actual.bytes == expected.bytes,
                             "Z-Image BF16/ConvRot blocks must have matching tensor layouts");
            block_bytes = checked_add(block_bytes, actual.bytes,
                                      "block byte count overflow");
        }
        if (!block)
            state_->block_total = block_bytes;
        require_metadata(block_bytes == state_->block_total,
                         "Z-Image block byte counts differ");
    }
    prepare_materializations();
    check_unchanged();
}

StreamingMetadata::~StreamingMetadata() = default;

void StreamingMetadata::prepare_materializations() {
    auto find = [](std::vector<TensorRecord> &records,
                   const std::string &name) -> TensorRecord & {
        auto it = std::find_if(records.begin(), records.end(),
                              [&](const auto &r) { return r.name == name; });
        require_metadata(it != records.end(), "missing ConvRot/suffix companion: " + name);
        return *it;
    };
    const auto prefix = state_->options.mlp_prefix_channels;
    if (prefix) {
        require_metadata(!state_->convrot || prefix % 256 == 0,
                         "ConvRot suffix must align to rotation groups");
        auto suffix = [&](std::vector<TensorRecord> &records, const std::string &key) {
            auto &w1 = find(records, key + ".feed_forward.w1.weight");
            auto &w2 = find(records, key + ".feed_forward.w2.weight");
            auto &w3 = find(records, key + ".feed_forward.w3.weight");
            require_metadata(w1.shape.size() == 2 && w2.shape.size() == 2 &&
                                 w1.shape == w3.shape && w1.shape[0] == w2.shape[1] &&
                                 w1.shape[1] == w2.shape[0] && prefix < w1.shape[0] &&
                                 w1.dtype == w2.dtype && w1.dtype == w3.dtype &&
                                 w1.dtype == (state_->convrot ? "I8" : "BF16"),
                             "invalid Z-Image hybrid suffix geometry");
            const uint64_t item = state_->convrot ? 1 : 2;
            for (auto *r : {&w1, &w3}) {
                const auto skipped = uint64_t(prefix) * r->shape[1] * item;
                r->file_offset += skipped;
                r->bytes -= skipped;
                r->shape[0] -= prefix;
                if (state_->convrot) {
                    auto &scale = find(records, r->name.substr(0, r->name.size() - 7) + ".weight_scale");
                    require_metadata(scale.dtype == "F32" &&
                                         scale.shape == std::vector<uint64_t>{r->shape[0] + prefix, 1},
                                     "invalid ConvRot suffix scale geometry");
                    scale.file_offset += uint64_t(prefix) * 4;
                    scale.bytes -= uint64_t(prefix) * 4;
                    scale.shape[0] -= prefix;
                }
            }
            state_->packs.push_back({w2.file_offset, state_->packed_bytes,
                                     w2.shape[0], w2.shape[1] * item, prefix * item});
            w2.file_offset = state_->packed_bytes;
            w2.shape[1] -= prefix;
            w2.bytes = w2.shape[0] * w2.shape[1] * item;
            w2.artifact = 1;
            state_->packed_bytes = checked_add(state_->packed_bytes, w2.bytes,
                                               "suffix pack size overflow");
        };
        for (uint32_t i = 0; i < 2; ++i)
            suffix(state_->fixed, "noise_refiner." + std::to_string(i));
        for (uint32_t i = 0; i < 30; ++i)
            suffix(state_->blocks[i], "layers." + std::to_string(i));
    }
    auto prepare = [&](std::vector<TensorRecord> &records) {
        std::vector<TensorRecord> output;
        for (auto r : records) {
            r.source = {r.artifact, r.file_offset, r.bytes, r.name, r.dtype, r.shape};
            if (!state_->convrot) {
                require_metadata(r.dtype == "BF16", "non-ConvRot streaming requires BF16 tensors");
            } else if (r.name.ends_with(".weight_scale")) {
                require_metadata(find(records, r.name.substr(0, r.name.size() - 13) + ".weight").dtype == "I8",
                                 "ConvRot scale has no signed INT8 weight");
                continue;
            } else if (r.dtype == "I8") {
                require_metadata(r.name.ends_with(".weight") && r.shape.size() == 2 &&
                                     r.shape[1] % 256 == 0,
                                 "invalid ConvRot Q8 geometry");
                const auto key = r.name.substr(0, r.name.size() - 7);
                const auto &metadata = find(records, key + ".comfy_quant");
                auto scale = find(records, key + ".weight_scale");
                require_metadata(metadata.dtype == "U8" && scale.dtype == "F32" &&
                                     scale.shape == std::vector<uint64_t>{r.shape[0], 1},
                                 "invalid ConvRot scale geometry");
                scale.source = {scale.artifact, scale.file_offset, scale.bytes,
                                scale.name, scale.dtype, scale.shape};
                scale.name = key + ".scales";
                scale.shape[1] = r.shape[1] / 32;
                scale.dtype = state_->options.fp32_scales ? "F32" : "BF16";
                scale.bytes = scale.shape[0] * scale.shape[1] * (state_->options.fp32_scales ? 4 : 2);
                scale.conversion = StreamingConversion::scales;
                output.push_back(scale);
                scale.name = key + ".biases";
                scale.conversion = StreamingConversion::biases;
                output.push_back(std::move(scale));
                r.shape[1] /= 4;
                r.dtype = "U32";
                r.conversion = StreamingConversion::signed_q8;
            } else if (r.dtype == "F32") {
                r.dtype = "BF16";
                r.bytes /= 2;
                r.conversion = StreamingConversion::bf16;
            } else if (r.dtype == "U8") {
                require_metadata(r.name.ends_with(".comfy_quant") &&
                                     find(records, r.name.substr(0, r.name.size() - 12) + ".weight").dtype == "I8",
                                 "unexpected ConvRot metadata tensor");
            }
            output.push_back(std::move(r));
        }
        uint64_t bytes = 0;
        for (auto &r : output) {
            if (r.name.starts_with("layers.")) r.suffix = layer_identity(r.name).second;
            bytes = checked_add(bytes, r.bytes, "materialized byte count overflow");
        }
        std::sort(output.begin(), output.end(), [](const auto &a, const auto &b) { return a.name < b.name; });
        records = std::move(output);
        return bytes;
    };
    state_->fixed_total = prepare(state_->fixed);
    state_->block_total = prepare(state_->blocks[0]);
    for (uint32_t i = 1; i < 30; ++i)
        require_metadata(prepare(state_->blocks[i]) == state_->block_total,
                         "materialized Z-Image block sizes differ");
}

uint32_t StreamingMetadata::tensors_per_block() const noexcept {
    return static_cast<uint32_t>(state_->blocks[0].size());
}
bool StreamingMetadata::convrot() const noexcept { return state_->convrot; }
uint64_t StreamingMetadata::scratch_bytes_per_slot() const noexcept {
    return state_->convrot ? 65536 : 0;
}
uint64_t StreamingMetadata::packed_bytes() const noexcept { return state_->packed_bytes; }
const StreamingWeightOptions &StreamingMetadata::options() const noexcept { return state_->options; }
const std::vector<StreamingTensor> &StreamingMetadata::fixed_records() const { return state_->fixed; }
const std::vector<StreamingTensor> &StreamingMetadata::block_records(uint32_t block) const {
    return state_->blocks.at(block);
}
const std::vector<StreamingSuffixPack> &StreamingMetadata::suffix_packs() const { return state_->packs; }

uint64_t StreamingMetadata::block_bytes() const noexcept {
    return state_ ? state_->block_total : 0;
}

uint64_t StreamingMetadata::fixed_bytes() const noexcept {
    return state_ ? state_->fixed_total : 0;
}

const std::string &StreamingMetadata::snapshot_identity() const noexcept {
    return state_->identity;
}

void StreamingMetadata::check_unchanged() const {
    require_metadata(state_ && state_->descriptor && state_->lease,
                     "checkpoint descriptor is unavailable");
    try {
        state_->lease->revalidate_open_files();
        state_->lease->revalidate_paths();
    } catch (const std::exception &error) {
        throw std::invalid_argument(
            std::string("z_image_streaming_metadata: checkpoint_changed: ") +
            error.what());
    }
}

const streaming::SourceLease &StreamingMetadata::lease() const {
    require_metadata(state_ && state_->lease,
                     "source lease is unavailable");
    return *state_->lease;
}

std::shared_ptr<const streaming::SourceLease>
StreamingMetadata::lease_ptr() const {
    require_metadata(state_ && state_->lease,
                     "source lease is unavailable");
    return state_->lease;
}

streaming::Descriptor StreamingMetadata::describe_verified(
    const StreamingWorkload &workload) const {
    require_metadata(lease().has_verified_content(),
                     "artifact_verification_required");
    auto descriptor = describe(workload);
    // The whole lease covers auxiliary inputs too, even though this stage's
    // materializations read only the transformer. No binding stat enters here.
    descriptor.checkpoint_identity =
        "artifact-content-v1:" + std::string(lease().artifact_digest());
    descriptor.artifacts.front().identity =
        lease().file(state_->logical_id).content_digest;
    descriptor.artifacts.front().identity_kind =
        streaming::SourceIdentityKind::content_sha256;
    return descriptor;
}

streaming::Descriptor StreamingMetadata::describe(
    const StreamingWorkload &workload) const {
    check_unchanged();
    require_metadata(workload.width >= 16 && workload.height >= 16 &&
                         workload.width % 16 == 0 &&
                         workload.height % 16 == 0 && workload.steps &&
                         workload.steps <= streaming::max_passes &&
                         workload.caption_rows &&
                         workload.caption_rows % 32 == 0,
                     "workload must be normalized before describe");
    const uint64_t image_tokens =
        static_cast<uint64_t>(workload.width / 16) *
        static_cast<uint64_t>(workload.height / 16);
    const uint64_t image_rows = padded_rows(image_tokens);
    const uint64_t unified_rows = checked_add(
        image_rows, workload.caption_rows, "unified token count overflow");

    streaming::Descriptor descriptor{
        "z-image-turbo", "snapshot:" + state_->identity,
        state_->convrot ? "z-image-mlx-convrot-streaming-v1" :
                          "z-image-mlx-bf16-streaming-v1", {}};
    descriptor.artifacts.push_back(
        {"transformer", state_->identity, state_->file_bytes,
         streaming::SourceIdentityKind::snapshot});
    descriptor.workload = {
        {"operation", "image.denoise"},
        {"format", state_->convrot ? "comfy-int8-convrot-single-file" : "comfy-bf16-single-file"},
        {"width", std::to_string(workload.width)},
        {"height", std::to_string(workload.height)},
        {"caption_rows", std::to_string(workload.caption_rows)},
        {"image_rows", std::to_string(image_rows)},
        {"unified_rows", std::to_string(unified_rows)},
        {"hidden", std::to_string(kHidden)},
        {"steps", std::to_string(workload.steps)},
        {"fixed_tensor_count", std::to_string(state_->fixed.size())},
        {"fixed_bytes", std::to_string(state_->fixed_total)},
        {"resident_auxiliary", "refiners-embedders-final-layer-not-in-slots"},
        {"reader_revision", "z-image-pread-bf16-v1"},
        {"reader_cache_policy", "f_nocache-payload-reader"},
    };

    if (state_->convrot) {
        descriptor.workload["reader_revision"] = "z-image-convrot-affine-q8-v1";
        descriptor.workload["quantization_group_size"] = "32";
        descriptor.workload["rotation_group_size"] = "256";
        descriptor.workload["scale_dtype"] = state_->options.fp32_scales ? "F32" : "BF16";
        descriptor.workload["scratch_bytes_per_slot"] = std::to_string(scratch_bytes_per_slot());
        descriptor.workload["kernel_revision"] = "z-image-convrot-packed-q8-v1";
    }
    if (state_->options.mlp_prefix_channels) {
        descriptor.workload["ane_mlp_prefix_channels"] = std::to_string(state_->options.mlp_prefix_channels);
        descriptor.workload["suffix_pack_revision"] = "z-image-suffix-row-pack-v1";
        descriptor.artifacts.push_back({"gpu-suffix", state_->identity + ":suffix-v1:" +
            std::to_string(state_->options.mlp_prefix_channels), state_->packed_bytes,
            streaming::SourceIdentityKind::snapshot});
    }

    streaming::StageDescriptor stage;
    stage.id = "denoiser";
    stage.adapter_revision = state_->convrot ? "z-image-convrot-refill-pool-v1" :
                                              "z-image-bf16-refill-pool-v2-metadata";
    stage.min_slots = 1;
    stage.max_slots = state_->convrot ? 3 : 2;
    stage.max_group_size = 1;
    stage.min_prefix = 0;
    stage.pass_count = workload.steps;
    stage.pass_transition = streaming::PassTransition::reload;
    for (uint32_t step = 0; step < workload.steps; ++step)
        stage.passes.push_back(
            {step, "denoise", {unified_rows, kHidden}});

    auto field_spec = [](const TensorRecord &record, const std::string &name,
                         const std::string &storage) {
        streaming::Materialization materialization;
        materialization.format = record.dtype;
        materialization.storage_mode = "mlx-metal-shared";
        switch (record.conversion) {
        case StreamingConversion::copy:
            materialization.conversion = record.dtype == "BF16" ? "copy-bf16-v1" : "copy-v1";
            break;
        case StreamingConversion::signed_q8:
            materialization.conversion = "signed-q8-xor128-pack-u32-v1";
            break;
        case StreamingConversion::bf16:
            materialization.conversion = "f32-to-bf16-v1";
            break;
        case StreamingConversion::scales:
            materialization.conversion = "convrot-row-scale-expand-g32-v1";
            break;
        case StreamingConversion::biases:
            materialization.conversion = "convrot-rounded-scale-minus128-g32-v1";
            break;
        }
        materialization.shape = record.shape;
        materialization.reads.push_back(record.source);
        return streaming::FieldSpec{name, storage, record.bytes, 256, std::move(materialization)};
    };
    // Preserve the existing BF16 public layout identities. New quantized
    // descriptors close over fixed weights as well as every main-block field.
    if (state_->convrot)
        for (const auto &record : state_->fixed)
            stage.resident_fields.push_back(field_spec(
                record, record.name, "z-image.fixed." + record.name));
    for (uint32_t block_id = 0; block_id < state_->blocks.size(); ++block_id) {
        streaming::BlockSpec block;
        block.id = block_id;
        block.layout_class = state_->convrot ? "z-image-convrot-main-block-v1" :
                                               "z-image-bf16-main-block-v1";
        block.streamable = true;
        block.safe_boundary_after = true;
        for (const auto &record : state_->blocks[block_id])
            block.fields.push_back(field_spec(record, record.suffix,
                "z-image.block." + std::to_string(block_id) + "." + record.suffix));
        stage.blocks.push_back(std::move(block));
    }
    descriptor.stages.push_back(std::move(stage));
    return descriptor;
}

GpuSuffixPlan StreamingMetadata::describe_gpu_suffix(
    const StreamingWorkload &workload, uint32_t first_gpu_channel) const {
    require_metadata(first_gpu_channel > 0 && first_gpu_channel < 10240,
                     "invalid hybrid GPU suffix channel");
    GpuSuffixPlan plan;
    plan.descriptor = lease().has_verified_content()
        ? describe_verified(workload) : describe(workload);
    const auto geometry = suffix_geometry(kHidden, 10240, first_gpu_channel, 2);
    // Validate the complete branch map before producing any partial plan.
    // Noise refiners are fixed, main blocks may be prefix or streamed, and
    // context refiners deliberately never enter this map.
    std::map<std::string, unsigned> projections;
    for (unsigned branch = 0; branch < 32; ++branch) {
        const auto prefix = branch < 2
            ? "noise_refiner." + std::to_string(branch)
            : "layers." + std::to_string(branch - 2);
        const auto &records = branch < 2 ? state_->fixed : state_->blocks[branch - 2];
        for (unsigned projection = 1; projection <= 3; ++projection) {
            const auto name = prefix + ".feed_forward.w" + std::to_string(projection) + ".weight";
            const auto it = std::find_if(records.begin(), records.end(),
                [&](const TensorRecord &r) { return r.name == name; });
            const std::vector<uint64_t> expected = projection == 2
                ? std::vector<uint64_t>{kHidden, 10240}
                : std::vector<uint64_t>{10240, kHidden};
            require_metadata(it != records.end() && it->shape == expected,
                             "missing or invalid hybrid FFN tensor");
            projections.emplace(name, projection);
        }
    }
    streaming::CanonicalEncoder recipe("tc-z-image-bf16-suffix-recipe-v1");
    recipe.string_field("parent_identity", plan.descriptor.artifacts.front().identity);
    recipe.string_field("parent_identity_kind", lease().has_verified_content() ? "content_sha256" : "snapshot");
    recipe.unsigned_field("parent_bytes", state_->file_bytes);
    recipe.unsigned_field("first_gpu_channel", first_gpu_channel);
    recipe.string_field("conversion", "bf16-row-suffix-v1");
    recipe.begin_list("tensors", projections.size());
    auto field = [&](const TensorRecord &r, const std::string &name,
                     const std::string &storage_id) {
        streaming::Materialization m;
        m.format = "BF16";
        m.storage_mode = "mlx-metal-shared";
        m.conversion = "copy-bf16-v1";
        m.shape = r.shape;
        streaming::SourceRange read{0, r.file_offset, r.bytes, r.name, "BF16", r.shape};
        uint64_t bytes = r.bytes;
        if (const auto it = projections.find(r.name); it != projections.end()) {
            recipe.string_field("tensor", r.name);
            recipe.unsigned_field("source_offset", r.file_offset);
            recipe.unsigned_field("source_bytes", r.bytes);
            recipe.unsigned_field("projection", it->second);
            if (it->second == 2) {
                m.shape[1] -= first_gpu_channel;
                bytes = geometry.down_suffix_bytes;
                plan.packing.push_back({read, plan.setup_write_bytes, bytes, first_gpu_channel});
                read.artifact = 1;
                read.offset = plan.setup_write_bytes;
                read.bytes = bytes;
                read.shape = m.shape;
                plan.setup_read_bytes = checked_add(plan.setup_read_bytes, r.bytes, "packing read overflow");
                plan.setup_write_bytes = checked_add(plan.setup_write_bytes, bytes, "packing write overflow");
            } else {
                m.shape[0] -= first_gpu_channel;
                bytes = geometry.up_suffix_bytes;
                read.offset = checked_add(r.file_offset, geometry.up_skip_bytes, "suffix offset overflow");
                read.bytes = bytes;
                read.shape = m.shape;
            }
        }
        m.reads.push_back(std::move(read));
        return streaming::FieldSpec{name, storage_id, bytes, 256, std::move(m)};
    };
    auto &stage = plan.descriptor.stages.front();
    stage.adapter_revision = "z-image-bf16-gpu-suffix-v1-metadata-only";
    // Foundation dictionary enumeration is not a canonical order.
    auto fixed = state_->fixed;
    std::sort(fixed.begin(), fixed.end(), [](const auto &a, const auto &b) { return a.name < b.name; });
    uint64_t fixed_bytes = 0;
    for (const auto &r : fixed) {
        auto f = field(r, r.name, "z-image.fixed." + r.name);
        fixed_bytes = checked_add(fixed_bytes, f.bytes, "suffix fixed bytes overflow");
        stage.resident_fields.push_back(std::move(f));
    }
    for (auto &block : stage.blocks) {
        block.layout_class = "z-image-bf16-suffix-main-block-v1";
        block.fields.clear();
        for (const auto &r : state_->blocks[block.id])
            block.fields.push_back(field(r, r.suffix,
                "z-image.block." + std::to_string(block.id) + "." + r.suffix));
    }
    require_metadata(plan.packing.size() == 32, "incomplete suffix packing map");
    plan.recipe_digest = recipe.sha256();
    // This is a metadata recipe, not the SHA-256 of materialized bytes. The
    // future execution source must bind and verify the private derived fd.
    plan.descriptor.artifacts.push_back({"transformer-gpu-suffix",
        "recipe:" + plan.recipe_digest, plan.setup_write_bytes,
        streaming::SourceIdentityKind::snapshot});
    plan.descriptor.backend_revision = "z-image-bf16-gpu-suffix-v1-metadata-only";
    auto &identity = plan.descriptor.workload;
    identity["fixed_bytes"] = std::to_string(fixed_bytes);
    identity["first_gpu_channel"] = std::to_string(first_gpu_channel);
    identity["suffix_recipe"] = plan.recipe_digest;
    identity["suffix_setup_read_bytes"] = std::to_string(plan.setup_read_bytes);
    identity["suffix_setup_write_bytes"] = std::to_string(plan.setup_write_bytes);
    identity["reader_revision"] = "z-image-derived-fd-bf16-v1-metadata-only";
    check_unchanged();
    return plan;
}

struct GpuSuffixSource::State {
    GpuSuffixPlan plan;
    std::shared_ptr<const streaming::SourceLease> parent;
    std::string logical_id, content_digest;
    streaming::OwnedSourceFd derived;
    struct stat identity{};
};

GpuSuffixSource::GpuSuffixSource(std::unique_ptr<State> state)
    : state_(std::move(state)) {}
GpuSuffixSource::~GpuSuffixSource() = default;
const GpuSuffixPlan &GpuSuffixSource::plan() const noexcept { return state_->plan; }
const std::string &GpuSuffixSource::content_digest() const noexcept { return state_->content_digest; }
uint64_t GpuSuffixSource::verification_read_bytes() const noexcept {
    return state_->plan.setup_write_bytes;
}
const streaming::SourceFileIdentity &GpuSuffixSource::parent_file() const {
    return state_->parent->file(state_->logical_id);
}
void GpuSuffixSource::check_unchanged() const {
    state_->parent->revalidate_after_drain();
    struct stat actual{};
    require_metadata(::fstat(state_->derived.get(), &actual) == 0,
                     "derived suffix fd unavailable");
    const auto &expected = state_->identity;
    require_metadata(S_ISREG(actual.st_mode) && actual.st_nlink == 0 &&
        actual.st_dev == expected.st_dev && actual.st_ino == expected.st_ino &&
        actual.st_size == expected.st_size &&
        actual.st_mtimespec.tv_sec == expected.st_mtimespec.tv_sec &&
        actual.st_mtimespec.tv_nsec == expected.st_mtimespec.tv_nsec &&
        actual.st_ctimespec.tv_sec == expected.st_ctimespec.tv_sec &&
        actual.st_ctimespec.tv_nsec == expected.st_ctimespec.tv_nsec,
        "derived suffix changed");
}
streaming::OwnedSourceFd GpuSuffixSource::duplicate_fd(uint32_t artifact) const {
    require_metadata(artifact < 2, "unknown suffix artifact");
    check_unchanged();
    if (artifact == 0) return state_->parent->duplicate_fd(state_->logical_id);
    const int fd = ::fcntl(state_->derived.get(), F_DUPFD_CLOEXEC, 0);
    require_metadata(fd >= 0, "cannot duplicate derived suffix fd");
    return streaming::OwnedSourceFd(fd);
}

std::unique_ptr<GpuSuffixSource> StreamingMetadata::materialize_gpu_suffix(
    const StreamingWorkload &workload, uint32_t first_gpu_channel,
    std::atomic<bool> &cancelled, const Event &event) const {
    checkpoint(cancelled);
    require_metadata(lease().has_verified_content(), "artifact_verification_required");
    auto ready = std::make_unique<GpuSuffixSource::State>();
    ready->plan = describe_gpu_suffix(workload, first_gpu_channel);
    ready->parent = lease_ptr();
    ready->logical_id = state_->logical_id;
    auto source = ready->parent->duplicate_fd(ready->logical_id);
    require_metadata(::fcntl(source.get(), F_NOCACHE, 1) == 0,
                     "cannot configure suffix source reader");
    // Open both handles while the exclusively-created file is named, verify
    // that they refer to the same inode, then unlink before writing payload.
    // Only the read-only handle is allowed to escape after final verification.
    auto pattern = (std::filesystem::path(NSTemporaryDirectory().UTF8String) /
                    "turbocider-z-derived-XXXXXX").string();
    streaming::OwnedSourceFd writable(::mkstemp(pattern.data()));
    struct Unlink {
        const std::string &path;
        bool pending = true;
        ~Unlink() { if (pending) ::unlink(path.c_str()); }
    } cleanup{pattern, bool(writable)};
    require_metadata(bool(writable), "cannot create derived suffix file");
    ready->derived = streaming::OwnedSourceFd(::open(pattern.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    require_metadata(bool(ready->derived), "cannot open derived suffix reader");
    struct stat writer_stat{}, reader_stat{};
    require_metadata(::fstat(writable.get(), &writer_stat) == 0 &&
                     ::fstat(ready->derived.get(), &reader_stat) == 0 &&
                     writer_stat.st_dev == reader_stat.st_dev && writer_stat.st_ino == reader_stat.st_ino,
                     "derived suffix binding changed");
    require_metadata(::unlink(pattern.c_str()) == 0, "cannot unlink derived suffix file");
    cleanup.pending = false;
    require_metadata(::fcntl(writable.get(), F_SETFD, FD_CLOEXEC) == 0 &&
                     ::fcntl(writable.get(), F_NOCACHE, 1) == 0 &&
                     ::fcntl(ready->derived.get(), F_NOCACHE, 1) == 0,
                     "cannot configure derived suffix file");
    SuffixPackMetrics actual;
    for (size_t i = 0; i < ready->plan.packing.size(); ++i) {
        checkpoint(cancelled);
        if (event) event("pack_z_image_suffix", int(i), int(ready->plan.packing.size()));
        const auto &record = ready->plan.packing[i];
        const auto geometry = suffix_geometry(record.source.shape[0], record.source.shape[1],
                                              record.first_gpu_channel, 2);
        pack_suffix_rows(source.get(), record.source.offset, writable.get(), record.destination_offset,
                         geometry, cancelled, actual);
    }
    require_metadata(actual.read_bytes == ready->plan.setup_read_bytes &&
                     actual.write_bytes == ready->plan.setup_write_bytes,
                     "derived suffix I/O differs from plan");
    // Close the last writable handle before computing the actual content hash.
    writable = streaming::OwnedSourceFd();
    require_metadata(::fstat(ready->derived.get(), &ready->identity) == 0 &&
                     ready->identity.st_nlink == 0 && ready->identity.st_size >= 0 &&
                     uint64_t(ready->identity.st_size) == ready->plan.setup_write_bytes,
                     "derived suffix size differs from plan");
    if (event) event("verify_z_image_suffix", 0, 1);
    ready->content_digest = memory_sha256_fd(ready->derived.get(), ready->plan.setup_write_bytes, &cancelled);
    if (event) event("verify_z_image_suffix", 1, 1);
    checkpoint(cancelled);
    auto result = std::unique_ptr<GpuSuffixSource>(new GpuSuffixSource(std::move(ready)));
    result->check_unchanged();
    return result;
}

StreamingPlanView::StreamingPlanView(
    const std::string &checkpoint, const StreamingConfig &config,
    const StreamingWorkload &workload)
    : metadata_(checkpoint, {workload.ane_mlp_prefix_channels, workload.fp32_scales}),
      descriptor_(metadata_.describe(workload)),
      layout_(streaming::compile_layout(config, descriptor_)) {
    validate();
}

StreamingPlanView::StreamingPlanView(
    std::shared_ptr<const streaming::SourceLease> lease,
    const StreamingConfig &config, const StreamingWorkload &workload)
    : metadata_(std::move(lease), "transformer",
                {workload.ane_mlp_prefix_channels, workload.fp32_scales}),
      descriptor_(metadata_.lease().has_verified_content()
          ? metadata_.describe_verified(workload) : metadata_.describe(workload)),
      layout_(streaming::compile_layout(config, descriptor_)) {
    validate();
}

void StreamingPlanView::validate() const {
    require_metadata(layout_.materializations_complete,
                     "Z-Image descriptor metadata is incomplete");
    require_metadata(layout_.stages.size() == 1,
                     "Z-Image shadow requires one stage");
    const auto &stage = layout_.stages.front();
    require_metadata(stage.id == "denoiser" && !stage.resident &&
                         stage.group_size == 1 && stage.slot_count >= 1 &&
                         stage.slot_count <= (metadata_.convrot() ? 3u : 2u) &&
                         stage.distance < stage.slot_count &&
                         stage.workers >= 1 && stage.workers <= stage.slot_count &&
                         stage.pass_transition ==
                             streaming::PassTransition::reload &&
                         stage.pools.size() == 1,
                     "Z-Image shadow requires K=1..2 (BF16) or 1..3 (ConvRot)/G=1/D<K/1<=Q<=K reload");
    require_metadata(stage.prefix < descriptor_.stages.front().blocks.size() &&
                         stage.groups.size() ==
                             descriptor_.stages.front().blocks.size() -
                                 stage.prefix,
                     "Z-Image compiled suffix differs from the descriptor");
    for (const auto &group : stage.groups)
        require_metadata(group.blocks.size() == 1 && group.pool == 0 &&
                             group.bytes == metadata_.block_bytes(),
                         "Z-Image shadow requires one compatible block per group");
}

} // namespace tc::z_image
