#import <Foundation/Foundation.h>

#include "../../models/z_image/streaming_descriptor.hpp"

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

struct TensorRecord {
    std::string name;
    std::string suffix;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
    uint64_t file_offset = 0;
    uint64_t bytes = 0;
};

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

StreamingMetadata::StreamingMetadata(const std::string &checkpoint)
    : StreamingMetadata(capture_checkpoint_lease(checkpoint),
                        "transformer") {}

StreamingMetadata::StreamingMetadata(
        std::shared_ptr<const streaming::SourceLease> lease,
        std::string logical_id)
    : state_(std::make_unique<State>()) {
    require_metadata(lease != nullptr, "source lease is unavailable");
    require_metadata(!logical_id.empty(), "source logical id is empty");
    const auto &file = lease->file(logical_id);
    require_metadata(file.path.extension() == ".safetensors",
                     "Z-Image shadow requires a safetensors checkpoint");
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
            require_metadata([record[@"dtype"] isKindOfClass:NSString.class] &&
                                 [record[@"dtype"] isEqualToString:@"BF16"],
                             "Z-Image shadow supports BF16 weights only");
            NSArray *shape = record[@"shape"];
            NSArray *offsets = record[@"data_offsets"];
            require_metadata([shape isKindOfClass:NSArray.class] &&
                                 shape.count > 0 && shape.count <= 8 &&
                                 [offsets isKindOfClass:NSArray.class] &&
                                 offsets.count == 2,
                             "invalid tensor shape or offsets");

            TensorRecord tensor;
            tensor.name = utf8;
            uint64_t bytes = sizeof(uint16_t);
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
                             "tensor range differs from BF16 shape");
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
    require_metadata(state_->blocks[0].size() == 13,
                     "Z-Image shadow requires 13 tensors per block");
    for (uint32_t block = 0; block < state_->blocks.size(); ++block) {
        require_metadata(state_->blocks[block].size() == 13,
                         "Z-Image shadow requires 30 dense blocks");
        uint64_t block_bytes = 0;
        for (uint32_t field = 0; field < state_->blocks[block].size(); ++field) {
            const auto &expected = state_->blocks[0][field];
            const auto &actual = state_->blocks[block][field];
            require_metadata(actual.suffix == expected.suffix &&
                                 actual.shape == expected.shape &&
                                 actual.bytes == expected.bytes,
                             "Z-Image blocks must have matching tensor layouts");
            block_bytes = checked_add(block_bytes, actual.bytes,
                                      "block byte count overflow");
        }
        if (!block)
            state_->block_total = block_bytes;
        require_metadata(block_bytes == state_->block_total,
                         "Z-Image block byte counts differ");
    }
    check_unchanged();
}

StreamingMetadata::~StreamingMetadata() = default;

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
        "z-image-mlx-bf16-streaming-v1", {}};
    descriptor.artifacts.push_back(
        {"transformer", state_->identity, state_->file_bytes,
         streaming::SourceIdentityKind::snapshot});
    descriptor.workload = {
        {"operation", "image.denoise"},
        {"format", "comfy-bf16-single-file"},
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

    streaming::StageDescriptor stage;
    stage.id = "denoiser";
    stage.adapter_revision = "z-image-bf16-refill-pool-v2-metadata";
    stage.min_slots = 1;
    stage.max_slots = 2;
    stage.max_group_size = 1;
    stage.min_prefix = 0;
    stage.pass_count = workload.steps;
    stage.pass_transition = streaming::PassTransition::reload;
    for (uint32_t step = 0; step < workload.steps; ++step)
        stage.passes.push_back(
            {step, "denoise", {unified_rows, kHidden}});

    for (uint32_t block_id = 0; block_id < state_->blocks.size(); ++block_id) {
        streaming::BlockSpec block;
        block.id = block_id;
        block.layout_class = "z-image-bf16-main-block-v1";
        block.streamable = true;
        block.safe_boundary_after = true;
        for (const auto &record : state_->blocks[block_id]) {
            streaming::Materialization materialization;
            materialization.format = "BF16";
            materialization.storage_mode = "mlx-metal-shared";
            materialization.conversion = "copy-bf16-v1";
            materialization.shape = record.shape;
            materialization.reads.push_back(
                {0, record.file_offset, record.bytes, record.name, "BF16",
                 record.shape});
            block.fields.push_back(
                {record.suffix,
                 "z-image.block." + std::to_string(block_id) + "." +
                     record.suffix,
                 record.bytes, 256, std::move(materialization)});
        }
        stage.blocks.push_back(std::move(block));
    }
    descriptor.stages.push_back(std::move(stage));
    return descriptor;
}

StreamingPlanView::StreamingPlanView(
    const std::string &checkpoint, const StreamingConfig &config,
    const StreamingWorkload &workload)
    : metadata_(checkpoint), descriptor_(metadata_.describe(workload)),
      layout_(streaming::compile_layout(config, descriptor_)) {
    validate();
}

StreamingPlanView::StreamingPlanView(
    std::shared_ptr<const streaming::SourceLease> lease,
    const StreamingConfig &config, const StreamingWorkload &workload)
    : metadata_(std::move(lease)), descriptor_(metadata_.describe(workload)),
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
                         stage.slot_count <= 2 &&
                         stage.distance < stage.slot_count &&
                         stage.workers >= 1 && stage.workers <= stage.slot_count &&
                         stage.pass_transition ==
                             streaming::PassTransition::reload &&
                         stage.pools.size() == 1,
                     "Z-Image shadow requires K=1 or K=2/G=1/D<K/1<=Q<=K reload");
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
