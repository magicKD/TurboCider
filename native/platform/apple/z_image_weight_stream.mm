#import <Foundation/Foundation.h>
#include "../../models/z_image/weight_stream.hpp"
#include "../../runtime/residency.hpp"
#include "platform.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tc {
namespace {
uint64_t integer(id value) {
    require([value isKindOfClass:NSNumber.class] &&
                CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            "streamed weight metadata must contain integers");
    double n = [value doubleValue];
    require(std::isfinite(n) && n >= 0 && n == std::floor(n) && n <= double(1ull << 50),
            "invalid streamed weight size");
    return uint64_t(n);
}
} // namespace

ZImageWeightStream::ReadResult ZImageWeightStream::read(const std::vector<Read> &reads) const {
    auto start = Clock::now();
    ReadResult result;
    for (const auto &r : reads) {
        uint64_t done = 0;
        while (done < r.bytes) {
            checkpoint(cancelled_);
            auto n = ::pread(r.fd < 0 ? fd_ : r.fd, r.destination + done,
                             size_t(std::min<uint64_t>(4ull << 20, r.bytes - done)),
                             off_t(r.offset + done));
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "Z-Image streamed weight read failed or file was truncated");
            done += uint64_t(n);
        }
        result.bytes += done;
    }
    result.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

void ZImageWeightStream::index(const std::filesystem::path &path) {
    require(std::filesystem::is_regular_file(path) && path.extension() == ".safetensors",
            "Z-Image streaming requires a single Comfy BF16 or INT8 ConvRot safetensors checkpoint");
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(fd_ >= 0, "cannot open Z-Image streamed checkpoint");
    struct stat status{};
    require(::fstat(fd_, &status) == 0 && status.st_size >= 8,
            "invalid Z-Image streamed checkpoint");
    file_bytes_ = uint64_t(status.st_size);
    modified_seconds_ = status.st_mtimespec.tv_sec;
    modified_nanos_ = status.st_mtimespec.tv_nsec;
    // A file-cache copy of every streamed layer would compete with our bounded
    // GPU working set and reintroduce the memory pressure this mode avoids.
    require(::fcntl(fd_, F_NOCACHE, 1) == 0, "cannot disable streamed weight file caching");
    unsigned char length[8];
    read({{0, 8, reinterpret_cast<char *>(length)}});
    uint64_t header_bytes = 0;
    for (int i = 0; i < 8; ++i) header_bytes |= uint64_t(length[i]) << (8 * i);
    require(header_bytes > 0 && header_bytes <= (16ull << 20) && header_bytes <= file_bytes_ - 8,
            "invalid Z-Image streamed safetensors header length");
    std::vector<char> header(header_bytes);
    read({{8, header_bytes, header.data()}});
    @autoreleasepool {
        auto data = [NSData dataWithBytes:header.data() length:header.size()];
        id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        require([object isKindOfClass:NSDictionary.class], "invalid streamed safetensors JSON");
        id qkv = object[@"layers.0.attention.qkv.weight"];
        convrot_ = [qkv isKindOfClass:NSDictionary.class] && [qkv[@"dtype"] isEqual:@"I8"] &&
                   object[@"layers.0.attention.qkv.comfy_quant"] != nil;
        std::vector<std::pair<uint64_t, uint64_t>> intervals;
        for (NSString *key in (NSDictionary *)object) {
            if ([key isEqualToString:@"__metadata__"]) continue;
            id value = object[key];
            require([value isKindOfClass:NSDictionary.class], "invalid streamed tensor record");
            NSString *dtype = value[@"dtype"];
            require([dtype isKindOfClass:NSString.class], "invalid streamed tensor dtype");
            require([dtype isEqual:@"BF16"] || (convrot_ &&
                        ([dtype isEqual:@"F32"] || [dtype isEqual:@"I8"] || [dtype isEqual:@"U8"])),
                    "Z-Image streaming supports Comfy BF16 or INT8 ConvRot weights only");
            NSArray *shape = value[@"shape"], *offsets = value[@"data_offsets"];
            require([shape isKindOfClass:NSArray.class] && shape.count <= 8 &&
                        [offsets isKindOfClass:NSArray.class] && offsets.count == 2,
                    "invalid streamed tensor shape or offsets");
            Record r;
            require(key.UTF8String && strlen(key.UTF8String) == [key lengthOfBytesUsingEncoding:NSUTF8StringEncoding],
                    "invalid streamed tensor name");
            r.name = key.UTF8String;
            r.dtype = [dtype isEqual:@"F32"] ? mx::float32 : [dtype isEqual:@"I8"] ? mx::int8 :
                      [dtype isEqual:@"U8"] ? mx::uint8 : mx::bfloat16;
            uint64_t bytes = r.dtype.size();
            for (id item in shape) {
                auto dim = integer(item);
                require(dim > 0 && dim <= INT32_MAX && bytes <= file_bytes_ / dim,
                        "invalid or overflowing streamed tensor shape");
                bytes *= dim;
                r.shape.push_back(int(dim));
            }
            auto lo = integer(offsets[0]), hi = integer(offsets[1]);
            require(hi >= lo && hi - lo == bytes && hi <= file_bytes_ - 8 - header_bytes,
                    "invalid or truncated streamed tensor data range");
            r.offset = 8 + header_bytes + lo;
            r.bytes = bytes;
            intervals.emplace_back(lo, hi);
            if (r.name.starts_with("layers.")) {
                auto end = r.name.find('.', 7);
                require(end != std::string::npos, "invalid Z-Image layer key");
                auto number = r.name.substr(7, end - 7);
                require(!number.empty() && number.size() <= 2 &&
                            number.find_first_not_of("0123456789") == std::string::npos,
                        "invalid Z-Image layer number");
                int block = std::stoi(number);
                require(block < 30 && number == std::to_string(block), "invalid Z-Image layer number");
                blocks_[block].push_back(std::move(r));
            } else {
                require(r.name.starts_with("noise_refiner.") || r.name.starts_with("context_refiner.") ||
                            r.name.starts_with("x_embedder.") || r.name.starts_with("cap_embedder.") ||
                            r.name.starts_with("t_embedder.") || r.name.starts_with("final_layer.") ||
                            r.name == "x_pad_token" || r.name == "cap_pad_token",
                        "streaming requires Comfy Z-Image key layout (Diffusers shards are not yet supported)");
                fixed_records_.push_back(std::move(r));
            }
        }
        std::sort(intervals.begin(), intervals.end());
        uint64_t end = 0;
        for (auto [lo, hi] : intervals) {
            require(lo == end, "overlapping or noncontiguous streamed tensor data");
            end = hi;
        }
        require(end == file_bytes_ - 8 - header_bytes, "unexpected streamed checkpoint payload length");
    }
    for (auto &block : blocks_) {
        std::sort(block.begin(), block.end(), [](const Record &a, const Record &b) { return a.name < b.name; });
        require(block.size() == (convrot_ ? 25 : 13),
                "streaming requires 30 matching Comfy Z-Image layers with fused QKV");
    }
    for (int i = 1; i < 30; ++i)
        for (size_t j = 0; j < blocks_[0].size(); ++j) {
            const auto &a = blocks_[0][j], &b = blocks_[i][j];
            require(a.name.substr(9) == b.name.substr(8 + std::to_string(i).size()) &&
                        a.shape == b.shape && a.dtype == b.dtype,
                    "Z-Image streamed layers must have matching tensor layouts");
        }
    require(std::any_of(fixed_records_.begin(), fixed_records_.end(), [](const Record &r) {
                return r.name == "x_embedder.weight";
            }), "missing Comfy Z-Image embedding weights");
}

void ZImageWeightStream::pack_suffix(int prefix_channels, const Event &event) {
    require(prefix_channels > 0, "invalid Z-Image ANE prefix width");
    require(!convrot_ || prefix_channels % 256 == 0,
            "ConvRot GPU suffix must align to the 256-channel rotation group");
    auto start = Clock::now();
    // Validate all 32 hybrid MLPs before changing any records. Context refiners
    // have no ANE branch and must retain their full weights.
    std::vector<std::array<Record *, 3>> groups;
    auto add = [&](std::vector<Record> &records, const std::string &prefix) {
        std::array<Record *, 3> group{};
        for (auto &r : records)
            for (int i = 0; i < 3; ++i)
                if (r.name == prefix + ".feed_forward.w" + std::to_string(i + 1) + ".weight")
                    group[i] = &r;
        require(group[0] && group[1] && group[2], "missing Z-Image hybrid MLP weights");
        const auto &a = group[0]->shape, &b = group[1]->shape, &c = group[2]->shape;
        require(a.size() == 2 && b.size() == 2 && a == c &&
                    a[0] == b[1] && a[1] == b[0] && prefix_channels < a[0],
                "invalid Z-Image hybrid MLP suffix geometry");
        groups.push_back(group);
    };
    for (int i = 0; i < 2; ++i) add(fixed_records_, "noise_refiner." + std::to_string(i));
    for (int i = 0; i < 30; ++i) add(blocks_[i], "layers." + std::to_string(i));

    // Only down-projection columns need repacking. The other two suffixes are
    // already contiguous in the original checkpoint. An unlinked private file
    // ties this derived data to this validated source and session, without stale
    // cache identities or persistent artifacts after cancellation / unload.
    auto pattern = (std::filesystem::path(NSTemporaryDirectory().UTF8String) /
                    "turbocider-z-suffix-XXXXXX").string();
    packed_fd_ = ::mkstemp(pattern.data());
    require(packed_fd_ >= 0, "cannot create Z-Image GPU suffix temporary file");
    require(::unlink(pattern.c_str()) == 0, "cannot unlink Z-Image GPU suffix temporary file");
    require(::fcntl(packed_fd_, F_SETFD, FD_CLOEXEC) == 0 &&
                ::fcntl(packed_fd_, F_NOCACHE, 1) == 0,
            "cannot configure Z-Image GPU suffix temporary file");
    uint64_t packed_offset = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
        checkpoint(cancelled_);
        event("pack_z_image_suffix", int(i), int(groups.size()));
        auto &group = groups[i];
        for (int j : {0, 2}) {
            auto &r = *group[j];
            const auto skipped = uint64_t(prefix_channels) * r.shape[1] * r.dtype.size();
            r.offset += skipped;
            r.bytes -= skipped;
            r.shape[0] -= prefix_channels;
            if (convrot_) {
                auto &records = i < 2 ? fixed_records_ : blocks_[i - 2];
                const auto scale_name = r.name.substr(0, r.name.size() - 7) + ".weight_scale";
                auto scale = std::find_if(records.begin(), records.end(), [&](const Record &v) {
                    return v.name == scale_name;
                });
                require(scale != records.end() && scale->dtype == mx::float32 &&
                            scale->shape == mx::Shape{r.shape[0] + prefix_channels, 1},
                        "invalid ConvRot suffix row scales");
                scale->offset += uint64_t(prefix_channels) * 4;
                scale->bytes -= uint64_t(prefix_channels) * 4;
                scale->shape[0] -= prefix_channels;
            }
        }
        auto &r = *group[1];
        const uint64_t row_bytes = uint64_t(r.shape[1]) * r.dtype.size();
        const uint64_t skip_bytes = uint64_t(prefix_channels) * r.dtype.size();
        const uint64_t suffix_bytes = row_bytes - skip_bytes;
        require(row_bytes <= (4ull << 20), "Z-Image MLP row exceeds suffix packing limit");
        const uint64_t batch_rows = std::max<uint64_t>(1, (4ull << 20) / row_bytes);
        std::vector<char> buffer(std::min<uint64_t>(r.shape[0], batch_rows) * row_bytes);
        for (uint64_t row = 0; row < uint64_t(r.shape[0]); row += batch_rows) {
            const auto count = std::min<uint64_t>(batch_rows, r.shape[0] - row);
            auto loaded = read({{r.offset + row * row_bytes, count * row_bytes, buffer.data()}});
            metrics_.request_pack_read_bytes += loaded.bytes;
            for (uint64_t j = 0; j < count; ++j)
                std::memmove(buffer.data() + j * suffix_bytes,
                             buffer.data() + j * row_bytes + skip_bytes, suffix_bytes);
            uint64_t done = 0, bytes = count * suffix_bytes;
            while (done < bytes) {
                checkpoint(cancelled_);
                auto n = ::pwrite(packed_fd_, buffer.data() + done, size_t(bytes - done),
                                  off_t(packed_offset + row * suffix_bytes + done));
                if (n < 0 && errno == EINTR) continue;
                require(n > 0, "cannot write Z-Image GPU suffix temporary file (check free disk space)");
                done += uint64_t(n);
            }
            metrics_.request_pack_write_bytes += done;
        }
        r.offset = packed_offset;
        r.shape[1] -= prefix_channels;
        r.bytes = uint64_t(r.shape[0]) * suffix_bytes;
        r.packed = true;
        packed_offset += r.bytes;
    }
    check_source();
    metrics_.mlp_prefix_channels = prefix_channels;
    metrics_.suffix_pack_bytes = packed_offset;
    metrics_.request_pack_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    event("pack_z_image_suffix", int(groups.size()), int(groups.size()));
}

void ZImageWeightStream::prepare_convrot(std::vector<Record> &records) {
    std::vector<Record> converted;
    auto find = [&](const std::string &name) -> const Record & {
        auto it = std::find_if(records.begin(), records.end(), [&](const Record &r) { return r.name == name; });
        require(it != records.end(), "missing ConvRot companion tensor: " + name);
        return *it;
    };
    for (auto r : records) {
        r.source_bytes = r.bytes;
        if (r.name.ends_with(".weight_scale")) {
            require(find(r.name.substr(0, r.name.size() - 13) + ".weight").dtype == mx::int8,
                    "ConvRot row scale has no signed INT8 weight");
            continue; // replaced by affine scales + biases
        }
        if (r.dtype == mx::int8) {
            require(r.name.ends_with(".weight") && r.shape.size() == 2 && r.shape[1] % 256 == 0,
                    "invalid streamed ConvRot Q8 geometry");
            const auto prefix = r.name.substr(0, r.name.size() - 7);
            const auto &metadata = find(prefix + ".comfy_quant");
            const auto &source_scale = find(prefix + ".weight_scale");
            require(metadata.dtype == mx::uint8 && source_scale.dtype == mx::float32 &&
                        source_scale.shape == mx::Shape{r.shape[0], 1},
                    "invalid streamed ConvRot scale geometry");
            Record scale = source_scale;
            scale.name = prefix + ".scales";
            scale.source_bytes = scale.bytes;
            scale.shape[1] = r.shape[1] / 32;
            scale.dtype = std::getenv("TURBOCIDER_Z_CONVROT_FP32_SCALES") ? mx::float32 : mx::bfloat16;
            scale.bytes = uint64_t(scale.shape[0]) * scale.shape[1] * scale.dtype.size();
            scale.conversion = Conversion::scales;
            converted.push_back(scale);
            scale.name = prefix + ".biases";
            scale.conversion = Conversion::biases;
            converted.push_back(std::move(scale));
            r.shape[1] /= 4;
            r.dtype = mx::uint32;
            r.conversion = Conversion::signed_q8;
        } else if (r.dtype == mx::float32) {
            r.dtype = mx::bfloat16;
            r.bytes /= 2;
            r.conversion = Conversion::bf16;
        } else if (r.dtype == mx::uint8) {
            require(r.name.ends_with(".comfy_quant"), "unexpected streamed ConvRot byte tensor");
            require(find(r.name.substr(0, r.name.size() - 12) + ".weight").dtype == mx::int8,
                    "ConvRot metadata has no signed INT8 weight");
        }
        converted.push_back(std::move(r));
    }
    std::sort(converted.begin(), converted.end(), [](const Record &a, const Record &b) { return a.name < b.name; });
    records = std::move(converted);
}

void ZImageWeightStream::allocate(Slot &slot, const std::vector<Record> &records) {
    if (convrot_) slot.scratch.resize(16384);
    for (const auto &r : records) {
        // Allocate MLX-owned shared buffers once; pread overwrites only slots
        // whose preceding GPU use has completed. No per-refill tensor copies.
        slot.arrays.emplace_back(mx::allocator::malloc(r.bytes), r.shape, r.dtype);
        slot.pointers.push_back(slot.arrays.back().data<char>());
    }
}
ZImageWeightStream::ReadResult ZImageWeightStream::fill(Slot &slot, const std::vector<Record> &records) const {
    const auto start = Clock::now();
    ReadResult result;
    for (size_t i = 0; i < records.size(); ++i) {
        const auto &r = records[i];
        auto *destination = slot.pointers[i];
        const int fd = r.packed ? packed_fd_ : fd_;
        if (r.conversion == Conversion::none || r.conversion == Conversion::signed_q8) {
            result.bytes += read({{r.offset, r.bytes, destination, fd}}).bytes;
            if (r.conversion == Conversion::signed_q8) {
                // q + 128, without dequantizing or allocating a tensor.
                for (uint64_t offset = 0; offset < r.bytes; offset += 4) {
                    uint32_t value;
                    std::memcpy(&value, destination + offset, 4);
                    value ^= 0x80808080u;
                    std::memcpy(destination + offset, &value, 4);
                }
            }
            continue;
        }
        const bool row_scale = r.conversion == Conversion::scales || r.conversion == Conversion::biases;
        const uint64_t repeat = row_scale ? r.shape[1] : 1;
        for (uint64_t offset = 0; offset < r.source_bytes / 4; offset += slot.scratch.size()) {
            const auto count = std::min<uint64_t>(slot.scratch.size(), r.source_bytes / 4 - offset);
            result.bytes += read({{r.offset + offset * 4, count * 4,
                                   reinterpret_cast<char *>(slot.scratch.data()), fd}}).bytes;
            for (uint64_t j = 0; j < count; ++j) {
                const float value = slot.scratch[j];
                if (r.dtype == mx::bfloat16) {
                    auto converted = mx::bfloat16_t(value);
                    if (r.conversion == Conversion::biases)
                        converted = mx::bfloat16_t(float(converted) * -128.f);
                    std::fill_n(reinterpret_cast<mx::bfloat16_t *>(destination) + (offset + j) * repeat,
                                repeat, converted);
                } else {
                    std::fill_n(reinterpret_cast<float *>(destination) + (offset + j) * repeat,
                                repeat, r.conversion == Conversion::biases ? value * -128.f : value);
                }
            }
        }
    }
    result.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}
void ZImageWeightStream::record(ReadResult result) {
    metrics_.request_bytes_loaded += result.bytes;
    metrics_.request_load_seconds += result.seconds;
}
Weights ZImageWeightStream::bind(const Slot &slot, int block) const {
    std::vector<std::string> names;
    for (const auto &r : blocks_[block]) names.push_back(r.name);
    Weights weights;
    weights.bind_arrays(names, slot.arrays);
    return weights;
}

ZImageWeightStream::ZImageWeightStream(const std::filesystem::path &path, uint64_t budget,
                                       uint64_t activation_reserve, Weights &fixed,
                                       const Event &event, std::atomic<bool> &cancelled,
                                       int prefix_channels, unsigned prefetch_layers)
    : prefetch_layers_(prefetch_layers), cancelled_(cancelled) {
    try {
        require(prefetch_layers >= 1 && prefetch_layers <= 8, "Z-Image prefetch must be 1 through 8 layers");
        const auto &optimizations = device_info().optimizations();
        require(prefetch_layers == 1 || optimizations.z_image_suffix_streaming,
                "Z-Image multi-layer prefetch is only enabled for Apple M5 Pro 24 GiB");
        require(prefix_channels == 0 || optimizations.z_image_suffix_streaming,
                "Z-Image suffix streaming is only enabled for Apple M5 Pro 24 GiB");
        index(path);
        require(optimizations.supports_z_image_streaming(convrot_),
                "Z-Image INT8 streaming is only enabled for Apple M5 Pro 24 GiB; use resident weights on this device");
        require(prefix_channels >= 0, "invalid Z-Image ANE prefix width");
        if (prefix_channels) pack_suffix(prefix_channels, event);
        if (convrot_) {
            prepare_convrot(fixed_records_);
            for (auto &records : blocks_) prepare_convrot(records);
        }
        uint64_t block_bytes = 0, fixed_bytes = 0;
        for (const auto &r : blocks_[0]) block_bytes += r.bytes;
        for (const auto &r : fixed_records_) fixed_bytes += r.bytes;
        const auto scratch_bytes = convrot_ ? uint64_t(prefetch_layers + 1) * 65536 : 0;
        const auto reserved = activation_reserve + fixed_bytes + scratch_bytes;
        const auto capacity = budget > reserved ? (budget - reserved) / block_bytes : 0;
        const auto slots = budget ? std::min<uint64_t>(prefetch_layers + 1, std::max<uint64_t>(2, capacity))
                                  : prefetch_layers + 1;
        auto plan = make_block_residency_plan(budget, reserved, block_bytes, 30, 0,
                                               unsigned(slots), false, convrot_);
        slots_.resize(plan.refill_slots);
        if (!slots_.empty()) prefetch_layers_ = std::min<unsigned>(prefetch_layers_, slots_.size() - 1);
        metrics_.enabled = true;
        metrics_.active_blocks = 30;
        metrics_.pinned_blocks = plan.pinned_blocks;
        metrics_.streamed_blocks = plan.streamed_blocks;
        metrics_.refill_slots = plan.refill_slots;
        metrics_.fully_resident = plan.fully_resident;
        metrics_.quantized = convrot_;
        metrics_.memory_budget_bytes = budget;
        metrics_.activation_reserve_bytes = reserved;
        metrics_.block_bytes = block_bytes;
        metrics_.estimated_working_set_bytes = plan.estimated_working_set_bytes;
        Slot shared;
        allocate(shared, fixed_records_);
        record(fill(shared, fixed_records_));
        std::vector<std::string> names;
        for (const auto &r : fixed_records_) names.push_back(r.name);
        fixed.bind_arrays(names, shared.arrays);
        for (unsigned i = 0; i < plan.pinned_blocks; ++i) {
            event("load_z_image_stream", int(i), int(plan.pinned_blocks));
            Slot slot;
            allocate(slot, blocks_[i]);
            record(fill(slot, blocks_[i]));
            pinned_.push_back(bind(slot, int(i)));
        }
        for (auto &slot : slots_) allocate(slot, blocks_[0]);
        metrics_.request_slot_allocations = slots_.size();
        event("load_z_image_stream", int(plan.pinned_blocks), int(plan.pinned_blocks));
    } catch (...) {
        if (packed_fd_ >= 0) ::close(packed_fd_);
        packed_fd_ = -1;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        fixed.clear();
        throw;
    }
}
ZImageWeightStream::~ZImageWeightStream() {
    for (auto &slot : slots_)
        if (slot.pending.valid()) slot.pending.wait();
    if (packed_fd_ >= 0) ::close(packed_fd_);
    if (fd_ >= 0) ::close(fd_);
}
void ZImageWeightStream::reset_metrics() {
    metrics_.request_bytes_loaded = metrics_.request_slot_allocations = metrics_.request_slot_refills = 0;
    metrics_.request_load_seconds = metrics_.request_wait_seconds = 0;
    metrics_.request_pack_read_bytes = metrics_.request_pack_write_bytes = 0;
    metrics_.request_pack_seconds = 0;
}
void ZImageWeightStream::prefetch(int block) {
    auto &slot = slots_[(block - metrics_.pinned_blocks) % slots_.size()];
    require(!slot.pending.valid(), "Z-Image stream slot still has a pending read");
    slot.block = block;
    slot.pending = std::async(std::launch::async, [this, &slot, block] {
        return fill(slot, blocks_[block]);
    });
    ++metrics_.request_slot_refills;
}
void ZImageWeightStream::check_source() const {
    struct stat status{};
    require(::fstat(fd_, &status) == 0 && uint64_t(status.st_size) == file_bytes_ &&
                status.st_mtimespec.tv_sec == modified_seconds_ && status.st_mtimespec.tv_nsec == modified_nanos_,
            "Z-Image streamed checkpoint changed during the session");
}
void ZImageWeightStream::begin_pass() {
    check_source();
    expected_block_ = 0;
    for (unsigned i = 0; i < prefetch_layers_ && metrics_.pinned_blocks + i < 30; ++i)
        prefetch(int(metrics_.pinned_blocks + i));
}
Weights ZImageWeightStream::acquire(int block) {
    checkpoint(cancelled_);
    require(block == expected_block_++ && block < 30, "Z-Image stream blocks must execute in order");
    if (unsigned(block) < metrics_.pinned_blocks) return pinned_[block];
    auto &slot = slots_[(block - metrics_.pinned_blocks) % slots_.size()];
    require(slot.block == block && slot.pending.valid(), "missing Z-Image prefetched layer");
    auto start = Clock::now();
    auto loaded = slot.pending.get();
    metrics_.request_wait_seconds += std::chrono::duration<double>(Clock::now() - start).count();
    record(loaded);
    if (block + int(prefetch_layers_) < 30) prefetch(block + int(prefetch_layers_));
    return bind(slot, block);
}

} // namespace tc
