#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
#include <unistd.h>

namespace tc::gguf {

struct TypeInfo { uint32_t id, elements, bytes; const char *name; };
// On-disk GGML IDs; file-level Q4_K_M etc. are not tensor types.
inline constexpr std::array<TypeInfo, 11> types{{
    {0, 1, 4, "F32"}, {1, 1, 2, "F16"}, {2, 32, 18, "Q4_0"},
    {3, 32, 20, "Q4_1"}, {6, 32, 22, "Q5_0"}, {7, 32, 24, "Q5_1"},
    {8, 32, 34, "Q8_0"}, {12, 256, 144, "Q4_K"},
    {13, 256, 176, "Q5_K"}, {14, 256, 210, "Q6_K"}, {30, 1, 2, "BF16"}
}};
inline const TypeInfo &type_info(uint32_t id) {
    for (const auto &type : types) if (type.id == id) return type;
    throw std::invalid_argument("gguf_type_unsupported: " + std::to_string(id));
}
inline bool native_mlx_type(uint32_t id) {
    return id == 0 || id == 1 || id == 2 || id == 3 || id == 8 || id == 30;
}
inline void check(bool valid, std::string_view reason) {
    if (!valid) throw std::invalid_argument("gguf_invalid: " + std::string(reason));
}
inline uint64_t checked_add(uint64_t a, uint64_t b) {
    check(b <= UINT64_MAX - a, "integer addition overflow"); return a + b;
}
inline uint64_t checked_mul(uint64_t a, uint64_t b) {
    check(!a || b <= UINT64_MAX / a, "integer multiplication overflow"); return a * b;
}

struct ParseLimits {
    uint64_t tensors = 65536, metadata_entries = 65536;
    uint64_t directory_bytes = 64ull << 20, retained_bytes = 32ull << 20;
    uint64_t string_bytes = 1ull << 20, array_elements = 1ull << 20;
    uint32_t max_rank = 4, max_array_depth = 4;
};
struct Metadata {
    std::string key, text;
    uint32_t type = 0, element_type = 0;
    uint64_t count = 0, unsigned_value = 0;
    int64_t signed_value = 0;
    double float_value = 0;
    // Checked payload span for component metadata readers (e.g. vocabulary),
    // including array element-type/count headers. Does not retain array data.
    uint64_t value_offset = 0, value_bytes = 0;
};
struct TensorDescriptor {
    std::string name;
    uint32_t type = 0;
    // GGML ne[0] is contiguous. Reverse shape descriptions, not payload bytes.
    std::vector<uint64_t> dimensions;
    uint64_t relative_offset = 0, file_offset = 0, bytes = 0, elements = 0;
    uint64_t columns() const { return dimensions.at(0); }
    uint64_t rows() const { return elements / columns(); }
    std::vector<uint64_t> logical_shape() const {
        return {dimensions.rbegin(), dimensions.rend()};
    }
};
struct Directory {
    uint32_t version = 0, alignment = 32;
    uint64_t file_bytes = 0, directory_end = 0, data_offset = 0, retained_upper_bytes = 0;
    std::vector<Metadata> metadata;
    std::vector<TensorDescriptor> tensors;
    const TensorDescriptor &tensor(std::string_view name) const {
        for (const auto &item : tensors) if (item.name == name) return item;
        throw std::invalid_argument("gguf_tensor_missing: " + std::string(name));
    }
    const Metadata *meta(std::string_view key) const {
        for (const auto &item : metadata) if (item.key == key) return &item;
        return nullptr;
    }
};

namespace detail {
class Reader {
  public:
    Reader(int fd, uint64_t bytes, const ParseLimits &limits)
        : fd_(fd), bytes_(bytes), limits_(limits) {
        check(fd >= 0 && bytes <= uint64_t(std::numeric_limits<off_t>::max()), "invalid fd/size");
    }
    uint64_t cursor() const { return cursor_; }
    uint64_t retained() const { return retained_; }
    void charge(uint64_t bytes) {
        retained_ = checked_add(retained_, bytes);
        check(retained_ <= limits_.retained_bytes, "retained metadata limit");
    }
    void skip(uint64_t bytes) {
        check(cursor_ <= bytes_ && bytes <= bytes_ - cursor_, "truncated directory");
        check(cursor_ <= limits_.directory_bytes && bytes <= limits_.directory_bytes - cursor_,
              "directory byte limit");
        cursor_ += bytes;
    }
    void read(void *destination, uint64_t bytes) {
        const uint64_t start = cursor_; skip(bytes);
        auto *output = static_cast<uint8_t *>(destination);
        uint64_t done = 0;
        while (done != bytes) {
            const auto amount = std::min<uint64_t>(bytes - done, 1ull << 20);
            const ssize_t n = ::pread(fd_, output + done, size_t(amount), off_t(start + done));
            if (n < 0 && errno == EINTR) continue;
            check(n > 0, "directory read failed or source changed");
            done += uint64_t(n);
        }
    }
    uint64_t integer(uint32_t bytes) {
        std::array<uint8_t, 8> data{};
        check(bytes && bytes <= data.size(), "invalid scalar width"); read(data.data(), bytes);
        uint64_t value = 0;
        for (uint32_t i = 0; i < bytes; ++i) value |= uint64_t(data[i]) << (8 * i);
        return value;
    }
    std::string string(bool retain = true) {
        const uint64_t bytes = integer(8);
        check(bytes <= limits_.string_bytes, "string byte limit");
        if (!retain) { skip(bytes); return {}; }
        charge(checked_add(bytes, 1));
        check(bytes <= bytes_ - cursor_ && bytes <= limits_.directory_bytes - cursor_, "truncated string");
        std::string result(size_t(bytes), '\0'); read(result.data(), bytes); return result;
    }
    void value(uint32_t type, Metadata *entry, uint32_t depth = 0) {
        if (type == 8) {
            auto text = string(entry != nullptr);
            if (entry) entry->text = std::move(text);
            return;
        }
        if (type == 9) {
            check(depth < limits_.max_array_depth, "nested array depth limit");
            const uint32_t element = uint32_t(integer(4)); const uint64_t count = integer(8);
            check(count <= limits_.array_elements, "array element limit");
            check(element <= 12, "invalid array element type");
            if (entry) { entry->element_type = element; entry->count = count; }
            if (element != 8 && element != 9 && element != 7) skip(checked_mul(count, scalar_width(element)));
            else for (uint64_t i = 0; i < count; ++i) value(element, nullptr, depth + 1);
            return;
        }
        const uint32_t width = scalar_width(type); const uint64_t bits = integer(width);
        if (type == 7) check(bits <= 1, "invalid boolean");
        if (!entry) return;
        entry->unsigned_value = bits;
        if (type == 1 || type == 3 || type == 5 || type == 11) {
            const uint64_t sign = uint64_t(1) << (width * 8 - 1);
            const uint64_t extended = width == 8 || !(bits & sign)
                ? bits : bits | (UINT64_MAX << (width * 8));
            entry->signed_value = std::bit_cast<int64_t>(extended);
        }
        if (type == 6) entry->float_value = std::bit_cast<float>(uint32_t(bits));
        if (type == 12) entry->float_value = std::bit_cast<double>(bits);
    }
  private:
    static uint32_t scalar_width(uint32_t type) {
        switch (type) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: throw std::invalid_argument("gguf_invalid: invalid metadata scalar type");
        }
    }
    int fd_;
    uint64_t bytes_, cursor_ = 0, retained_ = 0;
    const ParseLimits &limits_;
};
} // namespace detail

// The caller owns a held/lease fd. No path reopen, payload allocation or GPU work.
inline Directory read_directory(int fd, uint64_t file_bytes,
                                const ParseLimits &limits = {}, bool mlx_only = false) {
    detail::Reader reader(fd, file_bytes, limits);
    check(limits.max_rank <= 8 && limits.max_array_depth <= 8, "unsafe parser limits");
    check(reader.integer(4) == 0x46554747, "invalid GGUF magic/endian");
    Directory directory; directory.file_bytes = file_bytes;
    directory.version = uint32_t(reader.integer(4));
    check(directory.version == 2 || directory.version == 3, "unsupported GGUF version");
    const uint64_t tensor_count = reader.integer(8), metadata_count = reader.integer(8);
    check(tensor_count && tensor_count <= limits.tensors && tensor_count <= file_bytes / 24,
          "tensor count limit");
    check(metadata_count <= limits.metadata_entries && metadata_count <= file_bytes / 12,
          "metadata count limit");
    // Conservative vector/index/node capacity, not only string payload.
    reader.charge(checked_mul(tensor_count, sizeof(TensorDescriptor) + 256));
    reader.charge(checked_mul(metadata_count, sizeof(Metadata) + 128));
    directory.metadata.reserve(size_t(metadata_count)); directory.tensors.reserve(size_t(tensor_count));
    std::unordered_set<std::string> keys;
    for (uint64_t i = 0; i < metadata_count; ++i) {
        Metadata metadata; metadata.key = reader.string();
        check(!metadata.key.empty() && metadata.key.find('\0') == std::string::npos, "invalid metadata key");
        reader.charge(metadata.key.size() + 1);
        check(keys.insert(metadata.key).second, "duplicate metadata key");
        metadata.type = uint32_t(reader.integer(4)); metadata.value_offset = reader.cursor();
        reader.value(metadata.type, &metadata);
        metadata.value_bytes = reader.cursor() - metadata.value_offset;
        directory.metadata.push_back(std::move(metadata));
    }
    if (const auto *alignment = directory.meta("general.alignment")) {
        check(alignment->type == 4, "alignment must be uint32");
        const uint64_t value = alignment->unsigned_value;
        check(value && value <= (1u << 20) && !(value & (value - 1)), "invalid alignment");
        directory.alignment = uint32_t(value);
    }
    keys.clear();
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorDescriptor tensor; tensor.name = reader.string();
        check(!tensor.name.empty() && tensor.name.find('\0') == std::string::npos, "invalid tensor name");
        reader.charge(tensor.name.size() + 1);
        check(keys.insert(tensor.name).second, "duplicate tensor name");
        const uint32_t rank = uint32_t(reader.integer(4));
        check(rank && rank <= limits.max_rank, "invalid tensor rank"); tensor.elements = 1;
        for (uint32_t j = 0; j < rank; ++j) {
            const uint64_t dimension = reader.integer(8);
            check(dimension && dimension <= INT32_MAX, "invalid tensor dimension");
            tensor.elements = checked_mul(tensor.elements, dimension); tensor.dimensions.push_back(dimension);
        }
        tensor.type = uint32_t(reader.integer(4));
        if (mlx_only && !native_mlx_type(tensor.type))
            throw std::invalid_argument("unsupported native MLX GGUF tensor type " +
                std::to_string(tensor.type) + "; supported: F32, F16, BF16, Q4_0, Q4_1, Q8_0; no fallback backend");
        const auto &type = type_info(tensor.type);
        check(tensor.columns() % type.elements == 0, "continuous dimension is not block aligned");
        tensor.bytes = checked_mul(tensor.elements / type.elements, type.bytes);
        tensor.relative_offset = reader.integer(8);
        check(tensor.relative_offset % directory.alignment == 0, "unaligned tensor offset");
        directory.tensors.push_back(std::move(tensor));
    }
    directory.directory_end = reader.cursor(); const uint64_t mask = directory.alignment - 1;
    directory.data_offset = checked_add(directory.directory_end, mask) & ~mask;
    check(directory.data_offset <= file_bytes, "missing aligned tensor data");
    std::vector<std::pair<uint64_t, uint64_t>> ranges; ranges.reserve(directory.tensors.size());
    for (auto &tensor : directory.tensors) {
        tensor.file_offset = checked_add(directory.data_offset, tensor.relative_offset);
        check(tensor.file_offset <= file_bytes && tensor.bytes <= file_bytes - tensor.file_offset,
              "tensor payload out of file bounds");
        ranges.emplace_back(tensor.file_offset, checked_add(tensor.file_offset, tensor.bytes));
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i)
        check(ranges[i].first >= ranges[i - 1].second, "overlapping tensor payloads");
    directory.retained_upper_bytes = reader.retained(); return directory;
}

inline void validate_split_set(const std::vector<Directory> &shards, const ParseLimits &limits = {}) {
    check(!shards.empty() && shards.size() <= 64, "empty/excessive split set");
    uint64_t tensor_upper = 0, metadata_upper = 0;
    for (const auto &shard : shards) {
        tensor_upper = checked_add(tensor_upper, shard.tensors.size());
        metadata_upper = checked_add(metadata_upper, shard.retained_upper_bytes);
    }
    check(tensor_upper <= limits.tensors && metadata_upper <= limits.retained_bytes,
          "aggregate split metadata limit");
    std::unordered_set<uint64_t> numbers; std::unordered_set<std::string> names;
    std::optional<std::string> architecture; uint64_t count = 0;
    for (const auto &shard : shards) {
        const auto *number = shard.meta("split.no"), *total = shard.meta("split.count");
        const auto *tensors = shard.meta("split.tensors.count");
        if (shards.size() == 1 && !number && !total && !tensors) return;
        check(number && total && tensors && number->type == 2 && total->type == 2 &&
                  (tensors->type == 5 || tensors->type == 10), "invalid/missing split metadata");
        check(total->unsigned_value == shards.size() && number->unsigned_value < shards.size(),
              "missing split shard or invalid split number");
        check(numbers.insert(number->unsigned_value).second, "duplicate split number");
        check(tensors->type != 5 || tensors->signed_value >= 0, "negative split tensor count");
        if (!count) count = tensors->unsigned_value;
        check(count && tensors->unsigned_value == count, "inconsistent split tensor count");
        const auto *arch = shard.meta("general.architecture");
        check(arch && arch->type == 8 && !arch->text.empty(), "missing split architecture");
        if (!architecture) architecture = arch->text;
        check(*architecture == arch->text, "inconsistent split architecture");
        for (const auto &tensor : shard.tensors)
            check(names.insert(tensor.name).second, "duplicate split tensor");
    }
    check(names.size() == count, "split tensor count mismatch");
}

} // namespace tc::gguf
