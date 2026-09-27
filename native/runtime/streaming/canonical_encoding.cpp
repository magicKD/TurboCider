#include "canonical_encoding.hpp"

#include "../memory_manifest.hpp"

namespace tc::streaming {

CanonicalEncoder::CanonicalEncoder(std::string_view schema) {
    tag('H');
    append_string(schema);
}

void CanonicalEncoder::tag(char value) {
    bytes_.push_back(value);
}

void CanonicalEncoder::append_u64(uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
        bytes_.push_back(static_cast<char>((value >> shift) & 0xff));
}

void CanonicalEncoder::append_string(std::string_view value) {
    append_u64(value.size());
    bytes_.append(value.data(), value.size());
}

void CanonicalEncoder::field_name(std::string_view value) {
    append_string(value);
}

void CanonicalEncoder::string_field(
        std::string_view name, std::string_view value) {
    tag('S');
    field_name(name);
    append_string(value);
}

void CanonicalEncoder::optional_string_field(
        std::string_view name, const std::optional<std::string> &value) {
    tag('O');
    field_name(name);
    tag(value ? '1' : '0');
    if (value) append_string(*value);
}

void CanonicalEncoder::unsigned_field(
        std::string_view name, uint64_t value) {
    tag('U');
    field_name(name);
    append_u64(value);
}

void CanonicalEncoder::boolean_field(
        std::string_view name, bool value) {
    tag('B');
    field_name(name);
    tag(value ? '1' : '0');
}

void CanonicalEncoder::begin_list(
        std::string_view name, uint64_t count) {
    tag('L');
    field_name(name);
    append_u64(count);
}

std::string CanonicalEncoder::sha256() const {
    return memory_sha256_hex(bytes_);
}

} // namespace tc::streaming
