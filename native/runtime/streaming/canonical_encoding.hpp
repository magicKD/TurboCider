#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace tc::streaming {

// A compact deterministic encoder for release identities.  It deliberately
// does not depend on JSON, locales, Foundation, or platform number bridging.
// Every value carries an explicit type tag and every string is length-prefixed.
class CanonicalEncoder {
  public:
    explicit CanonicalEncoder(std::string_view schema);

    void string_field(std::string_view name, std::string_view value);
    void optional_string_field(
        std::string_view name, const std::optional<std::string> &value);
    void unsigned_field(std::string_view name, uint64_t value);
    void boolean_field(std::string_view name, bool value);
    void begin_list(std::string_view name, uint64_t count);

    const std::string &bytes() const noexcept { return bytes_; }
    std::string sha256() const;

  private:
    void tag(char value);
    void append_u64(uint64_t value);
    void append_string(std::string_view value);
    void field_name(std::string_view value);

    std::string bytes_;
};

} // namespace tc::streaming
