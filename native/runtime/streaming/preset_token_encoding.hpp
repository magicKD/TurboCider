#pragma once

#include "canonical_encoding.hpp"
#include "preset_catalog.hpp"

#include <string_view>
#include <vector>

namespace tc::streaming {

// The record and workload identity have distinct list names and share the
// same per-token field order in their published digest schemas.
static inline void encode_preset_token_shapes(
        CanonicalEncoder &out, std::string_view list_name,
        const std::vector<PresetTokenShape> &shapes) {
    out.begin_list(list_name, shapes.size());
    for (const auto &token : shapes) {
        out.string_field("token.encoder", token.encoder);
        out.string_field("token.tokenizer_revision", token.tokenizer_revision);
        out.string_field("token.template_revision", token.template_revision);
        out.unsigned_field("token.valid_rows", token.valid_rows);
        out.unsigned_field("token.padded_rows", token.padded_rows);
        out.unsigned_field("token.compute_rows", token.compute_rows);
    }
}

} // namespace tc::streaming
