#pragma once
#include <string_view>

namespace tc {
// Input must already be valid JSON. Checks decoded object keys, including
// escaped spellings, before a dictionary can hide duplicate configuration.
void reject_duplicate_json_keys(std::string_view);
}
