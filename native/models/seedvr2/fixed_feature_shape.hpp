#pragma once
#include <cstddef>
#include <utility>
#include <vector>

namespace tc::seedvr2 {
// Core ML's "unspecified" means unconstrained, not fixed. Accept exactly one
// enumerated shape or singleton ranges, never a merely matching default shape.
inline bool exact_fixed_feature_shape(int kind, const std::vector<size_t> &expected,
    const std::vector<std::vector<size_t>> &enumerated,
    const std::vector<std::pair<size_t,size_t>> &ranges) {
    if (expected.size() != 4) return false;
    for (auto value : expected) if (!value) return false;
    if (kind == 2) return enumerated.size() == 1 && enumerated.front() == expected;
    if (kind != 3 || ranges.size() != expected.size()) return false;
    for (size_t index = 0; index < expected.size(); ++index)
        if (ranges[index].first != expected[index] || ranges[index].second != 1) return false;
    return true;
}
} // namespace tc::seedvr2
