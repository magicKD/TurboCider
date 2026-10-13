#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tc::seedvr2 {
// Exact row partition only: no sampling, context truncation or dropped tail.
// Bounding groups preserves the qualified one-group path for t25 while
// avoiding unbounded caller-owned Core ML backing storage on longer videos.
inline std::vector<std::pair<int,int>> prefix_batch_ranges(int total, int rows, int maximum_chunks) {
    if (total <= 0 || rows < 128 || rows > 1024 || rows % 128 ||
            maximum_chunks <= 0 || maximum_chunks > 64)
        throw std::invalid_argument("invalid SeedVR2 prefix batch geometry");
    const int64_t capacity = int64_t(rows) * maximum_chunks;
    std::vector<std::pair<int,int>> result;
    for (int64_t first = 0; first < total; first += capacity)
        result.emplace_back(int(first),int(std::min(int64_t(total),first+capacity)));
    return result;
}
} // namespace tc::seedvr2
