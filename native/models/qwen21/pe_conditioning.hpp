#pragma once
#include "../../core/tokenizer.hpp"
#include "../../backends/mlx.hpp"
#include <array>

namespace tc::qwen21::pe {
struct ImageGrid { int height, width; }; // Patch grid before 2x2 spatial merge.
struct ImageSpan { int begin, count; };
struct MultimodalLayout {
    std::vector<int> ids, token_types;
    std::array<std::vector<int>, 3> positions;
    std::vector<ImageSpan> images;
    int next_position = 0;
    int rope_delta() const { return next_position - int(ids.size()); }
    Tensor position_tensor() const;
};

// Expands each PE image_pad placeholder and retains the complete chat, unlike
// diffusion conditioning which drops the system turn. Still images only.
MultimodalLayout image_layout(const Tokens &, const std::vector<ImageGrid> &,
                              int max_tokens = 32768);
Tensor image_embeddings(const MultimodalLayout &, const Tensor &table,
                        const std::vector<Tensor> &merged_features);
}
