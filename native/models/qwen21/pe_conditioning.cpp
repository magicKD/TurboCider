#include "pe_conditioning.hpp"
#include <algorithm>

namespace tc::qwen21::pe {
namespace {
constexpr int image_pad = 248056, video_pad = 248057;
constexpr int vision_start = 248053, vision_end = 248054;
}
MultimodalLayout image_layout(const Tokens &tokens, const std::vector<ImageGrid> &grids,
                              int max_tokens) {
    require(tokens.valid > 0 && size_t(tokens.valid) == tokens.ids.size(),
            "PE multimodal tokens must be nonempty and unpadded");
    require(grids.size() <= 10 && max_tokens > 0 && max_tokens <= 32768,
            "invalid PE reference/token limit");
    int64_t count = tokens.valid;
    for (const auto &grid : grids) {
        require(grid.height > 0 && grid.width > 0 && grid.height % 2 == 0 && grid.width % 2 == 0,
                "invalid PE image grid");
        count += int64_t(grid.height / 2) * (grid.width / 2) - 1;
        require(count <= max_tokens, "PE expanded prompt exceeds token limit");
    }
    require(count <= max_tokens && std::count(tokens.ids.begin(), tokens.ids.end(), image_pad) == int(grids.size()),
            "PE image placeholders/reference count mismatch or token limit exceeded");
    MultimodalLayout out;
    out.ids.reserve(size_t(count)); out.token_types.reserve(size_t(count));
    for (auto &axis : out.positions) axis.reserve(size_t(count));
    size_t image = 0;
    for (int i = 0; i < tokens.valid; ++i) {
        const int id = tokens.ids[i];
        require(id >= 0 && id != video_pad, "invalid PE token or unsupported video input");
        if (id == vision_start)
            require(i + 2 < tokens.valid && tokens.ids[i + 1] == image_pad && tokens.ids[i + 2] == vision_end,
                    "malformed PE vision delimiters");
        if (id == vision_end)
            require(i >= 2 && tokens.ids[i - 1] == image_pad && tokens.ids[i - 2] == vision_start,
                    "malformed PE vision delimiters");
        if (id != image_pad) {
            out.ids.push_back(id); out.token_types.push_back(0);
            for (auto &axis : out.positions) axis.push_back(out.next_position);
            ++out.next_position;
            continue;
        }
        require(i > 0 && i + 1 < tokens.valid && tokens.ids[i - 1] == vision_start && tokens.ids[i + 1] == vision_end,
                "PE image placeholder requires vision delimiters");
        const auto grid = grids.at(image++);
        const int h = grid.height / 2, w = grid.width / 2;
        out.images.push_back({int(out.ids.size()), h * w});
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            out.ids.push_back(image_pad); out.token_types.push_back(1);
            out.positions[0].push_back(out.next_position);
            out.positions[1].push_back(out.next_position + y);
            out.positions[2].push_back(out.next_position + x);
        }
        out.next_position += std::max(h, w);
    }
    return out;
}
Tensor MultimodalLayout::position_tensor() const {
    std::vector<int> flat;
    for (const auto &axis : positions) {
        require(axis.size() == ids.size(), "invalid PE position layout");
        flat.insert(flat.end(), axis.begin(), axis.end());
    }
    require(!ids.empty(), "empty PE position layout");
    return Tensor(flat.data(), {3, int(ids.size())}, mx::int32);
}
Tensor image_embeddings(const MultimodalLayout &layout, const Tensor &table,
                        const std::vector<Tensor> &features) {
    require(table.ndim() == 2 && table.shape(1) > 0 && !layout.ids.empty() &&
            features.size() == layout.images.size(), "invalid PE embedding inputs");
    for (int id : layout.ids)
        require(id >= 0 && id < table.shape(0), "PE token out of embedding vocabulary");
    std::vector<Tensor> chunks;
    int cursor = 0;
    auto text = [&](int end) {
        if (end > cursor) chunks.push_back(mx::take(table,
            Tensor(layout.ids.data() + cursor, {end - cursor}, mx::int32), 0));
    };
    for (size_t i = 0; i < features.size(); ++i) {
        auto span = layout.images[i];
        require(span.begin >= cursor && span.count > 0 && int64_t(span.begin) + span.count <= int64_t(layout.ids.size()),
                "invalid PE image span");
        require(features[i].shape() == mx::Shape{span.count, table.shape(1)}, "PE feature/grid shape mismatch");
        text(span.begin);
        chunks.push_back(mx::astype(features[i], table.dtype()));
        cursor = span.begin + span.count;
    }
    text(int(layout.ids.size()));
    return mx::expand_dims(mx::concatenate(chunks, 0), 0);
}
}
