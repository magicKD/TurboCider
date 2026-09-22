#include "sequence.hpp"
#include "../../core/common.hpp"
#include <algorithm>
#include <limits>

namespace tc::qwen21 {
SequenceGeometry make_sequence_geometry(int text_length, int height, int width,
                                       const std::vector<ReferenceGeometry> &references) {
    require(text_length > 0 && height > 0 && width > 0 && references.size() <= 10,
            "invalid Qwen21 sequence geometry/reference count");
    int previous = 0;
    int64_t tokens = int64_t(height) * width + text_length;
    for (const auto &r : references) {
        require(r.height > 0 && r.width > 0 && r.text_slot >= previous && r.text_slot <= text_length,
                "Qwen21 image slots must be ordered text boundaries with positive dimensions");
        previous = r.text_slot;
        tokens += int64_t(r.height) * r.width;
    }
    require(tokens <= std::numeric_limits<int>::max(), "Qwen21 sequence exceeds index range");
    SequenceGeometry result;
    result.positions.reserve(size_t(tokens));
    int cursor = 0;
    float position = 0.f;
    for (size_t i = 0; i <= references.size(); ++i) {
        int end_text = i < references.size() ? references[i].text_slot : text_length;
        int start = int(result.positions.size());
        if (end_text > cursor) {
            result.segments.push_back({start, start + end_text - cursor, true, cursor, -1});
            for (int t = cursor; t < end_text; ++t) {
                result.positions.push_back({position, position, position});
                position += 1.f;
            }
        }
        cursor = end_text;
        int h = i < references.size() ? references[i].height : height;
        int w = i < references.size() ? references[i].width : width;
        start = int(result.positions.size());
        result.segments.push_back({start, start + h * w, false, -1, int(i)});
        if (i == references.size()) result.prefix_length = start;
        for (int row = 0; row < h; ++row)
            for (int col = 0; col < w; ++col)
                result.positions.push_back({position,
                    float(row - (h - h / 2)) + .5f * (h % 2 - height % 2),
                    float(col - (w - w / 2)) + .5f * (w % 2 - width % 2)});
        position += float(std::max(h, w));
    }
    return result;
}
} // namespace tc::qwen21
