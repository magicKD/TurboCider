#pragma once
#include <array>
#include <vector>

namespace tc::qwen21 {
struct ReferenceGeometry {
    int height, width, text_slot;
    bool operator==(const ReferenceGeometry &) const = default;
};
struct SequenceSegment {
    int start, end;
    bool causal;
    int text_start; // -1 for an image segment
    int image_index; // -1 for text; references.size() for target
};
struct SequenceGeometry {
    std::vector<std::array<float, 3>> positions;
    std::vector<SequenceSegment> segments;
    int prefix_length = 0;
};
SequenceGeometry make_sequence_geometry(int text_length, int height, int width,
                                       const std::vector<ReferenceGeometry> &);
} // namespace tc::qwen21
