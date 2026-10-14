#pragma once
#include "model.hpp"
#include <algorithm>

namespace tc::seedvr2 {
using FrameHistory = std::vector<Tensor>;
using FrameHistoryMap = std::unordered_map<std::string, FrameHistory>;

// Each entry owns exactly one logical frame, or shares a known independently
// materialized singleton producer. Do not infer that ownership from contiguous
// flags: a contiguous slice may still pin a much larger activation.
inline FrameHistory advance_frame_history(const Tensor &input, const FrameHistory &previous,
                                          int count, bool stored, bool retain_owned_singleton) {
    require(input.ndim() == 5 && input.shape(1) >= 1 && count >= 1 && count <= 2 &&
            (!stored || int(previous.size()) == count), "invalid SeedVR2 frame history");
    auto own = [&](int index) {
        if (retain_owned_singleton && input.shape(1) == 1) {
            mx::eval(input);
            return input;
        }
        auto frame = mx::slice(input,{0,index,0,0,0},
            {input.shape(0),index+1,input.shape(2),input.shape(3),input.shape(4)});
        // Copy is a logical identity in MLX, and Contiguous may retain small
        // views. Full performs a real typed copy while the source remains
        // owned here, so an untrusted slice cannot pin its parent backing.
        auto owned = mx::full(frame.shape(),frame,frame.dtype());
        mx::eval(owned);
        return owned;
    };
    if (!stored && input.shape(1) == 1) return FrameHistory(count,own(0));
    FrameHistory next;
    const int incoming = std::min(count,input.shape(1)), missing = count-incoming;
    if (missing) {
        require(stored,"missing initial SeedVR2 causal operands");
        next.insert(next.end(),previous.end()-missing,previous.end());
    }
    for (int index = input.shape(1)-incoming; index < input.shape(1); ++index) next.push_back(own(index));
    return next;
}
} // namespace tc::seedvr2
