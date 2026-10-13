#pragma once
#include "../model.hpp"

namespace tc::seedvr2 {
// Retain only the original causal tail, not a view into a halo+activation
// concatenation. The replicated initial singleton owns one frame and exposes
// a zero-stride temporal view: its two identical history frames need not occupy
// two buffers. Subsequent histories preserve every distinct frame unchanged.
inline Tensor causal_history(const Tensor &input, const Tensor &head, int count, bool stored) {
    require(input.ndim() == 5 && head.ndim() == 5 && count > 0,
            "invalid SeedVR2 causal history geometry");
    auto tail = [](const Tensor &x, int frames) {
        auto first = mx::Shape(x.ndim(), 0), last = x.shape();
        first[1] = x.shape(1) - frames;
        return mx::slice(x, first, last);
    };
    if (input.shape(1) >= count) return mx::copy(mx::contiguous(tail(input, count)));
    if (!stored && input.shape(1) == 1) {
        auto owned = mx::copy(mx::contiguous(input));
        auto shape = input.shape();
        shape[1] = count;
        return mx::broadcast_to(owned, shape);
    }
    const int missing = count - input.shape(1);
    auto previous = stored ? tail(head, missing) : mx::repeat(mx::slice(input,
        {0,0,0,0,0}, {input.shape(0),1,input.shape(2),input.shape(3),input.shape(4)}), missing, 1);
    return mx::contiguous(mx::concatenate({previous, input}, 1));
}
} // namespace tc::seedvr2
