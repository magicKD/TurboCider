#pragma once
#include "model.hpp"

namespace tc::seedvr2 {
// Drop owned normalized/padded input handles only after the encoder's
// independent FP16 condition is materialized. Logical bytes are not a
// physical-memory guarantee: external aliases and allocator caches may live.
inline uint64_t release_encoded_input(Tensor &padded, Tensor &source, const Tensor &condition) {
    require(source.ndim() == 5 && source.shape(0) == 1 && source.shape(1) == 3 &&
            source.shape(2) > 0 && source.dtype() == mx::float32 &&
            padded.ndim() == 5 && padded.shape(0) == 1 && padded.shape(1) == 3 &&
            padded.shape(2) == padded_frames(source.shape(2)) && padded.dtype() == mx::float32 &&
            padded.shape(3) == source.shape(3) && padded.shape(4) == source.shape(4) &&
            source.shape(3) % 16 == 0 && source.shape(4) % 16 == 0,
            "invalid owned SeedVR2 encoder input geometry/precision");
    require(condition.is_available() && condition.dtype() == mx::float16 &&
            condition.shape() == mx::Shape{1,16,(padded.shape(2)-1)/4+1,source.shape(3)/8,source.shape(4)/8},
            "SeedVR2 input release requires completed matching FP16 conditioning");
    if (source.is_available())
        require(condition.buffer().ptr() != source.buffer().ptr(),
                "SeedVR2 conditioning aliases the source input allocation");
    if (padded.is_available())
        require(condition.buffer().ptr() != padded.buffer().ptr(),
                "SeedVR2 conditioning aliases the padded input allocation");
    const uint64_t bytes = source.nbytes();
    padded = Tensor(0);
    source = Tensor(0);
    return bytes;
}
} // namespace tc::seedvr2
