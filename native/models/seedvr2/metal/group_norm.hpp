#pragma once
#include "../model.hpp"

namespace tc::seedvr2::metal {
// Per-frame, 32-group statistics in FP32. Chunk-major dispatch keeps the
// channel-last pixels hot across adjacent groups. Centered variance avoids
// cancellation on constant/near-constant inputs; no FP32 full-frame transpose
// or affine intermediate escapes the two small statistical buffers.
inline Tensor group_norm(const Tensor &input, const Tensor &weight,
                         const Tensor &bias, bool activate) {
    require(input.ndim() == 5 && input.dtype() == mx::float16 && input.shape(4) % 32 == 0 &&
            weight.shape() == mx::Shape{input.shape(4)} && bias.shape() == weight.shape() &&
            weight.dtype() == input.dtype() && bias.dtype() == input.dtype(),
            "SeedVR2 fused group norm requires FP16 NTHWC and 32 groups");
    const int bt = input.shape(0) * input.shape(1), hw = input.shape(2) * input.shape(3), c = input.shape(4);
    const int cg = c / 32, elements = hw * cg, chunks = (elements + 4095) / 4096;
    require(bt > 0 && hw > 0 && cg > 0, "empty SeedVR2 group norm input");
    auto x = mx::contiguous(input);
    static auto statistics = mx::fast::metal_kernel(
        "tc_seedvr2_group_norm_statistics", {"x", "mean"}, {"partial"}, R"metal(
        uint task = threadgroup_position_in_grid.x;
        uint group = task % 32;
        uint chunk = (task / 32) % Chunks;
        uint frame = task / (32 * Chunks);
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        float total = 0.0f;
        float center = Variance ? mean[frame * 32 + group] : 0.0f;
        for (uint offset = tid; offset < 4096; offset += 256) {
            uint element = chunk * 4096 + offset;
            if (element < Elements) {
                uint pixel = element / CG, channel = element % CG;
                float value = float(x[(frame * HW + pixel) * C + group * CG + channel]);
                float difference = value - center;
                total += Variance ? difference * difference : value;
            }
        }
        threadgroup float sums[8];
        total = simd_sum(total);
        if (lane == 0) sums[sg] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float sum = simd_sum(lane < 8 ? sums[lane] : 0.0f);
            if (lane == 0) partial[(frame * 32 + group) * Chunks + chunk] = sum;
        }
        )metal");
    auto stats = [&](const Tensor &mean, bool variance) {
        return statistics({x, mean}, {{bt,32,chunks}}, {mx::float32},
            {bt*32*chunks*256,1,1}, {256,1,1},
            {{"C",c},{"CG",cg},{"HW",hw},{"Elements",elements},{"Chunks",chunks},{"Variance",variance}},
            {}, false, {})[0];
    };
    auto mean = mx::sum(stats(mx::zeros({1}, mx::float32), false), 2) / Tensor(float(elements));
    auto variance = mx::sum(stats(mean, true), 2) / Tensor(float(elements));
    auto inverse = mx::rsqrt(variance + Tensor(1e-6f));
    static auto affine = mx::fast::metal_kernel(
        "tc_seedvr2_group_norm_affine", {"x","mean","inverse","w","b"}, {"out"}, R"metal(
        uint index = thread_position_in_grid.x;
        if (index >= Size) return;
        uint channel = index % C;
        uint frame = index / (HW * C);
        uint group = channel / CG;
        float normalized = (float(x[index]) - mean[frame * 32 + group]) * inverse[frame * 32 + group];
        // Preserve the reference's FP32 multiplication then addition and its
        // FP16 rounding before sigmoid/multiplication.
        volatile float scaled = normalized * float(w[channel]);
        half value = half(scaled + float(b[channel]));
        if (Activate) {
            half sigmoid = half(1.0f / (1.0f + metal::precise::exp(-float(value))));
            value = half(float(value) * float(sigmoid));
        }
        out[index] = value;
        )metal");
    return affine({x,mean,inverse,weight,bias}, {input.shape()}, {input.dtype()},
        {int(input.size()),1,1}, {256,1,1},
        {{"C",c},{"CG",cg},{"HW",hw},{"Size",int(input.size())},{"Activate",activate}}, {},false,{})[0];
}
} // namespace tc::seedvr2::metal
