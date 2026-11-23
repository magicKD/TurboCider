#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <string>
#include <string_view>
#include <vector>

namespace tc::metal_qkv {
namespace mx = mlx::core;

// Each MPP tile covers one attention head and writes the head-major Q/K/V
// layout directly. The caller supplies its RMSNorm epsilon and RoPE loads.
inline mx::fast::CustomKernelFunction make_projection_kernel(
    const std::string &name, const std::vector<std::string> &input_names,
    std::string_view rms_epsilon, std::string_view rotary_loads) {
    std::string source = "constexpr float rms_epsilon = ";
    source += rms_epsilon;
    source += R"metal(;
        using namespace mpp::tensor_ops;
        uint row = threadgroup_position_in_grid.y * BM;
        uint col = threadgroup_position_in_grid.x * 128;
        auto a = tensor(const_cast<device T*>(x), dextents<int,2>{HIDDEN,M},
                        array<int,2>{1,HIDDEN});
        auto b = tensor(const_cast<device T*>(w), dextents<int,2>{HIDDEN,3 * HIDDEN},
                        array<int,2>{1,HIDDEN});
        auto aa = a.slice(0,row), bb = b.slice(0,col);
        matmul2d<matmul2d_descriptor(BM,128,HIDDEN,false,true,false),
                 execution_simdgroups<4>> op;
        auto acc = op.template get_destination_cooperative_tensor<
            decltype(aa),decltype(bb),float>();
        op.run(aa,bb,acc);

        threadgroup T tile[BM * 128];
        for (uint i=0; i<acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            auto coord = acc.get_multidimensional_index(i);
            tile[coord[1]*128+coord[0]] = T(acc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint kind = col / HIDDEN;
        uint head = (col % HIDDEN) / 128;
        for (uint local_row=sg; local_row<BM; local_row+=4) {
            uint token = row + local_row;
            if (token >= M) continue;
            uint d0 = lane * 4;
            uint src = local_row * 128 + d0;
            uint dst = (head * M + token) * 128 + d0;
            T values[4];
            float sum = 0.0f;
            for (uint i=0; i<4; ++i) {
                values[i] = tile[src+i];
                float value = float(values[i]);
                sum += value*value;
            }
            if (kind == 2) {
                for (uint i=0; i<4; ++i) v[dst+i] = values[i];
                continue;
            }
            float inv = metal::precise::rsqrt(simd_sum(sum)/128.0f+rms_epsilon);
            for (uint i=0; i<4; ++i) {
                uint d = d0+i;
                float weight = kind == 0 ? float(qw[d]) : float(kw[d]);
                values[i] = T((float(values[i])*inv)*weight);
            }
            device T *out = kind == 0 ? q : k;
            for (uint i=0; i<4; i+=2) {
                uint d = d0+i;
)metal";
    source += rotary_loads;
    source += R"metal(
                float first = float(values[i]), second = float(values[i+1]);
                out[dst+i] = T(first*c-second*s);
                out[dst+i+1] = T(first*s+second*c);
            }
        }
        )metal";
    return mx::fast::metal_kernel(
        name, input_names, {"q", "k", "v"}, source,
        "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
}

inline std::vector<mx::array> run_projection(
    const mx::fast::CustomKernelFunction &kernel, const std::vector<mx::array> &inputs,
    int bm) {
    const auto &x = inputs.front();
    const int m = x.shape(1);
    const int hidden = x.shape(2);
    const mx::Shape shape{1, hidden / 128, m, 128};
    return kernel(inputs, {shape, shape, shape}, {x.dtype(), x.dtype(), x.dtype()},
                  {3 * hidden, (m + bm - 1) / bm, 1}, {128, 1, 1},
                  {{"T", x.dtype()}, {"M", m}, {"BM", bm}, {"HIDDEN", hidden}},
                  {}, false, {});
}

} // namespace tc::metal_qkv
