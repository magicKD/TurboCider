#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::dense_gpu {
namespace mx = mlx::core;

// Original dense [out, physical_in] operands, FP32 accumulation. No widened
// W bank or compact source copy. Optional F32 partial omits only the final
// output narrowing; callers must perform the model boundary after joining.
inline mx::array projection_range(const mx::array &x, const mx::array &w,
    int row_begin, int row_end, int col_begin, int col_end, int bm = 32,
    bool retain_fp32 = false, int bn = 128, bool static_tiles = false) {
    if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(1) <= 0 || w.ndim() != 2 ||
        row_begin < 0 || row_end <= row_begin || row_end > w.shape(0) ||
        col_begin < 0 || col_end <= col_begin || col_end > w.shape(1) ||
        x.shape(2) != col_end-col_begin || !w.flags().row_contiguous ||
        (x.dtype() != mx::bfloat16 && x.dtype() != mx::float16) || w.dtype() != x.dtype() ||
        (bm != 16 && bm != 32 && bm != 64) || (bn != 64 && bn != 128))
        throw std::invalid_argument("dense Metal physical projection range geometry/dtype mismatch");
    const int m=x.shape(1), n=row_end-row_begin, k=col_end-col_begin, pitch=w.shape(1);
    const auto output_dtype=retain_fp32?mx::float32:x.dtype();
    // Keep the established range kernel's arithmetic/physical leading
    // dimension, including its name. O=T is the original output boundary.
    auto make_kernel=[](bool aligned) {
        const std::string source=std::string(R"metal(
        using namespace mpp::tensor_ops;
        uint row=threadgroup_position_in_grid.y*BM, col=threadgroup_position_in_grid.x*BN;
        auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
        auto b=tensor(const_cast<device T*>(w)+ROW_BEGIN*WP+COL_BEGIN,dextents<int,2>{K,N},array<int,2>{1,WP});
        )metal")+(aligned ? "auto aa=a.template slice<K,BM>(0,row);auto bb=b.template slice<K,BN>(0,col);\n" :
                                 "auto aa=a.slice(0,row),bb=b.slice(0,col);\n")+R"metal(
        matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
        for(uint i=0;i<acc.get_capacity();++i) {
            if(!acc.is_valid_element(i))continue;
            auto coord=acc.get_multidimensional_index(i);
            if(row+coord[1]<M&&col+coord[0]<N)out[(row+coord[1])*N+col+coord[0]]=O(acc[i]);
        }
    )metal";
        return mx::fast::metal_kernel(aligned ? "tc_dense_projection_static_range" : "tc_z_mpp_projection_range",
            {"x","w"},{"out"},source,"#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    };
    static auto dynamic_kernel=make_kernel(false),aligned_kernel=make_kernel(true);
    // Static extents are an explicit candidate, used only for full tiles.
    // Unaligned rows/columns retain the original checked dynamic geometry.
    const auto &kernel=static_tiles && m%bm==0 && n%bn==0 ? aligned_kernel : dynamic_kernel;
    return kernel({x,w},{{1,m,n}},{output_dtype}, {((n+bn-1)/bn)*128,(m+bm-1)/bm,1},{128,1,1},
        {{"T",x.dtype()},{"O",output_dtype},{"M",m},{"N",n},{"K",k},{"WP",pitch},
         {"ROW_BEGIN",row_begin},{"COL_BEGIN",col_begin},{"BM",bm},{"BN",bn}}, {},false,{})[0];
}
} // namespace tc::dense_gpu
