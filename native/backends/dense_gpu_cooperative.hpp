#pragma once
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::dense_gpu {
namespace mx=mlx::core;

// Four independent SIMD tiles per TG, explicit K-blocked accumulation and
// original typed physical W as a cooperative tensor or tensor view.
// No W compaction/widening or precision change. Research-only until measured.
inline mx::array projection_cooperative_range(const mx::array &x,const mx::array &w,
    int rb,int re,int cb,int ce,bool retain_fp32=true,int bk=32,int sn=32,int sm=64,bool register_right=true) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<=0 || w.ndim()!=2 ||
        rb<0 || re<=rb || re>w.shape(0) || cb<0 || ce<=cb || ce>w.shape(1) || x.shape(2)!=ce-cb ||
        !w.flags().row_contiguous || (x.dtype()!=mx::float16 && x.dtype()!=mx::bfloat16) || w.dtype()!=x.dtype() ||
        (bk!=32 && bk!=64) || (sn!=32 && sn!=64) || (bk==64 && sn==64) ||
        (sm!=16 && sm!=32 && sm!=64) || (sm!=16 && (bk!=32 || sn!=32)))
        throw std::invalid_argument("cooperative dense physical geometry/dtype/tile mismatch");
    const int m=x.shape(1),n=re-rb,k=ce-cb,pitch=w.shape(1);
    const bool masked=k%bk!=0;
    // Both cooperative inputs require elementary M/N/K in {16,32}.
    // Tail dispatches safely narrow the tile, not the arithmetic or source.
    const int dispatch_sm=masked && sm==64?32:sm,dispatch_sn=masked?32:sn;
    const int bm=2*dispatch_sm,bn=2*dispatch_sn;
    static auto kernel=mx::fast::metal_kernel("tc_dense_cooperative_kblocked_candidate",{"x","w"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        uint warp=simdgroup_index_in_threadgroup;
        uint row=threadgroup_position_in_grid.y*(2*SM)+(warp/2)*SM;
        uint col=threadgroup_position_in_grid.x*(2*SN)+(warp%2)*SN;
        // Unaligned K uses explicitly masked K32, M<=32 and N32 subtiles;
        // aligned K retains the requested fast tile. Not a precision change.
        constexpr uint STEP=MASKED_A ? 32 : BK;
        matmul2d<matmul2d_descriptor(SM,SN,STEP,false,true,false,
            matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroup> op;
        auto b=op.template get_right_input_cooperative_tensor<T,T,float>();
        auto aa=op.template get_left_input_cooperative_tensor<T,T,float>();
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(b),float>();
        #pragma unroll
        for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0;
        for(uint kb=0;kb<K;kb+=STEP) {
            if constexpr(MASKED_A) {
                #pragma unroll
                for(uint i=0;i<aa.get_capacity();++i) {
                    if(!aa.is_valid_element(i))continue;
                    auto c=aa.get_multidimensional_index(i);
                    uint r=row+c[1],k=kb+c[0];
                    aa[i]=(r<M && k<K) ? x[r*K+k] : T(0);
                }
            }
            if constexpr(REGISTER_B || MASKED_A) {
                #pragma unroll
                for(uint i=0;i<b.get_capacity();++i) {
                    if(!b.is_valid_element(i))continue;
                    auto c=b.get_multidimensional_index(i);
                    uint r=col+c[1],k=kb+c[0];
                    b[i]=(r<N && k<K) ? w[(RB+r)*WP+CB+k] : T(0);
                }
            }
            if constexpr(MASKED_A)op.run(aa,b,acc);
            else {
                // Tiny masked inputs may be constant-address-space MLX
                // arguments. Only the aligned tensor path needs device views.
                auto a=tensor(const_cast<device T*>(x),dextents<int,2>{K,M},array<int,2>{1,K});
                auto input=a.slice(kb,row);
                if constexpr(REGISTER_B)op.run(input,b,acc);
                else {
                    auto weight=tensor(const_cast<device T*>(w)+RB*WP+CB,dextents<int,2>{K,N},array<int,2>{1,WP});
                    auto weights=weight.slice(kb,col);op.run(input,weights,acc);
                }
            }
        }
        #pragma unroll
        for(uint i=0;i<acc.get_capacity();++i) {
            if(!acc.is_valid_element(i))continue;
            auto c=acc.get_multidimensional_index(i);
            if(row+c[1]<M && col+c[0]<N)out[(row+c[1])*N+col+c[0]]=O(acc[i]);
        }
    )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const auto dtype=retain_fp32?mx::float32:x.dtype();
    return kernel({x,w},{{1,m,n}},{dtype},{((n+bn-1)/bn)*128,(m+bm-1)/bm,1},{128,1,1},
        {{"T",x.dtype()},{"O",dtype},{"M",m},{"N",n},{"K",k},{"WP",pitch},{"RB",rb},{"CB",cb},{"BK",bk},{"SN",dispatch_sn},{"SM",dispatch_sm},{"MASKED_A",masked},{"REGISTER_B",register_right}}, {},false,{})[0];
}
} // namespace tc::dense_gpu
