#pragma once
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <stdexcept>

namespace tc::affine_gpu {
namespace mx=mlx::core;
// Explicit MLX affine Q4/Q8, not raw GGUF blocks. Decode in the ORIGINAL
// activation/metadata dtype, accumulate in F32 and retain an F32 partial.
// Never widen metadata before decode, or change a model's stored weights.
// Research consumer: no default routing or performance qualification.
inline mx::array projection_fp32(const mx::array &x,const mx::array &words,
    const mx::array &scales,const mx::array &biases,int bits,
    int row_begin,int row_end,int col_begin,int col_end) {
    if(x.ndim()!=3 || x.shape(0)!=1 || x.shape(1)<32 || words.ndim()!=2 ||
       scales.ndim()!=2 || biases.shape()!=scales.shape() || (bits!=4&&bits!=8) ||
       words.dtype()!=mx::uint32 || (x.dtype()!=mx::bfloat16&&x.dtype()!=mx::float16) ||
       scales.dtype()!=x.dtype() || biases.dtype()!=x.dtype() || !words.flags().row_contiguous ||
       scales.shape(0)!=words.shape(0) || scales.shape(1)<=0)
        throw std::invalid_argument("affine F32 partial requires produced same-dtype Q4/Q8 matrices and M>=32");
    const int physical=words.shape(1)*(32/bits),groups=scales.shape(1);
    const int group=physical/groups;
    if(physical%groups || (group!=32&&group!=64&&group!=128) ||
       row_begin<0 || row_end<=row_begin || row_end>words.shape(0) || col_begin<0 || col_end<=col_begin ||
       col_end>physical || col_begin%group || col_end%group || x.shape(2)!=col_end-col_begin)
        throw std::invalid_argument("affine F32 partial physical range/group mismatch");
    const int m=x.shape(1),n=row_end-row_begin,k=col_end-col_begin;
    static auto kernel=mx::fast::metal_kernel("tc_affine_decode_t_fp32_partial",
        {"x","words","scales","biases"},{"out"},R"metal(
        constexpr uint BM=32,BN=64,BK=32,STRIDE=36;
        threadgroup float xs[BM*STRIDE],ws[BN*STRIDE],result[BM*BN];
        uint lane=thread_index_in_threadgroup;
        uint warp=simdgroup_index_in_threadgroup;
        uint m0=threadgroup_position_in_grid.y*BM,n0=threadgroup_position_in_grid.x*BN;
        uint wm=warp/2,wn=warp%2;
        simdgroup_matrix<float,8,8> acc[2][4];
        for(uint i=0;i<2;++i)for(uint j=0;j<4;++j)acc[i][j]=make_filled_simdgroup_matrix<float,8,8>(0);
        for(uint kb=0;kb<K;kb+=BK) {
            for(uint at=lane;at<BM*BK;at+=128) {
                uint r=at/BK,c=at%BK;
                xs[r*STRIDE+c]=m0+r<M?float(x[(m0+r)*K+kb+c]):0;
            }
            for(uint at=lane;at<BN*BK;at+=128) {
                uint r=at/BK,c=at%BK;
                float value=0;
                if(n0+r<N) {
                    uint physical_row=ROW_BEGIN+n0+r,physical_col=COL_BEGIN+kb+c;
                    uint word=words[physical_row*(WP/(32/BITS))+physical_col/(32/BITS)];
                    uchar code=uchar((word>>((physical_col%(32/BITS))*BITS))&((1u<<BITS)-1));
                    uint meta=physical_row*(WP/GROUP)+physical_col/GROUP;
                    // Same T expression/storage boundary as MLX's affine
                    // QuantizedBlockLoader. The coefficient is NOT F32.
                    T decoded=scales[meta]*code+biases[meta];
                    value=float(decoded);
                }
                ws[r*STRIDE+c]=value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint kk=0;kk<BK;kk+=8) {
                simdgroup_matrix<float,8,8> a[2],b[4];
                for(uint i=0;i<2;++i)simdgroup_load(a[i],xs+(wm*16+i*8)*STRIDE+kk,STRIDE);
                for(uint j=0;j<4;++j)simdgroup_load(b[j],ws+(wn*32+j*8)*STRIDE+kk,STRIDE,ulong2(0),true);
                for(uint i=0;i<2;++i)for(uint j=0;j<4;++j)simdgroup_multiply_accumulate(acc[i][j],a[i],b[j],acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint i=0;i<2;++i)for(uint j=0;j<4;++j)
            simdgroup_store(acc[i][j],result+(wm*16+i*8)*BN+wn*32+j*8,BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint at=lane;at<BM*BN;at+=128) {
            uint r=at/BN,c=at%BN;
            if(m0+r<M&&n0+c<N)out[(m0+r)*N+n0+c]=result[at];
        }
    )metal");
    return kernel({x,words,scales,biases},{{1,m,n}},{mx::float32},
        {((n+63)/64)*128,(m+31)/32,1},{128,1,1},
        {{"T",x.dtype()},{"M",m},{"N",n},{"K",k},{"WP",physical},{"GROUP",group},{"BITS",bits},
         {"ROW_BEGIN",row_begin},{"COL_BEGIN",col_begin}}, {},false,{})[0];
}
}
