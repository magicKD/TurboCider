#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace tc::dense_gpu {
namespace mx=mlx::core;

// Approximate B operands only: keep original F32 A-ranks until this dispatch's
// small input conversion. F32 B accumulation/scale, optional F32 base add,
// then ONE requested output cast. Neither W nor the full base is compacted.
// Caller preserves stacked-adapter order and the delta aggregation boundary.
inline mx::array lora_b_epilogue(const mx::array &ranks,const mx::array &up,
    int first,int last,float scale,mx::Dtype output_dtype,
    const std::optional<mx::array> &base=std::nullopt,int base_begin=0,int bm=16,int bn=128) {
    if(ranks.ndim()!=3 || ranks.shape(0)!=1 || ranks.shape(1)<=0 || ranks.dtype()!=mx::float32 ||
        up.ndim()!=2 || first<0 || last<=first || last>up.shape(0) || ranks.shape(2)!=up.shape(1) ||
        !up.flags().row_contiguous || (up.dtype()!=mx::bfloat16 && up.dtype()!=mx::float16) ||
        !std::isfinite(scale) || (output_dtype!=mx::float32 && output_dtype!=mx::bfloat16 && output_dtype!=mx::float16) ||
        (bm!=16 && bm!=32 && bm!=64) || (bn!=64 && bn!=128))
        throw std::invalid_argument("LoRA B epilogue rank/source/range/dtype mismatch");
    const int m=ranks.shape(1),n=last-first,k=up.shape(1);
    if(base && (base->ndim()!=3 || base->shape(0)!=1 || base->shape(1)!=m || base_begin<0 ||
        base_begin>base->shape(2)-n || !base->flags().row_contiguous ||
        (base->dtype()!=mx::float32 && base->dtype()!=mx::float16 && base->dtype()!=mx::bfloat16)))
        throw std::invalid_argument("LoRA B epilogue requires a valid contiguous physical base range");
    if(!base && base_begin!=0)throw std::invalid_argument("LoRA B epilogue base offset without base");
    static auto kernel=mx::fast::metal_kernel("tc_lora_b_f32_epilogue",{"ranks","up","base","scale"},{"out"},R"metal(
        using namespace mpp::tensor_ops;
        uint row=threadgroup_position_in_grid.y*BM,col=threadgroup_position_in_grid.x*BN;
        auto a=tensor(const_cast<device T*>(ranks),dextents<int,2>{K,M},array<int,2>{1,K});
        auto b=tensor(const_cast<device T*>(up)+FIRST*K,dextents<int,2>{K,N},array<int,2>{1,K});
        auto aa=a.slice(0,row),bb=b.slice(0,col);
        matmul2d<matmul2d_descriptor(BM,BN,K,false,true,false),execution_simdgroups<4>> op;
        auto acc=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();op.run(aa,bb,acc);
        for(uint i=0;i<acc.get_capacity();++i) {
            if(!acc.is_valid_element(i))continue;
            auto coord=acc.get_multidimensional_index(i);
            if(row+coord[1]<M && col+coord[0]<N) {
                float value=acc[i]*scale;
                if constexpr(ADD_BASE)value+=float(base[(row+coord[1])*BASE_PITCH+BASE_BEGIN+col+coord[0]]);
                out[(row+coord[1])*N+col+coord[0]]=O(value);
            }
        }
    )metal","#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    const auto input=mx::astype(ranks,up.dtype());
    const auto backing=base?*base:mx::zeros({2},mx::float32);
    return kernel({input,up,backing,mx::array(scale,mx::float32)},{{1,m,n}},{output_dtype},
        {((n+bn-1)/bn)*128,(m+bm-1)/bm,1},{128,1,1},
        {{"T",up.dtype()},{"O",output_dtype},{"M",m},{"N",n},{"K",k},{"FIRST",first},
         {"ADD_BASE",bool(base)},{"BASE_PITCH",base?base->shape(2):1},{"BASE_BEGIN",base_begin},{"BM",bm},{"BN",bn}}, {},false,{})[0];
}
} // namespace tc::dense_gpu
