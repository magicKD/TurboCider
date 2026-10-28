#pragma once
#include "../model.hpp"

namespace tc::seedvr2::metal {
struct SpatialNormStatistics {Tensor mean,inverse;};
// Same complete-frame 4096-element centered FP32 statistics as group_norm.hpp.
// Tile boundaries never change the statistical scope or group membership.
inline SpatialNormStatistics spatial_norm_statistics(const Tensor &input){
    require(input.ndim()==5 && input.dtype()==mx::float16 && input.shape(4)%32==0,"invalid spatial norm input");
    const int bt=input.shape(0)*input.shape(1),hw=input.shape(2)*input.shape(3),c=input.shape(4),cg=c/32,elements=hw*cg,chunks=(elements+4095)/4096;
    require(bt>0 && hw>0 && cg>0,"empty spatial statistics input");
    auto x=mx::contiguous(input);
    static auto kernel=mx::fast::metal_kernel("tc_seedvr2_spatial_norm_statistics",{"x","mean"},{"partial"},R"metal(
        uint task=threadgroup_position_in_grid.x,group=task%32,chunk=(task/32)%Chunks,frame=task/(32*Chunks);
        uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
        float total=0.0f,center=Variance?mean[frame*32+group]:0.0f;
        for(uint offset=tid;offset<4096;offset+=256){uint element=chunk*4096+offset;
            if(element<Elements){uint pixel=element/CG,channel=element%CG;
                float value=float(x[(frame*HW+pixel)*C+group*CG+channel]),difference=value-center;
                total+=Variance?difference*difference:value;}}
        threadgroup float sums[8];total=simd_sum(total);if(lane==0)sums[sg]=total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sg==0){float sum=simd_sum(lane<8?sums[lane]:0.0f);if(lane==0)partial[(frame*32+group)*Chunks+chunk]=sum;}
    )metal");
    auto partial=[&](const Tensor &mean,bool variance){return kernel({x,mean},{{bt,32,chunks}},{mx::float32},
        {bt*32*chunks*256,1,1},{256,1,1},{{"C",c},{"CG",cg},{"HW",hw},{"Elements",elements},{"Chunks",chunks},{"Variance",variance}},{},false,{})[0];};
    auto mean=mx::sum(partial(mx::zeros({1},mx::float32),false),2)/Tensor(float(elements));
    auto variance=mx::sum(partial(mean,true),2)/Tensor(float(elements));
    auto inverse=mx::rsqrt(variance+Tensor(1e-6f));mx::eval({mean,inverse});
    return {mean,inverse};
}
inline Tensor spatial_norm_affine(const Tensor &rows,const SpatialNormStatistics &statistics,
                                  const Tensor &weight,const Tensor &bias,bool activate){
    require(rows.ndim()==5 && rows.dtype()==mx::float16 && rows.shape(4)%32==0 &&
        weight.shape()==mx::Shape{rows.shape(4)} && bias.shape()==weight.shape() &&
        weight.dtype()==rows.dtype() && bias.dtype()==rows.dtype() &&
        statistics.mean.dtype()==mx::float32 && statistics.inverse.dtype()==mx::float32 &&
        statistics.mean.shape()==mx::Shape{rows.shape(0)*rows.shape(1),32} && statistics.inverse.shape()==statistics.mean.shape(),
        "invalid global-statistics spatial affine geometry");
    auto x=mx::contiguous(rows);const int c=rows.shape(4),hw=rows.shape(2)*rows.shape(3);
    static auto kernel=mx::fast::metal_kernel("tc_seedvr2_spatial_norm_affine",{"x","mean","inverse","w","b"},{"out"},R"metal(
        uint index=thread_position_in_grid.x;if(index>=Size)return;
        uint channel=index%C,frame=index/(HW*C),group=channel/CG;
        float normalized=(float(x[index])-mean[frame*32+group])*inverse[frame*32+group];
        volatile float scaled=normalized*float(w[channel]);half value=half(scaled+float(b[channel]));
        if(Activate){half sigmoid=half(1.0f/(1.0f+metal::precise::exp(-float(value))));value=half(float(value)*float(sigmoid));}
        out[index]=value;
    )metal");
    return kernel({x,statistics.mean,statistics.inverse,weight,bias},{rows.shape()},{rows.dtype()},
        {int(rows.size()),1,1},{256,1,1},{{"C",c},{"CG",c/32},{"HW",hw},{"Size",int(rows.size())},{"Activate",activate}},{},false,{})[0];
}
} // namespace tc::seedvr2::metal
