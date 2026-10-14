#pragma once
#include "../model.hpp"
#include <limits>
namespace tc::seedvr2::metal {
// The GEMM result has already rounded to FP16. Fuse ONLY its separate FP16
// bias add and phase/layout shuffle; never fold bias into GEMM accumulation.
inline Tensor upscale_bias_shuffle(const Tensor &projection,const Tensor &bias,
                                  bool initial,bool temporal) {
    require(projection.ndim()==5 && projection.dtype()==mx::float16 && bias.ndim()==1 &&
        bias.dtype()==mx::float16,"fused upsampling requires FP16 projection/bias");
    const int b=projection.shape(0),t=projection.shape(1),h=projection.shape(2),w=projection.shape(3),
        expanded=projection.shape(4),tr=temporal?2:1;
    require(b>0 && t>0 && h>0 && w>0 && expanded>0 && expanded%(4*tr)==0 &&
        bias.shape(0)==expanded,"fused upsampling shape/phase mismatch");
    const int64_t wide_t=int64_t(t)*tr-(initial && temporal?1:0);
    require(wide_t>0 && wide_t<=std::numeric_limits<int>::max() &&
        h<=std::numeric_limits<int>::max()/2 && w<=std::numeric_limits<int>::max()/2 &&
        projection.size()<=uint64_t(std::numeric_limits<uint32_t>::max()),"fused upsampling extent exceeds bounded scope");
    const int c=expanded/(4*tr),out_t=int(wide_t);
    const uint64_t positions=uint64_t(b)*out_t*h*w*4;
    require(out_t>0 && positions<=uint64_t(std::numeric_limits<int>::max()) &&
        positions*c<=uint64_t(std::numeric_limits<int>::max()) &&
        projection.size()<=uint64_t(std::numeric_limits<uint32_t>::max()),"fused upsampling indexing exceeds bounded scope");
    static auto kernel=mx::fast::metal_kernel("tc_seedvr2_upscale_bias_shuffle",{"x","bias"},{"out"},R"metal(
        // Projection and output are BOTH channel-last. Consecutive lanes
        // access consecutive channels on both sides; an NCHW-style transpose
        // tile would unnecessarily scatter reads and add a shared-memory fence.
        uint index=thread_position_in_grid.x;if(index>=Size)return;
        uint channel=index%C,position=index/C;
        uint frame=position/(4*H*W),batch=frame/OutT,output_t=frame%OutT;
        uint expanded_t=output_t+(Skip && output_t>=1?1:0),source_t=expanded_t/Tr,tp=expanded_t%Tr;
        uint pixel=position%(4*H*W),row=pixel/(2*W),col=pixel%(2*W);
        uint oc=(((row%2)*2+col%2)*Tr+tp)*C+channel;
        uint at=(((batch*T+source_t)*H+row/2)*W+col/2)*(4*Tr*C)+oc;
        out[index]=half(float(x[at])+float(bias[oc]));
    )metal");
    return kernel({projection,bias},{{b,out_t,h*2,w*2,c}},{mx::float16},
        {int(positions*c),1,1},{256,1,1},
        {{"C",c},{"T",t},{"H",h},{"W",w},{"Tr",tr},{"OutT",out_t},
         {"Size",int(positions*c)},{"Skip",initial && temporal}},{},false,{})[0];
}
} // namespace tc::seedvr2::metal
