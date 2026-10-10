#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include "../../tools/native/qwen_lora_upload.hpp"
#include <bit>
#include <iostream>

using namespace tc;
using namespace tc::ane;
DeviceMatrixView matrix(const Tensor &a,int rows,int cols) {
    return {const_cast<void *>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),rows,cols,
        size_t(a.strides(-2))*a.itemsize(),a.dtype()==mx::float32?DType::FP32:a.dtype()==mx::bfloat16?DType::BF16:DType::FP16,
        std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
int main(){@autoreleasepool {try {
    configure_streams();gpu::Device device(false,false);
    int cases=0,rejected=0,guarded=0;float maximum=0;
    for (auto dtype:{mx::bfloat16,mx::float16}) for (int rank:{64,256}) for (bool swapped:{false,true}) {
        research::LoraUpload upload(rank,dtype==mx::bfloat16?DType::BF16:DType::FP16,160,swapped);
        for (int rows:{1,33,145}) for (auto boundary:{mx::bfloat16,mx::float16,mx::float32}) for(float scale:{.75f,-.4f}) {
            constexpr int n=83,first=7;
            auto ranks=mx::reshape(mx::sin(mx::arange((rows+4)*rank,mx::float32)*.037f)*.125f,{1,rows+4,rank});
            auto up=mx::astype(mx::reshape(mx::cos(mx::arange((n+15)*(rank+32),mx::float32)*.019f)*.035f,{n+15,rank+32}),dtype);
            auto selected=mx::contiguous(mx::slice(up,{0,0},{n+15,rank}));
            auto local=mx::contiguous(mx::slice(ranks,{0,3,0},{1,rows+3,rank}));
            auto expected=dense_gpu::lora_b_epilogue(local,selected,first,first+n,scale,boundary);
            mx::eval({ranks,up,expected});
            gpu::Surface destination(device,n,rows+5,gpu::Element::FP16);
            auto bv=matrix(up,n+15,rank);bv.row_stride_bytes=size_t(rank+32)*2;
            auto projection=research::LoraUploadProjection{matrix(ranks,rows+3,rank),bv,destination,3,first,scale,
                boundary==mx::float32?DType::FP32:boundary==mx::bfloat16?DType::BF16:DType::FP16};
            uint32_t flags=0;auto result=upload.run({projection},flags);require(result.ok && !flags,result.error);
            auto ef=mx::astype(expected,mx::float32);mx::eval(ef);const auto *reference=ef.data<float>();
            double error=0,energy=0;
            for(int c=0;c<n;++c)for(int r=0;r<rows+5;++r) {
                const float value=float(reinterpret_cast<const _Float16 *>(static_cast<const char *>(destination.data())+size_t(c)*destination.pitch())[r]);
                require(std::isfinite(value),"rank surface nonfinite");
                const float target=r<rows?float(_Float16(reference[r*n+c])):0;
                error+=(value-target)*(value-target);energy+=target*target;
                require(std::abs(value-target)<=.0003f,"rank upload typed boundary absolute error");
            }
            const float relative=std::sqrt(error/std::max(energy,1e-20));
            require(relative<.004f,"rank upload typed boundary relative error");maximum=std::max(maximum,relative);++cases;
            if(cases==1) {
                auto reject=[&](auto p){try {upload.run({p},flags);}catch(const std::invalid_argument &){++rejected;return;}
                    throw std::runtime_error("invalid rank upload accepted");};
                auto p=projection;p.begin_row=-1;reject(p);p=projection;p.first_channel=bv.rows;reject(p);
                p=projection;p.ranks.owner.reset();reject(p);p=projection;p.up.buffer_bytes=1;reject(p);
                p=projection;p.scale=std::numeric_limits<float>::infinity();reject(p);
                try {upload.run({projection,projection},flags);throw std::runtime_error("aliased rank destinations accepted");}
                catch(const std::invalid_argument &){++rejected;}
                p=projection;p.ranks.dtype=DType::BF16;reject(p);
                id<MTLDevice> metal=MTLCreateSystemDefaultDevice();
                id<MTLBuffer> alias=[metal newBufferWithBytesNoCopy:destination.data() length:destination.bytes()
                    options:MTLResourceStorageModeShared deallocator:nil];
                require(alias!=nil,"alias rejection fixture allocation");
                p=projection;p.ranks.buffer=(__bridge void *)alias;p.ranks.buffer_bytes=alias.length;p.ranks.offset_bytes=0;
                p.ranks.owner=std::make_shared<gpu::Surface>(destination);reject(p);
                for (float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
                    auto nonfinite=mx::full({1,rows+3,rank},bad,mx::float32);mx::eval(nonfinite);
                    p=projection;p.ranks=matrix(nonfinite,rows+3,rank);
                    const auto rejected_run=upload.run({p},flags);
                    require(!rejected_run.ok && !rejected_run.timed_out && (flags&1),"nonfinite ranks did not fail validation");++guarded;
                }
                auto nonfinite_b=mx::full({n+15,rank},std::numeric_limits<float>::infinity(),dtype);mx::eval(nonfinite_b);
                p=projection;p.up=matrix(nonfinite_b,n+15,rank);
                auto rejected_run=upload.run({p},flags);
                require(!rejected_run.ok && !rejected_run.timed_out && (flags&1),"nonfinite B did not fail validation");++guarded;
                p=projection;p.scale=1e35f;rejected_run=upload.run({p},flags);
                require(!rejected_run.ok && !rejected_run.timed_out && (flags&1),"FP16 carrier overflow did not fail validation");++guarded;
                const auto recovered=upload.run({projection},flags);
                require(recovered.ok && !flags,"finite producer did not recover after validation failure");
            }
        }
    }
    std::cout<<"PASS rank-to-surface numeric_cases="<<cases<<" rejected="<<rejected<<" maximum_relative_l2="<<maximum
             <<": original BF16/FP16 B, F32 ranks, both orientations, offsets/physical pitch, padding and typed boundary\n";
    std::cout<<"PASS rank-to-surface guarded_nonfinite_overflow="<<guarded<<" recovered_finite_producer=1\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}}
