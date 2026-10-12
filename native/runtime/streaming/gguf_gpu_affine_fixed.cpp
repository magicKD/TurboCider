#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <mlx/backend/metal/device.h>
#include <mlx/primitives.h>
#include "gguf_gpu_affine_fixed.hpp"

namespace tc::streaming {
namespace {
void validate_fixed_batch(const std::vector<Tensor> &raw,
                              const std::vector<GgufGpuAffineFixedTarget *> &targets) {
    require(!raw.empty() && raw.size()==targets.size(),"fixed GPU affine batch mismatch");
    std::unordered_set<const void *> raw_buffers,output_buffers;
    for (const auto &input:raw) {
        require(input.data_shared_ptr() && input.offset()>=0 && input.buffer_size()>=
            uint64_t(input.offset())*input.itemsize()+input.nbytes(),"fixed GPU raw storage too short/unprepared");
        raw_buffers.insert(input.buffer().ptr());
    }
    // Validate every span before any write; only already-owned, contiguous
    // allocator buffers on the GPU stream are accepted.
    for (size_t i=0;i<raw.size();++i) {
        require(targets[i],"missing fixed GPU affine target");const auto &t=*targets[i];
        const uint64_t block_bytes=t.type==8 ? 34 : t.type==2 ? 18 : t.type==3 ? 20 : 0;
        require(block_bytes && raw[i].dtype()==mx::uint8 && raw[i].flags().row_contiguous &&
            raw[i].shape()==mx::Shape{int(t.rows),int(t.columns/32*block_bytes)} && raw[i].nbytes()<=UINT32_MAX,
            "fixed GPU affine raw span mismatch");
        const uint64_t groups=gguf::checked_mul(t.rows,t.columns/32),bits=t.type==8 ? 8 : 4;
        require(groups && groups<=INT32_MAX && t.arrays[0].shape()==mx::Shape{int(t.rows),int(t.columns*bits/32)} &&
            t.arrays[1].shape()==mx::Shape{int(t.rows),int(t.columns/32)} && t.arrays[2].shape()==t.arrays[1].shape() &&
            t.arrays[3].shape()==mx::Shape{1} && t.arrays[0].dtype()==mx::uint32 && t.arrays[1].dtype()==mx::float16 &&
            t.arrays[2].dtype()==mx::float16 && t.arrays[3].dtype()==mx::uint32,"fixed GPU affine output span mismatch");
        for (const auto &out:t.arrays) {
            require(out.flags().row_contiguous && out.data_shared_ptr(),"fixed GPU affine target is not prepared");
            require(out.offset()>=0 && out.buffer_size()>=uint64_t(out.offset())*out.itemsize()+out.nbytes(),
                    "fixed GPU affine target storage too short");
            require(!raw_buffers.contains(out.buffer().ptr()) && output_buffers.insert(out.buffer().ptr()).second,
                    "fixed GPU affine source/target or target/target alias");
        }
    }
}
MTL::Library *fixed_library(mx::metal::Device &device) {
    return device.get_library("tc-gguf-fixed-affine-dependency-v1",{},[] {return std::string(R"metal(
        #include <metal_stdlib>
        using namespace metal;
        struct Params {uint groups,type,bits,block_bytes;};
        kernel void tc_gguf_fixed_clear(device atomic_uint *error [[buffer(0)]]) {
            atomic_store_explicit(error,0u,memory_order_relaxed);
        }
        kernel void tc_gguf_fixed_affine(device const uchar *raw [[buffer(0)]],
            device uint *codes [[buffer(1)]],device half *scales [[buffer(2)]],
            device half *biases [[buffer(3)]],device atomic_uint *error [[buffer(4)]],
            constant Params &p [[buffer(5)]],uint group [[thread_position_in_grid]]) {
            if (group>=p.groups) return;
            uint begin=group*p.block_bytes;
            ushort sb=ushort(raw[begin]) | (ushort(raw[begin+1])<<8);
            half scale=as_type<half>(sb);
            half bias;
            if (p.type==3) {
                ushort bb=ushort(raw[begin+2]) | (ushort(raw[begin+3])<<8);
                bias=as_type<half>(bb);
            } else bias=half(float(scale)*(p.type==8 ? -128.f : -8.f));
            float lo=float(bias),hi=float((1u<<p.bits)-1u)*float(scale)+lo;
            if (!isfinite(float(scale)) || !isfinite(float(bias)) || max(abs(lo),abs(hi))>65504.f)
                atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            scales[group]=scale;biases[group]=bias;
            for (uint word=0;word<p.bits;++word) {
                uint value=0;
                for (uint i=0;i<32/p.bits;++i) {
                    uint k=word*(32/p.bits)+i,code;
                    if (p.type==8) code=uint(raw[begin+2+k])^0x80u;
                    else {
                        uint q=uint(raw[begin+(p.type==3 ? 4 : 2)+(k%16)]);
                        code=k<16 ? (q&15u) : (q>>4);
                    }
                    value|=code<<(i*p.bits);
                }
                codes[group*p.bits+word]=value;
            }
        }
    )metal");});
}
void encode_fixed_batch(const std::vector<Tensor> &raw,
                        const std::vector<GgufGpuAffineFixedTarget *> &targets,mx::Stream stream) {
    auto &device=mx::metal::device(stream.device);
    auto *library=fixed_library(device);
    auto *pipeline=device.get_kernel("tc_gguf_fixed_affine",library);
    auto *clear=device.get_kernel("tc_gguf_fixed_clear",library);
    require(pipeline->maxTotalThreadsPerThreadgroup()>=256,"fixed GPU affine pipeline cannot use its declared threadgroup");
    auto &encoder=mx::metal::get_command_encoder(stream);
    for (size_t i=0;i<raw.size();++i) {
        auto &t=*targets[i];
        encoder.set_compute_pipeline_state(clear);
        encoder.set_output_array(t.arrays[3],0);
        encoder.dispatch_threads(MTL::Size(1,1,1),MTL::Size(1,1,1));
        struct Params {uint32_t groups,type,bits,block_bytes;};
        const Params params{uint32_t(t.rows*(t.columns/32)),t.type,t.type==8 ? 8u : 4u,t.type==8 ? 34u : t.type==2 ? 18u : 20u};
        encoder.set_compute_pipeline_state(pipeline);encoder.set_input_array(raw[i],0);
        for (size_t output=0;output<4;++output) encoder.set_output_array(t.arrays[output],int(output+1));
        encoder.set_bytes(params,5);
        encoder.dispatch_threads(MTL::Size(params.groups,1,1),MTL::Size(256,1,1));
    }
}
class FixedAffineDependency final : public mx::Primitive {
    std::vector<GgufGpuAffineFixedTarget> targets_;
  public:
    FixedAffineDependency(mx::Stream stream,const std::vector<GgufGpuAffineFixedTarget *> &targets)
        : mx::Primitive(stream) {for (const auto *target:targets) targets_.push_back(*target);}
    const char *name() const override {return "TcGgufFixedAffineDependencyV1";}
    // Never CSE two content tickets just because shapes/backings match.
    bool is_equivalent(const mx::Primitive &) const override {return false;}
    void eval_cpu(const std::vector<Tensor> &,std::vector<Tensor> &) override {
        throw std::invalid_argument("fixed affine dependency requires Metal GPU");
    }
    void eval_gpu(const std::vector<Tensor> &inputs,std::vector<Tensor> &outputs) override {
        require(inputs.size()==targets_.size() && outputs.size()==4*targets_.size(),"fixed affine dependency arity changed");
        std::vector<GgufGpuAffineFixedTarget> aliases;
        aliases.reserve(targets_.size());
        for (size_t i=0;i<targets_.size();++i) {
            for (size_t j=0;j<4;++j) outputs[i*4+j].copy_shared_buffer(targets_[i].arrays[j]);
            aliases.push_back({targets_[i].type,targets_[i].rows,targets_[i].columns,
                {outputs[i*4],outputs[i*4+1],outputs[i*4+2],outputs[i*4+3]}});
        }
        std::vector<GgufGpuAffineFixedTarget *> pointers;
        for (auto &alias:aliases) pointers.push_back(&alias);
        encode_fixed_batch(inputs,pointers,stream());
    }
};
} // namespace

void fill_gguf_gpu_affine_fixed(const std::vector<Tensor> &raw,
                              const std::vector<GgufGpuAffineFixedTarget *> &targets) {
    validate_fixed_batch(raw,targets);
    const auto stream=mx::default_stream(mx::Device(mx::Device::gpu));
    mx::synchronize(stream);
    encode_fixed_batch(raw,targets,stream);
    // Targets and raw arrays stay owned throughout the unretained-reference
    // command buffer; this method returns only after completion/error.
    mx::metal::get_command_encoder(stream).synchronize();
    for (const auto *target:targets)
        require(target->arrays[3].data<uint32_t>()[0]==0,"qe_decode_invalid: fixed GPU affine metadata/FP16 consumer overflow");
}

std::vector<Tensor> prepare_gguf_gpu_affine_fixed_dependency(const std::vector<Tensor> &raw,
                              const std::vector<GgufGpuAffineFixedTarget *> &targets) {
    validate_fixed_batch(raw,targets);
    std::vector<mx::Shape> shapes;std::vector<mx::Dtype> dtypes;
    for (const auto *target:targets) for (const auto &output:target->arrays) {
        shapes.push_back(output.shape());dtypes.push_back(output.dtype());
    }
    const auto stream=mx::default_stream(mx::Device(mx::Device::gpu));
    return Tensor::make_arrays(std::move(shapes),dtypes,std::make_shared<FixedAffineDependency>(stream,targets),raw);
}
void finish_gguf_gpu_affine_fixed_dependency(std::vector<Tensor> &outputs) {
    require(!outputs.empty() && outputs.size()%4==0,"fixed affine dependency output arity invalid");
    // Standalone callers also get a genuine completion fence. In the model,
    // every sibling is already available through the final QMM reader eval.
    mx::eval(outputs);
    bool valid=true;
    for (size_t i=3;i<outputs.size();i+=4) valid &= outputs[i].data<uint32_t>()[0]==0;
    for (auto &output:outputs) {output.detach();output.set_siblings({},0);}
    require(valid,"qe_decode_invalid: dependency GPU affine metadata/FP16 consumer overflow");
}
} // namespace tc::streaming
