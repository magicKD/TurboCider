#include "native_lora.hpp"

#import <Metal/Metal.h>

namespace tc::metal_graph {
namespace {
native_lora::Layout layout(const MatrixView& matrix) {
    if(!matrix.storage || !matrix.storage.buffer().native_handle())
        throw std::invalid_argument("native LoRA matrix has no retained native buffer");
    return {matrix.rows,matrix.columns,matrix.row_stride_bytes,matrix.storage.size_bytes(),
        matrix.storage.offset_bytes(),matrix.storage.scalar_type(),
        reinterpret_cast<uintptr_t>(matrix.storage.buffer().native_handle())};
}
void same_device(const MatrixView& matrix,void* device) {
    const id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)matrix.storage.buffer().native_handle();
    if(!buffer || !device || (__bridge void*)buffer.device!=device)
        throw std::invalid_argument("native LoRA operand belongs to another/empty Metal device");
}
void check_pipeline(const Pipeline& pipeline,size_t threads) {
    if(pipeline.thread_execution_width()!=32 || pipeline.max_threads_per_threadgroup()<threads)
        throw std::runtime_error("native LoRA requires checked 32-lane SIMD/threadgroup capacity");
}
}

NativeLoRA NativeLoRA::prepare(Backend& backend,Geometry geometry,native_lora::KernelVariant variant) {
    native_lora::validate_variant(geometry,variant);
    const id<MTLDevice> device=(__bridge id<MTLDevice>)backend.native_device();
    const auto needed=variant==native_lora::KernelVariant::SgMmaR128?
        native_lora::sg_mma_threadgroup_bytes:native_lora::scalar_threadgroup_bytes;
    if(!device || device.maxThreadgroupMemoryLength<needed)
        throw std::runtime_error("native LoRA shared FP32 tiles exceed device capability");
    NativeLoRA result;result.geometry_=geometry;result.variant_=variant;result.device_=backend.native_device();
    const auto library=backend.compile_library(native_lora::source(),"tc-native-lora-fp32-tiled8x32-v1",
        {LanguageVersion::Metal4_0});
    result.low_=backend.prepare_pipeline(library,"tc_native_lora_low");
    result.apply_=backend.prepare_pipeline(library,"tc_native_lora_apply");
    result.copy_=backend.prepare_pipeline(library,"tc_native_lora_copy");
    result.bias_=backend.prepare_pipeline(library,"tc_native_lora_bias");
    result.silu_up_=backend.prepare_pipeline(library,"tc_native_lora_silu_up");
    check_pipeline(result.low_,128);check_pipeline(result.apply_,128);
    check_pipeline(result.copy_,256);check_pipeline(result.bias_,256);check_pipeline(result.silu_up_,256);
    if(variant==native_lora::KernelVariant::SgMmaR128) {
        // Explicit opt-in: failure propagates as capability rejection. The
        // unchanged scalar/default library never parses the new candidate.
        const auto candidate=backend.compile_library(native_lora::sg_mma_source(),
            "tc-native-lora-fp32-sgmma8x8-r128-scalar-tail-v2",{LanguageVersion::Metal4_0});
        result.low_sgmma_=backend.prepare_pipeline(candidate,"tc_native_lora_low_sgmma");
        result.apply_sgmma_=backend.prepare_pipeline(candidate,"tc_native_lora_apply_sgmma");
        for(const auto* p:{&result.low_sgmma_,&result.apply_sgmma_})
            native_lora::validate_capability(variant,device.maxThreadgroupMemoryLength,
                uint32_t(p->thread_execution_width()),p->max_threads_per_threadgroup());
    }
    return result;
}

void NativeLoRA::encode_ordered_corrections(CommandGraph& graph,const MatrixView& input,
    std::span<const OrderedLoRATerm> terms,const MatrixView& base,const MatrixView& output,
    const MatrixView& low,std::optional<MatrixView> bias) const {
    if(terms.size()>64)throw std::invalid_argument("native LoRA ordered term count exceeds graph bound");
    std::vector<native_lora::TermLayout> metadata;
    metadata.reserve(terms.size());
    for(const auto& term:terms)metadata.push_back({layout(term.a),layout(term.b),term.effective_scale,
        term.first_output,term.output_count});
    const auto checked=native_lora::validate_bindings(geometry_,layout(input),metadata,layout(base),layout(output),
        terms.empty()?native_lora::Layout{}:layout(low),bias?std::optional{layout(*bias)}:std::nullopt);
    if(terms.empty())return; // No PSO, workspace, compilation or dispatch required.
    if(!low_ || !apply_ || !copy_ || !bias_)
        throw std::runtime_error("native LoRA correction bundle was not cold prepared");
    if(variant_==native_lora::KernelVariant::SgMmaR128 && (!low_sgmma_ || !apply_sgmma_))
        throw std::runtime_error("native LoRA SgMmaR128 bundle was not cold prepared");
    for(const auto* matrix:{&input,&base,&output,&low})same_device(*matrix,device_);
    for(const auto& term:terms) {same_device(term.a,device_);same_device(term.b,device_);}
    if(bias)same_device(*bias,device_);
    CommandGraph draft;
    if(checked.copy_base)draft.dispatch(copy_,{base.storage,output.storage},checked.copy,
        {(size_t(output.columns)+31)/32,(size_t(input.rows)+7)/8,1},{32,8,1});
    for(size_t i=0;i<terms.size();++i) {
        const auto& term=terms[i];const auto& parameters=checked.terms[i];
        const bool candidate=variant_==native_lora::KernelVariant::SgMmaR128 &&
            native_lora::sg_mma_eligible(parameters);
        draft.dispatch(candidate?low_sgmma_:low_,{input.storage,term.a.storage,low.storage},parameters.low,
            {(size_t(term.a.rows)+31)/32,(size_t(input.rows)+7)/8,1},{128,1,1});
        draft.dispatch(candidate?apply_sgmma_:apply_,{low.storage,term.b.storage,output.storage},parameters.apply,
            {(size_t(term.output_count)+31)/32,(size_t(input.rows)+7)/8,1},{128,1,1});
    }
    if(bias)draft.dispatch(bias_,{output.storage,bias->storage},checked.bias,
        {(size_t(output.columns)+31)/32,(size_t(input.rows)+7)/8,1},{32,8,1});
    graph.append_graph(std::move(draft));
}

ValidatedLoRAProjection NativeLoRA::validate_projection(const MatrixView& input,
    const OrderedLoRATerm& term,const MatrixView& output,const MatrixView& low) const {
    const auto checked=native_lora::validate_projection(geometry_,variant_,layout(input),
        {layout(term.a),layout(term.b),term.effective_scale,term.first_output,term.output_count},
        layout(output),layout(low));
    ValidatedLoRAProjection projection(input,term,output,low,checked,geometry_,variant_,device_);
    validate_projection_descriptor(projection);
    return projection;
}

void NativeLoRA::validate_projection_descriptor(const ValidatedLoRAProjection& projection) const {
    // Descriptor fields are private and retained: a caller cannot replace A/B,
    // a physical pitch/range, or only the low phase's eligibility after checking.
    // Accept another prepared bundle only when geometry, variant and device are
    // identical; its pipeline arithmetic is the same fixed source recipe.
    if(projection.geometry_.input_columns!=geometry_.input_columns ||
       projection.geometry_.output_columns!=geometry_.output_columns ||
       projection.geometry_.max_rank!=geometry_.max_rank || projection.bundle_variant_!=variant_ ||
       !device_ || projection.device_!=device_)
        throw std::invalid_argument("native LoRA split descriptor belongs to another bundle/device");
    if(!low_ || !apply_)
        throw std::runtime_error("native LoRA split low/apply bundle was not cold prepared");
    if(variant_==native_lora::KernelVariant::SgMmaR128 && (!low_sgmma_ || !apply_sgmma_))
        throw std::runtime_error("native LoRA split SgMmaR128 bundle was not cold prepared");
    for(const auto* matrix:{&projection.input_,&projection.term_.a,&projection.term_.b,
                           &projection.output_,&projection.low_})same_device(*matrix,device_);
}

void NativeLoRA::encode_low(CommandGraph& graph,const ValidatedLoRAProjection& projection) const {
    validate_projection_descriptor(projection);
    const bool candidate=projection.selected_variant()==native_lora::KernelVariant::SgMmaR128;
    CommandGraph draft;
    draft.dispatch(candidate?low_sgmma_:low_,
        {projection.input_.storage,projection.term_.a.storage,projection.low_.storage},
        projection.checked_.term.low,
        {(size_t(projection.term_.a.rows)+31)/32,(size_t(projection.input_.rows)+7)/8,1},{128,1,1});
    graph.append_graph(std::move(draft));
}

void NativeLoRA::encode_apply_inplace(CommandGraph& graph,const ValidatedLoRAProjection& projection) const {
    validate_projection_descriptor(projection);
    const bool candidate=projection.selected_variant()==native_lora::KernelVariant::SgMmaR128;
    CommandGraph draft;
    draft.dispatch(candidate?apply_sgmma_:apply_,
        {projection.low_.storage,projection.term_.b.storage,projection.output_.storage},
        projection.checked_.term.apply,
        {(size_t(projection.term_.output_count)+31)/32,(size_t(projection.input_.rows)+7)/8,1},{128,1,1});
    graph.append_graph(std::move(draft));
}

void NativeLoRA::encode_silu_up(CommandGraph& graph,const MatrixView& gate,
    const MatrixView& up,const MatrixView& hidden) const {
    const auto parameters=native_lora::validate_silu_up(geometry_,layout(gate),layout(up),layout(hidden));
    if(!silu_up_)throw std::runtime_error("native LoRA SiLU/up bundle was not cold prepared");
    for(const auto* matrix:{&gate,&up,&hidden})same_device(*matrix,device_);
    CommandGraph draft;
    draft.dispatch(silu_up_,{gate.storage,up.storage,hidden.storage},parameters,
        {(size_t(gate.columns)+31)/32,(size_t(gate.rows)+7)/8,1},{32,8,1});
    graph.append_graph(std::move(draft));
}

} // namespace tc::metal_graph
