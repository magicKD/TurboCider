#include "native_attention.hpp"
#include "kernels/dense_attention.hpp"

#import <Metal/Metal.h>

namespace tc::metal_graph {
namespace {
dense_attention::Layout layout(const HeadView& view) {
    if (!view.storage || !view.storage.buffer().native_handle())
        throw std::invalid_argument("native SDPA has no retained native head buffer");
    return {view.heads,view.tokens,view.dimension,view.token_stride_bytes,view.head_stride_bytes,
            view.storage.size_bytes(),view.storage.offset_bytes(),view.storage.scalar_type()};
}
void reject_alias(const HeadView& output,size_t output_bytes,const HeadView& source,size_t source_bytes) {
    // Repeated imports of the same native allocation are detected. Different
    // external bytesNoCopy wrappers require caller-owned storage provenance.
    if (output.storage.buffer().native_handle()==source.storage.buffer().native_handle() &&
        dense_attention::ranges_overlap(output.storage.offset_bytes(),output_bytes,
                                        source.storage.offset_bytes(),source_bytes))
        throw std::invalid_argument("native SDPA output aliases an immutable Q/K/V source");
}
void validate_pipeline(const Pipeline& pipeline) {
    if (pipeline.thread_execution_width()!=32 || pipeline.max_threads_per_threadgroup()<128)
        throw std::runtime_error("native SDPA Steel requires four 32-lane SIMD groups");
}
}

NativeAttention NativeAttention::prepare(Backend& backend,AttentionGeometry geometry) {
    dense_attention::validate_geometry(geometry);
    id<MTLDevice> device=(__bridge id<MTLDevice>)backend.native_device();
    if (!device || device.maxThreadgroupMemoryLength<(geometry.dimension==80?size_t((32*88+80*24)*4):dense_attention::threadgroup_bytes))
        throw std::runtime_error("native SDPA D128 requires 14848 bytes of static threadgroup memory");
    NativeAttention prepared;
    prepared.geometry_=geometry;
    auto library=backend.compile_library(dense_attention::source(),"tc-native-steel-sdpa-d128",
                                         {LanguageVersion::Metal4_0});
    prepared.noncausal_=backend.prepare_pipeline(library,geometry.dimension==80?"tc_native_sdpa_d80_f32":"tc_native_sdpa_d128");
    prepared.causal_=backend.prepare_pipeline(library,geometry.dimension==80?"tc_native_sdpa_d80_f32_causal":"tc_native_sdpa_d128_causal");
    validate_pipeline(prepared.noncausal_); validate_pipeline(prepared.causal_);
    return prepared;
}

void NativeAttention::encode(CommandGraph& graph,const AttentionBindings& bindings) const {
    const auto checked=dense_attention::validate_bindings(geometry_,layout(bindings.query),layout(bindings.key),
        layout(bindings.value),layout(bindings.output),bindings.options);
    reject_alias(bindings.output,checked.output_bytes,bindings.query,checked.query_bytes);
    reject_alias(bindings.output,checked.output_bytes,bindings.key,checked.key_bytes);
    reject_alias(bindings.output,checked.output_bytes,bindings.value,checked.value_bytes);
    graph.dispatch(bindings.options.causal?causal_:noncausal_,
        {bindings.query.storage,bindings.key.storage,bindings.value.storage,bindings.output.storage},
        checked.parameters,{(size_t(bindings.query.tokens)+31)/32,geometry_.heads,1},{32,4,1});
}

} // namespace tc::metal_graph
