#include "native_gate_up.hpp"

#import <Metal/Metal.h>

namespace tc::metal_graph {
namespace {

dense_gate_up::Layout layout(const MatrixView& matrix) {
    if (!matrix.storage || !matrix.storage.buffer().native_handle())
        throw std::invalid_argument("native paired gate/up matrix has no retained native buffer");
    return {matrix.rows,matrix.columns,matrix.row_stride_bytes,matrix.storage.size_bytes(),
        matrix.storage.offset_bytes(),matrix.storage.scalar_type(),
        reinterpret_cast<uintptr_t>(matrix.storage.buffer().native_handle())};
}

void same_device(const MatrixView& matrix, void* device) {
    const id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)matrix.storage.buffer().native_handle();
    if (!buffer || !device || (__bridge void*)buffer.device != device)
        throw std::invalid_argument("native paired gate/up operand belongs to another/empty Metal device");
}

} // namespace

NativeGateUp NativeGateUp::prepare(Backend& backend, Geometry geometry) {
    dense_ffn::validate_geometry(geometry);
    if (!backend.native_device())
        throw std::runtime_error("native paired gate/up has no Metal device");
    NativeGateUp prepared;
    prepared.geometry_ = geometry;
    prepared.device_ = backend.native_device();
    const auto library = backend.compile_library(dense_gate_up::source(geometry),
        "tc-native-paired-gate-up-h"+std::to_string(geometry.hidden)+
        "-c"+std::to_string(geometry.channels),{LanguageVersion::Metal4_0});
    prepared.paired_ = backend.prepare_pipeline(library,"tc_native_gate_up_bf16");
    if (prepared.paired_.thread_execution_width() != 32 ||
        prepared.paired_.max_threads_per_threadgroup() < 128)
        throw std::runtime_error("native paired gate/up MPP requires four 32-lane SIMD groups");
    return prepared;
}

void NativeGateUp::encode(CommandGraph& graph, const MatrixView& input,
    const MatrixView& gate, const MatrixView& up, const MatrixView& gate_output,
    const MatrixView& up_output, uint32_t first_channel) const {
    const auto checked = dense_gate_up::validate_bindings(geometry_,layout(input),layout(gate),
        layout(up),layout(gate_output),layout(up_output),first_channel);
    if (!paired_)
        throw std::invalid_argument("native paired gate/up was not prepared outside the execution path");
    for (const auto* matrix : {&input,&gate,&up,&gate_output,&up_output})
        same_device(*matrix,device_);
    CommandGraph draft;
    draft.dispatch(paired_,{input.storage,gate.storage,up.storage,gate_output.storage,up_output.storage},
        checked.parameters,{geometry_.channels/128,(size_t(input.rows)+31)/32,1},{128,1,1});
    graph.append_graph(std::move(draft));
}

} // namespace tc::metal_graph
