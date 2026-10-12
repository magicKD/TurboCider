#include "native_ffn.hpp"

#include <limits>
#include <string>

namespace tc::metal_graph {
namespace {

size_t used_bytes(const MatrixView& matrix) {
    if (!matrix.storage) throw std::invalid_argument("native FFN empty matrix storage");
    return dense_ffn::validate_layout({matrix.rows,matrix.columns,matrix.row_stride_bytes,
        matrix.storage.size_bytes(),matrix.storage.offset_bytes(),matrix.storage.scalar_type()});
}

void expect(const MatrixView& matrix, uint32_t rows, uint32_t columns,
            ScalarType type) {
    (void)used_bytes(matrix);
    if (matrix.rows != rows || matrix.columns != columns || matrix.storage.scalar_type() != type)
        throw std::invalid_argument("native FFN logical shape/dtype mismatch");
}

bool overlaps(const MatrixView& a, const MatrixView& b) {
    if (a.storage.buffer().native_handle() != b.storage.buffer().native_handle()) return false;
    const auto a_start = a.storage.offset_bytes(), b_start = b.storage.offset_bytes();
    const auto a_bytes = used_bytes(a), b_bytes = used_bytes(b);
    // Subtraction avoids adding an extent to an arbitrary imported offset.
    return dense_ffn::ranges_overlap(a_start,a_bytes,b_start,b_bytes);
}

void distinct_output(const MatrixView& output, const MatrixView& input) {
    if (overlaps(output,input))
        throw std::invalid_argument("native FFN writable view aliases a read operand");
}

void validate_first(dense_ffn::Geometry shape, uint32_t first) {
    if (first > shape.width-shape.channels)
        throw std::invalid_argument("native FFN channel slice exceeds physical source width");
}

void validate_gate(dense_ffn::Geometry shape, const MatrixView& input,
                   const MatrixView& gate, const MatrixView& up,
                   const MatrixView& hidden, uint32_t first) {
    validate_first(shape,first);
    expect(input,input.rows,shape.hidden,ScalarType::BF16);
    expect(gate,shape.width,shape.hidden,ScalarType::BF16);
    expect(up,shape.width,shape.hidden,ScalarType::BF16);
    expect(hidden,input.rows,shape.channels,ScalarType::BF16);
    distinct_output(hidden,input);
    distinct_output(hidden,gate);
    distinct_output(hidden,up);
}

void validate_down(dense_ffn::Geometry shape, const MatrixView& hidden,
                   const MatrixView& down, const MatrixView& output, uint32_t first) {
    validate_first(shape,first);
    expect(hidden,hidden.rows,shape.channels,ScalarType::BF16);
    expect(down,shape.hidden,shape.width,ScalarType::BF16);
    const auto type = output.storage.scalar_type();
    if (type != ScalarType::BF16 && type != ScalarType::FP32)
        throw std::invalid_argument("native FFN down output must be BF16 or explicit FP32 partial");
    expect(output,hidden.rows,shape.hidden,type);
    distinct_output(output,hidden);
    distinct_output(output,down);
}

uint32_t stride(const MatrixView& matrix) {
    return uint32_t(matrix.row_stride_bytes / scalar_bytes(matrix.storage.scalar_type()));
}

void validate_pipeline(const Pipeline& pipeline) {
    if (pipeline.thread_execution_width() != 32 || pipeline.max_threads_per_threadgroup() < 128)
        throw std::runtime_error("native dense FFN MPP requires four 32-lane SIMD groups");
}

} // namespace

NativeFfn NativeFfn::prepare(Backend& backend, dense_ffn::Geometry geometry) {
    return prepare_impl(backend,geometry,false);
}
NativeFfn NativeFfn::prepare_parallel(Backend& backend, dense_ffn::Geometry geometry) {
    dense_ffn::validate_geometry(geometry);
    if (geometry.channels >= geometry.width)
        throw std::invalid_argument("native parallel FFN requires a proper channel subset");
    return prepare_impl(backend,geometry,true);
}
NativeFfn NativeFfn::prepare_impl(Backend& backend, dense_ffn::Geometry geometry, bool parallel) {
    dense_ffn::validate_geometry(geometry);
    NativeFfn prepared;
    prepared.geometry_ = geometry;
    auto library = backend.compile_library(dense_ffn::source(geometry,parallel),
        "tc-native-dense-ffn-h"+std::to_string(geometry.hidden)+"-c"+std::to_string(geometry.channels)+
            (parallel ? "-parallel" : ""),
        {LanguageVersion::Metal4_0});
    prepared.gate_up_ = backend.prepare_pipeline(library,"tc_native_ffn_gate_up");
    prepared.down_bf16_ = backend.prepare_pipeline(library,"tc_native_ffn_down_bf16");
    prepared.down_f32_ = backend.prepare_pipeline(library,"tc_native_ffn_down_f32");
    validate_pipeline(prepared.gate_up_);
    validate_pipeline(prepared.down_bf16_);
    validate_pipeline(prepared.down_f32_);
    if (parallel) {
        prepared.join_ = backend.prepare_pipeline(library,"tc_native_ffn_join_partials");
        validate_pipeline(prepared.join_);
    }
    return prepared;
}

void NativeFfn::encode_join_partials(CommandGraph& graph,const MatrixView& gpu,
    const MatrixView& ane,const MatrixView& output) const {
    if (!join_) throw std::invalid_argument("native FFN join was not prepared outside the execution path");
    auto layout=[](const MatrixView& m) {
        if (!m.storage) throw std::invalid_argument("native FFN join empty matrix storage");
        return dense_ffn::Layout{m.rows,m.columns,m.row_stride_bytes,m.storage.size_bytes(),
            m.storage.offset_bytes(),m.storage.scalar_type()};
    };
    const auto checked=dense_ffn::validate_join(geometry_.hidden,layout(gpu),layout(ane),layout(output));
    distinct_output(output,gpu); distinct_output(output,ane); distinct_output(gpu,ane);
    graph.dispatch(join_,{gpu.storage,ane.storage,output.storage},checked.parameters,
        {geometry_.hidden/128,gpu.rows,1},{128,1,1});
}

void NativeFfn::encode_gate_up(CommandGraph& graph, const MatrixView& input,
    const MatrixView& gate, const MatrixView& up, const MatrixView& hidden, uint32_t first) const {
    validate_gate(geometry_,input,gate,up,hidden,first);
    dense_ffn::Parameters p{input.rows,first,stride(input),stride(gate),stride(up),stride(hidden),0,0};
    graph.dispatch(gate_up_,{input.storage,gate.storage,up.storage,hidden.storage},p,
                   {geometry_.channels/128,(size_t(input.rows)+31)/32,1},{128,1,1});
}

void NativeFfn::encode_down(CommandGraph& graph, const MatrixView& hidden,
    const MatrixView& down, const MatrixView& output, uint32_t first) const {
    validate_down(geometry_,hidden,down,output,first);
    dense_ffn::Parameters p{hidden.rows,first,0,0,0,stride(hidden),stride(down),stride(output)};
    const auto& pipeline = output.storage.scalar_type() == ScalarType::FP32 ? down_f32_ : down_bf16_;
    graph.dispatch(pipeline,{hidden.storage,down.storage,output.storage},p,
                   {geometry_.hidden/128,(size_t(hidden.rows)+31)/32,1},{128,1,1});
}

void NativeFfn::encode(CommandGraph& graph, const FfnBindings& bindings) const {
    validate_gate(geometry_,bindings.input,bindings.gate,bindings.up,
                  bindings.hidden,bindings.first_channel);
    validate_down(geometry_,bindings.hidden,bindings.down,bindings.output,bindings.first_channel);
    // A down weight aliasing the hidden scratch would be destroyed by step 1;
    // checking the full graph prevents this otherwise valid per-dispatch alias.
    distinct_output(bindings.hidden,bindings.down);
    distinct_output(bindings.output,bindings.input);
    distinct_output(bindings.output,bindings.gate);
    distinct_output(bindings.output,bindings.up);
    encode_gate_up(graph,bindings.input,bindings.gate,bindings.up,bindings.hidden,bindings.first_channel);
    encode_down(graph,bindings.hidden,bindings.down,bindings.output,bindings.first_channel);
}

} // namespace tc::metal_graph
