#include "native_linear.hpp"

namespace tc::metal_graph {
namespace {

dense_linear::Layout layout(const MatrixView& matrix) {
    if (!matrix.storage || !matrix.storage.buffer().native_handle())
        throw std::invalid_argument("native linear matrix has no retained native buffer");
    return {matrix.rows,matrix.columns,matrix.row_stride_bytes,
            matrix.storage.size_bytes(),matrix.storage.offset_bytes(),matrix.storage.scalar_type()};
}

void reject_alias(const MatrixView& output, size_t output_bytes,
                  const MatrixView& source, size_t source_bytes) {
    // Repeated imports of the same MTLBuffer are caught by native identity.
    // Distinct external bytesNoCopy wrappers require caller storage provenance;
    // this operator cannot infer physical aliases from different handles.
    if (output.storage.buffer().native_handle() == source.storage.buffer().native_handle() &&
        dense_linear::ranges_overlap(output.storage.offset_bytes(),output_bytes,
                                     source.storage.offset_bytes(),source_bytes))
        throw std::invalid_argument("native linear output aliases an immutable input or weight");
}

void validate_pipeline(const Pipeline& pipeline) {
    if (pipeline.thread_execution_width() != 32 || pipeline.max_threads_per_threadgroup() < 128)
        throw std::runtime_error("native dense linear MPP requires four 32-lane SIMD groups");
}

} // namespace

NativeLinear NativeLinear::prepare(Backend& backend, dense_linear::Geometry geometry) {
    dense_linear::validate_geometry(geometry);
    NativeLinear prepared;
    prepared.geometry_ = geometry;
    auto library = backend.compile_library(dense_linear::source(geometry),
        "tc-native-dense-linear-k"+std::to_string(geometry.input_columns)+
        "-n"+std::to_string(geometry.output_columns),{LanguageVersion::Metal4_0});
    prepared.bf16_ = backend.prepare_pipeline(library,"tc_native_linear_bf16");
    prepared.f32_ = backend.prepare_pipeline(library,"tc_native_linear_f32");
    prepared.bias_bf16_ = backend.prepare_pipeline(library,"tc_native_linear_bias_bf16");
    prepared.bias_f32_ = backend.prepare_pipeline(library,"tc_native_linear_bias_f32");
    validate_pipeline(prepared.bf16_);
    validate_pipeline(prepared.f32_);
    validate_pipeline(prepared.bias_bf16_);
    validate_pipeline(prepared.bias_f32_);
    return prepared;
}

void NativeLinear::encode(CommandGraph& graph, const LinearBindings& bindings) const {
    // Validate every extent and alias before appending the single compute step.
    const auto checked = dense_linear::validate_bindings(geometry_,layout(bindings.input),
        layout(bindings.weight),layout(bindings.output),
        bindings.bias ? std::optional{layout(*bindings.bias)} : std::nullopt);
    reject_alias(bindings.output,checked.output_bytes,bindings.input,checked.input_bytes);
    reject_alias(bindings.output,checked.output_bytes,bindings.weight,checked.weight_bytes);
    if (bindings.bias) reject_alias(bindings.output,checked.output_bytes,*bindings.bias,checked.bias_bytes);
    const bool f32 = bindings.output.storage.scalar_type() == ScalarType::FP32;
    const auto& pipeline = bindings.bias ? (f32 ? bias_f32_ : bias_bf16_) : (f32 ? f32_ : bf16_);
    std::vector<BufferView> sources{bindings.input.storage,bindings.weight.storage,bindings.output.storage};
    if (bindings.bias) sources.push_back(bindings.bias->storage);
    graph.dispatch(pipeline,std::move(sources),checked.parameters,
        {(size_t(geometry_.output_columns)+127)/128,(size_t(bindings.input.rows)+31)/32,1},{128,1,1});
}

} // namespace tc::metal_graph
