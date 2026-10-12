#include "native_qwen_ops.hpp"

#include <initializer_list>

namespace tc::metal_graph {
namespace {
struct UsedView { const BufferView* storage; size_t bytes; };

UsedView used(const MatrixView& matrix) {
    const auto type = matrix.storage.scalar_type();
    if (!matrix.storage || (type != ScalarType::BF16 && type != ScalarType::FP32))
        throw std::invalid_argument("native Qwen matrix requires retained BF16/FP32 buffer");
    return {&matrix.storage, qwen_ops::matrix_extent(matrix.rows, matrix.columns,
        matrix.row_stride_bytes, scalar_bytes(type), matrix.storage.offset_bytes(),
        matrix.storage.size_bytes())};
}
UsedView used(const HeadView& head) {
    if (!head.storage || head.storage.scalar_type() != ScalarType::BF16)
        throw std::invalid_argument("native Qwen heads require retained BF16 buffer");
    return {&head.storage, qwen_ops::head_extent(head.heads, head.tokens, head.dimension,
        head.token_stride_bytes, head.head_stride_bytes, 2, head.storage.offset_bytes(),
        head.storage.size_bytes())};
}
void expect(const MatrixView& matrix, uint32_t rows, uint32_t columns,
            ScalarType type = ScalarType::BF16) {
    (void)used(matrix);
    if (matrix.rows != rows || matrix.columns != columns || matrix.storage.scalar_type() != type)
        throw std::invalid_argument("native Qwen matrix logical shape/dtype mismatch");
}
void expect(const HeadView& head, qwen_ops::Geometry geometry, uint32_t rows) {
    (void)used(head);
    if (head.heads != geometry.heads || head.tokens != rows || head.dimension != geometry.head_dim)
        throw std::invalid_argument("native Qwen head-major shape mismatch");
}
void broadcast(const MatrixView& matrix, uint32_t rows, uint32_t hidden) {
    expect(matrix, matrix.rows, hidden);
    if (matrix.rows != 1 && matrix.rows != rows)
        throw std::invalid_argument("native Qwen modulation must have one row or match input rows");
}
void distinct(UsedView output, std::initializer_list<UsedView> reads) {
    for (const auto& input : reads)
        if (output.storage->buffer().native_handle() == input.storage->buffer().native_handle() &&
            qwen_ops::ranges_overlap(output.storage->offset_bytes(), output.bytes,
                                    input.storage->offset_bytes(), input.bytes))
            throw std::invalid_argument("native Qwen writable view aliases another operand");
}
uint32_t stride(const MatrixView& matrix) {
    return uint32_t(matrix.row_stride_bytes / scalar_bytes(matrix.storage.scalar_type()));
}
void pipeline_ready(const Pipeline& pipeline) {
    if (!pipeline) throw std::logic_error("native Qwen operators were not prepared");
}
Grid point_groups(qwen_ops::Geometry geometry, uint32_t rows) {
    return {(size_t(geometry.hidden) + 255) / 256, rows, 1};
}
} // namespace

NativeQwenOps NativeQwenOps::prepare(Backend& backend, qwen_ops::Geometry geometry) {
    qwen_ops::validate_geometry(geometry);
    NativeQwenOps result;
    result.geometry_ = geometry;
    auto library = backend.compile_library(qwen_ops::source(geometry),
        "tc-native-qwen-ops-h" + std::to_string(geometry.hidden));
    result.norm_ = backend.prepare_pipeline(library, "tc_qwen_layer_norm");
    result.modulation_ = backend.prepare_pipeline(library, "tc_qwen_modulation");
    result.residual_ = backend.prepare_pipeline(library, "tc_qwen_residual");
    result.qk_rope_ = backend.prepare_pipeline(library, "tc_qwen_qk_norm_rope");
    result.pack_ = backend.prepare_pipeline(library, "tc_qwen_pack_heads");
    result.unpack_ = backend.prepare_pipeline(library, "tc_qwen_unpack_heads");
    result.copy_heads_ = backend.prepare_pipeline(library, "tc_qwen_copy_heads");
    result.silu_ = backend.prepare_pipeline(library, "tc_qwen_silu");
    result.gelu_ = backend.prepare_pipeline(library, "tc_qwen_gelu");
    result.condition_norm_ = backend.prepare_pipeline(library, "tc_qwen_condition_norm");
    result.expand_modulation_ = backend.prepare_pipeline(library, "tc_qwen_expand_modulation");
    for (const auto* pipeline : {&result.norm_, &result.modulation_, &result.residual_,
                                &result.qk_rope_, &result.pack_, &result.unpack_, &result.copy_heads_,
                                &result.silu_, &result.gelu_, &result.condition_norm_, &result.expand_modulation_})
        if (pipeline->thread_execution_width() != 32 ||
            pipeline->max_threads_per_threadgroup() < (pipeline == &result.qk_rope_ ? 32 : 256))
            throw std::runtime_error("native Qwen kernel SIMD/threadgroup capability unavailable");
    return result;
}

void NativeQwenOps::validate_layer_norm(qwen_ops::Geometry geometry, const MatrixView& input,
    const MatrixView& output, float epsilon) {
    qwen_ops::validate_geometry(geometry);
    qwen_ops::validate_epsilon(epsilon);
    expect(input, input.rows, geometry.hidden);
    expect(output, input.rows, geometry.hidden);
    distinct(used(output), {used(input)});
}

void NativeQwenOps::validate_norm_modulation(qwen_ops::Geometry geometry, const MatrixView& input,
    const MatrixView& scale, const MatrixView& output, float epsilon, const MatrixView* shift) {
    validate_layer_norm(geometry, input, output, epsilon);
    broadcast(scale, input.rows, geometry.hidden);
    if (shift) broadcast(*shift, input.rows, geometry.hidden);
    distinct(used(output), {used(scale), used(shift ? *shift : scale)});
}

void NativeQwenOps::validate_modulation(qwen_ops::Geometry geometry, const MatrixView& input,
    const MatrixView& scale, const MatrixView& output, const MatrixView* shift) {
    qwen_ops::validate_geometry(geometry);
    expect(input, input.rows, geometry.hidden);
    expect(output, input.rows, geometry.hidden);
    broadcast(scale, input.rows, geometry.hidden);
    if (shift) broadcast(*shift, input.rows, geometry.hidden);
    distinct(used(output), {used(input), used(scale), used(shift ? *shift : scale)});
}

void NativeQwenOps::validate_gated_residual(qwen_ops::Geometry geometry, const MatrixView& residual,
    const MatrixView& branch, const MatrixView& gate, const MatrixView& output, bool gate_is_tanh) {
    (void)gate_is_tanh;
    qwen_ops::validate_geometry(geometry);
    expect(residual, residual.rows, geometry.hidden);
    expect(branch, residual.rows, geometry.hidden);
    expect(output, residual.rows, geometry.hidden);
    broadcast(gate, residual.rows, geometry.hidden);
    distinct(used(output), {used(residual), used(branch), used(gate)});
}

void NativeQwenOps::validate_qk_norm_rope(qwen_ops::Geometry geometry, const MatrixView& q,
    const MatrixView& k, const MatrixView& q_weight, const MatrixView& k_weight,
    const MatrixView& cosine, const MatrixView& sine, const HeadView& q_out,
    const HeadView& k_out, float epsilon) {
    qwen_ops::validate_geometry(geometry);
    qwen_ops::validate_epsilon(epsilon);
    expect(q, q.rows, geometry.hidden);
    expect(k, q.rows, geometry.hidden);
    expect(q_weight, 1, geometry.head_dim);
    expect(k_weight, 1, geometry.head_dim);
    expect(cosine, q.rows, geometry.head_dim / 2, ScalarType::FP32);
    expect(sine, q.rows, geometry.head_dim / 2, ScalarType::FP32);
    expect(q_out, geometry, q.rows);
    expect(k_out, geometry, q.rows);
    if (q.rows > UINT32_MAX / geometry.heads)
        throw std::invalid_argument("native Qwen QK task count exceeds uint indexing");
    const auto reads = {used(q), used(k), used(q_weight), used(k_weight), used(cosine), used(sine)};
    distinct(used(q_out), reads); distinct(used(k_out), reads);
    distinct(used(q_out), {used(k_out)});
}

void NativeQwenOps::validate_pack_heads(qwen_ops::Geometry geometry, const MatrixView& row_major,
    const HeadView& head_major) {
    qwen_ops::validate_geometry(geometry);
    expect(row_major, row_major.rows, geometry.hidden);
    expect(head_major, geometry, row_major.rows);
    distinct(used(head_major), {used(row_major)});
}

void NativeQwenOps::validate_unpack_heads(qwen_ops::Geometry geometry, const HeadView& head_major,
    const MatrixView& row_major) {
    validate_pack_heads(geometry, row_major, head_major);
}

void NativeQwenOps::validate_copy_heads(qwen_ops::Geometry geometry, const HeadView& source,
    const HeadView& destination) {
    qwen_ops::validate_geometry(geometry);
    expect(source, geometry, source.tokens);
    expect(destination, geometry, source.tokens);
    distinct(used(destination), {used(source)});
}

void NativeQwenOps::validate_silu(qwen_ops::Geometry geometry, const MatrixView& input,
    const MatrixView& output) {
    qwen_ops::validate_geometry(geometry);
    expect(input,input.rows,geometry.hidden);
    expect(output,input.rows,geometry.hidden);
    distinct(used(output),{used(input)});
}

void NativeQwenOps::validate_gelu(qwen_ops::Geometry geometry, const MatrixView& input,
    const MatrixView& output) {
    validate_silu(geometry,input,output);
}

void NativeQwenOps::validate_condition_norm(qwen_ops::Geometry geometry, const MatrixView& text,
    const MatrixView& weight, const MatrixView& output, float epsilon) {
    validate_silu(geometry,text,output);
    qwen_ops::validate_epsilon(epsilon);
    expect(weight,1,geometry.hidden);
    distinct(used(output),{used(weight)});
}

void NativeQwenOps::validate_expand_modulation(qwen_ops::Geometry geometry, const MatrixView& source,
    const std::array<MatrixView,4>& outputs, uint32_t prefix_rows) {
    qwen_ops::validate_geometry(geometry);
    const uint32_t rows = outputs[0].rows;
    qwen_ops::validate_modulation_rows(rows,prefix_rows);
    expect(source,2,4*geometry.hidden);
    // Validate every destination before recording any write; two overlapping
    // outputs are invalid even if their source row values happen to be equal.
    for (const auto& output : outputs) expect(output,rows,geometry.hidden);
    for (size_t i=0; i<outputs.size(); ++i) {
        distinct(used(outputs[i]),{used(source)});
        for (size_t j=0; j<i; ++j) distinct(used(outputs[i]),{used(outputs[j])});
    }
}

void NativeQwenOps::encode_layer_norm(CommandGraph& graph, const MatrixView& input,
    const MatrixView& output, float epsilon) const {
    pipeline_ready(norm_);
    validate_layer_norm(geometry_, input, output, epsilon);
    qwen_ops::NormParameters p{input.rows, stride(input), 0, 0, stride(output), 0, 0, 0, epsilon};
    // Flags prohibit accesses to the placeholder scale/shift bindings.
    graph.dispatch(norm_, {input.storage, input.storage, input.storage, output.storage},
                   p, {input.rows, 1, 1}, {256, 1, 1});
}

void NativeQwenOps::encode_norm_modulation(CommandGraph& graph, const MatrixView& input,
    const MatrixView& scale, const MatrixView& output, float epsilon, const MatrixView* shift) const {
    pipeline_ready(norm_);
    validate_norm_modulation(geometry_, input, scale, output, epsilon, shift);
    const auto& shift_binding = shift ? *shift : scale;
    qwen_ops::NormParameters p{input.rows, stride(input), stride(scale), stride(shift_binding),
        stride(output), scale.rows, shift_binding.rows, shift ? 3U : 1U, epsilon};
    graph.dispatch(norm_, {input.storage, scale.storage, shift_binding.storage, output.storage},
                   p, {input.rows, 1, 1}, {256, 1, 1});
}

void NativeQwenOps::encode_modulation(CommandGraph& graph, const MatrixView& input,
    const MatrixView& scale, const MatrixView& output, const MatrixView* shift) const {
    pipeline_ready(modulation_);
    validate_modulation(geometry_, input, scale, output, shift);
    const auto& shift_binding = shift ? *shift : scale;
    qwen_ops::PointParameters p{input.rows, stride(input), 0, stride(scale), stride(output),
        scale.rows, stride(shift_binding), shift_binding.rows, shift ? 1U : 0U};
    graph.dispatch(modulation_, {input.storage, scale.storage, shift_binding.storage, output.storage},
                   p, point_groups(geometry_, input.rows), {256, 1, 1});
}

void NativeQwenOps::encode_gated_residual(CommandGraph& graph, const MatrixView& residual,
    const MatrixView& branch, const MatrixView& gate, const MatrixView& output, bool gate_is_tanh) const {
    pipeline_ready(residual_);
    validate_gated_residual(geometry_, residual, branch, gate, output, gate_is_tanh);
    qwen_ops::PointParameters p{residual.rows, stride(residual), stride(branch), stride(gate),
        stride(output), gate.rows, 0, 0, gate_is_tanh ? 1U : 0U};
    graph.dispatch(residual_, {residual.storage, branch.storage, gate.storage, output.storage},
                   p, point_groups(geometry_, residual.rows), {256, 1, 1});
}

void NativeQwenOps::encode_qk_norm_rope(CommandGraph& graph, const MatrixView& q,
    const MatrixView& k, const MatrixView& q_weight, const MatrixView& k_weight,
    const MatrixView& cosine, const MatrixView& sine, const HeadView& q_out,
    const HeadView& k_out, float epsilon) const {
    pipeline_ready(qk_rope_);
    validate_qk_norm_rope(geometry_, q, k, q_weight, k_weight, cosine, sine, q_out, k_out, epsilon);
    qwen_ops::QkParameters p{q.rows, stride(q), stride(k), stride(cosine), stride(sine),
        uint32_t(q_out.token_stride_bytes / 2), uint32_t(q_out.head_stride_bytes / 2),
        uint32_t(k_out.token_stride_bytes / 2), uint32_t(k_out.head_stride_bytes / 2), epsilon};
    graph.dispatch(qk_rope_, {q.storage, k.storage, q_weight.storage, k_weight.storage,
                             cosine.storage, sine.storage, q_out.storage, k_out.storage},
                   p, {size_t(q.rows) * geometry_.heads, 1, 1}, {32, 1, 1});
}

void NativeQwenOps::encode_pack_heads(CommandGraph& graph, const MatrixView& row_major,
    const HeadView& head_major) const {
    pipeline_ready(pack_);
    validate_pack_heads(geometry_, row_major, head_major);
    qwen_ops::PackParameters p{row_major.rows, stride(row_major),
        uint32_t(head_major.token_stride_bytes / 2), uint32_t(head_major.head_stride_bytes / 2)};
    graph.dispatch(pack_, {row_major.storage, head_major.storage}, p,
                   point_groups(geometry_, row_major.rows), {256, 1, 1});
}

void NativeQwenOps::encode_unpack_heads(CommandGraph& graph, const HeadView& head_major,
    const MatrixView& row_major) const {
    pipeline_ready(unpack_);
    validate_unpack_heads(geometry_, head_major, row_major);
    qwen_ops::PackParameters p{row_major.rows, stride(row_major),
        uint32_t(head_major.token_stride_bytes / 2), uint32_t(head_major.head_stride_bytes / 2)};
    graph.dispatch(unpack_, {head_major.storage, row_major.storage}, p,
                   point_groups(geometry_, row_major.rows), {256, 1, 1});
}

void NativeQwenOps::encode_copy_heads(CommandGraph& graph, const HeadView& source,
    const HeadView& destination) const {
    pipeline_ready(copy_heads_);
    validate_copy_heads(geometry_, source, destination);
    qwen_ops::CopyParameters p{source.tokens, uint32_t(source.token_stride_bytes / 2),
        uint32_t(source.head_stride_bytes / 2), uint32_t(destination.token_stride_bytes / 2),
        uint32_t(destination.head_stride_bytes / 2)};
    graph.dispatch(copy_heads_, {source.storage, destination.storage}, p,
                   point_groups(geometry_, source.tokens), {256, 1, 1});
}

void NativeQwenOps::encode_silu(CommandGraph& graph, const MatrixView& input,
    const MatrixView& output) const {
    pipeline_ready(silu_);
    validate_silu(geometry_,input,output);
    qwen_ops::UnaryParameters p{input.rows,stride(input),stride(output)};
    graph.dispatch(silu_,{input.storage,output.storage},p,
                   point_groups(geometry_,input.rows),{256,1,1});
}

void NativeQwenOps::encode_gelu(CommandGraph& graph, const MatrixView& input,
    const MatrixView& output) const {
    pipeline_ready(gelu_);
    validate_gelu(geometry_,input,output);
    qwen_ops::UnaryParameters p{input.rows,stride(input),stride(output)};
    graph.dispatch(gelu_,{input.storage,output.storage},p,
                   point_groups(geometry_,input.rows),{256,1,1});
}

void NativeQwenOps::encode_condition_norm(CommandGraph& graph, const MatrixView& text,
    const MatrixView& weight, const MatrixView& output, float epsilon) const {
    pipeline_ready(condition_norm_);
    validate_condition_norm(geometry_,text,weight,output,epsilon);
    qwen_ops::ConditionParameters p{text.rows,stride(text),stride(output),epsilon};
    graph.dispatch(condition_norm_,{text.storage,weight.storage,output.storage},p,
                   {text.rows,1,1},{256,1,1});
}

void NativeQwenOps::encode_expand_modulation(CommandGraph& graph, const MatrixView& source,
    const std::array<MatrixView,4>& outputs, uint32_t prefix_rows) const {
    pipeline_ready(expand_modulation_);
    validate_expand_modulation(geometry_,source,outputs,prefix_rows);
    qwen_ops::ExpandParameters p{outputs[0].rows,prefix_rows,stride(source),stride(outputs[0]),
        stride(outputs[1]),stride(outputs[2]),stride(outputs[3])};
    graph.dispatch(expand_modulation_,{source.storage,outputs[0].storage,outputs[1].storage,
                                      outputs[2].storage,outputs[3].storage},p,
                   point_groups(geometry_,outputs[0].rows),{256,1,1});
}

} // namespace tc::metal_graph
