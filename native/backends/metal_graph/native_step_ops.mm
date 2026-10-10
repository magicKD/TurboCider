#include "native_step_ops.hpp"

#include <cstring>

namespace tc::metal_graph {
namespace {
step_ops::Layout layout(const MatrixView& matrix) {
    const auto type = matrix.storage.scalar_type();
    if (!matrix.storage || (type != ScalarType::BF16 && type != ScalarType::FP32))
        throw std::invalid_argument("native step operator requires retained BF16/FP32 matrix");
    return {matrix.storage.buffer().native_handle(), matrix.storage.offset_bytes(), matrix.storage.size_bytes(),
            matrix.row_stride_bytes, matrix.rows, matrix.columns,
            type == ScalarType::BF16 ? step_ops::Element::BF16 : step_ops::Element::FP32};
}
step_ops::FlagLayout flag_layout(const BufferView& flag) {
    if (!flag || flag.scalar_type() != ScalarType::UInt32)
        throw std::invalid_argument("native step finite flag must have UInt32 type");
    return {flag.buffer().native_handle(), flag.offset_bytes(), flag.size_bytes()};
}
uint32_t stride(const MatrixView& matrix) {
    return uint32_t(matrix.row_stride_bytes / scalar_bytes(matrix.storage.scalar_type()));
}
Grid groups(const MatrixView& matrix) { return {(size_t(matrix.columns) + 255) / 256, matrix.rows, 1}; }
void prepared(const Pipeline& pipeline) {
    if (!pipeline) throw std::logic_error("native step operators have not been prepared");
}
} // namespace

NativeStepOps NativeStepOps::prepare(Backend& backend) {
    NativeStepOps result;
    auto library = backend.compile_library(step_ops::source, "tc-native-step-ops");
    result.copy_bf16_ = backend.prepare_pipeline(library, "tc_step_copy_bf16");
    result.copy_f32_ = backend.prepare_pipeline(library, "tc_step_copy_f32");
    result.euler_ = backend.prepare_pipeline(library, "tc_step_euler_bf16");
    result.reset_ = backend.prepare_pipeline(library, "tc_step_reset_invalid");
    result.finite_bf16_ = backend.prepare_pipeline(library, "tc_step_finite_bf16");
    result.finite_f32_ = backend.prepare_pipeline(library, "tc_step_finite_f32");
    for (const auto *pipeline : {&result.copy_bf16_, &result.copy_f32_, &result.euler_,
                                &result.reset_, &result.finite_bf16_, &result.finite_f32_})
        if (pipeline->max_threads_per_threadgroup() < (pipeline == &result.reset_ ? 1 : 256))
            throw std::runtime_error("native step kernel threadgroup capability unavailable");
    return result;
}
void NativeStepOps::validate_copy(const MatrixView& source, const MatrixView& destination) {
    step_ops::validate_copy(layout(source), layout(destination));
}
void NativeStepOps::validate_euler_bf16(const MatrixView& latents, const MatrixView& noise,
                                      float dt, const MatrixView& output) {
    step_ops::validate_euler(layout(latents), layout(noise), dt, layout(output));
}
void NativeStepOps::validate_invalid_flag(const BufferView& invalid) { step_ops::validate_flag(flag_layout(invalid)); }
void NativeStepOps::validate_finite(const MatrixView& input, const BufferView& invalid) {
    step_ops::validate_finite(layout(input), flag_layout(invalid));
}
void NativeStepOps::encode_copy(CommandGraph& graph, const MatrixView& source, const MatrixView& destination) const {
    validate_copy(source, destination);
    const auto &pipeline = source.storage.scalar_type() == ScalarType::BF16 ? copy_bf16_ : copy_f32_;
    prepared(pipeline);
    step_ops::CopyParameters p{source.rows, source.columns, stride(source), stride(destination)};
    graph.dispatch(pipeline, {source.storage, destination.storage}, p, groups(source), {256, 1, 1});
}
void NativeStepOps::encode_euler_bf16(CommandGraph& graph, const MatrixView& latents, const MatrixView& noise,
                                    float dt, const MatrixView& output) const {
    validate_euler_bf16(latents, noise, dt, output); prepared(euler_);
    step_ops::EulerParameters p{latents.rows, latents.columns, stride(latents), stride(noise), stride(output),
                                step_ops::checked_dt(dt)};
    graph.dispatch(euler_, {latents.storage, noise.storage, output.storage}, p, groups(latents), {256, 1, 1});
}
void NativeStepOps::encode_reset_invalid_flag(CommandGraph& graph, const BufferView& invalid) const {
    validate_invalid_flag(invalid); prepared(reset_);
    graph.dispatch(reset_, {invalid}, {1, 1, 1}, {1, 1, 1});
}
void NativeStepOps::encode_finite(CommandGraph& graph, const MatrixView& input, const BufferView& invalid) const {
    validate_finite(input, invalid);
    const auto &pipeline = input.storage.scalar_type() == ScalarType::BF16 ? finite_bf16_ : finite_f32_;
    prepared(pipeline);
    step_ops::FiniteParameters p{input.rows, input.columns, stride(input)};
    graph.dispatch(pipeline, {input.storage, invalid}, p, groups(input), {256, 1, 1});
}
void NativeStepOps::require_finite(const Ticket& ticket, const BufferView& invalid) {
    validate_invalid_flag(invalid);
    const auto completion = ticket.status();
    if (!completion.ok || !completion.physically_complete || completion.timed_out)
        throw std::runtime_error("native step finite flag cannot be read before successful physical completion");
    const void *contents = invalid.contents();
    if (!contents) throw std::invalid_argument("native step finite flag requires Shared storage for host inspection");
    uint32_t flags;
    std::memcpy(&flags, contents, sizeof(flags));
    if (flags) throw std::runtime_error("native step detected nonfinite output (invalid flag=" + std::to_string(flags) + ")");
}

} // namespace tc::metal_graph
