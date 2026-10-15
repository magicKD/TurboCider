#include "mlx_ffn_island.hpp"

namespace tc::metal_graph {
namespace {
MatrixView matrix(ImportedMatrix view) {
    return {std::move(view.storage), view.rows, view.columns,
            view.row_stride_bytes};
}
}

MlxFfnIsland::MlxFfnIsland(Backend& backend, dense_ffn::Geometry geometry)
    : backend_(backend), ffn_(NativeFfn::prepare(backend, geometry)) {}

IslandResult MlxFfnIsland::run(const mlx::core::array& input,
    const mlx::core::array& gate, const mlx::core::array& up,
    const mlx::core::array& down, uint32_t first_channel,
    bool fp32_partial) const {
    // Import checks data/layout readiness. There are no eval, matmul, CPU
    // copies, shader builds or hidden allocation managers in this method.
    auto x = matrix(import_mlx_matrix(backend_, input));
    auto g = matrix(import_mlx_matrix(backend_, gate));
    auto u = matrix(import_mlx_matrix(backend_, up));
    auto d = matrix(import_mlx_matrix(backend_, down));
    const auto geometry = ffn_.geometry();
    auto hidden = allocate_mlx_matrix(x.rows, geometry.channels);
    auto output = allocate_mlx_matrix(x.rows, geometry.hidden,
        fp32_partial ? ScalarType::FP32 : ScalarType::BF16);
    auto h = matrix(import_mlx_matrix(backend_, hidden));
    auto y = matrix(import_mlx_matrix(backend_, output));
    CommandGraph graph;
    ffn_.encode(graph, {x, g, u, d, h, y, first_channel});
    auto ticket = backend_.submit(std::move(graph));
    auto completion = ticket.wait();
    if (!completion.physically_complete || !completion.ok) {
        // Submitted commands keep their own owners even when this call throws.
        // Do not publish uninitialized/partial outputs or force-signal events.
        throw std::runtime_error(completion.error.empty()
            ? "native FFN graph did not physically complete successfully"
            : completion.error);
    }
    return {std::move(output), std::move(hidden), std::move(completion)};
}
} // namespace tc::metal_graph
