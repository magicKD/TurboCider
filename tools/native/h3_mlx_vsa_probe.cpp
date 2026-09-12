#include "../../native/models/h3_mlx/vsa.hpp"
#include "../../native/models/h3_mlx/vsa_attention.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 1 || argc == 2,
                    "usage: h3-mlx-vsa-probe [OUTPUT]");
        using namespace tc;
        using namespace tc::h3_mlx;
        auto layout = build_packed_layout(5, 8, 16, 16, 3);
        auto geometry = build_vsa_geometry(
            vsa_prefix_segments(layout), vsa_video_shape(layout), 64);
        constexpr int heads = 2;
        constexpr int dim = 8;
        auto q = mx::astype(mx::random::normal(
            {geometry.total_sequence_length, heads, dim}, mx::float32,
            mx::random::key(2026)), mx::bfloat16);
        auto k = mx::astype(mx::random::normal(
            {geometry.total_sequence_length, heads, dim}, mx::float32,
            mx::random::key(2027)), mx::bfloat16);
        auto v = mx::astype(mx::random::normal(
            {geometry.total_sequence_length, heads, dim}, mx::float32,
            mx::random::key(2028)), mx::bfloat16);
        VSAStats sparse_stats;
        auto sparse = vsa_attention(q, k, v, geometry, 0.5f,
                                    VSAPrefixMode::exempt,
                                    VSAImplementation::reference, nullptr,
                                    &sparse_stats);
        mx::eval(sparse);
        require(sparse.shape() == q.shape(),
                "VSA sparse probe returned an unexpected shape");
        require(mx::all(mx::isfinite(sparse)).item<bool>(),
                "VSA sparse probe returned non-finite values");
        require(sparse_stats.video_keep == 4.f &&
                    sparse_stats.achieved_sparsity == 0.5f,
                "VSA sparse probe reported incorrect routing statistics");

        auto gate = mx::astype(mx::random::normal(
            {geometry.total_sequence_length, heads, dim}, mx::float32,
            mx::random::key(2029)), mx::bfloat16);
        auto dense_no_gate = vsa_attention(
            q, k, v, geometry, 0.f, VSAPrefixMode::exempt,
            VSAImplementation::reference, nullptr, nullptr);
        VSAStats dense_stats;
        auto dense = vsa_attention(q, k, v, geometry, 0.f,
                                   VSAPrefixMode::exempt,
                                   VSAImplementation::reference, &gate,
                                   &dense_stats);
        auto sparse_gate = vsa_attention(
            q, k, v, geometry, 0.5f, VSAPrefixMode::exempt,
            VSAImplementation::reference, &gate, nullptr);
        mx::eval(dense_no_gate, dense, sparse_gate);
        require(dense.shape() == q.shape(),
                "VSA dense/gate probe returned an unexpected shape");
        require(mx::all(mx::isfinite(dense)).item<bool>(),
                "VSA dense/gate probe returned non-finite values");
        require(dense_stats.implementation == "dense",
                "VSA dense probe did not report dense implementation");
        if (argc == 2) {
            mx::save_safetensors(
                std::filesystem::absolute(argv[1]).string(),
                {{"query", q}, {"key", k}, {"value", v}, {"gate", gate},
                 {"dense_no_gate", dense_no_gate}, {"dense_gate", dense},
                 {"sparse_no_gate", sparse}, {"sparse_gate", sparse_gate}});
        }
        std::cout << "{\"sequence\":" << geometry.total_sequence_length
                  << ",\"video_tiles\":" << geometry.num_video_tiles
                  << ",\"video_keep\":" << sparse_stats.video_keep
                  << ",\"achieved_sparsity\":"
                  << sparse_stats.achieved_sparsity << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
