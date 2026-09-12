#include "vsa_attention.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

namespace tc::h3_mlx {
namespace {

Tensor int_indices(const std::vector<int32_t> &values) {
    return Tensor(values.data(), {int(values.size())}, mx::int32);
}

Tensor dense_attention(const Tensor &query, const Tensor &key,
                       const Tensor &value, float scale) {
    const auto q = mx::expand_dims(mx::transpose(query, {1, 0, 2}), 0);
    const auto k = mx::expand_dims(mx::transpose(key, {1, 0, 2}), 0);
    const auto v = mx::expand_dims(mx::transpose(value, {1, 0, 2}), 0);
    auto output = mx::fast::scaled_dot_product_attention(q, k, v, scale);
    output = mx::transpose(output, {0, 2, 1, 3});
    return mx::reshape(output, {query.shape(0), query.shape(1), query.shape(2)});
}

Tensor tile_hidden(const Tensor &value, const VSAGeometry &geometry) {
    const int sequence = value.shape(0);
    const int heads = value.shape(1);
    const int dim = value.shape(2);
    require(sequence == geometry.total_sequence_length,
            "H3 VSA tensor length does not match tile geometry");
    auto padded = mx::concatenate({value, mx::zeros({1, heads, dim}, value.dtype())}, 0);
    return mx::take(padded, int_indices(geometry.tile_gather_index()), 0);
}

Tensor tile_prefix(const Tensor &value, const VSAGeometry &geometry) {
    const int prefix = geometry.prefix_length();
    const int heads = value.shape(1);
    const int dim = value.shape(2);
    require(value.shape(0) == prefix,
            "H3 VSA prefix output length does not match geometry");
    auto padded = mx::concatenate({value, mx::zeros({1, heads, dim}, value.dtype())}, 0);
    return mx::take(padded, int_indices(geometry.prefix_gather_index()), 0);
}

Tensor untile_hidden(const Tensor &value, const VSAGeometry &geometry) {
    return mx::take(value, int_indices(geometry.untile_combined_index), 0);
}

Tensor pool_tiles(const Tensor &value, const VSAGeometry &geometry) {
    const int heads = value.shape(1);
    const int dim = value.shape(2);
    auto pooled = mx::reshape(value, {geometry.num_tiles(), geometry.tile_elems,
                                      heads, dim});
    pooled = mx::sum(mx::astype(pooled, mx::float32), 1);
    auto sizes = mx::astype(Tensor(geometry.variable_block_sizes.data(),
                                   {int(geometry.variable_block_sizes.size())}, mx::int32),
                            mx::float32);
    pooled = pooled / mx::reshape(sizes, {geometry.num_tiles(), 1, 1});
    return mx::transpose(pooled, {1, 0, 2});
}

Tensor selected_kv(const Tensor &tiles, const Tensor &indices,
                   int query_tiles, int selected_tiles,
                   int tile_elems, int dim) {
    const int heads = tiles.shape(0);
    require(indices.shape() == mx::Shape{heads, query_tiles, selected_tiles},
            "H3 VSA block-index shape does not match tile gather");
    // Match FastVideo's `tiles[arange(heads)[:,None,None], block_idx]`
    // advanced-indexing contract directly.  The MLX gather primitive keeps
    // the [head, query-tile, selected-tile, tile-row, dim] layout and avoids
    // the flat-head offset rewrite, whose Metal take kernel can produce
    // isolated BF16 ULP differences on wide VSA tensors.
    auto head_indices = mx::broadcast_to(
        mx::reshape(mx::arange(heads, mx::int32), {heads, 1, 1}),
        {heads, query_tiles, selected_tiles});
    auto gathered = mx::gather(
        tiles, {head_indices, indices}, {0, 1}, {1, 1, tile_elems, dim});
    return mx::reshape(gathered,
                       {heads, query_tiles, selected_tiles * tile_elems, dim});
}

Tensor selected_mask(const Tensor &indices,
                     const std::vector<int32_t> &variable_sizes,
                     int tile_elems, int query_tiles, int selected_tiles) {
    const int heads = indices.shape(0);
    auto sizes = Tensor(variable_sizes.data(),
                        {int(variable_sizes.size())}, mx::int32);
    auto selected = mx::take(sizes, indices);
    auto offsets = mx::reshape(mx::arange(tile_elems, mx::int32),
                               {1, 1, 1, tile_elems});
    auto valid = mx::greater(mx::expand_dims(selected, -1), offsets);
    return mx::reshape(valid,
                       {1, heads * query_tiles, 1, selected_tiles * tile_elems});
}

Tensor video_sparse_attention(const Tensor &query_tiles,
                              const Tensor &key_tiles,
                              const Tensor &value_tiles,
                              const Tensor &block_indices,
                              const VSAGeometry &geometry,
                              float scale) {
    const int heads = query_tiles.shape(0);
    const int video_tiles = query_tiles.shape(1);
    const int tile_elems = geometry.tile_elems;
    const int dim = query_tiles.shape(3);
    const int selected = block_indices.shape(2);
    const int bytes_per_query = 4 * heads * std::max(selected, 1) * tile_elems * dim;
    const uint64_t target_bytes = 2ull * 1024 * 1024 * 1024;
    const int query_chunk = std::max(1, std::min(video_tiles,
        int(target_bytes / uint64_t(std::max(bytes_per_query, 1)))));
    std::vector<Tensor> chunks;
    for (int start = 0; start < video_tiles; start += query_chunk) {
        const int end = std::min(video_tiles, start + query_chunk);
        const int count = end - start;
        auto indices = slice_axis(block_indices, 1, start, end);
        auto gathered_k = selected_kv(key_tiles, indices, count, selected,
                                       tile_elems, dim);
        auto gathered_v = selected_kv(value_tiles, indices, count, selected,
                                       tile_elems, dim);
        auto valid = selected_mask(indices, geometry.variable_block_sizes,
                                   tile_elems, count, selected);
        auto q = mx::contiguous(slice_axis(query_tiles, 1, start, end));
        q = mx::reshape(q, {heads * count, tile_elems, dim});
        q = mx::expand_dims(q, 0);
        auto k = mx::reshape(mx::contiguous(gathered_k),
                             {heads * count, selected * tile_elems, dim});
        k = mx::expand_dims(k, 0);
        auto v = mx::reshape(mx::contiguous(gathered_v),
                             {heads * count, selected * tile_elems, dim});
        v = mx::expand_dims(v, 0);
        auto output = mx::fast::scaled_dot_product_attention(q, k, v, scale,
                                                              "", valid);
        output = mx::reshape(output,
                             {heads, count, tile_elems, dim});
        chunks.push_back(output);
        mx::eval(output);
    }
    require(!chunks.empty(), "H3 VSA produced no video query tiles");
    return chunks.size() == 1 ? chunks.front() : mx::concatenate(chunks, 1);
}

Tensor gate_output(const Tensor &scores, const Tensor &value_tiles,
                   const Tensor &gate_tiles, const VSAGeometry &geometry) {
    auto pooled = pool_tiles(value_tiles, geometry);
    auto compressed = mx::matmul(mx::softmax(scores, -1), pooled);
    compressed = mx::transpose(compressed, {1, 0, 2});
    const int heads = gate_tiles.shape(1);
    const int dim = gate_tiles.shape(2);
    auto gate = mx::reshape(gate_tiles,
                            {geometry.num_tiles(), geometry.tile_elems, heads, dim});
    return mx::reshape(mx::reshape(compressed,
                                   {geometry.num_tiles(), 1, heads, dim}) * gate,
                       {geometry.padded_length(), heads, dim});
}

} // namespace

Tensor vsa_attention(const Tensor &query, const Tensor &key, const Tensor &value,
                     const VSAGeometry &geometry, double sparsity,
                     VSAPrefixMode prefix_mode,
                     VSAImplementation implementation,
                     const Tensor *gate_compress,
                     VSAStats *stats) {
    require(query.shape() == key.shape() && query.shape() == value.shape() &&
                query.ndim() == 3,
            "H3 VSA Q/K/V shapes must match [sequence, heads, dim]");
    const int sequence = query.shape(0);
    const int heads = query.shape(1);
    const int dim = query.shape(2);
    require(sequence == geometry.total_sequence_length,
            "H3 VSA Q/K/V sequence does not match geometry");
    const uint64_t call_index = stats ? stats->attention_calls : 0;
    if (stats) {
        stats->configured_sparsity = sparsity;
        stats->tile_size = geometry.tile_elems;
        stats->num_prefix_tiles = geometry.num_prefix_tiles;
        stats->num_video_tiles = geometry.num_video_tiles;
        stats->prefix_mode = to_string(prefix_mode);
        stats->attention_calls++;
    }
    const float scale = 1.f / std::sqrt(float(dim));
    auto q_tiles = tile_hidden(query, geometry);
    auto k_tiles = tile_hidden(key, geometry);
    auto v_tiles = tile_hidden(value, geometry);
    auto q_pool = pool_tiles(q_tiles, geometry);
    auto k_pool = pool_tiles(k_tiles, geometry);
    // Keep the routed-score expression byte-for-byte equivalent to
    // FastVideo's `(q_pool @ k_pool.T) / sqrt(dim)`. Replacing the division
    // with multiplication by a precomputed reciprocal changes a small
    // number of FP32 score values; the trained gate-compression softmax then
    // amplifies those BF16 ULPs across 50 transformer blocks.
    auto scores = mx::matmul(q_pool,
                             mx::transpose(k_pool, {0, 2, 1})) /
                  std::sqrt(float(dim));
    if (stats && stats->capture_debug && !stats->debug_captured) {
        stats->debug_q_pool = q_pool;
        stats->debug_k_pool = k_pool;
        stats->debug_scores = scores;
    }

    std::optional<Tensor> output;
    std::string chosen = "reference";
    if (sparsity <= 0.f) {
        output = tile_hidden(dense_attention(query, key, value, scale), geometry);
        chosen = "dense";
    } else {
        const int keep_video = vsa_compute_topk(sparsity, geometry.num_video_tiles);
        const int prefix = geometry.num_prefix_tiles;
        const int total_tiles = geometry.num_tiles();
        auto video_scores = slice_axis(slice_axis(scores, 1, prefix,
                                                   total_tiles),
                                       2, prefix, total_tiles);
        std::optional<Tensor> block_indices;
        int selected = keep_video;
        if (prefix_mode == VSAPrefixMode::exempt) {
            auto top = mx::argpartition(-video_scores, keep_video - 1, -1);
            top = slice_axis(top, -1, 0, keep_video) + prefix;
            selected += prefix;
            if (prefix) {
                auto prefix_idx = mx::broadcast_to(
                    mx::reshape(mx::arange(prefix, mx::int32), {1, 1, prefix}),
                    {heads, geometry.num_video_tiles, prefix});
                block_indices = mx::concatenate({prefix_idx, top}, -1);
            } else {
                block_indices = top;
            }
        } else {
            selected = std::min(keep_video + prefix, total_tiles);
            auto top = mx::argpartition(-slice_axis(scores, 1, prefix, total_tiles),
                                        selected - 1, -1);
            block_indices = slice_axis(top, -1, 0, selected);
        }
        auto all_q = mx::transpose(mx::reshape(q_tiles,
            {geometry.num_tiles(), geometry.tile_elems, heads, dim}),
            {2, 0, 1, 3});
        auto q_video = slice_axis(all_q, 1, prefix, geometry.num_tiles());
        auto all_k = mx::transpose(mx::reshape(k_tiles,
            {geometry.num_tiles(), geometry.tile_elems, heads, dim}),
            {2, 0, 1, 3});
        auto all_v = mx::transpose(mx::reshape(v_tiles,
            {geometry.num_tiles(), geometry.tile_elems, heads, dim}),
            {2, 0, 1, 3});
        require(block_indices.has_value(), "H3 VSA routing did not produce block indices");
        if (stats && stats->capture_debug && !stats->debug_captured) {
            stats->debug_block_indices = *block_indices;
            stats->debug_captured = true;
            mx::eval(*stats->debug_q_pool, *stats->debug_k_pool,
                     *stats->debug_scores, *stats->debug_block_indices);
        }
        auto video_out = video_sparse_attention(q_video, all_k, all_v,
                                                *block_indices, geometry, scale);
        auto video_tiled = mx::reshape(mx::transpose(video_out, {1, 2, 0, 3}),
                                       {geometry.num_video_tiles * geometry.tile_elems,
                                        heads, dim});
        if (geometry.prefix_length()) {
            auto prefix_query = slice_axis(query, 0, 0, geometry.prefix_length());
            auto prefix_out = dense_attention(prefix_query, key, value, scale);
            output = mx::concatenate({tile_prefix(prefix_out, geometry), video_tiled}, 0);
        } else {
            output = video_tiled;
        }
        chosen = "reference";
        if (stats) {
            stats->sparse_calls++;
            const double keep = double(keep_video);
            const double achieved = 1.0 - keep /
                                    double(geometry.num_video_tiles);
            const double count = double(call_index + 1);
            stats->video_keep =
                (stats->video_keep * double(call_index) + keep) / count;
            stats->achieved_sparsity =
                (stats->achieved_sparsity * double(call_index) + achieved) / count;
            if (implementation == VSAImplementation::simd)
                stats->dense_fallback_reason = "simd implementation is not available in this build";
        }
    }
    if (gate_compress) {
        require(output.has_value(), "H3 VSA attention did not produce output");
        output = *output + mx::astype(gate_output(scores, v_tiles,
                                      tile_hidden(*gate_compress, geometry),
                                      geometry), output->dtype());
    }
    if (stats) {
        if (call_index == 0) stats->implementation = chosen;
        else if (stats->implementation != chosen) stats->implementation = "mixed";
        if (sparsity <= 0.f) {
            const double count = double(call_index + 1);
            const double keep = double(geometry.num_video_tiles);
            stats->video_keep =
                (stats->video_keep * double(call_index) + keep) / count;
            stats->achieved_sparsity =
                (stats->achieved_sparsity * double(call_index)) / count;
        }
    }
    require(output.has_value(), "H3 VSA attention did not produce output");
    return untile_hidden(*output, geometry);
}

} // namespace tc::h3_mlx
