#include "vsa.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tc::h3_mlx {
namespace {

int product(const std::array<int, 3> &shape) {
    return shape[0] * shape[1] * shape[2];
}

int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

std::array<int, 3> tile_shape(int tile_size) {
    if (tile_size == 64) return {4, 4, 4};
    if (tile_size == 256) return {4, 8, 8};
    throw std::runtime_error("H3 VSA tile_size must be 64 or 256");
}

std::vector<int32_t> axis_sizes(int length, int tile) {
    const int count = ceil_div(length, tile);
    std::vector<int32_t> result(static_cast<size_t>(count),
                                static_cast<int32_t>(tile));
    result.back() = int32_t(length - (count - 1) * tile);
    return result;
}

} // namespace

void VSAConfig::validate(int num_layers) const {
    require(num_layers > 0, "H3 VSA requires a positive transformer layer count");
    require(std::isfinite(sparsity) && sparsity >= 0.f && sparsity < 1.f,
            "H3 VSA sparsity must be in [0, 1)");
    require(tile_size == 64 || tile_size == 256,
            "H3 VSA tile_size must be 64 or 256");
    require(dense_first_n_steps >= 0,
            "H3 VSA dense_first_n_steps must be nonnegative");
    for (int layer : dense_layers)
        require(layer >= 0 && layer < num_layers,
                "H3 VSA dense layer index is outside the transformer");
}

double VSAConfig::layer_sparsity(int layer_index, int step_index) const {
    require(layer_index >= 0 && step_index >= 0,
            "H3 VSA layer and step indices must be nonnegative");
    if (!enabled || step_index < dense_first_n_steps ||
        std::find(dense_layers.begin(), dense_layers.end(), layer_index) !=
            dense_layers.end())
        return 0.0;
    return sparsity;
}

int VSAGeometry::prefix_length() const {
    return std::accumulate(prefix_segments.begin(), prefix_segments.end(), 0);
}

std::vector<int32_t> VSAGeometry::tile_gather_index() const {
    std::vector<int32_t> result(static_cast<size_t>(padded_length()),
                                static_cast<int32_t>(total_sequence_length));
    for (int row = 0; row < total_sequence_length; ++row)
        result.at(size_t(untile_combined_index.at(size_t(row)))) = int32_t(row);
    return result;
}

std::vector<int32_t> VSAGeometry::prefix_gather_index() const {
    const int prefix = prefix_length();
    std::vector<int32_t> result(size_t(num_prefix_tiles * tile_elems),
                                int32_t(prefix));
    for (int row = 0; row < prefix; ++row)
        result.at(size_t(untile_combined_index.at(size_t(row)))) = int32_t(row);
    return result;
}

bool VSABlockMask::at(int head, int query_tile, int key_tile) const {
    require(head >= 0 && head < heads && query_tile >= 0 &&
                query_tile < tiles && key_tile >= 0 && key_tile < tiles,
            "H3 VSA block-mask index is out of range");
    const size_t index = (size_t(head) * tiles + query_tile) * tiles + key_tile;
    return values.at(index) != 0;
}

std::string to_string(VSAPrefixMode value) {
    return value == VSAPrefixMode::exempt ? "exempt" : "compete";
}

std::string to_string(VSAImplementation value) {
    switch (value) {
        case VSAImplementation::automatic: return "auto";
        case VSAImplementation::reference: return "reference";
        case VSAImplementation::simd: return "simd";
    }
    throw std::runtime_error("unknown H3 VSA implementation");
}

VSAPrefixMode vsa_prefix_mode(const std::string &value) {
    if (value == "exempt") return VSAPrefixMode::exempt;
    if (value == "compete") return VSAPrefixMode::compete;
    throw std::runtime_error("H3 VSA prefix_mode must be exempt or compete");
}

VSAImplementation vsa_implementation(const std::string &value) {
    if (value == "auto") return VSAImplementation::automatic;
    if (value == "reference") return VSAImplementation::reference;
    if (value == "simd") return VSAImplementation::simd;
    throw std::runtime_error("H3 VSA implementation must be auto, reference, or simd");
}

int vsa_compute_topk(double sparsity, int num_video_tiles) {
    require(std::isfinite(sparsity) && sparsity >= 0.f && sparsity < 1.f,
            "H3 VSA sparsity must be in [0, 1)");
    require(num_video_tiles >= 0, "H3 VSA video tile count must be nonnegative");
    if (num_video_tiles == 0) return 0;
    const int keep = int(std::ceil((1.0 - double(sparsity)) * num_video_tiles));
    return std::clamp(keep, 1, num_video_tiles);
}

std::vector<int> vsa_prefix_segments(const PackedLayout &layout) {
    require(layout.sequence_length > 0, "H3 VSA cannot tile an empty packed layout");
    std::vector<int> result;
    for (int size : {int(layout.text_indices.size()), layout.condition_video_rows,
                     int(layout.audio_indices.size())})
        if (size > 0) result.push_back(size);
    return result;
}

std::array<int, 3> vsa_video_shape(
    const PackedLayout &layout, const std::array<int, 3> &patch_size) {
    require(patch_size[0] > 0 && patch_size[1] > 0 && patch_size[2] > 0 &&
                layout.video_latent_frames % patch_size[0] == 0 &&
                layout.latent_height % patch_size[1] == 0 &&
                layout.latent_width % patch_size[2] == 0,
            "H3 VSA video geometry is not patch divisible");
    const std::array<int, 3> result{
        layout.video_latent_frames / patch_size[0],
        layout.latent_height / patch_size[1],
        layout.latent_width / patch_size[2],
    };
    const auto segments = vsa_prefix_segments(layout);
    const int prefix = std::accumulate(segments.begin(), segments.end(), 0);
    require(prefix + product(result) == layout.sequence_length,
            "H3 VSA supports standard [text|condition|audio|video] packing only");
    return result;
}

VSAGeometry build_vsa_geometry(const std::vector<int> &raw_prefix_segments,
                               const std::array<int, 3> &video_shape,
                               int tile_size) {
    const auto shape = tile_shape(tile_size);
    require(std::all_of(video_shape.begin(), video_shape.end(),
                        [](int axis) { return axis > 0; }),
            "H3 VSA video axes must be positive");
    const int tile_elems = product(shape);
    VSAGeometry result;
    result.video_shape = video_shape;
    result.tile_shape = shape;
    result.tile_elems = tile_elems;
    for (int segment : raw_prefix_segments) {
        require(segment >= 0, "H3 VSA prefix segments must be nonnegative");
        if (segment == 0) continue;
        result.prefix_segments.push_back(segment);
        const int full = segment / tile_elems;
        const int remainder = segment % tile_elems;
        result.variable_block_sizes.insert(result.variable_block_sizes.end(),
                                           size_t(full), int32_t(tile_elems));
        if (remainder) result.variable_block_sizes.push_back(int32_t(remainder));
    }
    result.num_prefix_tiles = int(result.variable_block_sizes.size());

    const auto t_sizes = axis_sizes(video_shape[0], shape[0]);
    const auto h_sizes = axis_sizes(video_shape[1], shape[1]);
    const auto w_sizes = axis_sizes(video_shape[2], shape[2]);
    result.num_video_tiles = int(t_sizes.size() * h_sizes.size() * w_sizes.size());
    for (int32_t t : t_sizes)
        for (int32_t h : h_sizes)
            for (int32_t w : w_sizes)
                result.variable_block_sizes.push_back(t * h * w);

    const int prefix = result.prefix_length();
    result.total_sequence_length = prefix + product(video_shape);
    result.tile_partition_indices.reserve(size_t(result.total_sequence_length));
    for (int row = 0; row < prefix; ++row)
        result.tile_partition_indices.push_back(int32_t(row));
    for (int tt = 0; tt < int(t_sizes.size()); ++tt)
        for (int hh = 0; hh < int(h_sizes.size()); ++hh)
            for (int ww = 0; ww < int(w_sizes.size()); ++ww)
                for (int t = tt * shape[0]; t < std::min((tt + 1) * shape[0], video_shape[0]); ++t)
                    for (int h = hh * shape[1]; h < std::min((hh + 1) * shape[1], video_shape[1]); ++h)
                        for (int w = ww * shape[2]; w < std::min((ww + 1) * shape[2], video_shape[2]); ++w)
                            result.tile_partition_indices.push_back(int32_t(
                                prefix + (t * video_shape[1] + h) * video_shape[2] + w));

    std::vector<int32_t> non_pad;
    non_pad.reserve(size_t(result.total_sequence_length));
    for (int tile = 0; tile < result.num_tiles(); ++tile)
        for (int offset = 0; offset < result.variable_block_sizes.at(size_t(tile)); ++offset)
            non_pad.push_back(int32_t(tile * tile_elems + offset));
    require(non_pad.size() == size_t(result.total_sequence_length) &&
                result.tile_partition_indices.size() == non_pad.size(),
            "H3 VSA tile sizes do not cover the packed sequence");

    std::vector<int32_t> order(size_t(result.total_sequence_length));
    std::iota(order.begin(), order.end(), int32_t(0));
    std::stable_sort(order.begin(), order.end(), [&](int32_t left, int32_t right) {
        return result.tile_partition_indices[size_t(left)] <
               result.tile_partition_indices[size_t(right)];
    });
    result.untile_combined_index.resize(size_t(result.total_sequence_length));
    for (int row = 0; row < result.total_sequence_length; ++row)
        result.untile_combined_index[size_t(row)] = non_pad[size_t(order[size_t(row)])];

    require(*std::min_element(result.variable_block_sizes.begin(),
                              result.variable_block_sizes.end()) >= 1 &&
                *std::max_element(result.variable_block_sizes.begin(),
                                  result.variable_block_sizes.end()) <= tile_elems &&
                std::accumulate(result.variable_block_sizes.begin(),
                                result.variable_block_sizes.end(), 0) ==
                    result.total_sequence_length,
            "H3 VSA variable tile sizes are invalid");
    auto untile = result.untile_combined_index;
    std::sort(untile.begin(), untile.end());
    require(std::adjacent_find(untile.begin(), untile.end()) == untile.end() &&
                untile.front() >= 0 && untile.back() < result.padded_length(),
            "H3 VSA untile map is not injective");
    for (int32_t slot : result.untile_combined_index)
        require(slot % tile_elems <
                    result.variable_block_sizes.at(size_t(slot / tile_elems)),
                "H3 VSA untile map points into padding");
    return result;
}

VSABlockMask build_vsa_block_mask(const std::vector<float> &scores,
                                  int heads, int num_prefix_tiles,
                                  int num_video_tiles, double sparsity,
                                  VSAPrefixMode prefix_mode) {
    require(heads > 0 && num_prefix_tiles >= 0 && num_video_tiles > 0,
            "H3 VSA block-mask geometry is invalid");
    const int tiles = num_prefix_tiles + num_video_tiles;
    require(scores.size() == size_t(heads) * tiles * tiles,
            "H3 VSA score tensor does not match the tile geometry");
    VSABlockMask result{heads, tiles,
                        std::vector<uint8_t>(size_t(heads) * tiles * tiles, 0)};
    const int keep_video = vsa_compute_topk(sparsity, num_video_tiles);
    if (keep_video == num_video_tiles) {
        std::fill(result.values.begin(), result.values.end(), uint8_t(1));
        return result;
    }
    for (int head = 0; head < heads; ++head) {
        for (int query = 0; query < tiles; ++query) {
            if (query < num_prefix_tiles) {
                for (int key = 0; key < tiles; ++key)
                    result.values[(size_t(head) * tiles + query) * tiles + key] = 1;
                continue;
            }
            std::vector<int> candidates;
            if (prefix_mode == VSAPrefixMode::exempt) {
                for (int key = 0; key < num_prefix_tiles; ++key)
                    result.values[(size_t(head) * tiles + query) * tiles + key] = 1;
                for (int key = num_prefix_tiles; key < tiles; ++key)
                    candidates.push_back(key);
            } else {
                candidates.resize(size_t(tiles));
                std::iota(candidates.begin(), candidates.end(), 0);
            }
            std::stable_sort(candidates.begin(), candidates.end(), [&](int left, int right) {
                const size_t base = (size_t(head) * tiles + query) * tiles;
                return scores[base + size_t(left)] > scores[base + size_t(right)];
            });
            const int keep = prefix_mode == VSAPrefixMode::exempt
                ? keep_video : std::min(keep_video + num_prefix_tiles, tiles);
            for (int index = 0; index < keep; ++index)
                result.values[(size_t(head) * tiles + query) * tiles +
                              size_t(candidates[size_t(index)])] = 1;
        }
    }
    return result;
}

} // namespace tc::h3_mlx
