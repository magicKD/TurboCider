#pragma once

#include "geometry.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tc::h3_mlx {

enum class VSAPrefixMode {
    exempt,
    compete,
};

enum class VSAImplementation {
    automatic,
    reference,
    simd,
};

struct VSAConfig {
    bool enabled = false;
    // The trained gate branch is part of a VSA-capable checkpoint.  This
    // remains enabled for all product profiles; the opt-out is validation-only
    // so attention routing and the pooled gate can be compared independently.
    bool gate_compress = true;
    double sparsity = 0.9;
    int tile_size = 64;
    VSAPrefixMode prefix_mode = VSAPrefixMode::exempt;
    int dense_first_n_steps = 0;
    std::vector<int> dense_layers;
    VSAImplementation implementation = VSAImplementation::automatic;

    void validate(int num_layers) const;
    double layer_sparsity(int layer_index, int step_index) const;
};

struct VSAGeometry {
    std::vector<int> prefix_segments;
    std::array<int, 3> video_shape{};
    std::array<int, 3> tile_shape{};
    int tile_elems = 0;
    int total_sequence_length = 0;
    int num_prefix_tiles = 0;
    int num_video_tiles = 0;
    std::vector<int32_t> variable_block_sizes;
    std::vector<int32_t> untile_combined_index;
    std::vector<int32_t> tile_partition_indices;

    int num_tiles() const { return num_prefix_tiles + num_video_tiles; }
    int padded_length() const { return num_tiles() * tile_elems; }
    int prefix_length() const;
    std::vector<int32_t> tile_gather_index() const;
    std::vector<int32_t> prefix_gather_index() const;
};

struct VSABlockMask {
    int heads = 0;
    int tiles = 0;
    std::vector<uint8_t> values;

    bool at(int head, int query_tile, int key_tile) const;
};

std::string to_string(VSAPrefixMode);
std::string to_string(VSAImplementation);
VSAPrefixMode vsa_prefix_mode(const std::string &);
VSAImplementation vsa_implementation(const std::string &);
int vsa_compute_topk(double sparsity, int num_video_tiles);
std::vector<int> vsa_prefix_segments(const PackedLayout &);
std::array<int, 3> vsa_video_shape(
    const PackedLayout &, const std::array<int, 3> &patch_size = {1, 2, 2});
VSAGeometry build_vsa_geometry(const std::vector<int> &prefix_segments,
                               const std::array<int, 3> &video_shape,
                               int tile_size = 64);
VSABlockMask build_vsa_block_mask(const std::vector<float> &scores,
                                  int heads, int num_prefix_tiles,
                                  int num_video_tiles, double sparsity,
                                  VSAPrefixMode prefix_mode);

} // namespace tc::h3_mlx
