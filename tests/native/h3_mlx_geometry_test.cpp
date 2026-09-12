#include "../../native/models/h3_mlx/geometry.hpp"
#include "../../native/models/h3_mlx/conditioner_math.hpp"
#include "../../native/models/h3_mlx/vsa.hpp"

#include <cassert>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>

int main() {
    using namespace tc::h3_mlx;
    assert(align_frames(1) == 5);
    assert(align_frames(124) == 124);
    assert(video_latent_frames(124) == 37);
    assert(audio_latent_frames(124) == 207);
    auto layout = build_packed_layout(11, 37, 30, 52, 207);
    assert(layout.sequence_length == 11 + 414 + 37 * 15 * 26);
    assert(layout.text_indices.front() == 0 && layout.text_indices.back() == 10);
    assert(layout.audio_indices.front() == 11);
    assert(layout.video_indices.front() == 425);
    assert(layout.token_tags[10] == text_tag);
    assert(layout.token_tags[11] == audio_tag);
    assert(layout.token_tags[425] == video_tag);
    auto rows = build_row_timesteps(layout, 0.2f, 0.5f);
    assert(rows.unique.size() == 2);
    assert(rows.inverse[0] == 0);
    assert(rows.inverse[layout.audio_indices[0]] == 1);
    auto video = Scheduler::create(12.f, 4);
    auto audio = Scheduler::create(3.f, 4);
    assert(video.sigmas.size() == 5 && audio.timesteps.size() == 4);
    assert(std::abs(video.timesteps[3] - 0.2f) < 1e-6f);
    assert(std::abs(audio.timesteps[3] - 0.5f) < 1e-6f);
    auto union_values = four_step_adaln_union();
    assert(union_values.size() == 8);
    assert(union_values.front() == 0.f && union_values.back() == 1.f);

    VSAConfig vsa;
    vsa.enabled = true;
    vsa.dense_first_n_steps = 1;
    vsa.dense_layers = {3, 7};
    vsa.validate(50);
    assert(vsa.layer_sparsity(0, 0) == 0.0);
    assert(vsa.layer_sparsity(3, 1) == 0.0);
    assert(std::abs(vsa.layer_sparsity(4, 1) - 0.9) < 1e-12);
    assert(vsa_compute_topk(0.9, 280) == 28);

    auto prefix = vsa_prefix_segments(layout);
    assert((prefix == std::vector<int>{11, 414}));
    auto video_shape = vsa_video_shape(layout);
    assert((video_shape == std::array<int, 3>{37, 15, 26}));
    auto vsa64 = build_vsa_geometry(prefix, video_shape, 64);
    assert((vsa64.tile_shape == std::array<int, 3>{4, 4, 4}));
    assert(vsa64.num_prefix_tiles == 8);
    assert(vsa64.num_video_tiles == 280);
    assert(vsa64.total_sequence_length == layout.sequence_length);
    assert(vsa64.padded_length() == 288 * 64);
    assert(std::accumulate(vsa64.variable_block_sizes.begin(),
                           vsa64.variable_block_sizes.end(), 0) ==
           layout.sequence_length);
    auto gather = vsa64.tile_gather_index();
    for (int row = 0; row < layout.sequence_length; ++row)
        assert(gather.at(size_t(vsa64.untile_combined_index.at(size_t(row)))) == row);

    auto vsa256 = build_vsa_geometry(prefix, video_shape, 256);
    assert((vsa256.tile_shape == std::array<int, 3>{4, 8, 8}));
    assert(vsa256.num_prefix_tiles == 3);
    assert(vsa256.num_video_tiles == 80);
    assert(vsa256.padded_length() == 83 * 256);

    std::vector<float> scores(25);
    std::iota(scores.begin(), scores.end(), 0.f);
    auto exempt = build_vsa_block_mask(scores, 1, 2, 3, 0.5f,
                                       VSAPrefixMode::exempt);
    for (int key = 0; key < 5; ++key) {
        assert(exempt.at(0, 0, key));
        assert(exempt.at(0, 1, key));
    }
    assert(exempt.at(0, 2, 0) && exempt.at(0, 2, 1));
    assert(exempt.at(0, 2, 3) && exempt.at(0, 2, 4));
    assert(!exempt.at(0, 2, 2));
    auto compete = build_vsa_block_mask(scores, 1, 2, 3, 0.5f,
                                        VSAPrefixMode::compete);
    assert(!compete.at(0, 2, 0));
    for (int key = 1; key < 5; ++key) assert(compete.at(0, 2, key));

    auto exact = [](float input, uint32_t sine, uint32_t cosine) {
        const auto value = numpy_float32_sin_cos(input);
        assert(std::bit_cast<uint32_t>(value.sine) == sine);
        assert(std::bit_cast<uint32_t>(value.cosine) == cosine);
    };
    exact(0.f, 0x00000000u, 0x3f800000u);
    exact(1.f, 0x3f576aa5u, 0x3f0a5140u);
    exact(2.f, 0x3f68c7b7u, 0xbed51132u);
    exact(6.f, 0xbe8f0f8cu, 0x3f75cdb8u);
}
