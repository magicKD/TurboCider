#pragma once

#include "../../core/common.hpp"

#include <array>
#include <cstdint>

namespace tc::h3_mlx {

constexpr int video_tag = 0;
constexpr int text_tag = 1;
constexpr int audio_tag = 2;
constexpr int modality_count = 3;
constexpr int frames_per_chunk = 17;
constexpr int latents_per_chunk = 5;
constexpr int audio_latents_per_second = 40;
constexpr int audio_channels = 2;
constexpr float video_shift = 12.f;
constexpr float audio_shift = 3.f;

struct PackedLayout {
    int sequence_length = 0;
    std::vector<double> position_ids; // row-major [sequence, 3]
    std::vector<int32_t> token_tags;
    std::vector<int32_t> video_indices;
    std::vector<int32_t> audio_indices;
    std::vector<int32_t> text_indices;
    int condition_video_rows = 0;
    int condition_audio_rows = 0;
    int video_latent_frames = 0;
    int latent_height = 0;
    int latent_width = 0;
    int audio_latents = 0;
};

struct RowTimesteps {
    std::vector<float> unique;
    std::vector<int32_t> inverse;
};

struct Scheduler {
    float shift = 1.f;
    std::vector<float> sigmas;
    std::vector<float> timesteps;

    static Scheduler create(float shift, int steps);
};

int align_frames(int frames);
int video_latent_frames(int frames);
int audio_latent_frames(int frames, int fps = 24);
PackedLayout build_packed_layout(int text_tokens, int latent_frames,
                                 int latent_height, int latent_width,
                                 int audio_latents,
                                 const std::array<int, 3> &patch_size = {1, 2, 2},
                                 const std::vector<int32_t> &text_tags = {});
RowTimesteps build_row_timesteps(const PackedLayout &, float video_timestep,
                                 float audio_timestep,
                                 float condition_video_timestep = 1.f,
                                 float condition_audio_timestep = 1.f);
std::vector<float> four_step_adaln_union();

} // namespace tc::h3_mlx
