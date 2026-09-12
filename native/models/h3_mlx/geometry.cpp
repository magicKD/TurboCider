#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace tc::h3_mlx {
namespace {
constexpr double rope_frame_rescale = 5.0 / 3.0;
constexpr std::array<int, 5> rope_frames_per_latent{1, 4, 4, 4, 4};
constexpr double rope_spatial_scale = 32.0;

std::vector<double> spatial_grid(int dim, int patch, double sqrt_area) {
    require(dim > 0 && dim % patch == 0 && sqrt_area > 0,
            "invalid H3 spatial grid geometry");
    const int count = dim / patch;
    const double ratio = dim / sqrt_area;
    const double left = (1.0 - ratio) / 2.0;
    std::vector<double> result(count);
    for (int index = 0; index < count; ++index)
        result[index] = (left + ratio * index / count) * rope_spatial_scale;
    return result;
}

std::vector<double> temporal_grid(int frames, double origin) {
    require(frames > 0, "invalid H3 temporal grid geometry");
    std::vector<double> result(frames);
    double current = origin;
    for (int index = 0; index < frames; ++index) {
        result[index] = current;
        current += rope_frame_rescale * rope_frames_per_latent[index % 5];
    }
    return result;
}
} // namespace

int align_frames(int frames) {
    require(frames > 0, "H3 frame count must be positive");
    while (frames % frames_per_chunk != latents_per_chunk) ++frames;
    return frames;
}

int video_latent_frames(int frames) {
    require(frames > 0 && frames % frames_per_chunk == latents_per_chunk,
            "H3 frames must be 17n+5");
    return (frames - latents_per_chunk) / frames_per_chunk * latents_per_chunk + 2;
}

int audio_latent_frames(int frames, int fps) {
    require(frames > 0 && fps > 0, "invalid H3 audio duration");
    return int(std::llround(double(frames) / fps * audio_latents_per_second));
}

PackedLayout build_packed_layout(int text_tokens, int latent_frames,
                                 int latent_height, int latent_width,
                                 int audio_latents,
                                 const std::array<int, 3> &patch_size,
                                 const std::vector<int32_t> &text_tags) {
    require(text_tokens > 0 && latent_frames > 0 && latent_height > 0 &&
                latent_width > 0 && audio_latents > 0,
            "invalid H3 packed layout geometry");
    require(patch_size[0] == 1 && latent_height % patch_size[1] == 0 &&
                latent_width % patch_size[2] == 0,
            "unsupported H3 packed patch geometry");
    require(text_tags.empty() || int(text_tags.size()) == text_tokens,
            "H3 text tags do not match token count");
    for (int32_t tag : text_tags)
        require(tag == text_tag || tag == video_tag, "invalid H3 text presentation tag");

    PackedLayout result;
    const int rows_per_frame = latent_height / patch_size[1] *
                               (latent_width / patch_size[2]);
    const int audio_rows = audio_latents * audio_channels;
    const int video_rows = latent_frames * rows_per_frame;
    const int audio_start = text_tokens;
    const int video_start = audio_start + audio_rows;
    result.sequence_length = video_start + video_rows;
    result.position_ids.assign(size_t(result.sequence_length) * 3, 0.0);
    result.token_tags.resize(result.sequence_length);
    result.video_indices.resize(video_rows);
    result.audio_indices.resize(audio_rows);
    result.text_indices.resize(text_tokens);
    result.video_latent_frames = latent_frames;
    result.latent_height = latent_height;
    result.latent_width = latent_width;
    result.audio_latents = audio_latents;

    for (int row = 0; row < text_tokens; ++row) {
        result.position_ids[size_t(row) * 3] = row;
        result.token_tags[row] = text_tags.empty() ? text_tag : text_tags[row];
        result.text_indices[row] = row;
    }

    const double sqrt_area = std::sqrt(double(latent_height) * latent_width);
    auto heights = spatial_grid(latent_height, patch_size[1], sqrt_area);
    auto widths = spatial_grid(latent_width, patch_size[2], sqrt_area);
    for (int channel = 0; channel < audio_channels; ++channel)
        for (int time = 0; time < audio_latents; ++time) {
            int row = audio_start + channel * audio_latents + time;
            result.position_ids[size_t(row) * 3] = text_tokens + time;
            result.position_ids[size_t(row) * 3 + 2] =
                channel == 0 ? widths.front() : widths.back();
            result.token_tags[row] = audio_tag;
            result.audio_indices[row - audio_start] = row;
        }

    auto times = temporal_grid(latent_frames, text_tokens);
    int offset = 0;
    for (int frame = 0; frame < latent_frames; ++frame)
        for (double height : heights)
            for (double width : widths) {
                int row = video_start + offset;
                result.position_ids[size_t(row) * 3] = times[frame];
                result.position_ids[size_t(row) * 3 + 1] = height;
                result.position_ids[size_t(row) * 3 + 2] = width;
                result.token_tags[row] = video_tag;
                result.video_indices[offset++] = row;
            }
    return result;
}

RowTimesteps build_row_timesteps(const PackedLayout &layout, float video_timestep,
                                 float audio_timestep,
                                 float condition_video_timestep,
                                 float condition_audio_timestep) {
    require(layout.sequence_length > 0, "empty H3 packed layout");
    std::vector<float> rows(layout.sequence_length, video_timestep);
    for (int index = 0; index < layout.condition_video_rows; ++index)
        rows[layout.video_indices[index]] = condition_video_timestep;
    for (int index = layout.condition_audio_rows; index < int(layout.audio_indices.size()); ++index)
        rows[layout.audio_indices[index]] = audio_timestep;
    for (int index = 0; index < layout.condition_audio_rows; ++index)
        rows[layout.audio_indices[index]] = condition_audio_timestep;

    RowTimesteps result;
    result.unique = rows;
    std::sort(result.unique.begin(), result.unique.end());
    result.unique.erase(std::unique(result.unique.begin(), result.unique.end()), result.unique.end());
    result.inverse.resize(rows.size());
    for (size_t index = 0; index < rows.size(); ++index)
        result.inverse[index] = int32_t(std::lower_bound(
            result.unique.begin(), result.unique.end(), rows[index]) - result.unique.begin());
    return result;
}

Scheduler Scheduler::create(float shift_value, int steps) {
    require(shift_value > 0 && steps > 0, "invalid H3 scheduler parameters");
    Scheduler result;
    result.shift = shift_value;
    result.sigmas.reserve(steps + 1);
    for (int index = 0; index <= steps; ++index) {
        double base = 1.0 - double(index) / steps;
        result.sigmas.push_back(float(shift_value * base /
            (1.0 + (shift_value - 1.0) * base)));
    }
    result.timesteps.reserve(steps);
    for (int index = 0; index < steps; ++index)
        result.timesteps.push_back(1.f - result.sigmas[index]);
    return result;
}

std::vector<float> four_step_adaln_union() {
    auto video = Scheduler::create(video_shift, 4);
    auto audio = Scheduler::create(audio_shift, 4);
    std::vector<float> result{1.f};
    result.insert(result.end(), video.timesteps.begin(), video.timesteps.end());
    result.insert(result.end(), audio.timesteps.begin(), audio.timesteps.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

} // namespace tc::h3_mlx
