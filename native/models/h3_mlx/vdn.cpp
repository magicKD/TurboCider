#include "vdn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace tc::h3_mlx {
namespace {

int clamp_int(int value, int low, int high) {
    return value < low ? low : value > high ? high : value;
}

} // namespace

bool parse_vdn_anchor_frames(std::string_view name, VDNAnchorFrames *out) {
    VDNAnchorFrames value;
    if (name == "none") value = VDNAnchorFrames::none;
    else if (name == "columns") value = VDNAnchorFrames::columns;
    else if (name == "rows") value = VDNAnchorFrames::rows;
    else if (name == "both") value = VDNAnchorFrames::both;
    else return false;
    if (out) *out = value;
    return true;
}

const char *vdn_anchor_frames_name(VDNAnchorFrames mode) {
    switch (mode) {
        case VDNAnchorFrames::columns: return "columns";
        case VDNAnchorFrames::rows: return "rows";
        case VDNAnchorFrames::both: return "both";
        case VDNAnchorFrames::none: return "none";
    }
    return "none";
}

std::vector<VDNWindowBound> vdn_window_bounds(int num_frames, int radius,
                                               int chunk) {
    if (num_frames <= 0) return {};
    std::vector<VDNWindowBound> result;
    result.reserve(static_cast<size_t>(num_frames));
    for (int frame = 0; frame < num_frames; ++frame) {
        if (chunk <= 0) {
            result.push_back({frame - radius, frame + radius});
        } else {
            const int group = frame / chunk;
            result.push_back({(group - radius) * chunk,
                              (group + radius + 1) * chunk - 1});
        }
    }
    return result;
}

bool VDNWindowMask::allows(int query_row, int key_row) const {
    const int end = video_end();
    const bool query_video = query_row >= video_start && query_row < end;
    const bool key_video = key_row >= video_start && key_row < end;
    if (!(query_video && key_video)) return true;
    if (bounds.empty() || tokens_per_frame <= 0 || num_frames <= 0) return true;
    const int query_frame = clamp_int(
        (query_row - video_start) / tokens_per_frame, 0, num_frames - 1);
    const int key_frame = (key_row - video_start) / tokens_per_frame;
    const auto &bound = bounds.at(static_cast<size_t>(query_frame));
    if (key_frame >= bound.lo && key_frame <= bound.hi) return true;
    const bool key_anchor = key_frame == 0 || key_frame == num_frames - 1;
    const bool query_anchor = query_frame == 0 || query_frame == num_frames - 1;
    const bool anchor_columns = anchors == VDNAnchorFrames::columns ||
                                anchors == VDNAnchorFrames::both;
    const bool anchor_rows = anchors == VDNAnchorFrames::rows ||
                             anchors == VDNAnchorFrames::both;
    return (anchor_columns && key_anchor) || (anchor_rows && query_anchor);
}

VDNGatherIndex vdn_gather_indices(const std::vector<VDNWindowBound> &bounds,
                                  int num_frames) {
    VDNGatherIndex result;
    const size_t count = bounds.size();
    result.before_index.resize(count);
    result.after_index.resize(count);
    result.has_before.resize(count);
    result.has_after.resize(count);
    result.bridge_before.resize(count);
    result.bridge_after.resize(count);
    for (size_t frame = 0; frame < count; ++frame) {
        const int before = bounds[frame].lo - 1;
        const int after = bounds[frame].hi + 1;
        result.before_index[frame] = before < 0 ? 0 : before;
        result.after_index[frame] = std::min(after, num_frames - 1);
        result.has_before[frame] = static_cast<char>(before >= 0);
        result.has_after[frame] = static_cast<char>(after < num_frames);
        result.bridge_before[frame] = std::max(before + 1, 0);
        result.bridge_after[frame] = std::min(after, num_frames);
    }
    return result;
}

void vdn_alpha_bridge(const float *alpha, int num_frames, int heads,
                      int key_dim, const VDNGatherIndex &indices,
                      float *from_before, float *from_after) {
    if (num_frames <= 0 || heads <= 0 || key_dim <= 0) return;
    const size_t width = static_cast<size_t>(heads) * key_dim;
    std::vector<double> log_prefix(static_cast<size_t>(num_frames + 1) * width,
                                   0.0);
    for (int frame = 0; frame < num_frames; ++frame) {
        const double *previous = log_prefix.data() +
            static_cast<size_t>(frame) * width;
        double *current = log_prefix.data() +
            static_cast<size_t>(frame + 1) * width;
        const float *value = alpha + static_cast<size_t>(frame) * width;
        for (size_t i = 0; i < width; ++i)
            current[i] = previous[i] + std::log(std::max(value[i], 1e-12f));
    }
    for (int frame = 0; frame < num_frames; ++frame) {
        const double *end_before = log_prefix.data() +
            static_cast<size_t>(frame + 1) * width;
        const double *begin_before = log_prefix.data() +
            static_cast<size_t>(indices.bridge_before.at(frame)) * width;
        const double *end_after = log_prefix.data() +
            static_cast<size_t>(indices.bridge_after.at(frame)) * width;
        const double *begin_after = log_prefix.data() +
            static_cast<size_t>(frame) * width;
        float *before = from_before + static_cast<size_t>(frame) * width;
        float *after = from_after + static_cast<size_t>(frame) * width;
        for (size_t i = 0; i < width; ++i) {
            before[i] = static_cast<float>(std::exp(end_before[i] - begin_before[i]));
            after[i] = static_cast<float>(std::exp(end_after[i] - begin_after[i]));
        }
    }
}

void vdn_gather_linear_state(const float *prefix, const float *suffix,
                             const float *alpha,
                             const VDNGatherIndex &indices, int num_frames,
                             int heads, int value_dim, int key_dim,
                             bool bridge_alpha, const float *text_state,
                             float *out) {
    if (num_frames <= 0 || heads <= 0 || value_dim <= 0 || key_dim <= 0) return;
    const size_t matrix = static_cast<size_t>(value_dim) * key_dim;
    const size_t state_width = static_cast<size_t>(heads) * matrix;
    const size_t alpha_width = static_cast<size_t>(heads) * key_dim;
    std::vector<float> before_bridge;
    std::vector<float> after_bridge;
    if (bridge_alpha) {
        before_bridge.resize(static_cast<size_t>(num_frames) * alpha_width);
        after_bridge.resize(before_bridge.size());
        vdn_alpha_bridge(alpha, num_frames, heads, key_dim, indices,
                         before_bridge.data(), after_bridge.data());
    }
    for (int frame = 0; frame < num_frames; ++frame) {
        const float *before_state = prefix +
            static_cast<size_t>(indices.before_index.at(frame)) * state_width;
        const float *after_state = suffix +
            static_cast<size_t>(indices.after_index.at(frame)) * state_width;
        float *destination = out + static_cast<size_t>(frame) * state_width;
        for (int head = 0; head < heads; ++head) {
            const size_t base = static_cast<size_t>(head) * matrix;
            for (int value = 0; value < value_dim; ++value) {
                for (int key = 0; key < key_dim; ++key) {
                    const size_t index = base + static_cast<size_t>(value) * key_dim + key;
                    float left = before_state[index];
                    float right = after_state[index];
                    if (text_state) {
                        if (!indices.has_before.at(frame)) left = text_state[index];
                        if (!indices.has_after.at(frame)) right = text_state[index];
                    }
                    if (bridge_alpha) {
                        const size_t alpha_index = static_cast<size_t>(frame) * alpha_width +
                            static_cast<size_t>(head) * key_dim + key;
                        left *= before_bridge[alpha_index];
                        right *= after_bridge[alpha_index];
                    } else if (!text_state) {
                        left *= static_cast<float>(indices.has_before.at(frame) != 0);
                        right *= static_cast<float>(indices.has_after.at(frame) != 0);
                    }
                    destination[index] = left + right;
                }
            }
        }
    }
}

std::vector<VDNWindowBound> vdn_rebase_for_anchor_skip(
    const std::vector<VDNWindowBound> &bounds, int num_frames) {
    if (num_frames <= 2) return {};
    std::vector<VDNWindowBound> result;
    result.reserve(static_cast<size_t>(num_frames - 2));
    for (int frame = 1; frame < num_frames - 1; ++frame)
        result.push_back({bounds.at(static_cast<size_t>(frame)).lo - 1,
                          bounds.at(static_cast<size_t>(frame)).hi - 1});
    return result;
}

bool vdn_cholesky_lower(const float *a, int dimension, float *lower) {
    if (!a || !lower || dimension <= 0) return false;
    std::fill(lower, lower + static_cast<size_t>(dimension) * dimension, 0.f);
    for (int row = 0; row < dimension; ++row) {
        for (int column = 0; column <= row; ++column) {
            double sum = static_cast<double>(a[static_cast<size_t>(row) * dimension + column]);
            if (row == column) sum += 1.0;
            for (int k = 0; k < column; ++k)
                sum -= static_cast<double>(lower[static_cast<size_t>(row) * dimension + k]) *
                       lower[static_cast<size_t>(column) * dimension + k];
            if (row == column) {
                if (!(sum > 0.0)) return false;
                lower[static_cast<size_t>(row) * dimension + column] =
                    static_cast<float>(std::sqrt(sum));
            } else {
                const double diagonal = lower[static_cast<size_t>(column) * dimension + column];
                if (diagonal == 0.0) return false;
                lower[static_cast<size_t>(row) * dimension + column] =
                    static_cast<float>(sum / diagonal);
            }
        }
    }
    return true;
}

bool vdn_solve_state(const float *a, const float *alpha,
                     const float *state_in, const float *b, int key_dim,
                     int value_dim, float *state_out) {
    if (!a || !alpha || !b || !state_out || key_dim <= 0 || value_dim <= 0)
        return false;
    std::vector<float> lower(static_cast<size_t>(key_dim) * key_dim);
    if (!vdn_cholesky_lower(a, key_dim, lower.data())) return false;
    std::vector<double> rhs(static_cast<size_t>(key_dim) * value_dim);
    std::vector<double> solution(rhs.size());
    for (int value = 0; value < value_dim; ++value) {
        for (int key = 0; key < key_dim; ++key) {
            double sum = b[static_cast<size_t>(value) * key_dim + key];
            if (state_in) sum += static_cast<double>(state_in[static_cast<size_t>(value) * key_dim + key]) * alpha[key];
            rhs[static_cast<size_t>(key) * value_dim + value] = sum;
        }
    }
    for (int column = 0; column < value_dim; ++column) {
        for (int row = 0; row < key_dim; ++row) {
            double sum = rhs[static_cast<size_t>(row) * value_dim + column];
            for (int k = 0; k < row; ++k)
                sum -= static_cast<double>(lower[static_cast<size_t>(row) * key_dim + k]) *
                       solution[static_cast<size_t>(k) * value_dim + column];
            solution[static_cast<size_t>(row) * value_dim + column] =
                sum / lower[static_cast<size_t>(row) * key_dim + row];
        }
        for (int row = key_dim - 1; row >= 0; --row) {
            double sum = solution[static_cast<size_t>(row) * value_dim + column];
            for (int k = row + 1; k < key_dim; ++k)
                sum -= static_cast<double>(lower[static_cast<size_t>(k) * key_dim + row]) *
                       solution[static_cast<size_t>(k) * value_dim + column];
            solution[static_cast<size_t>(row) * value_dim + column] =
                sum / lower[static_cast<size_t>(row) * key_dim + row];
        }
    }
    for (int value = 0; value < value_dim; ++value)
        for (int key = 0; key < key_dim; ++key)
            state_out[static_cast<size_t>(value) * key_dim + key] =
                static_cast<float>(solution[static_cast<size_t>(key) * value_dim + value]);
    return true;
}

} // namespace tc::h3_mlx
