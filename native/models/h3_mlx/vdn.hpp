#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace tc::h3_mlx {

// VDN-H3 hybrid-attention geometry.  This is intentionally independent of
// MLX tensors so it can be used as a small CPU oracle before the Metal path is
// enabled.  The implementation follows the published OpenVDN stage-DMD v2
// contract; it does not link against the reference implementation at runtime.
enum class VDNAnchorFrames { none, columns, rows, both };

bool parse_vdn_anchor_frames(std::string_view name, VDNAnchorFrames *out);
const char *vdn_anchor_frames_name(VDNAnchorFrames mode);

struct VDNWindowBound {
    int lo = 0;
    int hi = 0;
    bool operator==(const VDNWindowBound &other) const {
        return lo == other.lo && hi == other.hi;
    }
};

std::vector<VDNWindowBound> vdn_window_bounds(int num_frames, int radius,
                                               int chunk);

struct VDNWindowMask {
    int sequence_length = 0;
    int video_start = 0;
    int num_frames = 0;
    int tokens_per_frame = 0;
    VDNAnchorFrames anchors = VDNAnchorFrames::none;
    std::vector<VDNWindowBound> bounds;

    int video_end() const {
        return video_start + num_frames * tokens_per_frame;
    }
    bool allows(int query_row, int key_row) const;
};

struct VDNGatherIndex {
    std::vector<int> before_index;
    std::vector<int> after_index;
    std::vector<char> has_before;
    std::vector<char> has_after;
    std::vector<int> bridge_before;
    std::vector<int> bridge_after;
};

VDNGatherIndex vdn_gather_indices(const std::vector<VDNWindowBound> &bounds,
                                  int num_frames);

void vdn_alpha_bridge(const float *alpha, int num_frames, int heads,
                      int key_dim, const VDNGatherIndex &indices,
                      float *from_before, float *from_after);

void vdn_gather_linear_state(const float *prefix, const float *suffix,
                             const float *alpha,
                             const VDNGatherIndex &indices, int num_frames,
                             int heads, int value_dim, int key_dim,
                             bool bridge_alpha, const float *text_state,
                             float *out);

std::vector<VDNWindowBound> vdn_rebase_for_anchor_skip(
    const std::vector<VDNWindowBound> &bounds, int num_frames);

// CPU reference for the per-frame vdn_solve recurrence.  It is deliberately
// kept small and deterministic; the MLX/Metal implementation will be checked
// against it on tiny fixtures.
bool vdn_cholesky_lower(const float *a, int dimension, float *lower);
bool vdn_solve_state(const float *a, const float *alpha,
                     const float *state_in, const float *b, int key_dim,
                     int value_dim, float *state_out);

} // namespace tc::h3_mlx
