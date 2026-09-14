#include "../../native/models/h3_mlx/vdn.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    using namespace tc::h3_mlx;

    VDNAnchorFrames anchors;
    assert(parse_vdn_anchor_frames("both", &anchors));
    assert(anchors == VDNAnchorFrames::both);
    assert(std::string(vdn_anchor_frames_name(anchors)) == "both");
    assert(!parse_vdn_anchor_frames("invalid", &anchors));

    const auto bounds = vdn_window_bounds(12, 1, 5);
    assert(bounds.size() == 12);
    assert((bounds[0] == VDNWindowBound{-5, 9}));
    assert((bounds[4] == VDNWindowBound{-5, 9}));
    assert((bounds[5] == VDNWindowBound{0, 14}));
    assert((bounds[11] == VDNWindowBound{5, 19}));

    VDNWindowMask mask{30, 3, 12, 2, VDNAnchorFrames::both, bounds};
    assert(mask.allows(0, 19));  // global rows remain dense.
    assert(mask.allows(3, 25));  // first video frame sees the final anchor.
    assert(mask.allows(9, 25));  // anchor column is dense under kBoth.
    assert(!mask.allows(5, 23)); // frame 10 is outside frame 1's chunk window.

    const auto gather = vdn_gather_indices(
        vdn_window_bounds(7, 1, 2), 7);
    assert(gather.before_index.size() == 7);
    assert(gather.before_index[0] == 0 && !gather.has_before[0]);
    assert(gather.after_index[6] == 6 && !gather.has_after[6]);
    assert(gather.bridge_before[0] == 0 && gather.bridge_after[6] == 7);

    const int frames = 3;
    const int heads = 1;
    const int dim = 2;
    const auto simple = vdn_window_bounds(frames, 0, 0);
    const auto simple_gather = vdn_gather_indices(simple, frames);
    std::vector<float> alpha{0.5f, 0.25f, 0.5f, 0.25f, 0.5f, 0.25f};
    std::vector<float> before(6), after(6);
    vdn_alpha_bridge(alpha.data(), frames, heads, dim, simple_gather,
                     before.data(), after.data());
    assert(std::abs(before[0] - 0.5f) < 1e-6f);
    assert(std::abs(after[4] - 0.5f) < 1e-6f);

    const int key_dim = 2;
    const int value_dim = 2;
    const float a[] = {1.f, 0.f, 0.f, 1.f};
    const float b[] = {2.f, 0.f, 0.f, 4.f};
    const float decay[] = {0.5f, 0.25f};
    const float state[] = {1.f, 2.f, 3.f, 4.f};
    float solved[4]{};
    assert(vdn_solve_state(a, decay, state, b, key_dim, value_dim, solved));
    assert(std::abs(solved[0] - 1.25f) < 1e-6f);
    assert(std::abs(solved[1] - 0.25f) < 1e-6f);
    assert(std::abs(solved[2] - 0.75f) < 1e-6f);
    assert(std::abs(solved[3] - 2.5f) < 1e-6f);

    const auto rebased = vdn_rebase_for_anchor_skip(bounds, 12);
    assert(rebased.size() == 10);
    assert((rebased.front() == VDNWindowBound{-6, 8}));
    assert((rebased.back() == VDNWindowBound{4, 18}));
}
