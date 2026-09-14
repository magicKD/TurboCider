#include "vdn_mlx.hpp"
#include "vdn_attention_metal.hpp"
#include "vdn_solve_metal.hpp"

#include "../../backends/mlx.hpp"

#include <mlx/linalg.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <numeric>
#include <set>
#include <vector>

namespace tc::h3_mlx {
namespace {

using ProfileClock = std::chrono::steady_clock;
thread_local int vdn_profile_current_block = -1;

/*
 * VDN's softmax mask is a small union of contiguous key spans.  The old MLX
 * path materialised one take/SDPA graph per frame group, which is convenient
 * as a reference but pays a graph and gather cost for every group.  This
 * kernel follows vpipe's span formulation: one SIMD group owns one
 * (head,query-row), walks the disjoint spans in order, and keeps the online
 * softmax state in registers.  It deliberately accepts the same BF16/FP16
 * activation dtype as the surrounding MLX graph and accumulates in FP32.
 */
inline constexpr const char *kVDNSpanSoftmaxHeader = R"METAL(
#include <metal_stdlib>
using namespace metal;

#define TC_VDN_MAX_PER 32
)METAL";

inline constexpr const char *kVDNSpanSoftmaxBody = R"METAL(
  const int h = (int)threadgroup_position_in_grid.y;
  const int qi = (int)threadgroup_position_in_grid.z;
  const int per = (D + 31) / 32;
  const int video_end = video_start + num_frames * tokens_per_frame;
  const int group = (qi < video_start || qi >= video_end)
      ? 0 : 1 + (qi - video_start) / tokens_per_frame;
  const device T *qh = q + ((uint)h * n_q + qi) * D;
  const device T *kh = k + (uint)h * n_q * D;
  const device T *vh = v + (uint)h * n_q * D;

  float qreg[TC_VDN_MAX_PER];
  float acc[TC_VDN_MAX_PER];
  for (int p = 0; p < per; ++p) {
    const int index = (int)thread_index_in_simdgroup * per + p;
    qreg[p] = index < D ? float(qh[index]) * scale : 0.0f;
    acc[p] = 0.0f;
  }

  float maximum = -INFINITY;
  float norm = 0.0f;
  for (int span = span_off[group]; span < span_off[group + 1]; ++span) {
    const int begin = span_start[span];
    const int end = span_end[span];
    for (int key = begin; key < end; ++key) {
      float dot = 0.0f;
      for (int p = 0; p < per; ++p) {
        const int index = (int)thread_index_in_simdgroup * per + p;
        if (index < D) dot += qreg[p] * float(kh[key * D + index]);
      }
      dot = simd_sum(dot);
      const float next_maximum = max(maximum, dot);
      const float correction = exp(maximum - next_maximum);
      const float probability = exp(dot - next_maximum);
      norm = norm * correction + probability;
      for (int p = 0; p < per; ++p) {
        const int index = (int)thread_index_in_simdgroup * per + p;
        if (index < D)
          acc[p] = acc[p] * correction + probability * float(vh[key * D + index]);
      }
      maximum = next_maximum;
    }
  }

  const float inv_norm = norm > 0.0f ? 1.0f / norm : 0.0f;
  device T *out_h = out + ((uint)h * n_q + qi) * D;
  for (int p = 0; p < per; ++p) {
    const int index = (int)thread_index_in_simdgroup * per + p;
    if (index < D) out_h[index] = T(acc[p] * inv_norm);
  }
)METAL";

bool vdn_profile_enabled() {
    static const bool enabled = std::getenv("TURBOCIDER_VDN_PROFILE") != nullptr;
    if (!enabled) return false;
    static const int selected_block = [] {
        const char *value = std::getenv("TURBOCIDER_VDN_PROFILE_BLOCK");
        return value && *value ? std::atoi(value) : -1;
    }();
    return selected_block < 0 || selected_block == vdn_profile_current_block;
}

void vdn_profile_eval(const char *stage, std::vector<Tensor> values) {
    if (!vdn_profile_enabled()) return;
    const auto started = ProfileClock::now();
    mx::eval(std::move(values));
    const double seconds = std::chrono::duration<double>(
        ProfileClock::now() - started).count();
    std::fprintf(stderr,
                 "{\"event\":\"h3_vdn_profile\",\"block\":%d,"
                 "\"phase\":\"%s\",\"seconds\":%.9f}\n",
                 vdn_profile_current_block, stage, seconds);
}

void vdn_profile_eval(const char *stage, const Tensor &value) {
    vdn_profile_eval(stage, std::vector<Tensor>{value});
}

class VDNProfileBlockScope {
    int previous_ = -1;

  public:
    explicit VDNProfileBlockScope(int block)
        : previous_(vdn_profile_current_block) {
        vdn_profile_current_block = block;
    }
    ~VDNProfileBlockScope() { vdn_profile_current_block = previous_; }
};

Tensor i32_indices(const std::vector<int32_t> &values) {
    return Tensor(values.data(), {int(values.size())}, mx::int32);
}

Tensor linear(const Tensor &x, const Tensor &weight) {
    require(weight.ndim() == 2 && x.shape(-1) == weight.shape(1),
            "invalid VDN linear projection geometry");
    return mx::matmul(x, mx::transpose(weight));
}

Tensor linear_fp32(const Tensor &x, const Tensor &weight) {
    return linear(mx::astype(x, mx::float32), mx::astype(weight, mx::float32));
}

Tensor silu(const Tensor &x) { return x * mx::sigmoid(x); }

struct VDNWindowSpans {
    std::vector<int32_t> offsets;
    std::vector<int32_t> starts;
    std::vector<int32_t> ends;
};

struct VDNBlockSpans {
    std::vector<int32_t> offsets;
    std::vector<int32_t> blocks;
};

void vdn_push_span(VDNWindowSpans *spans, size_t group_begin,
                   int begin, int end) {
    if (!spans || end <= begin) return;
    if (spans->starts.size() > group_begin &&
        begin <= spans->ends.back()) {
        if (end > spans->ends.back()) spans->ends.back() = end;
        return;
    }
    spans->starts.push_back(begin);
    spans->ends.push_back(end);
}

VDNWindowSpans vdn_build_window_spans(const PackedLayout &layout,
                                      int radius, int chunk,
                                      VDNAnchorFrames anchors) {
    const int sequence = layout.sequence_length;
    const int frames = layout.video_latent_frames;
    require(sequence > 0 && frames > 0 && !layout.video_indices.empty(),
            "invalid VDN span geometry");
    const int video_start = layout.video_indices.front();
    const int video_rows = int(layout.video_indices.size());
    require(video_start >= 0 && video_start + video_rows <= sequence &&
                video_rows % frames == 0,
            "VDN video rows must be a contiguous frame-major run");
    const int tokens = video_rows / frames;
    const int video_end = video_start + video_rows;
    const bool anchor_columns = anchors == VDNAnchorFrames::columns ||
                                anchors == VDNAnchorFrames::both;
    const bool anchor_rows = anchors == VDNAnchorFrames::rows ||
                             anchors == VDNAnchorFrames::both;
    const auto bounds = vdn_window_bounds(frames, radius, chunk);

    VDNWindowSpans result;
    result.offsets.reserve(static_cast<size_t>(frames) + 2u);
    result.offsets.push_back(0);
    vdn_push_span(&result, 0u, 0, sequence);
    result.offsets.push_back(static_cast<int32_t>(result.starts.size()));

    for (int frame = 0; frame < frames; ++frame) {
        const size_t group_begin = result.starts.size();
        const bool row_dense = anchor_rows &&
            (frame == 0 || frame == frames - 1);
        if (row_dense) {
            vdn_push_span(&result, group_begin, 0, sequence);
            result.offsets.push_back(static_cast<int32_t>(result.starts.size()));
            continue;
        }

        /* Leading/trailing text and audio rows are global. */
        vdn_push_span(&result, group_begin, 0, video_start);
        std::vector<std::pair<int, int>> frame_ranges;
        const auto &bound = bounds.at(static_cast<size_t>(frame));
        const int lo = std::max(0, bound.lo);
        const int hi = std::min(frames - 1, bound.hi);
        if (lo <= hi) frame_ranges.emplace_back(lo, hi);
        if (anchor_columns) {
            frame_ranges.emplace_back(0, 0);
            frame_ranges.emplace_back(frames - 1, frames - 1);
        }
        std::sort(frame_ranges.begin(), frame_ranges.end());
        for (const auto &range : frame_ranges) {
            vdn_push_span(&result, group_begin,
                          video_start + range.first * tokens,
                          video_start + (range.second + 1) * tokens);
        }
        vdn_push_span(&result, group_begin, video_end, sequence);
        result.offsets.push_back(static_cast<int32_t>(result.starts.size()));
    }
    require(result.offsets.size() == static_cast<size_t>(frames) + 2u,
            "VDN span offset count mismatch");
    return result;
}

VDNBlockSpans vdn_build_block_spans(const PackedLayout &layout,
                                    int radius, int chunk,
                                    VDNAnchorFrames anchors,
                                    int query_block, int key_block) {
    const int sequence = layout.sequence_length;
    const int frames = layout.video_latent_frames;
    const int video_start = layout.video_indices.front();
    const int video_rows = int(layout.video_indices.size());
    const int tokens = video_rows / frames;
    const int video_end = video_start + video_rows;
    const auto bounds = vdn_window_bounds(frames, radius, chunk);
    const bool anchor_columns = anchors == VDNAnchorFrames::columns ||
                                anchors == VDNAnchorFrames::both;
    const bool anchor_rows = anchors == VDNAnchorFrames::rows ||
                             anchors == VDNAnchorFrames::both;
    const int query_blocks = (sequence + query_block - 1) / query_block;
    const int key_blocks = (sequence + key_block - 1) / key_block;
    VDNBlockSpans result;
    result.offsets.reserve(static_cast<size_t>(query_blocks) + 1u);
    result.offsets.push_back(0);
    for (int qblock = 0; qblock < query_blocks; ++qblock) {
        const int q0 = qblock * query_block;
        const int q1 = std::min(sequence, q0 + query_block);
        bool dense = q0 < video_start || q1 > video_end;
        int frame0 = 0;
        int frame1 = 0;
        if (!dense) {
            frame0 = std::max(0, std::min(frames - 1,
                (q0 - video_start) / tokens));
            frame1 = std::max(0, std::min(frames - 1,
                (q1 - 1 - video_start) / tokens));
            if (anchor_rows) {
                for (int frame = frame0; frame <= frame1; ++frame)
                    if (frame == 0 || frame == frames - 1) dense = true;
            }
        }
        std::vector<std::pair<int, int>> ranges;
        if (dense) {
            ranges.emplace_back(0, sequence);
        } else {
            ranges.emplace_back(0, video_start);
            int lo = bounds[static_cast<size_t>(frame0)].lo;
            int hi = bounds[static_cast<size_t>(frame0)].hi;
            for (int frame = frame0 + 1; frame <= frame1; ++frame) {
                lo = std::min(lo, bounds[static_cast<size_t>(frame)].lo);
                hi = std::max(hi, bounds[static_cast<size_t>(frame)].hi);
            }
            lo = std::max(0, lo);
            hi = std::min(frames - 1, hi);
            if (lo <= hi)
                ranges.emplace_back(video_start + lo * tokens,
                                    video_start + (hi + 1) * tokens);
            if (anchor_columns) {
                ranges.emplace_back(video_start, video_start + tokens);
                ranges.emplace_back(video_end - tokens, video_end);
            }
            ranges.emplace_back(video_end, sequence);
        }
        std::vector<int> selected;
        for (const auto &range : ranges) {
            const int begin = std::max(0, range.first) / key_block;
            const int end = std::min(sequence, range.second);
            if (end <= range.first) continue;
            const int last = (end - 1) / key_block;
            for (int block = begin; block <= last && block < key_blocks; ++block)
                selected.push_back(block);
        }
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        result.blocks.insert(result.blocks.end(), selected.begin(), selected.end());
        result.offsets.push_back(static_cast<int32_t>(result.blocks.size()));
    }
    return result;
}

} // namespace

Tensor vdn_spd_inverse(const Tensor &a, bool cpu_reference) {
    require(a.ndim() >= 2 && a.shape(-1) == a.shape(-2) &&
                a.shape(-1) > 0 && a.shape(-1) <= 128,
            "invalid VDN SPD solve geometry");
    const int dim = a.shape(-1);
    const int batch = int(a.size() / (dim * dim));
    auto matrices = mx::contiguous(mx::reshape(mx::astype(a, mx::float32),
                                                {batch, dim, dim}));
    if (cpu_reference) {
        const auto cpu = mx::Device(mx::Device::cpu);
        auto eye = mx::reshape(mx::eye(dim, mx::float32), {1, dim, dim});
        auto lower = mx::linalg::cholesky(matrices + eye, false, cpu);
        auto inverse = mx::linalg::cholesky_inv(lower, false, cpu);
        return mx::reshape(inverse, a.shape());
    }

    static auto cholesky = mx::fast::metal_kernel(
        "tc_vdn_cholesky_f32", {"A", "d"}, {"L"},
        kVDNCholeskyBody, kVDNMetalHeader, true, false);
    static auto trinv = mx::fast::metal_kernel(
        "tc_vdn_trinv_f32", {"L", "d"}, {"X"},
        kVDNTrinvBody, kVDNMetalHeader, true, false);
    auto lower = cholesky(
        {matrices, Tensor(dim)}, {matrices.shape()}, {mx::float32},
        {batch * 128, 1, 1}, {128, 1, 1}, {}, {}, false, {})[0];
    const int block_columns = (dim + 31) / 32;
    auto lower_inverse = trinv(
        {lower, Tensor(dim)}, {matrices.shape()}, {mx::float32},
        {block_columns * 128, batch, 1}, {128, 1, 1}, {}, {}, false, {})[0];
    auto inverse = mx::matmul(mx::transpose(lower_inverse, {0, 2, 1}),
                              lower_inverse);
    vdn_profile_eval("solve", inverse);
    return mx::reshape(inverse, a.shape());
}

namespace {

Tensor spd_inverse(const Tensor &a) {
    return vdn_spd_inverse(a,
                           std::getenv("TURBOCIDER_VDN_CPU_SOLVE") != nullptr);
}

Tensor l2_normalize(const Tensor &x) {
    auto value = mx::astype(x, mx::float32);
    auto sum = mx::sum(value * value, -1, true);
    auto norm = mx::maximum(mx::sqrt(sum), Tensor(1e-6f, mx::float32));
    return value / norm;
}

Tensor depthwise_spatial(const Tensor &value, const Tensor &weight,
                         int frames, int grid_h, int grid_w, int channels) {
    // MLX uses NHWC and OHWI; the ModelScope branch stores OIHW.
    auto input = mx::reshape(value, {frames, grid_h, grid_w, channels});
    auto filter = mx::transpose(weight, {0, 2, 3, 1});
    auto output = mx::conv2d(input, filter, {1, 1}, {2, 2}, {1, 1}, channels);
    vdn_profile_eval("spatial_conv", output);
    return mx::reshape(output, {frames, grid_h * grid_w, channels});
}

Tensor depthwise_temporal(const Tensor &value, const Tensor &weight,
                          int frames, int tokens, int channels) {
    // Treat every spatial token as a batch item.  The branch's temporal
    // kernel is [channel, 1, 5] in OIK and MLX expects [channel, 5, 1].
    auto input = mx::transpose(mx::reshape(value, {frames, tokens, channels}),
                               {1, 0, 2});
    auto filter = mx::transpose(weight, {0, 2, 1});
    auto output = mx::conv1d(input, filter, 1, 2, 1, channels);
    vdn_profile_eval("temporal_conv", output);
    return mx::transpose(output, {1, 0, 2});
}

Tensor branch_features(const Tensor &raw, const Tensor &spatial,
                       const Tensor &temporal, int frames, int tokens,
                       int grid_h, int grid_w, int heads, int dim,
                       bool normalize) {
    const int channels = heads * dim;
    auto x = mx::reshape(raw, {frames * tokens, channels});
    x = depthwise_spatial(x, spatial, frames, grid_h, grid_w, channels);
    x = depthwise_temporal(x, temporal, frames, tokens, channels);
    x = mx::astype(silu(x), mx::float32);
    x = mx::reshape(x, {frames, tokens, heads, dim});
    return normalize ? l2_normalize(x) : x;
}

Tensor softmax_for_rows(const Tensor &q, const Tensor &k, const Tensor &v,
                        const Tensor &query_rows, const Tensor &key_rows,
                        int heads, int dim) {
    auto q_rows = mx::take(q, query_rows, 0);
    auto k_rows = mx::take(k, key_rows, 0);
    auto v_rows = mx::take(v, key_rows, 0);
    auto qh = mx::expand_dims(mx::transpose(q_rows, {1, 0, 2}), 0);
    auto kh = mx::expand_dims(mx::transpose(k_rows, {1, 0, 2}), 0);
    auto vh = mx::expand_dims(mx::transpose(v_rows, {1, 0, 2}), 0);
    auto result = mx::fast::scaled_dot_product_attention(
        qh, kh, vh, 1.f / std::sqrt(float(dim)));
    return mx::transpose(mx::squeeze(result, 0), {1, 0, 2});
}

struct VDNWindowIndexGroup {
    Tensor queries;
    Tensor keys;
};

struct VDNWindowIndexCache {
    int sequence = 0;
    int global = 0;
    int frames = 0;
    int tokens = 0;
    int chunk = 0;
    int radius = 0;
    std::vector<VDNWindowIndexGroup> groups;
};

const std::vector<VDNWindowIndexGroup> &window_index_groups(
    int sequence, int global, int frames, int tokens, int chunk, int radius) {
    static thread_local std::optional<VDNWindowIndexCache> cache;
    // Keep an explicit diagnostic escape hatch so the cache can be measured
    // against the former rebuild-on-every-block behavior in one binary.  The
    // production default is cached; this switch is never selected implicitly.
    static const bool disable_cache =
        std::getenv("TURBOCIDER_VDN_DISABLE_WINDOW_INDEX_CACHE") != nullptr;
    if (disable_cache) cache.reset();
    if (cache && cache->sequence == sequence && cache->global == global &&
        cache->frames == frames && cache->tokens == tokens &&
        cache->chunk == chunk && cache->radius == radius)
        return cache->groups;

    VDNWindowIndexCache next;
    next.sequence = sequence;
    next.global = global;
    next.frames = frames;
    next.tokens = tokens;
    next.chunk = chunk;
    next.radius = radius;
    next.groups.reserve(1 + (frames + chunk - 1) / chunk);
    if (global > 0) {
        std::vector<int32_t> queries(global);
        std::iota(queries.begin(), queries.end(), 0);
        std::vector<int32_t> keys(sequence);
        std::iota(keys.begin(), keys.end(), 0);
        next.groups.push_back({i32_indices(queries), i32_indices(keys)});
    }
    for (int first = 0; first < frames;) {
        const bool anchor_row = first == 0 || first == frames - 1;
        const int chunk_end = std::min(frames, (first / chunk + 1) * chunk);
        const int last = anchor_row ? first + 1
                                    : std::min(chunk_end, frames - 1);
        const int lo = anchor_row ? 0
                                  : std::max(0, (first / chunk - radius) * chunk);
        const int hi = anchor_row
                           ? frames
                           : std::min(frames,
                                      (first / chunk + radius + 1) * chunk);
        std::vector<int32_t> queries;
        std::vector<int32_t> keys;
        queries.reserve(static_cast<size_t>(last - first) * tokens);
        keys.reserve(static_cast<size_t>(global + (hi - lo + 2) * tokens));
        for (int frame = first; frame < last; ++frame)
            for (int token = 0; token < tokens; ++token)
                queries.push_back(global + frame * tokens + token);
        for (int row = 0; row < global; ++row) keys.push_back(row);
        for (int frame = lo; frame < hi; ++frame)
            for (int token = 0; token < tokens; ++token)
                keys.push_back(global + frame * tokens + token);
        for (int anchor : {0, frames - 1}) {
            if (anchor < lo || anchor >= hi) {
                for (int token = 0; token < tokens; ++token)
                    keys.push_back(global + anchor * tokens + token);
            }
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        next.groups.push_back({i32_indices(queries), i32_indices(keys)});
        first = last;
    }
    require(!next.groups.empty(), "VDN window softmax produced no groups");
    cache = std::move(next);
    return cache->groups;
}

Tensor window_softmax_reference(const Tensor &q, const Tensor &k, const Tensor &v,
                                const PackedLayout &layout, int chunk, int radius,
                                int heads, int dim) {
    const int sequence = q.shape(0);
    const int global = layout.video_indices.empty()
                           ? sequence
                           : layout.video_indices.front();
    const int frames = layout.video_latent_frames;
    const int tokens = int(layout.video_indices.size()) / std::max(frames, 1);
    require(frames > 0 && tokens > 0 && global + frames * tokens == sequence,
            "VDN packed video geometry is not contiguous");

    // Geometry stays fixed for all 50 blocks and all denoise steps. Cache the
    // host-built MLX index arrays instead of recreating them 300 times.
    const auto &index_groups = window_index_groups(
        sequence, global, frames, tokens, chunk, radius);
    std::vector<Tensor> groups;
    groups.reserve(index_groups.size());
    for (const auto &group : index_groups)
        groups.push_back(softmax_for_rows(
            q, k, v, group.queries, group.keys, heads, dim));
    auto result = groups.size() == 1 ? groups.front() : mx::concatenate(groups, 0);
    vdn_profile_eval("window_softmax", result);
    return result;
}

Tensor window_softmax_spans(const Tensor &q, const Tensor &k, const Tensor &v,
                            const PackedLayout &layout, int chunk, int radius,
                            VDNAnchorFrames anchors) {
    require(q.ndim() == 3 && k.shape() == q.shape() && v.shape() == q.shape(),
            "invalid VDN span attention tensors");
    const int sequence = q.shape(0);
    const int heads = q.shape(1);
    const int dim = q.shape(2);
    require(heads > 0 && dim > 0 && dim <= 1024,
            "invalid VDN span attention dimensions");
    const auto spans = vdn_build_window_spans(layout, radius, chunk, anchors);

    std::vector<int32_t> offsets = spans.offsets;
    std::vector<int32_t> starts = spans.starts;
    std::vector<int32_t> ends = spans.ends;
    auto span_offsets = Tensor(offsets.data(), {int(offsets.size())}, mx::int32);
    auto span_starts = Tensor(starts.data(), {int(starts.size())}, mx::int32);
    auto span_ends = Tensor(ends.data(), {int(ends.size())}, mx::int32);

    /* The custom kernel uses [heads, sequence, dim], one SIMD group per row. */
    auto q_heads = mx::contiguous(mx::transpose(q, {1, 0, 2}));
    auto k_heads = mx::contiguous(mx::transpose(k, {1, 0, 2}));
    auto v_heads = mx::contiguous(mx::transpose(v, {1, 0, 2}));
    static auto kernel = mx::fast::metal_kernel(
        "tc_vdn_sdpa_spans",
        {"q", "k", "v", "scale", "n_q", "D", "span_off",
         "span_start", "span_end", "video_start", "tokens_per_frame",
         "num_frames"},
        {"out"}, kVDNSpanSoftmaxBody, kVDNSpanSoftmaxHeader, true, false);
    const int video_start = layout.video_indices.empty()
                                ? sequence
                                : layout.video_indices.front();
    const int tokens = int(layout.video_indices.size()) /
                       std::max(layout.video_latent_frames, 1);
    auto output = kernel(
        {q_heads, k_heads, v_heads,
         Tensor(1.f / std::sqrt(float(dim))), Tensor(sequence), Tensor(dim),
         span_offsets, span_starts, span_ends, Tensor(video_start),
         Tensor(tokens), Tensor(layout.video_latent_frames)},
        {q_heads.shape()}, {q_heads.dtype()},
        {32, heads, sequence}, {32, 1, 1}, {{"T", q_heads.dtype()}}, {},
        false, {})[0];
    auto result = mx::transpose(output, {1, 0, 2});
    vdn_profile_eval("window_softmax", result);
    return result;
}

Tensor window_softmax_mma_spans(const Tensor &q, const Tensor &k,
                                const Tensor &v,
                                const PackedLayout &layout,
                                int chunk, int radius,
                                VDNAnchorFrames anchors) {
    require(q.ndim() == 3 && k.shape() == q.shape() && v.shape() == q.shape(),
            "invalid VDN MMA span attention tensors");
    const int sequence = q.shape(0);
    const int heads = q.shape(1);
    const int dim = q.shape(2);
    require(heads > 0 && dim == 128,
            "VDN MMA span attention requires head dimension 128");
    constexpr int query_block = 32;
    constexpr int key_block = 16;
    const int query_blocks = (sequence + query_block - 1) / query_block;
    const auto spans = vdn_build_block_spans(
        layout, radius, chunk, anchors, query_block, key_block);
    auto block_offsets = Tensor(spans.offsets.data(),
                                {int(spans.offsets.size())}, mx::int32);
    auto block_ids = Tensor(spans.blocks.data(),
                            {int(spans.blocks.size())}, mx::int32);
    const auto bounds = vdn_window_bounds(layout.video_latent_frames,
                                          radius, chunk);
    std::vector<int32_t> packed_bounds;
    packed_bounds.reserve(bounds.size() * 2u);
    for (const auto &bound : bounds) {
        packed_bounds.push_back(bound.lo);
        packed_bounds.push_back(bound.hi);
    }
    auto bound_values = Tensor(packed_bounds.data(),
                               {int(packed_bounds.size())}, mx::int32);
    auto q_heads = mx::contiguous(mx::transpose(q, {1, 0, 2}));
    auto k_heads = mx::contiguous(mx::transpose(k, {1, 0, 2}));
    auto v_heads = mx::contiguous(mx::transpose(v, {1, 0, 2}));
    static auto kernel = mx::fast::metal_kernel(
        "tc_vdn_sdpa_mma_block_spans",
        {"q", "k", "v", "scale", "n_q", "block_off", "block_ids",
         "video_start", "video_end", "tokens_per_frame", "num_frames",
         "bounds"},
        {"out"}, kVDNMmaBody, kVDNMmaHeader, true, false);
    const int video_start = layout.video_indices.front();
    const int video_end = video_start + int(layout.video_indices.size());
    const int tokens = int(layout.video_indices.size()) /
                       layout.video_latent_frames;
    auto output = kernel(
        {q_heads, k_heads, v_heads,
         Tensor(1.f / std::sqrt(float(dim))), Tensor(sequence),
         block_offsets, block_ids, Tensor(video_start), Tensor(video_end),
         Tensor(tokens), Tensor(layout.video_latent_frames), bound_values},
        {q_heads.shape()}, {q_heads.dtype()},
        {128, heads, query_blocks}, {128, 1, 1}, {{"T", q_heads.dtype()}},
        {}, false, {})[0];
    auto result = mx::transpose(output, {1, 0, 2});
    vdn_profile_eval("window_softmax_mma", result);
    return result;
}

Tensor window_softmax(const Tensor &q, const Tensor &k, const Tensor &v,
                      const PackedLayout &layout, int chunk, int radius,
                      int heads, int dim) {
    const auto anchors = VDNAnchorFrames::both;
    if (std::getenv("TURBOCIDER_VDN_EXPERIMENTAL_MMA_SPANS"))
        return window_softmax_mma_spans(q, k, v, layout, chunk, radius,
                                        anchors);
    if (std::getenv("TURBOCIDER_VDN_EXPERIMENTAL_SCALAR_SPANS"))
        return window_softmax_spans(q, k, v, layout, chunk, radius, anchors);
    return window_softmax_reference(q, k, v, layout, chunk, radius,
                                    heads, dim);
}

Tensor alpha_from_mean(const Checkpoint &weights, const Tensor &x,
                       const std::string &prefix, int frames, int tokens,
                       const Tensor &a_log, const Tensor &dt_bias) {
    auto mean = mx::mean(mx::reshape(x, {frames, tokens, x.shape(-1)}), 1);
    auto down = linear_fp32(mean, weights.vdn_at(prefix + "alpha.down.weight"));
    auto up = linear_fp32(down, weights.vdn_at(prefix + "alpha.up.weight"));
    const int heads = a_log.shape(0);
    const int dim = dt_bias.shape(0) / heads;
    auto delta = mx::reshape(up + mx::astype(dt_bias, mx::float32),
                             {frames, heads, dim});
    auto scale = mx::reshape(mx::exp(mx::astype(a_log, mx::float32)),
                             {1, heads, 1});
    auto alpha = mx::exp(-scale * mx::logaddexp(delta, Tensor(0.f)));
    return alpha;
}

std::pair<Tensor, Tensor> frame_stats(const Tensor &key, const Tensor &value,
                                     const Tensor &beta, bool a_fp32) {
    auto k = mx::transpose(key, {0, 2, 1, 3});
    auto v = mx::transpose(value, {0, 2, 1, 3});
    auto b = mx::expand_dims(mx::transpose(beta, {0, 2, 1}), -1);
    auto weighted_k = k * b;
    auto weighted_v = v * b;
    auto a = mx::matmul(mx::transpose(weighted_k, {0, 1, 3, 2}), k);
    auto bt = mx::matmul(mx::transpose(weighted_v, {0, 1, 3, 2}), k);
    a = (a + mx::transpose(a, {0, 1, 3, 2})) * 0.5f;
    if (a_fp32) a = mx::astype(a, mx::float32);
    return {std::move(a), mx::astype(bt, mx::float32)};
}

Tensor text_state(const Checkpoint &weights, const Tensor &x,
                  const Tensor &k_raw, const Tensor &v_raw,
                  const std::string &prefix, int heads, int dim) {
    const int tokens = x.shape(0);
    auto key = l2_normalize(mx::reshape(silu(k_raw), {1, tokens, heads, dim}));
    auto value = mx::reshape(silu(v_raw), {1, tokens, heads, dim});
    auto beta = mx::sigmoid(linear(x, weights.vdn_at(prefix + "beta_proj.weight")));
    auto [a, b] = frame_stats(key, value,
                              mx::reshape(beta, {1, tokens, heads}), true);
    auto inv = spd_inverse(a);
    auto state = mx::matmul(b, inv);
    state = mx::reshape(state * 0.5f, {heads, dim, dim});
    vdn_profile_eval("text_state", state);
    return state;
}

Tensor scan_states(const Tensor &transition, const Tensor &injection,
                   const Tensor &initial, bool reverse) {
    const int frames = transition.shape(0);
    const int heads = transition.shape(1);
    const int dim = transition.shape(2);
    require(transition.ndim() == 4 && injection.shape() == transition.shape() &&
                initial.ndim() == 3 && initial.shape(0) == heads &&
                initial.shape(1) == dim && initial.shape(2) == dim,
            "invalid VDN scan geometry");
    std::vector<Tensor> states;
    states.reserve(static_cast<size_t>(frames));
    auto state = initial;
    for (int index = 0; index < frames; ++index) {
        const int frame = reverse ? frames - 1 - index : index;
        state = mx::matmul(state, mx::take(transition, frame, 0)) +
                mx::take(injection, frame, 0);
        states.push_back(state);
    }
    if (reverse) std::reverse(states.begin(), states.end());
    auto result = mx::stack(states, 0);
    vdn_profile_eval(reverse ? "scan_reverse" : "scan_forward", result);
    return result;
}

Tensor bridge_product(const Tensor &alpha, int begin, int end) {
    if (end <= begin)
        return mx::ones({alpha.shape(1), alpha.shape(2)}, mx::float32);
    auto value = mx::ones({alpha.shape(1), alpha.shape(2)}, mx::float32);
    for (int index = begin; index < end; ++index)
        value = value * mx::take(alpha, index, 0);
    return value;
}

Tensor gather_states(const Tensor &prefix, const Tensor &suffix,
                     const Tensor &alpha, const Tensor &text,
                     const std::vector<VDNWindowBound> &bounds,
                     bool bridge, int frames, int heads, int dim) {
    std::vector<Tensor> output;
    output.reserve(frames);
    for (int frame = 0; frame < frames; ++frame) {
        const int before = std::max(0, bounds[frame].lo - 1);
        const int after = std::min(frames - 1, bounds[frame].hi + 1);
        const bool has_before = bounds[frame].lo - 1 >= 0;
        const bool has_after = bounds[frame].hi + 1 < frames;
        auto left = has_before ? mx::take(prefix, before, 0) : text;
        auto right = has_after ? mx::take(suffix, after, 0) : text;
        if (bridge) {
            const int left_begin = std::max(bounds[frame].lo, 0);
            const int right_end = std::min(bounds[frame].hi + 1, frames);
            left = left * mx::expand_dims(
                bridge_product(alpha, left_begin, frame + 1), 1);
            right = right * mx::expand_dims(
                bridge_product(alpha, frame, right_end), 1);
        }
        output.push_back(left + right);
    }
    (void)heads;
    (void)dim;
    return mx::stack(output, 0);
}

Tensor linear_readout(const Checkpoint &weights, const Tensor &x,
                      const Tensor &q_raw, const Tensor &k_raw,
                      const Tensor &v_raw, const PackedLayout &layout,
                      const std::string &prefix, int heads, int dim) {
    const int frames = layout.video_latent_frames;
    const int tokens = int(layout.video_indices.size()) / frames;
    const int global = layout.video_indices.front();
    const int grid_h = layout.latent_height / 2;
    const int grid_w = layout.latent_width / 2;
    require(grid_h * grid_w == tokens, "VDN spatial geometry mismatch");
    require(frames >= 3, "VDN requires at least three latent video frames");
    const int inner_frames = frames - 2;
    const int inner_rows = inner_frames * tokens;
    auto x_inner = slice_axis(x, 0, global + tokens, global + (frames - 1) * tokens);
    auto q_inner = slice_axis(q_raw, 0, global + tokens, global + (frames - 1) * tokens);
    auto k_inner = slice_axis(k_raw, 0, global + tokens, global + (frames - 1) * tokens);
    auto v_inner = slice_axis(v_raw, 0, global + tokens, global + (frames - 1) * tokens);
    auto base = prefix + "linear_attention.";
    auto k = branch_features(k_inner, weights.vdn_at(base + "short_conv.k_sp.weight"),
                             weights.vdn_at(base + "short_conv.k_tm.weight"),
                             inner_frames, tokens, grid_h, grid_w, heads, dim, true);
    auto v = branch_features(v_inner, weights.vdn_at(base + "short_conv.v_sp.weight"),
                             weights.vdn_at(base + "short_conv.v_tm.weight"),
                             inner_frames, tokens, grid_h, grid_w, heads, dim, false);
    auto q = l2_normalize(mx::reshape(silu(q_inner), {inner_frames, tokens, heads, dim}));
    auto beta = mx::sigmoid(linear(mx::reshape(x_inner, {inner_rows, x.shape(-1)}),
                                   weights.vdn_at(base + "beta_proj.weight")));
    beta = mx::reshape(beta, {inner_frames, tokens, heads});
    vdn_profile_eval("linear_features", {k, v, q, beta});
    auto [a, b] = frame_stats(k, v, beta, true);
    vdn_profile_eval("frame_statistics", {a, b});
    auto inv = spd_inverse(a);
    auto alpha = alpha_from_mean(weights, x_inner, base, inner_frames, tokens,
                                 weights.vdn_at(base + "alpha.A_log"),
                                 weights.vdn_at(base + "alpha.dt_bias"));
    vdn_profile_eval("alpha", alpha);
    // S_out = (S_in Diag(alpha) + B) (I + A)^-1, so the transition is
    // Diag(alpha) times the inverse.  Alpha therefore scales inverse ROWS.
    // Inserting the singleton after the key dimension would scale columns and
    // silently compute inverse * Diag(alpha), which is a different recurrence.
    auto transition = mx::expand_dims(alpha, 3) * inv;
    auto injection = mx::matmul(b, inv);
    vdn_profile_eval("state_parameters", {transition, injection});

    Tensor initial = mx::zeros({heads, dim, dim}, mx::float32);
    if (weights.identity().vdn.enable_text_state && global > 0)
        initial = text_state(weights,
                             slice_axis(x, 0, 0, int(layout.text_indices.size())),
                             slice_axis(k_raw, 0, 0, int(layout.text_indices.size())),
                             slice_axis(v_raw, 0, 0, int(layout.text_indices.size())), base,
                             heads, dim);
    auto prefix_state = scan_states(transition, injection, initial, false);
    auto suffix_state = scan_states(transition, injection, initial, true);
    auto bounds = vdn_rebase_for_anchor_skip(
        vdn_window_bounds(frames, weights.identity().vdn.softmax_radius,
                          weights.identity().vdn.softmax_chunk), frames);
    auto state = gather_states(prefix_state, suffix_state, alpha, initial,
                               bounds, weights.identity().vdn.bridge_alpha,
                               inner_frames, heads, dim);
    vdn_profile_eval("state_gather", state);
    auto qh = mx::transpose(q, {0, 2, 1, 3});
    auto q_column = mx::expand_dims(qh, -1);
    auto state_h = mx::expand_dims(state, 2);
    auto read = mx::matmul(state_h, q_column);
    read = mx::squeeze(read, -1);
    read = mx::transpose(read, {0, 2, 1, 3});
    auto norm = mx::rsqrt(mx::mean(read * read, -1, true) + 1e-6f);
    read = read * norm;
    read = read * mx::reshape(mx::astype(weights.vdn_at(base + "norm.weight"),
                                         mx::float32),
                              {1, 1, 1, dim});
    vdn_profile_eval("readout_norm", read);
    auto gate_down = linear(mx::reshape(x_inner, {inner_rows, x.shape(-1)}),
                            weights.vdn_at(base + "output_gate.down.weight"));
    auto gate_up = linear(gate_down,
                          weights.vdn_at(base + "output_gate.up.weight"));
    gate_up = gate_up + weights.vdn_at(base + "output_gate.up.bias");
    auto gate = mx::sigmoid(mx::reshape(gate_up, {inner_frames, tokens, heads, dim}));
    vdn_profile_eval("output_gate", gate);
    read = read * gate;
    auto read_rows = mx::reshape(read, {inner_rows, heads * dim});
    /* The compact form reduces the peak temporary, but on some MLX releases
     * its extra concatenate inputs cost a little wall time.  Keep it explicit
     * and opt-in until a production-size memory/speed matrix justifies it. */
    std::optional<Tensor> result;
    if (std::getenv("TURBOCIDER_VDN_COMPACT_LINEAR_OUTPUT")) {
        auto leading_zeros = mx::zeros(
            {global + tokens, heads * dim}, read_rows.dtype());
        auto trailing_zeros = mx::zeros(
            {tokens, heads * dim}, read_rows.dtype());
        result = mx::concatenate(
            {leading_zeros, read_rows, trailing_zeros}, 0);
    } else {
        // The branch output is video-only and leaves both anchor frames zero.
        auto zeros = mx::zeros({global + frames * tokens, heads * dim},
                               read_rows.dtype());
        result = mx::concatenate({slice_axis(zeros, 0, 0, global + tokens),
                                  read_rows,
                                  slice_axis(zeros, 0,
                                             global + (frames - 1) * tokens,
                                             global + frames * tokens)}, 0);
    }
    vdn_profile_eval("linear_readout", *result);
    return *result;
}

} // namespace

Tensor vdn_window_softmax_for_test(const Tensor &q, const Tensor &k,
                                   const Tensor &v,
                                   const PackedLayout &layout,
                                   int chunk, int radius,
                                   bool reference) {
    const int heads = q.ndim() == 3 ? q.shape(1) : 0;
    const int dim = q.ndim() == 3 ? q.shape(2) : 0;
    return reference
        ? window_softmax_reference(q, k, v, layout, chunk, radius,
                                   heads, dim)
        : window_softmax_spans(q, k, v, layout, chunk, radius,
                               VDNAnchorFrames::both);
}

Tensor vdn_window_softmax_mma_for_test(const Tensor &q, const Tensor &k,
                                       const Tensor &v,
                                       const PackedLayout &layout,
                                       int chunk, int radius) {
    return window_softmax_mma_spans(q, k, v, layout, chunk, radius,
                                    VDNAnchorFrames::both);
}

VDNForwardOutput vdn_forward(const Checkpoint &weights, const Tensor &x,
                             const Tensor &q_raw, const Tensor &k_raw,
                             const Tensor &v_raw, const Tensor &q_softmax,
                             const Tensor &k_softmax, const Tensor &v_softmax,
                             const PackedLayout &layout, int block_index) {
    VDNProfileBlockScope profile_scope(block_index);
    const auto &config = weights.config();
    const int heads = config.num_heads;
    const int dim = config.head_dim;
    const auto prefix = "transformer_blocks." + std::to_string(block_index) +
                        ".attn.";
    vdn_profile_eval("base_qkv", {q_raw, k_raw, v_raw, q_softmax, k_softmax,
                                   v_softmax});
    auto attended = window_softmax(
        q_softmax, k_softmax, v_softmax, layout,
        weights.identity().vdn.softmax_chunk,
        weights.identity().vdn.softmax_radius, heads, dim);
    auto gate = mx::sigmoid(
        linear(x, weights.vdn_at("transformer_blocks." +
                                std::to_string(block_index) +
                                ".attn.softmax_gate.up.weight")) +
        weights.vdn_at("transformer_blocks." + std::to_string(block_index) +
                       ".attn.softmax_gate.up.bias"));
    attended = attended * mx::expand_dims(gate, -1);
    vdn_profile_eval("softmax_gate", attended);
    return {std::move(attended), linear_readout(
        weights, x, q_raw, k_raw, v_raw, layout, prefix, heads, dim)};
}

} // namespace tc::h3_mlx
