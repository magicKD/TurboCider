#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace tc::ane {

// Scheduling is independent of Core ML and MLX. One state per layer/row count
// prevents a short decode or unusually slow refiner from tuning long prefill.
class RowScheduler {
    // A visit is one layer at one row count, not one arbitrary model block.
    // Reprobing every eight visits made a short diffusion request repeatedly
    // pay full-GPU probes after a stable split had already been measured.
    static constexpr int kReprobeVisits = 32;
    // These margins compare COMPLETE blocks, including attention/staging,
    // not isolated FFNs. Raising them to a 5%/8% full-block win discarded
    // useful Z-Image splits without improving Qwen LoRA end-to-end latency.
    // Keep the measured 2%/5% policy; the 5% partition-change gate below is
    // a separate decision. Fixed partitions bypass both on/off thresholds.
    static constexpr double kKeepHybridBlockRatio = .98;
    static constexpr double kReenableHybridBlockRatio = .95;
    struct State {
        int visits = 0, chunks = 1, gpu_samples = 0;
        double hybrid = 0, gpu = 0, gpu_row = 0, ane_chunk = 0;
        bool disabled = false;
        std::set<int> warmed_chunks;
    };
    std::map<std::pair<int, int>, State> states_;
    int chunk_, fixed_chunks_;
    static double ema(double old, double value) { return old ? .75 * old + .25 * value : value; }
  public:
    enum class Mode { Hybrid, HybridUntimed, GpuProbe, SplitProbe, Gpu };
    struct Plan {
        Mode mode;
        int chunks;
        bool split() const {
            return mode == Mode::Hybrid || mode == Mode::HybridUntimed || mode == Mode::SplitProbe;
        }
        bool measured() const { return mode != Mode::Gpu && mode != Mode::HybridUntimed; }
    };
    // fixed_chunks: -1 adaptive, 0 GPU-only split-boundary ablation, >0 fixed.
    RowScheduler(int chunk, int fixed_chunks = -1) : chunk_(chunk), fixed_chunks_(fixed_chunks) {}
    Plan plan(int layer, int rows) {
        const int max_chunks = (rows - 1) / chunk_;
        // The explicit zero-chunk ablation keeps the measured split boundary.
        // An unprofitable/too-short automatic route must instead be able to
        // run the family's ordinary unsplit GPU block, without a timing fence.
        if (fixed_chunks_ == 0) return {Mode::SplitProbe, 0};
        if (max_chunks < 1) return {Mode::Gpu, 0};
        if (fixed_chunks_ > 0) return {Mode::Hybrid, std::min(fixed_chunks_, max_chunks)};
        auto &s = states_[{layer, rows}];
        ++s.visits;
        // Warm two hybrid executions, then measure GPU on the same layer.
        // A disabled layer retries after 32 visits; an enabled one refreshes
        // the GPU baseline at the same interval. No duplicate FFN is needed.
        if (s.visits == 3 || s.visits == 4 ||
            (s.visits > 4 && s.visits % kReprobeVisits == 3)) return {Mode::GpuProbe, 0};
        if (s.disabled && s.visits % kReprobeVisits != 4) return {Mode::Gpu, 0};
        // Once both routes and the current partition have useful samples,
        // keep steady hybrid blocks free of whole-block timing fences. Sample
        // around each GPU reprobe, and continuously while a changed partition
        // still needs warmup/measurement. Stage-only callers remain compatible
        // through select(): they keep their existing in-FFN observation path.
        if (!s.disabled && s.visits > 6 && s.gpu && s.hybrid &&
                s.visits % kReprobeVisits != 2 && s.visits % kReprobeVisits != 4)
            return {Mode::HybridUntimed, std::min(s.chunks, max_chunks)};
        return {Mode::Hybrid, std::min(s.chunks, max_chunks)};
    }
    // Compatibility for callers that still keep a split GPU path. Do not call
    // both plan and select for one visit: each advances the controller once.
    int select(int layer, int rows) { return plan(layer, rows).chunks; }
    void observe(int layer, int rows, int chunks, double wall,
                 double gpu_seconds = 0, double ane_seconds = 0) {
        if (fixed_chunks_ >= 0 || !std::isfinite(wall) || wall <= 0) return;
        auto &s = states_[{layer, rows}];
        if (chunks == 0) {
            // Exclude each shape's first GPU-only call (possible compilation).
            if (++s.gpu_samples > 1) s.gpu = ema(s.gpu, wall);
        } else {
            // A different split changes the GPU head shape and may compile
            // a new kernel. Warm each split once, not just the first one.
            if (s.warmed_chunks.insert(chunks).second) return;
            s.hybrid = ema(s.hybrid, wall);
            if (std::isfinite(gpu_seconds) && std::isfinite(ane_seconds) &&
                gpu_seconds > 0 && ane_seconds > 0) {
                s.gpu_row = ema(s.gpu_row, gpu_seconds / (rows - chunks * chunk_));
                s.ane_chunk = ema(s.ane_chunk, ane_seconds / chunks);
                int best = chunks;
                auto cost = [&](int c) { return std::max(s.gpu_row * (rows - c * chunk_), s.ane_chunk * c); };
                for (int c = 1; c <= (rows - 1) / chunk_; ++c)
                    if (cost(c) < cost(best)) best = c;
                if (best != chunks && cost(best) < .95 * cost(chunks)) {
                    s.chunks = best;
                    // The old, slower partition must not disable a newly
                    // selected candidate before that candidate is measured.
                    // This matters at long sequences, where one initial
                    // chunk offloads too little work to overcome staging.
                    s.hybrid = 0;
                }
            }
        }
        if (s.gpu && s.hybrid) {
            if (!s.disabled && s.hybrid >= kKeepHybridBlockRatio * s.gpu) s.disabled = true;
            else if (s.disabled && chunks && s.hybrid < kReenableHybridBlockRatio * s.gpu) s.disabled = false;
        }
    }
};

} // namespace tc::ane
