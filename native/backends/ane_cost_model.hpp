#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc::ane {

// CPU-only shared policy, independent of Core ML/Private/MLX. Times are
// measured complete GPU-side work / independent ANE / both host spans, NOT
// exposed finish waits. Units are seconds throughout.
struct CalibrationPoint {
    double share = 0, gpu = 0, ane = 0, both = 0;
};
inline double calibration_layer_seconds(double one, double four) {
    if (!std::isfinite(one) || !std::isfinite(four) || one <= 0 || four <= one)
        throw std::invalid_argument("calibration requires positive ordered one/four layer spans");
    return (four - one) / 3;
}
struct ChannelCostDecision {
    int ane_channels = 0;
    double gpu_seconds = 0, predicted_seconds = 0;
    bool requires_candidate_trial = true;
    std::string reason;
};

class ChannelCostModel {
    struct Line {
        double intercept, slope;
        double operator()(double share) const { return intercept + slope * share; }
    };
    struct Coupling { double gpu, ane; };
    Line gpu_{0,0}, ane_{0,0};
    std::vector<Coupling> feasible_;
    double low_ = 0, high_ = 0;

    // Feasible bandwidth fractions in [0,1]^2. Rather than choose an
    // underidentified one-binding heuristic, retain its uncertainty and use
    // the conservative envelope. No claimed physical engine placement.
    static void clip(std::vector<Coupling> &polygon, double g, double a, double limit) {
        std::vector<Coupling> next;
        if (polygon.empty()) return;
        auto distance = [&](Coupling p) { return g*p.gpu + a*p.ane - limit; };
        const double epsilon=1e-12*std::max(std::abs(g)+std::abs(a)+std::abs(limit),std::numeric_limits<double>::min());
        auto previous = polygon.back(); double before = distance(previous);
        for (auto current : polygon) {
            const double after = distance(current);
            if ((before <= epsilon) != (after <= epsilon)) {
                const double fraction = std::clamp(before / (before - after),0.,1.);
                next.push_back({previous.gpu + fraction*(current.gpu-previous.gpu),
                                previous.ane + fraction*(current.ane-previous.ane)});
            }
            if (after <= epsilon) next.push_back(current);
            previous = current; before = after;
        }
        polygon = std::move(next);
    }

  public:
    static std::optional<ChannelCostModel> fit(CalibrationPoint p, CalibrationPoint q,
                                               double tolerance = .03) {
        if (!std::isfinite(tolerance) || tolerance < 0 || tolerance > .1) return std::nullopt;
        auto valid = [&](const CalibrationPoint &x) {
            return std::isfinite(x.share) && x.share > 0 && x.share < 1 &&
                std::isfinite(x.gpu) && x.gpu > 0 && std::isfinite(x.ane) && x.ane > 0 &&
                std::isfinite(x.both) && x.both > 0 &&
                x.both >= (1-tolerance)*std::max(x.gpu,x.ane) &&
                x.both <= (1+tolerance)*(x.gpu+x.ane);
        };
        if (!valid(p) || !valid(q) || std::abs(q.share-p.share) < 1e-6) return std::nullopt;
        if (q.share < p.share) std::swap(p,q);
        const auto line = [&](double a, double b) {
            const double slope = (b-a)/(q.share-p.share);
            return Line{a-slope*p.share,slope};
        };
        ChannelCostModel model;
        model.gpu_ = line(p.gpu,q.gpu); model.ane_ = line(p.ane,q.ane);
        model.low_=p.share; model.high_=q.share;
        model.feasible_={{0,0},{1,0},{1,1},{0,1}};
        for (const auto x : {p,q}) {
            clip(model.feasible_,x.gpu,x.ane,(1+tolerance)*x.both);
            // A non-binding observation gives only an upper bound. A clear
            // excess over max(G,A) also supplies a lower bandwidth bound.
            if (x.both > (1+tolerance)*std::max(x.gpu,x.ane))
                clip(model.feasible_,-x.gpu,-x.ane,-(1-tolerance)*x.both);
        }
        if (model.feasible_.empty()) return std::nullopt;
        return model;
    }

    std::optional<double> predict(double share) const {
        // Avoid silently extrapolating across unmeasured program/tile widths.
        if (!std::isfinite(share) || share < low_-1e-12 || share > high_+1e-12) return std::nullopt;
        const double g=gpu_(share),a=ane_(share);
        if (!std::isfinite(g) || !std::isfinite(a) || g<=0 || a<=0) return std::nullopt;
        double value=std::max(g,a);
        for (const auto coupling : feasible_)
            value=std::max(value,coupling.gpu*g+coupling.ane*a);
        return std::isfinite(value)?std::optional<double>(value):std::nullopt;
    }

    ChannelCostDecision select(int full_width, int channel_unit, double measured_full_gpu,
                              const std::function<bool(int)> &admitted,
                              double minimum_gain = .05, double near_optimal = .01) const {
        if (full_width<=0 || channel_unit<=0 || full_width%channel_unit || full_width/channel_unit<2 ||
            !std::isfinite(measured_full_gpu) || measured_full_gpu<=0 || !admitted ||
            !std::isfinite(minimum_gain) || minimum_gain<0 || minimum_gain>=1 ||
            !std::isfinite(near_optimal) || near_optimal<0 || near_optimal>.1)
            throw std::invalid_argument("invalid channel calibration selection/baseline/admission");
        ChannelCostDecision result{0,measured_full_gpu,measured_full_gpu,true,"no measured-range memory-admitted candidate"};
        std::vector<std::pair<int,double>> choices;
        double best=std::numeric_limits<double>::infinity();
        for (int channels=channel_unit; channels<full_width; channels+=channel_unit) {
            if (!admitted(channels)) continue;
            const auto cost=predict(double(channels)/full_width);
            if (!cost) continue;
            choices.emplace_back(channels,*cost); best=std::min(best,*cost);
        }
        if (choices.empty()) return result;
        if (best>(1-minimum_gain)*measured_full_gpu) {
            result.reason="predicted gain below minimum versus measured optimized GPU"; return result;
        }
        for (const auto &[channels,cost] : choices) if (cost<=(1+near_optimal)*best) {
            result.ane_channels=channels;result.predicted_seconds=cost;
            result.reason="smallest admitted measured-range near-optimal share; independent candidate trial required";
            return result;
        }
        throw std::logic_error("channel calibration lost its minimum candidate");
    }
};
} // namespace tc::ane
