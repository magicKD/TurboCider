#include "../../native/backends/coreml.hpp"
#include "../../native/models/h3_mlx/conditioner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {
using tc::Tensor;

Tensor dense(const Tensor &input, const Tensor &weight) {
    tc::require(weight.ndim() == 2 && input.shape(-1) == weight.shape(1),
                "invalid H3 MLP probe projection geometry");
    return tc::mx::matmul(input, tc::mx::transpose(weight));
}

double median(std::vector<double> values) {
    tc::require(!values.empty(), "cannot summarize empty H3 MLP timings");
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle]
                             : (values[middle - 1] + values[middle]) * 0.5;
}

struct Quality {
    double relative_l2 = 0;
    double cosine = 1;
    double max_abs = 0;
    double relative_max_abs = 0;
};

Quality quality(const Tensor &candidate, const Tensor &reference) {
    auto a = tc::mx::astype(candidate, tc::mx::float32);
    auto b = tc::mx::astype(reference, tc::mx::float32);
    auto finite = tc::mx::logical_and(tc::mx::all(tc::mx::isfinite(a)),
                                      tc::mx::all(tc::mx::isfinite(b)));
    auto delta = a - b;
    auto scale = tc::mx::maximum(
        tc::mx::maximum(tc::mx::max(tc::mx::abs(a)),
                        tc::mx::max(tc::mx::abs(b))),
        Tensor(1.f, tc::mx::float32));
    auto scaled_a = a / scale;
    auto scaled_b = b / scale;
    auto scaled_delta = delta / scale;
    auto delta2 = tc::mx::sum(scaled_delta * scaled_delta);
    auto a2 = tc::mx::sum(scaled_a * scaled_a);
    auto b2 = tc::mx::sum(scaled_b * scaled_b);
    auto dot = tc::mx::sum(scaled_a * scaled_b);
    auto max_abs = tc::mx::max(tc::mx::abs(delta));
    auto reference_max = tc::mx::max(tc::mx::abs(b));
    tc::mx::eval({finite, delta2, a2, b2, dot, max_abs, reference_max});
    tc::require(finite.item<bool>(), "H3 MLP probe produced nonfinite values");
    const double delta_energy = delta2.item<float>();
    const double a_energy = a2.item<float>();
    const double b_energy = b2.item<float>();
    const double denominator = std::sqrt(a_energy * b_energy);
    Quality result;
    result.relative_l2 = b_energy > 0
        ? std::sqrt(delta_energy / b_energy)
        : (delta_energy == 0 ? 0 : INFINITY);
    result.cosine = std::clamp(
        denominator > 0 ? dot.item<float>() / denominator
                        : (a_energy == 0 && b_energy == 0 ? 1.0 : 0.0),
        -1.0, 1.0);
    result.max_abs = max_abs.item<float>();
    const double amplitude = reference_max.item<float>();
    result.relative_max_abs = amplitude > 0
        ? result.max_abs / amplitude
        : (result.max_abs == 0 ? 0 : INFINITY);
    return result;
}
} // namespace

int main(int argc, char **argv) {
    try {
        tc::require(argc == 6,
                    "usage: h3-mlx-mlp-hybrid-probe TEXT_ENCODER "
                    "COMPILED_MANIFEST ROWS RUNS OUTPUT.safetensors");
        tc::configure_streams();
        const auto root = std::filesystem::absolute(argv[1]);
        const auto manifest = std::filesystem::absolute(argv[2]);
        const int rows = std::stoi(argv[3]);
        const int runs = std::stoi(argv[4]);
        tc::require(rows > 0 && rows <= 512 && runs > 0 && runs <= 20,
                    "H3 MLP probe requires 1...512 rows and 1...20 runs");
        const auto checkpoint = root / "model.safetensors.index.json";
        tc::require(std::filesystem::is_regular_file(checkpoint) &&
                        !std::filesystem::is_symlink(checkpoint),
                    "H3 MLP probe requires an indexed checkpoint");

        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::h3_mlx::ShardIndex weights(root);
        const auto setup_started = tc::Clock::now();
        tc::HybridSession hybrid(manifest, root, rows, event, cancelled, 0,
                                 std::filesystem::canonical(checkpoint));
        const double setup_seconds = std::chrono::duration<double>(
            tc::Clock::now() - setup_started).count();
        tc::require(hybrid.hidden == 5120 && hybrid.mlp_width == 25600 &&
                        hybrid.ane_mlp_start == 0 && hybrid.ane_mlp_end > 0 &&
                        hybrid.ane_mlp_end < hybrid.mlp_width &&
                        hybrid.block_count >= 1,
                    "compiled H3 MLP manifest geometry is invalid");

        auto input = tc::mx::random::normal(
            {rows, hybrid.hidden}, tc::mx::float32, tc::mx::random::key(42));
        tc::mx::eval(input);
        const std::string prefix = "model.language_model.layers.0.mlp.";
        auto gpu = [&] {
            auto gate = dense(input, weights.tensor(prefix + "gate_proj.weight"));
            auto up = dense(input, weights.tensor(prefix + "up_proj.weight"));
            auto output = dense(gate * tc::mx::sigmoid(gate) * up,
                                weights.tensor(prefix + "down_proj.weight"));
            tc::mx::eval(output);
            return output;
        };
        auto split = [&] {
            const int end = hybrid.ane_mlp_end;
            const int width = hybrid.mlp_width;
            auto gate = dense(input, weights.slice(
                prefix + "gate_proj.weight", end, width, 0, hybrid.hidden));
            auto up = dense(input, weights.slice(
                prefix + "up_proj.weight", end, width, 0, hybrid.hidden));
            auto suffix = dense(
                gate * tc::mx::sigmoid(gate) * up,
                weights.slice(prefix + "down_proj.weight", 0, hybrid.hidden,
                              end, width));
            tc::mx::async_eval({suffix});
            auto packed = tc::mx::expand_dims(
                tc::mx::astype(input, tc::mx::float16), 0);
            tc::mx::eval(packed);
            auto ane = tc::mx::squeeze(hybrid.predict(0, packed), 0);
            if (hybrid.output_scale != 1.f)
                ane = ane * Tensor(hybrid.output_scale, ane.dtype());
            tc::mx::eval({suffix, ane});
            auto output = suffix + tc::mx::astype(ane, tc::mx::float32);
            tc::mx::eval(output);
            return output;
        };

        Tensor reference = input;
        Tensor candidate = input;
        std::vector<double> gpu_seconds;
        std::vector<double> hybrid_seconds;
        for (int iteration = 0; iteration <= runs; ++iteration) {
            const bool hybrid_first = iteration % 2 == 1;
            auto measure = [&](bool use_hybrid) {
                tc::mx::clear_cache();
                const auto started = tc::Clock::now();
                auto output = use_hybrid ? split() : gpu();
                const double seconds = std::chrono::duration<double>(
                    tc::Clock::now() - started).count();
                if (iteration > 0)
                    (use_hybrid ? hybrid_seconds : gpu_seconds).push_back(seconds);
                if (use_hybrid)
                    candidate = std::move(output);
                else
                    reference = std::move(output);
            };
            measure(hybrid_first);
            measure(!hybrid_first);
        }
        const auto metrics = hybrid.metrics();
        const auto q = quality(candidate, reference);
        tc::mx::save_safetensors(
            std::filesystem::absolute(argv[5]).string(),
            {{"input", input}, {"gpu", reference}, {"hybrid", candidate}});
        const double gpu_median = median(gpu_seconds);
        const double hybrid_median = median(hybrid_seconds);
        std::cout << std::setprecision(10)
                  << "{\"rows\":" << rows
                  << ",\"runs\":" << runs
                  << ",\"ane_mlp_end\":" << hybrid.ane_mlp_end
                  << ",\"setup_seconds\":" << setup_seconds
                  << ",\"gpu_median_seconds\":" << gpu_median
                  << ",\"hybrid_median_seconds\":" << hybrid_median
                  << ",\"speedup\":" << gpu_median / hybrid_median
                  << ",\"coreml_prediction_seconds_session_total\":"
                  << metrics.prediction_seconds
                  << ",\"coreml_runtime_calls_session_total\":"
                  << metrics.runtime_calls
                  << ",\"output_copy_bytes_session_total\":"
                  << metrics.copied_bytes
                  << ",\"relative_l2\":" << q.relative_l2
                  << ",\"cosine\":" << q.cosine
                  << ",\"max_abs\":" << q.max_abs
                  << ",\"relative_max_abs\":" << q.relative_max_abs
                  << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
