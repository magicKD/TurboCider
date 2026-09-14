#include "../../native/models/h3_mlx/vdn_mlx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle]
                             : 0.5 * (values[middle - 1] + values[middle]);
}

double time_attention(const tc::Tensor &q, const tc::Tensor &k,
                      const tc::Tensor &v,
                      const tc::h3_mlx::PackedLayout &layout,
                      bool reference, bool mma, int runs) {
    for (int index = 0; index < 2; ++index) {
        auto out = mma
            ? tc::h3_mlx::vdn_window_softmax_mma_for_test(q, k, v, layout, 5, 1)
            : tc::h3_mlx::vdn_window_softmax_for_test(q, k, v, layout, 5, 1, reference);
        tc::mx::eval(out);
    }
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(runs));
    for (int index = 0; index < runs; ++index) {
        const auto started = Clock::now();
        auto out = mma
            ? tc::h3_mlx::vdn_window_softmax_mma_for_test(q, k, v, layout, 5, 1)
            : tc::h3_mlx::vdn_window_softmax_for_test(q, k, v, layout, 5, 1, reference);
        tc::mx::eval(out);
        samples.push_back(std::chrono::duration<double>(
            Clock::now() - started).count());
    }
    return median(std::move(samples));
}

} // namespace

int main() {
    try {
        tc::configure_streams();
        constexpr int text_tokens = 32;
        constexpr int latent_frames = 17;
        constexpr int latent_height = 32;
        constexpr int latent_width = 32;
        constexpr int audio_latents = 16;
        constexpr int heads = 4;
        constexpr int dim = 128;
        const auto layout = tc::h3_mlx::build_packed_layout(
            text_tokens, latent_frames, latent_height, latent_width,
            audio_latents);
        const size_t elements = static_cast<size_t>(layout.sequence_length) *
                                heads * dim;
        std::vector<float> qv(elements), kv(elements), vv(elements);
        for (size_t index = 0; index < elements; ++index) {
            const float phase = float(index % 65521u);
            qv[index] = 0.4f * std::sin(phase * 0.013f);
            kv[index] = 0.4f * std::cos(phase * 0.017f);
            vv[index] = 0.6f * std::sin(phase * 0.007f + 0.3f);
        }
        auto q = tc::mx::astype(tc::Tensor(qv.data(),
            {layout.sequence_length, heads, dim}, tc::mx::float32),
            tc::mx::bfloat16);
        auto k = tc::mx::astype(tc::Tensor(kv.data(),
            {layout.sequence_length, heads, dim}, tc::mx::float32),
            tc::mx::bfloat16);
        auto v = tc::mx::astype(tc::Tensor(vv.data(),
            {layout.sequence_length, heads, dim}, tc::mx::float32),
            tc::mx::bfloat16);

        auto reference = tc::h3_mlx::vdn_window_softmax_for_test(
            q, k, v, layout, 5, 1, true);
        auto candidate = tc::h3_mlx::vdn_window_softmax_for_test(
            q, k, v, layout, 5, 1, false);
        auto mma = tc::h3_mlx::vdn_window_softmax_mma_for_test(
            q, k, v, layout, 5, 1);
        tc::mx::eval(reference, candidate, mma);
        auto rf = tc::mx::astype(reference, tc::mx::float32);
        auto cf = tc::mx::astype(candidate, tc::mx::float32);
        auto mf = tc::mx::astype(mma, tc::mx::float32);
        auto difference = cf - rf;
        const float max_abs = tc::mx::max(tc::mx::abs(difference)).item<float>();
        const float rmse = tc::mx::sqrt(
            tc::mx::mean(tc::mx::square(difference))).item<float>();
        const float reference_rms = tc::mx::sqrt(
            tc::mx::mean(tc::mx::square(rf))).item<float>();
        const float relative_rmse = rmse / std::max(reference_rms, 1e-12f);
        const float cosine = (tc::mx::sum(rf * cf) /
            tc::mx::maximum(
                tc::mx::sqrt(tc::mx::sum(rf * rf) * tc::mx::sum(cf * cf)),
                tc::Tensor(1e-12f))).item<float>();
        auto mma_difference = mf - rf;
        const float mma_max_abs = tc::mx::max(tc::mx::abs(mma_difference)).item<float>();
        const float mma_rmse = tc::mx::sqrt(
            tc::mx::mean(tc::mx::square(mma_difference))).item<float>();
        const float mma_relative_rmse = mma_rmse / std::max(reference_rms, 1e-12f);
        const float mma_cosine = (tc::mx::sum(rf * mf) /
            tc::mx::maximum(
                tc::mx::sqrt(tc::mx::sum(rf * rf) * tc::mx::sum(mf * mf)),
                tc::Tensor(1e-12f))).item<float>();

        const double reference_seconds = time_attention(q, k, v, layout, true, false, 7);
        const double candidate_seconds = time_attention(q, k, v, layout, false, false, 7);
        const double mma_seconds = time_attention(q, k, v, layout, false, true, 7);
        tc::require(relative_rmse < 0.02f,
                    "VDN span attention relative RMSE exceeds 2%");
        tc::require(cosine > 0.999f,
                    "VDN span attention cosine is below 0.999");
        tc::require(mma_relative_rmse < 0.03f,
                    "VDN MMA span attention relative RMSE exceeds 3%");
        tc::require(mma_cosine > 0.999f,
                    "VDN MMA span attention cosine is below 0.999");
        std::cout << std::setprecision(9)
                  << "{\"sequence\":" << layout.sequence_length
                  << ",\"heads\":" << heads
                  << ",\"dim\":" << dim
                  << ",\"max_abs\":" << max_abs
                  << ",\"relative_rmse\":" << relative_rmse
                  << ",\"cosine\":" << cosine
                  << ",\"reference_seconds\":" << reference_seconds
                  << ",\"span_seconds\":" << candidate_seconds
                  << ",\"speedup\":" << reference_seconds / candidate_seconds
                  << ",\"mma_max_abs\":" << mma_max_abs
                  << ",\"mma_relative_rmse\":" << mma_relative_rmse
                  << ",\"mma_cosine\":" << mma_cosine
                  << ",\"mma_seconds\":" << mma_seconds
                  << ",\"mma_speedup\":" << reference_seconds / mma_seconds
                  << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
