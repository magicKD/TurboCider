#include "../../native/models/ltx_mlx/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {

std::vector<uint8_t> read_bytes(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    tc::require(input.good(), "cannot open fixture: " + path.string());
    const auto size = input.tellg();
    tc::require(size >= 0, "cannot stat fixture: " + path.string());
    std::vector<uint8_t> data(static_cast<size_t>(size));
    input.seekg(0);
    if (!data.empty())
        input.read(reinterpret_cast<char *>(data.data()),
                   static_cast<std::streamsize>(data.size()));
    tc::require(input.good() || input.eof(),
                "cannot read fixture: " + path.string());
    return data;
}

tc::Tensor load_bf16(const std::filesystem::path &path,
                     const tc::mx::Shape &shape) {
    auto bytes = read_bytes(path);
    const auto elements = static_cast<size_t>(std::accumulate(
        shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()));
    tc::require(bytes.size() == elements * 2,
                "BF16 fixture byte count mismatch: " + path.string());
    const auto *raw = reinterpret_cast<const uint16_t *>(bytes.data());
    std::vector<float> values(elements);
    for (size_t i = 0; i < elements; ++i) {
        uint32_t bits = uint32_t(raw[i]) << 16;
        std::memcpy(&values[i], &bits, sizeof(float));
    }
    return tc::mx::astype(tc::Tensor(values.data(), shape, tc::mx::float32),
                          tc::mx::bfloat16);
}

tc::Tensor load_f32(const std::filesystem::path &path,
                    const tc::mx::Shape &shape) {
    auto bytes = read_bytes(path);
    const auto elements = static_cast<size_t>(std::accumulate(
        shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()));
    tc::require(bytes.size() == elements * sizeof(float),
                "F32 fixture byte count mismatch: " + path.string());
    return tc::Tensor(reinterpret_cast<const float *>(bytes.data()), shape,
                      tc::mx::float32);
}

struct Metrics {
    double rel_l2 = 0.0;
    double cosine = 0.0;
    double max_abs = 0.0;
};

struct TensorStats {
    double rms = 0.0;
    uint64_t nonfinite = 0;
};

TensorStats stats(const tc::Tensor &value) {
    auto f32 = tc::mx::astype(value, tc::mx::float32);
    tc::mx::eval(f32);
    const auto *values = f32.data<float>();
    long double square = 0.0L;
    uint64_t nonfinite = 0;
    for (size_t i = 0; i < f32.size(); ++i) {
        if (!std::isfinite(values[i])) {
            ++nonfinite;
            continue;
        }
        square += static_cast<long double>(values[i]) * values[i];
    }
    return {std::sqrt(static_cast<double>(square / f32.size())), nonfinite};
}

Metrics compare(const tc::Tensor &actual, const tc::Tensor &expected) {
    auto a = tc::mx::astype(actual, tc::mx::float32);
    auto e = tc::mx::astype(expected, tc::mx::float32);
    tc::mx::eval(a, e);
    tc::require(a.shape() == e.shape(), "probe output shape mismatch");
    const auto *ap = a.data<float>();
    const auto *ep = e.data<float>();
    const size_t count = a.size();
    long double delta_norm = 0.0L;
    long double expected_norm = 0.0L;
    long double dot = 0.0L;
    long double actual_norm = 0.0L;
    double max_abs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const long double d = static_cast<long double>(ap[i]) - ep[i];
        delta_norm += d * d;
        expected_norm += static_cast<long double>(ep[i]) * ep[i];
        dot += static_cast<long double>(ap[i]) * ep[i];
        actual_norm += static_cast<long double>(ap[i]) * ap[i];
        max_abs = std::max(max_abs, std::abs(static_cast<double>(d)));
    }
    return {
        std::sqrt(static_cast<double>(delta_norm)) /
            std::max(std::sqrt(static_cast<double>(expected_norm)), 1.0e-30),
        static_cast<double>(dot) /
            std::max(std::sqrt(static_cast<double>(actual_norm * expected_norm)),
                     1.0e-30),
        max_abs};
}

} // namespace

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 3 && argc <= 6,
                    "usage: ltx-mlx-model-probe CHECKPOINT FIXTURE "
                    "[ITERATIONS [BLOCKS [CACHE_CAPACITY]]]");
        const auto checkpoint = std::filesystem::absolute(argv[1]);
        const auto fixture = std::filesystem::absolute(argv[2]);
        const int iterations = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 1;
        const int blocks = argc >= 5 ? std::clamp(std::atoi(argv[4]), 1, 48) : 1;
        const int capacity = argc >= 6 ?
            std::clamp(std::atoi(argv[5]), 1, 48) : blocks;

        tc::configure_streams();
        tc::ltx_mlx::Transformer transformer;
        tc::ltx_mlx::TransformerRunOptions options;
        options.block_count = blocks;
        options.block_cache_capacity = capacity;
        options.force_eval_each_block = true;
        // The exported dev-checkpoint fixture records the upstream-iso gate
        // scale. Production LTX-2.5 config.json supplies 1000 instead.
        options.av_ca_timestep_scale_multiplier = 1.0f;
        transformer.load(checkpoint, options);

        auto video = load_bf16(fixture / "video_patch_input.bf16", {1, 4, 128});
        auto audio = load_bf16(fixture / "audio_patch_input.bf16", {1, 4, 128});
        auto video_text = load_bf16(fixture / "video_context.bf16", {1, 8, 4096});
        auto audio_text = load_bf16(fixture / "audio_context.bf16", {1, 8, 2048});
        auto mask = load_bf16(fixture / "text_mask.bf16", {1, 1, 1, 8});
        auto video_positions = load_f32(fixture / "video_positions.f32", {1, 4, 3});
        auto audio_positions = load_f32(fixture / "audio_positions.f32", {1, 4, 1});

        auto prepared = transformer.prepare_for_probe(
            video, audio, 0.5f, video_positions, audio_positions);
        auto expected_video_patch = load_f32(
            fixture / "video_patch_output.f32", {1, 4, 4096});
        auto expected_audio_patch = load_f32(
            fixture / "audio_patch_output.f32", {1, 4, 2048});
        auto expected_video_params = load_bf16(
            fixture / "video_adaln_params.bf16", {1, 36864});
        auto expected_audio_params = load_bf16(
            fixture / "audio_adaln_params.bf16", {1, 18432});
        auto expected_video_embedded = load_bf16(
            fixture / "video_embedded_timestep.bf16", {1, 4096});
        auto expected_audio_embedded = load_bf16(
            fixture / "audio_embedded_timestep.bf16", {1, 2048});
        const auto video_patch_metrics = compare(prepared.video_hidden,
                                                 expected_video_patch);
        const auto audio_patch_metrics = compare(prepared.audio_hidden,
                                                 expected_audio_patch);
        const auto video_params_metrics = compare(prepared.video_adaln,
                                                  expected_video_params);
        const auto audio_params_metrics = compare(prepared.audio_adaln,
                                                  expected_audio_params);
        const auto video_embedded_metrics = compare(
            prepared.video_embedded_timestep, expected_video_embedded);
        const auto audio_embedded_metrics = compare(
            prepared.audio_embedded_timestep, expected_audio_embedded);

        auto run = [&] {
            auto output = transformer.forward(
                video, audio, 0.5f, video_text, audio_text,
                video_positions, audio_positions, {}, {}, mask);
            tc::mx::eval(output.first, output.second);
            return output;
        };
        auto output = run();
        std::vector<double> timings;
        timings.reserve(iterations);
        for (int i = 0; i < iterations; ++i) {
            const auto started = std::chrono::steady_clock::now();
            output = run();
            timings.push_back(std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count());
        }
        std::sort(timings.begin(), timings.end());
        const double p50 = timings[(timings.size() - 1) / 2];
        Metrics video_metrics{};
        Metrics audio_metrics{};
        if (blocks == 1) {
            auto expected_video = load_f32(
                fixture / "video_head_output.f32", {1, 4, 128});
            auto expected_audio = load_f32(
                fixture / "audio_head_output.f32", {1, 4, 128});
            video_metrics = compare(output.first, expected_video);
            audio_metrics = compare(output.second, expected_audio);
        }
        const auto video_stats = stats(output.first);
        const auto audio_stats = stats(output.second);
        const auto &cache = transformer.cache_metrics();
        std::cout << "{\"block_count\":" << blocks
                  << ",\"cache_capacity\":" << capacity
                  << ",\"iterations\":" << iterations
                  << ",\"p50_ms\":" << p50
                  << ",\"top_weight_bytes\":" << transformer.top_weight_bytes()
                  << ",\"cache_loads\":" << cache.loads
                  << ",\"cache_hits\":" << cache.hits
                  << ",\"cache_evictions\":" << cache.evictions
                  << ",\"video_patch_rel_l2\":" << video_patch_metrics.rel_l2
                  << ",\"audio_patch_rel_l2\":" << audio_patch_metrics.rel_l2
                  << ",\"video_adaln_rel_l2\":" << video_params_metrics.rel_l2
                  << ",\"audio_adaln_rel_l2\":" << audio_params_metrics.rel_l2
                  << ",\"video_embedded_rel_l2\":" << video_embedded_metrics.rel_l2
                  << ",\"audio_embedded_rel_l2\":" << audio_embedded_metrics.rel_l2
                  << ",\"video_rms\":" << video_stats.rms
                  << ",\"audio_rms\":" << audio_stats.rms
                  << ",\"nonfinite\":"
                  << video_stats.nonfinite + audio_stats.nonfinite
                  << ",\"video_rel_l2\":" << video_metrics.rel_l2
                  << ",\"video_cosine\":" << video_metrics.cosine
                  << ",\"video_max_abs\":" << video_metrics.max_abs
                  << ",\"audio_rel_l2\":" << audio_metrics.rel_l2
                  << ",\"audio_cosine\":" << audio_metrics.cosine
                  << ",\"audio_max_abs\":" << audio_metrics.max_abs << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
