#include "../../native/models/ltx_mlx/block.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <chrono>

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
    tc::require(input.good() || input.eof(), "cannot read fixture: " + path.string());
    return data;
}

tc::Tensor load_bf16(const std::filesystem::path &path, const tc::mx::Shape &shape) {
    auto bytes = read_bytes(path);
    tc::require(bytes.size() == size_t(std::accumulate(shape.begin(), shape.end(), 1LL,
                                                       std::multiplies<int64_t>())) * 2,
                "BF16 fixture byte count mismatch: " + path.string());
    const auto *raw = reinterpret_cast<const uint16_t *>(bytes.data());
    std::vector<float> values(bytes.size() / 2);
    for (size_t i = 0; i < values.size(); ++i) {
        uint32_t bits = uint32_t(raw[i]) << 16;
        std::memcpy(&values[i], &bits, sizeof(float));
    }
    return tc::mx::astype(tc::Tensor(values.data(), shape, tc::mx::float32),
                          tc::mx::bfloat16);
}

tc::Tensor load_f32(const std::filesystem::path &path, const tc::mx::Shape &shape) {
    auto bytes = read_bytes(path);
    tc::require(bytes.size() == size_t(std::accumulate(shape.begin(), shape.end(), 1LL,
                                                       std::multiplies<int64_t>())) * sizeof(float),
                "F32 fixture byte count mismatch: " + path.string());
    return tc::Tensor(reinterpret_cast<const float *>(bytes.data()), shape,
                      tc::mx::float32);
}

tc::Tensor params(const std::filesystem::path &fixture, const char *name, int elements) {
    return load_bf16(fixture / name, {1, elements});
}

struct Metrics {
    double rel_l2 = 0.0;
    double cosine = 0.0;
    double max_abs = 0.0;
};

Metrics compare(const tc::Tensor &actual, const tc::Tensor &expected) {
    auto a = tc::mx::astype(actual, tc::mx::float32);
    auto e = tc::mx::astype(expected, tc::mx::float32);
    tc::mx::eval(a, e);
    const auto *ap = a.data<float>();
    const auto *ep = e.data<float>();
    const size_t count = a.size();
    long double delta_norm = 0.0L, expected_norm = 0.0L;
    long double dot = 0.0L, actual_norm = 0.0L;
    double max_abs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const long double d = static_cast<long double>(ap[i]) -
                              static_cast<long double>(ep[i]);
        delta_norm += d * d;
        expected_norm += static_cast<long double>(ep[i]) *
                         static_cast<long double>(ep[i]);
        dot += static_cast<long double>(ap[i]) * static_cast<long double>(ep[i]);
        actual_norm += static_cast<long double>(ap[i]) *
                       static_cast<long double>(ap[i]);
        max_abs = std::max(max_abs, std::abs(double(d)));
    }
    Metrics result;
    result.rel_l2 = std::sqrt(double(delta_norm)) /
                    std::max(std::sqrt(double(expected_norm)), 1.0e-30);
    result.cosine = double(dot) /
                    std::max(std::sqrt(double(actual_norm * expected_norm)), 1.0e-30);
    result.max_abs = max_abs;
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 3 && argc <= 4,
                    "usage: ltx-mlx-block-probe CHECKPOINT FIXTURE [ITERATIONS]");
        const auto checkpoint = std::filesystem::absolute(argv[1]);
        const auto fixture = std::filesystem::absolute(argv[2]);
        const int iterations = argc == 4 ? std::max(1, std::atoi(argv[3])) : 1;
        tc::configure_streams();
        tc::ltx_mlx::BasicAVTransformerBlock block;
        block.load(checkpoint, 0, 64);

        auto video = load_bf16(fixture / "video_input.bf16", {1, 4, 4096});
        auto audio = load_bf16(fixture / "audio_input.bf16", {1, 4, 2048});
        auto video_text = load_bf16(fixture / "video_context.bf16", {1, 8, 4096});
        auto audio_text = load_bf16(fixture / "audio_context.bf16", {1, 8, 2048});
        auto mask = load_bf16(fixture / "text_mask.bf16", {1, 1, 1, 8});
        auto video_positions = load_f32(fixture / "video_positions.f32", {1, 4, 3});
        auto audio_positions = load_f32(fixture / "audio_positions.f32", {1, 4, 1});
        auto video_rope = tc::ltx_mlx::precompute_rope(
            video_positions, 4096, 32, {20, 2048, 2048});
        auto audio_rope = tc::ltx_mlx::precompute_rope(
            audio_positions, 2048, 32, {20});
        auto video_cross_rope = tc::ltx_mlx::precompute_rope(
            tc::mx::slice(video_positions, {0, 0, 0}, {1, 4, 1}),
            2048, 32, {20});

        auto video_adaln = params(fixture, "video_adaln_params.bf16", 9 * 4096);
        auto audio_adaln = params(fixture, "audio_adaln_params.bf16", 9 * 2048);
        auto video_prompt = params(fixture, "video_prompt_params.bf16", 2 * 4096);
        auto audio_prompt = params(fixture, "audio_prompt_params.bf16", 2 * 2048);
        auto av_video = params(fixture, "av_video_params.bf16", 4 * 4096);
        auto av_audio = params(fixture, "av_audio_params.bf16", 4 * 2048);
        auto a2v_gate = params(fixture, "a2v_gate_params.bf16", 4096);
        auto v2a_gate = params(fixture, "v2a_gate_params.bf16", 2048);

        auto run = [&] {
            auto value = block.forward(
                video, audio, video_adaln, audio_adaln, video_prompt, audio_prompt,
                av_video, av_audio, a2v_gate, v2a_gate,
                video_text, audio_text, video_rope, audio_rope,
                video_cross_rope, audio_rope, {}, {}, mask);
            tc::mx::eval(value.first, value.second);
            return value;
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
        auto expected_video = load_f32(fixture / "video_output.f32", {1, 4, 4096});
        auto expected_audio = load_f32(fixture / "audio_output.f32", {1, 4, 2048});
        const auto vm = compare(output.first, expected_video);
        const auto am = compare(output.second, expected_audio);
        std::cout << "{\"block\":0,\"iterations\":" << iterations
                  << ",\"bytes\":" << block.bytes()
                  << ",\"p50_ms\":" << p50
                  << ",\"video_rel_l2\":" << vm.rel_l2
                  << ",\"video_cosine\":" << vm.cosine
                  << ",\"video_max_abs\":" << vm.max_abs
                  << ",\"audio_rel_l2\":" << am.rel_l2
                  << ",\"audio_cosine\":" << am.cosine
                  << ",\"audio_max_abs\":" << am.max_abs << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
