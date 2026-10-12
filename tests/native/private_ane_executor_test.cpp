#include "../../native/backends/private/ane_executor.hpp"
#include <bit>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>

using namespace tc::ane;
int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) return 2;
    try {
        if (argc == 3) {
            const std::string profile = argv[2];
            GraphShape shape{Kind::SwiGLU, 0, 0, 0, 1024, 512, false};
            if (profile == "z512" || profile == "z1024") {
                shape.hidden = 3840; shape.width = 10240; shape.rows = profile == "z512" ? 352 : 704;
            } else if (profile == "qwen512" || profile == "qwen1024") {
                shape.hidden = 4096; shape.width = 12288; shape.rows = profile == "qwen512" ? 320 : 1792;
            } else return 2;
            const auto start = std::chrono::steady_clock::now();
            PrivateGraph graph(shape, 2ull << 30, argv[1]);
            const double load = graph.load_seconds();
            std::string error;
            if (!graph.self_test(error)) throw std::runtime_error(error);
            std::cout << "PASS model-shape-only sparse A/B/A: " << profile << " rows=" << shape.rows
                << " H=" << shape.hidden << " F=" << shape.width << " slots=" << graph.slot_bytes()
                << " estimate=" << graph.estimated_bytes() << " load_s=" << load << " total_s="
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                << "; no checkpoint/full-model/performance qualification\n";
            return 0;
        }
        for (Kind kind : {Kind::Matmul, Kind::SwiGLU}) for (bool lora : {false, true}) {
            if (kind == Kind::Matmul && lora) continue;
            const GraphShape shape{kind, 33, 64, 96, 32, 48, lora};
            PrivateGraph graph(shape, 256u << 20, argv[1]);
            std::string error;
            if (!graph.self_test(error)) throw std::runtime_error(error);
            std::vector<float> w0(96 * 64, .01f), w1(96 * 64, -.02f), w2(64 * 96, .03f);
            auto view = [](const std::vector<float> &v, int rows, int cols) { return MatrixView{v.data(), v.size() * 4, rows, cols, 0, DType::FP32}; };
            std::vector<WeightView> weights{view(w0, 96, 64)};
            if (kind == Kind::SwiGLU) { weights.push_back(view(w1, 96, 64)); weights.push_back(view(w2, 64, 96)); }
            const int rows = 66, cols = shape.output_width();
            std::vector<float> x(rows * 64, .1f), dg(rows * 96, .02f), du(rows * 96, -.03f);
            std::vector<uint16_t> output(rows * cols), hidden(rows * 96);
            auto run = [&](bool adapter) {
                graph.stage_weights(weights);
                if (!graph.wait_stage().ok) throw std::runtime_error("staging failed");
                graph.launch(view(x, rows, 64), output.data(), output.size(), DType::BF16,
                    adapter ? std::optional<AdapterInput>({view(dg, rows, 96), view(du, rows, 96), hidden.data(), hidden.size()}) : std::nullopt);
                auto result = graph.finish();
                if (!result.ok || result.calls != 2) throw std::runtime_error(result.error);
                const float gate = 64 * .01f * .1f + (adapter ? .02f : 0.f);
                const float up = 64 * -.02f * .1f + (adapter ? -.03f : 0.f);
                const float h = gate / (1 + std::exp(-gate)) * up;
                const float expected = kind == Kind::Matmul ? gate : h * 96 * .03f;
                for (uint16_t bits : output) {
                    const float actual = std::bit_cast<float>(uint32_t(bits) << 16);
                    if (std::abs(actual - expected) > .0003f + .02f * std::abs(expected)) throw std::runtime_error("private FFN dense oracle mismatch");
                }
                if (adapter) for (uint16_t bits : hidden)
                    if (std::abs(std::bit_cast<float>(uint32_t(bits) << 16) - h) > .0002f) throw std::runtime_error("private hidden LoRA BF16 oracle mismatch");
                return output;
            };
            auto base = run(false);
            if (lora) { if (run(true) == base) throw std::runtime_error("LoRA ignored"); if (run(false) != base) throw std::runtime_error("adapter leakage to base"); }
            // Bad staging invalidates the previous slots, but a clean refill
            // can recover. No stale partial output is declared successful.
            w0[0] = std::numeric_limits<float>::infinity();
            graph.stage_weights(weights); if (graph.wait_stage().ok) throw std::runtime_error("bad weights accepted");
            graph.launch(view(x, rows, 64), output.data(), output.size());
            if (graph.finish().ok) throw std::runtime_error("stale weights launched");
            w0[0] = .01f; if (run(false) != base) throw std::runtime_error("fresh refill failed");
            if (kind == Kind::SwiGLU) {
                std::fill(x.begin(), x.end(), 8.f); std::fill(w1.begin(), w1.end(), 32.f);
                graph.stage_weights(weights); if (!graph.wait_stage().ok) throw std::runtime_error("headroom stage failed");
                graph.launch(view(x, rows, 64), output.data(), output.size(), DType::BF16,
                    lora ? std::optional<AdapterInput>({view(dg, rows, 96), view(du, rows, 96), hidden.data(), hidden.size()}) : std::nullopt);
                auto result = graph.finish();
                if (!result.ok || result.headroom_scale < 4 || !result.overflow_retries) throw std::runtime_error("headroom recovery failed: " + result.error);
                const float g = 64 * .01f * 8 + (lora ? .02f : 0.f), u = 64 * 32.f * 8 + (lora ? -.03f : 0.f);
                const float h = g / (1 + std::exp(-g)) * u, expected = h * 96 * .03f;
                for (uint16_t bits : output) if (std::abs(std::bit_cast<float>(uint32_t(bits) << 16) - expected) > .02f * expected)
                    throw std::runtime_error("headroom changed FFN nonlinear math");
                if (lora) for (uint16_t bits : hidden) if (std::abs(std::bit_cast<float>(uint32_t(bits) << 16) - h) > .02f * h)
                    throw std::runtime_error("headroom/adapter hidden dtype or scaling mismatch");
                graph.stage_weights(weights); if (!graph.wait_stage().ok) throw std::runtime_error("scaled layer refill failed");
                graph.launch(view(x, rows, 64), output.data(), output.size(), DType::BF16);
                result = graph.finish(); if (!result.ok || result.overflow_retries) throw std::runtime_error("headroom not retained across layer refill");
            }
            std::cout << "PASS private executor kind=" << int(kind) << " lora=" << lora << " tiled/tail/multichunk/source-failure/base-A-base slots=" << graph.slot_bytes() << "\n";
        }
    } catch (const std::exception &error) { std::cerr << error.what() << "\n"; return 1; }
}
