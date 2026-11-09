// Independent full-request experiment. This executable is never packaged or
// exposed by the public LTX 0.9.8 session. CPU_AND_NE placement is unknown.
#define TURBOCIDER_LTX098_BIAS_GRAPH_ONLY 1
#include "ltx098_ane_bias_split_probe.mm"

#include "../../native/core/unigram_tokenizer.hpp"
#include "../../native/media/image.hpp"
#include "../../native/media/video.hpp"
#include "pipeline.hpp" // Experimental overlay, generated only for this probe.

#include <array>
#include <limits>

struct HybridStats {
    double stage_seconds = 0, stage_join_seconds = 0, input_eval_seconds = 0;
    double branch_seconds = 0, branch_join_seconds = 0, prediction_seconds = 0;
    double input_copy_seconds = 0, output_copy_seconds = 0;
    uint64_t ffn_calls = 0, hybrid_ffn_calls = 0, coreml_calls = 0;
    std::vector<uint64_t> blocks = std::vector<uint64_t>(28, 0);
    std::array<uint64_t, 8> steps{}, hybrid_steps{};
};

static tc::Tensor gpu_ffn(const tc::Tensor &input, const tc::Weights &weights) {
    return weights.project(tc::ltx098::gelu_tanh(weights.project(input, "ff.net.0.proj")), "ff.net.2");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc >= 9 && argc <= 11, "usage: ltx098-ane-hybrid-generate ROOT IMAGE COMPILED_MODEL NEW_OUTPUT_DIR PROMPT SEED gpu|passthrough|hybrid CHUNKS [STEP_MASK [REQUESTS]]");
            const auto began = Clock::now();
            const auto root = std::filesystem::absolute(argv[1]);
            const auto image_path = std::filesystem::absolute(argv[2]);
            const auto model_path = std::filesystem::absolute(argv[3]);
            const auto output = std::filesystem::absolute(argv[4]);
            const std::string prompt(argv[5]), seed_text(argv[6]), mode(argv[7]);
            const int chunks = std::stoi(argv[8]);
            const unsigned step_mask = argc >= 10 ? unsigned(std::stoul(argv[9])) : 255u;
            const int requests = argc == 11 ? std::stoi(argv[10]) : 1;
            check(!std::filesystem::exists(output) && !prompt.empty() && !seed_text.empty() && seed_text.front() != '-' &&
                  (mode == "gpu" || mode == "passthrough" || mode == "hybrid") && chunks >= 1 && chunks <= 15 &&
                  step_mask > 0 && step_mask <= 255 && requests >= 1 && requests <= 2,
                  "invalid experimental generation arguments");
            size_t consumed = 0;
            const uint64_t base_seed = std::stoull(seed_text, &consumed);
            check(consumed == seed_text.size() && base_seed <= std::numeric_limits<uint64_t>::max() - uint64_t(requests - 1),
                  "invalid experimental seed");
            check(std::filesystem::create_directories(output), "cannot create experimental output directory");
            std::unique_ptr<BiasGraph> graph;
            double graph_seconds = 0;
            if (mode == "hybrid") {
                const auto start = Clock::now();
                graph = std::make_unique<BiasGraph>(model_path, false);
                graph->self_test();
                graph_seconds = seconds(start);
            }
            tc::UnigramTokenizer tokenizer(root / "tokenizer", tc::UnigramFlavor::t5_v1_1);
            tc::ltx098::Pipeline pipeline(root);
            auto image = tc::load_image_center_crop_tensor(image_path, 512, 512);
            mx::eval(image);
            auto tokens = tokenizer.t5_prompt(prompt);
            for (int request_index = 0; request_index < requests; ++request_index) {
                const auto request_started = request_index == 0 ? began : Clock::now();
                const auto request_output =
                    requests == 1 ? output : output / ("request-" + std::to_string(request_index));
                if (requests != 1)
                    check(std::filesystem::create_directory(request_output),
                          "cannot create request output directory");
                const uint64_t seed = base_seed + uint64_t(request_index);
                HybridStats stats;
                tc::ltx098::GenerateOptions options;
                options.width = options.height = 512;
                options.frames = 121;
                options.fps = 24;
                options.seed = seed;
                options.trace = [&](const std::string &name, const tc::Tensor &value) {
                    if (name == "step_7") {
                        mx::eval(value);
                        mx::save_safetensors((request_output / "final-latent.safetensors").string(),
                                             {{"value", value}});
                    }
                };
                if (mode == "passthrough") {
                    options.ffn_hook = [&](const tc::Tensor &value, const tc::Weights &weights,
                                           int index) {
                        check(index >= 0 && index < 28, "invalid hooked block index");
                        const uint64_t ordinal = stats.ffn_calls++;
                        check(index == int(ordinal % 28) && ordinal / 28 < 8,
                              "experimental FFN call order differs");
                        ++stats.blocks[index];
                        ++stats.steps[ordinal / 28];
                        return gpu_ffn(value, weights);
                    };
                } else if (mode == "hybrid") {
                    options.ffn_hook = [&](const tc::Tensor &value, const tc::Weights &weights,
                                           int index) {
                        check(index >= 0 && index < 28 &&
                                  value.shape() == mx::Shape{1, 4096, 2048} &&
                                  value.dtype() == mx::bfloat16,
                              "experimental FFN geometry mismatch");
                        const uint64_t ordinal = stats.ffn_calls++;
                        check(index == int(ordinal % 28) && ordinal / 28 < 8,
                              "experimental FFN call order differs");
                        ++stats.blocks[index];
                        ++stats.steps[ordinal / 28];
                        if ((step_mask & (1u << unsigned(ordinal / 28))) == 0)
                            return gpu_ffn(value, weights);
                        const auto &wu = weights.at("ff.net.0.proj.weight"),
                                   &bu = weights.at("ff.net.0.proj.bias");
                        const auto &wd = weights.at("ff.net.2.weight"),
                                   &bd = weights.at("ff.net.2.bias");
                        check(wu.flags().row_contiguous && wd.flags().row_contiguous &&
                                  bu.flags().row_contiguous && bd.flags().row_contiguous &&
                                  wu.dtype() == mx::bfloat16 && wd.dtype() == mx::bfloat16 &&
                                  bu.dtype() == mx::bfloat16 && bd.dtype() == mx::bfloat16,
                              "experimental FFN weights must be contiguous BF16");
                        // The CPU stages the current block's fixed weights while
                        // MLX evaluates the independent attention/norm producer.
                        std::exception_ptr stage_failure;
                        double stage_time = 0;
                        std::jthread stager([&] {
                            @autoreleasepool {
                                try {
                                    const auto started = Clock::now();
                                    graph->stage(wu, wd, bu, bd);
                                    stage_time = seconds(started);
                                } catch (...) {
                                    stage_failure = std::current_exception();
                                }
                            }
                        });
                        const auto input_started = Clock::now();
                        auto flat = mx::contiguous(mx::reshape(value, {4096, 2048}));
                        mx::eval(flat);
                        stats.input_eval_seconds += seconds(input_started);
                        const auto stage_join = Clock::now();
                        stager.join();
                        stats.stage_join_seconds += seconds(stage_join);
                        stats.stage_seconds += stage_time;
                        if (stage_failure)
                            std::rethrow_exception(stage_failure);

                        const int tail_rows = chunks * 256, head_rows = 4096 - tail_rows;
                        const auto *tail_input =
                            reinterpret_cast<const uint16_t *>(flat.data<mx::bfloat16_t>()) +
                            size_t(head_rows) * 2048;
                        std::vector<uint16_t> tail_output(size_t(tail_rows) * 2048);
                        graph->input_seconds = graph->prediction_seconds = graph->output_seconds =
                            0;
                        graph->calls = 0;
                        std::exception_ptr branch_failure;
                        const auto branch_started = Clock::now();
                        std::jthread worker([&] {
                            @autoreleasepool {
                                try {
                                    for (int chunk = 0; chunk < chunks; ++chunk) {
                                        graph->fill_input(tail_input + size_t(chunk) * 256 * 2048);
                                        graph->predict(tail_output.data() +
                                                           size_t(chunk) * 256 * 2048,
                                                       tc::ane::DType::BF16);
                                    }
                                } catch (...) {
                                    branch_failure = std::current_exception();
                                }
                            }
                        });
                        auto head = gpu_ffn(mx::slice(flat, {0, 0}, {head_rows, 2048}), weights);
                        mx::eval(head);
                        const auto branch_join = Clock::now();
                        worker.join();
                        stats.branch_join_seconds += seconds(branch_join);
                        if (branch_failure)
                            std::rethrow_exception(branch_failure);
                        check(graph->calls == uint64_t(chunks),
                              "experimental Core ML chunk count mismatch");
                        auto tail =
                            mx::array(reinterpret_cast<const mx::bfloat16_t *>(tail_output.data()),
                                      {tail_rows, 2048}, mx::bfloat16);
                        auto combined = mx::concatenate({head, tail}, 0);
                        mx::eval(combined);
                        stats.branch_seconds += seconds(branch_started);
                        stats.prediction_seconds += graph->prediction_seconds;
                        stats.input_copy_seconds += graph->input_seconds;
                        stats.output_copy_seconds += graph->output_seconds;
                        ++stats.hybrid_ffn_calls;
                        ++stats.hybrid_steps[ordinal / 28];
                        stats.coreml_calls += graph->calls;
                        return mx::reshape(combined, {1, 4096, 2048});
                    };
                }
                std::atomic<bool> cancelled{false};
                auto [pixels, metrics] = pipeline.generate(
                    tokens, image, options,
                    [](const std::string &phase, int done, int total) {
                        if (phase == "denoise")
                            std::cerr << phase << ' ' << done << '/' << total << '\n';
                    },
                    cancelled);
                check(metrics.denoise_steps == 8 && (mode == "gpu" || stats.ffn_calls == 28 * 8) &&
                          (mode != "hybrid" ||
                           (stats.hybrid_ffn_calls == uint64_t(28 * std::popcount(step_mask)) &&
                            stats.coreml_calls == stats.hybrid_ffn_calls * uint64_t(chunks))),
                      "experimental full-request call counts differ");
                auto scaled = (mx::astype(pixels, mx::float32) + tc::Tensor(1.f, mx::float32)) *
                              tc::Tensor(127.5f, mx::float32);
                auto rgb =
                    mx::astype(mx::contiguous(mx::clip(scaled, tc::Tensor(0.f), tc::Tensor(255.f))),
                               mx::uint8);
                mx::eval(rgb);
                const auto movie = request_output / "0.mp4";
                tc::write_video_rgb24(movie, rgb.data<uint8_t>(), 121, 512, 512, 24);
                const auto info = tc::probe_video(movie);
                check(info.frames == 121 && info.width == 512 && info.height == 512 &&
                          info.fps == 24,
                      "experimental video geometry differs");
                const double wall = seconds(request_started);
                NSMutableArray *per_block = [NSMutableArray array];
                for (auto count : stats.blocks)
                    [per_block addObject:@(count)];
                NSMutableArray *per_step = [NSMutableArray array],
                               *per_hybrid_step = [NSMutableArray array];
                for (auto count : stats.steps)
                    [per_step addObject:@(count)];
                for (auto count : stats.hybrid_steps)
                    [per_hybrid_step addObject:@(count)];
                NSDictionary *report = @{
                    @"schema" : @"ltx098-experimental-hybrid-generate-v1",
                    @"mode" : @(mode.c_str()),
                    @"seed" : @(seed),
                    @"physical_ane_placement" : @"unknown",
                    @"scope" : @"probe-only full request; CPU_AND_NE may fall back; no production "
                               @"enablement",
                    @"width" : @512,
                    @"height" : @512,
                    @"frames" : @121,
                    @"fps" : @24,
                    @"actual_steps" : @(metrics.denoise_steps),
                    @"hybrid_step_mask" : @(step_mask),
                    @"hybrid_ffn_calls" : @(stats.hybrid_ffn_calls),
                    @"request_index" : @(request_index),
                    @"requests_in_process" : @(requests),
                    @"wall_seconds" : @(wall),
                    @"graph_load_and_selftest_seconds" : @(request_index == 0 ? graph_seconds : 0),
                    @"text_seconds" : @(metrics.text_seconds),
                    @"image_seconds" : @(metrics.image_seconds),
                    @"text_cache_hit" : @(metrics.text_cache_hit),
                    @"cross_attention_cache_hit" : @(metrics.cross_attention_cache_hit),
                    @"cross_attention_prepare_seconds" : @(metrics.cross_attention_prepare_seconds),
                    @"denoise_seconds" : @(metrics.denoise_seconds),
                    @"decode_seconds" : @(metrics.decode_seconds),
                    @"ffn_calls" : @(stats.ffn_calls),
                    @"coreml_calls" : @(stats.coreml_calls),
                    @"per_block_calls" : per_block,
                    @"per_step_ffn_calls" : per_step,
                    @"per_step_hybrid_ffn_calls" : per_hybrid_step,
                    @"stage_seconds" : @(stats.stage_seconds),
                    @"stage_join_seconds" : @(stats.stage_join_seconds),
                    @"input_eval_seconds" : @(stats.input_eval_seconds),
                    @"branch_seconds" : @(stats.branch_seconds),
                    @"branch_join_seconds" : @(stats.branch_join_seconds),
                    @"prediction_seconds" : @(stats.prediction_seconds),
                    @"input_copy_seconds" : @(stats.input_copy_seconds),
                    @"output_copy_seconds" : @(stats.output_copy_seconds),
                    @"mlx_peak_bytes" : @(mx::get_peak_memory())
                };
                NSData *json = [NSJSONSerialization dataWithJSONObject:report
                                                               options:NSJSONWritingPrettyPrinted
                                                                 error:nil];
                check(json &&
                          [json writeToURL:[NSURL fileURLWithPath:@((request_output / "report.json")
                                                                        .c_str())]
                                atomically:YES],
                      "cannot write experimental report");
                std::cout << "mode=" << mode << " request=" << request_index
                          << " wall_seconds=" << wall << " ffn_calls=" << stats.ffn_calls
                          << " coreml_calls=" << stats.coreml_calls << '\n';
            }
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
