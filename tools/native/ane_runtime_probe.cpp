// Runtime-weight component screen. Not a model/quality or ANE-residency claim.
#include "../../native/backends/ane_runtime.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace mx = mlx::core;
using Clock = std::chrono::steady_clock;
using tc::ane::Kind;

static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
// One measured iteration, excluding the two warmups. Worker time overlaps
// GPU head work: do not add runtime_seconds to split_seconds or gpu_seconds.
struct ProbeSample {
    double gpu_seconds = 0, split_seconds = 0, split_with_stage_seconds = 0;
    double stage_seconds = 0, predict_seconds = 0, join_seconds = 0;
    double input_seconds = 0, output_seconds = 0, runtime_seconds = 0;
    uint64_t calls = 0, overflow_retries = 0;
    float headroom_scale = 1.f;
};
// Summary fields and raw evidence share one source of truth. Compute summaries
// after the measurement loop; do not maintain parallel timing vectors.
static double median(const std::vector<ProbeSample> &samples, double ProbeSample::*field) {
    if (samples.empty()) throw std::runtime_error("cannot summarize empty probe samples");
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto &sample : samples) values.push_back(sample.*field);
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) * .5;
}
static tc::ane::MatrixView view(const mx::array &tensor) {
    // Synchronization is explicit BEFORE this borrowed host view is created.
    tc::ane::DType dtype;
    const void *data;
    if (tensor.dtype() == mx::bfloat16) { dtype = tc::ane::DType::BF16; data = tensor.data<mx::bfloat16_t>(); }
    else if (tensor.dtype() == mx::float16) { dtype = tc::ane::DType::FP16; data = tensor.data<mx::float16_t>(); }
    else if (tensor.dtype() == mx::float32) { dtype = tc::ane::DType::FP32; data = tensor.data<float>(); }
    else throw std::runtime_error("probe needs BF16/FP16/FP32 inputs");
    if (tensor.ndim() != 2 || !tensor.flags().row_contiguous) throw std::runtime_error("probe needs contiguous matrices");
    return {data, tensor.nbytes(), tensor.shape(0), tensor.shape(1), 0, dtype};
}

int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[2]) == "lease-source-move-self-test") {
            // This probe-only mode operates on a disposable exported fixture.
            // The loaded graph must keep working when the original compiled
            // directory is moved; a new load from that manifest must fail.
            const std::filesystem::path manifest(argv[1]);
            tc::ane::RuntimeGraph loaded(manifest, 2ull << 30);
            const auto source = manifest.parent_path() / "graph.mlmodelc";
            const auto moved = manifest.parent_path() / "graph.moved";
            if (std::filesystem::exists(moved)) throw std::runtime_error("lease test destination exists");
            std::filesystem::rename(source, moved);
            std::string error;
            if (!loaded.self_test(error)) throw std::runtime_error("leased graph failed: " + error);
            bool rejected = false;
            try { tc::ane::RuntimeGraph missing(manifest, 2ull << 30); }
            catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("missing original artifact was accepted");
            std::cout << "PASS private runtime ANE artifact lease" << std::endl;
            return 0;
        }
        if (argc < 2 || argc > 11) throw std::runtime_error(
            "usage: ane-runtime-probe MANIFEST [REPEATS=6] [ROWS=2*chunk] [cpu|ne] [BUNDLE_OR_CHECKPOINT|-] [ANE_CHUNKS] [simd|scalar] [INPUT_NPY|-] [BLOCK=0] [qkv|qkv-separate-gpu|qkv-parts]");
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 6;
        if (repeats < 2 || repeats > 100) throw std::runtime_error("repeats must be in [2,100]");
        if (argc > 4 && std::string(argv[4]) != "cpu" && std::string(argv[4]) != "ne")
            throw std::runtime_error("compute units must be cpu or ne");
        const bool scalar = argc > 7 && std::string(argv[7]) == "scalar";
        if (argc > 7 && !scalar && std::string(argv[7]) != "simd") throw std::runtime_error("invalid staging mode");
        const bool separate_qkv_gpu = argc > 10 && std::string(argv[10]) == "qkv-separate-gpu";
        const bool parts_qkv = argc > 10 && std::string(argv[10]) == "qkv-parts";
        const bool pack_qkv = argc > 10 && (std::string(argv[10]) == "qkv" || separate_qkv_gpu);
        if (argc > 10 && !pack_qkv && !parts_qkv) throw std::runtime_error("invalid projection pack");
        tc::ane::RuntimeGraph graph(argv[1], 2ull << 30, argc > 4 && std::string(argv[4]) == "cpu", scalar);
        const auto &s = graph.shape();
        if ((pack_qkv || parts_qkv) && (s.kind != Kind::Matmul || s.width != 3 * s.hidden ||
                                       argc < 6 || std::string(argv[5]) == "-"))
            throw std::runtime_error("QKV source needs a MatMul graph with three square projections and a checkpoint");
        std::string error;
        if (!graph.self_test(error)) throw std::runtime_error("self-test: " + error);
        const int rows = argc > 3 ? std::stoi(argv[3]) : 2 * s.rows;
        if (rows <= s.rows || rows > 32768) throw std::runtime_error("probe must leave positive GPU rows");
        // Half the rows, rounded down to chunks, leaving at least one chunk.
        const int chunks = argc > 6 ? std::stoi(argv[6]) :
            std::min((rows - 1) / s.rows, std::max(1, rows / (2 * s.rows)));
        if (chunks < 1 || chunks > (rows - 1) / s.rows) throw std::runtime_error("invalid ANE chunk count");
        const int ane_rows = chunks * s.rows;
        const int gpu_rows = rows - ane_rows;
        std::vector<mx::array> weights;
        std::string weight_source = "synthetic";
        auto random = [](int r, int c, int seed) {
            return mx::contiguous(mx::astype(mx::random::normal({r, c}, mx::float32, mx::random::key(seed)) *
                                              (1.f / std::sqrt(float(c))), mx::bfloat16));
        };
        // Activations do not shrink with hidden width. Scaling x by 1/sqrt(H)
        // as for weights artificially pushes a whole FFN into FP16 underflow.
        auto x = mx::contiguous(mx::astype(mx::random::normal({rows, s.hidden},
                      mx::float32, mx::random::key(41)) * .5f, mx::bfloat16));
        if (s.kind == Kind::SwiGLU) weights.push_back(random(s.width, s.hidden, 42));
        weights.push_back(random(s.width, s.hidden, 43));
        if (s.kind != Kind::Matmul) weights.push_back(random(s.hidden, s.width, 44));
        if (argc >= 6 && std::string(argv[5]) != "-") {
            auto [tensors, metadata] = mx::load_safetensors(argv[5]);
            const int block = argc > 9 ? std::stoi(argv[9]) : 0;
            if (block < 0 || block >= 32) throw std::runtime_error("probe block must be in [0,31]");
            if (tensors.count("x")) x = mx::contiguous(tensors.at("x"));
            const std::vector<std::string> names = s.kind == Kind::Matmul ? std::vector<std::string>{"w"} :
                s.kind == Kind::GELU ? std::vector<std::string>{"wu", "wd"} : std::vector<std::string>{"wg", "wu", "wd"};
            weights.clear();
            const auto qwen = "transformer_blocks." + std::to_string(block) + ".img_mlp.";
            const auto z = (block < 2 ? "noise_refiner." + std::to_string(block) :
                                        "layers." + std::to_string(block - 2)) + ".feed_forward.";
            const auto z_attention = (block < 2 ? "noise_refiner." + std::to_string(block) :
                                                 "layers." + std::to_string(block - 2)) + ".attention.qkv.weight";
            const auto qwen_q = "transformer_blocks." + std::to_string(block) + ".attn.to_q.weight";
            if (pack_qkv || parts_qkv) {
                std::vector<mx::array> projections;
                weight_source.clear();
                for (const auto *name : {"to_q.weight", "to_k.weight", "to_v.weight"}) {
                    const auto key = "transformer_blocks." + std::to_string(block) + ".attn." + name;
                    if (!tensors.count(key) || tensors.at(key).shape() != mx::Shape{s.hidden, s.hidden})
                        throw std::runtime_error("missing or mismatched QKV checkpoint tensor: " + key);
                    projections.push_back(tensors.at(key));
                    if (!weight_source.empty()) weight_source += "+";
                    weight_source += key;
                }
                if (parts_qkv) weights = std::move(projections);
                else weights.push_back(mx::contiguous(mx::concatenate(projections, 0)));
            } else if (s.kind == Kind::Matmul && !tensors.count("w") && tensors.count(qwen_q)) {
                weights.push_back(mx::contiguous(tensors.at(qwen_q)));
                weight_source = qwen_q;
            } else if (s.kind == Kind::Matmul && !tensors.count("w") && tensors.count(z_attention)) {
                weights.push_back(mx::contiguous(tensors.at(z_attention)));
                weight_source = z_attention;
            } else if (s.kind == Kind::SwiGLU && tensors.count(qwen + "gate_up.weight")) {
                const auto &fused = tensors.at(qwen + "gate_up.weight");
                weights = {mx::contiguous(mx::slice(fused, {0, 0}, {s.width, s.hidden})),
                           mx::contiguous(mx::slice(fused, {s.width, 0}, {2 * s.width, s.hidden})),
                           mx::contiguous(tensors.at(qwen + "out.weight"))};
                weight_source = qwen + "gate_up.weight";
            } else if (s.kind == Kind::SwiGLU && tensors.count(z + "w1.weight")) {
                for (const auto *name : {"w1.weight", "w3.weight", "w2.weight"})
                    weights.push_back(mx::contiguous(tensors.at(z + name)));
                weight_source = z + "w1.weight";
            } else {
                for (const auto &name : names) weights.push_back(mx::contiguous(tensors.at(name)));
                weight_source = names.front();
            }
        }
        const bool captured_input = argc > 8 && std::string(argv[8]) != "-";
        if (captured_input) x = mx::contiguous(mx::astype(mx::reshape(mx::load(argv[8]), {rows, s.hidden}), weights[0].dtype()));
        if (x.shape() != mx::Shape{rows, s.hidden}) throw std::runtime_error("bundle input geometry mismatch");
        mx::eval(weights);
        mx::eval(x);
        std::vector<tc::ane::MatrixView> sources;
        for (const auto &w : weights) sources.push_back(view(w));
        auto operation = [kind = s.kind, separate_qkv_gpu, parts_qkv, hidden = s.hidden](const std::vector<mx::array> &a) {
            if (parts_qkv) {
                std::vector<mx::array> projections;
                for (int i = 1; i <= 3; ++i)
                    projections.push_back(mx::matmul(a[0], mx::transpose(a[i])));
                return std::vector<mx::array>{mx::concatenate(projections, 1)};
            }
            if (separate_qkv_gpu) {
                std::vector<mx::array> projections;
                for (int begin = 0; begin < 3 * hidden; begin += hidden) {
                    auto weight = mx::slice(a[1], {begin, 0}, {begin + hidden, hidden});
                    projections.push_back(mx::matmul(a[0], mx::transpose(weight)));
                }
                return std::vector<mx::array>{mx::concatenate(projections, 1)};
            }
            auto y = mx::matmul(a[0], mx::transpose(a[1]));
            if (kind == Kind::SwiGLU)
                y = mx::matmul((y * mx::sigmoid(y)) * mx::matmul(a[0], mx::transpose(a[2])), mx::transpose(a[3]));
            else if (kind == Kind::GELU) {
                y = .5f * y * (1.f + mx::tanh(.7978845608f * (y + .044715f * y * y * y)));
                y = mx::matmul(y, mx::transpose(a[2]));
            }
            return std::vector<mx::array>{y};
        };
        auto compiled = mx::compile(operation);
        auto gpu = [&](const mx::array &input) {
            std::vector<mx::array> args{input};
            args.insert(args.end(), weights.begin(), weights.end());
            return compiled(args)[0];
        };
        auto input = view(x);
        input.data = static_cast<const char *>(input.data) + size_t(gpu_rows) * s.hidden * x.itemsize();
        input.bytes -= size_t(gpu_rows) * s.hidden * x.itemsize();
        input.rows = ane_rows;
        std::vector<uint16_t> output(size_t(ane_rows) * s.output_width());
        const auto output_dtype = x.dtype() == mx::bfloat16 ? tc::ane::DType::BF16 : tc::ane::DType::FP16;
        std::vector<ProbeSample> samples;
        samples.reserve(repeats);
        uint64_t overflow_retries = 0;
        float headroom_scale = 1.f;
        auto reference = gpu(x);
        mx::eval(reference);
        auto candidate = reference;
        auto run_gpu = [&] {
            auto start = Clock::now();
            reference = gpu(x);
            mx::eval(reference);
            return elapsed(start);
        };
        auto run_split = [&] {
            const auto full_start = Clock::now();
            if (parts_qkv) graph.stage_matmul_parts(sources);
            else graph.stage(sources);
            auto staged = graph.wait_stage();
            if (!staged.ok) throw std::runtime_error("stage: " + staged.error);
            const auto start = Clock::now();
            graph.launch(input, output.data(), output.size(), output_dtype);
            auto head = gpu(mx::slice(x, {0, 0}, {gpu_rows, s.hidden}));
            mx::eval(head); // independent GPU rows run while Core ML processes the tail
            const auto join_start = Clock::now();
            auto result = graph.finish();
            const double join = elapsed(join_start);
            if (!result.ok) throw std::runtime_error("predict: " + result.error);
            if (result.calls != uint64_t(ane_rows / s.rows) + result.overflow_retries)
                throw std::runtime_error("runtime chunk count mismatch");
            overflow_retries += result.overflow_retries;
            headroom_scale = result.headroom_scale;
            auto tail = output_dtype == tc::ane::DType::BF16
                ? mx::array(reinterpret_cast<const mx::bfloat16_t *>(output.data()), {ane_rows, s.output_width()}, mx::bfloat16)
                : mx::array(reinterpret_cast<const mx::float16_t *>(output.data()), {ane_rows, s.output_width()}, mx::float16);
            candidate = mx::concatenate({head, mx::astype(tail, head.dtype())}, 0);
            mx::eval(candidate);
            const double split = elapsed(start), total = elapsed(full_start);
            return ProbeSample{0, split, total, staged.stage_seconds, result.prediction_seconds, join,
                               result.input_seconds, result.output_seconds, result.total_seconds,
                               result.calls, result.overflow_retries, result.headroom_scale};
        };
        for (int iteration = 0; iteration < repeats + 2; ++iteration) {
            double full;
            ProbeSample split;
            // Alternating order, two excluded warmups. Staging is reported both
            // separately and exposed: there is NO attention overlap in this tool.
            if (iteration % 2) { split = run_split(); full = run_gpu(); }
            else { full = run_gpu(); split = run_split(); }
            if (iteration < 2) continue;
            split.gpu_seconds = full;
            samples.push_back(split);
        }
        auto ref32 = mx::astype(reference, mx::float32), got32 = mx::astype(candidate, mx::float32);
        auto difference = ref32 - got32;
        const float relative = mx::sqrt(mx::sum(difference * difference) / mx::sum(ref32 * ref32)).item<float>();
        const float maximum = mx::max(mx::abs(difference)).item<float>();
        const float cosine = (mx::sum(ref32 * got32) / mx::sqrt(mx::sum(ref32 * ref32) * mx::sum(got32 * got32))).item<float>();
        if (!std::isfinite(relative) || relative > .05f) throw std::runtime_error(
            "runtime-weight split relative L2 exceeds probe gate: " + std::to_string(relative));
        // Independently check dense reductions against CPU FP32 on small graphs.
        bool cpu_checked = false;
        if (s.hidden <= 128 && s.width <= 128 && rows <= 256) {
            auto xf = mx::astype(x, mx::float32);
            std::vector<mx::array> wf;
            for (const auto &w : weights) wf.push_back(mx::astype(w, mx::float32));
            mx::eval(wf); mx::eval(xf, got32);
            const float *xp = xf.data<float>(), *yp = got32.data<float>();
            double squared = 0, norm = 0;
            for (int r = gpu_rows; r < rows; ++r) {
                std::vector<float> hidden(s.width);
                for (int c = 0; c < s.width; ++c) {
                    float g = 0, u = 0;
                    for (int k = 0; k < s.hidden; ++k) {
                        const int part = parts_qkv ? c / s.hidden : 0;
                        const int local = parts_qkv ? c % s.hidden : c;
                        g += xp[r * s.hidden + k] * wf[part].data<float>()[local * s.hidden + k];
                        if (s.kind == Kind::SwiGLU) u += xp[r * s.hidden + k] * wf[1].data<float>()[c * s.hidden + k];
                    }
                    hidden[c] = s.kind == Kind::SwiGLU ? g / (1.f + std::exp(-g)) * u :
                        s.kind == Kind::GELU ? .5f * g * (1.f + std::tanh(.7978845608f * (g + .044715f * g * g * g))) : g;
                }
                for (int c = 0; c < s.output_width(); ++c) {
                    float expected = s.kind == Kind::Matmul ? hidden[c] : 0;
                    if (s.kind != Kind::Matmul) for (int k = 0; k < s.width; ++k)
                        expected += hidden[k] * wf.back().data<float>()[c * s.width + k];
                    const float delta = yp[r * s.output_width() + c] - expected;
                    squared += delta * delta; norm += expected * expected;
                }
            }
            if (std::sqrt(squared / norm) > .03) throw std::runtime_error("CPU FP32 dense reduction check failed");
            cpu_checked = true;
        }
        // Malformed calls must return failure rather than old/partial output.
        auto invalid = input; invalid.bytes = 1;
        graph.launch(invalid, output.data(), output.size());
        if (graph.finish().ok) throw std::runtime_error("short input incorrectly accepted");
        graph.launch(input, output.data(), output.size() - 1);
        if (graph.finish().ok) throw std::runtime_error("short output incorrectly accepted");
        auto invalid_sources = sources;
        invalid_sources.front().bytes = 1;
        if (parts_qkv) graph.stage_matmul_parts(invalid_sources);
        else graph.stage(invalid_sources);
        if (graph.wait_stage().ok) throw std::runtime_error("short weights incorrectly accepted");
        graph.launch(input, output.data(), output.size(), output_dtype);
        if (graph.finish().ok) throw std::runtime_error("failed stage reused stale weights");
        if (parts_qkv) {
            auto incomplete = sources;
            incomplete.pop_back();
            graph.stage_matmul_parts(incomplete);
            if (graph.wait_stage().ok) throw std::runtime_error("incomplete MatMul weight slot was accepted");
            graph.launch(input, output.data(), output.size(), output_dtype);
            if (graph.finish().ok) throw std::runtime_error("incomplete stage reused stale weights");
        }
        if (parts_qkv) graph.stage_matmul_parts(sources);
        else graph.stage(sources);
        if (!graph.wait_stage().ok) throw std::runtime_error("stage did not recover after rejection");
        graph.launch(input, output.data(), output.size(), output_dtype);
        if (!graph.finish().ok) throw std::runtime_error("prediction did not recover after rejection");
        const double gpu_median = median(samples, &ProbeSample::gpu_seconds);
        const double split_median = median(samples, &ProbeSample::split_seconds);
        const double total_median = median(samples, &ProbeSample::split_with_stage_seconds);
        std::cout << std::setprecision(17)
                  << "{\"scope\":\"runtime-weight FFN/matmul component only\",\"observed_ane_residency\":\"unknown\""
                  << ",\"scalar_staging\":" << (scalar ? "true" : "false")
                  << ",\"real_weights\":" << (argc >= 6 && std::string(argv[5]) != "-" ? "true" : "false")
                  << ",\"weight_source\":\"" << weight_source << "\""
                  << ",\"projection_pack\":\"" << (parts_qkv ? "parts" : pack_qkv ? "qkv" : "none") << "\""
                  << ",\"gpu_projection\":\"" << (separate_qkv_gpu || parts_qkv ? "separate" : "single") << "\""
                  << ",\"captured_input\":" << (captured_input ? "true" : "false")
                  << ",\"overflow_retries_including_warmup\":" << overflow_retries
                  << ",\"headroom_scale\":" << headroom_scale
                  << ",\"cpu_reference_checked\":" << (cpu_checked ? "true" : "false")
                  << ",\"rows\":" << rows << ",\"ane_rows\":" << ane_rows << ",\"chunk\":" << s.rows
                  << ",\"gpu_rows\":" << gpu_rows << ",\"hidden\":" << s.hidden << ",\"width\":" << s.width
                  << ",\"tile_k\":" << s.tile_k << ",\"tile_n\":" << s.tile_n
                  << ",\"lora_inputs\":" << (s.lora_inputs ? "true" : "false")
                  << ",\"warmup_iterations\":2,\"measured_iterations\":" << repeats
                  << ",\"load_seconds\":" << graph.load_seconds() << ",\"slot_bytes\":" << graph.slot_bytes()
                  << ",\"estimated_bytes\":" << graph.estimated_bytes()
                  << ",\"relative_l2\":" << relative << ",\"max_abs\":" << maximum << ",\"cosine\":" << cosine
                  << ",\"gpu_seconds\":" << gpu_median << ",\"split_seconds\":" << split_median
                  << ",\"split_with_stage_seconds\":" << total_median
                  << ",\"stage_seconds\":" << median(samples, &ProbeSample::stage_seconds)
                  << ",\"predict_seconds\":" << median(samples, &ProbeSample::predict_seconds)
                  << ",\"join_seconds\":" << median(samples, &ProbeSample::join_seconds)
                  << ",\"input_seconds\":" << median(samples, &ProbeSample::input_seconds)
                  << ",\"output_seconds\":" << median(samples, &ProbeSample::output_seconds)
                  << ",\"runtime_seconds\":" << median(samples, &ProbeSample::runtime_seconds)
                  << ",\"speedup_without_stage\":" << gpu_median / split_median
                  << ",\"speedup_with_stage\":" << gpu_median / total_median
                  << ",\"samples\":[";
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto &sample = samples[i];
            if (i) std::cout << ',';
            std::cout << "{\"iteration\":" << i << ",\"order\":\""
                      << (i % 2 ? "split,gpu" : "gpu,split") << "\""
                      << ",\"gpu_seconds\":" << sample.gpu_seconds
                      << ",\"split_seconds\":" << sample.split_seconds
                      << ",\"split_with_stage_seconds\":" << sample.split_with_stage_seconds
                      << ",\"stage_seconds\":" << sample.stage_seconds
                      << ",\"predict_seconds\":" << sample.predict_seconds
                      << ",\"join_seconds\":" << sample.join_seconds
                      << ",\"input_seconds\":" << sample.input_seconds
                      << ",\"output_seconds\":" << sample.output_seconds
                      << ",\"runtime_seconds\":" << sample.runtime_seconds
                      << ",\"calls\":" << sample.calls
                      << ",\"overflow_retries\":" << sample.overflow_retries
                      << ",\"headroom_scale\":" << sample.headroom_scale << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
