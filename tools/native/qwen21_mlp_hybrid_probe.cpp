// Experimental component benchmark, not an end-to-end hybrid qualification.
#include "../../native/backends/coreml.hpp"
#include <algorithm>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4 || argc == 5, "usage: qwen21-mlp-hybrid-probe CHECKPOINT COMPILED_MANIFEST OUTPUT [INPUT]");
        tc::configure_streams();
        namespace mx = tc::mx;
        std::atomic<bool> cancelled{false};
        const auto checkpoint = std::filesystem::absolute(argv[1]);
        tc::Event event = [](const std::string &phase, int completed, int total) {
            std::cerr << phase << ' ' << completed << '/' << total << std::endl;
        };
        tc::HybridSession ane(std::filesystem::absolute(argv[2]), checkpoint.parent_path(), 1024,
                              event, cancelled, 1, checkpoint, {}, 1024, 1);
        tc::require(ane.hidden == 4096 && ane.mlp_width == 12288 && ane.ane_mlp_start == 0 &&
                    ane.ane_mlp_end < 12288, "Qwen21 experimental MLP shape mismatch");
        tc::Weights weights;
        weights.load_file(checkpoint);
        auto fused = weights.at("transformer_blocks.0.img_mlp.gate_up.weight");
        auto down = weights.at("transformer_blocks.0.img_mlp.out.weight");
        int width = ane.ane_mlp_end;
        auto suffix = mx::contiguous(mx::concatenate({tc::slice_axis(fused, 0, width, 12288),
                                                     tc::slice_axis(fused, 0, 12288 + width, 24576)}, 0));
        auto suffix_down = mx::contiguous(tc::slice_axis(down, 1, width, 12288));
        auto prefix = mx::contiguous(mx::concatenate({tc::slice_axis(fused, 0, 0, width),
                                                     tc::slice_axis(fused, 0, 12288, 12288 + width)}, 0));
        auto prefix_down = mx::contiguous(tc::slice_axis(down, 1, 0, width));
        auto x = mx::random::normal({1, 1024, 4096}, mx::float32, mx::random::key(42)) * .2f;
        if (argc == 5) {
            auto [inputs, metadata] = mx::load_safetensors(argv[4]);
            const auto &saved = inputs.at(inputs.count("mlp_input") ? "mlp_input" : "tensor");
            x = tc::slice_axis(saved, 1, saved.shape(1) - 1024, saved.shape(1));
        }
        x = mx::astype(x, mx::bfloat16);
        mx::eval(fused, down, suffix, suffix_down, prefix, prefix_down, x);
        weights.clear();
        auto mlp = mx::compile([](const std::vector<tc::Tensor> &a) {
            auto parts = mx::split(mx::matmul(a[0], mx::transpose(a[1])), 2, -1);
            return std::vector<tc::Tensor>{mx::matmul(tc::silu(parts[0]) * parts[1], mx::transpose(a[2]))};
        });
        auto hybrid = [&]() {
            auto packed = mx::contiguous(mx::astype(x, mx::float16));
            mx::eval(packed);
            auto gpu = mlp({x, suffix, suffix_down})[0];
            mx::async_eval(gpu);
            auto prediction = ane.predict(0, packed);
            auto result = gpu + mx::astype(prediction, gpu.dtype()) * tc::Tensor(ane.output_scale, gpu.dtype());
            mx::eval(result); // consume shared Core ML output before the next call
            return result;
        };
        auto reference = mlp({x, fused, down})[0];
        auto prefix_reference = mlp({x, prefix, prefix_down})[0];
        mx::eval(reference, prefix_reference);
        auto result = hybrid();
        std::vector<float> gpu_seconds, hybrid_seconds;
        for (int i = 0; i < 12; ++i) {
            auto start = tc::Clock::now();
            auto gpu = mlp({x, fused, down})[0];
            mx::eval(gpu);
            double gpu_elapsed = std::chrono::duration<double>(tc::Clock::now() - start).count();
            start = tc::Clock::now();
            result = hybrid();
            double hybrid_elapsed = std::chrono::duration<double>(tc::Clock::now() - start).count();
            if (i >= 2) { gpu_seconds.push_back(gpu_elapsed); hybrid_seconds.push_back(hybrid_elapsed); }
            std::cout << "{\"iteration\":" << i << ",\"gpu_seconds\":" << gpu_elapsed
                      << ",\"hybrid_seconds\":" << hybrid_elapsed << "}\n";
        }
        auto expected = mx::astype(reference, mx::float32);
        auto delta = mx::astype(result, mx::float32) - expected;
        float rrms = mx::sqrt(mx::mean(delta * delta) / mx::mean(expected * expected)).item<float>();
        tc::require(mx::all(mx::isfinite(result)).item<bool>(), "nonfinite experimental hybrid output");
        std::sort(gpu_seconds.begin(), gpu_seconds.end());
        std::sort(hybrid_seconds.begin(), hybrid_seconds.end());
        auto median = [](const std::vector<float> &v) { return (v[4] + v[5]) / 2; };
        std::cout << "{\"scope\":\"single MLP only; CPU+ANE policy does not prove ANE placement\",\"relative_rmse\":"
                  << rrms << ",\"gpu_median\":" << median(gpu_seconds) << ",\"hybrid_median\":"
                  << median(hybrid_seconds) << ",\"speedup\":" << median(gpu_seconds) / median(hybrid_seconds) << "}\n";
        mx::save_safetensors(argv[3], {{"input", x}, {"reference", reference}, {"hybrid", result},
            {"prefix_reference", prefix_reference}, {"gpu_seconds", tc::Tensor(gpu_seconds.data(), {10}, mx::float32)},
            {"hybrid_seconds", tc::Tensor(hybrid_seconds.data(), {10}, mx::float32)}});
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
