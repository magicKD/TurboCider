// Component-only GPU screen: three original projections vs packed QKV.
#include "../../native/backends/mlx.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 5 && !std::filesystem::exists(argv[2]),
                    "usage: ltx098-qkv-screen CHECKPOINT NEW_OUTPUT ROWS REPEATS");
        const int rows = std::stoi(argv[3]), repeats = std::stoi(argv[4]);
        tc::require(rows > 0 && rows <= 4096 && repeats >= 2 && repeats <= 20, "invalid screen dimensions");
        tc::Weights weights;
        weights.load_file(argv[1], "model.diffusion_model.transformer_blocks.0.");
        for (const auto *suffix : {"to_q", "to_k", "to_v"}) {
            const auto prefix = "attn1." + std::string(suffix);
            tc::require(weights.has(prefix + ".weight") && weights.has(prefix + ".bias") &&
                        weights.at(prefix + ".weight").shape() == tc::mx::Shape{2048, 2048},
                        "unexpected LTX 0.9.8 QKV checkpoint geometry");
        }
        weights.materialize();
        const auto started = std::chrono::steady_clock::now();
        auto packed_weight = tc::mx::concatenate({weights.at("attn1.to_q.weight"), weights.at("attn1.to_k.weight"),
                                                   weights.at("attn1.to_v.weight")}, 0);
        auto packed_bias = tc::mx::concatenate({weights.at("attn1.to_q.bias"), weights.at("attn1.to_k.bias"),
                                                 weights.at("attn1.to_v.bias")}, 0);
        tc::mx::eval(packed_weight, packed_bias);
        const double prepare = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        auto x = tc::mx::astype(tc::mx::random::normal({1, rows, 2048}, tc::mx::float32, tc::mx::random::key(71)) * .5f,
                                tc::mx::bfloat16);
        tc::mx::eval(x);
        auto separate = [&] { return weights.project_many(x, {"attn1.to_q", "attn1.to_k", "attn1.to_v"}); };
        auto fused = [&] {
            auto value = tc::mx::matmul(x, tc::mx::transpose(packed_weight)) + packed_bias;
            return tc::mx::split(value, 3, -1);
        };
        std::vector<std::vector<double>> times(4);
        std::vector<tc::Tensor> reference, candidate;
        for (int group = 0; group < 4; ++group) {
            const bool packed = group == 1 || group == 2;
            for (int iteration = 0; iteration < repeats + 1; ++iteration) {
                const auto clock = std::chrono::steady_clock::now();
                auto values = packed ? fused() : separate();
                tc::mx::eval(values);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - clock).count();
                if (iteration) times[group].push_back(seconds);
                if (group == 0) reference = values;
                if (group == 2) candidate = values;
            }
        }
        tc::mx::save_safetensors(argv[2], {{"ref_q", reference[0]}, {"ref_k", reference[1]}, {"ref_v", reference[2]},
                                          {"packed_q", candidate[0]}, {"packed_k", candidate[1]}, {"packed_v", candidate[2]}});
        std::cout << std::setprecision(12) << "{\"rows\":" << rows << ",\"prepare_seconds\":" << prepare
                  << ",\"packed_bytes\":" << packed_weight.nbytes() + packed_bias.nbytes()
                  << ",\"peak_mlx_bytes\":" << tc::mx::get_peak_memory() << ",\"runs\":[";
        for (int group = 0; group < 4; ++group) {
            std::cout << (group ? "," : "") << "{\"packed\":" << ((group == 1 || group == 2) ? "true" : "false") << ",\"seconds\":[";
            for (size_t i = 0; i < times[group].size(); ++i) std::cout << (i ? "," : "") << times[group][i];
            std::cout << "]}";
        }
        std::cout << "]}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
