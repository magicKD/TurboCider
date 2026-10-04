#include "../../native/models/qwen21/runtime_activation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <unordered_map>

// Tiny real compiled GPU FFNs, not model inference. In particular, the eager
// oracle must never compile a multi-output split: that would contaminate the
// live-buffer measurement with the very ownership regression under test.
namespace {
using namespace tc;
constexpr int hidden = 64, intermediate = 96, rank = 4;
constexpr float adapter_strength = .75f;
const std::string prefix = "transformer_blocks.0.img_mlp.";
using AdapterArrays = std::unordered_map<std::string, Tensor>;
using Graph = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;
enum class Variant { ParameterizedBase, CapturedBase, CapturedLoRA };

const char *name(Variant variant) {
    switch (variant) {
        case Variant::ParameterizedBase: return "parameterized_base";
        case Variant::CapturedBase: return "captured_weights_base";
        case Variant::CapturedLoRA: return "captured_weights_runtime_lora";
    }
    throw std::logic_error("unknown runtime activation variant");
}

Tensor pattern(const mx::Shape &shape, int salt, float scale) {
    size_t count = 1;
    for (int dimension : shape) count *= size_t(dimension);
    std::vector<float> values(count);
    // Host construction avoids a persistent random-generator tensor changing
    // the cold active-memory baseline when a new shape is first encountered.
    // SplitMix64 decorrelates rows, columns and fixtures. A sinusoidal grid
    // can accidentally make one input/weight pair nearly orthogonal, hiding
    // useful activation values for a particular row count.
    for (size_t i = 0; i < count; ++i) {
        uint64_t bits = (uint64_t(uint32_t(salt)) << 32) ^ uint64_t(i + 1);
        bits += 0x9e3779b97f4a7c15ull;
        bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ull;
        bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebull;
        bits ^= bits >> 31;
        values[i] = scale * (float(bits >> 40) / 8388608.f - 1.f);
    }
    auto result = mx::astype(Tensor(values.data(), shape, mx::float32), mx::bfloat16);
    mx::eval(result);
    return result;
}

void write_adapter(const std::filesystem::path &path) {
    AdapterArrays arrays;
    int salt = 30;
    auto pair = [&](const std::string &branch, int output, int input) {
        const auto stem = "transformer." + prefix + branch;
        arrays.emplace(stem + ".lora_A.weight", pattern({rank, input}, salt++, .15f));
        arrays.emplace(stem + ".lora_B.weight", pattern({output, rank}, salt++, .12f));
        arrays.emplace(stem + ".alpha", Tensor(float(rank), mx::float32));
    };
    // These separate logical branches exercise the real loader's fused
    // gate_up row mapping as well as the output-projection adapter.
    pair("gate_layer", intermediate, hidden);
    pair("proj", intermediate, hidden);
    pair("out", hidden, intermediate);
    mx::save_safetensors(path.string(), arrays);
}

std::vector<float> host_values(const Tensor &value) {
    auto converted = mx::astype(value, mx::float32);
    mx::eval(converted);
    const auto *data = converted.data<float>();
    return {data, data + converted.size()};
}

Tensor eager_projection(const Tensor &input, const Tensor &weight,
                        const AdapterArrays &adapters, const std::string &branch) {
    auto output = mx::matmul(input, mx::transpose(weight));
    mx::eval(output);
    if (!adapters.empty()) {
        const auto stem = "transformer." + prefix + branch;
        const auto &a = adapters.at(stem + ".lora_A.weight");
        const auto &b = adapters.at(stem + ".lora_B.weight");
        // Match the documented runtime low-rank FP32 accumulation and final
        // BF16 cast, independently of Weights::project and its row mapping.
        auto low = mx::matmul(mx::astype(input, mx::float32),
                              mx::transpose(mx::astype(a, mx::float32)));
        auto delta = mx::matmul(low, mx::transpose(mx::astype(b, mx::float32))) *
                     Tensor(adapter_strength, mx::float32); // alpha/rank = 1
        output = mx::astype(mx::astype(output, mx::float32) + delta, mx::bfloat16);
        mx::eval(output);
    }
    return output;
}

Tensor eager_ffn(const Tensor &input, const Tensor &gate_up, const Tensor &down,
                 const AdapterArrays &adapters) {
    // Separate matrix projections and explicit eager SiLU are the oracle.
    // Neither the production activation helper nor a compiled split is used.
    auto gate_weight = slice_axis(gate_up, 0, 0, intermediate);
    auto up_weight = slice_axis(gate_up, 0, intermediate, 2 * intermediate);
    mx::eval(gate_weight, up_weight);
    auto gate = eager_projection(input, gate_weight, adapters, "gate_layer");
    auto up = eager_projection(input, up_weight, adapters, "proj");
    auto sigmoid = mx::sigmoid(gate);
    mx::eval(sigmoid);
    auto activated = gate * sigmoid;
    mx::eval(activated);
    auto joined = activated * up;
    mx::eval(joined);
    return eager_projection(joined, down, adapters, "out");
}

struct Sample {
    std::vector<float> values;
    double max_error = 0, relative_l2 = 0;
    bool exact = true;
};

Sample compare(const Tensor &actual, const Tensor &expected, int rows,
               Variant variant) {
    require(actual.shape() == mx::Shape({1, rows, hidden}) &&
                expected.shape() == actual.shape(), "runtime FFN output shape changed");
    require(actual.dtype() == mx::bfloat16 && expected.dtype() == mx::bfloat16,
            "runtime FFN output must remain BF16");
    Sample result;
    result.values = host_values(actual);
    const auto oracle = host_values(expected);
    double squared_error = 0, squared_reference = 0, maximum_reference = 0;
    for (size_t i = 0; i < oracle.size(); ++i) {
        require(std::isfinite(oracle[i]) && std::isfinite(result.values[i]),
                "nonfinite runtime FFN output");
        const double difference = double(result.values[i]) - oracle[i];
        result.max_error = std::max(result.max_error, std::abs(difference));
        maximum_reference = std::max(maximum_reference, std::abs(double(oracle[i])));
        squared_error += difference * difference;
        squared_reference += double(oracle[i]) * oracle[i];
        result.exact = result.exact && result.values[i] == oracle[i];
    }
    require(squared_reference > 1e-12 && maximum_reference > 1e-4,
            "runtime FFN oracle is numerically trivial");
    result.relative_l2 = std::sqrt(squared_error / squared_reference);
    // Compilation may fuse BF16 SiLU/multiply boundaries that the eager
    // oracle rounds separately. Keep both RMS and absolute error bounded;
    // two BF16 relative quantization intervals are not a memory allowance.
    const double max_allowed_error = maximum_reference / 64.0 + 1e-7;
    std::cout << "numeric variant=" << name(variant) << " rows=" << rows
              << " exact=" << result.exact << " max_abs=" << result.max_error
              << " relative_l2=" << result.relative_l2 << '\n';
    require(result.relative_l2 <= .01 && result.max_error <= max_allowed_error,
            std::string("compiled runtime FFN disagrees with eager gate/up math: ") +
                name(variant) + " rows=" + std::to_string(rows));
    return result;
}

std::array<Sample, 2> run_cycle(Variant variant, const std::filesystem::path &adapter) {
    // Every invocation owns fresh materialized allocations. A reference
    // capture still lets MLX's traced graph capture those arrays internally,
    // exactly like the production request-local runtime LoRA GPU closure.
    auto gate_up = pattern({2 * intermediate, hidden}, 10, .07f);
    auto down = pattern({hidden, intermediate}, 20, .05f);
    auto gate_snapshot = host_values(gate_up), down_snapshot = host_values(down);
    Weights weights;
    weights.bind_arrays({prefix + "gate_up.weight", prefix + "out.weight"}, {gate_up, down});
    AdapterArrays adapters;
    if (variant == Variant::CapturedLoRA) {
        std::atomic<bool> cancelled{false};
        require(weights.apply_loras({{adapter.string(), adapter_strength, "transformer"}},
                    "transformer", [](const std::string &, int, int) {}, cancelled,
                    true, true) == 3, "tiny runtime adapter must bind all three projections");
        require(weights.has_runtime_loras(), "tiny adapter was merged instead of retained");
        adapters = mx::load_safetensors(adapter.string()).first;
    }
    weights.materialize();
    require(host_values(weights.at(prefix + "gate_up.weight")) == gate_snapshot &&
                host_values(weights.at(prefix + "out.weight")) == down_snapshot,
            "runtime adapter changed the immutable base weights");

    Graph graph;
    if (variant == Variant::ParameterizedBase) {
        graph = mx::compile([](const std::vector<Tensor> &a) {
            auto fused = mx::matmul(a[0], mx::transpose(a[1]));
            return std::vector<Tensor>{mx::matmul(qwen21::runtime_ffn_activation(fused),
                                                mx::transpose(a[2]))};
        });
    } else {
        graph = mx::compile([&weights](const std::vector<Tensor> &a) {
            auto fused = weights.project(a[0], prefix + "gate_up");
            return std::vector<Tensor>{weights.project(qwen21::runtime_ffn_activation(fused),
                                                       prefix + "out")};
        });
    }
    std::array<Sample, 2> samples;
    int slot = 0;
    for (int rows : {33, 97}) {
        auto input = pattern({1, rows, hidden}, 40 + rows, .4f);
        auto expected = eager_ffn(input, gate_up, down, adapters);
        auto outputs = variant == Variant::ParameterizedBase ?
            graph({input, weights.at(prefix + "gate_up.weight"), weights.at(prefix + "out.weight")}) :
            graph({input});
        require(outputs.size() == 1, "runtime FFN must expose one completed output");
        mx::eval(outputs);
        samples[slot++] = compare(outputs[0], expected, rows, variant);
        if (variant == Variant::CapturedLoRA) {
            const auto base = host_values(eager_ffn(input, gate_up, down, {}));
            const auto &actual = samples[slot - 1].values;
            double changed = 0;
            for (size_t i = 0; i < base.size(); ++i)
                changed = std::max(changed, std::abs(double(actual[i]) - base[i]));
            require(changed > 1e-5, "tiny LoRA output did not exercise its low-rank branches");
        }
    }
    // Only host vectors leave this scope. The callable dies before weights;
    // every MLX input, oracle, output and adapter allocation dies before the
    // caller synchronizes and measures active memory.
    return samples;
}

size_t idle_active() {
    mx::synchronize();
    mx::clear_cache();
    require(mx::get_cache_memory() == 0, "MLX allocator cache did not drain");
    return mx::get_active_memory();
}
}

int main(int argc, char **argv) {
    try {
        require(argc == 2 && std::filesystem::is_directory(argv[1]),
                "usage: qwen21_runtime_activation_test TEMPORARY_FIXTURE_DIRECTORY");
        require(std::getenv("MLX_DISABLE_COMPILE") == nullptr,
                "runtime activation regression needs real MLX compilation");
        configure_streams();
        const size_t initial = idle_active();
        std::cout << "initial_active_bytes=" << initial << " memory_tolerance_bytes=0\n";
        const auto adapter = std::filesystem::path(argv[1]) / "tiny-runtime-activation.safetensors";
        write_adapter(adapter);
        require(idle_active() == initial, "tiny adapter writer retained active MLX allocations");

        constexpr std::array<Variant, 3> variants{Variant::ParameterizedBase,
                                                Variant::CapturedBase, Variant::CapturedLoRA};
        std::array<std::array<Sample, 2>, 3> first;
        for (int repetition = 0; repetition < 3; ++repetition) {
            for (size_t index = 0; index < variants.size(); ++index) {
                auto samples = run_cycle(variants[index], adapter);
                const size_t active = idle_active();
                std::cout << "ownership repetition=" << repetition << " variant=" << name(variants[index])
                          << " active_bytes=" << active << " baseline_bytes=" << initial << '\n';
                // No warm compiled-capture pass and no 32-KiB tolerance that
                // could hide the previous single-call multi-output split leak.
                require(active == initial,
                        std::string("compiled runtime FFN retained live arrays after destruction: ") +
                            name(variants[index]) + " repetition=" + std::to_string(repetition) +
                            " initial=" + std::to_string(initial) + " active=" + std::to_string(active));
                if (repetition == 0) first[index] = std::move(samples);
                else for (size_t shape = 0; shape < samples.size(); ++shape)
                    require(samples[shape].values == first[index][shape].values,
                            "fresh runtime FFN reconstruction changed deterministic output");
            }
        }
        require(idle_active() == initial, "runtime FFN idle memory grew after all rebuilds");
        std::cout << "PASS Qwen runtime activation ownership\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
