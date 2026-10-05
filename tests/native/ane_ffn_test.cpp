#include "../../native/backends/ane_ffn.hpp"
#include "../../native/backends/ane_runtime_quant.hpp"
#include <cassert>
#include <array>
#include <iostream>
#include <limits>

namespace {
void lora_alpha_tests(const std::filesystem::path &directory) {
    using namespace tc;
    const std::string prefix = "transformer_blocks.0.img_mlp.gate_up";
    const std::string stem = "transformer." + prefix;
    const auto path = (directory / "alpha-scalar.safetensors").string();
    std::atomic<bool> cancelled{false};
    auto bind = [&](const Tensor &alpha, Weights &weights, float strength = -.75f) {
        mx::save_safetensors(path, {{stem + ".lora_A.weight", mx::full({2, 64}, .125f, mx::bfloat16)},
                                  {stem + ".lora_B.weight", mx::full({192, 2}, .25f, mx::bfloat16)},
                                  {stem + ".alpha", alpha}});
        weights.bind_arrays({prefix + ".weight"}, {mx::zeros({192, 64}, mx::bfloat16)});
        weights.apply_loras({{path, strength}}, "transformer",
                            [](const std::string &, int, int) {}, cancelled, true);
        weights.materialize();
    };
    auto input = mx::full({1, 3, 64}, .5f, mx::bfloat16);
    // X A^T B^T = 2, alpha/rank = 1/2, strength = -3/4.
    auto expected = mx::full({1, 3, 192}, -.75f, mx::bfloat16);
    for (auto dtype : {mx::float32, mx::bfloat16, mx::float16, mx::int32}) {
        Weights weights;
        bind(Tensor(1.f, dtype), weights);
        auto projected = weights.project(input, prefix);
        auto correction = weights.lora_delta_slice(input, prefix, 0, 192, 0, 64);
        if (!mx::all(projected == expected).item<bool>() ||
            !mx::all(correction == expected).item<bool>())
            throw std::runtime_error("LoRA alpha dtype changed the mathematical scale");
        assert(mx::all(weights.at(prefix + ".weight") == 0).item<bool>());
    }
    for (auto alpha : {Tensor(std::numeric_limits<float>::infinity()),
                       Tensor(std::numeric_limits<float>::quiet_NaN()),
                       mx::ones({2}, mx::float32)}) {
        Weights weights;
        bool rejected = false;
        try { bind(alpha, weights); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected && weights.bytes() == 0);
    }
    for (float strength : {std::numeric_limits<float>::infinity(), 4.f}) {
        Weights weights;
        bool rejected = false;
        try { bind(Tensor(std::numeric_limits<float>::max()), weights, strength); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected && weights.bytes() == 0);
    }
    std::cout << "PASS LoRA alpha: F32/BF16/F16/integer numeric conversion, signed strength, "
                 "unchanged base and nonfinite/nonscalar/overflow rejection\n";
}

void qwen_lora_correction_tests(const std::filesystem::path &directory) {
    using namespace tc;
    const std::string prefix = "transformer_blocks.0.img_mlp.gate_up";
    int seed = 730;
    auto random = [&](mx::Shape shape) {
        return mx::astype(mx::random::normal(shape, mx::float32, mx::random::key(seed++)) * .2f,
                          mx::bfloat16);
    };
    auto save = [&](const char *name, const std::vector<std::string> &branches) {
        std::unordered_map<std::string, Tensor> tensors;
        for (const auto &branch : branches) {
            const auto p = "transformer.transformer_blocks.0.img_mlp." + branch;
            tensors.emplace(p + ".lora_A.weight", random({4, 64}));
            tensors.emplace(p + ".lora_B.weight", random({branch == "gate_up" ? 192 : 96, 4}));
        }
        const auto path = directory / name;
        mx::save_safetensors(path.string(), tensors);
        return path.string();
    };
    const auto separate = save("slice-separate.safetensors", {"gate_layer", "proj"});
    const auto gate = save("slice-gate.safetensors", {"gate_layer"});
    const auto up = save("slice-up.safetensors", {"proj"});
    const auto fused = save("slice-fused.safetensors", {"gate_up"});
    const std::vector<std::vector<LoRAAsset>> cases{
        {}, {{separate, 1.f}}, {{separate, -.75f}}, {{gate, 1.f}}, {{up, 1.f}},
        {{fused, 1.f}}, {{separate, 1.f}, {fused, -.75f}}, {{separate, 0.f}}};
    std::atomic<bool> cancelled{false};
    for (auto dtype : {mx::bfloat16, mx::float16, mx::float32}) {
        for (bool narrow : {false, true}) {
            for (size_t which = 0; which < cases.size(); ++which) {
                Weights weights;
                auto base = mx::astype(random({192, 64}), dtype);
                auto snapshot = mx::copy(base);
                mx::eval(base, snapshot);
                weights.bind_arrays({prefix + ".weight"}, {base});
                weights.set_runtime_lora_fp16(narrow);
                if (!cases[which].empty())
                    assert(weights.apply_loras(cases[which], "transformer",
                        [](const std::string &, int, int) {}, cancelled, true) > 0);
                weights.materialize();
                auto packed = mx::compile([&](const std::vector<Tensor> &a) {
                    auto gu = mx::split(weights.lora_delta_slice(a[0], prefix, 0, 192, 0, 64), 2, -1);
                    return std::vector<Tensor>{mx::contiguous(gu[0]), mx::contiguous(gu[1])};
                });
                auto halves = mx::compile([&](const std::vector<Tensor> &a) {
                    return std::vector<Tensor>{
                        mx::contiguous(weights.lora_delta_slice(a[0], prefix, 0, 96, 0, 64)),
                        mx::contiguous(weights.lora_delta_slice(a[0], prefix, 96, 192, 0, 64))};
                });
                for (int rows : {33, 97}) {
                    auto input = mx::astype(random({1, rows, 64}), dtype);
                    auto expected = packed({input}), actual = halves({input});
                    mx::eval(expected); mx::eval(actual);
                    for (int i = 0; i < 2; ++i) {
                        assert(actual[i].shape() == mx::Shape({1, rows, 96}));
                        assert(actual[i].dtype() == dtype && actual[i].flags().row_contiguous);
                        assert(mx::all(mx::isfinite(actual[i])).item<bool>());
                        const bool equal = mx::all(actual[i] == expected[i]).item<bool>();
                        if (!equal) std::cerr << "Qwen correction mismatch: case=" << which
                            << " narrow=" << narrow << " half=" << i << " rows=" << rows << '\n';
                        assert(equal);
                    }
                }
                assert(mx::all(weights.at(prefix + ".weight") == snapshot).item<bool>());
            }
        }
    }
    std::cout << "PASS Qwen LoRA correction halves: separate/fused/stacked/partial/zero/signed, "
                 "BF16/FP16/FP32, FP32/FP16 rank branches, unchanged base\n";
}

void lora_tests(const char *manifest) {
    using namespace tc;
    std::atomic<bool> cancelled{false};
    auto random = [](mx::Shape shape, int seed, float scale) {
        return mx::astype(mx::random::normal(shape, mx::float32, mx::random::key(seed)) * scale, mx::bfloat16);
    };
    std::vector<Tensor> weights{random({96, 64}, 123, .08f), random({96, 64}, 124, .08f),
                                random({64, 96}, 125, .08f)};
    std::vector<Tensor> a{random({4, 64}, 126, .2f), random({4, 64}, 127, .2f),
                         random({4, 96}, 128, .2f)};
    std::vector<Tensor> b{random({96, 4}, 129, .3f), random({96, 4}, 130, .3f),
                         random({64, 4}, 131, .3f)};
    mx::eval(weights); mx::eval(a); mx::eval(b);
    auto x = random({1, 97, 64}, 132, .5f);
    float strength = 0.f;
    auto delta = [&](const Tensor &input, int i) {
        return mx::astype(mx::matmul(mx::matmul(mx::astype(input, mx::float32),
            mx::transpose(mx::astype(a[i], mx::float32))), mx::transpose(mx::astype(b[i], mx::float32))) * strength,
            input.dtype());
    };
    auto gpu = [&](const Tensor &input) {
        auto gate = mx::matmul(input, mx::transpose(weights[0])) + delta(input, 0);
        auto up = mx::matmul(input, mx::transpose(weights[1])) + delta(input, 1);
        auto hidden = (gate * mx::sigmoid(gate)) * up;
        return mx::matmul(hidden, mx::transpose(weights[2])) + delta(hidden, 2);
    };
    ane::HybridFfn::Adapter adapter{
        [&](const Tensor &input) { return std::make_pair(delta(input, 0), delta(input, 1)); },
        {}};
    ane::HybridFfn runtime(manifest, 64, 96, 128ull << 20, cancelled, true);
    std::optional<Tensor> first_base;
    std::optional<Tensor> retained_hidden, retained_base, hidden_snapshot, base_snapshot, lazy_tail;
    double worst = 0;
    for (float scale : {0.f, 0.f, 1.f, -.75f, 0.f, 0.f}) {
        const auto before = runtime.metrics();
        strength = scale;
        // Recreate a lazy input on EVERY request. The combined readiness
        // fence must own both this upstream graph and adapter corrections.
        x = random({1, 97, 64}, 132, .5f);
        // Like Qwen's request-local graph: never capture another adapter's
        // strength/weights in a closure reused across requests.
        auto down_add = mx::compile([&](const std::vector<Tensor> &inputs) {
            return std::vector<Tensor>{mx::astype(mx::astype(inputs[1], mx::float32) +
                mx::astype(delta(inputs[0], 2), mx::float32), inputs[1].dtype())};
        });
        adapter.down_and_add = [&, down_add](const Tensor &hidden, const Tensor &base) {
            if (retained_hidden) {
                assert(mx::all(mx::astype(*retained_hidden, mx::float32) == *hidden_snapshot).item<bool>());
                assert(mx::all(mx::astype(*retained_base, mx::float32) == *base_snapshot).item<bool>());
                assert(mx::all(*lazy_tail == mx::sum(*hidden_snapshot, -1) +
                                           mx::sum(*base_snapshot, -1)).item<bool>());
            }
            retained_hidden = hidden; retained_base = base;
            hidden_snapshot = mx::astype(hidden, mx::float32);
            base_snapshot = mx::astype(base, mx::float32);
            mx::eval({*hidden_snapshot, *base_snapshot});
            // Consume these borrowed callback views only AFTER another
            // prediction. The runtime must preserve their MLX ownership.
            lazy_tail = mx::sum(mx::astype(hidden, mx::float32), -1) +
                        mx::sum(mx::astype(base, mx::float32), -1);
            return down_add({hidden, base})[0];
        };
        auto probe_hidden = random({1, 64, 96}, 140, .5f);
        auto probe_base = random({1, 64, 64}, 141, .5f);
        auto packed_gate_up = mx::compile([&](const std::vector<Tensor> &inputs) {
            auto fused = mx::concatenate({delta(inputs[0], 0), delta(inputs[0], 1)}, -1);
            auto parts = mx::split(fused, 2, -1);
            return std::vector<Tensor>{mx::contiguous(parts[0]), mx::contiguous(parts[1])};
        });
        auto corrections = packed_gate_up({probe_base});
        mx::eval(corrections);
        for (int i = 0; i < 2; ++i) {
            assert(corrections[i].flags().row_contiguous);
            assert(mx::all(corrections[i] == delta(probe_base, i)).item<bool>());
        }
        auto probe_delta = delta(probe_hidden, 2);
        mx::eval(probe_delta); // the previous separated delta boundary
        auto separate = mx::astype(mx::astype(probe_base, mx::float32) +
            mx::astype(probe_delta, mx::float32), probe_base.dtype());
        assert(mx::all(down_add({probe_hidden, probe_base})[0] == separate).item<bool>());
        runtime.begin_request(std::to_string(scale));
        runtime.stage(0, 97, weights);
        auto y = runtime.run(0, x, gpu, cancelled, scale == 0.f ? nullptr : &adapter);
        auto expected = gpu(x);
        const float relative = mx::sqrt(mx::sum(mx::square(mx::astype(y, mx::float32) -
            mx::astype(expected, mx::float32))) / mx::sum(mx::square(mx::astype(expected, mx::float32)))).item<float>();
        worst = std::max(worst, double(relative));
        assert(std::isfinite(relative) && relative < .035f && !runtime.metrics().runtime_failed);
        const auto after = runtime.metrics();
        // Base consumes only y; a real adapter still owns a complete hidden
        // copy for down-LoRA. These are TWO 32-row chunks, not padded rows.
        assert(after.copied_bytes - before.copied_bytes ==
               uint64_t(64 * (64 + (scale == 0.f ? 0 : 96)) * 2));
        const double gate_time = after.runtime_weight_lora_gate_up_seconds - before.runtime_weight_lora_gate_up_seconds;
        const double ready_time = after.runtime_weight_lora_input_ready_seconds - before.runtime_weight_lora_input_ready_seconds;
        const double pre_time = after.runtime_weight_pre_seconds - before.runtime_weight_pre_seconds;
        const double post_time = after.runtime_weight_post_join_seconds - before.runtime_weight_post_join_seconds;
        const double ffn_time = after.runtime_weight_wall_seconds - before.runtime_weight_wall_seconds;
        assert(gate_time == 0); // no separate post-input correction barrier
        assert(std::isfinite(ready_time) && (scale == 0.f ? ready_time == 0 : ready_time > 0));
        assert(ready_time <= pre_time);
        assert(std::isfinite(post_time) && post_time > 0 && gate_time + post_time <= ffn_time);
        if (!first_base) first_base = y;
        else if (scale == 0.f) assert(mx::all(y == *first_base).item<bool>());
        else {
            // item<T>() reads storage as T; it does NOT convert BF16 to
            // float. Compute metrics in FP32 before reading their scalar.
            const float difference = mx::max(mx::abs(mx::astype(y, mx::float32) -
                mx::astype(*first_base, mx::float32))).item<float>();
            if (!(difference > .05f))
                std::cerr << "LoRA switch lost effect: scale=" << scale << " difference=" << difference
                          << " reference_difference=" << mx::max(mx::abs(mx::astype(expected, mx::float32) -
                              mx::astype(*first_base, mx::float32))).item<float>()
                          << " relative=" << relative << '\n';
            assert(difference > .05f);
        }
    }
    assert(runtime.metrics().runtime_calls == 12);
    // Reject a malformed gate/up result before exposing any borrowed host
    // view; the next request must drain a possibly outstanding weight stage.
    auto valid_gate_up = adapter.gate_up;
    for (bool bad_gate : {false, true}) {
        adapter.gate_up = [&, bad_gate](const Tensor &input) {
            auto deltas = valid_gate_up(input);
            if (bad_gate) deltas.first = slice_axis(deltas.first, 1, 0, input.shape(1) - 1);
            else deltas.second = slice_axis(deltas.second, 2, 0, 95);
            return deltas;
        };
        runtime.stage(0, 97, weights);
        bool rejected = false;
        try { runtime.run(0, x, gpu, cancelled, &adapter); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected && !runtime.metrics().runtime_failed);
        runtime.begin_request("malformed-correction-recovery");
    }
    adapter.gate_up = valid_gate_up;
    std::cout << "PASS runtime LoRA readiness: lazy upstream input, malformed corrections and stage drain\n";
    assert(mx::all(mx::astype(*retained_hidden, mx::float32) == *hidden_snapshot).item<bool>());
    assert(mx::all(mx::astype(*retained_base, mx::float32) == *base_snapshot).item<bool>());
    // A malformed complete-tail callback must fail closed, then release its
    // borrowed graph buffers so a subsequent request can still run normally.
    auto valid_down_add = adapter.down_and_add;
    for (bool wrong_shape : {false, true}) {
        adapter.down_and_add = [wrong_shape](const Tensor &, const Tensor &base) {
            return wrong_shape ? slice_axis(base, 1, 0, base.shape(1) - 1)
                               : mx::astype(base, mx::float32);
        };
        runtime.stage(0, 97, weights);
        bool rejected = false;
        try { runtime.run(0, x, gpu, cancelled, &adapter); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected && !runtime.metrics().runtime_failed);
    }
    adapter.down_and_add = valid_down_add;
    runtime.stage(0, 97, weights);
    auto recovered = runtime.run(0, x, gpu, cancelled, &adapter);
    assert(mx::all(recovered == *first_base).item<bool>());
    // Actual low-rank gate/up/down contributions remain outside the weights.
    assert(mx::all(weights[0] == random({96, 64}, 123, .08f)).item<bool>());
    // Nonzero up correction must scale on EVERY headroom retry, and hidden
    // must be restored before GPU down-LoRA. A large hidden with small base
    // down also catches overflow hidden by a finite y output.
    weights = {mx::full({96, 64}, 1.25f, mx::bfloat16),
               mx::full({96, 64}, 25.f, mx::bfloat16),
               mx::full({64, 96}, .001f, mx::bfloat16)};
    x = mx::ones({1, 97, 64}, mx::bfloat16);
    {
        // The v2 hidden can overflow even when the small down weights make
        // y fit FP16. A base-only launch must still inspect hidden and retry,
        // despite no longer materializing that unused output on the host.
        ane::HybridFfn base_runtime(manifest, 64, 96, 128ull << 20, cancelled, true);
        auto base_gpu = [&](const Tensor &input) {
            auto gate = mx::matmul(input, mx::transpose(weights[0]));
            auto up = mx::matmul(input, mx::transpose(weights[1]));
            return mx::matmul((gate * mx::sigmoid(gate)) * up, mx::transpose(weights[2]));
        };
        base_runtime.stage(1, 97, weights);
        auto actual = base_runtime.run(1, x, base_gpu, cancelled);
        auto expected_base = base_gpu(x);
        assert(mx::all(mx::isfinite(actual)).item<bool>());
        assert(mx::max(mx::abs(mx::astype(actual, mx::float32) - mx::astype(expected_base, mx::float32)) /
            mx::abs(mx::astype(expected_base, mx::float32))).item<float>() < .025f);
        assert(base_runtime.metrics().runtime_weight_overflow_retries > 0 && !base_runtime.metrics().runtime_failed);
    }
    auto high_gate_up = [](const Tensor &input) {
        return std::make_pair(mx::full({1, input.shape(1), 96}, 16.f, mx::bfloat16),
                              mx::full({1, input.shape(1), 96}, 800.f, mx::bfloat16));
    };
    auto high_down = [](const Tensor &hidden) {
        return mx::matmul(hidden, mx::full({96, 64}, .125f, mx::bfloat16));
    };
    ane::HybridFfn::Adapter high{high_gate_up,
        [&](const Tensor &hidden, const Tensor &base) {
            return mx::astype(mx::astype(base, mx::float32) +
                mx::astype(high_down(hidden), mx::float32), base.dtype());
        }};
    auto high_gpu = [&](const Tensor &input) {
        auto [dg, du] = high_gate_up(input);
        auto gate = mx::matmul(input, mx::transpose(weights[0])) + dg;
        auto up = mx::matmul(input, mx::transpose(weights[1])) + du;
        auto hidden = (gate * mx::sigmoid(gate)) * up;
        return mx::matmul(hidden, mx::transpose(weights[2])) + high_down(hidden);
    };
    runtime.stage(1, 97, weights);
    auto y = runtime.run(1, x, high_gpu, cancelled, &high);
    auto expected = high_gpu(x);
    assert(mx::all(mx::isfinite(y)).item<bool>());
    assert(mx::max(mx::abs(mx::astype(y, mx::float32) - mx::astype(expected, mx::float32)) /
        mx::abs(mx::astype(expected, mx::float32))).item<float>() < .025f);
    assert(runtime.metrics().runtime_weight_overflow_retries > 0 && !runtime.metrics().runtime_failed);
    // Poison only the second correction chunk: one successful chunk must
    // NOT be concatenated with stale hidden/down-LoRA after failure.
    high.gate_up = [](const Tensor &input) {
        auto gate = mx::concatenate({mx::full({1, 32, 96}, 16.f, mx::bfloat16),
            mx::full({1, input.shape(1) - 32, 96}, 100000.f, mx::bfloat16)}, 1);
        return std::make_pair(gate, mx::full({1, input.shape(1), 96}, 800.f, mx::bfloat16));
    };
    runtime.stage(2, 97, weights);
    y = runtime.run(2, x, high_gpu, cancelled, &high);
    assert(mx::all(y == expected).item<bool>());
    assert(runtime.metrics().runtime_failed && runtime.metrics().runtime_weight_fallback_blocks == 1);
    std::cout << "PASS runtime LoRA activation corrections: base/adapter/signed-adapter/base, "
                 "all projections, headroom+hidden restore, whole-tail fallback; worst relative L2=" << worst << '\n';
    std::cout << "PASS runtime base v2: validation-only hidden, exact copy accounting, consecutive base, "
                 "adapter/base isolation and hidden overflow recovery\n";
}

void correction_input_isolation_tests(const char *manifest) {
    using namespace tc;
    for (bool cpu_only : {true, false}) for (bool scalar : {false, true}) {
        ane::RuntimeGraph graph(manifest, 128ull << 20, cpu_only, scalar);
        ane::RuntimeGraph reference(manifest, 128ull << 20, cpu_only, scalar);
        const auto &shape = graph.shape();
        assert(shape.lora_inputs && shape.rows == 32 && shape.hidden == 64 && shape.width == 96);
        std::string error;
        assert(graph.self_test(error));
        auto view = [](const Tensor &tensor) {
            assert(tensor.dtype() == mx::bfloat16 && tensor.ndim() == 2);
            return ane::MatrixView{tensor.data<mx::bfloat16_t>(), tensor.nbytes(),
                tensor.shape(0), tensor.shape(1), 0, ane::DType::BF16};
        };
        std::vector<Tensor> weights{mx::full({96, 64}, .015625f, mx::bfloat16),
            mx::full({96, 64}, .03125f, mx::bfloat16), mx::full({64, 96}, .0078125f, mx::bfloat16)};
        auto input = mx::full({64, 64}, .125f, mx::bfloat16);
        auto stage = [&](ane::RuntimeGraph &target) {
            mx::eval(weights);
            std::vector<ane::MatrixView> sources;
            for (const auto &weight : weights) sources.push_back(view(weight));
            target.stage(std::move(sources));
            assert(target.wait_stage().ok);
        };
        std::vector<uint16_t> base;
        auto check_base = [&] {
            mx::eval(input);
            stage(graph);
            base.resize(input.size());
            graph.launch(view(input), base.data(), base.size(), ane::DType::BF16);
            auto result = graph.finish();
            assert(result.ok && result.calls >= uint64_t(input.shape(0) / shape.rows));
            // Re-running capability patterns forces the independent oracle
            // to clear nonzero dg/du before EVERY reference base prediction.
            assert(reference.self_test(error));
            stage(reference);
            std::vector<uint16_t> expected(input.size());
            reference.launch(view(input), expected.data(), expected.size(), ane::DType::BF16);
            assert(reference.finish().ok && base == expected);
            return result;
        };
        check_base();
        check_base();
        weights[0] = mx::full({96, 64}, .0625f, mx::bfloat16);
        input = mx::full({96, 64}, .25f, mx::bfloat16);
        check_base(); // new layer weights and three chunks
        for (float sign : {1.f, -1.f}) {
            auto dg = mx::full({96, 96}, sign * .5f, mx::bfloat16);
            auto du = mx::full({96, 96}, sign * .25f, mx::bfloat16);
            mx::eval(dg, du);
            std::vector<uint16_t> hidden(96 * 96), output(input.size());
            stage(graph);
            graph.launch(view(input), output.data(), output.size(), ane::DType::BF16,
                         ane::AdapterInput{view(dg), view(du), hidden.data(), hidden.size()});
            auto result = graph.finish();
            assert(result.ok && output != base);
            check_base();
            check_base();
        }
        for (bool poison_gate : {true, false}) {
            auto good = mx::full({96, 96}, .5f, mx::bfloat16);
            auto bad = mx::concatenate({mx::full({32, 96}, .5f, mx::bfloat16),
                mx::full({64, 96}, std::numeric_limits<float>::quiet_NaN(), mx::bfloat16)}, 0);
            mx::eval(good, bad);
            std::vector<uint16_t> hidden(96 * 96), output(input.size());
            graph.launch(view(input), output.data(), output.size(), ane::DType::BF16,
                         ane::AdapterInput{view(poison_gate ? bad : good),
                             view(poison_gate ? good : bad), hidden.data(), hidden.size()});
            auto result = graph.finish();
            assert(!result.ok && result.calls == 1); // reject partially computed tail
            check_base();
            check_base();
        }
        // Base conversion failure and capability checks must not leak stale
        // correction inputs into subsequent base requests.
        auto invalid = mx::full({96, 64}, std::numeric_limits<float>::infinity(), mx::bfloat16);
        mx::eval(invalid);
        graph.launch(view(invalid), base.data(), base.size(), ane::DType::BF16);
        assert(!graph.finish().ok);
        check_base();
        assert(graph.self_test(error));
        check_base();
        check_base();
        weights = {mx::full({96, 64}, 1.25f, mx::bfloat16),
            mx::full({96, 64}, 25.f, mx::bfloat16), mx::full({64, 96}, .001f, mx::bfloat16)};
        input = mx::ones({64, 64}, mx::bfloat16);
        auto overflow = check_base();
        assert(overflow.overflow_retries > 0 && overflow.headroom_scale > 1.f);
        assert(check_base().overflow_retries == 0);
    }
    std::cout << "PASS correction input isolation: CPU/NE, scalar/SIMD, layers/chunks, signed adapters, "
                 "partial failure/base recovery, repeated self-test and headroom retries\n";
}

void output_lifetime_tests(const char *manifest) {
    using namespace tc;
    std::atomic<bool> cancelled{false};
    std::vector<Tensor> saved, snapshots, consumers;
    {
        ane::HybridFfn runtime(manifest, 64, 96, 128ull << 20, cancelled);
        for (auto dtype : {mx::bfloat16, mx::float16, mx::float32}) {
            std::vector<Tensor> weights{mx::full({96, 64}, .01f, dtype),
                                        mx::full({96, 64}, .01f, dtype),
                                        mx::full({64, 96}, .01f, dtype)};
            mx::eval(weights);
            auto gpu = [&](const Tensor &x) {
                auto gate = mx::matmul(x, mx::transpose(weights[0]));
                return mx::matmul((gate * mx::sigmoid(gate)) *
                    mx::matmul(x, mx::transpose(weights[1])), mx::transpose(weights[2]));
            };
            for (int rows : {33, 97, 65}) {
                auto input = mx::full({1, rows, 64}, float(rows) / 128.f, dtype);
                runtime.stage(0, rows, weights);
                auto output = runtime.run(0, input, gpu, cancelled);
                assert(output.dtype() == dtype && output.shape() == input.shape());
                assert(mx::all(mx::isfinite(output)).item<bool>());
                assert(mx::max(mx::abs(mx::astype(output, mx::float32) -
                                      mx::astype(gpu(input), mx::float32))).item<float>() < .03f);
                saved.push_back(output);
                snapshots.push_back(mx::copy(output));
                mx::eval(snapshots.back());
                consumers.push_back(mx::astype(output, mx::float32) * .75f + 1.f);
                // Throw/cancel AFTER launch, while the worker may still
                // write its output buffers. Both paths must join before
                // unwinding, then allow a clean prediction on the same graph.
                runtime.stage(0, rows, weights);
                bool rejected = false;
                try {
                    runtime.run(0, input, [](const Tensor &) -> Tensor {
                        throw std::runtime_error("test GPU failure after launch");
                    }, cancelled);
                } catch (const std::runtime_error &error) {
                    rejected = std::string(error.what()) == "test GPU failure after launch";
                }
                assert(rejected);
                runtime.stage(0, rows, weights);
                rejected = false;
                try {
                    runtime.run(0, input, [&](const Tensor &x) {
                        cancelled = true;
                        return gpu(x);
                    }, cancelled);
                } catch (const Cancelled &) { rejected = true; }
                cancelled = false;
                assert(rejected);
                runtime.stage(0, rows, weights);
                assert(mx::all(runtime.run(0, input, gpu, cancelled) == output).item<bool>());
                assert(!runtime.metrics().runtime_failed);
            }
        }
    }
    // Consumers may outlive the runtime itself, not merely its next launch.
    for (size_t i = 0; i < saved.size(); ++i) {
        assert(mx::all(saved[i] == snapshots[i]).item<bool>());
        assert(mx::all(consumers[i] == mx::astype(snapshots[i], mx::float32) * .75f + 1.f).item<bool>());
    }
    std::cout << "PASS output lifetime: BF16/FP16/FP32, changing rows, lazy consumers, "
                 "runtime destruction and in-flight GPU failure/cancellation\n";
}

void async_head_tests(const char *base_manifest, const char *lora_manifest) {
    using namespace tc;
    {
        std::atomic<bool> cancelled{false};
        for (const auto &policy : {std::array<const char*,3>{"2","1","0"}, {"1","auto","0"},
                                   {"1","0","0"}, {"1","1","1"}}) {
            setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC",policy[0],1);
            setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS",policy[1],1);
            setenv("TURBOCIDER_RUNTIME_ANE_PROFILE",policy[2],1);
            bool rejected=false;
            try { ane::HybridFfn invalid(base_manifest,64,96,128ull<<20,cancelled); }
            catch (const std::invalid_argument&) { rejected=true; }
            assert(rejected);
        }
    }
    for (bool fixed : {false,true})
    for (bool profile : {false, true}) for (bool with_adapter : {false, true})
    for (auto dtype : {mx::bfloat16, mx::float16, mx::float32}) {
        if (fixed && profile) continue; // invalid combinations tested above
        setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", fixed ? "1" : "auto", 1);
        setenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC", fixed ? "1" : "0", 1);
        setenv("TURBOCIDER_RUNTIME_ANE_PROFILE", profile ? "1" : "0", 1);
        std::atomic<bool> cancelled{false};
        std::vector<Tensor> weights{mx::full({96, 64}, .01f, dtype),
                                    mx::full({96, 64}, .01f, dtype),
                                    mx::full({64, 96}, .01f, dtype)};
        mx::eval(weights);
        auto gpu = [&](const Tensor &x) {
            auto gate = mx::matmul(x, mx::transpose(weights[0]));
            auto up = mx::matmul(x, mx::transpose(weights[1]));
            if (with_adapter) { gate = gate + .0625f; up = up + .125f; }
            auto hidden = (gate * mx::sigmoid(gate)) * up;
            auto base = mx::matmul(hidden, mx::transpose(weights[2]));
            return with_adapter ? mx::astype(base + mx::sum(hidden, -1, true) * .015625f, dtype) : base;
        };
        std::optional<Tensor> saved, snapshot, retained_hidden, hidden_snapshot;
        bool bad_down = false, poison_gate = false;
        ane::HybridFfn::Adapter adapter{
            [&](const Tensor &x) {
                return std::make_pair(mx::full({1, x.shape(1), 96}, poison_gate ?
                    std::numeric_limits<float>::infinity() : .0625f, dtype),
                    mx::full({1, x.shape(1), 96}, .125f, dtype));
            },
            [&](const Tensor &hidden, const Tensor &base) {
                if (bad_down) throw std::runtime_error("test async down callback failure");
                retained_hidden = hidden;
                hidden_snapshot = mx::copy(hidden);
                return mx::astype(base + mx::sum(hidden, -1, true) * .015625f, base.dtype());
            }};
        ane::HybridFfn runtime(with_adapter ? lora_manifest : base_manifest,
                              64, 96, 128ull << 20, cancelled, with_adapter);
        // One legal chunk fixes geometry; synthetic whole-block times only
        // make the controller choose its steady plan, never claim a speedup.
        for (int visit = 1; visit <= 12; ++visit) {
            const auto plan = runtime.plan_block(0, 64);
            if (fixed) assert(plan.mode==ane::RowScheduler::Mode::HybridUntimed && plan.chunks==1 && !plan.measured());
            auto input = mx::full({1, 64, 64}, visit == 11 ? .25f : .5f, dtype);
            const auto before = runtime.metrics();
            if (plan.split()) {
                runtime.stage(0, 64, weights);
                if (visit == 9 || (visit == 10 && with_adapter)) {
                    bad_down = visit == 10;
                    bool caught = false;
                    try {
                        runtime.run(0, input, [&](const Tensor &x) {
                            if (visit == 9) cancelled = true;
                            return gpu(x);
                        }, cancelled, with_adapter ? &adapter : nullptr);
                    } catch (const Cancelled &) { caught = visit == 9; }
                    catch (const std::runtime_error &error) {
                        caught = bad_down && std::string(error.what()) == "test async down callback failure";
                    }
                    cancelled = false; bad_down = false;
                    assert(caught && runtime.metrics().runtime_weight_async_hybrid_blocks ==
                                     before.runtime_weight_async_hybrid_blocks);
                    continue; // next plan must work without a dangling sample
                }
                // An invalid ANE correction must still reconstruct the whole
                // tail with GPU while its independently submitted head lives.
                poison_gate = with_adapter && visit == 12;
                auto output = runtime.run(0, input, gpu, cancelled, with_adapter ? &adapter : nullptr);
                auto expected = gpu(input);
                assert(output.dtype() == dtype && mx::all(mx::isfinite(output)).item<bool>());
                assert(mx::max(mx::abs(mx::astype(output, mx::float32) -
                                      mx::astype(expected, mx::float32))).item<float>() < .03f);
                if (poison_gate) assert(mx::all(output == expected).item<bool>());
                const auto after = runtime.metrics();
                const bool asynchronous = !profile && (fixed || visit >= 7);
                assert(after.runtime_weight_async_hybrid_blocks - before.runtime_weight_async_hybrid_blocks ==
                       uint64_t(asynchronous));
                if (asynchronous) {
                    assert(after.runtime_weight_gpu_seconds == before.runtime_weight_gpu_seconds);
                    assert(after.runtime_weight_join_seconds == before.runtime_weight_join_seconds);
                    assert(after.runtime_weight_async_ane_wait_seconds > before.runtime_weight_async_ane_wait_seconds);
                } else assert(after.runtime_weight_async_ane_wait_seconds == before.runtime_weight_async_ane_wait_seconds);
                if (visit == 7) { saved = output; snapshot = mx::copy(output); mx::eval(*snapshot); }
                if (visit == 11) {
                    assert(mx::all(*saved == *snapshot).item<bool>());
                    assert(!mx::all(output == *snapshot).item<bool>());
                }
            } else mx::eval(gpu(input));
            if (plan.measured()) runtime.observe_block(0, 64, plan.split() ? .1 : 1.);
        }
        assert(runtime.metrics().runtime_failed == with_adapter);
        if (with_adapter) assert(runtime.metrics().runtime_weight_fallback_blocks == 1);
        if (retained_hidden) assert(mx::all(*retained_hidden == *hidden_snapshot).item<bool>());
        assert(mx::all(*saved == *snapshot).item<bool>());
    }
    unsetenv("TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC");
    setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", "2", 1);
    unsetenv("TURBOCIDER_RUNTIME_ANE_PROFILE");
    std::cout << "PASS async head join: base/LoRA, BF16/FP16/FP32, profile isolation, "
                 "owned output, cancellation/down failure drain and complete tail fallback\n";
    std::cout << "PASS fixed async row plan: policy validation, untimed ownership, base/LoRA, failure/cancellation drain\n";
}

void request_scheduler_tests(const char *manifest) {
    using namespace tc;
    using Mode = ane::RowScheduler::Mode;
    setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", "auto", 1);
    std::atomic<bool> cancelled{false};
    std::vector<Tensor> weights{mx::full({96, 64}, .01f, mx::bfloat16),
                                mx::full({96, 64}, .01f, mx::bfloat16),
                                mx::full({64, 96}, .01f, mx::bfloat16)};
    auto input = mx::full({1, 64, 64}, .5f, mx::bfloat16);
    mx::eval(weights);
    float correction = 0;
    auto gpu = [&](const Tensor &x) {
        auto gate = mx::matmul(x, mx::transpose(weights[0])) + correction;
        auto up = mx::matmul(x, mx::transpose(weights[1])) + correction;
        auto hidden = (gate * mx::sigmoid(gate)) * up;
        return mx::astype(mx::matmul(hidden, mx::transpose(weights[2])) +
                          mx::sum(hidden, -1, true) * correction, x.dtype());
    };
    ane::HybridFfn::Adapter adapter{
        [&](const Tensor &x) {
            auto delta = mx::full({1, x.shape(1), 96}, correction, x.dtype());
            return std::make_pair(delta, delta);
        },
        [&](const Tensor &hidden, const Tensor &base) {
            return mx::astype(base + mx::sum(hidden, -1, true) * correction, base.dtype());
        }};
    // One v2 executable serves base, A, same A, B, then base again.
    // One legal chunk isolates request warmup/reset from partition tuning.
    ane::HybridFfn runtime(manifest, 64, 96, 128ull << 20, cancelled, true);
    auto execute = [&](Mode expected_mode) {
        auto plan = runtime.plan_block(0, 64);
        assert(plan.mode == expected_mode);
        auto output = gpu(input);
        if (plan.split()) {
            runtime.stage(0, 64, weights);
            output = runtime.run(0, input, gpu, cancelled, correction ? &adapter : nullptr);
        }
        mx::eval(output);
        assert(mx::all(mx::isfinite(output)).item<bool>());
        assert(mx::max(mx::abs(mx::astype(output, mx::float32) -
                              mx::astype(gpu(input), mx::float32))).item<float>() < .03f);
        // Synthetic costs exercise state, NOT real performance measurements.
        if (plan.measured()) runtime.observe_block(0, 64, plan.split() ? 1. : .5);
        return output;
    };
    std::optional<Tensor> base_snapshot;
    std::string previous_identity;
    for (const std::string identity : {"", "adapter-A", "adapter-A", "adapter-B", ""}) {
        const bool same_adapter = !identity.empty() && identity == previous_identity;
        runtime.begin_request(identity);
        previous_identity = identity;
        correction = identity.empty() ? 0.f : identity == "adapter-A" ? .0625f : -.03125f;
        // Same adapter retains the disabled layer; changing identity resets
        // warmup. Base and adapter both retain hybrid-first scheduling.
        if (same_adapter) {
            execute(Mode::Gpu);
            continue;
        }
        for (int visit = 1; visit <= 4; ++visit) {
            const bool probe = visit >= 3;
            auto output = execute(probe ? Mode::GpuProbe : Mode::Hybrid);
            if (identity.empty() && visit == 1) {
                if (base_snapshot) assert(mx::all(output == *base_snapshot).item<bool>());
                else { base_snapshot = mx::copy(output); mx::eval(*base_snapshot); }
            }
        }
    }
    assert(!runtime.metrics().runtime_failed && runtime.metrics().runtime_calls == 8);
    assert(runtime.metrics().runtime_weight_full_gpu_probe_blocks == 8);
    setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", "2", 1);
    std::cout << "PASS request scheduler isolation: base/A/same-A/B/base on one v2 graph\n";
}

void quantized_tests(const char *manifest) {
    using namespace tc;
    auto matrix_view = [](const Tensor &a) {
        const auto dtype = a.dtype() == mx::float32 ? ane::DType::FP32 :
            a.dtype() == mx::bfloat16 ? ane::DType::BF16 : ane::DType::FP16;
        return ane::MatrixView{a.data<void>(), a.nbytes(), a.shape(0), a.shape(1), 0, dtype};
    };
    std::atomic<bool> cancelled{false};
    auto x = mx::astype(mx::random::normal({1, 97, 64}, mx::float32, mx::random::key(74)) * .5f,
                        mx::bfloat16);
    double worst = 0;
    for (int bits : {4, 8}) for (auto dtype : {mx::float16, mx::bfloat16, mx::float32})
    for (bool biased : {false, true}) {
        std::vector<ane::FfnWeight> weights;
        std::vector<Tensor> reference;
        std::vector<Tensor> gpu_offsets;
        for (int projection = 0; projection < 3; ++projection) {
            const int rows = projection == 2 ? 64 : 96, cols = projection == 2 ? 96 : 64;
            auto original = mx::astype(mx::random::normal({rows, cols}, mx::float32,
                mx::random::key(91 + projection)) / std::sqrt(float(cols)), dtype);
            auto quant = mx::quantize(original, 32, bits, "affine");
            mx::eval(quant);
            weights.push_back({quant[0], quant[1], biased ? std::optional<Tensor>(quant[2]) : std::nullopt,
                               32, bits});
            // MLX's affine API requires a bias array even for a mathematically
            // zero offset. The stager accepts omitted offsets without a new
            // tensor allocation; provide explicit zeros only to the oracle.
            gpu_offsets.push_back(biased ? quant[2] : mx::zeros_like(quant[1]));
            // Compare direct CPU slot conversion with MLX dequantize, which
            // independently unpacks the actual native uint32 representation.
            // Promote metadata first: MLX otherwise rounds affine decode in
            // the metadata dtype (e.g. BF16) even with a requested FP32 output.
            // The slot contract is FP32 affine arithmetic -> FP16, not BF16
            // rounding followed by a second FP16 rounding.
            auto dense = mx::dequantize(quant[0], mx::astype(quant[1], mx::float32),
                                       mx::astype(gpu_offsets.back(), mx::float32),
                                       32, bits, "affine", std::nullopt, mx::float32);
            mx::eval(dense);
            reference.push_back(dense);
            ane::AffineView source{quant[0].data<uint32_t>(), quant[0].nbytes(), rows, cols, 0,
                                   32, bits, matrix_view(quant[1]), std::nullopt};
            if (biased) source.offsets = matrix_view(quant[2]);
            ane::validate_affine_view(source);
            std::vector<uint16_t> converted(size_t(rows) * cols);
            for (int row = 0; row < rows; ++row)
                assert(ane::affine_fp16_row(source, row, converted.data() + size_t(row) * cols));
            Tensor decoded(reinterpret_cast<const mx::float16_t *>(converted.data()), {rows, cols}, mx::float16);
            const float error = mx::max(mx::abs(mx::astype(decoded, mx::float32) - dense)).item<float>();
            const float bound = .001f * mx::max(mx::abs(dense)).item<float>() + 1e-6f;
            if (error > bound) {
                const auto index = mx::argmax(mx::reshape(mx::abs(mx::astype(decoded, mx::float32) - dense), {-1})).item<uint32_t>();
                std::cerr << "affine decode mismatch bits=" << bits << " metadata=" << dtype
                          << " biased=" << biased << " projection=" << projection << " error=" << error
                          << " bound=" << bound << " index=" << index
                          << " cpu=" << float(std::bit_cast<_Float16>(converted[index]))
                          << " mlx=" << dense.data<float>()[index] << " output_dtype=" << dense.dtype() << '\n';
            }
            assert(error <= bound);
        }
        auto gpu = [&](const Tensor &input) {
            auto project = [&](const Tensor &value, size_t i) {
                const auto &w = weights[i];
                return mx::quantized_matmul(value, w.values, *w.scales, gpu_offsets[i], true, 32, bits, "affine");
            };
            auto gate = project(input, 0);
            return project((gate * mx::sigmoid(gate)) * project(input, 1), 2);
        };
        ane::HybridFfn runtime(manifest, 64, 96, 128ull << 20, cancelled);
        runtime.stage_weights(0, 97, weights);
        auto y = runtime.run(0, x, gpu, cancelled);
        auto expected = gpu(x);
        const float relative = mx::sqrt(mx::sum(mx::square(mx::astype(y, mx::float32) -
            mx::astype(expected, mx::float32))) / mx::sum(mx::square(mx::astype(expected, mx::float32)))).item<float>();
        assert(std::isfinite(relative) && relative < .03f);
        worst = std::max(worst, double(relative));
        assert(runtime.metrics().runtime_calls == 2 && !runtime.metrics().runtime_failed);
        // Same graph now receives a mixed dense/Q4 or Q8 layer. Drop all
        // caller-side packed owners after stage; bridge must keep them alive.
        weights[0] = {mx::astype(reference[0], mx::bfloat16), std::nullopt, std::nullopt};
        auto dense_gpu = [&](const Tensor &input) {
            auto value = mx::astype(input, mx::float32);
            auto gate = mx::matmul(value, mx::transpose(reference[0]));
            return mx::astype(mx::matmul((gate * mx::sigmoid(gate)) *
                mx::matmul(value, mx::transpose(reference[1])), mx::transpose(reference[2])), input.dtype());
        };
        runtime.stage_weights(1, 97, std::move(weights));
        weights.clear();
        gpu_offsets.clear();
        y = runtime.run(1, x, dense_gpu, cancelled);
        assert(mx::all(mx::isfinite(y)).item<bool>());
        assert(runtime.metrics().runtime_calls == 4 && !runtime.metrics().runtime_failed);
        expected = dense_gpu(x);
        const float mixed_error = mx::sqrt(mx::sum(mx::square(mx::astype(y, mx::float32) -
            mx::astype(expected, mx::float32))) / mx::sum(mx::square(mx::astype(expected, mx::float32)))).item<float>();
        assert(mixed_error < .03f);
        // Invalid source metadata must degrade to a COMPLETE GPU result.
        auto q = mx::quantize(mx::astype(reference[0], mx::bfloat16), 32, bits, "affine");
        std::vector<ane::FfnWeight> invalid{
            {q[0], mx::zeros({96, 1}, mx::bfloat16), q[2], 32, bits},
            {reference[1], std::nullopt, std::nullopt}, {reference[2], std::nullopt, std::nullopt}};
        runtime.stage_weights(2, 97, std::move(invalid));
        y = runtime.run(2, x, dense_gpu, cancelled);
        assert(mx::all(y == expected).item<bool>());
        assert(runtime.metrics().runtime_failed && runtime.metrics().runtime_calls == 4);
    }
    std::cout << "PASS Q4/Q8 staging vs MLX dequantize and packed GPU FFN; mixed sources, owned metadata, "
                 "invalid-source fallback; worst relative L2=" << worst << '\n';
}
}

int main(int argc, char **argv) {
    using namespace tc;
    assert(argc == 2 || argc == 3);
    std::atomic<bool> cancelled{false};
    std::vector<Tensor> weights{
        mx::full({96, 64}, .01f, mx::bfloat16),
        mx::full({96, 64}, .01f, mx::bfloat16),
        mx::full({64, 96}, .01f, mx::bfloat16)};
    mx::eval(weights);
    auto compiled = mx::compile([](const std::vector<Tensor> &a) {
        auto gate = mx::matmul(a[0], mx::transpose(a[1]));
        return std::vector<Tensor>{mx::matmul((gate * mx::sigmoid(gate)) *
            mx::matmul(a[0], mx::transpose(a[2])), mx::transpose(a[3]))};
    });
    auto gpu = [&](const Tensor &x) { return compiled({x, weights[0], weights[1], weights[2]})[0]; };
    auto x = mx::full({1, 97, 64}, .5f, mx::bfloat16);
    setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", "auto", 1);
    {
        ane::HybridFfn adaptive(argv[1], 64, 96, 128ull << 20, cancelled);
        // Only one chunk is legal: isolate on/off from the independently
        // timed row balancer, which may legitimately try a larger partition.
        constexpr int adaptive_rows = 64;
        auto adaptive_x = slice_axis(x, 1, 0, adaptive_rows);
        for (int visit = 1; visit <= 4; ++visit) {
            const auto plan = adaptive.plan_block(0, adaptive_rows);
            assert(plan.measured() && plan.split() == (visit <= 2));
            if (plan.split()) {
                adaptive.stage(0, adaptive_rows, weights);
                mx::eval(adaptive.run(0, adaptive_x, gpu, cancelled));
            } else {
                bool rejected = false;
                try { adaptive.stage(0, adaptive_rows, weights); }
                catch (const std::invalid_argument &) { rejected = true; }
                assert(rejected); // full GPU probe never borrows/stages weights
                mx::eval(gpu(adaptive_x));
            }
            bool wrong_sample = false;
            try { adaptive.observe_block(1, adaptive_rows, .1); }
            catch (const std::invalid_argument &) { wrong_sample = true; }
            assert(wrong_sample);
            // Synthetic timings deliberately make hybrid unprofitable. They
            // test the controller contract, NOT the micrograph's throughput.
            adaptive.observe_block(0, adaptive_rows, visit <= 2 ? 1.0 : .5);
            // Two hybrid warmups, two measured GPU probes; stage must not
            // consume a second plan and skip one of these controller states.
            assert(adaptive.metrics().runtime_calls == uint64_t(std::min(visit, 2)));
        }
        assert(adaptive.metrics().runtime_weight_full_gpu_probe_blocks == 2);
        assert(adaptive.metrics().runtime_weight_full_gpu_probe_seconds == 1.0);
        assert(adaptive.plan_block(0, adaptive_rows).mode == ane::RowScheduler::Mode::Gpu);
        assert(adaptive.metrics().runtime_weight_unsplit_gpu_blocks == 3);
        bool duplicate = false;
        try { adaptive.observe_block(0, adaptive_rows, .1); }
        catch (const std::invalid_argument &) { duplicate = true; }
        assert(duplicate);
    }
    {
        ane::HybridFfn adaptive(argv[1], 64, 96, 128ull << 20, cancelled);
        auto input = slice_axis(x, 1, 0, 64);
        std::optional<Tensor> saved_output, saved_snapshot, lazy_residual;
        for (int visit = 1; visit <= 8; ++visit) {
            if (visit == 8) input = mx::full({1, 64, 64}, .25f, mx::bfloat16);
            const auto plan = adaptive.plan_block(0, 64);
            if (plan.split()) {
                adaptive.stage(0, 64, weights);
                auto output = adaptive.run(0, input, gpu, cancelled);
                const float error = mx::max(mx::abs(mx::astype(output, mx::float32) -
                    mx::astype(gpu(input), mx::float32))).item<float>();
                assert(error < .03f);
                if (visit == 7) {
                    saved_output = output;
                    saved_snapshot = mx::copy(output);
                    mx::eval(*saved_snapshot);
                    // Model residuals may remain lazy across the next graph
                    // prediction. Construct now, evaluate only after slots
                    // have been reused for a different input.
                    lazy_residual = mx::astype(output, mx::float32) * .75f +
                                    mx::astype(input, mx::float32);
                } else if (visit == 8) {
                    assert(mx::all(*saved_output == *saved_snapshot).item<bool>());
                    assert(!mx::all(output == *saved_snapshot).item<bool>());
                    auto expected = mx::astype(*saved_snapshot, mx::float32) * .75f + .5f;
                    assert(mx::all(*lazy_residual == expected).item<bool>());
                }
            } else mx::eval(gpu(input));
            if (plan.measured()) adaptive.observe_block(0, 64, plan.split() ? .1 : 1.);
            if (visit >= 7) {
                assert(plan.mode == ane::RowScheduler::Mode::HybridUntimed);
                bool rejected = false;
                try { adaptive.observe_block(0, 64, .1); }
                catch (const std::invalid_argument &) { rejected = true; }
                assert(rejected); // run completed the plan; no stray full-block sample
            }
        }
        assert(adaptive.metrics().runtime_weight_hybrid_blocks == 6);
        assert(adaptive.metrics().runtime_weight_untimed_hybrid_blocks == 2);
        assert(adaptive.metrics().runtime_weight_gpu_blocks == 2);
        // A failed untimed stage must also release the plan after full GPU
        // fallback, so the next block cannot be stuck behind an observation.
        assert(adaptive.plan_block(0, 64).mode == ane::RowScheduler::Mode::HybridUntimed);
        adaptive.stage(0, 64, {weights[0]});
        assert(mx::all(adaptive.run(0, input, gpu, cancelled) == gpu(input)).item<bool>());
        assert(adaptive.plan_block(0, 64).mode == ane::RowScheduler::Mode::Gpu);
    }
    setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", "0", 1);
    {
        ane::HybridFfn ablation(argv[1], 64, 96, 128ull << 20, cancelled);
        assert(ablation.plan_block(0, 97).mode == ane::RowScheduler::Mode::SplitProbe);
        ablation.stage(0, 97, weights);
        assert(mx::all(ablation.run(0, x, gpu, cancelled) == gpu(x)).item<bool>());
        ablation.observe_block(0, 97, .1);
        assert(ablation.metrics().runtime_calls == 0);
        assert(ablation.metrics().runtime_weight_gpu_blocks == 1);
        assert(ablation.metrics().runtime_weight_full_gpu_probe_blocks == 0);
        assert(ablation.metrics().runtime_weight_unsplit_gpu_blocks == 0);
    }
    setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", "2", 1);
    ane::HybridFfn runtime(argv[1], 64, 96, 128ull << 20, cancelled);
    runtime.begin_request();
    // An early decision must be consumed once by stage, not advance the
    // scheduler a second time. Short rows bypass without a stage/prediction.
    assert(!runtime.plan_block(0, 32).measured());
    assert(runtime.metrics().runtime_weight_unsplit_gpu_blocks == 1);
    assert(runtime.metrics().runtime_calls == 0);
    assert(runtime.plan_block(0, 97).split());
    bool plan_mismatch = false;
    try { runtime.stage(1, 97, weights); }
    catch (const std::invalid_argument &) { plan_mismatch = true; }
    assert(plan_mismatch);
    runtime.stage(0, 97, weights);
    auto y = runtime.run(0, x, gpu, cancelled);
    runtime.observe_block(0, 97, .1);
    auto expected = gpu(x);
    const float relative = mx::sqrt(mx::sum(mx::square(mx::astype(y, mx::float32) -
        mx::astype(expected, mx::float32))) / mx::sum(mx::square(mx::astype(expected, mx::float32)))).item<float>();
    assert(relative < .03f);
    assert(runtime.metrics().runtime_calls == 2); // fixed two chunks in test env
    assert(runtime.metrics().runtime_weight_ane_rows == 64);
    // Values exceed FP16 input range, but the small weights keep the BF16
    // GPU FFN finite. The Core ML error must recompute the COMPLETE tail.
    x = mx::full({1, 97, 64}, 100000.f, mx::bfloat16);
    runtime.stage(1, 97, weights);
    y = runtime.run(1, x, gpu, cancelled);
    expected = gpu(x);
    assert(mx::all(mx::isfinite(y)).item<bool>());
    assert(mx::all(y == expected).item<bool>());
    auto metrics = runtime.metrics();
    assert(metrics.runtime_failed && metrics.runtime_failures == 1);
    assert(metrics.runtime_weight_fallback_blocks == 1);
    runtime.stage(2, 97, weights);
    y = runtime.run(2, x, gpu, cancelled);
    assert(mx::all(y == expected).item<bool>());
    assert(runtime.metrics().runtime_calls == metrics.runtime_calls);
    // Cancellation while staging is pending must be drained before buffers
    // and the graph are destroyed, and must not poison the next request.
    ane::HybridFfn cancelled_runtime(argv[1], 64, 96, 128ull << 20, cancelled);
    cancelled_runtime.stage(0, 97, weights);
    cancelled = true;
    bool rejected = false;
    try { cancelled_runtime.run(0, x, gpu, cancelled); }
    catch (const Cancelled &) { rejected = true; }
    assert(rejected);
    cancelled_runtime.drain();
    cancelled = false;
    cancelled_runtime.begin_request();
    x = mx::full({1, 97, 64}, .5f, mx::bfloat16);
    cancelled_runtime.stage(0, 97, weights);
    y = cancelled_runtime.run(0, x, gpu, cancelled);
    assert(mx::all(mx::isfinite(y)).item<bool>());
    // Memory admission failure is a real GPU fallback, not uncomputed output.
    ane::HybridFfn no_budget(argv[1], 64, 96, 1, cancelled);
    assert(!no_budget.available());
    no_budget.stage(0, 97, weights);
    y = no_budget.run(0, x, gpu, cancelled);
    expected = gpu(x);
    assert(mx::all(y == expected).item<bool>());
    assert(no_budget.metrics().runtime_calls == 0);
    // A resident request can lose its headroom after a successful first
    // request. Inject a pressure observation; no real memory is occupied.
    ane::HybridFfn pressured(argv[1], 64, 96, 128ull << 20, cancelled);
    assert(pressured.available());
    pressured.stage(0, 97, weights);
    y = pressured.run(0, x, gpu, cancelled);
    assert(pressured.metrics().runtime_calls == 2);
    const ane::MemoryObservation low{true, 64ull << 30, 24ull << 30,
                                      (4ull << 30) - 1, 18ull << 30};
    pressured.begin_request("next-adapter", low); // drains, releases graph + scratch
    assert(!pressured.available() && pressured.reason().find("memory admission") != std::string::npos);
    pressured.stage(0, 97, weights);
    y = pressured.run(0, x, gpu, cancelled);
    assert(mx::all(y == gpu(x)).item<bool>() && pressured.metrics().runtime_calls == 2);
    std::cout << "PASS resident memory pressure: graph release and complete GPU result\n";
    // Selecting a graph for the wrong model is a request error, not a memory
    // fallback. Verify model geometry even when no slot can be allocated.
    for (size_t budget : {size_t(1), size_t(128ull << 20)}) {
        bool wrong_geometry = false;
        try { ane::HybridFfn wrong(argv[1], 65, 96, budget, cancelled); }
        catch (const std::runtime_error &error) {
            wrong_geometry = std::string(error.what()).find("geometry") != std::string::npos;
        }
        assert(wrong_geometry);
    }
    no_budget.stage(0, 97, weights);
    bool wrong_input = false;
    try { no_budget.run(0, mx::zeros({1, 97, 65}, mx::bfloat16), gpu, cancelled); }
    catch (const std::invalid_argument &error) {
        wrong_input = std::string(error.what()).find("stage/run mismatch") != std::string::npos;
    }
    assert(wrong_input);
    // A GGUF FP32 residual must retain its dtype and restored exponent range.
    // The FFN result exceeds FP16, even though every input fits in FP16.
    // Headroom recovery followed by BF16 -> FP32 must not force GPU fallback.
    weights = {mx::full({96, 64}, .125f, mx::float32),
               mx::full({96, 64}, .125f, mx::float32),
               mx::full({64, 96}, .125f, mx::float32)};
    x = mx::full({1, 97, 64}, 16.f, mx::float32);
    ane::HybridFfn fp32_runtime(argv[1], 64, 96, 128ull << 20, cancelled);
    fp32_runtime.stage(0, 97, weights);
    y = fp32_runtime.run(0, x, gpu, cancelled);
    expected = gpu(x);
    assert(y.dtype() == mx::float32 && mx::all(mx::isfinite(y)).item<bool>());
    assert(mx::min(expected).item<float>() > 65504.f);
    assert(mx::max(mx::abs(y - expected) / mx::abs(expected)).item<float>() < .01f);
    assert(fp32_runtime.metrics().runtime_calls >= 2 && !fp32_runtime.metrics().runtime_failed);
    assert(fp32_runtime.metrics().runtime_weight_overflow_retries > 0);
    quantized_tests(argv[1]);
    output_lifetime_tests(argv[1]);
    if (argc == 3) {
        lora_alpha_tests(std::filesystem::path(argv[2]).parent_path());
        qwen_lora_correction_tests(std::filesystem::path(argv[2]).parent_path());
        lora_tests(argv[2]);
        correction_input_isolation_tests(argv[2]);
        async_head_tests(argv[1], argv[2]);
        request_scheduler_tests(argv[2]);
        // LoRA graph compatibility is checked before memory fallback.
        bool rejected = false;
        try { ane::HybridFfn unsupported(argv[1], 64, 96, 1, cancelled, true); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected);
    }
    std::cout << "PASS runtime-weight model wrapper: rows, GPU tail fallback, degradation, cancellation/drain\n";
}
