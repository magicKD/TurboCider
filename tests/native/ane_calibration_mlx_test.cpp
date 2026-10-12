#include "../../native/backends/ane_calibration_mlx.hpp"
#include "../../native/backends/ane_smoothquant_mlx.hpp"
#include <fstream>

namespace cal = tc::ane::calibration;
namespace mx = tc::mx;
namespace sq = tc::ane::smoothquant;

static void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class Function> static void rejected(Function function) {
    bool failed = false;
    try { function(); } catch (const std::exception &) { failed = true; }
    check(failed, "invalid bridge capture accepted");
}

int main(int argc, char **argv) {
    try {
        check(argc == 2 || argc == 3, "expected CPU fixture parent and optional S1 profile");
        // Do this before making any array. No configure_streams, Metal kernel,
        // Core ML, private ANE, checkpoint loader or actual model is exercised.
        mx::set_default_device(mx::Device(mx::Device::cpu));
        const auto root = std::filesystem::path(argv[1]);
        ::unsetenv("TURBOCIDER_ANE_CALIBRATION_DIR");
        ::unsetenv("TURBOCIDER_PRIVATE_ANE_S1_PROFILE");
        const auto missing = root / "does-not-exist.safetensors";
        tc::Request request;
        sq::validate_request(request); // Unset S1 is a zero-operation even with no source.
        ::setenv("TURBOCIDER_PRIVATE_ANE_S1_PROFILE", missing.c_str(), 1);
        rejected([&] { sq::validate_request(request); }); // Wrong route rejects before file loading.
        ::unsetenv("TURBOCIDER_PRIVATE_ANE_S1_PROFILE");
        {
            cal::ScopedCapture disabled(request, missing, "");
            check(!disabled.enabled() && !disabled.wants(0) && !cal::ScopedCapture::current(),
                  "default bridge capture was enabled");
            disabled.finish();
        }
        check(!std::filesystem::exists(missing), "disabled capture accessed/created source");
        const auto checkpoint = root / "fake-checkpoint.safetensors";
        if (argc == 2) std::ofstream(checkpoint) << "CPU stat identity fixture, not a checkpoint";
        const auto capture_parent = root / "captures";
        ::setenv("TURBOCIDER_ANE_CALIBRATION_DIR", capture_parent.c_str(), 1);
        ::setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", "1", 1);
        ::setenv("TURBOCIDER_RUNTIME_ANE_PROFILE", "0", 1);
        request.model = "qwen-image-2.1"; request.hybrid_mlp_mode = "runtime";
        request.execution = "gpu_ane";
        request.width = request.height = 512; request.steps = 6; request.seed = 42;
        tc::InputAsset reference;
        reference.kind = "image"; reference.role = "reference"; reference.path = "fake-reference-never-read";
        request.inputs.push_back(reference);
        request.qwen21_reference_size = 512;
        request.loras.push_back({"fake-lora-never-read.safetensors", .7f, "transformer"});
        if (argc == 3) {
            const auto expected = sq::request_binding(request,checkpoint,"existing-fake-adapter-sha-and-strength",8);
            const auto profile = sq::ExperimentalProfile::load(argv[2],expected);
            check(profile.provenance().model_fingerprint == sq::checkpoint_fingerprint(checkpoint) &&
                  profile.provenance().loras[0].fingerprint == expected.loras[0].fingerprint &&
                  profile.provenance().runtime_recipe == expected.runtime_recipe &&
                  profile.provenance().configured_backend == expected.configured_backend,
                  "capture and S1 request binding identity disagree");
            const auto tensors = sq::scale_tensors(profile,8);
            check(tensors.size() == 32, "S1 scale tensor layer count mismatch");
            for (int layer = 0; layer < 32; ++layer) {
                const auto &tensor = tensors[size_t(layer)];
                check(tensor.shape() == mx::Shape({1,8}) && tensor.dtype() == mx::float32,
                      "S1 CPU scale tensor shape/dtype mismatch");
                mx::eval(tensor);
                const auto expected_scales = profile.scales_for(layer);
                for (int channel = 0; channel < 8; ++channel)
                    check(tensor.data<float>()[channel] == expected_scales[size_t(channel)],
                          "S1 CPU scale tensor content mismatch");
            }
            ::unsetenv("TURBOCIDER_ANE_CALIBRATION_DIR");
            std::cout << "PASS CPU S1 binding/scales no model/no hardware\n";
            return 0;
        }

        std::vector<float> weight_values(257 * 8, 2.f); weight_values[256 * 8 + 3] = -4.f;
        const tc::Tensor fused(weight_values.data(), {257, 8}, mx::float32);
        tc::Weights weights;
        weights.bind_arrays({"fixture.gate_up.weight"}, {fused});
        auto make_input = [](int rows) {
            std::vector<float> values(size_t(rows) * 8, 1.f);
            values[97 * 8 + 3] = 16.f;
            auto input = tc::Tensor(values.data(), {1, rows, 8}, mx::float32);
            mx::eval(input); return input;
        };
        auto prefill = make_input(272), decode = make_input(129), z_input = make_input(1056);
        {
            cal::ScopedCapture outer(request, checkpoint, "existing-fake-adapter-sha-and-strength");
            check(outer.enabled() && cal::ScopedCapture::current() == &outer, "enabled bridge not installed");
            // A failed constructor cannot replace the owner's TLS context.
            rejected([&] { cal::ScopedCapture invalid(request, missing, ""); });
            check(cal::ScopedCapture::current() == &outer, "failed constructor lost outer TLS context");
            for (const char *chunks : {static_cast<const char *>(nullptr), "auto", "2"}) {
                if (chunks) ::setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", chunks, 1);
                else ::unsetenv("TURBOCIDER_RUNTIME_ANE_CHUNKS");
                rejected([&] { cal::ScopedCapture invalid(request, checkpoint, ""); });
                check(cal::ScopedCapture::current() == &outer, "invalid chunks lost outer TLS context");
            }
            ::setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", "0", 1);
            {
                cal::ScopedCapture gpu_boundary(request, checkpoint, "existing-fake-adapter-sha-and-strength");
                check(gpu_boundary.enabled() && cal::ScopedCapture::current() == &gpu_boundary,
                      "forced GPU runtime boundary capture was rejected");
            }
            check(cal::ScopedCapture::current() == &outer, "forced GPU boundary scope lost TLS context");
            ::setenv("TURBOCIDER_RUNTIME_ANE_CHUNKS", "1", 1);
            for (const char *profile : {static_cast<const char *>(nullptr), "1"}) {
                if (profile) ::setenv("TURBOCIDER_RUNTIME_ANE_PROFILE", profile, 1);
                else ::unsetenv("TURBOCIDER_RUNTIME_ANE_PROFILE");
                rejected([&] { cal::ScopedCapture invalid(request, checkpoint, ""); });
                check(cal::ScopedCapture::current() == &outer, "invalid profile lost outer TLS context");
            }
            ::setenv("TURBOCIDER_RUNTIME_ANE_PROFILE", "0", 1);
            rejected([&] {
                cal::ScopedCapture failed(request, checkpoint, "existing-fake-adapter-sha-and-strength");
                check(cal::ScopedCapture::current() == &failed, "nested scope not installed");
                failed.observe(0, prefill, weights, {"fixture.gate_up"},
                               {0, 0, "prefill", {{"target", 0, 271}}});
            });
            check(cal::ScopedCapture::current() == &outer, "failed observation lost outer TLS context");
            for (const int step : {0, 3, 5}) {
                outer.step(step);
                for (int layer = 0; layer < 32; ++layer) {
                    const auto point = step == 0 ? cal::Point{layer, step, "prefill",
                        {{"text-0", 0, 16}, {"reference-0", 16, 144}, {"target", 144, 272}}} :
                        cal::Point{layer, step, "denoise", {{"target", 0, 129}}};
                    check(outer.wants(layer), "selected point absent");
                    outer.observe(layer, step == 0 ? prefill : decode, weights, {"fixture.gate_up"}, point);
                    check(!outer.wants(layer), "captured point requested twice");
                }
            }
            outer.finish(); outer.finish();
        }
        check(!cal::ScopedCapture::current(), "finished request leaked TLS capture");
        request.model = "z-image-turbo"; request.inputs.clear(); request.loras.clear();
        tc::Weights z_weights;
        z_weights.bind_arrays({"fixture.w1.weight", "fixture.w3.weight"}, {fused, fused * tc::Tensor(.5f)});
        {
            cal::ScopedCapture z(request, checkpoint, "");
            for (const int step : {0, 3, 5}) {
                z.step(step);
                for (int layer = 0; layer < 32; ++layer)
                    z.observe(layer, z_input, z_weights, {"fixture.w1", "fixture.w3"},
                              {layer, step, "denoise", {{"target", 0, 1024}, {"text", 1024, 1056}}});
            }
            z.finish();
        }
        check(!cal::ScopedCapture::current(), "second request leaked TLS capture");
        ::unsetenv("TURBOCIDER_ANE_CALIBRATION_DIR");
        std::cout << "PASS CPU MLX calibration default-off/fake-source/regions/32x3/stats/TLS-rollback\n";
    } catch (const std::exception &error) {
        ::unsetenv("TURBOCIDER_ANE_CALIBRATION_DIR");
        std::cerr << error.what() << '\n'; return 1;
    }
}
