#include "../../native/models/qwen21/pe_generation.hpp"
#include "../../native/models/qwen21/vision.hpp"
#include <cassert>
#include <iostream>
#include <limits>

int main(int argc, char **argv) {
    using namespace tc::qwen21::pe;
    tc::mx::set_default_device(tc::mx::Device::cpu);
    TextGenerationResult result;
    assert(!result.complete());
    result.rewrite.parse_ok = true;
    assert(!result.complete()); // parseable but truncated
    result.stopped_eos = true;
    assert(result.complete());
    result.rewrite.parse_ok = false;
    assert(!result.complete()); // EOS without a valid answer
    if (argc != 2) {
        std::cout << "PASS PE completion contract; SKIP input checks (tokenizer directory required)\n";
        return 0;
    }
    tc::Tokenizer tokenizer(argv[1]);
    tc::Weights weights; // Deliberately empty: all failures must precede inference.
    for (bool pe : {false, true}) {
        tc::qwen21::VisionConfig config;
        config.pe_qwen35 = pe;
        config.pe_fp32 = true;
        config.pe_compensated_projection = pe;
        config.deepstack_layers.clear();
        bool rejected = false;
        try { tc::qwen21::VisionEncoder encoder(weights, config); }
        catch (const std::exception &error) {
            rejected = std::string(error.what()).find("FP32 vision") != std::string::npos;
        }
        assert(rejected);
    }
    std::atomic<bool> cancelled{false};
    auto rejects = [&](const SamplingProfile &profile, const std::string &system,
                       const std::string &prompt, const std::string &expected) {
        try {
            generate_text(weights, tokenizer, system, prompt, profile, 42, {}, cancelled);
        } catch (const std::exception &error) {
            assert(std::string(error.what()).find(expected) != std::string::npos);
            return;
        }
        assert(false);
    };
    SamplingProfile profile;
    cancelled = true;
    rejects(profile, "system", "prompt", "generation cancelled");
    cancelled = false;
    for (int budget : {0, -1, 16257, 24000}) {
        profile.max_new_tokens = budget;
        rejects(profile, "system", "prompt", "token budget");
    }
    profile = {};
    profile.temperature = std::numeric_limits<double>::quiet_NaN();
    rejects(profile, "system", "prompt", "sampling profile");
    profile = {};
    rejects(profile, "", "prompt", "empty");
    rejects(profile, "system", " \n", "empty");
    auto edit_rejects = [&](const SamplingProfile &sampling, const std::vector<ImageGrid> &grids,
                            const std::vector<tc::Tensor> &features, const std::string &expected) {
        try {
            generate_edit_features(weights, tokenizer, "system", "edit", grids, features,
                sampling, 42, {}, cancelled);
        } catch (const std::exception &error) {
            assert(std::string(error.what()).find(expected) != std::string::npos);
            return;
        }
        assert(false);
    };
    auto edit = SamplingProfile::for_edit();
    cancelled = true;
    edit_rejects(edit, {}, {}, "generation cancelled");
    cancelled = false;
    for (int budget : {0, -1, 24001}) {
        auto invalid = edit;
        invalid.max_new_tokens = budget;
        edit_rejects(invalid, {}, {}, "token budget");
    }
    auto invalid = edit;
    invalid.temperature = std::numeric_limits<double>::quiet_NaN();
    edit_rejects(invalid, {}, {}, "sampling profile");
    edit_rejects(edit, {}, {}, "matching grids/features");
    edit_rejects(edit, {{2,2}}, {}, "matching grids/features");
    auto small = tc::mx::zeros({1, 1});
    edit_rejects(edit, std::vector<ImageGrid>(11, {2,2}), std::vector<tc::Tensor>(11, small), "matching grids/features");
    edit_rejects(edit, {{3,2}}, {small}, "image grid");
    edit_rejects(edit, {{256,256}}, {small}, "token limit");
    edit_rejects(edit, {{2,2}}, {small}, "feature/grid shape");
    edit_rejects(edit, {{2,2}}, {tc::mx::zeros({1,4096}, tc::mx::int32)}, "floating point");
    edit_rejects(edit, {{2,2}}, {tc::mx::full({1,4096}, std::numeric_limits<float>::infinity())}, "nonfinite");
    auto image_rejects = [&](const std::vector<tc::Tensor> &images, const SamplingProfile &sampling,
                             const std::string &expected) {
        for (bool fp32_visual : {false, true}) {
            bool rejected = false;
            try {
                generate_edit_images(weights, tokenizer, "system", "edit", images,
                    sampling, 42, {}, cancelled, {}, fp32_visual);
            } catch (const std::exception &error) {
                assert(std::string(error.what()).find(expected) != std::string::npos);
                rejected = true;
            }
            assert(rejected);
        }
    };
    cancelled = true;
    image_rejects({}, edit, "generation cancelled");
    cancelled = false;
    image_rejects({}, edit, "1...10 images");
    image_rejects(std::vector<tc::Tensor>(11, small), edit, "1...10 images");
    invalid = edit; invalid.max_new_tokens = 0;
    image_rejects({small}, invalid, "token budget");
    invalid = edit; invalid.max_new_tokens = 24001;
    image_rejects({small}, invalid, "token budget");
    invalid = edit; invalid.temperature = std::numeric_limits<double>::quiet_NaN();
    image_rejects({small}, invalid, "sampling profile");
    image_rejects({small}, edit, "uint8 NHWC RGB/RGBA");
    image_rejects({tc::mx::zeros({1,32,32,4}, tc::mx::float32)}, edit, "uint8 NHWC RGB/RGBA");
    image_rejects({tc::mx::zeros({1,32,32,2}, tc::mx::uint8)}, edit, "uint8 NHWC RGB/RGBA");
    std::cout << "PASS PE completion, cancellation and pre-inference input validation\n";
    std::cout << "PASS PE-I2I feature API cancellation, budgets, grids, context, shape, dtype and finite validation\n";
    std::cout << "PASS PE-I2I image API cancellation, image count, sampling and pixel validation\n";
}
