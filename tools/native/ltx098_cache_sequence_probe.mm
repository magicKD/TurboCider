#import <Foundation/Foundation.h>
#include "../../native/core/unigram_tokenizer.hpp"
#include "../../native/media/image.hpp"
#include "../../native/models/ltx098/pipeline.hpp"
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            tc::require(argc == 3, "usage: ltx098-cache-sequence-probe ROOT IMAGE");
            const auto root = std::filesystem::absolute(argv[1]);
            tc::UnigramTokenizer tokenizer(root / "tokenizer", tc::UnigramFlavor::t5_v1_1);
            const auto a = tokenizer.t5_prompt("A lighthouse above the sea, gentle waves and drifting clouds.");
            const auto b = tokenizer.t5_prompt("A yellow tram crosses the rainy city street at night.");
            tc::ltx098::Pipeline pipeline(root);
            auto image = tc::load_image_center_crop_tensor(argv[2], 64, 64);
            tc::mx::eval(image);
            std::atomic<bool> cancelled{false};
            std::optional<tc::Tensor> baseline, other, current;
            const std::vector<const tc::Tokens *> sequence{&a, &a, &b, &a};
            const std::vector<bool> expected_hit{false, true, false, false};
            for (size_t i = 0; i < sequence.size(); ++i) {
                tc::ltx098::GenerateOptions options;
                options.width = options.height = 64; options.frames = 9; options.fps = 24; options.seed = 42;
                current.reset();
                options.trace = [&](const std::string &name, const tc::Tensor &value) {
                    if (name == "step_7") { tc::mx::eval(value); current = value; }
                };
                auto [pixels, metrics] = pipeline.generate(*sequence[i], image, options,
                    [](const std::string &, int, int) {}, cancelled);
                tc::require(current.has_value() && metrics.text_cache_hit == expected_hit[i] &&
                            metrics.cross_attention_cache_hit == expected_hit[i] && metrics.denoise_steps == 8,
                            "prompt and cross-attention cache hit sequence differs");
                if (i == 0) baseline = *current;
                else if (i == 2) other = *current;
                else tc::require(tc::mx::all(*baseline == *current).item<bool>(),
                                 "same prompt changed after a hit or invalidation");
            }
            tc::require(!tc::mx::all(*baseline == *other).item<bool>(), "changed prompt did not change the latent");
            std::cout << "PASS: prompt A,A,B,A cache hit/invalidation and exact repeat outputs\n";
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
