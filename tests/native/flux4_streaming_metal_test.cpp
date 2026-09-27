#include "models/flux2/flux_streaming.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 2, "transformer fixture required");
        std::atomic<bool> cancelled{false};
        tc::StreamingConfig config;
        config.enabled = true; config.schema_version = 1;
        config.selection = "manual"; config.retention = "request";
        config.stages["denoiser"] = {"streamed", 1, 2, 0, 1, 2};
        tc::Weights resident;
        uint32_t events = 0;
        tc::Event event = [&](const std::string &phase, int block, int total) {
            if (phase == "transformer_block") {
                tc::require(total == 25 && block == int(events % 25), "wrong 4B block order");
                ++events;
            }
        };
        tc::FluxExactStream stream(argv[1], "flux2-klein-4b", config,
                                  {16, 16, 1, 0, 2}, resident, event, cancelled, 1);
        const auto mod = tc::mx::ones({1, 1, 3072}, tc::mx::bfloat16);
        const std::vector<tc::Tensor> dual(6, mod), single(3, mod);
        const auto cosine = tc::mx::ones({2, 64}, tc::mx::float32);
        const auto sine = tc::mx::zeros({2, 64}, tc::mx::float32);
        for (uint32_t pass = 0; pass < 2; ++pass) {
            auto image = tc::mx::reshape(tc::mx::astype(tc::mx::arange(3072), tc::mx::bfloat16), {1, 1, 3072});
            auto context = tc::mx::full({1, 1, 3072}, float(pass + 2), tc::mx::bfloat16);
            const auto expected = tc::mx::concatenate({context, image}, 1);
            tc::mx::eval(expected);
            stream.run_pass(pass, pass, image, context, dual, dual, single, cosine, sine, 1, 2);
            tc::require(image.shape() == tc::mx::Shape({1, 2, 3072}) &&
                            tc::mx::all(image == expected).item<bool>(),
                        "zero-weight 4B stream changed its residual or class boundary");
        }
        stream.finish();
        const auto counters = stream.counters();
        tc::require(events == 50 && counters.fills == 50 && counters.groups_submitted == 50 &&
                        counters.pool_creates == 2 && counters.slot_bundles == 4,
                    "4B stream did not reuse two pools across both passes");
        std::cout << "PASS FLUX 4B Metal: actual 3072/24-head dual/single kernels, 50 blocks, residual oracle and retained pools (synthetic zero weights)\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
