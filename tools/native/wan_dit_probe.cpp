// Development parity probe. It links the production native components and
// accepts precomputed inputs only to isolate DiT arithmetic from tokenization.
#include "../../native/models/wan/dit.hpp"
#include "../../native/components/diffusion/wan.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 4, "usage: wan-dit-probe CHECKPOINT INPUT.safetensors OUTPUT.safetensors");
        auto inputs = tc::mx::load_safetensors(argv[2]).first;
        auto input = [&](const std::string &name) -> const tc::Tensor & { return inputs.at(name); };
        std::atomic<bool> cancelled{false};
        tc::Event event = [](const std::string &, int, int) {};
        tc::wan::Checkpoint checkpoint;
        checkpoint.load(std::filesystem::absolute(argv[1]), event, cancelled);
        tc::wan::DiT dit(checkpoint);
        std::unordered_map<std::string, tc::Tensor> outputs;
        auto save = [&](const std::string &key, const tc::Tensor &value) {
            tc::mx::eval(value);
            outputs.emplace(key, value);
        };
        auto x = dit.patch_embed(input("latent"));
        save("patch", x);
        auto condition = dit.condition(input("timestep"), input("text"));
        save("time", condition[0]);
        save("modulation", condition[1]);
        save("context", condition[2]);
        for (int index = 0; index < checkpoint.config().layers; ++index) {
            x = dit.block(x, condition[2], condition[1], input("cosine"), input("sine"), index);
            save("block_" + std::to_string(index), x);
        }
        auto predicted = dit.output(x, condition[0], input("latent").shape());
        save("output", predicted);
        save("compiled_output", dit.forward_compiled(input("latent"), input("text"), input("timestep"),
            input("cosine"), input("sine"), event, cancelled));
        auto sigmas = tc::components::wan_qad_sigmas({1000.f, 757.f, 522.f});
        save("sigmas", tc::Tensor(sigmas.data(), {3}, tc::mx::float32));
        for (int index = 0; index < 3; ++index) {
            std::optional<float> next;
            if (index < 2) next = sigmas[index + 1];
            save("dmd_" + std::to_string(index), tc::components::wan_dmd_step(
                input("latent"), predicted, sigmas[index], next, input("latent")));
        }
        if (inputs.count("renoise_0") && inputs.count("renoise_1")) {
            for (bool compiled : {false, true}) {
                auto latent = input("latent");
                const float timesteps[] = {1000.f, 757.f, 522.f};
                const std::string mode = compiled ? "compiled" : "eager";
                for (int index = 0; index < 3; ++index) {
                    auto time = tc::Tensor(&timesteps[index], {1}, tc::mx::float32);
                    auto prediction = compiled
                        ? dit.forward_compiled(latent, input("text"), time, input("cosine"), input("sine"), event, cancelled)
                        : dit.forward(latent, input("text"), time, input("cosine"), input("sine"), event, cancelled);
                    std::optional<float> next;
                    std::optional<tc::Tensor> noise;
                    if (index < 2) {
                        next = sigmas[index + 1];
                        noise = input("renoise_" + std::to_string(index));
                    }
                    latent = tc::mx::astype(tc::components::wan_dmd_step(
                        tc::mx::astype(latent, tc::mx::float32),
                        tc::mx::astype(prediction, tc::mx::float32), sigmas[index], next, noise), tc::mx::float16);
                    save(mode + "_rollout_" + std::to_string(index), latent);
                }
            }
        }
        tc::mx::save_safetensors(argv[3], outputs);
        std::cout << "Validated " << checkpoint.bytes() << " checkpoint bytes; wrote "
                  << outputs.size() << " native DiT stages\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
