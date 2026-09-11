// Evaluation-only executable. Compile the unchanged production image pipeline
// with a conditioning replay seam; never link this object into the App.
#include "../../native/components/text/qwen3.hpp"
#include "../../native/runtime/execution.hpp"
#include <iostream>
#include <fstream>

namespace {
std::string replay_path;
}
namespace tc::components {
Tensor replay_conditioning(const Tokens &, const Weights &,
                           const Qwen3Conditioning &, const Event &, std::atomic<bool> &);
}
#define qwen3_conditioning replay_conditioning
#include "../../native/models/z_image/z_image.cpp"
#undef qwen3_conditioning

namespace tc::components {
Tensor replay_conditioning(const Tokens &tokens, const Weights &weights,
                           const Qwen3Conditioning &config, const Event &event,
                           std::atomic<bool> &cancelled) {
    if (replay_path.empty())
        return qwen3_conditioning(tokens, weights, config, event, cancelled);
    checkpoint(cancelled);
    auto data = mx::load_safetensors(replay_path).first;
    require(data.count("conditioning") && data.count("token_ids"), "missing replay tensors");
    auto value = data.at("conditioning");
    auto ids = data.at("token_ids");
    require(value.shape() == mx::Shape{1, int(tokens.ids.size()), 2560} &&
            value.dtype() == mx::bfloat16, "invalid conditioning geometry/dtype");
    require(ids.dtype() == mx::int32 && ids.size() == tokens.ids.size(), "invalid replay IDs");
    mx::eval(value, ids);
    for (size_t i = 0; i < tokens.ids.size(); ++i)
        require(ids.data<int>()[i] == tokens.ids[i], "replay prompt token mismatch");
    require(mx::all(mx::isfinite(value)).item<bool>(), "nonfinite conditioning");
    return value;
}
}

int main(int argc, char **argv) {
    try {
        tc::require(argc >= 4, "usage: replay tokens ROOT PROMPT | generate ROOT PROMPT SEED SIZE CONDITIONING|- OUTPUT");
        if (std::string(argv[1]) == "tokens") {
            tc::Tokenizer tokenizer(std::filesystem::path(argv[2]) / "tokenizer");
            auto t = tokenizer.z_image_prompt(argv[3], true);
            std::cout << "{\"ids\":[";
            for (size_t i=0; i<t.ids.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << t.ids[i];
            }
            std::cout << "],\"valid\":" << t.valid << "}\n";
            return 0;
        }
        tc::require(argc == 8 && std::string(argv[1]) == "generate", "invalid generation arguments");
        tc::DeviceLease lease;
        tc::configure_streams();
        tc::Request request;
        request.model="z-image-turbo";
        request.audio=false;
        request.prompt=argv[3];
        request.seed=std::stoull(argv[4]);
        request.width=request.height=std::stoi(argv[5]);
        request.steps=9;
        request.output=argv[7];
        tc::require(!std::filesystem::exists(request.output), "refuse to overwrite output");
        if (std::string(argv[6])!="-") replay_path=argv[6];
        tc::ZImage model(argv[2]);
        std::atomic<bool> cancelled{false};
        tc::mx::reset_peak_memory();
        auto result=model.generate(request, [](const std::string &,int,int){}, cancelled);
        std::cout << "{\"conditioning_replay\":" << (replay_path.empty()?"false":"true")
                  << ",\"image_stage_wall_seconds\":" << result.timings.wall
                  << ",\"denoise_seconds\":" << result.timings.denoise
                  << ",\"vae_seconds\":" << result.timings.decode
                  << ",\"mlx_image_stage_peak_bytes\":" << result.peak_bytes << "}\n";
        model.unload();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
