// Research fixture preparation using the same tokenizer/Qwen path as ZImage.
#include "components/text/qwen3.hpp"
#include "runtime/streaming/source_lease.hpp"
#include <fstream>
#include <iostream>
using namespace tc;
namespace fs = std::filesystem;
int main(int argc, char **argv) {
    try {
        require(argc == 5, "expected text checkpoint, tokenizer, prompt text file, output directory");
        const fs::path output = argv[4];
        require(!fs::exists(output), "fixture output already exists");
        std::ifstream prompt_file(argv[3]);
        require(bool(prompt_file), "cannot open prompt");
        const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
        streaming::SourceFileIdentity text, tokenizer;
        text.logical_id = "text_encoder"; text.path = argv[1];
        tokenizer.logical_id = "tokenizer"; tokenizer.path = argv[2];
        std::atomic<bool> cancelled{false};
        auto lease = streaming::SourceLease::capture_verified({text, tokenizer}, &cancelled);
        auto fd = lease->duplicate_fd("tokenizer");
        Tokenizer parser(fd.get(), lease->file("tokenizer").bytes);
        auto tokens = parser.z_image_prompt(prompt, true);
        require(tokens.valid > 32 && tokens.valid <= 64 && tokens.ids.size() == size_t(tokens.valid),
                "prompt must occupy the frozen 64-row bucket with 33..64 valid tokens");
        mx::set_default_device(mx::Device(mx::Device::gpu, 0));
        Weights weights;
        const Event event = [](const std::string &phase, int done, int total) {
            if (phase == "text_encode") std::cout << phase << ' ' << done << '/' << total << std::endl;
        };
        weights.load_lease(lease, {"text_encoder"}, event, cancelled);
        auto encoded = components::qwen3_conditioning(tokens, weights,
            components::Qwen3Conditioning::z_image(), event, cancelled);
        auto caption = mx::astype(slice_axis(mx::squeeze(encoded, 0), 0, 0, tokens.valid), mx::float32);
        auto latent = mx::random::normal({16, 1, 64, 64}, mx::float32, 0.f, 1.f, mx::random::key(42));
        mx::eval({caption, latent});
        require(mx::all(mx::isfinite(caption)).item<bool>() && mx::all(mx::isfinite(latent)).item<bool>(), "nonfinite fixture");
        mx::synchronize();
        lease->revalidate_after_drain();
        weights.clear(); mx::clear_cache();
        fs::create_directories(output);
        mx::save((output / "caption.npy").string(), caption);
        mx::save((output / "initial.npy").string(), latent);
        std::ofstream report(output / "inputs.json");
        report << "{\"seed\":42,\"valid_tokens\":" << tokens.valid
               << ",\"text_compute_tokens\":" << tokens.ids.size()
               << ",\"source_artifact_digest\":\"" << lease->artifact_digest()
               << "\",\"caption_dtype\":\"float32 transport of native conditioning\"}\n";
        require(bool(report), "cannot write input report");
        std::cout << "prepared real prompt: " << tokens.valid << "/64 tokens\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
