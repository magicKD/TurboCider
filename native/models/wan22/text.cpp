#include "text.hpp"
#include "../../components/text/umt5.hpp"

namespace tc::wan22 {

TextEncoder::TextEncoder(std::filesystem::path checkpoint, std::filesystem::path tokenizer_root)
    : checkpoint_(std::move(checkpoint)), tokenizer_root_(std::move(tokenizer_root)) {
    require(checkpoint_.is_absolute() && std::filesystem::is_regular_file(checkpoint_) &&
                tokenizer_root_.is_absolute() && std::filesystem::is_directory(tokenizer_root_),
            "Wan2.2 UMT5 requires absolute checkpoint and tokenizer paths");
}

Tokens TextEncoder::tokenize(const std::string &prompt) const {
    return UnigramTokenizer(tokenizer_root_).prompt_extended(prompt);
}

Tensor TextEncoder::encode(const Tokens &tokens, const Event &event, std::atomic<bool> &cancelled) const {
    checkpoint(cancelled);
    Weights weights;
    weights.load_file(checkpoint_);
    components::UMT5Config config;
    config.max_tokens = 1024;
    config.comfy_t5_sdpa = true;
    config.comfy_t5_fp32 = true;
    components::UMT5Encoder encoder(weights, config);
    auto output = mx::astype(encoder.encode(tokens, event, cancelled), mx::bfloat16);
    mx::eval(output);
    checkpoint(cancelled);
    return output;
}

} // namespace tc::wan22
