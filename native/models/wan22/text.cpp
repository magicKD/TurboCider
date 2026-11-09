#include "text.hpp"
#include "../../components/text/umt5.hpp"

#include <algorithm>

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

Tensor TextEncoder::encode(const Tokens &tokens, const Event &event, std::atomic<bool> &cancelled,
                           int padding_alignment) const {
    checkpoint(cancelled);
    require(!tokens.ids.empty() && tokens.ids.size() <= 1024 && tokens.valid > 0 &&
                size_t(tokens.valid) <= tokens.ids.size(), "invalid Wan2.2 token mask");
    require(padding_alignment == 0 || padding_alignment == 64 || padding_alignment == 128 || padding_alignment == 256,
            "invalid Wan2.2 experimental text alignment");
    auto encoded_tokens = tokens;
    if (padding_alignment) {
        const auto aligned = size_t((tokens.valid + padding_alignment - 1) / padding_alignment) * padding_alignment;
        encoded_tokens.ids.resize(std::min(tokens.ids.size(), aligned));
    }
    Weights weights;
    weights.load_file(checkpoint_);
    components::UMT5Config config;
    config.max_tokens = 1024;
    config.comfy_t5_sdpa = true;
    config.comfy_t5_fp32 = true;
    components::UMT5Encoder encoder(weights, config);
    auto output = mx::astype(encoder.encode(encoded_tokens, event, cancelled), mx::bfloat16);
    if (encoded_tokens.ids.size() < tokens.ids.size()) {
        auto padding = mx::zeros({1, int(tokens.ids.size() - encoded_tokens.ids.size()), int(output.shape(2))},
                                 mx::bfloat16);
        output = mx::concatenate({output, padding}, 1);
    }
    mx::eval(output);
    checkpoint(cancelled);
    return output;
}

} // namespace tc::wan22
