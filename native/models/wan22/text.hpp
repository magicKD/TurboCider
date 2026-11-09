#pragma once

#include "../../backends/mlx.hpp"
#include "../../core/unigram_tokenizer.hpp"

namespace tc::wan22 {

// Single-file Comfy UMT5 checkpoint with a separately verified tokenizer.json.
class TextEncoder {
    std::filesystem::path checkpoint_, tokenizer_root_;

  public:
    TextEncoder(std::filesystem::path checkpoint, std::filesystem::path tokenizer_root);
    Tokens tokenize(const std::string &) const;
    Tensor encode(const Tokens &, const Event &, std::atomic<bool> &, int padding_alignment = 0) const;
};

} // namespace tc::wan22
