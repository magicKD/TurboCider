#pragma once
#include "tokenizer.hpp"

namespace tc {

// Native reader for UMT5's tokenizer.json Unigram/Metaspace contract. This is
// intentionally separate from Qwen BPE and does not load a Python tokenizer.
class UnigramTokenizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit UnigramTokenizer(const std::filesystem::path &root);
    ~UnigramTokenizer();
    // Mirrors the Wan reference: append EOS, truncate to limit, pad with 0.
    Tokens prompt(const std::string &, int limit = 512) const;
};

} // namespace tc
