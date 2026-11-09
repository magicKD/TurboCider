#pragma once
#include "tokenizer.hpp"

namespace tc {

enum class UnigramFlavor { umt5, t5_v1_1 };

// Native reader for UMT5's tokenizer.json Unigram/Metaspace contract. This is
// intentionally separate from Qwen BPE and does not load a Python tokenizer.
class UnigramTokenizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit UnigramTokenizer(const std::filesystem::path &root,
                              UnigramFlavor flavor = UnigramFlavor::umt5);
    ~UnigramTokenizer();
    // Mirrors the Wan reference: append EOS, truncate to limit, pad with 0.
    Tokens prompt(const std::string &, int limit = 512) const;
    // Standard T5 uses a shared precompiled Unicode map and first-only
    // Metaspace prefix. It pads to at least 128 and rejects overlong prompts.
    Tokens t5_prompt(const std::string &, int max_tokens = 512) const;
};

} // namespace tc
