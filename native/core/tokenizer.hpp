#pragma once
#include "common.hpp"
namespace tc {
struct Tokens {
    std::vector<int> ids;
    int valid = 0;
};
// Unicode normalization and regex remain in the Apple adapter.
class Tokenizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit Tokenizer(const std::filesystem::path &);
    ~Tokenizer();
    Tokens raw(const std::string &) const;
    // Explicit longer PE context; existing diffusion token limits stay intact.
    Tokens raw_bounded(const std::string &, int max_tokens) const;
    static constexpr int qwen21_limit = 2048;
    // Decode token IDs using the tokenizer's byte-level vocabulary. This is
    // used by native PE sampling; generation never shells out to Python.
    // Returns raw UTF-8 bytes: a single token may split a UTF-8 character.
    std::string decode(const std::vector<int> &) const;
    Tokens prompt(const std::string &, bool dynamic = true);
    static constexpr int z_image_limit = 1024;
    Tokens z_image_tokens(const std::string &);
    Tokens z_image_prompt(const std::string &, bool dynamic = true);
    Tokens llada_image_prompt(const std::string &);
};
} // namespace tc
