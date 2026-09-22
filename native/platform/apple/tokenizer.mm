#include "bridge.hpp"
#include <limits>
#include <map>
#include <unordered_map>
#include <array>
namespace tc {
struct Tokenizer::Impl {
    NSDictionary *vocab_;
    NSRegularExpression *pattern_;
    std::unordered_map<std::string, int> ranks_;
    std::vector<std::string> byte_encoder_;
    std::unordered_map<int, std::string> token_decoder_;
    std::map<std::string, int> special_;
    std::vector<int> encode(const std::string &);
    explicit Impl(const std::filesystem::path &);
};
Tokenizer::Impl::Impl(const std::filesystem::path &root) {
    auto config = read_json(root / "tokenizer.json");
    require([config[@"model"][@"type"] isEqual:@"BPE"], "tokenizer must be BPE");
    vocab_ = config[@"model"][@"vocab"];
    NSArray *merges = config[@"model"][@"merges"];
    int rank = 0;
    for (id pair in merges) {
        std::string a, b;
        if ([pair isKindOfClass:NSArray.class]) {
            require([pair count] == 2, "unsupported BPE merges");
            a = [pair[0] UTF8String];
            b = [pair[1] UTF8String];
        } else if ([pair isKindOfClass:NSString.class]) {
            // Tokenizers >= 0.20 serialise Qwen3 BPE merges as a single
            // space-separated string. Byte-level symbols encode spaces as Ġ,
            // so the first ASCII space is an unambiguous separator.
            NSString *line = (NSString *)pair;
            NSRange separator = [line rangeOfString:@" "];
            require(separator.location != NSNotFound && separator.location > 0 &&
                        separator.location + 1 < line.length,
                    "unsupported BPE merge string");
            a = [[line substringToIndex:separator.location] UTF8String];
            b = [[line substringFromIndex:separator.location + 1] UTF8String];
        } else {
            require(false, "unsupported BPE merges");
        }
        ranks_[a + '\0' + b] = rank++;
    }
    NSString *regex = config[@"pre_tokenizer"][@"pretokenizers"][0][@"pattern"][@"Regex"];
    require([regex isKindOfClass:NSString.class], "missing tokenizer split regex");
    NSError *e = nil;
    pattern_ = [NSRegularExpression regularExpressionWithPattern:regex options:0 error:&e];
    require(pattern_ != nil, "unsupported tokenizer regex");
    for (NSDictionary *t in config[@"added_tokens"])
        special_ [[t [@"content"] UTF8String]] = [t[@"id"] intValue];
    byte_encoder_.resize(256);
    std::array<int, 512> byte_decoder;
    byte_decoder.fill(-1);
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        int cp = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) ? b : 256 + extra++;
        unichar c = cp;
        byte_encoder_[b] = [[NSString stringWithCharacters:&c length:1] UTF8String];
        byte_decoder[cp] = b;
    }
    // Added tokens are literal text and can be outside the base vocabulary.
    for (const auto &[text, id] : special_) token_decoder_[id] = text;
    for (NSString *key in vocab_) {
        int id = [vocab_[key] intValue];
        require(id >= 0, "negative tokenizer vocabulary ID");
        if (token_decoder_.count(id)) continue;
        std::string decoded;
        for (NSUInteger i = 0; i < key.length; ++i) {
            unichar cp = [key characterAtIndex:i];
            require(cp < byte_decoder.size() && byte_decoder[cp] >= 0,
                    "unsupported non-byte-level tokenizer vocabulary");
            decoded.push_back(char(byte_decoder[cp]));
        }
        token_decoder_.emplace(id, std::move(decoded));
    }
}
std::vector<int> Tokenizer::Impl::encode(const std::string &raw) {
    NSString *s = [[NSString alloc] initWithBytes:raw.data() length:raw.size() encoding:NSUTF8StringEncoding];
    require(s != nil, "tokenizer input is not UTF-8");
    s = [s precomposedStringWithCanonicalMapping];
    NSData *normalized = [s dataUsingEncoding:NSUTF8StringEncoding];
    std::string text(static_cast<const char *>(normalized.bytes), normalized.length);
    std::vector<int> result;
    auto ordinary = [&](const std::string &sub) {
        NSString *part = [[NSString alloc] initWithBytes:sub.data() length:sub.size() encoding:NSUTF8StringEncoding];
        for (NSTextCheckingResult *m in [pattern_ matchesInString:part
                                                          options:0
                                                            range:NSMakeRange(0, part.length)]) {
            NSData *bytes =
                [[part substringWithRange:m.range] dataUsingEncoding:NSUTF8StringEncoding];
            std::vector<std::string> symbols;
            for (NSUInteger j = 0; j < bytes.length; ++j)
                symbols.push_back(byte_encoder_[((const uint8_t *)bytes.bytes)[j]]);
            while (symbols.size() > 1) {
                int best = std::numeric_limits<int>::max();
                size_t at = 0;
                for (size_t j = 0; j + 1 < symbols.size(); ++j) {
                    auto it = ranks_.find(symbols[j] + '\0' + symbols[j + 1]);
                    if (it != ranks_.end() && it->second < best) {
                        best = it->second;
                        at = j;
                    }
                }
                if (best == std::numeric_limits<int>::max())
                    break;
                symbols[at] += symbols[at + 1];
                symbols.erase(symbols.begin() + at + 1);
            }
            for (auto &symbol : symbols) {
                NSNumber *n = vocab_[@(symbol.c_str())];
                require(n != nil, "BPE symbol missing from vocabulary");
                result.push_back(n.intValue);
            }
        }
    };
    size_t start = 0;
    while (start < text.size()) {
        size_t next = text.size();
        std::string found;
        int id = 0;
        for (auto &[special, value] : special_) {
            auto at = text.find(special, start);
            if (at < next || (at == next && special.size() > found.size())) {
                next = at;
                found = special;
                id = value;
            }
        }
        ordinary(text.substr(start, next - start));
        if (found.empty())
            break;
        result.push_back(id);
        start = next + found.size();
    }
    return result;
}
Tokenizer::Tokenizer(const std::filesystem::path &root) : impl_(std::make_unique<Impl>(root)) {}
Tokenizer::~Tokenizer() = default;
Tokens Tokenizer::raw(const std::string &s) const {
    require(!s.empty(), "prompt must not be empty");
    require(s.size() <= 32768, "prompt exceeds 32 KiB");
    auto ids = impl_->encode(s);
    require(!ids.empty() && ids.size() <= 512,
            "raw prompt must contain 1...512 tokens");
    int valid = int(ids.size());
    return {std::move(ids), valid};
}
std::string Tokenizer::decode(const std::vector<int> &ids) const {
    require(ids.size() <= 262144, "decode sequence exceeds token limit");
    std::string decoded;
    for (int id : ids) {
        auto found = impl_->token_decoder_.find(id);
        require(found != impl_->token_decoder_.end(), "token ID cannot be decoded");
        decoded += found->second;
    }
    return decoded;
}
Tokens Tokenizer::raw_bounded(const std::string &s, int max_tokens) const {
    require(max_tokens > 0 && max_tokens <= 32768, "invalid extended tokenizer limit");
    require(!s.empty() && s.size() <= 1024 * 1024, "extended prompt must contain 1 byte...1 MiB");
    auto ids = impl_->encode(s);
    require(!ids.empty() && ids.size() <= size_t(max_tokens), "extended prompt exceeds token budget");
    int valid = int(ids.size());
    return {std::move(ids), valid};
}
Tokens Tokenizer::prompt(const std::string &s, bool dynamic) {
    require(!s.empty(), "prompt must not be empty");
    require(s.size() <= 32768, "prompt exceeds 32 KiB");
    auto ids = impl_->encode("<|im_start|>user\n" + s +
                             "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    require(ids.size() <= 512, "prompt exceeds 512 tokens; no silent truncation");
    Tokens t{ids, int(ids.size())};
    if (!dynamic)
        t.ids.resize(512, 151643);
    return t;
}
Tokens Tokenizer::z_image_tokens(const std::string &s) {
    require(s.size() <= 32768, "prompt exceeds 32 KiB");
    // Share the exact generation template with the UI counter, including role tokens.
    auto ids = impl_->encode("<|im_start|>user\n" + s +
                             "<|im_end|>\n<|im_start|>assistant\n");
    return {ids, int(ids.size())};
}
Tokens Tokenizer::z_image_prompt(const std::string &s, bool dynamic) {
    require(!s.empty(), "prompt must not be empty");
    auto t = z_image_tokens(s);
    // 512 is the upstream default, not an ANE restriction. Bound our extended
    // context below the 1536-position caption RoPE table (including padding).
    require(t.valid <= z_image_limit,
            "Z-Image prompt exceeds 1024 tokens; no silent truncation; GPU and ANE share this limit");
    if (!dynamic)
        t.ids.resize(std::max(512, t.valid), 151643);
    return t;
}
Tokens Tokenizer::llada_image_prompt(const std::string &s) {
    require(!s.empty(), "prompt must not be empty");
    require(s.size() <= 32768, "prompt exceeds 32 KiB");
    // LLaDA-Image uses its own role/image tokens and does not add BOS/EOS.
    // Keep this template byte-for-byte aligned with the official pipeline;
    // the trailing newline before <IMAGE1> is part of the token sequence.
    auto ids = impl_->encode("<role>HUMAN</role> Generate an image: " + s +
                             "\n<role>ASSISTANT</role>\n<IMAGE1>");
    require(ids.size() <= 2048, "LLaDA prompt exceeds 2048 tokens; no silent truncation");
    const int valid = int(ids.size());
    return {std::move(ids), valid};
}
} // namespace tc
