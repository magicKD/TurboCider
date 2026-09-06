#include "bridge.hpp"
#include <limits>
#include <map>
#include <unordered_map>
namespace tc {
struct Tokenizer::Impl {
    NSDictionary *vocab_;
    NSRegularExpression *pattern_;
    std::unordered_map<std::string, int> ranks_;
    std::vector<std::string> byte_encoder_;
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
        require([pair isKindOfClass:NSArray.class] && [pair count] == 2, "unsupported BPE merges");
        std::string a = [pair[0] UTF8String], b = [pair[1] UTF8String];
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
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        int cp = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) ? b : 256 + extra++;
        unichar c = cp;
        byte_encoder_[b] = [[NSString stringWithCharacters:&c length:1] UTF8String];
    }
}
std::vector<int> Tokenizer::Impl::encode(const std::string &raw) {
    NSString *s = [@(raw.c_str()) precomposedStringWithCanonicalMapping];
    std::string text = s.UTF8String;
    std::vector<int> result;
    auto ordinary = [&](const std::string &sub) {
        NSString *part = @(sub.c_str());
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
Tokens Tokenizer::z_image_prompt(const std::string &s, bool dynamic) {
    require(!s.empty(), "prompt must not be empty");
    require(s.size() <= 32768, "prompt exceeds 32 KiB");
    // Z-Image-Turbo uses Qwen3's chat template with enable_thinking=true.
    // Unlike the FLUX conditioning template above, it must not prefill an
    // empty <think>...</think> block after the assistant generation prompt.
    auto ids = impl_->encode("<|im_start|>user\n" + s +
                             "<|im_end|>\n<|im_start|>assistant\n");
    require(ids.size() <= 512, "prompt exceeds 512 tokens; no silent truncation");
    Tokens t{ids, int(ids.size())};
    if (!dynamic)
        t.ids.resize(512, 151643);
    return t;
}
} // namespace tc
