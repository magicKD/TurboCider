#import <Foundation/Foundation.h>
#include "../../core/unigram_tokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace tc {
namespace {
id restore_json_strings(id value) {
    if ([value isKindOfClass:NSString.class]) return [value substringFromIndex:1];
    if ([value isKindOfClass:NSArray.class]) {
        NSMutableArray *result = [NSMutableArray arrayWithCapacity:[value count]];
        for (id item in value) [result addObject:restore_json_strings(item)];
        return result;
    }
    if ([value isKindOfClass:NSDictionary.class]) {
        NSMutableDictionary *result = [NSMutableDictionary dictionaryWithCapacity:[value count]];
        for (NSString *key in value) result[[key substringFromIndex:1]] = restore_json_strings(value[key]);
        return result;
    }
    return value;
}
} // namespace
struct UnigramTokenizer::Impl {
    struct Node {
        std::unordered_map<unsigned char, size_t> next;
        int token = -1;
    };
    std::vector<Node> trie{1};
    std::vector<double> scores;
    std::vector<std::pair<std::string, int>> special;
    double unknown_score = 0;
    int unknown = 3;
    bool t5 = false;
    std::vector<uint32_t> normalizer_trie;
    std::string normalizer_replacements;
    void encode_piece(const std::string &, std::vector<int> &) const;
    void ordinary(const std::string &, std::vector<int> &) const;
    std::optional<std::string> t5_transform(const std::string &) const;
    std::string t5_normalize(const std::string &) const;
    void t5_ordinary(const std::string &, bool, std::vector<int> &) const;
};

UnigramTokenizer::UnigramTokenizer(const std::filesystem::path &root, UnigramFlavor flavor) : impl_(std::make_unique<Impl>()) {
    require(root.is_absolute(), "Unigram tokenizer root must be absolute");
    auto path = root / "tokenizer.json";
    NSData *data = [NSData dataWithContentsOfFile:@(path.c_str())];
    require(data != nil, "Unigram tokenizer.json missing");
    // Foundation consumes a leading U+FEFF inside JSON strings as a BOM, even
    // when escaped. Prefix every JSON string (keys included) before parsing,
    // then remove exactly that prefix structurally. No sentinel collision and
    // no token-specific replacement: embedded NUL and BOM remain vocabulary.
    std::string source((const char *)data.bytes, data.length), protected_json;
    protected_json.reserve(source.size());
    bool inside = false, escaped = false;
    for (char byte : source) {
        protected_json += byte;
        if (!inside && byte == '"') { protected_json += 'x'; inside = true; }
        else if (inside) {
            if (escaped) escaped = false;
            else if (byte == '\\') escaped = true;
            else if (byte == '"') inside = false;
        }
    }
    data = [NSData dataWithBytes:protected_json.data() length:protected_json.size()];
    id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    require([parsed isKindOfClass:NSDictionary.class], "invalid Unigram tokenizer JSON");
    NSDictionary *config = restore_json_strings(parsed);
    NSDictionary *model = config[@"model"];
    impl_->t5 = flavor == UnigramFlavor::t5_v1_1;
    impl_->unknown = impl_->t5 ? 2 : 3;
    require([model isKindOfClass:NSDictionary.class] &&
                ([model[@"type"] isEqual:@"Unigram"] || (impl_->t5 && model[@"type"] == nil)) &&
                [model[@"unk_id"] isEqual:@(impl_->unknown)] &&
                ([model[@"byte_fallback"] isEqual:@NO] || (impl_->t5 && model[@"byte_fallback"] == nil)),
            "unsupported Unigram model contract");
    if (impl_->t5) {
        NSDictionary *normalizer = config[@"normalizer"], *metaspace = config[@"pre_tokenizer"];
        NSArray *steps = [normalizer isKindOfClass:NSDictionary.class] ? normalizer[@"normalizers"] : nil;
        const bool comfy_format = [normalizer[@"type"] isEqual:@"Sequence"] && [steps isKindOfClass:NSArray.class] && steps.count == 3 &&
                    [steps[0][@"type"] isEqual:@"Precompiled"] && [steps[0][@"precompiled_charsmap"] isKindOfClass:NSString.class] &&
                    [steps[1] isEqual:@{@"type":@"Strip", @"strip_left":@NO, @"strip_right":@YES}] &&
                    [steps[2] isEqual:@{@"type":@"Replace", @"pattern":@{@"Regex":@" {2,}"}, @"content":@"▁"}] &&
                    [metaspace[@"type"] isEqual:@"Metaspace"] && [metaspace[@"replacement"] isEqual:@"▁"] &&
                    [metaspace[@"prepend_scheme"] isEqual:@"first"] && [metaspace[@"add_prefix_space"] isEqual:@YES];
        // The public google-t5/t5-base JSON stores the same T5 vocabulary and
        // charsmap using an older schema. Run the explicitly selected Comfy
        // LTX T5 normalization contract rather than its legacy pre-tokenizer.
        const bool google_format = [normalizer[@"type"] isEqual:@"Precompiled"] &&
                    [normalizer[@"precompiled_charsmap"] isKindOfClass:NSString.class] &&
                    [metaspace isEqual:@{@"type":@"Sequence", @"pretokenizers":@[
                        @{@"type":@"WhitespaceSplit"},
                        @{@"type":@"Metaspace", @"replacement":@"▁", @"str_rep":@"▁", @"add_prefix_space":@YES}]}];
        require(comfy_format || google_format,
                "unsupported T5 normalization/pre-tokenization");
        NSString *charsmap = comfy_format ? steps[0][@"precompiled_charsmap"] : normalizer[@"precompiled_charsmap"];
        NSData *blob = [[NSData alloc] initWithBase64EncodedString:charsmap options:0];
        require(blob && blob.length > 1028 && blob.length <= (16u << 20), "invalid T5 precompiled charsmap");
        const auto *bytes = static_cast<const unsigned char *>(blob.bytes);
        const uint32_t trie_bytes = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
                                    (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
        require(trie_bytes >= 1024 && trie_bytes % 1024 == 0 && size_t(trie_bytes) < blob.length - 4,
                "invalid T5 precompiled trie size");
        impl_->normalizer_trie.resize(trie_bytes / 4);
        for (size_t i = 0; i < impl_->normalizer_trie.size(); ++i) {
            const auto *unit = bytes + 4 + i * 4;
            impl_->normalizer_trie[i] = uint32_t(unit[0]) | (uint32_t(unit[1]) << 8) |
                                        (uint32_t(unit[2]) << 16) | (uint32_t(unit[3]) << 24);
        }
        impl_->normalizer_replacements.assign(reinterpret_cast<const char *>(bytes + 4 + trie_bytes),
                                               blob.length - 4 - trie_bytes);
        require(!impl_->normalizer_replacements.empty() && impl_->normalizer_replacements.back() == '\0',
                "invalid T5 normalization replacement table");
    } else {
        NSDictionary *normalizer = @{@"type":@"Sequence", @"normalizers":@[
            @{@"type":@"Replace", @"pattern":@{@"Regex":@" {2,}"}, @"content":@" "}]};
        NSDictionary *metaspace = @{@"type":@"Metaspace", @"replacement":@"▁", @"prepend_scheme":@"always", @"split":@YES};
        require([config[@"normalizer"] isEqual:normalizer] && [config[@"pre_tokenizer"] isEqual:metaspace],
                "unsupported Unigram normalization/pre-tokenization");
    }
    NSDictionary *post = config[@"post_processor"];
    require([post isKindOfClass:NSDictionary.class] && [post[@"type"] isEqual:@"TemplateProcessing"] &&
                [post[@"single"] isEqual:@[@{@"Sequence":@{@"id":@"A", @"type_id":@0}},
                                           @{@"SpecialToken":@{@"id":@"</s>", @"type_id":@0}}]] &&
                [post[@"special_tokens"][@"</s>"][@"ids"] isEqual:@[@1]],
            "unsupported Unigram EOS template");
    NSArray *vocab = model[@"vocab"];
    require([vocab isKindOfClass:NSArray.class] && vocab.count > 4 && vocab.count <= 1000000,
            "invalid Unigram vocabulary");
    double minimum = 0;
    int vocabulary_index = 0;
    for (NSArray *record in vocab) {
        require([record isKindOfClass:NSArray.class] && record.count == 2 &&
                    [record[0] isKindOfClass:NSString.class] && [record[1] isKindOfClass:NSNumber.class],
                "invalid Unigram vocabulary entry");
        // Prefixing also prevents NSString UTF-8 conversion from treating an
        // initial U+FEFF as an encoding marker. Preserve embedded NUL bytes.
        NSData *encoded = [[@"x" stringByAppendingString:record[0]] dataUsingEncoding:NSUTF8StringEncoding];
        require(encoded != nil && encoded.length >= 1, "Unigram token is not UTF-8");
        std::string token((const char *)encoded.bytes + 1, encoded.length - 1);
        const double score = [record[1] doubleValue];
        require(!token.empty() && std::isfinite(score),
                "invalid Unigram token/score at index " + std::to_string(vocabulary_index) +
                " utf16=" + std::to_string([record[0] length]) + " score=" + std::to_string(score));
        size_t node = 0;
        for (unsigned char byte : token) {
            auto found = impl_->trie[node].next.find(byte);
            if (found == impl_->trie[node].next.end()) {
                const size_t next = impl_->trie.size();
                impl_->trie[node].next.emplace(byte, next);
                impl_->trie.emplace_back();
                node = next;
            } else node = found->second;
        }
        require(impl_->trie[node].token < 0, "duplicate Unigram token");
        impl_->trie[node].token = int(impl_->scores.size());
        impl_->scores.push_back(score);
        minimum = std::min(minimum, score);
        ++vocabulary_index;
    }
    impl_->unknown_score = minimum - 10.;
    NSArray *added = config[@"added_tokens"];
    require([added isKindOfClass:NSArray.class], "Unigram special tokens missing");
    for (NSDictionary *entry in added) {
        require([entry isKindOfClass:NSDictionary.class] && [entry[@"content"] isKindOfClass:NSString.class] &&
                    [entry[@"id"] isKindOfClass:NSNumber.class] &&
                    [entry[@"special"] isEqual:@YES] && [entry[@"normalized"] isEqual:@NO] &&
                    [entry[@"single_word"] isEqual:@NO] && [entry[@"lstrip"] isEqual:@NO] &&
                    [entry[@"rstrip"] isEqual:@NO], "unsupported added-token rules");
        std::string token([entry[@"content"] UTF8String]);
        int id = [entry[@"id"] intValue];
        require(!token.empty() && id >= 0 && id < int(vocab.count), "invalid added token");
        impl_->special.emplace_back(token, id);
    }
}

UnigramTokenizer::~UnigramTokenizer() = default;

void UnigramTokenizer::Impl::encode_piece(const std::string &piece, std::vector<int> &output) const {
    struct Path { double score = -std::numeric_limits<double>::infinity(); size_t previous = 0; int token = -1; };
    std::vector<Path> best(piece.size() + 1);
    best[0].score = 0;
    for (size_t start = 0; start < piece.size();) {
        size_t end_char = start + 1;
        while (end_char < piece.size() && (static_cast<unsigned char>(piece[end_char]) & 0xc0) == 0x80) ++end_char;
        bool has_character = false;
        size_t node = 0;
        auto update = [&](size_t end, int token, double score) {
            const double candidate = best[start].score + score;
            if (candidate > best[end].score) best[end] = {candidate, start, token};
        };
        for (size_t end = start; end < piece.size(); ++end) {
            auto found = trie[node].next.find(static_cast<unsigned char>(piece[end]));
            if (found == trie[node].next.end()) break;
            node = found->second;
            const int token = trie[node].token;
            if (token >= 0) {
                update(end + 1, token, scores[token]);
                if (end + 1 == end_char) has_character = true;
            }
        }
        if (!has_character) update(end_char, unknown, unknown_score);
        start = end_char;
    }
    std::vector<int> reversed;
    for (size_t at = piece.size(); at > 0; at = best[at].previous) {
        require(best[at].token >= 0 && best[at].previous < at, "Unigram segmentation failed");
        reversed.push_back(best[at].token);
    }
    int previous = -1;
    for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
        if (*it != unknown || previous != unknown) output.push_back(*it);
        previous = *it;
    }
}

void UnigramTokenizer::Impl::ordinary(const std::string &raw, std::vector<int> &output) const {
    if (raw.empty()) return;
    const std::string marker = "▁";
    std::string normalized;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == ' ') {
            normalized += marker;
            while (i + 1 < raw.size() && raw[i + 1] == ' ') ++i;
        } else normalized += raw[i];
    }
    if (!normalized.starts_with(marker)) normalized = marker + normalized;
    size_t start = 0;
    while (start < normalized.size()) {
        auto end = normalized.find(marker, start + marker.size());
        if (end == std::string::npos) end = normalized.size();
        encode_piece(normalized.substr(start, end - start), output);
        start = end;
    }
}

// Read the SentencePiece precompiled charsmap and first-match double-array
// trie in the same order as Hugging Face's Apache-2.0 spm_precompiled crate.
// Bounds checks below are TurboCider additions for untrusted local models.
std::optional<std::string> UnigramTokenizer::Impl::t5_transform(const std::string &chunk) const {
    auto offset = [](uint32_t unit) -> size_t { return size_t(unit >> 10) << ((unit & (1u << 9)) >> 6); };
    size_t position = offset(normalizer_trie.at(0));
    for (unsigned char byte : chunk) {
        if (byte == 0) break;
        position ^= byte;
        require(position < normalizer_trie.size(), "T5 normalization trie node outside bounds");
        const uint32_t unit = normalizer_trie[position];
        if ((unit & (0x80000000u | 0xffu)) != byte) break;
        position ^= offset(unit);
        require(position < normalizer_trie.size(), "T5 normalization trie offset outside bounds");
        if ((unit >> 8) & 1u) {
            const size_t index = normalizer_trie[position] & 0x7fffffffu;
            require(index < normalizer_replacements.size(), "T5 normalization replacement outside bounds");
            const auto end = normalizer_replacements.find('\0', index);
            require(end != std::string::npos, "unterminated T5 normalization replacement");
            return normalizer_replacements.substr(index, end - index);
        }
    }
    return std::nullopt;
}

std::string UnigramTokenizer::Impl::t5_normalize(const std::string &raw) const {
    NSString *input = [[NSString alloc] initWithBytes:raw.data() length:raw.size() encoding:NSUTF8StringEncoding];
    require(input != nil, "T5 prompt must be valid UTF-8");
    __block std::string normalized;
    const Impl *owner = this;
    [input enumerateSubstringsInRange:NSMakeRange(0, input.length)
                              options:NSStringEnumerationByComposedCharacterSequences
                           usingBlock:^(NSString *piece, NSRange, NSRange, BOOL *) {
        NSData *encoded = [[@"x" stringByAppendingString:piece] dataUsingEncoding:NSUTF8StringEncoding];
        std::string bytes(static_cast<const char *>(encoded.bytes) + 1, encoded.length - 1);
        if (bytes.size() < 6) {
            auto replacement = owner->t5_transform(bytes);
            if (replacement) { normalized += *replacement; return; }
        }
        for (size_t at = 0; at < bytes.size();) {
            size_t end = at + 1;
            while (end < bytes.size() && (static_cast<unsigned char>(bytes[end]) & 0xc0) == 0x80) ++end;
            auto scalar = bytes.substr(at, end - at);
            auto replacement = owner->t5_transform(scalar);
            normalized += replacement ? *replacement : scalar;
            at = end;
        }
    }];
    NSString *text = [[NSString alloc] initWithBytes:normalized.data() length:normalized.size() encoding:NSUTF8StringEncoding];
    require(text != nil, "invalid T5 normalization output");
    NSCharacterSet *spaces = NSCharacterSet.whitespaceAndNewlineCharacterSet;
    NSUInteger end = text.length;
    while (end > 0 && [spaces characterIsMember:[text characterAtIndex:end - 1]]) --end;
    NSData *trimmed = [[@"x" stringByAppendingString:[text substringToIndex:end]] dataUsingEncoding:NSUTF8StringEncoding];
    return std::string(static_cast<const char *>(trimmed.bytes) + 1, trimmed.length - 1);
}

void UnigramTokenizer::Impl::t5_ordinary(const std::string &raw, bool prepend, std::vector<int> &output) const {
    if (raw.empty()) return;
    const std::string marker = "▁";
    const std::string text = t5_normalize(raw);
    std::string encoded;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == ' ') {
            encoded += marker;
            while (i + 1 < text.size() && text[i + 1] == ' ') ++i;
        } else encoded += text[i];
    }
    if (encoded.empty()) return;
    if (prepend && !encoded.starts_with(marker)) encoded = marker + encoded;
    for (size_t start = 0; start < encoded.size();) {
        const size_t search = encoded.compare(start, marker.size(), marker) == 0 ? start + marker.size() : start;
        auto end = encoded.find(marker, search);
        if (end == std::string::npos) end = encoded.size();
        encode_piece(encoded.substr(start, end - start), output);
        start = end;
    }
}

Tokens UnigramTokenizer::prompt(const std::string &raw, int limit) const {
    require(!impl_->t5, "T5 tokenizer must use t5_prompt");
    require(raw.size() <= 32768 && limit >= 1 && limit <= 512, "Unigram prompt/limit exceeds supported bounds");
    NSString *utf8 = [[NSString alloc] initWithBytes:raw.data() length:raw.size() encoding:NSUTF8StringEncoding];
    require(utf8 != nil, "Unigram prompt must be valid UTF-8");
    std::vector<int> ids;
    size_t start = 0;
    while (start < raw.size()) {
        size_t next = raw.size();
        const std::pair<std::string, int> *selected = nullptr;
        for (const auto &entry : impl_->special) {
            auto at = raw.find(entry.first, start);
            if (at < next || (at == next && selected && entry.first.size() > selected->first.size())) {
                next = at; selected = &entry;
            }
        }
        impl_->ordinary(raw.substr(start, next - start), ids);
        if (!selected) break;
        ids.push_back(selected->second);
        start = next + selected->first.size();
    }
    if (ids.size() > size_t(limit - 1)) ids.resize(limit - 1);
    ids.push_back(1);
    int valid = int(ids.size());
    ids.resize(limit, 0);
    return {std::move(ids), valid};
}

Tokens UnigramTokenizer::t5_prompt(const std::string &raw, int max_tokens) const {
    require(impl_->t5 && raw.size() <= 32768 && max_tokens >= 128 && max_tokens <= 512,
            "T5 prompt/limit exceeds supported bounds");
    NSString *utf8 = [[NSString alloc] initWithBytes:raw.data() length:raw.size() encoding:NSUTF8StringEncoding];
    require(utf8 != nil, "T5 prompt must be valid UTF-8");
    std::vector<int> ids;
    size_t start = 0;
    bool first = true;
    while (start < raw.size()) {
        size_t next = raw.size();
        const std::pair<std::string, int> *selected = nullptr;
        for (const auto &entry : impl_->special) {
            auto at = raw.find(entry.first, start);
            if (at < next || (at == next && selected && entry.first.size() > selected->first.size())) {
                next = at; selected = &entry;
            }
        }
        impl_->t5_ordinary(raw.substr(start, next - start), first, ids);
        if (!selected) break;
        ids.push_back(selected->second);
        first = false;
        start = next + selected->first.size();
    }
    require(ids.size() < size_t(max_tokens), "T5 prompt exceeds the supported token limit");
    ids.push_back(1);
    const int valid = int(ids.size());
    ids.resize(std::max<size_t>(128, ids.size()), 0);
    return {std::move(ids), valid};
}

} // namespace tc
