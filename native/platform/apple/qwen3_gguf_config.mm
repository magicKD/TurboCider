#import <Foundation/Foundation.h>
#include "../../components/text/qwen3_gguf.hpp"
#include "../../core/json_keys.hpp"
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace tc::components {
namespace {
std::string read_span(int fd, uint64_t offset, uint64_t bytes, uint64_t maximum) {
    require(fd >= 0 && bytes && bytes <= maximum && offset <= INT64_MAX - bytes,
            "qe_adapter_mismatch: invalid bounded metadata span");
    std::string raw(size_t(bytes), '\0');
    uint64_t done = 0;
    while (done < bytes) {
        const auto n = ::pread(fd, raw.data() + done, size_t(bytes - done), off_t(offset + done));
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "qe_source_changed: short metadata read"); done += uint64_t(n);
    }
    return raw;
}
NSDictionary *json_object(int fd, uint64_t bytes, uint64_t maximum) {
    const auto raw = read_span(fd, 0, bytes, maximum);
    reject_duplicate_json_keys(raw);
    NSData *data = [NSData dataWithBytes:raw.data() length:raw.size()];
    id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    require([object isKindOfClass:NSDictionary.class], "qe_adapter_mismatch: metadata must be JSON object");
    return object;
}
double number(id value) {
    require([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            "qe_adapter_mismatch: config/token ID must be numeric");
    const double result = [value doubleValue];
    require(std::isfinite(result), "qe_adapter_mismatch: nonfinite config"); return result;
}
uint32_t integer(id value, uint32_t limit) {
    const double result = number(value);
    require(result >= 0 && result <= limit && std::floor(result) == result,
            "qe_adapter_mismatch: invalid config/token ID"); return uint32_t(result);
}
bool boolean(id value) {
    require(value && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID(),
            "qe_adapter_mismatch: config boolean has wrong type"); return [value boolValue];
}
}
Qwen3GgufConfig read_qwen3_gguf_config(int fd, uint64_t bytes) {
    @autoreleasepool {
        NSDictionary *d = json_object(fd, bytes, 1u << 20);
        require([d[@"model_type"] isEqual:@"qwen3"] && [d[@"hidden_act"] isEqual:@"silu"] &&
            boolean(d[@"tie_word_embeddings"]) &&
            (!d[@"attention_bias"] || !boolean(d[@"attention_bias"])) &&
            (!d[@"rope_scaling"] || d[@"rope_scaling"] == NSNull.null) &&
            (!d[@"use_sliding_window"] || !boolean(d[@"use_sliding_window"])),
            "qe_adapter_mismatch: only standard tied dense Qwen3 is supported");
        Qwen3GgufConfig c;
        c.hidden = integer(d[@"hidden_size"], 32768); c.intermediate = integer(d[@"intermediate_size"], 32768);
        c.vocabulary = integer(d[@"vocab_size"], 1u << 20); c.layers = integer(d[@"num_hidden_layers"], 128);
        c.heads = integer(d[@"num_attention_heads"], 128); c.kv_heads = integer(d[@"num_key_value_heads"], 128);
        c.head_dim = integer(d[@"head_dim"], 256); c.context_length = integer(d[@"max_position_embeddings"], 1u << 20);
        c.rope_theta = float(number(d[@"rope_theta"])); c.epsilon = float(number(d[@"rms_norm_eps"]));
        require(c.hidden == 2560 && c.intermediate && c.vocabulary && c.layers >= 35 &&
            c.heads == 32 && c.kv_heads == 8 && c.head_dim == 128 && c.rope_theta == 1000000.f &&
            c.epsilon == 1e-6f && c.context_length >= 1024 &&
            integer(d[@"bos_token_id"], c.vocabulary - 1) == 151643 &&
            integer(d[@"eos_token_id"], c.vocabulary - 1) == 151645,
            "qe_adapter_mismatch: config differs from Z Qwen3-4B conditioning contract");
        return c;
    }
}

void verify_qwen3_gguf_tokenizer(int fd, uint64_t bytes, int source_fd,
        const gguf::Directory &directory, const Qwen3GgufConfig &c) {
    @autoreleasepool {
        NSDictionary *d = json_object(fd, bytes, 32u << 20);
        NSDictionary *model = d[@"model"];
        require([model isKindOfClass:NSDictionary.class] && [model[@"type"] isEqual:@"BPE"] &&
                [model[@"vocab"] isKindOfClass:NSDictionary.class], "qe_adapter_mismatch: unsupported tokenizer");
        std::unordered_map<uint32_t, std::string> words;
        auto add = [&](NSString *word, id value) {
            require([word isKindOfClass:NSString.class], "qe_adapter_mismatch: invalid token string");
            const auto id = integer(value, c.vocabulary - 1);
            NSData *utf8 = [word dataUsingEncoding:NSUTF8StringEncoding];
            require(utf8 != nil, "qe_adapter_mismatch: invalid UTF8 token");
            std::string text(static_cast<const char *>(utf8.bytes), utf8.length);
            auto [found, fresh] = words.emplace(id, text);
            require(fresh || found->second == text, "qe_adapter_mismatch: conflicting token IDs");
        };
        NSDictionary *vocab = model[@"vocab"];
        for (NSString *word in vocab) add(word, vocab[word]);
        require([d[@"added_tokens"] isKindOfClass:NSArray.class], "qe_adapter_mismatch: missing added tokens");
        for (NSDictionary *token in d[@"added_tokens"]) {
            require([token isKindOfClass:NSDictionary.class], "qe_adapter_mismatch: malformed added token");
            add(token[@"content"], token[@"id"]);
        }
        const auto *meta = directory.meta("tokenizer.ggml.tokens");
        require(meta && meta->type == 9 && meta->element_type == 8 && meta->count == c.vocabulary,
                "qe_adapter_mismatch: GGUF vocabulary missing or wrong count");
        const auto raw = read_span(source_fd, meta->value_offset, meta->value_bytes, 32u << 20);
        size_t position = 12; uint64_t checked = 0;
        auto u64 = [&](size_t at) {
            require(at <= raw.size() && raw.size() - at >= 8, "qe_decode_invalid: truncated vocab");
            uint64_t value = 0; for (unsigned i = 0; i < 8; ++i) value |= uint64_t(uint8_t(raw[at + i])) << (8 * i);
            return value;
        };
        for (uint32_t id = 0; id < c.vocabulary; ++id) {
            const uint64_t length = u64(position); position += 8;
            require(length <= raw.size() - position, "qe_decode_invalid: token string span invalid");
            const auto found = words.find(id);
            if (found != words.end()) {
                require(found->second.size() == length && std::memcmp(found->second.data(), raw.data() + position, size_t(length)) == 0,
                        "qe_adapter_mismatch: tokenizer token-ID mapping differs from GGUF at " + std::to_string(id));
                ++checked;
            }
            position += size_t(length);
        }
        require(position == raw.size() && checked == words.size() && checked > 151643,
                "qe_adapter_mismatch: tokenizer vocabulary verification incomplete");
    }
}
} // namespace tc::components
