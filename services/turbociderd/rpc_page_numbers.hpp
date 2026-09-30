#pragma once
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tc_service {
// Foundation may round a tiny fractional part away. Check the original
// decimal value, preserving valid integral forms such as 1.0 and 100e-2.
inline void exact_page_integer(std::string_view token, const std::string &key,
                               int minimum, int maximum) {
    auto bad = [&]() { throw std::invalid_argument(key + " must be an exact integer in the supported range"); };
    size_t at = 0;
    const bool negative = !token.empty() && token[0] == '-';
    if (negative) ++at;
    std::string digits;
    while (at < token.size() && token[at] >= '0' && token[at] <= '9') digits += token[at++];
    if (digits.empty()) bad();
    int64_t fraction = 0;
    if (at < token.size() && token[at] == '.') {
        ++at;
        while (at < token.size() && token[at] >= '0' && token[at] <= '9') {
            digits += token[at++]; ++fraction;
        }
        if (!fraction) bad();
    }
    int64_t exponent = 0;
    if (at < token.size() && (token[at] == 'e' || token[at] == 'E')) {
        ++at;
        bool minus = false;
        if (at < token.size() && (token[at] == '+' || token[at] == '-')) minus = token[at++] == '-';
        const auto begin = at;
        // Input is at most 1 MiB, so a larger exponent cannot be cancelled
        // by its decimal digits. Saturation avoids integer overflow.
        while (at < token.size() && token[at] >= '0' && token[at] <= '9')
            exponent = std::min<int64_t>(10000000, exponent * 10 + token[at++] - '0');
        if (at == begin) bad();
        if (minus) exponent = -exponent;
    }
    if (at != token.size()) bad();
    const auto first = digits.find_first_not_of('0');
    if (first == std::string::npos) { if (minimum > 0) bad(); return; }
    if (negative) bad();
    digits.erase(0, first);
    int64_t shift = exponent - fraction;
    if (shift < 0) {
        const uint64_t remove = uint64_t(-shift);
        if (remove >= digits.size()) bad();
        const auto begin = digits.size() - size_t(remove);
        if (digits.find_first_not_of('0', begin) != std::string::npos) bad();
        digits.resize(begin); shift = 0;
    }
    if (digits.size() + uint64_t(shift) > 10) bad();
    int64_t value = 0;
    for (char digit : digits) value = value * 10 + digit - '0';
    while (shift-- > 0) value *= 10;
    if (value < minimum || value > maximum) bad();
}

// Called only after the shared JSON scanner, Foundation decoding and jobs
// envelope validation. This envelope has only a string action and numbers;
// no second general JSON parser or native-request validator is introduced.
inline void validate_page_number_tokens(std::string_view text) {
    size_t at = 0;
    auto ws = [&]() { while (at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\r' || text[at] == '\n')) ++at; };
    auto quoted = [&]() {
        const auto begin = at++;
        while (at < text.size()) {
            const char value = text[at++];
            if (value == '\\') { ++at; continue; }
            if (value == '"') return text.substr(begin, at - begin);
        }
        throw std::invalid_argument("invalid RPC page key structure");
    };
    ws(); ++at; ws();
    while (at < text.size() && text[at] != '}') {
        const auto raw_key = quoted();
        NSData *data = [NSData dataWithBytes:raw_key.data() length:raw_key.size()];
        NSString *key = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingFragmentsAllowed error:nil];
        ws(); ++at; ws();
        if ([key isEqual:@"action"]) (void)quoted();
        else {
            const auto begin = at;
            while (at < text.size() && text[at] != ',' && text[at] != '}' && text[at] != ' ' && text[at] != '\t' && text[at] != '\r' && text[at] != '\n') ++at;
            const bool limit = [key isEqual:@"limit"];
            exact_page_integer(text.substr(begin, at - begin), key.UTF8String,
                               limit ? 1 : 0, limit ? 100 : 2147483647);
        }
        ws(); if (at < text.size() && text[at] == ',') { ++at; ws(); } else break;
    }
}
} // namespace tc_service
