#include "json_keys.hpp"
#include <set>
#include <stdexcept>
#include <string>

namespace tc {
namespace {
class KeyScanner {
    std::string_view text;
    size_t at = 0;
    [[noreturn]] void bad() const { throw std::invalid_argument("invalid JSON key structure"); }
    char take() { if (at == text.size()) bad(); return text[at++]; }
    void ws() { while (at < text.size() && (text[at]==' ' || text[at]=='\n' || text[at]=='\r' || text[at]=='\t')) ++at; }
    uint32_t hex4() {
        uint32_t n = 0;
        for (int j=0; j<4; ++j) {
            char c=take(); n <<= 4;
            if (c>='0' && c<='9') n += c-'0';
            else if (c>='a' && c<='f') n += c-'a'+10;
            else if (c>='A' && c<='F') n += c-'A'+10;
            else bad();
        }
        return n;
    }
    static void utf8(std::string &s, uint32_t n) {
        if (n < 0x80) s += char(n);
        else if (n < 0x800) { s += char(0xc0|(n>>6)); s += char(0x80|(n&63)); }
        else if (n < 0x10000) {
            s += char(0xe0|(n>>12)); s += char(0x80|((n>>6)&63)); s += char(0x80|(n&63));
        } else {
            s += char(0xf0|(n>>18)); s += char(0x80|((n>>12)&63));
            s += char(0x80|((n>>6)&63)); s += char(0x80|(n&63));
        }
    }
    std::string string(bool decode) {
        if (take() != '"') bad();
        std::string result;
        while (true) {
            char c = take();
            if (c == '"') return result;
            if (c != '\\') { if (decode) result += c; continue; }
            c = take();
            if (c == 'u') {
                uint32_t n = hex4();
                if (n >= 0xd800 && n <= 0xdbff) {
                    if (take() != '\\' || take() != 'u') bad();
                    auto lo = hex4(); if (lo < 0xdc00 || lo > 0xdfff) bad();
                    n = 0x10000 + ((n-0xd800)<<10) + lo-0xdc00;
                } else if (n >= 0xdc00 && n <= 0xdfff) bad();
                if (decode) utf8(result, n);
            } else {
                switch(c) {
                case 'b': c='\b'; break; case 'f': c='\f'; break;
                case 'n': c='\n'; break; case 'r': c='\r'; break;
                case 't': c='\t'; break;
                case '"': case '\\': case '/': break;
                default: bad();
                }
                if (decode) result += c;
            }
        }
    }
    void value(unsigned depth) {
        if (depth > 128) throw std::invalid_argument("JSON nesting exceeds 128");
        ws(); if (at == text.size()) bad();
        char c = text[at];
        if (c == '"') { string(false); return; }
        if (c == '{' || c == '[') {
            ++at; ws(); const char end = c == '{' ? '}' : ']';
            std::set<std::string> keys;
            if (at < text.size() && text[at] == end) { ++at; return; }
            while (true) {
                ws();
                if (c == '{') {
                    auto key = string(true);
                    if (!keys.insert(key).second)
                        throw std::invalid_argument("duplicate JSON field: " + key);
                    ws(); if (take() != ':') bad();
                }
                value(depth+1); ws();
                const char delimiter = take();
                if (delimiter == end) return;
                if (delimiter != ',') bad();
            }
        }
        const size_t begin = at;
        while (at < text.size() && text[at]!=',' && text[at]!='}' && text[at]!=']' &&
               text[at]!=' ' && text[at]!='\r' && text[at]!='\n' && text[at]!='\t') ++at;
        if (begin == at) bad();
    }
public:
    explicit KeyScanner(std::string_view input) : text(input) {}
    void run() { value(0); ws(); if (at != text.size()) bad(); }
};
}
void reject_duplicate_json_keys(std::string_view json) { KeyScanner(json).run(); }
}
