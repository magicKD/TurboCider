#import <Foundation/Foundation.h>
#include "../../models/qwen21/prompt_rewrite.hpp"
#include "../../core/common.hpp"
#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <map>

namespace tc::qwen21 {
namespace {
std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}
std::vector<std::string> objects(const std::string &answer) {
    std::vector<std::string> result;
    size_t depth = 0, start = 0;
    bool in_string = false, escaped = false;
    for (size_t i = 0; i < answer.size(); ++i) {
        const char c = answer[i];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"' && depth) in_string = true;
        else if (c == '{') { if (!depth) start = i; ++depth; }
        else if (c == '}' && depth && !--depth) result.push_back(answer.substr(start, i - start + 1));
    }
    std::reverse(result.begin(), result.end());
    return result;
}
bool has_trailing_comma(const std::string &json) {
    bool in_string = false, escaped = false;
    for (size_t i = 0; i < json.size(); ++i) {
        char c = json[i];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') { in_string = true; continue; }
        if (c == ',') {
            size_t j = i + 1;
            while (j < json.size() && std::isspace(static_cast<unsigned char>(json[j]))) ++j;
            if (j < json.size() && (json[j] == '}' || json[j] == ']')) return true;
        }
    }
    return false;
}
std::string text(id value) {
    if (![value isKindOfClass:NSString.class]) return {};
    NSData *data = [(NSString *)value dataUsingEncoding:NSUTF8StringEncoding];
    return data ? trim(std::string(static_cast<const char *>(data.bytes), data.length)) : "";
}
int aligned(double value) {
    double x = value / 32., lo = std::floor(x), fraction = x - lo;
    return int(lo + (fraction > .5 || (fraction == .5 && int64_t(lo) % 2))) * 32;
}
}
PromptRewriteResult parse_prompt_rewrite_answer(const std::string &decoded, bool edit) {
    @autoreleasepool {
        PromptRewriteResult result;
        require(decoded.size() <= 4 * 1024 * 1024, "prompt-enhancer answer exceeds 4 MiB");
        auto end = decoded.find("</think>");
        std::string answer = decoded;
        if (end != std::string::npos) {
            result.thinking = decoded.substr(0, end);
            auto begin = result.thinking.find("<think>");
            if (begin != std::string::npos) result.thinking.erase(0, begin + 7);
            result.thinking = trim(result.thinking);
            answer = decoded.substr(end + 8);
        } else if (auto begin = decoded.find("<think>"); begin != std::string::npos) {
            result.thinking = trim(decoded.substr(begin + 7));
            return result; // unfinished thinking must never become the image prompt
        }
        for (const auto &candidate : objects(answer)) {
            // Foundation accepts trailing commas; the answer contract is JSON.
            if (has_trailing_comma(candidate)) continue;
            NSData *data = [NSData dataWithBytes:candidate.data() length:candidate.size()];
            id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
            if (![parsed isKindOfClass:NSDictionary.class]) continue;
            NSDictionary *object = parsed;
            auto rewritten = text(object[@"rewritten_prompt"]);
            if (rewritten.empty()) rewritten = text(object[@"rewrited_prompt"]);
            if (rewritten.empty() || rewritten.find('\0') != std::string::npos) continue;
            result.positive_prompt = std::move(rewritten);
            result.wh_ratio = text(object[@"wh_ratio"]);
            result.ratio_follow = edit ? text(object[@"ratio_follow"]) : "";
            result.parse_ok = true;
            return result;
        }
        result.positive_prompt = trim(answer);
        return result; // expose raw fallback, but never mark it successfully parsed
    }
}
std::string prompt_rewrite_chat(const std::string &system_prompt,
    const std::string &user_prompt, size_t references) {
    @autoreleasepool {
        require(references <= 10 && system_prompt.size() <= 1024 * 1024 && user_prompt.size() <= 1024 * 1024,
                "PE chat exceeds input limits");
        auto unicode_trim = [](const std::string &value) {
            NSString *text = [[NSString alloc] initWithBytes:value.data() length:value.size() encoding:NSUTF8StringEncoding];
            require(text != nil, "PE chat is not UTF-8");
            text = [text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
            NSData *data = [text dataUsingEncoding:NSUTF8StringEncoding];
            return std::string(static_cast<const char *>(data.bytes), data.length);
        };
        std::string content;
        for (size_t i = 0; i < references; ++i) content += "<|vision_start|><|image_pad|><|vision_end|>";
        content += user_prompt;
        auto system = unicode_trim(system_prompt), user = unicode_trim(content);
        require(!system.empty() && !unicode_trim(user_prompt).empty(), "PE system/user prompt is empty");
        return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user +
               "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    }
}
PromptRewriteCanvas prompt_rewrite_canvas(const PromptRewriteResult &rewrite,
    const std::vector<std::pair<int, int>> &references, int follow_resolution) {
    require(rewrite.parse_ok && !rewrite.positive_prompt.empty(), "cannot apply an unparsed prompt rewrite");
    require(rewrite.wh_ratio.empty() || rewrite.ratio_follow.empty(),
            "prompt rewrite specifies both wh_ratio and ratio_follow");
    PromptRewriteCanvas canvas;
    if (!rewrite.ratio_follow.empty()) {
        const auto &value = rewrite.ratio_follow;
        require(value.starts_with("<image") && value.ends_with(">") && value.size() > 7,
                "invalid prompt-rewrite ratio_follow");
        const char *begin = value.data() + 6, *end = value.data() + value.size() - 1;
        int index = 0;
        auto parsed = std::from_chars(begin, end, index);
        require(parsed.ec == std::errc{} && parsed.ptr == end && index >= 1 && index <= 10 &&
                    size_t(index) <= references.size(), "prompt-rewrite reference is out of range");
        const auto [width, height] = references[size_t(index - 1)];
        require(width > 0 && height > 0 && follow_resolution >= 32 && follow_resolution <= 2048,
                "invalid prompt-rewrite reference dimensions/resolution");
        double ratio = double(width) / height;
        double area = double(follow_resolution) * follow_resolution;
        // Validate before converting an extreme aspect ratio to integer pixels.
        require(std::sqrt(area * ratio) <= 4112 && std::sqrt(area / ratio) <= 4112,
                "prompt-rewrite reference aspect ratio exceeds canvas limits");
        canvas.width = aligned(std::sqrt(area * ratio));
        canvas.height = aligned(std::sqrt(area / ratio));
        require(canvas.width >= 64 && canvas.height >= 64 && canvas.width <= 4096 && canvas.height <= 4096 &&
                    int64_t(canvas.width) * canvas.height <= 8388608,
                "prompt-rewrite canvas exceeds Qwen21 limits");
        canvas.follows_reference = true;
        canvas.reference_index = index - 1;
        return canvas;
    }
    static const std::map<std::string, std::pair<int, int>> sizes = {
        {"1:1", {2048, 2048}}, {"4:3", {2400, 1792}}, {"3:4", {1792, 2400}},
        {"3:2", {2528, 1696}}, {"2:3", {1696, 2528}}, {"16:9", {2752, 1536}}, {"9:16", {1536, 2752}}
    };
    // Same documented fallback as the official pipeline integration example.
    auto found = sizes.find(rewrite.wh_ratio);
    auto size = found == sizes.end() ? sizes.at("1:1") : found->second;
    canvas.width = size.first; canvas.height = size.second;
    return canvas;
}
} // namespace tc::qwen21
