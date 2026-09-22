#include "../../native/models/qwen21/prompt_rewrite.hpp"
#include <cassert>
#include <iostream>
int main() {
    using namespace tc::qwen21;
    auto t2i = parse_prompt_rewrite_answer("<think>plan</think> prose {\"rewritten_prompt\":\"A {red} fox\",\"wh_ratio\":\"16:9\"} trailing", false);
    assert(t2i.parse_ok && t2i.thinking == "plan" && t2i.positive_prompt == "A {red} fox" && t2i.wh_ratio == "16:9" && t2i.ratio_follow.empty());
    auto t2i_size = prompt_rewrite_canvas(t2i); assert(t2i_size.width == 2752 && t2i_size.height == 1536);
    auto edit = parse_prompt_rewrite_answer("noise {\"rewrited_prompt\":\"Replace sky\",\"wh_ratio\":\"\",\"ratio_follow\":\"<image2>\"}", true);
    assert(edit.parse_ok && edit.ratio_follow == "<image2>");
    auto follow = prompt_rewrite_canvas(edit, {{640, 320}, {320, 640}}, 1024);
    assert(follow.follows_reference && follow.reference_index == 1 && follow.width == 736 && follow.height == 1440);
    auto bad = parse_prompt_rewrite_answer("not json", true); assert(!bad.parse_ok && bad.positive_prompt == "not json");
    auto malformed = parse_prompt_rewrite_answer("{\"rewritten_prompt\":\"ok\"}", false); assert(malformed.parse_ok);
    auto escaped = parse_prompt_rewrite_answer(R"({"rewritten_prompt":"\u4f60\u597d\n\"{fox}\"","wh_ratio":"3:4"})", false);
    assert(escaped.parse_ok && escaped.positive_prompt == "你好\n\"{fox}\"");
    auto last = parse_prompt_rewrite_answer(R"({"rewritten_prompt":"first"} {"rewritten_prompt":"last"})", true);
    assert(last.parse_ok && last.positive_prompt == "last");
    auto prose = parse_prompt_rewrite_answer(R"(an unmatched " in prose {"rewritten_prompt":"comma ,} inside text"})", false);
    assert(prose.parse_ok && prose.positive_prompt == "comma ,} inside text");
    auto unfinished = parse_prompt_rewrite_answer(R"(<think>{"rewritten_prompt":"not an answer"})", false);
    assert(!unfinished.parse_ok && unfinished.positive_prompt.empty());
    for (const auto &input : {R"({"rewritten_prompt":false,"wh_ratio":"4:3"})",
                             R"({"rewritten_prompt":"invalid",})", R"({"rewritten_prompt":"\q"})",
                             R"({"rewritten_prompt":"invalid","extra":[1, ]})"})
        assert(!parse_prompt_rewrite_answer(input, false).parse_ok);
    auto rejects = [](auto action) { try { action(); } catch (const std::exception &) { return; } assert(false); };
    rejects([&] { prompt_rewrite_canvas(bad); });
    rejects([&] { prompt_rewrite_canvas(edit, {{320, 640}}); });
    edit.wh_ratio = "1:1";
    rejects([&] { prompt_rewrite_canvas(edit, {{320, 640}, {320, 640}}); });
    std::cout << "PASS Qwen21 prompt-rewrite answer parser and ratio canvas contract\n";
}
