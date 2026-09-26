#pragma once
#include <string>
#include <utility>
#include <vector>

namespace tc::qwen21 {
// Official two-turn PE chat, thinking enabled; image placeholders stay in
// input order. The vision processor later expands their image-pad spans.
std::string prompt_rewrite_chat(const std::string &system_prompt,
                                const std::string &user_prompt, size_t references = 0);
struct PromptRewriteResult {
    std::string positive_prompt;
    std::string wh_ratio;
    std::string ratio_follow;
    std::string thinking;
    bool parse_ok = false;
};
struct PromptRewriteCanvas {
    int width = 2048;
    int height = 2048;
    bool follows_reference = false;
    int reference_index = -1; // zero-based; -1 for an explicit/new canvas
};

// Parses the official Qwen-Image-2.1 prompt-enhancer answer contract without
// Python. This is an answer parser only; loading/running Qwen3.5-VL PE weights
// remains a separate native backend requirement.
PromptRewriteResult parse_prompt_rewrite_answer(const std::string &decoded, bool edit);
PromptRewriteCanvas prompt_rewrite_canvas(const PromptRewriteResult &,
    const std::vector<std::pair<int, int>> &references = {}, int follow_resolution = 2048);
}
