#pragma once
#include "../../core/common.hpp"
#include <random>
#include <unordered_set>

namespace tc::qwen21::pe {
struct SamplingProfile {
    double temperature = 1., top_p = .95, presence_penalty = 1.5;
    int top_k = 20, max_new_tokens = 16256;
    static SamplingProfile for_edit() {
        SamplingProfile profile;
        profile.presence_penalty = 0.; profile.max_new_tokens = 24000;
        return profile;
    }
};
struct TokenDistribution {
    std::vector<int> ids;
    std::vector<double> probabilities;
};
// Presence penalty is applied once to generated IDs only. The prompt must not
// be added to this set. Native seeded sampling is not Torch RNG compatibility.
TokenDistribution token_distribution(const std::vector<float> &logits,
    const std::unordered_set<int> &generated, const SamplingProfile &);
int draw_token(const TokenDistribution &, std::mt19937_64 &);
}
