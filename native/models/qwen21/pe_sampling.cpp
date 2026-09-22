#include "pe_sampling.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace tc::qwen21::pe {
TokenDistribution token_distribution(const std::vector<float> &logits,
    const std::unordered_set<int> &generated, const SamplingProfile &profile) {
    require(!logits.empty() && logits.size() <= 1048576, "invalid PE logits size");
    require(std::isfinite(profile.temperature) && profile.temperature >= 0 &&
            std::isfinite(profile.top_p) && profile.top_p > 0 && profile.top_p <= 1 &&
            profile.top_k >= 0 && std::isfinite(profile.presence_penalty), "invalid PE sampling profile");
    std::vector<double> values(logits.begin(), logits.end());
    for (int id : generated) {
        require(id >= 0 && size_t(id) < values.size(), "generated ID outside PE vocabulary");
        values[id] -= profile.presence_penalty;
    }
    TokenDistribution result;
    for (size_t i = 0; i < values.size(); ++i) {
        require(!std::isnan(values[i]) && values[i] != INFINITY, "nonfinite PE logits");
        if (values[i] != -INFINITY) result.ids.push_back(int(i));
    }
    require(!result.ids.empty(), "all PE tokens are masked");
    const auto compare = [&](int a, int b) { return values[a] == values[b] ? a < b : values[a] > values[b]; };
    const size_t keep = profile.temperature == 0 ? 1 :
        (profile.top_k == 0 ? result.ids.size() : std::min(size_t(profile.top_k), result.ids.size()));
    std::partial_sort(result.ids.begin(), result.ids.begin() + keep, result.ids.end(), compare);
    if (profile.temperature == 0) { result.ids.resize(1); result.probabilities = {1.}; return result; }
    // HF top-k uses a score threshold, keeping ties at the cutoff too.
    const double cutoff = values[result.ids[keep - 1]];
    result.ids.erase(std::remove_if(result.ids.begin(), result.ids.end(),
        [&](int id) { return values[id] < cutoff; }), result.ids.end());
    std::sort(result.ids.begin(), result.ids.end(), compare);
    double total = 0;
    for (int id : result.ids) {
        double p = std::exp((values[id] - values[result.ids[0]]) / profile.temperature);
        result.probabilities.push_back(p); total += p;
    }
    double retained = 0;
    size_t count = 0;
    do { retained += result.probabilities[count++] / total; }
    while (count < result.ids.size() && retained < profile.top_p);
    result.ids.resize(count); result.probabilities.resize(count);
    const double normalizer = std::accumulate(result.probabilities.begin(), result.probabilities.end(), 0.);
    for (double &p : result.probabilities) p /= normalizer;
    return result;
}
int draw_token(const TokenDistribution &distribution, std::mt19937_64 &rng) {
    require(!distribution.ids.empty() && distribution.ids.size() == distribution.probabilities.size(),
            "invalid PE token distribution");
    double total = 0;
    for (double p : distribution.probabilities) {
        require(std::isfinite(p) && p >= 0, "invalid PE sampling probability");
        total += p;
    }
    require(std::isfinite(total) && total > 0, "empty PE sampling mass");
    // Explicit 53-bit conversion gives a stable native draw across STL vendors.
    double needle = double(rng() >> 11) * (1. / 9007199254740992.) * total;
    for (size_t i = 0; i < distribution.ids.size(); ++i)
        if ((needle -= distribution.probabilities[i]) < 0) return distribution.ids[i];
    return distribution.ids.back();
}
}
