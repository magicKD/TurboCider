#include "../../native/models/qwen21/pe_sampling.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace tc::qwen21::pe;
    SamplingProfile c;
    assert(c.max_new_tokens == 16256 && c.presence_penalty == 1.5);
    auto edit = SamplingProfile::for_edit();
    assert(edit.max_new_tokens == 24000 && edit.presence_penalty == 0);
    c.top_k = 0; c.top_p = 1;
    auto d = token_distribution({1, 2, 3}, {2}, c);
    assert((d.ids == std::vector<int>{1, 2, 0}));
    const double z = std::exp(2.) + std::exp(1.5) + std::exp(1.);
    assert(std::abs(d.probabilities[0] - std::exp(2.)/z) < 1e-12);
    c.top_k = 2; c.top_p = .7;
    d = token_distribution({std::log(.6f), std::log(.3f), std::log(.1f)}, {}, c);
    assert((d.ids == std::vector<int>{0, 1})); // retain crossing token
    c.top_p = .6;
    assert(token_distribution({std::log(.6f), std::log(.3f), std::log(.1f)}, {}, c).ids.size() == 1);
    c.top_k = 1; c.top_p = 1;
    assert((token_distribution({4, 4, 2}, {}, c).ids == std::vector<int>{0, 1}));
    c.temperature = 0;
    assert(token_distribution({4, 4, 2}, {}, c).ids == std::vector<int>{0});
    assert(token_distribution({4, 4, 2}, {0}, c).ids == std::vector<int>{1});
    assert(token_distribution({-INFINITY, 0}, {}, c).ids == std::vector<int>{1});
    auto rejects = [](auto fn) { bool failed = false; try { fn(); } catch (const std::exception &) { failed = true; } assert(failed); };
    rejects([&] { token_distribution({NAN}, {}, c); });
    rejects([&] { token_distribution({INFINITY}, {}, c); });
    rejects([&] { token_distribution({-INFINITY}, {}, c); });
    rejects([&] { token_distribution({0}, {1}, c); });
    c.temperature = -1;
    rejects([&] { token_distribution({0}, {}, c); });
    std::mt19937_64 a(42), b(42);
    d = {{1, 2}, {.25, .75}};
    int first = 0;
    for (int i = 0; i < 10000; ++i) { int x = draw_token(d, a); assert(x == draw_token(d, b)); first += x == 1; }
    assert(first > 2300 && first < 2700);
    std::cout << "PASS PE task sampling profiles, presence, top-k/p, greedy, invalid logits, native seeded draws\n";
}
