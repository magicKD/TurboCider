#pragma once
#include "pe_language.hpp"
#include "pe_sampling.hpp"
#include "pe_conditioning.hpp"
#include "prompt_rewrite.hpp"
#include "../../core/tokenizer.hpp"

namespace tc::qwen21::pe {
struct TextGenerationResult {
    std::string raw;
    PromptRewriteResult rewrite;
    int prompt_tokens = 0, generated_tokens = 0;
    // prompt_tokens is the expanded prefill length; retain the unexpanded
    // count, mRoPE origin and actual cache length for independent validation.
    int input_prompt_tokens = 0, next_position = 0, cached_tokens = 0;
    std::vector<int> generated_ids;
    bool stopped_eos = false;
    bool chunked_prefill = false;
    double prefill_seconds = 0., decode_seconds = 0.;
    // A parseable object at the token limit is not a completed generation.
    bool complete() const { return stopped_eos && rewrite.parse_ok; }
};

// Official PE-T2I text-only checkpoint. Weights/tokenizer must come from the
// same installation. No edit/vision support is implied by this interface.
// Owns fresh recurrent/KV state per request; caller serializes GPU use and
// controls checkpoint residency. Exceptions/cancellation never return a
// partial result as a successful rewrite.
TextGenerationResult generate_text(const Weights &, const Tokenizer &,
    const std::string &system_prompt, const std::string &user_prompt,
    const SamplingProfile &, uint64_t seed, const Event &,
    std::atomic<bool> &cancelled, LanguageConfig language = {});

// Experimental PE-I2I generation from caller-validated visual features.
// No production routing/vision-quality acceptance is implied. Ordered grids
// and merged features must come from the same PE checkpoint as the tokenizer.
// prompt_tokens counts the expanded multimodal prompt. Owns fresh cache state.
TextGenerationResult generate_edit_features(const Weights &, const Tokenizer &,
    const std::string &system_prompt, const std::string &user_prompt,
    const std::vector<ImageGrid> &, const std::vector<Tensor> &merged_features,
    const SamplingProfile &, uint64_t seed, const Event &,
    std::atomic<bool> &cancelled, LanguageConfig language = {});

// Native PE-I2I image entry point. Inputs are uint8 NHWC RGB/RGBA tensors;
// preprocessing and Qwen3.5 visual encoding stay in C++/MLX. Production
// Session/App routing remains separately gated by visual parity/quality.
// fp32_visual is experimental and promotes only visual arithmetic; the
// checkpoint and language model precision are unchanged. It is not BF16 parity.
TextGenerationResult generate_edit_images(const Weights &, const Tokenizer &,
    const std::string &system_prompt, const std::string &user_prompt,
    const std::vector<Tensor> &images, const SamplingProfile &, uint64_t seed,
    const Event &, std::atomic<bool> &, LanguageConfig language = {},
    bool fp32_visual = false);
}
